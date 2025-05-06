# info sur les paramètres et résultats de rainbow
# https://paperswithcode.com/paper/rainbow-combining-improvements-in-deep/review/?hl=19877
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import math
import numpy as np
from collections import deque
import random, os, sys
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass
from datetime import datetime
import atexit
from flask import Flask, send_from_directory, render_template_string
from colorama import Fore, Style
import time

class GraphWebServer:
    def __init__(
        self, graph_dir="graphs", host="0.0.0.0", port=5000, auto_display_latest=True
    ):
        """
        graph_dir: dossier où sont sauvegardés les graphiques (doit exister ou sera créé)
        host: adresse d'écoute (0.0.0.0 pour être accessible sur le réseau local)
        port: port d'écoute
        auto_display_latest: si True, affiche directement le dernier PNG en date ; sinon, affiche la liste des fichiers
        """
        self.graph_dir = graph_dir
        self.host = host
        self.port = port
        self.auto_display_latest = auto_display_latest
        self.app = Flask(__name__)
        self._setup_routes()

    def _setup_routes(self):
        @self.app.route("/")
        def index():
            # Crée le dossier s'il n'existe pas
            if not os.path.exists(self.graph_dir):
                os.makedirs(self.graph_dir)
            if self.auto_display_latest:
                # Récupère la liste des fichiers PNG dans le dossier
                files = [
                    f for f in os.listdir(self.graph_dir) if f.lower().endswith(".png")
                ]
                if files:
                    # Sélectionne le fichier le plus récent en fonction de sa date de modification
                    latest_file = max(
                        files,
                        key=lambda f: os.path.getmtime(os.path.join(self.graph_dir, f)),
                    )
                else:
                    latest_file = None
                # Template HTML qui affiche directement le dernier graphique
                html = """
                <!doctype html>
                <html>
                  <head>
                    <meta charset="utf-8">
                    <title>Graphique des parties</title>
                  </head>
                  <body>
                    <h1>Graphique des parties</h1>
                    {% if latest_file %}
                      <img src="/graphs/{{ latest_file }}" alt="Graphique des parties" style="max-width:100%; height:auto;">
                    {% else %}
                      <p>Aucun graphique disponible.</p>
                    {% endif %}
                  </body>
                </html>
                """
                return render_template_string(html, latest_file=latest_file)
            else:
                # Sinon, affiche la liste des fichiers disponibles
                files = [
                    f for f in os.listdir(self.graph_dir) if f.lower().endswith(".png")
                ]
                html = """
                <!doctype html>
                <html>
                  <head>
                    <meta charset="utf-8">
                    <title>Évolution des parties</title>
                  </head>
                  <body>
                    <h1>Graphiques des parties</h1>
                    {% if files %}
                      <ul>
                      {% for file in files %}
                        <li><a href="/graphs/{{ file }}">{{ file }}</a></li>
                      {% endfor %}
                      </ul>
                    {% else %}
                      <p>Aucun graphique disponible.</p>
                    {% endif %}
                  </body>
                </html>
                """
                return render_template_string(html, files=files)

        @self.app.route("/graphs/<path:filename>")
        def serve_graph(filename):
            # Sert le fichier depuis le dossier graph_dir
            return send_from_directory(self.graph_dir, filename)

    def run(self):
        # Démarre le serveur sur l'adresse et le port spécifiés
        self.app.run(host=self.host, port=self.port)

    def start(self):
        """Démarre le serveur web en appelant run()."""
        self.run()

class TeeLogger:
    """Capture les sorties du terminal (stdout et stderr) tout en les enregistrant dans un fichier."""

    def __init__(self, log_dir="logs"):
        os.makedirs(log_dir, exist_ok=True)  # Crée le dossier logs s'il n'existe pas
        log_filename = (
            f"{log_dir}/training_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        )

        self.terminal = sys.stdout  # Sauvegarde la sortie standard originale
        self.log_file = open(log_filename, "w", encoding="utf-8")
        self.log_filename = log_filename

        sys.stdout = self
        sys.stderr = self  # Redirige aussi les erreurs vers le fichier log

        print(
            f"\n📜 [LOGGING ACTIVÉ] Toutes les sorties seront enregistrées dans {log_filename}\n"
        )

    def write(self, message):
        """Écrit simultanément dans le terminal et dans le fichier log"""
        self.terminal.write(message)  # Affiche dans le terminal
        self.terminal.flush()  # Assure l'affichage en temps réel
        self.log_file.write(message)  # Sauvegarde dans le fichier
        self.log_file.flush()  # Évite la perte de logs en cas de crash

    def flush(self):
        """Flush les buffers (nécessaire pour sys.stdout)"""
        self.terminal.flush()
        self.log_file.flush()

    def close(self):
        """Ferme le fichier de log et restaure stdout"""
        sys.stdout = self.terminal
        sys.stderr = self.terminal
        self.log_file.close()
        print(f"\n✅ [LOGGING FERMÉ] Logs sauvegardés dans {self.log_filename}\n")
# Activation automatique du logger
logger = TeeLogger()
# ✅ Fermeture propre des logs à la fin du script (PLACÉ À LA FIN)
atexit.register(logger.close)

@dataclass
class TrainingConfig:
    """Configuration class for DQN training parameters"""

    input_size: any  # Pour MLP, un int ; pour CNN, un tuple (channels, height, width)
    state_history_size: int
    hidden_size: int
    output_size: int
    hidden_layers: int
    learning_rate: float
    gamma: float
    use_noisy: bool  # Active NoisyNet si True (remplace epsilon-greedy)
    epsilon_start: float
    epsilon_end: float
    epsilon_linear: float
    epsilon_decay: float
    epsilon_add: float
    buffer_capacity: int
    current_episode: int = 0  # episode courant
    rainbow_eval: int = 200_000 # Nombre de step avant chaque phase d'évaluation Rainbow (250_000 dans Rainbow)
    rainbow_eval_pourcent: int = 3 # Pourcentage du temps d'évaluation (5% = pourcentage du nb de step d'évaluation)
    batch_size: int = 32  # Taille du batch pour l'entraînement
    min_history_size: int = 0  # Taille minimale du buffer avant d'entraîner
    device: str = "cuda:0" if torch.cuda.is_available() else "cpu"
    #device: str = "cpu"  # Utiliser CPU pour le débogage
    double_dqn: bool = False  # Active le Double DQN si True
    target_update_freq: int = 32000  # 🔁 Fréquence de mise à jour du réseau cible
    dueling: bool = False  # 🆕 Active le Dueling DQN
    prioritized_replay: bool = False
    nstep: bool = False   # ← option nstep activable
    nstep_n: int = 3      # ← valeur par défaut (3 ou 5)
    model_type: str = "mlp"  # Choix de l'architecture : "mlp" ou "cnn"
    cnn_type: str = "default"  # ✅ AJOUTER ICI
    state_extractor: callable = None  # Fonction d'extraction d'état
    mode: str = (
        "exploration"  # "exploration" pour l'entraînement, "exploitation" (inference) pour la phase finale
    )

class GPUReplayBuffer:
    def __init__(self, capacity, config, prioritized=False, alpha=0.5, optimize_memory=False):
        self.capacity = capacity
        self.config = config
        self.device = torch.device(config.device)
        self.prioritized = prioritized
        self.optimize_memory = optimize_memory
        self.pos = 0
        self.size = 0

        if config.model_type.lower() == "cnn" and self.optimize_memory:
            self.store_on_cpu = True
            state_shape = config.input_size  # (C, H, W)
            self.states = np.empty((capacity, *state_shape), dtype=np.uint8)
            self.next_states = np.empty((capacity, *state_shape), dtype=np.uint8)
        else:
            self.store_on_cpu = False
            if config.model_type.lower() == "cnn":
                state_shape = config.input_size
            else:
                state_shape = (config.input_size * config.state_history_size,)

            self.states = torch.empty((capacity, *state_shape), dtype=torch.float32, device=self.device)
            self.next_states = torch.empty((capacity, *state_shape), dtype=torch.float32, device=self.device)

        self.actions = torch.empty((capacity,), dtype=torch.int64, device=self.device)
        self.rewards = torch.empty((capacity,), dtype=torch.float32, device=self.device)
        self.dones = torch.empty((capacity,), dtype=torch.float32, device=self.device)

        if self.prioritized:
            self.priorities = torch.zeros(capacity, dtype=torch.float32, device=self.device)
            self.alpha = alpha
            self.beta = 0.4
            self.beta_increment = (1.0 - self.beta) / (self.capacity / 2)


    def push(self, state, action, reward, next_state, done):
        if self.config.model_type.lower() == "mlp":
            state = state.flatten()
            next_state = next_state.flatten()

        if self.store_on_cpu:
            self.states[self.pos] = (state * 255).astype(np.uint8)
            self.next_states[self.pos] = (next_state * 255).astype(np.uint8)
        else:
            if not torch.is_tensor(state):
                state = torch.tensor(state, dtype=torch.float32, device=self.device)
            if not torch.is_tensor(next_state):
                next_state = torch.tensor(next_state, dtype=torch.float32, device=self.device)
            self.states[self.pos].copy_(state)
            self.next_states[self.pos].copy_(next_state)

        self.actions[self.pos] = action
        self.rewards[self.pos] = reward
        self.dones[self.pos] = float(done)

        if self.prioritized:
            max_priority = self.priorities[: self.size].max() if self.size > 0 else 1.0
            self.priorities[self.pos] = max_priority

        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)


            
    def sample(self, batch_size, beta=None):
        indices = torch.randint(0, self.size, (batch_size,), device=self.device)
        weights = torch.ones(batch_size, device=self.device)
        if self.store_on_cpu:
            states_batch = torch.tensor(self.states[indices.cpu().numpy()], dtype=torch.float32, device=self.device) / 255.0
            next_states_batch = torch.tensor(self.next_states[indices.cpu().numpy()], dtype=torch.float32, device=self.device) / 255.0
        else:
            states_batch = self.states.index_select(0, indices)
            next_states_batch = self.next_states.index_select(0, indices)

        actions_batch = self.actions.index_select(0, indices)
        rewards_batch = self.rewards.index_select(0, indices)
        dones_batch = self.dones.index_select(0, indices)

        if self.prioritized:
            prios = self.priorities[: self.size] ** self.alpha
            probs = prios / prios.sum()
            indices = torch.multinomial(probs, batch_size, replacement=False)
            weights = (self.size * probs[indices]) ** (-beta if beta else self.beta)
            weights = weights / weights.max()
            self.beta = min(1.0, self.beta + self.beta_increment)

        return (
            states_batch,
            actions_batch,
            rewards_batch,
            next_states_batch,
            dones_batch,
            indices,
            weights,
        )


    def update_priorities(self, batch_indices, batch_priorities):
        if self.priorities is None:
            return
        self.priorities[batch_indices] = batch_priorities.detach().abs()

    def __len__(self):
        return self.size
class NStepTransitionWrapper:
    def __init__(self, n, gamma):
        self.n = n
        self.gamma = gamma
        self.buffer = []
    
    def append(self, state, action, reward, done, next_state):
        self.buffer.append((state, action, reward, done, next_state))
        if len(self.buffer) < self.n and not done:
            return None  # Pas assez pour un n-step

        R, next_s, d = 0, None, False
        for i in range(len(self.buffer)):
            r = self.buffer[i][2]
            R += (self.gamma ** i) * r
            if self.buffer[i][3]:  # done
                d = True
                next_s = self.buffer[i][4]
                break
        else:
            next_s = self.buffer[-1][4]
            d = self.buffer[-1][3]

        first = self.buffer[0]
        nstep_transition = (first[0], first[1], R, next_s, d)
        self.buffer.pop(0)
        return nstep_transition

    def flush(self):
        result = []
        while self.buffer:
            # Pour forcer le vidage sans dépendre de append
            state, action, _, _, _ = self.buffer[0]
            R, next_s, d = 0, None, False
            for i in range(len(self.buffer)):
                r = self.buffer[i][2]
                R += (self.gamma ** i) * r
                if self.buffer[i][3]:  # done
                    d = True
                    next_s = self.buffer[i][4]
                    break
            else:
                next_s = self.buffer[-1][4]
                d = self.buffer[-1][3]

            nstep_transition = (state, action, R, next_s, d)
            result.append(nstep_transition)
            self.buffer.pop(0)
        return result


class NoisyLinear(nn.Module):
    def __init__(self, in_features, out_features, sigma_init=0.5):
        super(NoisyLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.mu_weight = nn.Parameter(torch.empty(out_features, in_features))
        self.sigma_weight = nn.Parameter(torch.full((out_features, in_features), sigma_init))
        self.mu_bias = nn.Parameter(torch.empty(out_features))
        self.sigma_bias = nn.Parameter(torch.full((out_features,), sigma_init))

        # ✅ Permet l'entraînement de sigma
        self.sigma_weight.requires_grad = True
        self.sigma_bias.requires_grad = True

        self.register_buffer("epsilon_weight", torch.zeros(out_features, in_features))
        self.register_buffer("epsilon_bias", torch.zeros(out_features))

        self.sigma_init = sigma_init
        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self):
        mu_range = 1 / math.sqrt(self.in_features)
        self.mu_weight.data.uniform_(-mu_range, mu_range)
        self.mu_bias.data.uniform_(-mu_range, mu_range)
        
        # Ne pas réinitialiser sigma s'il a déjà été mis à jour
        if self.sigma_weight.data.mean() == self.sigma_init:
            self.sigma_weight.data.fill_(self.sigma_init)
            self.sigma_bias.data.fill_(self.sigma_init)

    def _scale_noise(self, size):
        # Version améliorée de la génération de bruit factoriel
        # plus stable pour l'apprentissage
        x = torch.randn(size)
        return x.sign() * torch.sqrt(torch.abs(x))

    def reset_noise(self):
        device = self.mu_weight.device
        epsilon_in = self._scale_noise(self.in_features).to(device)
        epsilon_out = self._scale_noise(self.out_features).to(device)
        self.epsilon_weight = epsilon_out.ger(epsilon_in)
        self.epsilon_bias = epsilon_out

    def forward(self, input):
        if self.training:
            weight = self.mu_weight + self.sigma_weight * self.epsilon_weight
            bias = self.mu_bias + self.sigma_bias * self.epsilon_bias
        else:
            weight = self.mu_weight
            bias = self.mu_bias
        return F.linear(input, weight, bias)

    def get_sigma(self):
        return self.sigma_weight.abs().mean().item(), self.sigma_bias.abs().mean().item()

############################ Comparatif des architectures CNN #######################################
#  Mode       | Nb conv | Stride fort | MaxPool | FC size estimé | Total Ops        | Vitesse       #
# ------------|---------|-------------|---------|----------------|------------------|---------------#
#  precise    | 2       | Non         | Oui     | ~64×26×28      | Moyen + FC lourd | 🟡 Moyen Lent #
#  original   | 3       | Oui (4,2,2) | Non     | ~128×13×14     | Conv rapide      | 🟢 Rapide     #
#  deepmind   | 3       | Oui (4,2,1) | Non     | ~64×11×12      | Léger + FC 512   | ✅ Efficace   #
#####################################################################################################
class DQNModel(nn.Module):
    """
    DQNModel est une architecture de réseau neuronal modulaire utilisée pour l'entraînement par renforcement
    dans l'environnement Space Invaders (MAME).

    Elle supporte plusieurs types de réseaux :
    - MLP (Perceptron Multicouche) pour un vecteur d'état structuré
    - CNN "original" : 3 convolutions avec stride (4, 2, 2) et BatchNorm
    - CNN "precise" : 2 convolutions avec stride=1 et MaxPooling (haute fidélité spatiale)
    - CNN "deepmind" : architecture classique de DeepMind pour Atari (8x8, 4) → (4x4, 2) → (3x3, 1)

    Paramètres :
    - input_size : tuple (channels, height, width) ou int (MLP)
    - hidden_size : taille des couches fully connected
    - cnn_type : "original", "precise" ou "deepmind"
    - output_size : nombre d'actions discrètes (ex: 6 pour Space Invaders)

    La méthode forward applique le bon pipeline en fonction du type de modèle sélectionné.
    """
    def __init__(self, config):
        super().__init__()
        self.config = config

        # 🔹 Choix entre couche standard ou NoisyNet
        Linear = NoisyLinear if config.use_noisy else nn.Linear

        if config.model_type == "cnn":
            channels, height, width = config.input_size

            if config.cnn_type == "deepmind":
                self.encoder = nn.Sequential(
                    nn.Conv2d(channels, 32, kernel_size=8, stride=4),
                    nn.ReLU(),
                    nn.Conv2d(32, 64, kernel_size=4, stride=2),
                    nn.ReLU(),
                    nn.Conv2d(64, 64, kernel_size=3, stride=1),
                    nn.ReLU(),
                )

            elif config.cnn_type == "precise":
                self.encoder = nn.Sequential(
                    nn.Conv2d(channels, 32, kernel_size=5, stride=1, padding=2),
                    nn.BatchNorm2d(32),
                    nn.ReLU(),
                    nn.MaxPool2d(kernel_size=2),
                    nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
                    nn.BatchNorm2d(64),
                    nn.ReLU(),
                    nn.MaxPool2d(kernel_size=2),
                )

            elif config.cnn_type == "original":
                self.encoder = nn.Sequential(
                    nn.Conv2d(channels, 32, kernel_size=8, stride=4),
                    nn.BatchNorm2d(32),
                    nn.ReLU(),
                    nn.Conv2d(32, 64, kernel_size=4, stride=2),
                    nn.BatchNorm2d(64),
                    nn.ReLU(),
                    nn.Conv2d(64, 64, kernel_size=3, stride=2),
                    nn.BatchNorm2d(64),
                    nn.ReLU(),
                )

            else:
                raise ValueError(f"Unknown cnn_type: {config.cnn_type}")

            # Calcul automatique de flatten size
            with torch.no_grad():
                dummy = torch.zeros(1, *config.input_size)
                encoded = self.encoder(dummy)
                self.flatten_size = encoded.view(1, -1).size(1)

            self.fc1 = Linear(self.flatten_size, config.hidden_size)
            self.bn_fc1 = nn.BatchNorm1d(config.hidden_size)
            self.act_fc1 = nn.LeakyReLU(0.01)
            if config.dueling:
                self.value_head = Linear(config.hidden_size, 1)
                self.advantage_head = Linear(config.hidden_size, config.output_size)
            else:
                self.output_layer = Linear(config.hidden_size, config.output_size)

        elif config.model_type == "mlp":
            input_dim = int(np.prod(config.input_size)) * config.state_history_size

            self.mlp_layers = nn.ModuleList()
            in_dim = input_dim
            for _ in range(config.hidden_layers):
                self.mlp_layers.append(Linear(in_dim, config.hidden_size))
                in_dim = config.hidden_size

            if config.dueling:
                self.value_head = Linear(in_dim, 1)
                self.advantage_head = Linear(in_dim, config.output_size)
            else:
                self.output_layer = Linear(in_dim, config.output_size)

            self.activation = nn.LeakyReLU(0.01)


        else:
            raise NotImplementedError("Unknown model_type")

    def forward(self, x):
        # Cas CNN
        if self.config.model_type.lower() == "cnn":
            # 1) Encodage convolutionnel
            x = self.encoder(x)                         # -> (batch, C', H', W')
            # 2) Flatten
            x = x.view(x.size(0), -1)                   # -> (batch, flatten_size)
            # 3) Fully‑connected + BatchNorm + activation
            x = self.act_fc1(self.bn_fc1(self.fc1(x)))  # -> (batch, hidden_size)
            # 4) Tête du réseau (dueling ou non)
            if self.config.dueling:
                value     = self.value_head(x)          # -> (batch, 1)
                advantage = self.advantage_head(x)      # -> (batch, output_size)
                # recombinaison dueling
                return value + (advantage - advantage.mean(dim=1, keepdim=True))
            else:
                return self.output_layer(x)             # -> (batch, output_size)

        # Cas MLP (vecteur d’état)
        else:
            # x est déjà de forme (batch, input_dim) après state_hist concatenation
            for layer in self.mlp_layers:
                x = layer(x)
                x = self.activation(x)
            if self.config.dueling:
                value     = self.value_head(x)
                advantage = self.advantage_head(x)
                return value + (advantage - advantage.mean(dim=1, keepdim=True))
            else:
                return self.output_layer(x)

    def get_sigma_values(self):
        return {
            name: module.sigma_weight.abs().mean().item()
            for name, module in self.named_modules()
            if isinstance(module, NoisyLinear)
        }

    def reset_noise(self):
        if not self.config.use_noisy:
            return
        for module in self.modules():
            if isinstance(module, NoisyLinear):
                module.reset_noise()

class DQNTrainer:
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.device = torch.device(config.device)

        # Instanciation unique de DQNModel pour CNN et MLP
        self.dqn = DQNModel(config).to(self.device)
        self.target_dqn = DQNModel(config).to(self.device)
        self.copy_weights(self.dqn, self.target_dqn)
        self.set_mode(config.mode)
        # Initialisation des autres composants d'entraînement...
        self.optimizer = optim.Adam(self.dqn.parameters(), lr=config.learning_rate,eps=1.5e-4)
        #self.optimizer = optim.SGD(self.dqn.parameters(), lr=config.learning_rate)

        self.criterion = nn.MSELoss()
        # Utilisation du ReplayBuffer classique (échantillonnage uniforme)
        self.replay_buffer = GPUReplayBuffer(
            config.buffer_capacity, config, prioritized=config.prioritized_replay, optimize_memory=(config.model_type == "cnn")
        )

        if config.nstep:
            self.nstep_wrapper = NStepTransitionWrapper(config.nstep_n, config.gamma)
        else:
            self.nstep_wrapper = None
        self.epsilon = config.epsilon_start
        self.state_history = deque(maxlen=config.state_history_size)
        self.criterion = nn.SmoothL1Loss(reduction="none")  # ✅ Rainbow-style

        self.training_steps = 0
        self.eval_steps = 0

    @staticmethod
    def copy_weights(source: nn.Module, target: nn.Module) -> None:
        """Copie les poids du réseau source vers le réseau cible"""
        target.load_state_dict(source.state_dict())
    def set_mode(self, mode: str = "exploration"):
        self.config.mode = mode.lower()
        if self.config.mode == "exploitation":
            self.dqn.eval()
            self.target_dqn.eval()
        else:
            self.dqn.train()
            self.target_dqn.train()    
            
    def select_action(self, state: np.ndarray) -> int:
        state_tensor = torch.from_numpy(state).float().unsqueeze(0).to(self.device)

        if self.config.use_noisy:
            if self.config.mode != "exploitation":
                self.dqn.reset_noise()  # <--- bruit à chaque step
            was_training = self.dqn.training
            self.dqn.eval()
            with torch.no_grad():
                q_values = self.dqn(state_tensor)
            if was_training:
                self.dqn.train()
            return torch.argmax(q_values).item()
        else:
            # Version epsilon-greedy classique
            if self.config.mode == "exploitation" or self.epsilon == 0:
                was_training = self.dqn.training
                self.dqn.eval()
                with torch.no_grad():
                    q_values = self.dqn(state_tensor)
                if was_training:
                    self.dqn.train()
                return torch.argmax(q_values).item()
            else:
                if random.random() >= self.epsilon:
                    was_training = self.dqn.training
                    self.dqn.eval()
                    with torch.no_grad():
                        q_values = self.dqn(state_tensor)
                    if was_training:
                        self.dqn.train()
                    return torch.argmax(q_values).item()
                else:
                    return random.randint(0, self.config.output_size - 1)

    def OLD_select_action(self, state: np.ndarray) -> int:
        state_tensor = torch.from_numpy(state).float().unsqueeze(0).to(self.device)

        if self.config.use_noisy:
            if self.config.mode != "exploitation":
                self.dqn.reset_noise()  # <--- bruit à chaque step
            with torch.no_grad():
                q_values = self.dqn(state_tensor)
                return torch.argmax(q_values).item()
        else:
            # Version epsilon-greedy classique
            if self.config.mode == "exploitation" or self.epsilon == 0:
                with torch.no_grad():
                    q_values = self.dqn(state_tensor)
                    return torch.argmax(q_values).item()
            else:
                if random.random() >= self.epsilon:
                    with torch.no_grad():
                        q_values = self.dqn(state_tensor)
                        return torch.argmax(q_values).item()
                else:
                    return random.randint(0, self.config.output_size - 1)

    def update_epsilon(self):
        """Met à jour epsilon (ou ne fait rien si NoisyNet est activé)."""
        if self.config.use_noisy:
            # NoisyNet gère l'exploration, pas besoin d'epsilon
            return
        if self.config.epsilon_linear > 0:
            self.epsilon = max(
                self.config.epsilon_end,
                self.epsilon - self.config.epsilon_linear
            )
        else:
            self.epsilon = max(
                self.config.epsilon_end,
                self.epsilon * self.config.epsilon_decay 
            )

    def train_step(self) -> Optional[float]:
        # Évaluation périodique Rainbow
        if self.config.rainbow_eval > 0:
            # Démarrage de l'évaluation
            if (self.training_steps+1) % self.config.rainbow_eval == 0 and self.eval_steps == 0:
                self.config.mode = "exploitation"
                self.set_mode(self.config.mode)
                self.eval_steps = 1

            # Pendant l'évaluation
            if self.eval_steps > 0:
                self.eval_steps += 1
                if self.eval_steps >= self.config.rainbow_eval_pourcent/100*self.config.rainbow_eval:
                    self.config.mode = "exploration"
                    self.set_mode(self.config.mode)
                    self.eval_steps = 0

        # Ne pas entraîner en mode exploitation ou si buffer insuffisant
        if self.config.mode == "exploitation" or len(self.replay_buffer) < max(self.config.batch_size,self.config.min_history_size):
            return None
        
        # 1) Sample (PER ou uniforme)
        state_batch, action_batch, reward_batch, next_state_batch, done_batch, indices, is_weights = \
                self.replay_buffer.sample(self.config.batch_size)

        # 2) Reshape pour MLP vs CNN
        if self.config.model_type.lower() == "mlp":
            state_batch      = state_batch.view(self.config.batch_size, -1)
            next_state_batch = next_state_batch.view(self.config.batch_size, -1)
        else:  # cnn
            state_batch      = state_batch.view(self.config.batch_size, *self.config.input_size)
            next_state_batch = next_state_batch.view(self.config.batch_size, *self.config.input_size)

        action_batch = action_batch.unsqueeze(-1)  # (batch, 1)

        # 3) NoisyNet → reset noise si actif
        if self.config.use_noisy:
            self.dqn.reset_noise()
            self.target_dqn.reset_noise()

        # 4) Q-values courantes 
        q_values      = self.dqn(state_batch)                     # (batch, output_size)
        curr_q_values = q_values.gather(1, action_batch).view(-1) # (batch,) 
        
        # 5) Q-values cibles (Double DQN si activé) et n_step implémentation
        with torch.no_grad():
            if self.config.double_dqn:
                best_next = self.dqn(next_state_batch).argmax(dim=1, keepdim=True)
                next_q_vals = self.target_dqn(next_state_batch).gather(1, best_next).view(-1)
            else:
                next_q_vals = self.target_dqn(next_state_batch).max(dim=1)[0]
            not_done = 1.0 - done_batch
            if self.config.nstep:
                gamma_n = self.config.gamma ** self.config.nstep_n
                target_q = reward_batch + gamma_n * not_done * next_q_vals
            else:
                target_q = reward_batch + self.config.gamma * not_done * next_q_vals


        # 6) Loss Huber par échantillon 
        loss_per_sample = F.smooth_l1_loss(curr_q_values, target_q, reduction="none")

        if self.config.prioritized_replay:
            # on applique directement les IS‑weights
            loss = (is_weights * loss_per_sample).mean()
            # mise à jour des priorités
            td_errors = (curr_q_values - target_q).abs().detach()
            self.replay_buffer.update_priorities(indices, td_errors)
        else:
            loss = loss_per_sample.mean()

        # 7) Rétropropagation et optimisation
        self.optimizer.zero_grad()
        loss.backward()
        # on peut clipper pour la stabilité   
        torch.nn.utils.clip_grad_norm_(self.dqn.parameters(), max_norm=10)   
        self.optimizer.step() 

        # 8) Mise à jour du réseau cible
        self.training_steps += 1
        if self.training_steps % self.config.target_update_freq == 0:
            self.copy_weights(self.dqn, self.target_dqn)

        # 9) Décroissance d’epsilon si on n'est pas en NoisyNet
        self.update_epsilon()
        
        
        # ───────── Critère de sanity logging ─────────
        # On vérifie que la loss est bien un nombre fini
        if not torch.isfinite(loss):
            print(f"⚠️ [train_step] Step {self.training_steps}: loss non finie ! ({loss})")
        elif self.training_steps%5000==0:
            print(f"✅ [train_step] Step {self.training_steps}: loss OK = {loss.item():.4f}")

        return loss.item()
    def save_model(self, path: str) -> None:
        """Sauvegarde les poids du modèle dans un fichier"""
        torch.save(self.dqn.state_dict(), path)

    def load_model(self, path: str) -> None:
        """Charge les poids du modèle depuis un fichier si disponible"""
        if not os.path.exists(path):
            print(
                "\033[33m"
                + f"⚠️ Le fichier {path} n'existe pas. Entraînement à partir de zéro."
                + "\033[0m"
            )
            return  # Ne copie pas les poids si aucun modèle n'a été chargé

        # Charger les poids du modèle
        self.dqn.load_state_dict(torch.load(path, map_location=self.device))
        self.target_dqn.load_state_dict(torch.load(path, map_location=self.device))

        # Copier les poids si le modèle est bien chargé
        self.copy_weights(self.dqn, self.target_dqn)

        print("\033[32m" + f"✅ Modèle chargé depuis {path}" + "\033[0m")

    def update_state_history(self, state: np.ndarray) -> np.ndarray:
        """Met à jour l'historique d'état et retourne l'état correct"""
        self.state_history.append(state)
        while len(self.state_history) > self.config.state_history_size:
            self.state_history.popleft()  # Supprime les frames les plus anciennes

        assert (
            len(self.state_history) == self.config.state_history_size
        ), f"Erreur: state_history contient {len(self.state_history)} frames, attendu {self.config.state_history_size}."

        if self.config.model_type.lower() == "cnn":
            return np.stack(list(self.state_history), axis=0)  # Retourne (N, H, W)
        else:
            return np.concatenate(
                self.state_history, axis=0
            )  # ✅ Retourne `N` états concaténés

    def save_buffer(self, filename="buffer.of.states"):
        try:
            torch.save(self.replay_buffer, filename)
            print(f"✅ Buffer sauvegardé dans {filename}")
        except Exception as e:
            print(f"❌ Erreur de sauvegarde du buffer : {e}")

    def load_buffer(self, filename="buffer.of.states"):
        if not os.path.exists(filename):
            print("🟡 Aucun buffer sauvegardé trouvé.")
            return -1
        try:
            loaded = torch.load(filename, map_location=self.config.device)
            current = self.replay_buffer

            def print_progress_bar(current_index, total, bar_length=30):
                fraction = current_index / total
                filled_length = int(bar_length * fraction)
                bar = '█' * filled_length + '-' * (bar_length - filled_length)
                percent = int(fraction * 100)
                sys.stdout.write(f'\r🔄 Chargement du buffer : |{bar}| {percent}% ({current_index}/{total})')
                sys.stdout.flush()

            if len(loaded) <= current.capacity:
                print(f"🟢 Démarrage du chargement du buffer ({len(loaded)}/{current.capacity})")
                last_percent = -1
                for i in range(len(loaded)):
                    current.push(
                        loaded.states[i],
                        int(loaded.actions[i].item()),
                        float(loaded.rewards[i].item()),
                        loaded.next_states[i],
                        bool(loaded.dones[i].item()),
                    )
                    percent = int((i + 1) * 100 / len(loaded))
                    if percent != last_percent:
                        print_progress_bar(i + 1, len(loaded))
                        last_percent = percent
                print("\n✅ Chargement terminé !")
            else:
                print(f"🟠 Buffer trop gros ({len(loaded)} > {current.capacity})")
                last_percent = -1
                # Même logique que ci-dessus :
                if hasattr(loaded, "priorities") and loaded.prioritized:
                    top = torch.topk(
                        loaded.priorities[: len(loaded)], current.capacity
                    ).indices
                    for n, i in enumerate(top):
                        current.push(
                            loaded.states[i],
                            int(loaded.actions[i].item()),
                            float(loaded.rewards[i].item()),
                            loaded.next_states[i],
                            bool(loaded.dones[i].item()),
                        )
                        percent = int((n + 1) * 100 / current.capacity)
                        if percent != last_percent:
                            print_progress_bar(n + 1, current.capacity)
                            last_percent = percent
                else:
                    for i in range(current.capacity):
                        current.push(
                            loaded.states[i],
                            int(loaded.actions[i].item()),
                            float(loaded.rewards[i].item()),
                            loaded.next_states[i],
                            bool(loaded.dones[i].item()),
                        )
                        percent = int((i + 1) * 100 / current.capacity)
                        if percent != last_percent:
                            print_progress_bar(i + 1, current.capacity)
                            last_percent = percent
                print("✅ Buffer rechargé avec adaptation à la capacité actuelle.")
            return 0
        except Exception as e:
            print(f"❌ Erreur de chargement du buffer : {e}")
            return -1

    def reset_all_noisy_sigma(self, sigma0=0.5):
        """
        Reset les sigma_weight et sigma_bias des NoisyLinear comme dans l'ancien invaders.py.
        Gère le cas dueling et non-dueling, MLP/CNN.
        """
        for name, module in self.dqn.named_modules():
            if isinstance(module, NoisyLinear):
                reset = False
                # MLP
                if self.config.model_type.lower() == 'mlp':
                    if 'mlp_layers.0' in name:
                        N = module.in_features
                        reset = True
                    elif name.endswith('output_layer'):
                        N = module.out_features
                        reset = True
                    elif name.endswith('value_head') or name.endswith('advantage_head'):
                        N = module.out_features
                        reset = True
                # CNN
                elif self.config.model_type.lower() == 'cnn':
                    if name.endswith('fc1'):
                        N = module.in_features #(on va considérer les neurone de la couche hidden car sinon sigma_fc1 trop petit )
                        reset = True
                    elif name.endswith('output_layer'):
                        N = module.out_features
                        reset = True
                    elif name.endswith('value_head') or name.endswith('advantage_head'):
                        N = module.out_features
                        reset = True
                if reset:
                    init_val = sigma0 / math.sqrt(N)
                    with torch.no_grad():
                        module.sigma_weight.fill_(init_val)
                        module.sigma_bias.fill_(init_val)
                    print(f"🔁 Reset sigma de {name} à {init_val:.3f}")
