# Dans AI_Mame.py
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random, os, sys
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass
from datetime import datetime
import atexit
from flask import Flask, send_from_directory, render_template_string
from colorama import Fore, Style

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
    batch_size: int
    device: str = "cuda:0" if torch.cuda.is_available() else "cpu"
    double_dqn: bool = False  # Active le Double DQN si True
    prioritized_replay: bool = False
    model_type: str = "mlp"  # Choix de l'architecture : "mlp" ou "cnn"
    cnn_type: str = "default"  # ✅ AJOUTER ICI
    state_extractor: callable = None  # Fonction d'extraction d'état
    mode: str = (
        "exploration"  # "exploration" pour l'entraînement, "exploitation" (inference) pour la phase finale
    )
    target_update_freq: int = 5000  # 🔁 Fréquence de mise à jour du réseau cible

class GPUReplayBuffer:
    def __init__(self, capacity, config, prioritized=True, alpha=0.6):
        self.capacity = capacity
        self.config = config
        self.device = torch.device(config.device)
        self.prioritized = prioritized
        self.pos = 0
        self.size = 0

        # Déterminer la forme d'un état selon le type de modèle
        if config.model_type.lower() == "cnn":
            state_shape = config.input_size  # ex: (channels, height, width)
        else:
            state_shape = (config.input_size * config.state_history_size,)

        # Préallouer les tenseurs sur GPU
        self.states = torch.empty(
            (capacity, *state_shape), dtype=torch.float32, device=self.device
        )
        self.next_states = torch.empty(
            (capacity, *state_shape), dtype=torch.float32, device=self.device
        )
        self.actions = torch.empty((capacity,), dtype=torch.int64, device=self.device)
        self.rewards = torch.empty((capacity,), dtype=torch.float32, device=self.device)
        self.dones = torch.empty((capacity,), dtype=torch.float32, device=self.device)

        if self.prioritized:
            # Stocker les priorités directement sur GPU
            self.priorities = torch.zeros(
                capacity, dtype=torch.float32, device=self.device
            )
            self.alpha = alpha
            self.beta = 0.4  # Valeur initiale
            self.beta_increment = 0.000001
        else:
            self.priorities = None

    def push(self, state, action, reward, next_state, done):
        # Convertir state et next_state en tenseurs sur GPU si besoin
        if not torch.is_tensor(state):
            state = torch.tensor(state, dtype=torch.float32, device=self.device)
        if not torch.is_tensor(next_state):
            next_state = torch.tensor(
                next_state, dtype=torch.float32, device=self.device
            )
        if self.config.model_type.lower() == "mlp":
            state = state.flatten()
            next_state = next_state.flatten()

        self.states[self.pos].copy_(state)
        self.actions[self.pos] = action
        self.rewards[self.pos] = reward
        self.next_states[self.pos].copy_(next_state)
        self.dones[self.pos] = float(done)

        if self.prioritized:
            # Utiliser la valeur maximale actuelle ou 1.0 par défaut
            max_priority = self.priorities[: self.size].max() if self.size > 0 else 1.0
            self.priorities[self.pos] = max_priority

        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size, beta=None):
        if self.size == 0:
            return None
        if self.prioritized:
            if beta is None:
                beta = self.beta
            # Calculer les probabilités d'échantillonnage sur GPU
            prios = self.priorities[: self.size] ** self.alpha
            probs = prios / prios.sum()
            # torch.multinomial pour échantillonner les indices
            indices = torch.multinomial(probs, batch_size, replacement=False)
            weights = (self.size * probs[indices]) ** (-beta)
            weights = weights / weights.max()
            states_batch = self.states.index_select(0, indices)
            actions_batch = self.actions.index_select(0, indices)
            rewards_batch = self.rewards.index_select(0, indices)
            next_states_batch = self.next_states.index_select(0, indices)
            dones_batch = self.dones.index_select(0, indices)
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
        else:
            # Échantillonnage uniforme sur GPU
            indices = torch.randperm(self.size, device=self.device)[:batch_size]
            weights = torch.ones(batch_size, device=self.device)
            states_batch = self.states.index_select(0, indices)
            actions_batch = self.actions.index_select(0, indices)
            rewards_batch = self.rewards.index_select(0, indices)
            next_states_batch = self.next_states.index_select(0, indices)
            dones_batch = self.dones.index_select(0, indices)
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
        self.priorities[batch_indices] = batch_priorities

    def __len__(self):
        return self.size

class OLD_NoisyLinear(nn.Module):
    def __init__(self, in_features, out_features, sigma_init=0.017):
        super(NoisyLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Learnable parameters
        self.mu_weight = nn.Parameter(torch.empty(out_features, in_features))
        self.sigma_weight = nn.Parameter(torch.empty(out_features, in_features))
        self.register_buffer("epsilon_weight", torch.zeros(out_features, in_features))

        self.mu_bias = nn.Parameter(torch.empty(out_features))
        self.sigma_bias = nn.Parameter(torch.empty(out_features))
        self.register_buffer("epsilon_bias", torch.zeros(out_features))

        self.sigma_init = sigma_init
        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self):
        mu_range = 1 / self.in_features**0.5
        self.mu_weight.data.uniform_(-mu_range, mu_range)
        self.mu_bias.data.uniform_(-mu_range, mu_range)
        self.sigma_weight.data.fill_(self.sigma_init)
        self.sigma_bias.data.fill_(self.sigma_init)

    def _scale_noise(self, size):
        x = torch.randn(size)
        return x.sign().mul_(x.abs().sqrt_())


    def reset_noise(self):
        device = self.mu_weight.device  # Récupère le bon device (CPU ou GPU)

        epsilon_in = self._scale_noise(self.in_features).to(device)
        epsilon_out = self._scale_noise(self.out_features).to(device)

        self.epsilon_weight = epsilon_out.ger(epsilon_in)  # Produit extérieur (outer product)
        self.epsilon_bias = epsilon_out


    def forward(self, x):
        if self.training:
            weight = self.mu_weight + self.sigma_weight * self.epsilon_weight
            bias = self.mu_bias + self.sigma_bias * self.epsilon_bias
        else:
            weight = self.mu_weight
            bias = self.mu_bias
        return torch.nn.functional.linear(x, weight, bias)

    def get_sigma(self):
        return self.sigma_weight.abs().mean().item(), self.sigma_bias.abs().mean().item()

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class NoisyLinear(nn.Module):
    def __init__(self, in_features, out_features, sigma_init=0.017):
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
        self.sigma_weight.data.fill_(self.sigma_init)
        self.sigma_bias.data.fill_(self.sigma_init)

    def _scale_noise(self, size):
        x = torch.randn(size)
        return x.sign().mul_(x.abs().sqrt_())

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
            self.fc2 = Linear(config.hidden_size, config.output_size)

        elif config.model_type == "mlp":
            input_dim = (
                99 if config.state_extractor else int(np.prod(config.input_size))
            )
            self.fc1 = Linear(input_dim, config.hidden_size)
            self.bn_fc1 = nn.BatchNorm1d(config.hidden_size)
            self.act_fc1 = nn.LeakyReLU(0.01)
            self.fc2 = Linear(config.hidden_size, config.output_size)

        else:
            raise NotImplementedError("Unknown model_type")

    def forward(self, x):
        if self.config.model_type == "cnn":
            x = self.encoder(x)
            x = x.view(x.size(0), -1)
        elif self.config.model_type == "mlp":
            x = x.view(x.size(0), -1)
        else:
            raise NotImplementedError("Unknown model_type")

        x = self.fc1(x)

        # ⚠️ Applique BatchNorm uniquement si use_noisy est désactivé
        if not self.config.use_noisy:
            x = self.bn_fc1(x)

        x = self.act_fc1(x)
        x = self.fc2(x)
        return x


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

        # Initialisation des autres composants d'entraînement...
        self.optimizer = optim.Adam(self.dqn.parameters(), lr=config.learning_rate)
        self.criterion = nn.MSELoss()
        # Utilisation du ReplayBuffer classique (échantillonnage uniforme)
        self.replay_buffer = GPUReplayBuffer(
            config.buffer_capacity, config, prioritized=config.prioritized_replay
        )

        self.epsilon = config.epsilon_start
        self.state_history = deque(maxlen=config.state_history_size)
        self.criterion = nn.SmoothL1Loss(reduction="none")  # ✅ Rainbow-style

        self.training_steps = 0

    @staticmethod
    def copy_weights(source: nn.Module, target: nn.Module) -> None:
        """Copie les poids du réseau source vers le réseau cible"""
        target.load_state_dict(source.state_dict())

    def select_action(self, state: np.ndarray) -> int:
        state_tensor = torch.from_numpy(state).float().unsqueeze(0).to(self.device)

        if self.config.use_noisy:
            self.dqn.reset_noise()  # 🧠 Reset du bruit à chaque décision

        with torch.no_grad():
            was_training = self.dqn.training
            self.dqn.eval()
            q_values = self.dqn(state_tensor)
            if was_training and self.config.mode.lower() != "exploitation":
                self.dqn.train()

        if self.config.mode.lower() == "exploitation":
            return torch.argmax(q_values).item()
        else:
            if self.config.use_noisy:
                return torch.argmax(q_values).item()
            elif random.random() < self.epsilon:
                return random.randint(0, self.config.output_size - 1)
            else:
                return torch.argmax(q_values).item()

    def update_epsilon(self) -> None:
        if self.config.use_noisy or self.config.mode.lower() == "exploitation":
            return
        if self.config.epsilon_decay > 0:  # cas décroissance exponentielle decay
            self.epsilon = max(
                min(
                    self.epsilon * self.config.epsilon_decay, self.config.epsilon_start
                ),
                self.config.epsilon_end,
            )
        else:  # cas décroissance linéaire
            self.epsilon = max(
                min(
                    self.epsilon - self.config.epsilon_linear, self.config.epsilon_start
                ),
                self.config.epsilon_end,
            )

        if (
            self.config.mode.lower() == "exploitation"
            or len(self.replay_buffer) < self.config.batch_size
        ):
            return None

        (
            state_batch,
            action_batch,
            reward_batch,
            next_state_batch,
            done_batch,
            indices,
            weights,
        ) = self.replay_buffer.sample(self.config.batch_size)

        # Assurer la bonne forme des états
        if self.config.model_type.lower() == "mlp":
            state_batch = state_batch.view(self.config.batch_size, -1)
            next_state_batch = next_state_batch.view(self.config.batch_size, -1)
        elif self.config.model_type.lower() == "cnn":
            state_batch = state_batch.view(
                self.config.batch_size, *self.config.input_size
            )
            next_state_batch = next_state_batch.view(
                self.config.batch_size, *self.config.input_size
            )

        action_batch = action_batch.unsqueeze(-1)

        # --- Forward des Q-values actuelles ---
        if self.config.use_noisy:
            self.dqn.reset_noise()
            self.target_dqn.reset_noise()
        q_values = self.dqn(state_batch)
        if self.config.use_noisy:
            curr_q_values = q_values.gather(1, action_batch).reshape(-1)
        else:
            curr_q_values = q_values.gather(1, action_batch).squeeze(-1)

        # --- Target Q-values ---
        with torch.no_grad():
            if self.config.double_dqn:
                dqn_out = self.dqn(next_state_batch)
                target_out = self.target_dqn(next_state_batch)
                best_actions = dqn_out.argmax(dim=1, keepdim=True)
                if self.config.use_noisy:
                    next_q_values = target_out.gather(1, best_actions).reshape(-1)
                else:
                    next_q_values = target_out.gather(1, best_actions).squeeze(-1)
            else:
                target_out = self.target_dqn(next_state_batch)
                next_q_values = target_out.max(dim=1)[0]

            not_done = 1.0 - done_batch
            target_q_values = (
                reward_batch + self.config.gamma * not_done * next_q_values
            )

        # --- Loss ---
        #OLD diff = curr_q_values - target_q_values
        #OLD loss = (weights * diff.pow(2)).mean()
        loss = self.criterion(curr_q_values, target_q_values)
        loss = (weights * loss).mean()

        # --- Backward ---
        td_errors = diff.abs().detach()
        self.replay_buffer.update_priorities(indices, td_errors)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.training_steps += 1
        if self.training_steps % self.config.target_update_freq == 0:
            self.copy_weights(self.dqn, self.target_dqn)

    def train_step(self) -> Optional[float]:
        if (
            self.config.mode.lower() == "exploitation"
            or len(self.replay_buffer) < self.config.batch_size
        ):
            return None

        (
            state_batch,
            action_batch,
            reward_batch,
            next_state_batch,
            done_batch,
            indices,
            weights,
        ) = self.replay_buffer.sample(self.config.batch_size)

        # Format des états selon le modèle
        if self.config.model_type.lower() == "mlp":
            state_batch = state_batch.view(self.config.batch_size, -1)
            next_state_batch = next_state_batch.view(self.config.batch_size, -1)
        elif self.config.model_type.lower() == "cnn":
            state_batch = state_batch.view(
                self.config.batch_size, *self.config.input_size
            )
            next_state_batch = next_state_batch.view(
                self.config.batch_size, *self.config.input_size
            )

        action_batch = action_batch.unsqueeze(-1)

        # --- Forward des Q-values actuelles ---
        if self.config.use_noisy:
            self.dqn.reset_noise()
            self.target_dqn.reset_noise()

        q_values = self.dqn(state_batch)
        curr_q_values = (
            q_values.gather(1, action_batch).reshape(-1)
            if self.config.use_noisy
            else q_values.gather(1, action_batch).squeeze(-1)
        )

        # --- Target Q-values ---
        with torch.no_grad():
            if self.config.double_dqn:
                dqn_out = self.dqn(next_state_batch)
                target_out = self.target_dqn(next_state_batch)
                best_actions = dqn_out.argmax(dim=1, keepdim=True)
                next_q_values = (
                    target_out.gather(1, best_actions).reshape(-1)
                    if self.config.use_noisy
                    else target_out.gather(1, best_actions).squeeze(-1)
                )
            else:
                target_out = self.target_dqn(next_state_batch)
                next_q_values = target_out.max(dim=1)[0]

            not_done = 1.0 - done_batch
            target_q_values = (
                reward_batch + self.config.gamma * not_done * next_q_values
            )

        # --- Loss ---
        loss = self.criterion(curr_q_values, target_q_values)
        loss = (weights * loss).mean()
        td_errors = (curr_q_values - target_q_values).abs().detach()

        self.replay_buffer.update_priorities(indices, td_errors)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.dqn.parameters(), 10.0)  # 🛡️ Clipping ici
        self.optimizer.step()

        self.training_steps += 1
        if self.training_steps % self.config.target_update_freq == 0:
            self.copy_weights(self.dqn, self.target_dqn)
        # --- 🧠 SANITY CHECK: valeurs extrêmes ---
        if self.training_steps % 10000 == 0:  # toutes les 10000 itérations
            q_max = q_values.max().item()
            q_min = q_values.min().item()
            q_mean = q_values.mean().item()
            td_mean = td_errors.mean().item()
            td_max = td_errors.max().item()

            print(
                f"{Fore.GREEN}[Sanity]{Style.RESET_ALL} Step {self.training_steps:,} | Q=[{q_min:.2f}, {q_max:.2f}], Mean={q_mean:.2f} | TD err={td_mean:.2f}, Max={td_max:.2f}"
            )

            # 🚨 Alerte si Q-value explose
            if abs(q_max) > 1000 or abs(q_min) > 1000:
                print(
                    f"⚠️ {Fore.RED}ALERT:{Style.RESET_ALL} Q-values exploding! Q range: [{q_min:.1f}, {q_max:.1f}]"
                )

            # 🚨 Alertes supplémentaires de dérive
            if q_mean > 80:
                print(f"⚠️ {Fore.RED}ALERT:{Style.RESET_ALL} Q_mean dérive haut >80 → {q_mean:.2f}")
            if q_min < -200:
                print(f"⚠️ {Fore.RED}ALERT:{Style.RESET_ALL} Q_min très bas <200 → {q_min:.2f}")
            if td_max > 100:
                print(f"⚠️ {Fore.RED}ALERT:{Style.RESET_ALL} TD-error MAX anormal >100 → {td_max:.2f}")


        return loss.item()  # utile pour logger la loss

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
            return
        try:
            loaded = torch.load(filename, map_location=self.config.device)
            current = self.replay_buffer
            if len(loaded) <= current.capacity:
                print(f"🟢 Chargement du buffer ({len(loaded)}/{current.capacity})")
                current.__dict__.update(loaded.__dict__)
            else:
                print(f"🟠 Buffer trop gros ({len(loaded)} > {current.capacity})")
                if hasattr(loaded, "priorities") and loaded.prioritized:
                    top = torch.topk(
                        loaded.priorities[: len(loaded)], current.capacity
                    ).indices
                    for i in top:
                        current.push(
                            loaded.states[i],
                            int(loaded.actions[i].item()),
                            float(loaded.rewards[i].item()),
                            loaded.next_states[i],
                            bool(loaded.dones[i].item()),
                        )
                else:
                    for i in range(current.capacity):
                        current.push(
                            loaded.states[i],
                            int(loaded.actions[i].item()),
                            float(loaded.rewards[i].item()),
                            loaded.next_states[i],
                            bool(loaded.dones[i].item()),
                        )
                print("✅ Buffer rechargé avec adaptation à la capacité actuelle.")
        except Exception as e:
            print(f"❌ Erreur de chargement du buffer : {e}")
