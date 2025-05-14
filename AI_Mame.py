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
    rainbow_eval: int = 250_000 # Nombre de step avant chaque phase d'évaluation Rainbow (250_000 dans Rainbow)
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
    mode: str = "exploration"  # "exploration" pour l'entraînement, "exploitation" (inference) pour la phase finale

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
            else: # MLP
                if not isinstance(config.input_size, int):
                    raise ValueError(f"For MLP, config.input_size must be an int, got {config.input_size}")
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
        if self.store_on_cpu: # CNN with memory optimization
            # Assuming state and next_state are numpy arrays in [0,1] float
            self.states[self.pos] = (state * 255).astype(np.uint8)
            self.next_states[self.pos] = (next_state * 255).astype(np.uint8)
        else: # MLP or CNN without memory optimization
            # Ensure states are tensors on the correct device
            if not torch.is_tensor(state):
                state = torch.tensor(state, dtype=torch.float32)
            if not torch.is_tensor(next_state):
                next_state = torch.tensor(next_state, dtype=torch.float32)

            self.states[self.pos].copy_(state.to(self.device))
            self.next_states[self.pos].copy_(next_state.to(self.device))

        self.actions[self.pos] = torch.tensor(action, dtype=torch.int64).to(self.device)
        self.rewards[self.pos] = torch.tensor(reward, dtype=torch.float32).to(self.device)
        self.dones[self.pos] = torch.tensor(float(done), dtype=torch.float32).to(self.device)

        self.actions[self.pos] = action
        self.rewards[self.pos] = reward
        self.dones[self.pos] = float(done)

        if self.prioritized:
            max_priority = self.priorities[: self.size].max() if self.size > 0 else 1.0
            self.priorities[self.pos] = max_priority

        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)


    def sample(self, batch_size):
        if self.prioritized:
            if self.size == 0: return None # Cannot sample from empty buffer
            # Calculate probabilities from priorities
            prios = self.priorities[:self.size] ** self.alpha
            probs = prios / prios.sum()
            indices = torch.multinomial(probs, batch_size, replacement=True) # Allow replacement if batch_size > self.size
            
            # Anneal beta
            self.beta = min(1.0, self.beta + self.beta_increment)
            weights = (self.size * probs[indices]) ** (-self.beta)
            weights = weights / weights.max() # Normalize weights
        else:
            indices = torch.randint(0, self.size, (batch_size,), device=self.device)
            weights = torch.ones(batch_size, device=self.device) # Uniform weights if not prioritized

        if self.store_on_cpu:
            states_batch = torch.tensor(self.states[indices.cpu().numpy()], dtype=torch.float32, device=self.device) / 255.0
            next_states_batch = torch.tensor(self.next_states[indices.cpu().numpy()], dtype=torch.float32, device=self.device) / 255.0
        else:
            states_batch = self.states[indices]
            next_states_batch = self.next_states[indices]

        actions_batch = self.actions[indices]
        rewards_batch = self.rewards[indices]
        dones_batch = self.dones[indices]


        return (
            states_batch,
            actions_batch,
            rewards_batch,
            next_states_batch,
            dones_batch,
            indices, # For PER updates
            weights, # For PER loss weighting
        )            
    def old_sample(self, batch_size, beta=None):
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
        if not self.prioritized:
            return
        # Ensure batch_priorities is a tensor on the same device
        if not torch.is_tensor(batch_priorities):
            batch_priorities = torch.tensor(batch_priorities, dtype=torch.float32)
        batch_priorities = batch_priorities.to(self.device)
        
        self.priorities[batch_indices] = batch_priorities.abs() + 1e-5 # Add epsilon to avoid zero priority

    def __len__(self):
        return self.size
    
class NStepTransitionWrapper:
    def __init__(self, n, gamma):
        self.n = n
        self.gamma = gamma
        self.buffer = deque() # Using deque for efficient appends and pops from left

    def append(self, state, action, reward, done, next_state):
        self.buffer.append((state, action, reward, next_state, done)) # Storing next_state, done at the end

        if len(self.buffer) < self.n:
            return None  # Not enough transitions for an n-step return yet

        # Calculate n-step return
        n_step_reward = 0.0
        discount = 1.0
        
        # The actual next_state and done for the n-step transition
        # will be from the n-th transition in the buffer
        final_next_state = self.buffer[-1][3] # next_state of the n-th transition
        final_done = self.buffer[-1][4]       # done state of the n-th transition

        for i in range(self.n):
            s, a, r, ns, d = self.buffer[i]
            n_step_reward += discount * r
            if d: # If any intermediate state is done, the n-step sequence terminates there
                final_next_state = ns # This was the state that led to termination
                final_done = True
                # The reward calculation should stop here for this path
                # The effective n for this transition becomes i+1
                break 
            discount *= self.gamma

        # The state and action are from the first transition in the n-step sequence
        s0, a0, _, _, _ = self.buffer.popleft()

        # If the loop broke early due to an intermediate 'done', final_done is True.
        # If the loop completed self.n steps without an intermediate 'done',
        # final_done is the 'done' status of the n-th transition.
        return s0, a0, n_step_reward, final_next_state, final_done


    def flush(self):
        # Flushes remaining transitions when an episode ends before a full n-step sequence is formed
        multi_step_transitions = []
        while len(self.buffer) > 0:
            n_step_reward = 0.0
            discount = 1.0
            final_next_state = self.buffer[-1][3]
            final_done = self.buffer[-1][4]
            
            # Effective n for this flush is the current buffer size
            current_n = len(self.buffer)
            for i in range(current_n):
                s, a, r, ns, d = self.buffer[i]
                n_step_reward += discount * r
                if d:
                    final_next_state = ns
                    final_done = True
                    break
                discount *= self.gamma
            
            s0, a0, _, _, _ = self.buffer.popleft()
            multi_step_transitions.append((s0, a0, n_step_reward, final_next_state, final_done))
        return multi_step_transitions

class NoisyLinear(nn.Module):
    def __init__(self, in_features, out_features, sigma_init=0.5):
        super(NoisyLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        # sigma_init est la valeur de base pour sigma (sigma_0 dans le papier)
        # Elle sera utilisée dans reset_parameters pour calculer la valeur finale de sigma.
        self.sigma_init_value = sigma_init # Stocker sigma_init pour l'utiliser dans reset_parameters

        # Création des paramètres mu (poids moyens)
        self.mu_weight = nn.Parameter(torch.empty(out_features, in_features))
        self.mu_bias = nn.Parameter(torch.empty(out_features))

        # Création des paramètres sigma (paramètres de bruit)
        # Leurs valeurs initiales ici (sigma_init) seront écrasées par reset_parameters()
        self.sigma_weight = nn.Parameter(torch.full((out_features, in_features), self.sigma_init_value))
        self.sigma_bias = nn.Parameter(torch.full((out_features,), self.sigma_init_value))

        # Permettre l'entraînement de sigma
        self.sigma_weight.requires_grad = True
        self.sigma_bias.requires_grad = True

        # Buffers pour le bruit epsilon (non entraînables)
        self.register_buffer("epsilon_weight", torch.zeros(out_features, in_features))
        self.register_buffer("epsilon_bias", torch.zeros(out_features))

        self.sigma_init_flag = False # Indicateur pour savoir si sigma_init a été utilisé plutot que sigma0/sqrt(p)

        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self):
        # Initialisation des poids moyens (mu)
        mu_range = 1 / math.sqrt(self.in_features) if not self.sigma_init_flag else 1
        self.mu_weight.data.uniform_(-mu_range, mu_range)
        self.mu_bias.data.uniform_(-mu_range, mu_range)

        # Initialisation des paramètres de bruit (sigma)
        # selon la recommandation du papier pour le bruit factorisé: sigma_0 / sqrt(p)
        # où sigma_0 est self.sigma_init_value et p est self.in_features
        
        # Pour les poids sigma
        sigma_w_initial_value = self.sigma_init_value / math.sqrt(self.in_features) if not self.sigma_init_flag else self.sigma_init_value
        self.sigma_weight.data.fill_(sigma_w_initial_value)

        # Pour les biais sigma
        # Le papier (Eq. 11) utilise f(epsilon_j) pour epsilon_j^b,
        # et sigma^b accompagne ce bruit. Si on suit la logique de sigma_0 / sqrt(p_bias),
        # et que p_bias (nombre d'entrées pour le biais) est 1, alors sigma_b = sigma_0.
        # Cependant, il est aussi courant d'utiliser la même initialisation que pour les poids par simplicité.
        # Option 1: Suivre strictement la logique p_bias=1 -> sigma_b = sigma_0
        # self.sigma_bias.data.fill_(self.sigma_init_value)
        # Option 2: Utiliser la même mise à l'échelle que pour les poids (plus courant par analogie)
        sigma_b_initial_value = self.sigma_init_value / math.sqrt(self.in_features) if not self.sigma_init_flag else self.sigma_init_value
        self.sigma_bias.data.fill_(sigma_b_initial_value)

        # Note: La ligne `if torch.all(self.sigma_weight.data == self.sigma_init_value)...`
        # que nous avions précédemment n'est plus nécessaire ici si reset_parameters
        # est la source de vérité pour l'initialisation.

    def _scale_noise(self, size):
        x = torch.randn(size)
        return x.sign() * torch.sqrt(torch.abs(x))

    def reset_noise(self):
        device = self.mu_weight.device # S'assurer que le bruit est sur le même device que les paramètres
        epsilon_in = self._scale_noise(self.in_features).to(device)
        epsilon_out = self._scale_noise(self.out_features).to(device)
        self.epsilon_weight.copy_(epsilon_out.ger(epsilon_in)) # Utiliser copy_ pour les buffers
        self.epsilon_bias.copy_(epsilon_out)                   # Utiliser copy_ pour les buffers

    def forward(self, input):
        if self.training:
            # Il est bon de s'assurer que reset_noise() a été appelé avant cette passe
            # si l'intention est d'avoir un nouveau bruit pour chaque forward pass en training.
            # Typiquement, DQNTrainer.select_action ou DQNTrainer.train_step s'en chargerait.
            weight = self.mu_weight + self.sigma_weight * self.epsilon_weight
            bias = self.mu_bias + self.sigma_bias * self.epsilon_bias
        else:
            weight = self.mu_weight
            bias = self.mu_bias
        return F.linear(input, weight, bias)

    def get_sigma(self):
        # Retourne la moyenne absolue des paramètres sigma (pour monitoring)
        return self.sigma_weight.abs().mean().item(), self.sigma_bias.abs().mean().item()


class OLD_NoisyLinear(nn.Module):
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
        
        # S'assurer que sigma est initialisé à sigma_init si ce n'est pas déjà modifié
        # (par exemple, par un chargement de state_dict ou une modification manuelle)
        # Cette condition est un peu simpliste, mais vise à éviter de réinitialiser des sigmas déjà appris.
        # Une meilleure approche pourrait impliquer de vérifier si les données sont exactement égales à sigma_init.
        if torch.all(self.sigma_weight.data == self.sigma_init) and torch.all(self.sigma_bias.data == self.sigma_init):
             pass # Déjà initialisé ou pas besoin de réinitialiser
        else: # Ou si vous voulez toujours forcer l'initialisation de sigma à sigma_init au début:
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

        elif config.model_type.lower() == "mlp":
            # Calcul de la dimension d'entrée pour le MLP
            # S'assurer que config.input_size est un entier pour MLP
            if not isinstance(config.input_size, int):
                raise ValueError(f"Pour MLP, config.input_size doit être un entier (taille du vecteur d'état), reçu: {config.input_size}")
            input_dim = config.input_size * config.state_history_size
            self.activation = nn.LeakyReLU(0.01) # Activation commune pour les couches MLP

            # --- MODIFICATION COMMENCE ICI ---
            self.hidden_modules = nn.ModuleList()
            current_dim = input_dim
            for _ in range(config.hidden_layers):
                self.hidden_modules.append(
                    nn.Sequential(
                        Linear (current_dim, config.hidden_size),
                        nn.BatchNorm1d(config.hidden_size), # BatchNorm ajouté ici
                        self.activation # Activation après BatchNorm
                    )
                )
                current_dim = config.hidden_size # La dimension d'entrée de la couche suivante est la sortie de la précédente
            # --- MODIFICATION FIN ICI ---

            # Couches de sortie (Dueling ou simple)
            if config.dueling:
                self.value_head = Linear (current_dim, 1) # current_dim est hidden_size après les boucles
                self.advantage_head = Linear (current_dim, config.output_size)
            else:
                self.output_layer = Linear (current_dim, config.output_size)
        else:
            raise NotImplementedError(f"Unknown model_type: {config.model_type}")

    def forward(self, x):
        if self.config.model_type.lower() == "cnn":
            x = self.encoder(x)
            x = x.view(x.size(0), -1)
            x = self.fc1(x)
            x = self.bn_fc1(x) # BatchNorm avant l'activation pour la couche FC du CNN
            x = self.act_fc1(x)

            if self.config.dueling:
                value = self.value_head(x)
                advantage = self.advantage_head(x)
                return value + (advantage - advantage.mean(dim=1, keepdim=True))
            else:
                return self.output_layer(x)
        elif self.config.model_type.lower() == "mlp":
            # x est déjà de forme (batch, input_dim) après la concaténation de l'historique d'état
            for hidden_block in self.hidden_modules:
                x = hidden_block(x)
            if self.config.dueling:
                value = self.value_head(x)
                advantage = self.advantage_head(x)
                return value + (advantage - advantage.mean(dim=1, keepdim=True))
            else:
                return self.output_layer(x)

    def get_sigma_values(self):
        # S'assure que le modèle utilise NoisyLinear
        if not self.config.use_noisy:
            return {} # Retourne un dictionnaire vide si NoisyNet n'est pas utilisé
        return {
            name: module.sigma_weight.abs().mean().item()
            for name, module in self.named_modules()
            if isinstance(module, NoisyLinear)
        }

    def reset_noise(self):
        if not self.config.use_noisy: return
        for module in self.modules():
            if isinstance(module, NoisyLinear):  module.reset_noise()

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
        action: int

        if self.config.use_noisy:
            # For NoisyNet, model mode is set by self.set_mode().
            # If in exploration mode, self.dqn is .train(), NoisyLinear adds noise.
            # If in exploitation mode, self.dqn is .eval(), NoisyLinear uses mu_weights.
            if self.config.mode == "exploration": # Implies self.dqn.training is True
                self.dqn.reset_noise() # Sample new noise for this action selection

                # --- MODIFICATION START ---
                # Temporarily set BatchNorm1d layers to eval mode if the model is in train mode
                # to handle batch_size=1 correctly, while allowing NoisyLinear to add noise.
                bn_layers_original_training_state = []
                if self.dqn.training: # Check if the main model is indeed in training mode
                    for module in self.dqn.modules():
                        if isinstance(module, nn.BatchNorm1d):
                            bn_layers_original_training_state.append((module, module.training))
                            module.eval() # Force BatchNorm1d to use running stats
                
                try:
                    with torch.no_grad():
                        q_values = self.dqn(state_tensor)
                finally:
                    # Restore original training state of BatchNorm1d layers
                    for module, original_state in bn_layers_original_training_state:
                        if original_state: # Only set back to train if it was originally in train mode
                            module.train()
                        # else: module.eval(); # If it was eval, it remains eval
                # --- MODIFICATION END ---
            else: # Exploitation mode, self.dqn is already .eval()
                with torch.no_grad():
                    q_values = self.dqn(state_tensor)
            
            action = torch.argmax(q_values).item()

        else: # Epsilon-greedy
            original_training_state = self.dqn.training
            self.dqn.eval() # Set to eval mode for action selection
            try:
                if self.config.mode == "exploitation" or random.random() >= self.epsilon:
                    with torch.no_grad():
                        q_values = self.dqn(state_tensor)
                    action = torch.argmax(q_values).item()
                else: # Explore
                    action = random.randint(0, self.config.output_size - 1)
            finally:
                if original_training_state: # Restore original mode only if it was training
                    self.dqn.train()
                # If it was eval, it stays eval, which is fine.
        return action            
    def update_epsilon(self):
        if self.config.use_noisy: return # No epsilon decay for NoisyNet
        if self.config.epsilon_linear > 0:
            self.epsilon = max(self.config.epsilon_end, self.epsilon - self.config.epsilon_linear)
        else: # Exponential decay
            self.epsilon = max(self.config.epsilon_end, self.epsilon * self.config.epsilon_decay)

    def train_step(self) -> Optional[float]:
        if self.config.rainbow_eval > 0:
            if (self.training_steps + 1) % self.config.rainbow_eval == 0 and self.eval_steps == 0:
                print(f"{Fore.MAGENTA}🌈 Starting Rainbow evaluation phase...{Style.RESET_ALL}")
                self.set_mode("exploitation") # Switch to exploitation for evaluation
                self.eval_steps = 1
            if self.eval_steps > 0:
                self.eval_steps += 1
                if self.eval_steps >= (self.config.rainbow_eval_pourcent / 100 * self.config.rainbow_eval):
                    print(f"{Fore.MAGENTA}🌈 ...Rainbow evaluation phase ended. Resuming exploration.{Style.RESET_ALL}")
                    self.set_mode("exploration") # Switch back to exploration
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
               
        if not torch.isfinite(loss):
            print(f"{Fore.RED}⚠️ [train_step] Step {self.training_steps}: loss non finie ! ({loss.item()}){Style.RESET_ALL}")
        elif self.training_steps % 5000 == 0: # Log loss periodically
            print(f"{Fore.GREEN}✅ [train_step] Step {self.training_steps}: loss = {loss.item():.4f}{Style.RESET_ALL}")

        return loss.item()
    def save_model(self, path: str) -> None:
        torch.save(self.dqn.state_dict(), path)
        print(f"{Fore.CYAN}💾 Modèle sauvegardé dans {path}{Style.RESET_ALL}")
    def load_model(self, path: str) -> None:
        if not os.path.exists(path):
            print(f"{Fore.YELLOW}⚠️ Le fichier {path} n'existe pas. Entraînement à partir de zéro.{Style.RESET_ALL}")
            return
        try:
            self.dqn.load_state_dict(torch.load(path, map_location=self.device))
            self.copy_weights(self.dqn, self.target_dqn) # Ensure target network is also updated
            print(f"{Fore.GREEN}✅ Modèle chargé depuis {path}{Style.RESET_ALL}")
        except Exception as e:
            print(f"{Fore.RED}❌ Erreur lors du chargement du modèle depuis {path}: {e}{Style.RESET_ALL}")
            print(f"{Fore.YELLOW}⚠️ Entraînement à partir de zéro suite à l'erreur de chargement.{Style.RESET_ALL}")

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

    def save_buffer(self, filename="buffer.pt"): # Changed extension to .pt for PyTorch convention
        try:
            # It's better to save the components of the buffer rather than the whole object
            # if the object contains non-serializable parts or complex device logic.
            # However, if GPUReplayBuffer is designed to be serializable:
            torch.save({
                'states': self.replay_buffer.states[:self.replay_buffer.size].cpu() if self.replay_buffer.store_on_cpu else self.replay_buffer.states[:self.replay_buffer.size].cpu(),
                'actions': self.replay_buffer.actions[:self.replay_buffer.size].cpu(),
                'rewards': self.replay_buffer.rewards[:self.replay_buffer.size].cpu(),
                'next_states': self.replay_buffer.next_states[:self.replay_buffer.size].cpu() if self.replay_buffer.store_on_cpu else self.replay_buffer.next_states[:self.replay_buffer.size].cpu(),
                'dones': self.replay_buffer.dones[:self.replay_buffer.size].cpu(),
                'priorities': self.replay_buffer.priorities[:self.replay_buffer.size].cpu() if self.config.prioritized_replay else None,
                'pos': self.replay_buffer.pos,
                'size': self.replay_buffer.size,
                'store_on_cpu_flag_during_save': self.replay_buffer.store_on_cpu # Save the flag used during save
            }, filename)
            print(f"{Fore.CYAN}💾 Buffer sauvegardé dans {filename} ({self.replay_buffer.size} transitions){Style.RESET_ALL}")
        except Exception as e:
            print(f"{Fore.RED}❌ Erreur de sauvegarde du buffer : {e}{Style.RESET_ALL}")

    def load_buffer(self, filename="buffer.pt"):
        if not os.path.exists(filename):
            print(f"{Fore.YELLOW}🟡 Aucun buffer sauvegardé trouvé ({filename}).{Style.RESET_ALL}")
            return -1 # Indicate buffer was not loaded
        try:
            checkpoint = torch.load(filename, map_location='cpu') # Load to CPU first
            
            num_to_load = min(checkpoint['size'], self.replay_buffer.capacity)
            print(f"{Fore.GREEN}🟢 Démarrage du chargement du buffer ({num_to_load} transitions depuis {filename}){Style.RESET_ALL}")

            # Determine if the loaded states were stored as uint8 (optimized CNN)
            # The flag 'store_on_cpu_flag_during_save' tells us how they were saved.
            loaded_states_are_uint8 = checkpoint.get('store_on_cpu_flag_during_save', False)

            for i in range(num_to_load):
                state = checkpoint['states'][i]
                next_state = checkpoint['next_states'][i]

                if loaded_states_are_uint8: # If saved as uint8, convert back to float [0,1]
                    state = state.astype(np.float32) / 255.0
                    next_state = next_state.astype(np.float32) / 255.0
                
                # The push method handles moving to the correct device and dtype for storage
                self.replay_buffer.push(
                    state, # Numpy array or tensor
                    checkpoint['actions'][i].item(),
                    checkpoint['rewards'][i].item(),
                    next_state, # Numpy array or tensor
                    bool(checkpoint['dones'][i].item())
                )
                if self.config.prioritized_replay and checkpoint['priorities'] is not None:
                    # Find the position where the pushed element was stored.
                    # self.replay_buffer.pos is the *next* position to write.
                    # So, the element was written at (self.replay_buffer.pos - 1 + self.replay_buffer.capacity) % self.replay_buffer.capacity
                    idx_in_buffer = (self.replay_buffer.pos - 1 + self.replay_buffer.capacity) % self.replay_buffer.capacity
                    self.replay_buffer.priorities[idx_in_buffer] = checkpoint['priorities'][i].to(self.device)


                if (i + 1) % (num_to_load // 100 + 1) == 0 or i == num_to_load - 1: # Progress update
                    print(f"\r🔄 Chargement du buffer : {i+1}/{num_to_load} ({((i+1)/num_to_load*100):.1f}%)", end="")
            
            # If fewer transitions were loaded than the buffer's current reported size, adjust.
            # This scenario is less likely with the current push logic which updates size correctly.
            # self.replay_buffer.size = num_to_load 
            # self.replay_buffer.pos = num_to_load % self.replay_buffer.capacity

            print(f"\n{Fore.GREEN}✅ Chargement du buffer terminé.{Style.RESET_ALL}")
            return 0 # Indicate success
        except Exception as e:
            print(f"{Fore.RED}❌ Erreur de chargement du buffer : {e}{Style.RESET_ALL}")
            return -1 # Indicate failure
