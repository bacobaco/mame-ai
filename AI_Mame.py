# Dans AI_Mame.py

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random, os
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass

@dataclass
class TrainingConfig:
    """Configuration class for DQN training parameters"""
    input_size: any           # Pour MLP, un int ; pour CNN, un tuple (channels, height, width)
    hidden_size: int
    output_size: int
    hidden_layers: int
    learning_rate: float
    gamma: float
    epsilon_start: float
    epsilon_end: float
    epsilon_decay: float
    epsilon_add: float
    buffer_capacity: int
    batch_size: int
    state_history_size: int
    device: str = "cuda:0" if torch.cuda.is_available() else "cpu"
    double_dqn: bool = False    # Active le Double DQN si True
    model_type: str = "mlp"     # Choix de l'architecture : "mlp" ou "cnn"

class ReplayBuffer:
    """Generic replay buffer for DQN training"""
    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)

    def push(self, state: np.ndarray, action: int, reward: float, 
             next_state: np.ndarray, done: bool) -> None:
        state = np.expand_dims(state, 0)
        next_state = np.expand_dims(next_state, 0)
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int) -> Tuple:
        state, action, reward, next_state, done = zip(
            *random.sample(self.buffer, batch_size)
        )
        return np.concatenate(state), action, reward, np.concatenate(next_state), done

    def __len__(self) -> int:
        return len(self.buffer)

# Architecture MLP existante (DQN)
class DQN(nn.Module):
    def __init__(self, config: TrainingConfig):
        super(DQN, self).__init__()
        self.config = config
        
        # Couche d'entrée
        self.fc_input = nn.Linear(config.input_size, config.hidden_size)
        
        # Couches cachées
        self.fc_hidden = nn.ModuleList(
            [nn.Linear(config.hidden_size, config.hidden_size) 
             for _ in range(config.hidden_layers)]
        )
        
        # Couche de sortie
        self.fc_output = nn.Linear(config.hidden_size, config.output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.fc_input(x))
        for layer in self.fc_hidden:
            x = torch.relu(layer(x))
        return self.fc_output(x)

# Nouvelle architecture CNN
class CNN_DQN(nn.Module):
    def __init__(self, config: TrainingConfig):
        super(CNN_DQN, self).__init__()
        # Pour le CNN, on attend que input_size soit un tuple (channels, height, width)
        if not isinstance(config.input_size, tuple):
            raise ValueError("Pour CNN_DQN, input_size doit être un tuple (channels, height, width)")
        channels, height, width = config.input_size
        
        # Définition de deux couches convolutionnelles
        self.conv1 = nn.Conv2d(in_channels=channels, out_channels=16, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=4, stride=2)
        
        # Calcul de la taille de sortie après convolutions
        def conv2d_output_size(size, kernel_size, stride):
            return (size - kernel_size) // stride + 1
        
        convw = conv2d_output_size(conv2d_output_size(width, 8, 4), 4, 2)
        convh = conv2d_output_size(conv2d_output_size(height, 8, 4), 4, 2)
        linear_input_size = convw * convh * 32
        
        # Couches entièrement connectées
        self.fc1 = nn.Linear(linear_input_size, config.hidden_size)
        self.fc2 = nn.Linear(config.hidden_size, config.output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x doit être de taille (batch, channels, height, width)
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        return self.fc2(x)

class DQNTrainer:
    """Handles DQN training process independent of specific game"""
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.device = torch.device(config.device)
        
        # Choix de l'architecture en fonction du paramètre model_type
        if config.model_type.lower() == "cnn":
            self.dqn = CNN_DQN(config).to(self.device)
            self.target_dqn = CNN_DQN(config).to(self.device)
        else:
            self.dqn = DQN(config).to(self.device)
            self.target_dqn = DQN(config).to(self.device)
            
        self.copy_weights(self.dqn, self.target_dqn)
        
        # Initialisation des composants d'entraînement
        self.optimizer = optim.Adam(self.dqn.parameters(), lr=config.learning_rate)
        self.criterion = nn.MSELoss()
        self.replay_buffer = ReplayBuffer(config.buffer_capacity)
        
        # État d'entraînement
        self.epsilon = config.epsilon_start
        self.state_history = deque(maxlen=config.state_history_size)

    @staticmethod
    def copy_weights(source: nn.Module, target: nn.Module) -> None:
        """Copie les poids du réseau source vers le réseau cible"""
        target.load_state_dict(source.state_dict())

    def select_action(self, state: np.ndarray) -> int:
        """Sélectionne une action via une politique epsilon-greedy"""
        if random.random() < self.epsilon:
            return random.randint(0, self.config.output_size - 1)
        
        with torch.no_grad():
            # Pour le CNN, il faudra s'assurer que l'état est correctement mis en forme (batch, channels, height, width)
            state_tensor = torch.tensor(state, dtype=torch.float32).to(self.device)
            q_values = self.dqn(state_tensor)
            return torch.argmax(q_values).item()

    def update_epsilon(self, mean_score: float, mean_score_old: float) -> None:
        """Mise à jour d'epsilon en fonction du progrès de l'entraînement"""
        self.epsilon = max(
            min(self.epsilon * self.config.epsilon_decay, self.config.epsilon_start),
            self.config.epsilon_end
        )
        if mean_score_old > mean_score:
            self.epsilon += self.config.epsilon_add

    def train_step(self) -> Optional[float]:
        """Exécute une étape d'entraînement si suffisamment d'échantillons sont disponibles"""
        if len(self.replay_buffer) < self.config.batch_size:
            return None

        # Échantillonnage du replay buffer
        state_batch, action_batch, reward_batch, next_state_batch, done_batch = \
            self.replay_buffer.sample(self.config.batch_size)

        # Conversion en tenseurs
        state_batch = torch.tensor(state_batch, dtype=torch.float32).to(self.device)
        action_batch = torch.tensor(action_batch, dtype=torch.int64).to(self.device)
        reward_batch = torch.tensor(reward_batch, dtype=torch.float32).to(self.device)
        next_state_batch = torch.tensor(next_state_batch, dtype=torch.float32).to(self.device)
        done_batch = torch.tensor(done_batch, dtype=torch.float32).to(self.device)

        # Calcul des Q-values courantes
        curr_q_values = self.dqn(state_batch).gather(1, action_batch.unsqueeze(-1)).squeeze(-1)

        with torch.no_grad():
            if self.config.double_dqn:
                # Double DQN : sélection de la meilleure action via le réseau principal
                best_actions = self.dqn(next_state_batch).argmax(dim=1, keepdim=True)
                next_q_values = self.target_dqn(next_state_batch).gather(1, best_actions).squeeze(1)
            else:
                # DQN classique
                next_q_values = self.target_dqn(next_state_batch).max(1)[0]
            target_q_values = reward_batch + self.config.gamma * (1 - done_batch) * next_q_values

        # Calcul de la loss et mise à jour des poids
        loss = self.criterion(curr_q_values, target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def save_model(self, path: str) -> None:
        """Sauvegarde les poids du modèle dans un fichier"""
        torch.save(self.dqn.state_dict(), path)

    def load_model(self, path: str) -> None:
        """Charge les poids du modèle depuis un fichier"""
        if os.path.exists(path):
            self.dqn.load_state_dict(torch.load(path, map_location=self.device))
        else:
            print("\033[33m" + f"Le fichier {path} n'existe pas. Impossible de charger le modèle." + "\033[0m")
        self.copy_weights(self.dqn, self.target_dqn)

    def update_state_history(self, state: np.ndarray) -> np.ndarray:
        """Met à jour l'historique d'état et retourne l'état concaténé"""
        self.state_history.append(state)
        return np.concatenate(list(self.state_history))
