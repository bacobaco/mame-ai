import os
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
from collections import deque
from torch.distributions import Normal, Independent
from torch.distributions.kl import kl_divergence

class ConvEncoder(nn.Module):
    def __init__(self, in_channels=3, latent_dim=64, image_height=192, image_width=176, activation=nn.ReLU): # Ajout de image_height, image_width
        super().__init__()
        self.activation = activation()
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=4, stride=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=4, stride=2)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=4, stride=2)

        with torch.no_grad():
            # Utilisez les dimensions d'image passées en argument
            dummy_input_shape = (1, in_channels, image_height, image_width)
            dummy = torch.zeros(dummy_input_shape, dtype=torch.float32) # Spécifier dtype peut aider
            dummy_out = self.conv4(self.activation(self.conv3(self.activation(self.conv2(self.activation(self.conv1(dummy)))))))
            self.n_flat = dummy_out.view(1, -1).size(1) # Stocker pour que DreamerTrainer puisse le passer au Decoder

        self.fc = nn.Linear(self.n_flat, latent_dim)
        print(f"[ConvEncoder] Initialized with dynamic input: {dummy_input_shape}, Flattened size for FC: {self.n_flat}, Output latent_dim: {latent_dim}")

    def forward(self, x):
        x = self.activation(self.conv1(x))
        x = self.activation(self.conv2(x))
        x = self.activation(self.conv3(x))
        x = self.activation(self.conv4(x))
        x = x.view(x.size(0), -1)
        return self.fc(x)

# Dans dreamerv2.py

class ConvDecoder(nn.Module):
    # Ajout de encoder_flat_size, target_height, target_width
    def __init__(self, latent_dim, hidden_dim_rssm, out_channels=3, encoder_flat_size=23040, target_height=192, target_width=176, activation=nn.ReLU):
        super().__init__()
        self.activation = activation()
        self.encoder_flat_size = encoder_flat_size
        self.target_height = target_height
        self.target_width = target_width

        # La couche FC prend l'état du RSSM et produit la taille aplatie de l'encodeur
        self.fc = nn.Linear(hidden_dim_rssm + latent_dim, self.encoder_flat_size)

        # Déterminer les dimensions spatiales avant aplatissement dans l'encodeur
        # C_conv_out est typiquement 256 (sortie de conv4 de l'Encoder)
        self.num_encoder_conv_filters = 256 # Supposant que c'est la sortie de la dernière conv de l'encodeur

        # H_conv_out * W_conv_out = encoder_flat_size / num_encoder_conv_filters
        # Pour reformer le tenseur avant les déconvolutions, il nous faut H_conv_out et W_conv_out.
        # On peut les recalculer en simulant les convolutions inverses ou, plus simplement,
        # les déduire en sachant qu'il y a 4 couches de stride 2.
        # H_conv_out approx target_height / 16, W_conv_out approx target_width / 16
        # Pour (192,176) -> (10,9) après l'encodeur. Pour (96,88) -> (4,3) après l'encodeur.
        # Ce calcul doit être précis.
        # Créons un mini-encodeur temporaire pour obtenir ces dimensions :
        temp_conv1 = nn.Conv2d(out_channels, 32, kernel_size=4, stride=2)
        temp_conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        temp_conv3 = nn.Conv2d(64, 128, kernel_size=4, stride=2)
        temp_conv4 = nn.Conv2d(128, self.num_encoder_conv_filters, kernel_size=4, stride=2)
        with torch.no_grad():
            dummy_target_img = torch.zeros((1, out_channels, self.target_height, self.target_width), dtype=torch.float32)
            x = self.activation(temp_conv1(dummy_target_img))
            x = self.activation(temp_conv2(x))
            x = self.activation(temp_conv3(x))
            encoded_shape_calc = temp_conv4(x) # La dernière activation est appliquée avant cette ligne
        self.H_conv_out = encoded_shape_calc.shape[2]
        self.W_conv_out = encoded_shape_calc.shape[3]

        print(f"[ConvDecoder] Deduced encoder output spatial HxW: {self.H_conv_out}x{self.W_conv_out} for target {self.target_height}x{self.target_width}")

        # Les couches de déconvolution doivent être ajustées pour partir de (num_encoder_conv_filters, H_conv_out, W_conv_out)
        # et arriver à (out_channels, target_height, target_width)
        # Exemple (doit être recalculé précisément) :
        # self.deconv1 = nn.ConvTranspose2d(self.num_encoder_conv_filters, 128, kernel_size=4, stride=2, padding=?)
        # ... et ainsi de suite, en ajustant padding et output_padding pour atteindre les bonnes dimensions.
        # Ceci est la partie la plus délicate pour rendre le décodeur dynamique.
        # Pour l'instant, le code du décodeur est fixe pour une sortie (192,176) et une entrée de l'encodeur correspondante.
        # Si encoder_flat_size change, les valeurs (10,9) dans x.view(-1, 256, 10, 9) doivent changer.

        # Dans forward de ConvDecoder:
        # x = x.view(-1, self.num_encoder_conv_filters, self.H_conv_out, self.W_conv_out)

        # === Pour l'instant, gardons l'ancien code du décodeur et supposons que le problème est en amont ===
        # Si les prints révèlent que l'entrée de l'encodeur est bien 192x176, alors la modification dynamique
        # n'est pas la cause, et il faudra investiguer pourquoi les convolutions réduisent plus que prévu.

        # Version actuelle du code du décodeur (qui suppose une entrée de l'encodeur de 192x176):
        self.deconv1 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=0, output_padding=(0,1)) # Adjusted for width
        self.deconv2 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=0, output_padding=(0,0))
        self.deconv3 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=0, output_padding=(1,1))
        self.deconv4 = nn.ConvTranspose2d(32, out_channels, kernel_size=4, stride=2, padding=0, output_padding=(0,0))
        print(f"[ConvDecoder] Initialized (Note: Deconv layers might need adjustment if input image size for encoder changed significantly from 192x176)")

    def forward(self, h, z_posterior):
        x = torch.cat([h, z_posterior], dim=-1)
        x = self.fc(x) # Sortie (batch, self.encoder_flat_size)
        # Doit correspondre à la sortie de conv4 de l'Encoder AVANT aplatissement
        # Si encoder_flat_size a été calculé dynamiquement, il faut utiliser H_conv_out et W_conv_out
        # x = x.view(-1, self.num_encoder_conv_filters, self.H_conv_out, self.W_conv_out)
        x = x.view(-1, 256, self.H_conv_out, self.W_conv_out) # Utiliser les H_conv_out, W_conv_out calculés
        x = self.activation(self.deconv1(x))
        x = self.activation(self.deconv2(x))
        x = self.activation(self.deconv3(x))
        reconstructed_image = self.deconv4(x)
        # Vérifier que la taille de sortie est correcte
        # if reconstructed_image.shape[2:] != (self.target_height, self.target_width):
        #    print(f"WARNING [ConvDecoder]: Output shape {reconstructed_image.shape} does not match target ({self.target_height},{self.target_width})")
        return reconstructed_image

class RSSM(nn.Module):
    def __init__(self, latent_dim, action_dim, rnn_hidden_dim, activation=nn.ReLU):
        super().__init__()
        self.latent_dim = latent_dim
        self.action_dim = action_dim
        self.rnn_hidden_dim = rnn_hidden_dim
        self.activation = activation()

        # GRU input: encoded_observation_dim (latent_dim from ConvEncoder) + action_dim
        # Non, GRU input: posterior_latent_dim + action_dim
        self.rnn = nn.GRU(latent_dim + action_dim, rnn_hidden_dim, batch_first=True)

        # Prior: de h_t -> z_prior_{t+1}
        self.fc_prior = nn.Linear(rnn_hidden_dim, 2 * latent_dim) # mean and std_dev

        # Posterior: de h_t et e_{t+1} (encoded_obs) -> z_posterior_{t+1}
        self.fc_posterior = nn.Linear(rnn_hidden_dim + latent_dim, 2 * latent_dim) # mean and std_dev

    def dynamics_predict(self, prev_z_posterior, prev_action, prev_h):
        # prev_z_posterior: (batch, latent_dim)
        # prev_action: (batch, action_dim)
        # prev_h: (batch, rnn_hidden_dim)
        rnn_input = torch.cat([prev_z_posterior, prev_action], dim=-1).unsqueeze(1) # (batch, 1, latent_dim + action_dim)
        if prev_h.dim() == 2: # (batch, rnn_hidden_dim)
             prev_h = prev_h.unsqueeze(0) # (1, batch, rnn_hidden_dim) pour GRU

        current_h, _ = self.rnn(rnn_input, prev_h)
        current_h = current_h.squeeze(1) # (batch, rnn_hidden_dim)

        prior_stats = self.fc_prior(current_h)
        prior_mean, prior_std = torch.chunk(prior_stats, 2, dim=-1)
        prior_std = F.softplus(prior_std) + 1e-4 # Ensure positive std_dev
        return current_h, prior_mean, prior_std

    def update_posterior(self, current_h, encoded_obs):
        # current_h: (batch, rnn_hidden_dim)
        # encoded_obs: (batch, latent_dim from ConvEncoder)
        posterior_input = torch.cat([current_h, encoded_obs], dim=-1)
        posterior_stats = self.fc_posterior(posterior_input)
        posterior_mean, posterior_std = torch.chunk(posterior_stats, 2, dim=-1)
        posterior_std = F.softplus(posterior_std) + 1e-4 # Ensure positive std_dev
        return posterior_mean, posterior_std

    def get_initial_state(self, batch_size, device):
        # z est l'état latent stochastique, h est l'état déterministe du RNN
        return torch.zeros(batch_size, self.latent_dim, device=device), \
               torch.zeros(batch_size, self.rnn_hidden_dim, device=device)


class RewardModel(nn.Module):
    def __init__(self, rnn_hidden_dim, latent_dim, hidden_units=256, activation=nn.ReLU):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(rnn_hidden_dim + latent_dim, hidden_units), activation(),
            nn.Linear(hidden_units, hidden_units), activation(),
            nn.Linear(hidden_units, 1)
        )
    def forward(self, h, z): # Prend h_t et z_posterior_t (ou z_prior_t pendant l'imagination)
        return self.mlp(torch.cat([h, z], dim=-1)).squeeze(-1)

class ValueModel(nn.Module): # Critic
    def __init__(self, rnn_hidden_dim, latent_dim, hidden_units=256, activation=nn.ReLU):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(rnn_hidden_dim + latent_dim, hidden_units), activation(),
            nn.Linear(hidden_units, hidden_units), activation(),
            nn.Linear(hidden_units, 1)
        )
    def forward(self, h, z): # Prend h_t et z_posterior_t (ou z_prior_t pendant l'imagination)
        return self.mlp(torch.cat([h, z], dim=-1)).squeeze(-1)

class ActorModel(nn.Module):
    def __init__(self, rnn_hidden_dim, latent_dim, action_dim, hidden_units=256, activation=nn.ReLU):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(rnn_hidden_dim + latent_dim, hidden_units), activation(),
            nn.Linear(hidden_units, hidden_units), activation(),
            nn.Linear(hidden_units, action_dim)
        )
    def forward(self, h, z): # Prend h_t et z_posterior_t (ou z_prior_t pendant l'imagination)
        logits = self.mlp(torch.cat([h, z], dim=-1))
        # Pour des actions discrètes, on retourne les logits. La distribution sera créée dans DreamerTrainer.
        return logits


class DreamerTrainer:
    def __init__(self, config):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.config = config
        self.action_dim = config.output_size
        self.latent_dim = getattr(config, 'latent_dim', 32) # à définir dans TrainingConfig
        self.rnn_hidden_dim = getattr(config, 'rnn_hidden_dim', 64) # à définir dans TrainingConfig
        self.sequence_length = getattr(config, 'sequence_length', 30)
        self.imagination_horizon = getattr(config, 'imagination_horizon', 10)
        self.kl_loss_scale = getattr(config, 'kl_loss_scale', 1.0)
        self.reward_loss_scale = getattr(config, 'reward_loss_scale', 1.0)
        self.value_loss_scale = getattr(config, 'value_loss_scale', 1.0)
        self.actor_loss_scale = getattr(config, 'actor_loss_scale', 1.0)
        self.reconstruction_loss_scale = getattr(config, 'reconstruction_loss_scale', 1.0)
        self.action_entropy_scale = getattr(config, 'action_entropy_scale', 0.005)
        self.gamma = config.gamma
        self.lambda_gae = getattr(config, 'lambda_gae', 0.95)


        # Models
        # config.input_size est (N_frames_stack, H, W)
        # ConvEncoder prend N_frames_stack comme in_channels
        num_input_channels = config.input_size[0]
        effective_H = config.input_size[1]
        effective_W = config.input_size[2]
        self.encoder = ConvEncoder(
                                    in_channels=num_input_channels,
                                    latent_dim=self.latent_dim,
                                    image_height=effective_H,
                                    image_width=effective_W
                                ).to(self.device)
        # IMPORTANT : ConvDecoder devra aussi être adapté si la taille de sortie de l'encodeur change.
        # La taille de l'entrée de la première couche fc du ConvDecoder dépend de la sortie de l'encodeur.
        # self.encoder.n_flat contiendra la taille aplatie correcte APRÈS l'initialisation de l'encodeur.
        encoder_output_flat_size = self.encoder.n_flat
        self.decoder = ConvDecoder(
            latent_dim=self.latent_dim,
            hidden_dim_rssm=self.rnn_hidden_dim,
            out_channels=num_input_channels,
            encoder_flat_size=encoder_output_flat_size, # Nouvelle entrée
            target_height=effective_H, # Pour reconstruire à la bonne taille
            target_width=effective_W   # Pour reconstruire à la bonne taille
        ).to(self.device)
        self.rssm = RSSM(self.latent_dim, self.action_dim, self.rnn_hidden_dim).to(self.device)
        self.reward_model = RewardModel(self.rnn_hidden_dim, self.latent_dim).to(self.device)
        self.value_model = ValueModel(self.rnn_hidden_dim, self.latent_dim).to(self.device) # Critic
        self.actor = ActorModel(self.rnn_hidden_dim, self.latent_dim, self.action_dim).to(self.device)

        # Optimizers
        self.world_model_params = list(self.encoder.parameters()) + list(self.decoder.parameters()) + \
                                  list(self.rssm.parameters()) + list(self.reward_model.parameters())
        self.actor_params = list(self.actor.parameters())
        self.value_params = list(self.value_model.parameters())

        self.world_model_optim = torch.optim.Adam(self.world_model_params, lr=config.learning_rate, eps=1e-5)
        self.actor_optim = torch.optim.Adam(self.actor_params, lr=config.learning_rate * 0.5, eps=1e-5) # Souvent lr plus bas
        self.value_optim = torch.optim.Adam(self.value_params, lr=config.learning_rate, eps=1e-5)

        self.buffer = deque(maxlen=config.buffer_capacity)
        self.training_steps = 0
        self.epsilon = config.epsilon_start

        # État interne pour l'interaction avec l'environnement
        self.h_prev = None
        self.z_prev_posterior = None # z_posterior échantillonné

    def encode_state_initial(self, initial_observation_stack):
        # initial_observation_stack: (N_frames, H, W) numpy array
        # Convertir en tensor (1, N_frames, H, W) pour l'encodeur
        obs_tensor = torch.tensor(initial_observation_stack, dtype=torch.float32, device=self.device).unsqueeze(0)
        encoded_obs = self.encoder(obs_tensor) # (1, latent_dim)

        # Initialiser h et z pour le RSSM
        self.z_prev_posterior, self.h_prev = self.rssm.get_initial_state(batch_size=1, device=self.device)

        # Mettre à jour le premier z_posterior avec la première observation
        # Pour cela, on a besoin d'un h_prev (qui est initialement zéro)
        # Et on utilise l'encoded_obs pour obtenir le premier z_posterior
        # Note: au tout premier pas, il n'y a pas d'action précédente. On peut utiliser une action nulle.
        
        # Alternative: RSSM update_posterior est appelé avec h_prev (zero) et encoded_obs.
        post_mean, post_std = self.rssm.update_posterior(self.h_prev, encoded_obs)
        posterior_dist = Independent(Normal(post_mean, post_std), 1) # Distribution normale multivariée
        self.z_prev_posterior = posterior_dist.rsample() # (1, latent_dim)

        # Le h_prev reste celui initial (zero) car il n'y a pas eu de dynamique encore.
        # Ou, on peut faire un pas de dynamique avec une action nulle pour obtenir h_1
        # Pour simplifier, on garde h_prev=zero et z_prev_posterior basé sur la première obs.
        # Le premier vrai h sera calculé lors du premier `dreamer_step`.


    def dreamer_step(self, current_observation_stack, prev_action_value):
        # current_observation_stack: (N_frames, H, W) numpy array
        # prev_action_value: integer
        
        # 1. Encoder l'observation actuelle (si on en a besoin pour le posterior, mais pour l'action, on utilise le prior)
        # obs_tensor = torch.tensor(current_observation_stack, dtype=torch.float32, device=self.device).unsqueeze(0)
        # encoded_current_obs = self.encoder(obs_tensor) # (1, latent_dim)

        # 2. Préparer l'action précédente pour le RSSM
        prev_action_tensor = F.one_hot(torch.tensor([prev_action_value], device=self.device), num_classes=self.action_dim).float()

        # 3. Prédire l'état suivant (prior) et le h actuel avec le RSSM
        # self.z_prev_posterior et self.h_prev sont de l'étape t-1
        current_h, prior_mean, prior_std = self.rssm.dynamics_predict(self.z_prev_posterior, prev_action_tensor, self.h_prev)
        # Ici, prior_mean/std est pour z_prior_t. On peut l'échantillonner si besoin.
        # Pour la sélection d'action, on utilise souvent h_t et z_prior_t (ou z_posterior_t-1).
        # Dans l'article original, l'acteur prend h_t et z_prior_t.
        
        z_prior_dist = Independent(Normal(prior_mean, prior_std), 1)
        z_t_prior_sample = z_prior_dist.sample() # Ou .mean pour une version déterministe

        # 4. Sélectionner l'action avec l'acteur
        # L'acteur prend l'état actuel prédit (current_h, z_t_prior_sample)
        action_logits = self.actor(current_h, z_t_prior_sample)
        action_probs = F.softmax(action_logits, dim=-1)
        
        if self.config.mode == "exploration": # Ou self.training_steps < N_exploration_random_steps
            if self.training_steps < getattr(self.config, 'random_steps_explore', 1000): # Exemple
                 action = random.randint(0, self.action_dim - 1)
            else:
                 action = torch.multinomial(action_probs, 1).item()
        else: # Exploitation
            action = torch.argmax(action_probs, dim=-1).item()


        # 5. Mettre à jour l'état interne pour le prochain pas
        # Normalement, après avoir reçu la VRAIE observation suivante (o_t), on calculerait z_posterior_t.
        # Ici, pour `dreamer_step`, on se prépare pour le prochain appel.
        # On met à jour h_prev avec current_h.
        # z_prev_posterior sera mis à jour dans store_transition ou au début du prochain dreamer_step
        # si on reçoit l'observation réelle o_t.
        
        self.h_prev = current_h 
        # self.z_prev_posterior sera mis à jour une fois que l'environnement aura donné o_t
        # et qu'on l'aura encodé pour obtenir e_t, puis z_posterior_t via rssm.update_posterior(current_h, e_t)

        return action

    def store_transition(self, obs_stack_prev, action, reward, obs_stack_current, done):
        # obs_stack_prev: (N,H,W) utilisé pour l'action qui a mené à reward et obs_stack_current
        # obs_stack_current: (N,H,W) résultat de l'action
        # On stocke la pile d'observations qui a été utilisée pour prendre l'action.
        # Et la pile d'observations résultante.
        # Pour l'entraînement du RSSM, on a besoin de o_t, a_t, r_t, o_{t+1}, done_t
        
        # Dans la version typique, le buffer stocke (o_t, a_t, r_t, done_t)
        # o_t est la pile de frames (N, H, W)
        # a_t est l'action prise à partir de o_t
        # r_t est la récompense reçue après a_t
        # done_t indique si l'épisode s'est terminé après (o_t, a_t, r_t)
        
        # Ici, obs_stack_current est o_{t+1} (la conséquence de a_t prise sur obs_stack_prev)
        # Donc on stocke (obs_stack_prev, action, reward, obs_stack_current, done)
        # Pour la formation des séquences, c'est plus simple de stocker les (o_t, a_t, r_t, d_t) individuels.
        # L'implémentation du buffer et de l'échantillonnage doit gérer la formation de (o_t, a_t, r_t, o_{t+1}, d_t).

        # On va supposer que le buffer stocke des transitions (o_t, a_t, r_t, d_t) où o_t est la pile de frames.
        # La frame (ou pile de frames) `obs_stack_current` sera le `o_t` de la *prochaine* transition.
        if isinstance(obs_stack_prev, torch.Tensor): obs_stack_prev = obs_stack_prev.cpu().numpy()
        self.buffer.append((obs_stack_prev, action, reward, done))

        # Après avoir stocké la transition, et avant le prochain `dreamer_step`,
        # on doit mettre à jour `self.z_prev_posterior` avec le `z_posterior` de l'état actuel.
        # L'état actuel est `obs_stack_current`. `self.h_prev` est déjà `current_h` du pas précédent.
        if not done:
            obs_tensor = torch.tensor(obs_stack_current, dtype=torch.float32, device=self.device).unsqueeze(0)
            encoded_current_obs = self.encoder(obs_tensor) # (1, latent_dim)
            
            # self.h_prev est le current_h du pas où `action` a été prise.
            # On utilise ce h et l'observation actuelle encodée pour obtenir le z_posterior actuel.
            post_mean, post_std = self.rssm.update_posterior(self.h_prev.detach(), encoded_current_obs) # detach h?
            posterior_dist = Independent(Normal(post_mean, post_std), 1)
            self.z_prev_posterior = posterior_dist.rsample()
        else: # Fin d'épisode, réinitialiser pour le prochain
            self.z_prev_posterior = None
            self.h_prev = None


    def lambda_return(self, rewards, values, dones, gamma, lambda_gae):
        # rewards, values, dones: (T, B) ou (T*B)
        # T = imagination_horizon, B = batch_size
        # values inclut V(s_T) comme dernier élément
        returns = torch.zeros_like(rewards)
        advantage = torch.zeros_like(rewards)
        
        last_value = values[-1] # V(s_T)
        last_advantage = 0
        
        for t in reversed(range(rewards.size(0))): # T-1 down to 0
            mask = (1.0 - dones[t]) # 0 if done, 1 if not done
            delta = rewards[t] + gamma * last_value * mask - values[t]
            advantage[t] = delta + gamma * lambda_gae * last_advantage * mask
            returns[t] = advantage[t] + values[t]
            
            last_value = values[t]
            last_advantage = advantage[t]
        return returns, advantage

    def train_step(self):
        if len(self.buffer) < max(self.config.batch_size, getattr(self.config, 'min_history_size', 1000)):
            return
        
        # 1. Échantillonner des séquences du buffer
        # Chaque élément du buffer est (obs_stack_t, action_t, reward_t, done_t)
        # On a besoin de former des séquences de (o_t, a_t, r_t, d_t) pour t=0..L-1
        # et aussi des o_{L} pour la dernière transition.
        
        batch_size = self.config.batch_size
        seq_len = self.sequence_length

        sequences = []
        for _ in range(batch_size):
            start_index = random.randint(0, len(self.buffer) - seq_len)
            sequences.append(list(self.buffer)[start_index : start_index + seq_len])
        
        # Convertir en tensors:
        # obs_seq: (batch_size, seq_len, N_frames, H, W)
        # action_seq: (batch_size, seq_len)
        # reward_seq: (batch_size, seq_len)
        # done_seq: (batch_size, seq_len)
        
        obs_list, action_list, reward_list, done_list = [], [], [], []
        for seq in sequences:
            o, a, r, d = zip(*seq)
            obs_list.append(np.stack(o, axis=0))
            action_list.append(np.array(a))
            reward_list.append(np.array(r))
            done_list.append(np.array(d))

        obs_seq = torch.tensor(np.stack(obs_list, axis=0), dtype=torch.float32, device=self.device)
        action_seq = torch.tensor(np.stack(action_list, axis=0), dtype=torch.long, device=self.device)
        reward_seq = torch.tensor(np.stack(reward_list, axis=0), dtype=torch.float32, device=self.device)
        done_seq = torch.tensor(np.stack(done_list, axis=0), dtype=torch.float32, device=self.device) # float pour les masques

        # Préparer les actions en one-hot
        action_seq_onehot = F.one_hot(action_seq, num_classes=self.action_dim).float()

        # Encoder toutes les observations dans les séquences
        # obs_seq: (B, L, N_stack, H, W) -> (B*L, N_stack, H, W) pour l'encodeur
        # encoded_obs_seq: (B, L, latent_dim)
        encoded_obs_seq = self.encoder(obs_seq.view(batch_size * seq_len, *obs_seq.shape[2:]))
        encoded_obs_seq = encoded_obs_seq.view(batch_size, seq_len, self.latent_dim)

        # Initialiser z_posterior et h pour le début de chaque séquence
        z_posterior, h = self.rssm.get_initial_state(batch_size, self.device)
        
        kl_losses = []
        reconstruction_losses = []
        reward_pred_losses = []
        
        # Stocker les états pour l'apprentissage de l'acteur-critique
        posterior_states_z = [] # z_posterior échantillonné
        deterministic_states_h = [] # h (état du GRU)

        # --- Entraînement du Modèle du Monde ---
        for t in range(seq_len -1): # On a besoin de o_t, a_t, r_t, o_{t+1}
            # Action a_t, observation o_t (encoded_obs_seq[:, t])
            # Observation suivante o_{t+1} (encoded_obs_seq[:, t+1])
            # Récompense r_t (reward_seq[:, t])
            
            # Prédire le prior pour z_{t+1} et obtenir h_{t+1}
            # Utilise z_posterior_t (actuel), action_t, h_t (actuel)
            h, prior_mean, prior_std = self.rssm.dynamics_predict(z_posterior, action_seq_onehot[:, t], h)
            prior_dist = Independent(Normal(prior_mean, prior_std), 1)
            
            # Calculer le posterior pour z_{t+1} en utilisant h_{t+1} (qui est `h` ici) et e_{t+1}
            post_mean, post_std = self.rssm.update_posterior(h.detach(), encoded_obs_seq[:, t+1]) # detach h pour ne pas backpropager KL vers dynamique
            posterior_dist = Independent(Normal(post_mean, post_std), 1)
            
            # Perte KL
            kl_loss = kl_divergence(posterior_dist, prior_dist).mean() # Moyenne sur le batch
            kl_losses.append(kl_loss)
            
            # Échantillonner z_posterior_{t+1} pour le pas suivant et pour les prédicteurs
            z_posterior = posterior_dist.rsample() # (batch_size, latent_dim)

            posterior_states_z.append(z_posterior)
            deterministic_states_h.append(h)

            # Prédire la récompense r_t (basée sur h_{t+1} et z_posterior_{t+1})
            # Ici h et z_posterior sont pour le temps t+1. La récompense est r_t.
            # On prédit r_t à partir de l'état (h_t, z_t) ou (h_{t+1}, z_{t+1}) ?
            # Typiquement on prédit r_t depuis (h_t, z_t). Donc ici, on prédit reward_seq[:, t]
            # en utilisant le h et z_posterior qui viennent d'être calculés pour le pas t+1 (basé sur o_{t+1}).
            # C'est un peu un décalage.
            # Alternativement, prédire r_t avec (h, z_prior) de l'état qui a généré r_t.
            # Pour simplifier: prédire reward_seq[:, t] en utilisant h (de t+1) et z_posterior (de t+1).
            # Ou, RewardModel(h_prev_step, z_posterior_prev_step) pour prédire reward_seq[:,t]
            # L'article original prédit r_t en utilisant s_t (h_t, z_t).
            # Si h et z_posterior sont l'état s_{t+1}, alors on prédit r_t.
            
            predicted_reward = self.reward_model(h, z_posterior) # h et z_posterior sont s_{t+1}
            reward_loss = F.mse_loss(predicted_reward, reward_seq[:, t]) # prédit r_t
            reward_pred_losses.append(reward_loss)

            # Perte de reconstruction d'image
            # Reconstruire obs_seq[:, t] en utilisant (h, z_posterior) de l'état s_{t+1} ?
            # Non, reconstruire obs_seq[:, t+1] en utilisant (h, z_posterior) de l'état s_{t+1}
            # La cible est l'observation qui a mené à l'état (h, z_posterior)
            reconstructed_obs = self.decoder(h, z_posterior) # h et z_posterior sont pour t+1
            # La cible est obs_seq[:, t+1] (la pile de frames originale)
            recon_target = obs_seq[:, t+1]
            reconstruction_loss = F.mse_loss(reconstructed_obs, recon_target)
            reconstruction_losses.append(reconstruction_loss)

        total_kl_loss = torch.stack(kl_losses).mean()
        total_reward_loss = torch.stack(reward_pred_losses).mean()
        total_reconstruction_loss = torch.stack(reconstruction_losses).mean()

        world_model_loss = self.kl_loss_scale * total_kl_loss + \
                           self.reward_loss_scale * total_reward_loss + \
                           self.reconstruction_loss_scale * total_reconstruction_loss
        
        self.world_model_optim.zero_grad()
        world_model_loss.backward()
        nn.utils.clip_grad_norm_(self.world_model_params, getattr(self.config, 'grad_clip_world_model', 100.0))
        self.world_model_optim.step()

        # --- Entraînement de l'Acteur et de la Valeur (Critic) sur des trajectoires imaginées ---
        # Les états de départ pour l'imagination sont les états appris par le modèle du monde
        # On prend les `h` et `z_posterior` de la dernière étape de la boucle ci-dessus (seq_len-1)
        # Ou on peut prendre tous les (h, z_posterior) calculés.
        # Pour Dreamer, on imagine à partir de chaque état de la séquence d'apprentissage.
        
        # Récupérer les états stockés (detach pour ne pas backpropager dans le world model depuis l'acteur/critique)
        imag_h_initial = torch.stack(deterministic_states_h).detach() # (L-1, B, rnn_hidden_dim)
        imag_z_initial = torch.stack(posterior_states_z).detach()   # (L-1, B, latent_dim)
        
        # Reshape pour imagination : ( (L-1)*B, dim )
        batch_imag_h = imag_h_initial.view(-1, self.rnn_hidden_dim)
        batch_imag_z = imag_z_initial.view(-1, self.latent_dim)

        imagined_rewards = []
        imagined_hs = []
        imagined_zs_prior = [] # On stocke les z_prior échantillonnés
        imagined_action_log_probs = []
        imagined_action_entropies = []

        current_h_imag = batch_imag_h
        current_z_imag = batch_imag_z # Ceci est z_posterior, utilisé comme z_t pour prédire a_t

        for _ in range(self.imagination_horizon):
            # Action de l'acteur
            action_logits = self.actor(current_h_imag, current_z_imag)
            action_dist = Normal(action_logits, 1.0) # Si actions continues, sinon Categorical
            # Pour actions discrètes:
            action_probs_imag = F.softmax(action_logits, dim=-1)
            action_dist_imag = torch.distributions.Categorical(logits=action_logits) # Utiliser logits directement
            
            action_imag = action_dist_imag.sample() # ( (L-1)*B, )
            imagined_action_log_probs.append(action_dist_imag.log_prob(action_imag))
            imagined_action_entropies.append(action_dist_imag.entropy())

            # Prédire état suivant et récompense avec le modèle du monde
            action_imag_onehot = F.one_hot(action_imag, num_classes=self.action_dim).float()
            
            next_h_imag, prior_mean_imag, prior_std_imag = self.rssm.dynamics_predict(current_z_imag, action_imag_onehot, current_h_imag)
            next_z_prior_dist_imag = Independent(Normal(prior_mean_imag, prior_std_imag), 1)
            next_z_imag = next_z_prior_dist_imag.rsample() # Ou .mean

            predicted_reward_imag = self.reward_model(next_h_imag, next_z_imag)

            imagined_rewards.append(predicted_reward_imag)
            imagined_hs.append(next_h_imag)
            imagined_zs_prior.append(next_z_imag)
            
            current_h_imag = next_h_imag
            current_z_imag = next_z_imag

        imagined_rewards = torch.stack(imagined_rewards) # (H, (L-1)*B)
        imagined_hs = torch.stack(imagined_hs)           # (H, (L-1)*B, rnn_hidden_dim)
        imagined_zs_prior = torch.stack(imagined_zs_prior) # (H, (L-1)*B, latent_dim)
        imagined_action_log_probs = torch.stack(imagined_action_log_probs) # (H, (L-1)*B)
        imagined_action_entropies = torch.stack(imagined_action_entropies) # (H, (L-1)*B)

        # Prédire les valeurs avec le ValueModel (Critic)
        # Les valeurs sont V(s_imag_t) pour t=0..H-1
        # Et on a besoin de V(s_imag_H) pour le bootstrap du lambda return.
        # Donc on prédit H+1 valeurs. La dernière valeur est pour l'état après la dernière action imaginée.
        
        # Concaténer l'état initial de l'imagination pour avoir H+1 états
        all_imag_h = torch.cat([batch_imag_h.unsqueeze(0), imagined_hs], dim=0) # (H+1, (L-1)*B, h_dim)
        all_imag_z = torch.cat([batch_imag_z.unsqueeze(0), imagined_zs_prior], dim=0) # (H+1, (L-1)*B, z_dim)

        imagined_values = self.value_model(all_imag_h, all_imag_z) # (H+1, (L-1)*B)
        
        # Calculer les lambda returns
        # Pour les lambda returns, on n'a pas de "dones" dans l'imagination, sauf si on prédit un "continue flag"
        # On suppose que l'imagination ne se termine pas (mask=1)
        dones_imag = torch.zeros_like(imagined_rewards) # (H, (L-1)*B)
        
        lambda_returns, advantages = self.lambda_return(imagined_rewards, imagined_values, dones_imag, self.gamma, self.lambda_gae)
        # lambda_returns et advantages sont de taille (H, (L-1)*B)

        # Perte de l'Acteur
        # On veut maximiser sum_t (log_prob(a_t) * (lambda_return_t - V(s_t)).detach() + entropy_scale * entropy_t)
        # advantages = lambda_returns - imagined_values[:-1].detach() # Si on utilise advantage
        
        actor_loss = -(imagined_action_log_probs * advantages.detach()).mean() # detach advantages
        actor_entropy_loss = -self.action_entropy_scale * imagined_action_entropies.mean()
        total_actor_loss = self.actor_loss_scale * (actor_loss + actor_entropy_loss)

        self.actor_optim.zero_grad()
        total_actor_loss.backward(retain_graph=True)
        nn.utils.clip_grad_norm_(self.actor_params, getattr(self.config, 'grad_clip_actor', 100.0))
        self.actor_optim.step()

        # Perte de la Valeur (Critic)
        # V(s_t) doit approcher lambda_return_t
        value_loss = F.mse_loss(imagined_values[:-1], lambda_returns.detach()) # detach returns
        total_value_loss = self.value_loss_scale * value_loss

        self.value_optim.zero_grad()
        total_value_loss.backward()
        nn.utils.clip_grad_norm_(self.value_params, getattr(self.config, 'grad_clip_value', 100.0))
        self.value_optim.step()

        self.training_steps += 1
        if self.training_steps % 100 == 0:
            print(f"[Dreamer Train Step {self.training_steps}] WorldModel Loss: {world_model_loss.item():.3f} (KL: {total_kl_loss.item():.3f}, Reward: {total_reward_loss.item():.3f}, Recon: {total_reconstruction_loss.item():.3f})")
            print(f"    Actor Loss: {total_actor_loss.item():.3f} (Policy: {actor_loss.item():.3f}, Entropy: {actor_entropy_loss.item():.3f}), Value Loss: {total_value_loss.item():.3f}")


    def save_model(self, path):
        torch.save({
            'encoder': self.encoder.state_dict(),
            'decoder': self.decoder.state_dict(),
            'rssm': self.rssm.state_dict(),
            'reward_model': self.reward_model.state_dict(),
            'value_model': self.value_model.state_dict(),
            'actor': self.actor.state_dict(),
            'world_model_optim': self.world_model_optim.state_dict(),
            'actor_optim': self.actor_optim.state_dict(),
            'value_optim': self.value_optim.state_dict(),
            'training_steps': self.training_steps
        }, path)
        print(f"DreamerV2 model saved to {path}")

    def load_model(self, path):
        if not os.path.exists(path):
            print(f"Model file not found: {path}")
            return -1
        try:
            data = torch.load(path, map_location=self.device)
            self.encoder.load_state_dict(data['encoder'])
            self.decoder.load_state_dict(data['decoder'])
            self.rssm.load_state_dict(data['rssm'])
            self.reward_model.load_state_dict(data['reward_model'])
            self.value_model.load_state_dict(data['value_model'])
            self.actor.load_state_dict(data['actor'])
            self.world_model_optim.load_state_dict(data['world_model_optim'])
            self.actor_optim.load_state_dict(data['actor_optim'])
            self.value_optim.load_state_dict(data['value_optim'])
            self.training_steps = data.get('training_steps', 0) # Pour la compatibilité
            print(f"DreamerV2 model loaded from {path}, training_steps: {self.training_steps}")
            return 1
        except Exception as e:
            print(f"Error loading model: {e}")
            return -1

    def save_buffer(self, path):
        try:
            with open(path, 'wb') as f:
                pickle.dump(self.buffer, f)
            print(f"Buffer saved to {path}")
        except Exception as e:
            print(f"Erreur sauvegarde buffer : {e}")

    def load_buffer(self, path):
        if not os.path.exists(path):
            print(f"Buffer file not found: {path}")
            return -1
        try:
            with open(path, 'rb') as f:
                self.buffer = pickle.load(f)
            print(f"Buffer loaded from {path}, size: {len(self.buffer)}")
            return 1
        except Exception as e:
            print(f"Erreur chargement buffer : {e}")
            return -1