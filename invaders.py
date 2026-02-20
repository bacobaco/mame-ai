"""
invaders_ai.py

Version refactorisée et structurée de invaders.py pour l'entraînement d'une IA sur Space Invaders via MAME.
Ce script sépare la logique de communication, la gestion de l'environnement, la visualisation et la boucle d'entraînement.

Fonctionnalités identiques à invaders.py, mais avec une architecture orientée objet.
"""

import os
import time
import shutil
import random
import subprocess
import threading
import keyboard
import numpy as np
import pygame
import psutil
import win32gui
from collections import deque
from datetime import datetime
from colorama import Fore, Style
import matplotlib
import matplotlib.pyplot as plt

# Imports locaux
from MameCommSocket import MameCommunicator
from ScreenRecorder import ScreenRecorder
from AI_Mame import TrainingConfig, DQNTrainer, GraphWebServer
from dreamerv2 import DreamerTrainer

# ==================================================================================================
# CONSTANTES & ADRESSES MÉMOIRE
# ==================================================================================================

class Memory:
    """Adresses mémoire du jeu Space Invaders (Arcade)."""
    NUM_COINS       = "20EB"  # Nombre de pièces
    P1_SCORE_L      = "20F8"  # Score P1 (Bas)
    P1_SCORE_M      = "20F9"  # Score P1 (Haut)
    NUM_ALIENS      = "2082"  # Nombre d'aliens
    PLAYER_XR       = "201B"  # Position X du joueur
    PLAYER_ALIVE    = "2015"  # Statut joueur (FF=Vivant)
    PLAYER_1_ALIVE  = "20E7"  # 1 si vivant, 0 si mort (Game Over)
    PLAYER_OK       = "2068"  # 1=OK, 0=Explosion
    P1_SHIPS_REM    = "21FF"  # Vies restantes
    
    # Aliens & Tirs
    REF_ALIEN_YR    = "2009"  # Y Alien référence
    REF_ALIEN_XR    = "200A"  # X Alien référence
    SAUCER_XR       = "208A"  # X Soucoupe
    
    # Tirs Aliens (Rolling, Squiggly, Plunger)
    ROL_SHOT_YR     = "203D"
    ROL_SHOT_XR     = "203E"
    SQU_SHOT_YR     = "205D"
    SQU_SHOT_XR     = "205E"
    PLU_SHOT_YR     = "204D"
    PLU_SHOT_XR     = "204E"
    
    # Tirs Joueur
    OBJ1_COOR_XR    = "202A"
    OBJ1_COOR_YR    = "2029"
    PLYR_SHOT_STATUS = "2025"  # Status tir joueur (0=peut tirer, >0=tir en cours)
    
    # Flags
    PLAYER_ALIEN_DEAD = "2100" # Flag mort (55 octets à partir de là pour les aliens)

class GameConstants:
    ACTIONS = {
        0: ("left", False), 1: ("left", True),
        2: ("rght", False), 3: ("rght", True),
        4: ("stop", False), 5: ("stop", True),
    }
    FULL_FRAME_SIZE = (192, 176) # Taille utile après crop

# ==================================================================================================
# GESTION DE L'INTERFACE DE JEU (MAME)
# ==================================================================================================

class InvadersInterface:
    """Gère la communication avec MAME, l'exécution des actions et la lecture de l'état du jeu."""
    
    def __init__(self, communicator: MameCommunicator):
        self.comm = communicator

    def execute_action(self, action_code: int):
        """Envoie les commandes Lua pour exécuter une action."""
        direction, tirer = GameConstants.ACTIONS[action_code]
        self.comm.communicate([
            f"execute P1_left({int(direction=='left')})",
            f"execute P1_right({int(direction=='rght')})",
            f"execute P1_Button_1({int(tirer)})"
        ])

    def get_state_mlp(self, flag_coord_aliens=True, flag_boucliers=False, mult_reward_state=0.0, colonnes_deja_detruites_input=None):
        """
        Récupère l'état du jeu sous forme de vecteur pour le modèle MLP.
        Logique identique à l'ancien `get_state`.
        """
        if colonnes_deja_detruites_input is None:
            colonnes_deja_detruites_input = [False] * 11

        messages = [
            f"read_memory {Memory.SAUCER_XR}", 
            f"read_memory {Memory.ROL_SHOT_YR}", f"read_memory {Memory.ROL_SHOT_XR}",
            f"read_memory {Memory.SQU_SHOT_YR}", f"read_memory {Memory.SQU_SHOT_XR}", 
            f"read_memory {Memory.PLU_SHOT_YR}", f"read_memory {Memory.PLU_SHOT_XR}", 
            f"read_memory {Memory.NUM_ALIENS}", f"read_memory {Memory.PLAYER_XR}",
            f"read_memory {Memory.OBJ1_COOR_XR}", f"read_memory {Memory.OBJ1_COOR_YR}", 
            f"read_memory {Memory.REF_ALIEN_YR}", f"read_memory {Memory.REF_ALIEN_XR}",
            f"read_memory {Memory.PLYR_SHOT_STATUS}", f"read_memory {Memory.P1_SHIPS_REM}",
        ]
        if flag_coord_aliens: 
            messages.append(f"read_memory_range {Memory.PLAYER_ALIEN_DEAD}(55)")

        response = self.comm.communicate(messages)
        
        # Calcul de la taille attendue (11 flags + 2 new + 17 aliens + 2 ref + 4 boucliers)
        expected_size = 11 + 2 + (17 if flag_coord_aliens else 0) + 2 + (4 if flag_boucliers else 0)

        # Gestion d'erreur basique
        if not response or len(response) < 15: 
            return np.zeros(expected_size, dtype=np.float32), 0.0

        values = list(map(int, response[:15]))
        
        # Normalisation des valeurs
        flags_normalized = [
            (values[0] - 41) / (224 - 41) if (224 - 41) != 0 else 0.0, # saucerXr
            values[1] / 223.0, values[2] / 223.0, values[3] / 223.0, values[4] / 223.0,
            values[5] / 223.0, values[6] / 223.0, values[7] / 55.0 if values[7] != 0 else 0.0, # numAliens
            values[8] / 223.0, values[9] / 223.0, values[10] / 223.0,
        ]
        refAlienYr_val = values[11]
        refAlienXr_val = values[12]
        plyrShotStatus_normalized = 1.0 if values[13] > 0 else 0.0  # 0=peut tirer, 1=tir en vol
        nb_vies_normalized = min(values[14], 3) / 3.0               # 0-3 vies → [0, 1]
        
        penalty_descente = 0.0
        rewards_colonne_detruite_total = 0.0

        if flag_coord_aliens:
            if len(response) < 16: return np.zeros(expected_size, dtype=np.float32), 0.0
            alien_flags_str = response[15]
            try:
                alien_flags = list(map(int, alien_flags_str.split(",")))
                if len(alien_flags) != 55: alien_flags = [0] * 55
            except ValueError:
                alien_flags = [0] * 55

            nb_aliens_par_colonne = []
            for col_idx in range(11):
                count = sum(alien_flags[row * 11 + col_idx] for row in range(5))
                nb_aliens_par_colonne.append(count / 5.0)
                if count == 0 and not colonnes_deja_detruites_input[col_idx]:
                    rewards_colonne_detruite_total += 1000.0
                    colonnes_deja_detruites_input[col_idx] = True

            positions = [(col_idx, row) for row in range(5) for col_idx in range(11) if alien_flags[row * 11 + col_idx] == 1]
            if positions:
                xs, ys = zip(*positions)
                mean_x, mean_y = sum(xs) / len(xs), sum(ys) / len(ys)
                min_x, max_x, min_y, max_y = min(xs), max(xs), min(ys), max(ys)
                max_y_pixel = refAlienYr_val + max_y * 16
            else:
                mean_x = mean_y = min_x = max_x = min_y = max_y = max_y_pixel = 0

            state_vector_list = (
                flags_normalized +
                [plyrShotStatus_normalized, nb_vies_normalized] +
                [mean_x / 10.0, mean_y / 4.0, min_x / 10.0, max_x / 10.0, min_y / 4.0, max_y / 4.0] +
                nb_aliens_par_colonne + [refAlienXr_val / 255.0, refAlienYr_val / 255.0]
            )
            # CORRECTION FINALE : On utilise refAlienYr_val (120->200+) pour ne pas biaiser l'IA à tuer le haut.
            # (120 - ref) commence à 0 et devient négatif quand les aliens descendent.
            penalty_descente = ((120 - refAlienYr_val) / 10.0 + rewards_colonne_detruite_total) * mult_reward_state
        else:
            penalty_descente = ((120 - refAlienYr_val) / 10.0) * mult_reward_state
            state_vector_list = flags_normalized + [plyrShotStatus_normalized, nb_vies_normalized] + [refAlienXr_val / 255.0, refAlienYr_val / 255.0]

        state_vector_np = np.array(state_vector_list, dtype=np.float32)
        state_vector_np = (state_vector_np - 0.5) * 2.0

        if flag_boucliers:
            boucliers_np = np.zeros(4, dtype=np.float32) # Simulation boucliers
            state_vector_np = np.concatenate((state_vector_np, boucliers_np))
            
        return state_vector_np, penalty_descente

    def get_state_cnn(self, factor_div=2, mult_reward_state=0.0):
        """
        Récupère l'écran complet cropé et redimensionné pour CNN/Dreamer.
        Lit aussi la hauteur des aliens pour le reward shaping.
        """
        response = self.comm.communicate([
            "read_memory_range 2400(7168)",
            f"read_memory {Memory.REF_ALIEN_YR}"
        ])
        if not response or len(response) < 2: raise ValueError("Aucune réponse reçue de Lua.")

        raw_str = response[0]
        refAlienYr_val = int(response[1])
        
        all_bytes = np.array(list(map(int, raw_str.split(","))), dtype=np.uint8)

        if all_bytes.size != 7168: raise ValueError(f"Expected 7168 bytes, got {all_bytes.size}")
        
        columns = all_bytes.reshape(224, 32)
        columns = columns[16:208, :]  # Crop Horizontal
        cropped_columns = columns[:, 3:25]  # Crop Vertical
        image = np.unpackbits(cropped_columns, axis=1, bitorder="little")

        if factor_div > 1:
            h, w = image.shape
            h_ = h // factor_div
            w_ = w // factor_div
            image = image[:h_ * factor_div, :w_ * factor_div]
            image = image.reshape(h_, factor_div, w_, factor_div).max(axis=(1, 3))
            
        # Calcul pénalité (identique MLP) : (120 - Y) devient négatif quand Y augmente (descend)
        penalty_descente = ((120 - refAlienYr_val) / 10.0) * mult_reward_state
            
        return image.astype(np.float32), penalty_descente

    def get_score_and_status(self, last_score):
        """
        Récupère le score et le statut du joueur en un seul appel socket optimisé.
        """
        response_grouped = self.comm.communicate([
            f"read_memory {Memory.P1_SCORE_L}", f"read_memory {Memory.P1_SCORE_M}",
            f"read_memory {Memory.PLAYER_OK}", f"read_memory {Memory.PLAYER_1_ALIVE}"
        ])
        
        if not response_grouped or len(response_grouped) < 4:
            return last_score, 0, 0 # Erreur, on garde le score et on suppose mort/fin pour sécurité
        
        P1ScorL_v, P1ScorM_v, PlayerIsOK, NotEndOfGame = list(map(int, response_grouped))
        score = (P1ScorL_v >> 4) * 10 + (P1ScorM_v & 0x0F) * 100 + ((P1ScorM_v) >> 4) * 1000
        
        return score, PlayerIsOK, NotEndOfGame

class StateExtractor:
    """Classe helper pour extraire l'état selon la configuration (MLP/CNN)."""
    def __init__(self, interface: InvadersInterface, model_type, flag_aliens, flag_boucliers, factor_div_frame, mult_reward_state, colonnes_deja_detruites_ref):
        self.interface = interface
        self.model_type = model_type
        self.flag_aliens = flag_aliens
        self.flag_boucliers = flag_boucliers
        self.factor_div_frame = factor_div_frame
        self.mult_reward_state = mult_reward_state
        self.colonnes_deja_detruites_ref = colonnes_deja_detruites_ref

    def __call__(self):
        if self.model_type in ("cnn", "dreamer"):
            # On passe mult_reward_state pour récupérer la pénalité
            return self.interface.get_state_cnn(factor_div=self.factor_div_frame, mult_reward_state=self.mult_reward_state)
        else: # MLP
            return self.interface.get_state_mlp(
                flag_coord_aliens=self.flag_aliens,
                flag_boucliers=self.flag_boucliers,
                mult_reward_state=self.mult_reward_state,
                colonnes_deja_detruites_input=self.colonnes_deja_detruites_ref
            )

# ==================================================================================================
# VISUALISATION & LOGS
# ==================================================================================================

class Visualizer:
    """Gère la création des graphiques et l'affichage des frames."""
    
    @staticmethod
    def afficher_frame(frame, factor_div=1):
        frame_img = np.rot90(frame, 1)
        plt.figure(figsize=(6, 6))
        plt.imshow(frame_img, cmap="gray", interpolation="nearest")
        plt.title("Frame Space Invaders")
        plt.axis("off")
        filename = f"frame(shrink factor={factor_div})_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        plt.savefig(filename, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Frame sauvegardée : {filename}")

    @staticmethod
    def afficher_get_state(interface: InvadersInterface):
        state, _ = interface.get_state_mlp(flag_coord_aliens=True, flag_boucliers=True)
        print("🧠 État (get_state) avec labels (valeurs dénormalisées) :")
        state_denorm = (state / 2.0 + 0.5)
        # ... (Logique d'affichage détaillée identique à invaders.py) ...
        # Pour simplifier ici, on garde l'essentiel, mais le code original peut être copié/collé si besoin de debug précis.
        print(f"Vecteur d'état brut: {state}")

    @staticmethod
    def create_fig(trainer, NB_DE_FRAMES_STEP, nb_parties, scores_moyens, fenetre_lissage, 
                   epsilons_or_sigmas, high_score, flag_aliens, flag_boucliers, 
                   steps_cumules, reward_str="", filename="Invaders_fig", nb_parties_pente=1000):
        
        matplotlib.use("Agg")
        fig, ax1 = plt.subplots(figsize=(12, 8), constrained_layout=True)
        
        # Axe Sigma/Epsilon
        ax1.set_xlabel("Nombre d'épisodes", fontsize=10)
        ax1.set_ylabel("Sigma FC-In (bleu)/FC-Out (orange)" if trainer.config.use_noisy else "Epsilon", color="tab:blue", fontsize=7, fontweight="bold")
        
        if trainer.config.use_noisy:
            ax1.plot(epsilons_or_sigmas[0], color="tab:blue", linestyle="dashed", lw=0.8)
            ax1.plot(epsilons_or_sigmas[1], color="tab:orange", linestyle="dashdot", lw=0.8)
        else:
            ax1.plot(epsilons_or_sigmas[0], color="tab:blue", linestyle="dashed")
            
        # Score moyen
        ax2 = ax1.twinx()
        ax2.set_ylabel("Score moyen", color="tab:red", rotation=270, labelpad=15, fontsize=8, fontweight="bold")
        ax2.plot(scores_moyens, color="tab:red")
        
        # Steps cumulés
        ax3 = ax1.twinx()
        ax3.spines["left"].set_position(("axes", -0.04))
        ax3.spines["left"].set_visible(True)
        ax3.yaxis.set_label_position("left")
        ax3.yaxis.set_ticks_position("left")
        steps_k = [s / 1000 for s in steps_cumules]
        ax3.plot(steps_k, color="tab:green", linestyle=":", label="Steps (k)")
        ax3.set_ylabel("Steps cumulés (k)", color="tab:green", fontsize=7, fontweight="bold")
        
        # Trendlines
        if len(scores_moyens) > 1:
            x_all = np.arange(len(scores_moyens))
            z = np.polyfit(x_all, scores_moyens, 1)
            p = np.poly1d(z)
            ax2.plot(x_all, p(x_all), "tab:orange", alpha=0.6)
            
            # Pente locale
            n = min(nb_parties_pente, len(scores_moyens))
            x_recent = np.arange(len(scores_moyens)-n, len(scores_moyens))
            z_recent = np.polyfit(x_recent, scores_moyens[-n:], 1)
            p_recent = np.poly1d(z_recent)
            ax2.plot(x_recent, p_recent(x_recent), "tab:purple", linestyle="--")
            pente_recent = z_recent[0]
            
            midpoint = len(scores_moyens) - n // 2
            ax2.text(midpoint, p_recent(len(scores_moyens)-n//2), f"Pente: {pente_recent:.2f}", color="tab:purple", fontweight="bold")
        else:
            pente_recent = 0

        # Config info
        config_str = (f"Model={trainer.config.model_type}, Batch={trainer.config.batch_size}, LR={trainer.config.learning_rate}\n"
                      f"{reward_str}")
        ax1.text(0.01, 0.99, config_str, transform=ax1.transAxes, fontsize=6, va="top", color="gray")

        plt.title(f"Invaders AI: Score moyen {fenetre_lissage} derniers eps - HiSc: {high_score}")
        plt.savefig(filename + ".png", dpi=300)
        plt.close()
        return pente_recent

# ==================================================================================================
# APPLICATION PRINCIPALE (TRAINING SESSION)
# ==================================================================================================

class InvadersApp:
    """Classe principale orchestrant l'entraînement."""
    
    def __init__(self):
        self.debug = 0
        self.debug_lua = True
        self.flag_F8_pressed = False
        self.flag_F11_pressed = False
        self.flag_quit = False
        self.flag_create_fig = False
        self.vitesse_de_jeu = 50
        
        # Serveur Web
        self.web_server = GraphWebServer(graph_dir=".\\", host="0.0.0.0", port=5000, auto_display_latest=True)
        threading.Thread(target=self.web_server.start, daemon=True).start()
        
        # Audio
        pygame.mixer.init()

    def launch_mame(self, desactiver_video_son=False):
        """Lance le processus MAME."""
        command = [
            "D:\\Emulateurs\\Mame Officiel\\mame.exe",
            "-window", "-resolution", "448x576",
            "-skip_gameinfo",
            "-artwork_crop",
            "-sound", "none",
            "-console",
            "-noautosave",
            "invaders",
            "-autoboot_delay", "1",
            "-autoboot_script", "D:\\Emulateurs\\Mame Sets\\MAME EXTRAs\\plugins\\PythonBridgeSocket.lua",
        ]
        if desactiver_video_son:
            command.extend(["-video", "none", "-sound", "none"])
            
        startupinfo = subprocess.STARTUPINFO()
        startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
        startupinfo.wShowWindow = 7 # Minimisé
        
        self.process = subprocess.Popen(command, cwd="D:\\Emulateurs\\Mame Officiel", startupinfo=startupinfo, stderr=subprocess.DEVNULL)
        time.sleep(15) # Attente init MAME

    def setup_keyboard(self, trainer, config):
        """Configure les raccourcis clavier."""
        def on_key_press(event):
            def is_terminal_in_focus():
                hwnd = win32gui.GetForegroundWindow()
                title = win32gui.GetWindowText(hwnd)
                return title[-20:] == "- Visual Studio Code"

            if is_terminal_in_focus():
                if keyboard.is_pressed("shift") and keyboard.is_pressed("ctrl"):
                    if keyboard.is_pressed("f2"):
                        print("\n[F2] Sortie demandée.")
                        self.flag_quit = True
                    elif keyboard.is_pressed("f3") and self.debug > 0:
                        self.debug = 0
                        print(f"\n[F3] Debug RESET = {self.debug}")
                    elif keyboard.is_pressed("f4"):
                        self.debug += int(self.debug < 3)
                        print(f"\n[F4] Debug = {self.debug}")
                    elif keyboard.is_pressed("f5"):
                        trainer.epsilon += 0.01
                        print(f"\n[F5] Epsilon + : {trainer.epsilon:.3f}")
                    elif keyboard.is_pressed("f6"):
                        trainer.epsilon -= 0.01
                        print(f"\n[F6] Epsilon - : {trainer.epsilon:.3f}")
                    elif keyboard.is_pressed("f7") and not self.flag_create_fig:
                        print("\n[F7] Création figure demandée.")
                        self.flag_create_fig = True
                    elif keyboard.is_pressed("f8"):
                        self.flag_F8_pressed = True
                        print("\n[F8] Affichage Frame/State demandé.")
                    elif keyboard.is_pressed("f9"):
                        new_mode = "exploitation" if config.mode == "exploration" else "exploration"
                        print(f"\n[F9] Mode changé vers : {new_mode}")
                        config.mode = new_mode
                        trainer.set_mode(new_mode)
                    elif keyboard.is_pressed("f11"):
                        self.flag_F11_pressed = True
                        print("\n[F11] Toggle Debug Lua demandé.")

        keyboard.on_press(on_key_press)

    def run(self):
        """Boucle principale de l'application."""
        
        # 1. Configuration
        N = 4 # Historique
        NB_DE_FRAMES_STEP = 3
        model_type = "cnn" # "mlp", "cnn", "dreamer"
        
        # Paramètres spécifiques au modèle
        flag_aliens = False
        flag_boucliers = False
        cnn_type = None
        factor_div_frame = None
        
        if model_type.lower() == "cnn":
            cnn_type = "precise"
            full_frame_2D = (192, 176)
            factor_div_frame = 2
            input_size = (N,) + tuple(x // factor_div_frame for x in full_frame_2D)
            NB_DE_DEMANDES_PAR_STEP = str(2 + 3 + 4) # 2(state: img+aliens) + 3(action) + 4(score/status)
            TRAIN_EVERY_N_GLOBAL_STEPS = 4 # Optimisation vitesse : on entraîne moins souvent (tous les 8 steps)
        elif model_type.lower() == "mlp":
            flag_aliens = True
            flag_boucliers = False
            input_size = 11 + 2 + (6 + 11) * flag_aliens + 4 * flag_boucliers + 2
            NB_DE_DEMANDES_PAR_STEP = str((11 + 2 + 2) + flag_aliens + 3 + 2 + 2) # 15(state)+1(aliens)+3(action)+4(score/status) = 23
            TRAIN_EVERY_N_GLOBAL_STEPS = 4
        elif model_type.lower() == "dreamer":
            factor_div_frame = 2
            full_frame_2D = (96, 176)
            input_size = (N,) + full_frame_2D
            NB_DE_DEMANDES_PAR_STEP = str(2 + 3 + 4) # 2(state: img+aliens) + 3(action) + 4(score/status)
            TRAIN_EVERY_N_GLOBAL_STEPS = 10
        else:
            raise ValueError(f"Modèle {model_type} non supporté.")

        # Rewards
        reward_clipping_deepmind = False  # Ne PAS clipper avec des rewards custom (sinon step=-1 == mort=-1)
        reward_aliens_mult = 0.1         # Alien 10pts -> +1.0 reward.
        reward_kill = -10.0              # Augmenté (-10) pour rendre le suicide tactique non rentable (vs +3 max par alien)
        reward_alive = 0.0              # Réduit drastiquement pour éviter le camping passif.
        reward_mult_step = 0.0  # très léger
        mult_reward_state = 0.0          # Augmenté: la descente doit coûter plus cher que la survie ne rapporte.
        reward_end_of_game = -10.0       # game over sévère
        
        # Exploration
        use_noisy = False                # Désactivé pour garantir la précision en fin d'apprentissage
        epsilon_start = 1.0
        epsilon_end = 0.02               # On garde 2% d'aléatoire pour ne pas figer totalement
        target_steps_for_epsilon_end = 500_000 # Exploration plus longue (500k steps) pour bien apprendre le début
        epsilon_linear = (epsilon_start - epsilon_end) / target_steps_for_epsilon_end
        epsilon_decay = 0
        epsilon_add = ((epsilon_start - epsilon_end) / target_steps_for_epsilon_end) * NB_DE_FRAMES_STEP * 100 if not use_noisy else 0.0

        # Config Object
        config = TrainingConfig(
            state_history_size=N,
            input_size=input_size,
            hidden_layers=2,
            hidden_size=256,
            output_size=6,
            learning_rate=0.00025,
            gamma=0.99,
            use_noisy=use_noisy,
            rainbow_eval=250_000,
            rainbow_eval_pourcent=5,  # 5% du temps en évaluation (standard Rainbow)
            epsilon_start=epsilon_start,
            epsilon_end=epsilon_end,
            epsilon_linear=epsilon_linear,
            epsilon_decay=epsilon_decay,
            epsilon_add=epsilon_add,
            buffer_capacity=200_000, # Réduit à ~13 Go pour éviter de saturer la RAM/Disque
            batch_size=64,
            min_history_size=50000,  # compromis raisonnable (Rainbow = 80k)
            prioritized_replay=True,
            target_update_freq=10000,
            double_dqn=True,
            dueling=True,
            nstep=True,
            nstep_n=3,
            model_type=model_type,
            cnn_type=cnn_type,
            mode="exploration"
        )

        # 2. Initialisation Système
        self.launch_mame()
        comm = MameCommunicator("localhost", 12345)
        game = InvadersInterface(comm)
        
        # State Extractor setup
        # Note: colonnes_deja_detruites est réinitialisé à chaque épisode, donc on passera une ref mutable plus tard
        # Ici on initialise avec une liste dummy qui sera remplacée ou modifiée
        colonnes_deja_detruites = [False] * 11
        config.state_extractor = StateExtractor(game, model_type, flag_aliens, flag_boucliers, factor_div_frame, mult_reward_state, colonnes_deja_detruites)

        # Trainer setup
        if config.model_type.lower() == "dreamer":
            trainer = DreamerTrainer(config)
        else:
            trainer = DQNTrainer(config)
        
        print(f"Device: {trainer.device}")
        trainer.load_model("./invaders.pth")
        trainer.load_buffer("./invaders.buffer")

        self.setup_keyboard(trainer, config)
        
        # Recorder
        record = False
        recorder = ScreenRecorder() if record else None

        # Stats containers
        fenetre_moyenne = 100
        collection_score = deque(maxlen=fenetre_moyenne)
        list_mean_scores = []
        list_eps_sigmas = [[], []]
        list_cumul_steps = []
        mean_score = mean_score_old = last_score = high_score = best_mean_score = 0
        nb_steps_total = 0

        # Init MAME params
        comm.communicate([
            f"write_memory {Memory.NUM_COINS}(1)",
            "execute P1_start(1)",
            f"execute throttle_rate({self.vitesse_de_jeu})",
            "execute throttled(0)",
            f"frame_per_step {NB_DE_FRAMES_STEP}",
        ])

        print(f"Démarrage entraînement... Mode: {config.model_type}")

        # 3. Boucle d'épisodes
        num_episodes = 99999
        for episode in range(num_episodes):
            trainer.config.current_episode = episode
            if record: recorder.start_recording()
            
            step = 0
            sum_rewards = 0
            score = 0
            
            # Reset variables épisode
            for i in range(11): colonnes_deja_detruites[i] = False # Reset in-place pour StateExtractor
            
            if self.flag_quit:
                print("Arrêt demandé.")
                break

            # Attente début partie
            comm.communicate([f"write_memory {Memory.NUM_COINS}(1)"])

            # Remplissage historique initial
            if hasattr(trainer, 'state_history') and isinstance(trainer.state_history, deque):
                trainer.state_history.clear()
            else:
                local_state_history = deque(maxlen=config.state_history_size)

            for _ in range(config.state_history_size):
                frame, _ = config.state_extractor()
                if hasattr(trainer, 'state_history'): trainer.state_history.append(frame)
                else: local_state_history.append(frame)
            
            if hasattr(trainer, 'state_history'):
                initial_obs_stack = np.stack(trainer.state_history, axis=0)
            else:
                initial_obs_stack = np.stack(local_state_history, axis=0)

            if config.model_type.lower() == "dreamer":
                trainer.encode_state_initial(initial_obs_stack)
                prev_action_value = 0

            # Synchro MAME
            NotEndOfGame = 0
            PlayerIsOK = 0
            NewGameStarting = 0
            while NotEndOfGame == 0:
                NotEndOfGame = int(comm.communicate([f"read_memory {Memory.PLAYER_1_ALIVE}"])[0])
                time.sleep(0.01)
            while PlayerIsOK == 0:
                PlayerIsOK = int(comm.communicate([f"read_memory {Memory.PLAYER_OK}"])[0])
                time.sleep(0.01)
            while NewGameStarting != 55:
                NewGameStarting = int(comm.communicate([f"read_memory {Memory.NUM_ALIENS}"])[0])
                time.sleep(0.01)
            
            comm.communicate([f"wait_for {NB_DE_DEMANDES_PAR_STEP}"])
            start_steps_time = time.time()
            nb_vies = 3
            
            current_loop_obs_stack = initial_obs_stack
            invaders_loop_frame_history = deque(maxlen=config.state_history_size)
            for f in initial_obs_stack: invaders_loop_frame_history.append(f)

            # --- Boucle de jeu (Step) ---
            while NotEndOfGame == 1:
                # 1. Action
                if config.model_type.lower() == "dreamer":
                    action = trainer.dreamer_step(current_loop_obs_stack, prev_action_value)
                else:
                    if config.model_type.lower() == "cnn":
                        state_input = current_loop_obs_stack
                    else:
                        state_input = np.concatenate(list(invaders_loop_frame_history), axis=0)
                    action = trainer.select_action(state_input)

                # 2. Exécution
                game.execute_action(action)

                # 3. Observation & Score
                next_frame, reward_state_comp = config.state_extractor()
                _last_score = score
                score, PlayerIsOK, NotEndOfGame = game.get_score_and_status(_last_score)

                # 4. Reward
                if PlayerIsOK == 1:
                    reward = reward_alive
                elif nb_vies > 1:
                    reward = reward_kill
                else:
                    reward = reward_kill + reward_end_of_game
                reward += ((score - _last_score) * reward_aliens_mult) + (reward_mult_step) + reward_state_comp
                if reward_clipping_deepmind: reward = np.sign(reward)
                
                sum_rewards += reward
                done = (PlayerIsOK == 0)

                # 5. Stack Update
                invaders_loop_frame_history.append(next_frame)
                next_obs_stack = np.stack(invaders_loop_frame_history, axis=0)

                # 6. Store & Train
                if config.model_type.lower() == "dreamer":
                    trainer.store_transition(current_loop_obs_stack, action, reward, next_obs_stack, False)
                elif config.mode.lower() == "exploration":
                    if config.model_type.lower() == "cnn":
                        s_buf = current_loop_obs_stack
                        ns_buf = next_obs_stack
                    else:
                        s_buf = np.concatenate(list(invaders_loop_frame_history), axis=0) # Utilise l'état AVANT append pour s_buf? Non, history a déjà next_frame.
                        # Correction logique originale:
                        # invaders_loop_frame_history a déjà next_frame.
                        # current_loop_obs_stack est l'état t.
                        # next_obs_stack est l'état t+1.
                        # Pour MLP, il faut reconstruire t et t+1 concaténés.
                        # current_loop_obs_stack est (N, InputSize) pour MLP? Non, c'est (N, H, W) ou (N, VecSize) si state_extractor renvoie VecSize.
                        # state_extractor MLP renvoie un vecteur 1D.
                        # Donc current_loop_obs_stack est (N, VecSize).
                        # np.concatenate sur axis=0 donne (N*VecSize).
                        s_buf = np.concatenate(current_loop_obs_stack, axis=0)
                        ns_buf = np.concatenate(next_obs_stack, axis=0)

                    if trainer.config.nstep:
                        nstep_tr = trainer.nstep_wrapper.append(s_buf, action, reward, done, ns_buf)
                        if nstep_tr: trainer.replay_buffer.push(*nstep_tr)
                        if done:
                            for tr in trainer.nstep_wrapper.flush(): trainer.replay_buffer.push(*tr)
                    else:
                        trainer.replay_buffer.push(s_buf, action, reward, ns_buf, done)

                current_loop_obs_stack = next_obs_stack
                if config.model_type.lower() == "dreamer": prev_action_value = action
                else: trainer.update_epsilon()

                if nb_steps_total % TRAIN_EVERY_N_GLOBAL_STEPS == 0:
                    trainer.train_step()
                
                step += 1
                nb_steps_total += 1

                # 7. Debug / Affichage
                if self.debug >= 1:
                    elapsed = (time.time() - start_steps_time)
                    d_str, t_str = GameConstants.ACTIONS[action]
                    act_str = f"{d_str}{'+tir' if t_str else ''}"
                    print(
                        f"⏱️={(elapsed / step)*1000.0:2.2f}ms "
                        f"N°={episode:<5d}action={act_str:<8} PlayerIsOK={PlayerIsOK:<2d}NotEndOfGame={NotEndOfGame:<2d}"
                        f"reward={reward:<6.3f}score={score:<5d} sum_rewards={sum_rewards:<6.0f}"
                        f"nb_mess_step={comm.number_of_messages:<4d}nb_step={step:<5d}"
                    )
                
                comm.number_of_messages = 0
                
                # Gestion Touches
                if self.flag_F11_pressed:
                    self.debug_lua = not self.debug_lua
                    comm.communicate([f"debug {'on' if self.debug_lua else 'off'}"])
                    self.flag_F11_pressed = False
                
                if self.flag_F8_pressed:
                    if config.model_type == "mlp":
                        Visualizer.afficher_get_state(game)
                    else:
                        Visualizer.afficher_frame(next_frame, factor_div_frame)
                    self.flag_F8_pressed = False

                # 8. Mort / Fin vie
                if PlayerIsOK == 0:
                    comm.communicate(["wait_for 4"])  # 4 = nb de messages dans get_score_and_status
                    while PlayerIsOK == 0 and NotEndOfGame == 1:
                        _, PlayerIsOK, NotEndOfGame = game.get_score_and_status(score)
                    if PlayerIsOK == 1:
                        nb_vies -= 1
                        comm.communicate([f"wait_for {NB_DE_DEMANDES_PAR_STEP}"])
                    if NotEndOfGame == 0 and config.model_type.lower() == "dreamer":
                        trainer.store_transition(current_loop_obs_stack, action, reward, next_obs_stack, True)

            # --- Fin Episode ---
            comm.communicate(["wait_for 1"])
            comm.communicate([f'draw_text(25,1,"Game number: {episode+1:04d} - mean score={mean_score:04.0f} - ba(c)o 2026")'])
            if record:
                recorder.stop_recording()
                if score > last_score:
                    if last_score != 0 and os.path.exists(f"best_game_{last_score}.avi"):
                        os.remove(f"best_game_{last_score}.avi")
                    shutil.copy("output-obs.mp4", f"best_game_{score}.avi")
                    last_score = score

            # Stats & Graphiques
            if self.flag_create_fig:
                reward_str = f"R_alive={reward_alive}, R_step={reward_mult_step}, R_kill={reward_kill}"
                Visualizer.create_fig(trainer, NB_DE_FRAMES_STEP, episode, list_mean_scores, fenetre_moyenne,
                                      list_eps_sigmas, high_score, flag_aliens, flag_boucliers, list_cumul_steps, reward_str, "Invaders_fig_ask")
                self.flag_create_fig = False

            # Logging Sigmas/Epsilon
            if use_noisy:
                sigmas = trainer.dqn.get_sigma_values()
                s1 = sigmas.get("fc1", 0.0) if config.model_type == "cnn" else sigmas.get("hidden_modules.0.0", 0.0)
                s2 = sigmas.get("advantage_head", 0.0) if config.dueling else sigmas.get("output_layer", 0.0)
                list_eps_sigmas[0].append(s1)
                list_eps_sigmas[1].append(s2)
            else:
                list_eps_sigmas[0].append(trainer.epsilon)

            list_cumul_steps.append(nb_steps_total)
            collection_score.append(score)
            mean_score_old = mean_score
            mean_score = round(sum(collection_score) / len(collection_score), 2)
            list_mean_scores.append(mean_score)
            
            if score > high_score: high_score = score
            
            # Sauvegarde Best Model
            if episode >= fenetre_moyenne and mean_score > best_mean_score:
                best_mean_score = mean_score
                trainer.save_model("./invaders_best.pth")
                print(f"{Fore.YELLOW}🏆 Record moyen ({best_mean_score}) ! Sauvegarde best.{Style.RESET_ALL}")
                try: pygame.mixer.Sound("c:\\Windows\\Media\\tada.wav").play()
                except: pass

            # Update Epsilon (si pas noisy)
            if not (config.model_type.lower() == "dreamer" or config.use_noisy):
                if mean_score_old > mean_score:
                    trainer.epsilon += config.epsilon_add if trainer.epsilon < config.epsilon_start else 0.0
                if trainer.epsilon < 0.001 and trainer.config.mode == "exploration":
                    trainer.config.mode = "exploitation"
                    trainer.set_mode("exploitation")
                    print("===> Passage auto en EXPLOITATION")

            # Console Log
            _d = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            if config.use_noisy:
                sigma_vals = trainer.dqn.get_sigma_values()
                sigma_str = ", ".join([f"{name} = {val:.6f}" for name, val in sigma_vals.items()])
                exploration_str = f"[sigma {sigma_str}]"
            else:
                exploration_str = f"[ε={trainer.epsilon:.4f}]"
            exploration_str = exploration_str if config.mode != "exploitation" else "[*** Mode EXPLOITATION ***]"
            exploration_str = exploration_str if config.model_type != "dreamer" else "[Mode DREAMER.V2]"

            print(
                f"N°{episode+1} [{_d}][steps_ep,all={step:4d},{str(nb_steps_total//1000)+'k':>5}]"
                + (f"[Buffer={trainer.replay_buffer.size/1000:.0f}k/{trainer.config.buffer_capacity/1000:.0f}k]" if hasattr(trainer, "replay_buffer") else "")
                + exploration_str + f"[rewards={sum_rewards:5.0f}]" + f"[score={score:3d}][score moyen={mean_score:3.0f}]"
            )

            # Sauvegarde périodique
            if (episode + 1) % 10 == 0:
                trainer.save_model("./invaders.pth")
                reward_str = f"R_alive={reward_alive}, R_step={reward_mult_step}, R_kill={reward_kill}"
                Visualizer.create_fig(trainer, NB_DE_FRAMES_STEP, episode, list_mean_scores, fenetre_moyenne,
                                      list_eps_sigmas, high_score, flag_aliens, flag_boucliers, list_cumul_steps, reward_str)


            # Sauvegarde buffer périodique (sécurité anti-crash)
            if (episode + 1) % 100 == 0:
                trainer.save_buffer("./invaders.buffer")


        # Fin du programme
        final_tag = f"{config.model_type}_N={N}_batch={config.batch_size}_ep={episode}"
        trainer.save_model(f"./invaders_{final_tag}.pth")
        trainer.save_buffer("./invaders.buffer")
        
        if record:
            recorder.stop_recording()
            recorder.ws.disconnect()
        
        self.process.terminate()

if __name__ == "__main__":
    app = InvadersApp()
    app.run()