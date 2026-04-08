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
import sys
from collections import deque
from datetime import datetime
from colorama import Fore, Style
import matplotlib
import matplotlib.pyplot as plt

# ==================================================================================================
# CONFIGURATION DES CHEMINS (RELATIFS)
# ==================================================================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPT_DIR)
CORE_DIR = os.path.join(ROOT_DIR, "core")
MEDIA_DIR = os.path.join(ROOT_DIR, "media")

# Assurer la présence des dossiers
os.makedirs(MEDIA_DIR, exist_ok=True)

# Ajout du dossier 'core' au sys.path pour les imports locaux
if CORE_DIR not in sys.path:
    sys.path.append(CORE_DIR)

# Imports locaux (cherchés dans CORE_DIR grâce au sys.path.append)
from MameCommSocket import MameCommunicator
from ScreenRecorder import ScreenRecorder
from AI_Mame import TrainingConfig, DQNTrainer, GraphWebServer
from dreamerv2 import DreamerTrainer

# ==================================================================================================
# CONSTANTES & ADRESSES MÉMOIRE https://www.computerarcheology.com/Arcade/SpaceInvaders/RAMUse.html
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

    def get_state_cnn(self, factor_div=2, mult_reward_state=0.0, is_dreamer=False):
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
        
        if is_dreamer:
            # Dreamer exige un multiple parfait de 16. On prend 24 colonnes = 192 pixels.
            # Cela donne une image source de 192x192 parfaitement centrée sur l'action.
            cropped_columns = columns[:, 4:28]
        else:
            # Crop standard (25 colonnes = 200px)
            cropped_columns = columns[:, 3:28]
            
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
            f"read_memory {Memory.PLAYER_OK}", f"read_memory {Memory.PLAYER_1_ALIVE}",
            f"read_memory {Memory.P1_SHIPS_REM}"
        ])
        
        if not response_grouped or len(response_grouped) < 5:
            return last_score, 0, 0, 0 
        
        P1ScorL_v, P1ScorM_v, PlayerIsOK, NotEndOfGame, lives = list(map(int, response_grouped))
        score = (P1ScorL_v >> 4) * 10 + (P1ScorM_v & 0x0F) * 100 + ((P1ScorM_v) >> 4) * 1000
        
        return score, PlayerIsOK, NotEndOfGame, lives

    def get_complete_step(self, action_code, factor_div=1, mult_reward_state=0.0):
        """
        ENVOIE l'action, FAIT un pas de frame, et LIT tout sur la frame cible (Image + Score + Status).
        Méthode atomique pour garantir la cohérence temporelle.
        """
        direction, tirer = GameConstants.ACTIONS[action_code]
        messages = [
            f"execute P1_left({int(direction=='left')})",
            f"execute P1_right({int(direction=='rght')})",
            f"execute P1_Button_1({int(tirer)})",
            "read_memory_range 2400(7168)",
            f"read_memory {Memory.REF_ALIEN_YR}",
            f"read_memory {Memory.P1_SCORE_L}", f"read_memory {Memory.P1_SCORE_M}",
            f"read_memory {Memory.PLAYER_OK}", f"read_memory {Memory.PLAYER_1_ALIVE}",
            f"read_memory {Memory.P1_SHIPS_REM}"
        ]
        
        response = self.comm.communicate(messages)
        if not response or len(response) < 10: 
            return None, 0.0, 0, 0, 0, 0

        # --- IMAGE ---
        raw_str = response[3]; refAlienYr_val = int(response[4])
        all_bytes = np.array(list(map(int, raw_str.split(","))), dtype=np.uint8)
        columns = all_bytes.reshape(224, 32)[16:208, :]
        cropped_columns = columns[:, 3:28]
        image = np.unpackbits(cropped_columns, axis=1, bitorder="little")
        if factor_div > 1:
            h, w = image.shape
            h_, w_ = h // factor_div, w // factor_div
            image = image[:h_ * factor_div, :w_ * factor_div].reshape(h_, factor_div, w_, factor_div).max(axis=(1, 3))
        penalty_descente = ((120 - refAlienYr_val) / 10.0) * mult_reward_state
        
        # --- STATUS ---
        P1ScorL_v, P1ScorM_v, PlayerIsOK, NotEndOfGame, lives = list(map(int, response[5:10]))
        score = (P1ScorL_v >> 4) * 10 + (P1ScorM_v & 0x0F) * 100 + ((P1ScorM_v) >> 4) * 1000
        
        return image.astype(np.float32), penalty_descente, score, PlayerIsOK, NotEndOfGame, lives

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
            return self.interface.get_state_cnn(factor_div=self.factor_div_frame, mult_reward_state=self.mult_reward_state, is_dreamer=(self.model_type == "dreamer"))
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
        
        # 1. Sauvegarde de l'image brute au pixel près (ex: 96x96)
        raw_filename = f"frame_RAW_{frame_img.shape[1]}x{frame_img.shape[0]}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        plt.imsave(raw_filename, frame_img, cmap="gray")

        # 2. Sauvegarde du graphique de visualisation (agrandi avec titre)
        plt.figure(figsize=(6, 6))
        plt.imshow(frame_img, cmap="gray", interpolation="nearest")
        plt.title(f"Vision de l'IA (Résolution interne : {frame_img.shape[1]}x{frame_img.shape[0]})")
        plt.axis("off")
        filename = f"frame(shrink factor={factor_div})_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        plt.savefig(filename, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Frames sauvegardées :\n -> {filename} (Agrandie pour humains)\n -> {raw_filename} (Pixels bruts exacts de l'IA)")

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
                   steps_cumules, reward_str="", filename="Invaders_fig", nb_parties_pente=1000, max_scores=None):
        
        matplotlib.use("Agg")
        fig, ax1 = plt.subplots(figsize=(12, 8), constrained_layout=True)
        
        # Axe Sigma/Epsilon
        ax1.set_xlabel("Nombre d'épisodes", fontsize=10)
        ax1.set_ylabel("Sigma FC-In (bleu)/FC-Out (orange)" if trainer.config.use_noisy else "Epsilon", color="tab:blue", fontsize=7, fontweight="bold", rotation=90)
        
        ax1.tick_params(axis='y', labelcolor="tab:blue")
        if trainer.config.use_noisy:
            ax1.plot(epsilons_or_sigmas[0], color="tab:blue", linestyle="dashed", lw=0.8, label="Sigma In (Cachée)")
            ax1.plot(epsilons_or_sigmas[1], color="tab:orange", linestyle="dashdot", lw=0.8, label="Sigma Out (Actions)")
            ax1.legend(loc="upper left", fontsize=8)
        else:
            ax1.plot(epsilons_or_sigmas[0], color="tab:blue", linestyle="dashed", label="Epsilon")
            ax1.legend(loc="upper left", fontsize=8)
            
        # Score moyen
        ax2 = ax1.twinx()
        ax2.set_ylabel("Score Moyen", color="tab:red", rotation=270, labelpad=15, fontsize=8, fontweight="bold")
        ax2.plot(scores_moyens, color="tab:red", label="Score Moyen")
        ax2.tick_params(axis='y', labelcolor="tab:red")
        
        # Ajout courbe Max Score
        if max_scores and len(max_scores) == len(scores_moyens):
            ax4 = ax1.twinx()
            ax4.set_ylabel("Max Score (fenêtre)", color="tab:green", rotation=270, labelpad=15, fontsize=7, fontweight="bold", y=0.85)
            ax4.plot(max_scores, color="tab:green", alpha=0.3, linestyle="-", linewidth=1, label="Max Score (fenêtre)")
            ax4.tick_params(axis='y', labelcolor="tab:green", labelsize=7)
            ax4.legend(loc="lower right", fontsize=7)
        
        ax2.legend(loc="upper right", fontsize=8)
        
        # Steps cumulés
        ax3 = ax1.twinx()
        ax3.spines["left"].set_visible(True)
        ax3.spines["right"].set_visible(False)
        ax3.yaxis.set_label_position("left")
        ax3.yaxis.set_ticks_position("left")
        steps_k = [s / 1000 for s in steps_cumules]
        ax3.plot(steps_k, color="tab:green", linestyle=":", label="Steps (k)")
        ax3.set_ylabel("Steps cumulés (k)", color="tab:green", fontsize=7, fontweight="bold", rotation=90)
        ax3.tick_params(axis='y', labelcolor="tab:green")
        
        # Trendlines
        if len(scores_moyens) > 1:
            x_all = np.arange(len(scores_moyens))
            z = np.polyfit(x_all, scores_moyens, 1)
            p = np.poly1d(z)
            ax2.plot(x_all, p(x_all), "tab:orange", alpha=0.6)
            
            # Pente locale
            n = min(nb_parties_pente, len(scores_moyens))
            x_recent = np.arange(len(scores_moyens)-n, len(scores_moyens))
            y_recent = scores_moyens[-n:]
            z_recent = np.polyfit(x_recent, y_recent, 1)
            p_recent = np.poly1d(z_recent)
            ax2.plot(x_recent, p_recent(x_recent), "tab:purple", linestyle="--")
            pente_recent = z_recent[0]
            
            # Calcul du Coefficient de Variation (Stabilité en %)
            std_recent = np.std(y_recent)
            mean_recent = np.mean(y_recent)
            cv_percent = (std_recent / mean_recent * 100) if mean_recent != 0 else 0.0
            
            midpoint = len(scores_moyens) - n // 2
            ax2.text(midpoint, p_recent(len(scores_moyens)-n//2), f"Pente: {pente_recent:.2f}\nVar: {cv_percent:.1f}%", color="tab:purple", fontweight="bold")
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
        print(f"🐍 Python exécutable : {sys.executable}")
        self.debug = 0
        self.debug_lua = True
        self.flag_F8_pressed = False
        self.flag_quit = False
        self.flag_create_fig = False
        self.is_normal_speed = False
        self.training_speed = 50.0
        self.slow_speed = 1.0
        self.pending_commands = [] # File d'attente pour les commandes thread-safe
        
        # Serveur Web
        self.web_server = GraphWebServer(graph_dir=".\\", host="0.0.0.0", port=5000, auto_display_latest=True)
        threading.Thread(target=self.web_server.start, daemon=True).start()
        
        # Audio
        pygame.mixer.init()

    def kill_existing_mame(self):
        """Force la fermeture des processus MAME existants pour libérer le port."""
        for proc in psutil.process_iter(['pid', 'name']):
            try:
                if proc.info['name'] and 'mame' in proc.info['name'].lower():
                    print(f"{Fore.YELLOW}🧹 Nettoyage processus MAME orphelin (PID {proc.info['pid']})...{Style.RESET_ALL}")
                    proc.terminate()
                    proc.wait(timeout=2)
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                pass

    def generate_best_gif(self, video_path, score):
        """Génère un GIF léger et recadré de la meilleure partie."""
        try:
            # Essai d'importer les composants pour MoviePy v2+
            from moviepy import VideoFileClip
            try:
                from moviepy.video.fx.all import crop, resize
            except ImportError:
                # Compatibilité v2.x si fx.all est absent
                def crop(clip, **kwargs):
                    return clip.cropped(**kwargs) if hasattr(clip, 'cropped') else clip.crop(**kwargs)
                def resize(clip, **kwargs):
                    return clip.resized(**kwargs) if hasattr(clip, 'resized') else clip.resize(**kwargs)
            is_v2 = True
        except ImportError:
            try:
                # Fallback pour MoviePy v1
                from moviepy.editor import VideoFileClip
                is_v2 = False
            except ImportError as e:
                # On n'affiche le warning qu'une fois pour ne pas spammer
                if not hasattr(self, 'moviepy_warning_shown'):
                    print(f"{Fore.YELLOW}⚠️ Pour générer des GIFs, installez moviepy : pip install moviepy")
                    print(f"   Erreur : {e}{Style.RESET_ALL}")
                    self.moviepy_warning_shown = True
                return

        gif_path = os.path.join(MEDIA_DIR, f"invaders_best_game_{score}.gif")
        mp4_path = os.path.join(MEDIA_DIR, f"invaders_best_game_{score}_crop.mp4")
        clip = None
        clip_cropped = None
        clip_gif = None
        clip_mp4 = None
        try:
            clip = VideoFileClip(video_path)
            w, h = clip.size
            
            # --- CONFIGURATION DU RECADRAGE (CROP) ---
            # Ajustez 'crop_margin' pour retirer les bandes noires à gauche et à droite
            # OBS capture souvent en 1920x1080, mais si c'est la fenêtre native (448px), 553px est trop grand !
            # On vérifie la largeur avant de cropper.
            crop_margin = 0
            if w > 1000: # Probablement 1920x1080
                crop_margin = 553 
            
            clip_cropped = clip
            if crop_margin > 0 and w > 2 * crop_margin:
                if not is_v2: # MoviePy v1
                    clip_cropped = clip.crop(x1=crop_margin, width=w - 2 * crop_margin)
                else: # MoviePy v2
                    clip_cropped = crop(clip, x1=crop_margin, width=w - 2 * crop_margin)
                print(f"✂️ Crop appliqué (Marge {crop_margin}px)")
            elif crop_margin > 0:
                print(f"{Fore.YELLOW}⚠️ Crop annulé : Largeur vidéo ({w}) trop petite pour marge {crop_margin}.{Style.RESET_ALL}")

            # --- 1. Génération GIF (Léger, 10fps, Hauteur 256) ---
            target_height_gif = 256
            clip_gif = clip_cropped
            if not is_v2: # MoviePy v1
                clip_gif = clip_gif.resize(height=target_height_gif)
            else: # MoviePy v2
                clip_gif = resize(clip_gif, height=target_height_gif)
            clip_gif.write_gif(gif_path, fps=10, logger=None)
            print(f"{Fore.GREEN}🎞️ GIF créé : {gif_path}{Style.RESET_ALL}")
            
            # --- 2. Génération MP4 Crop (Qualité, Original FPS, Hauteur 320) ---
            # Approche ROBUSTE pour FFMPEG (Largeur/Hauteur paires obligatoires)
            target_h_mp4 = 320
            
            # Etape A : Resize sur la hauteur uniquement (préserve le ratio)
            if not is_v2: # MoviePy v1
                clip_mp4 = clip_cropped.resize(height=target_h_mp4)
            else: # MoviePy v2
                clip_mp4 = resize(clip_cropped, height=target_h_mp4)
            
            # Etape B : Vérification Dimensions Paires (Fix libx264)
            w_curr, h_curr = clip_mp4.size
            new_w = w_curr if w_curr % 2 == 0 else w_curr - 1
            new_h = h_curr if h_curr % 2 == 0 else h_curr - 1
            
            if new_w != w_curr or new_h != h_curr:
                print(f"⚠️ Correction dimensions impaires : {w_curr}x{h_curr} -> {new_w}x{new_h}")
                # Crop explicite (Top-Left) pour éviter les erreurs de centrage flottant
                if not is_v2: # MoviePy v1
                    clip_mp4 = clip_mp4.crop(x1=0, y1=0, width=new_w, height=new_h)
                else: # MoviePy v2
                    clip_mp4 = crop(clip_mp4, x1=0, y1=0, width=new_w, height=new_h)

            # Utilisation du FPS original pour garder toutes les images
            original_fps = clip.fps if clip.fps else 30
            clip_mp4.write_videofile(mp4_path, fps=original_fps, codec='libx264', ffmpeg_params=['-pix_fmt', 'yuv420p'], logger=None)
            print(f"{Fore.GREEN}🎥 MP4 recadré créé : {mp4_path} ({original_fps} fps){Style.RESET_ALL}")

        except Exception as e:
            print(f"{Fore.RED}❌ Erreur création GIF/AVI : {e}{Style.RESET_ALL}")
        finally:
            # Nettoyage des clips
            for c in [clip_gif, clip_mp4, clip_cropped, clip]:
                if c:
                    try: c.close()
                    except: pass
                    
    def get_next_experiment_id(self):
        """Lit le fichier de résultats pour déterminer le prochain ID d'expérience."""
        filename = os.path.join(SCRIPT_DIR, "resultats_invaders.txt")
        last_idx = 0
        if os.path.exists(filename):
            try:
                with open(filename, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line: continue
                        
                        idx = -1
                        # Ancien format: "30[input..."
                        if line[0].isdigit():
                            parts = line.split('[')
                            if parts[0].strip().isdigit():
                                idx = int(parts[0].strip())
                        # Nouveau format: "[31][model..."
                        elif line.startswith('['):
                            parts = line.split(']')
                            if parts:
                                val = parts[0][1:].strip()
                                if val.isdigit():
                                    idx = int(val)
                                    
                        if idx > last_idx: last_idx = idx
            except Exception:
                pass
        return last_idx + 1

    def log_results(self, experiment_id, nb_episodes, mean_scores, config: TrainingConfig, nb_mess, nb_step_frame, r_kill, r_step):
        filename = os.path.join(SCRIPT_DIR, "resultats_invaders.txt")
        
        # Stats
        start_mean = mean_scores[0] if mean_scores else 0
        end_mean = mean_scores[-1] if mean_scores else 0
        max_mean = max(mean_scores) if mean_scores else 0

        # --- Construction dynamique des lignes de log ---
        
        # Ligne 1: Architecture et hyperparamètres principaux
        params1_list = [f"{experiment_id}"]
        
        model_str = f"model={config.model_type}"
        if config.model_type.lower() == 'cnn' and config.cnn_type:
            model_str += f"({config.cnn_type})"
        params1_list.append(model_str)
        
        params1_list.append(f"input={str(config.input_size)}")
        
        if config.model_type.lower() == 'mlp':
            params1_list.append(f"hidden={config.hidden_size}*{config.hidden_layers}")
        elif config.model_type.lower() == 'dreamer':
            params1_list.append(f"latent_dim={getattr(config, 'latent_dim', 'N/A')}")
            params1_list.append(f"rnn_hidden={getattr(config, 'rnn_hidden_dim', 'N/A')}")
        else: # CNN
            params1_list.append(f"fc_hidden={config.hidden_size}")

        if config.model_type.lower() != 'dreamer':
            params1_list.append(f"output={config.output_size}")
            
        params1_list.append(f"gamma={config.gamma}")
        params1_list.append(f"lr={config.learning_rate}")
        params1_list.append(f"r_kill={r_kill}")
        params1_list.append(f"r_step={r_step}")
        
        param_line1 = "".join([f"[{p}]" for p in params1_list])

        # Ligne 2: Exploration, Replay Buffer et Synchro
        params2_list = []
        if config.model_type.lower() != 'dreamer' and not config.use_noisy:
            decay_val = config.epsilon_linear if config.epsilon_linear > 0 else config.epsilon_decay
            decay_type = "linear" if config.epsilon_linear > 0 else "decay"
            params2_list.append(f"epsilon start={config.epsilon_start} end={config.epsilon_end} {decay_type}={decay_val}")
        
        params2_list.append(f"Replay_size={config.buffer_capacity}")
        params2_list.append(f"batch={config.batch_size}")
        
        if config.model_type.lower() == 'dreamer':
            params2_list.append(f"seq_len={getattr(config, 'sequence_length', 'N/A')}")
            params2_list.append(f"imag_horizon={getattr(config, 'imagination_horizon', 'N/A')}")
        else:
            params2_list.append(f"mess_step={nb_mess}")
            
        params2_list.append(f"frames_step={nb_step_frame}")
        params2_list.append(f"speed={self.training_speed}")
        
        param_line2 = "".join([f"[{p}]" for p in params2_list])

        # Ligne 3: Commentaire résumé
        if config.model_type.lower() == 'dreamer':
            algo_str = "DreamerV2 (Model-Based)"
        else:
            rainbow_features = []
            if config.use_noisy: rainbow_features.append("Noisy")
            if config.double_dqn: rainbow_features.append("Double")
            if config.dueling: rainbow_features.append("Dueling")
            if config.prioritized_replay: rainbow_features.append("PER")
            if config.nstep: rainbow_features.append(f"N-Step({config.nstep_n})")
            algo_str = f"Rainbow: [{', '.join(rainbow_features) if rainbow_features else 'Vanilla DQN'}]"

        comment = (f"=> {nb_episodes} parties. Score moyen: début={start_mean:.2f}, fin={end_mean:.2f}, max={max_mean:.2f}. {algo_str}")

        try:
            with open(filename, "a", encoding="utf-8") as f:
                f.write(f"\n\n{param_line1}\n{param_line2}\n{comment}")
            print(f"{Fore.CYAN}📝 Résultats sauvegardés dans {filename}{Style.RESET_ALL}")
        except Exception as e:
            print(f"{Fore.RED}❌ Erreur écriture resultats: {e}{Style.RESET_ALL}")

    def launch_obs(self):
        """Tente de lancer OBS Studio si non détecté."""
        obs_path = r"C:\Program Files\obs-studio\bin\64bit\obs64.exe"
        if os.path.exists(obs_path):
            try:
                print(f"{Fore.YELLOW}🚀 Lancement d'OBS Studio...{Style.RESET_ALL}")
                subprocess.Popen([obs_path], cwd=os.path.dirname(obs_path))
                time.sleep(8) # Temps de démarrage
            except Exception as e:
                print(f"{Fore.RED}Erreur lancement OBS: {e}{Style.RESET_ALL}")

    def launch_mame(self, desactiver_video_son=False, visible=False, record_inp=False):
        """Lance le processus MAME."""
        # --- CONFIGURATION MAME ---
        MAME_BIN_PATH = "D:\\Emulateurs\\Mame Officiel\\mame.exe"
        MAME_DIR = os.path.dirname(MAME_BIN_PATH)
        LUA_SCRIPT_PATH = os.path.join(CORE_DIR, "PythonBridgeSocket.lua").replace('\\', '/')
        
        command = [
            MAME_BIN_PATH,
            "-window", "-resolution", "448x576",
            "-skip_gameinfo",
            "-artwork_crop",
            "-sound", "none",
            "-console",
            "-noautosave",
            "invaders",
            "-autoboot_delay", "1",
            "-autoboot_script", LUA_SCRIPT_PATH,
        ]
        if desactiver_video_son:
            command.extend(["-video", "none", "-sound", "none", "-nothrottle"])
            
        if record_inp:
            # Correction critique : MAME sur un autre disque (D:) a souvent du mal avec les chemins absolus (C:) pour le -record
            # On enregistre localement dans le dossier MAME_DIR
            filename = f"invaders_session_{datetime.now().strftime('%Y%m%d_%H%M%S')}.inp"
            command.extend(["-record", filename])
            print(f"{Fore.CYAN}📼 Enregistrement INP activé : {Fore.YELLOW}{filename}{Style.RESET_ALL} (Dans le dossier MAME)")

        startupinfo = subprocess.STARTUPINFO()
        startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
        startupinfo.wShowWindow = 1 if visible else 7 # 1=Visible, 7=Minimisé
        
        self.process = subprocess.Popen(command, cwd=MAME_DIR, startupinfo=startupinfo)
        time.sleep(5) # Attente init MAME

    def setup_keyboard(self, trainer, config, comm):
        """Configure les raccourcis clavier."""
        def on_key_press(event):
            def is_terminal_in_focus():
                hwnd = win32gui.GetForegroundWindow()
                title = win32gui.GetWindowText(hwnd).lower()
                # Plus flexible : VS Code, CMD, PowerShell ou le script lui-même
                return "code" in title or "powershell" in title or "cmd" in title or "invaders" in title or "python" in title

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
                        self.is_normal_speed = not self.is_normal_speed
                        new_speed = self.slow_speed if self.is_normal_speed else self.training_speed
                        self.pending_commands.append(f"execute throttle_rate({new_speed})")
                        print(f"\n[F5] Vitesse : {'Low (' + str(self.slow_speed) + ')' if self.is_normal_speed else f'Rapide ({self.training_speed})'}")
                    elif keyboard.is_pressed("f6"):
                        self.slow_speed = round(self.slow_speed + 0.1, 1)
                        print(f"\n[F6] Vitesse lente ajustée : {self.slow_speed}")
                        if self.is_normal_speed:
                            self.pending_commands.append(f"execute throttle_rate({self.slow_speed})")
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
                    elif keyboard.is_pressed("f10"):
                        trainer.epsilon = min(1.0, trainer.epsilon + 0.1)
                        print(f"\n[F10] Epsilon augmenté : {trainer.epsilon:.3f}")
                    elif keyboard.is_pressed("f11"):
                        trainer.epsilon = max(config.epsilon_end, trainer.epsilon - 0.1)
                        print(f"\n[F11] Epsilon diminué : {trainer.epsilon:.3f}")
                    elif keyboard.is_pressed("f12"):
                        self.debug_lua = not self.debug_lua
                        self.pending_commands.append(f"debug {'on' if self.debug_lua else 'off'}")
                        print(f"\n[F12] Debug Lua : {'ON' if self.debug_lua else 'OFF'}")

        keyboard.on_press(on_key_press)

    def run(self):
        """Boucle principale de l'application."""
        
        # 1. Configuration
        # RESUME = True # Reprise de l'entraînement (Charge invaders_best.pth)
        RESUME = True 
        N = 4 # Historique
        NB_DE_FRAMES_STEP = 3
        model_type = "cnn" # "mlp", "cnn", "dreamer"
        
        # Paramètres spécifiques au modèle
        flag_aliens = False
        flag_boucliers = False
        cnn_type = None
        factor_div_frame = None
        
        if model_type.lower() == "cnn":
            cnn_type = "deepmind" # Plus rapide et standard pour Atari
            cnn_type = "precise" # Architecture demandée, plus lourde mais potentiellement plus fine
            full_frame_2D = (192, 200) # Taille ajustée pour voir la soucoupe sans le score (192x200)
            factor_div_frame = 2
            input_size = (N,) + tuple(x // factor_div_frame for x in full_frame_2D)
            NB_DE_DEMANDES_PAR_STEP = str(2 + 3 + 5) # 2(state: img+aliens) + 3(action) + 5(score/status/lives)
            TRAIN_EVERY_N_GLOBAL_STEPS = 8 # Poussé à 8 pour accélérer la boucle (réduit les pauses PyTorch)
        elif model_type.lower() == "mlp":
            flag_aliens = True
            flag_boucliers = False
            input_size = 11 + 2 + (6 + 11) * flag_aliens + 4 * flag_boucliers + 2
            # Calcul des messages par step pour la synchro Lua `wait_for`:
            # - get_state_mlp: 15 (reads) + 1 (range_read) = 16
            # - execute_action: 3
            # - get_score_and_status: 5
            NB_DE_DEMANDES_PAR_STEP = str(16 + 3 + 5) # Total 24
            TRAIN_EVERY_N_GLOBAL_STEPS = 4 # Entraîner tous les 4 steps
        elif model_type.lower() == "dreamer":
            factor_div_frame = 1 # V3 VISION : On garde la résolution native HD pour voir les bombes
            full_frame_2D = (192, 192)
            input_size = (N, 192, 192)
            NB_DE_DEMANDES_PAR_STEP = str(2 + 3 + 5) # 2(state: img+aliens) + 3(action) + 5(score/status/lives)
            # On augmente drastiquement la fréquence d'entraînement pour rentabiliser l'attente du GPU
            TRAIN_EVERY_N_GLOBAL_STEPS = 4 
        else:
            raise ValueError(f"Modèle {model_type} non supporté.")

        # Rewards
        reward_clipping_deepmind = False  # Ne PAS clipper avec des rewards custom (sinon step=-1 == mort=-1)
        reward_aliens_mult = 1.0         # Alien 10pts -> +10.0 reward. (Doublé pour agressivité MAX)
        reward_kill = -20.0              # Réduit (-50 -> -20) pour moins de peur
        reward_alive = 0.0              # Réduit drastiquement pour éviter le camping passif.
        reward_mult_step = -0.01        # Légère pression temporelle (comme Pacman)
        mult_reward_state = 0.01         # ACTIVÉ (Très léger) : Force l'IA à nettoyer vite sans encourager le suicide.
        reward_end_of_game = -100.0      # Game Over inacceptable (-10 -> -100)
        reward_fire_penalty = 0.0        # DÉSACTIVÉ (Laisser tirer à volonté pour débloquer l'agressivité)
        
        # Exploration
        use_noisy = True                # ACTIVÉ (Comme Pacman Elite)
        epsilon_start = 0.6 if RESUME else 1.0
        epsilon_end = 0.02               # On garde 2% d'aléatoire pour ne pas figer totalement
        target_steps_for_epsilon_end = 3_000_000 # Exploration étendue (3M steps) pour viser le "Max Learning"
        epsilon_linear = (epsilon_start - epsilon_end) / target_steps_for_epsilon_end
        epsilon_decay = 0
        epsilon_add = ((epsilon_start - epsilon_end) / target_steps_for_epsilon_end) * NB_DE_FRAMES_STEP * 100 if not use_noisy else 0.0

        # Config Object
        config = TrainingConfig(
            state_history_size=N,
            input_size=input_size,
            hidden_layers=1, # NA en cnn utile uniquement en mlp...
            hidden_size=1024, # Aligné sur Ape-X Multi Session 40
            output_size=6,
            learning_rate=0.0001, # Retour à 1e-4 pour apprendre plus vite au début
            gamma=0.999, # Aligné sur Ape-X Multi
            use_noisy=use_noisy,
            rainbow_eval=250_000,
            rainbow_eval_pourcent=2,  # 5% du temps en évaluation (standard Rainbow)
            epsilon_start=epsilon_start,
            epsilon_end=epsilon_end,
            epsilon_linear=epsilon_linear,
            epsilon_decay=epsilon_decay,
            epsilon_add=epsilon_add,
            buffer_capacity=100_000, 
            batch_size=32 if model_type.lower() == "dreamer" else 256, # Aligné sur Ape-X Multi
            min_history_size=20000,  # Démarrage plus rapide (comme Pacman)
            prioritized_replay=True,
            target_update_freq=5000, # Aligné sur Ape-X Multi
            double_dqn=True,
            dueling=True,
            nstep=True,
            nstep_n=5, # Aligné sur Ape-X Multi
            model_type=model_type,
            cnn_type=cnn_type,
            mode="exploitation", # CHANGÉ ICI : Démarrage direct à 100% QI (sans test au hasard)
            optimize_memory=True # ✅ Invaders utilise [0,1], l'optimisation est sûre et économise la RAM
        )

        # Injection des hyperparamètres spécifiques à DreamerV2
        if model_type.lower() == "dreamer":
            config.latent_dim = 512
            config.rnn_hidden_dim = 256
            config.sequence_length = 16
            config.imagination_horizon = 10
            config.kl_loss_scale = 1.0
            config.reward_loss_scale = 1.0
            config.reconstruction_loss_scale = 10.0 # Ajusté pour la nouvelle perte BCE en haute résolution
            config.actor_loss_scale = 1.0
            config.value_loss_scale = 1.0
            config.action_entropy_scale = 0.01 # Pousse l'Acteur à l'exploration pour éviter la paralysie

        # Récupération ID expérience et création du tag complet pour les fichiers
        experiment_id = self.get_next_experiment_id()
        run_tag = f"{experiment_id}_{config.model_type}_N={N}_bs={config.batch_size}_lr={config.learning_rate}_g={config.gamma}"
        print(f"🆔 ID Session: {experiment_id} | Tag fichier: {run_tag}")

        # 2. Initialisation Système
        # Gestion OBS (Avant MAME pour décider de la visibilité)
        record = True # ✅ Active l'enregistrement OBS pour capturer vos exploits (si WebSocket OK)
        record_inp = False # ❌ Désactivé par défaut car cause des crashs MAME sur certains systèmes
        recorder = None
        if record:
            recorder = ScreenRecorder()
            if not recorder.ws:
                print(f"{Fore.YELLOW}⚠️ OBS non détecté. Tentative de lancement...{Style.RESET_ALL}")
                self.launch_obs()
                recorder = ScreenRecorder()
                if not recorder.ws:
                    print(f"{Fore.RED}❌ Echec connexion OBS. Enregistrement DÉSACTIVÉ.{Style.RESET_ALL}")
                    record = False
                    recorder = None

        self.kill_existing_mame() # Nettoyage préventif
        
        try:
            # 1. Démarrer le serveur (Bind) AVANT de lancer MAME pour réserver le port
            comm = MameCommunicator("127.0.0.1", 12345, deferred_accept=True)
            # 2. Lancer MAME
            # Pour voir MAME sans enregistrer, on force visible=True et desactiver_video_son=False
            self.launch_mame(desactiver_video_son=False, visible=True, record_inp=record_inp)
            # 3. Attendre que MAME se connecte
            comm.accept_connection()
        except Exception as e:
            print(f"{Fore.RED}❌ Impossible de démarrer le serveur. Arrêt de MAME.{Style.RESET_ALL}")
            if self.process: self.process.terminate()
            sys.exit(1)
            
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
        
        # Gestion Sauvegarde / Reprise / Archivage
        model_filename = os.path.join(SCRIPT_DIR, "invaders.pth")
        best_model_filename = os.path.join(SCRIPT_DIR, "invaders_best.pth")
        buffer_filename = os.path.join(SCRIPT_DIR, "invaders.buffer")

        if RESUME:
            if os.path.exists(best_model_filename):
                print(f"{Fore.CYAN}♻️ RÉCUPÉRATION : Chargement du modèle BEST ({best_model_filename}){Style.RESET_ALL}")
                trainer.load_model(best_model_filename)
            elif os.path.exists(model_filename):
                trainer.load_model(model_filename)
        else:
            # Archivage automatique
            if os.path.exists(model_filename) or os.path.exists(best_model_filename):
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                print(f"{Fore.YELLOW}⚠️ Archivage de la session précédente vers *_backup_{ts}...{Style.RESET_ALL}")
                if os.path.exists(model_filename): shutil.move(model_filename, f"{model_filename}_backup_{ts}")
                if os.path.exists(best_model_filename): shutil.move(best_model_filename, f"{best_model_filename}_backup_{ts}")
                if os.path.exists(buffer_filename): shutil.move(buffer_filename, f"{buffer_filename}_backup_{ts}")

        if RESUME and os.path.exists(buffer_filename):
            trainer.load_buffer(buffer_filename)

        self.setup_keyboard(trainer, config, comm)
        
        # Stats containers
        fenetre_moyenne = 100
        collection_score = deque(maxlen=fenetre_moyenne)
        list_mean_scores = []
        max_scores_hist = []
        list_eps_sigmas = [[], []]
        list_cumul_steps = []
        mean_score = mean_score_old = last_score = high_score = best_mean_score = 0
        nb_steps_total = 0

        # Init MAME params
        comm.communicate([
            f"write_memory {Memory.NUM_COINS}(1)",
            "execute P1_start(1)",
            f"execute throttle_rate({self.training_speed})",
            "execute throttled(0)",
            f"frame_per_step {NB_DE_FRAMES_STEP}",
        ])

        print(f"Démarrage entraînement... Mode: {config.model_type}")

        # --- ARCHITECTURE ASYNCHRONE (ACTOR-LEARNER) ---
        self.training_active = False
        self.stop_learner = False
        self.train_queue = 0
        
        def learner_thread_func():
            print(f"\n{Fore.MAGENTA}🧠 [Learner] Processus GPU Asynchrone Démarré en arrière-plan !{Style.RESET_ALL}")
            while not self.stop_learner:
                if self.training_active and self.train_queue > 0:
                    trainer.train_step()
                    self.train_queue -= 1
                else:
                    time.sleep(0.002) # Micro-pause pour relâcher le GIL
                    
        if config.model_type.lower() != "dreamer":
            learner_thread = threading.Thread(target=learner_thread_func, daemon=True)
            learner_thread.start()
        # -----------------------------------------------

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
            
            start_steps_time = time.time()
            
            current_loop_obs_stack = initial_obs_stack
            invaders_loop_frame_history = deque(maxlen=config.state_history_size)
            for f in initial_obs_stack: invaders_loop_frame_history.append(f)

            # Fonction locale pour gérer l'envoi synchronisé avec les commandes clavier en attente
            def send_sync_and_pending():
                cmds = []
                if self.pending_commands:
                    cmds = self.pending_commands[:] # Copie
                    self.pending_commands.clear()   # Vide la liste
                
                total_msgs = int(NB_DE_DEMANDES_PAR_STEP) + len(cmds)
                comm.communicate([f"wait_for {total_msgs}"])
                if cmds: comm.communicate(cmds)

            # --- Boucle de jeu (Step) ---
            while NotEndOfGame == 1:
                send_sync_and_pending()

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
                score, PlayerIsOK, NotEndOfGame, lives = game.get_score_and_status(_last_score)

                # 4. Reward
                if NotEndOfGame == 0:
                    # Game Over (Invasion ou fin de vies)
                    reward = reward_end_of_game
                    if PlayerIsOK == 1: # Le joueur est vivant mais le jeu coupe -> INVASION !
                         reward -= 500.0 # PÉNALITÉ MASSIVE (Pire que tout)
                    done = True
                elif PlayerIsOK == 1:
                    reward = reward_alive
                    done = False
                elif lives > 0: # Si on a encore des vies en réserve (mémoire 21FF)
                    reward = reward_kill
                    done = False
                else:
                    reward = reward_kill + reward_end_of_game
                    done = True

                # Ajout pénalité de tir (si pas mort/fini)
                if not done and action in [1, 3, 5]: 
                    reward += reward_fire_penalty

                reward += ((score - _last_score) * reward_aliens_mult) + (reward_mult_step) + reward_state_comp
                if reward_clipping_deepmind: reward = np.sign(reward)
                
                sum_rewards += reward

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
                        # Pour MLP, l'état est un vecteur 1D. On concatène l'historique de N vecteurs.
                        # current_loop_obs_stack est l'état t, next_obs_stack est l'état t+1.
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

                # --- SYNCHRONISATION ASYNCHRONE ---
                if config.model_type.lower() == "dreamer":
                    if nb_steps_total % TRAIN_EVERY_N_GLOBAL_STEPS == 0:
                        trainer.train_step()
                else:
                    # Signale au Thread Learner qu'il peut travailler en continu
                    if not self.training_active and hasattr(trainer, 'replay_buffer') and trainer.replay_buffer.size >= config.min_history_size:
                        self.training_active = True
                        print(f"\n{Fore.GREEN}✅ [Actor] Buffer prêt ({config.min_history_size} éléments). Début de l'entraînement GPU asynchrone régulé !{Style.RESET_ALL}")
                    
                    if self.training_active and nb_steps_total % TRAIN_EVERY_N_GLOBAL_STEPS == 0:
                        self.train_queue += 1
                # ----------------------------------
                
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
                if self.flag_F8_pressed:
                    if config.model_type == "mlp":
                        Visualizer.afficher_get_state(game)
                    else:
                        Visualizer.afficher_frame(next_frame, factor_div_frame)
                    self.flag_F8_pressed = False

                # 8. Mort / Fin vie
                if PlayerIsOK == 0:
                    comm.communicate(["wait_for 5"])  # 5 = nb de messages dans get_score_and_status
                    while PlayerIsOK == 0 and NotEndOfGame == 1:
                        score, PlayerIsOK, NotEndOfGame, lives = game.get_score_and_status(score)
                    if PlayerIsOK == 1:
                        send_sync_and_pending()
                    if NotEndOfGame == 0 and config.model_type.lower() == "dreamer":
                        trainer.store_transition(current_loop_obs_stack, action, reward, next_obs_stack, True)

            # --- Fin Episode ---
            comm.communicate(["wait_for 1"])
            #comm.communicate([f'draw_text(25,1,"Game number: {episode+1:04d} - mean score={mean_score:04.0f} - ba(c)o 2026")'])
            if record:
                if score > last_score:
                    time.sleep(0.5) # Attente pour capturer la fin de partie uniquement si record 
                                    # (marche pas car en fait il faudrait que mame soit libéré wait_for 0?)
                video_path = recorder.stop_recording()

                if video_path and os.path.exists(video_path):
                    if score > last_score:
                        # Supprimer l'ancienne meilleure vidéo si elle existe
                        if last_score != 0:
                            old_video = os.path.join(MEDIA_DIR, f"invaders_best_game_{last_score}.mp4")
                            if os.path.exists(old_video):
                                try:
                                    os.remove(old_video)
                                except OSError as e:
                                    print(f"Erreur suppression ancienne vidéo : {e}")
                        
                        # Supprimer l'ancien meilleur GIF et AVI crop si ils existent
                        if last_score != 0:
                            old_gif = os.path.join(MEDIA_DIR, f"invaders_best_game_{last_score}.gif")
                            if os.path.exists(old_gif):
                                try:
                                    os.remove(old_gif)
                                except OSError as e:
                                    print(f"Erreur suppression ancien GIF : {e}")
                            
                            old_crop = os.path.join(MEDIA_DIR, f"invaders_best_game_{last_score}_crop.mp4")
                            if os.path.exists(old_crop):
                                try:
                                    os.remove(old_crop)
                                except OSError as e:
                                    print(f"Erreur suppression ancien MP4 crop : {e}")

                        # Copier la nouvelle meilleure vidéo
                        try:
                            dst_mp4 = os.path.join(MEDIA_DIR, f"invaders_best_game_{score}.mp4")
                            shutil.copy(video_path, dst_mp4)
                            print(f"📼 Nouvelle meilleure partie enregistrée : {dst_mp4}")
                            
                            # Génération du GIF
                            self.generate_best_gif(video_path, score)
                            
                            # Tentative de suppression du fichier original avec re-essais
                            deleted = False
                            for attempt in range(5): # 5 tentatives
                                try:
                                    os.remove(video_path)
                                    deleted = True
                                    break
                                except PermissionError:
                                    print(f"  (Tentative {attempt+1}/5) Fichier vidéo verrouillé par OBS, attente...")
                                    time.sleep(0.5) # Attendre 500ms
                                except Exception as e:
                                    print(f"Erreur inattendue lors de la suppression de {video_path}: {e}")
                                    break # Sortir en cas d'autre erreur
                            if not deleted:
                                print(f"{Fore.YELLOW}⚠️ Impossible de supprimer le fichier vidéo original {video_path}. Il est peut-être encore utilisé.{Style.RESET_ALL}")
                        except (shutil.Error, OSError) as e:
                            print(f"Erreur copie/suppression vidéo : {e}")

                        last_score = score
                    else:
                        # Suppression de la vidéo si ce n'est pas un record pour économiser l'espace
                        try:
                            os.remove(video_path)
                        except Exception as e:
                            print(f"⚠️ Erreur suppression vidéo non-record : {e}")
                elif video_path: # Le chemin a été retourné mais le fichier n'existe pas
                    print(f"{Fore.YELLOW}⚠️ OBS a retourné un chemin ({video_path}) mais le fichier est introuvable.{Style.RESET_ALL}")

            # Stats & Graphiques
            if self.flag_create_fig:
                reward_str = f"R_alive={reward_alive}, R_step={reward_mult_step}, R_kill={reward_kill}"
                Visualizer.create_fig(trainer, NB_DE_FRAMES_STEP, episode, list_mean_scores, fenetre_moyenne,
                                      list_eps_sigmas, high_score, flag_aliens, flag_boucliers, list_cumul_steps, reward_str, os.path.join(MEDIA_DIR, f"Invaders_fig_{run_tag}_ask"), max_scores=max_scores_hist)
                self.flag_create_fig = False

            # Logging Sigmas/Epsilon
            if config.model_type.lower() == "dreamer":
                list_eps_sigmas[0].append(0.0)
                list_eps_sigmas[1].append(0.0)
                avg_sigma = min_sigma = max_sigma = 0.0
            elif use_noisy:
                sigmas = trainer.dqn.get_sigma_values()
                s1 = sigmas.get("fc1", 0.0) if config.model_type == "cnn" else sigmas.get("hidden_modules.0.0", 0.0)
                s2 = sigmas.get("advantage_head", 0.0) if config.dueling else sigmas.get("output_layer", 0.0)
                list_eps_sigmas[0].append(s1)
                list_eps_sigmas[1].append(s2)
                
                # Calculs pour affichage console
                avg_sigma = sum(sigmas.values()) / len(sigmas) if sigmas else 0
                min_sigma = min(sigmas.values()) if sigmas else 0
                max_sigma = max(sigmas.values()) if sigmas else 0
            else:
                list_eps_sigmas[0].append(trainer.epsilon)

            list_cumul_steps.append(nb_steps_total)
            collection_score.append(score)
            mean_score_old = mean_score
            mean_score = round(sum(collection_score) / len(collection_score), 2)
            list_mean_scores.append(mean_score)
            max_scores_hist.append(max(collection_score) if collection_score else 0)
            
            # Mise à jour intelligente du Learning Rate si plateau détecté
            # On attend que la fenêtre soit pleine pour éviter le "Pic de démarrage" (ex: Ep1=800 -> Mean=800 -> Record faussé)
            if len(collection_score) >= fenetre_moyenne:
                if hasattr(trainer, 'update_learning_rate'):
                    trainer.update_learning_rate(mean_score)
            
            if score > high_score: high_score = score
            
            # Sauvegarde Best Model
            if episode >= fenetre_moyenne and mean_score > (best_mean_score + 5):
                best_mean_score = mean_score
                trainer.save_model(os.path.join(SCRIPT_DIR, f"invaders_best_{run_tag}.pth"))
                trainer.save_model(best_model_filename) # Copie générique pour faciliter le resume
                print(f"{Fore.YELLOW}🏆 Record moyen ({best_mean_score}) ! Sauvegarde best.{Style.RESET_ALL}")
                try: pygame.mixer.Sound("c:\\Windows\\Media\\tada.wav").play()
                except: pass

            # Update Epsilon (si pas noisy)
            if not (config.model_type.lower() == "dreamer" or config.use_noisy):
                if mean_score_old > mean_score:
                    trainer.epsilon += config.epsilon_add if trainer.epsilon < config.epsilon_start else 0.0
                if trainer.epsilon < 0.001 and trainer.config.mode == "exploration":
                    trainer.config.mode = "exploitation"
                    if hasattr(trainer, 'set_mode'):
                        trainer.set_mode("exploitation")
                    print("===> Passage auto en EXPLOITATION")

            # Console Log
            _d = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            if config.use_noisy:
                exploration_str = f"[sigma avg={avg_sigma:.4f} min={min_sigma:.4f} max={max_sigma:.4f}]"
            else:
                exploration_str = f"[ε={trainer.epsilon:.4f}]"
            exploration_str = exploration_str if config.mode != "exploitation" else "[*** Mode EXPLOITATION ***]"
            exploration_str = exploration_str if config.model_type != "dreamer" else "[Mode DREAMER.V2]"

            if config.model_type.lower() == "dreamer":
                current_lr = trainer.actor_optim.param_groups[0]['lr']
                buffer_str = f"[Buffer={len(trainer.buffer)/1000:.0f}k/{trainer.config.buffer_capacity/1000:.0f}k]"
            else:
                current_lr = trainer.optimizer.param_groups[0]['lr']
                buffer_str = f"[Buffer={trainer.replay_buffer.size/1000:.0f}k/{trainer.config.buffer_capacity/1000:.0f}k]" if hasattr(trainer, "replay_buffer") else ""

            print(
                f"N°{episode+1} [{_d}][steps_ep,all={step:4d},{str(nb_steps_total//1000)+'k':>5}]"
                + buffer_str
                + exploration_str + f"[LR={current_lr:.1e}][rewards={sum_rewards:5.0f}]" + f"[score={score:3d}][score moyen={mean_score:3.0f}]"
            )

            # Sauvegarde périodique
            if (episode + 1) % 10 == 0:
                trainer.save_model(os.path.join(SCRIPT_DIR, f"invaders_{run_tag}.pth"))
                trainer.save_model(model_filename) # Copie générique pour faciliter le resume
                reward_str = f"R_alive={reward_alive}, R_step={reward_mult_step}, R_kill={reward_kill}"
                Visualizer.create_fig(trainer, NB_DE_FRAMES_STEP, episode, list_mean_scores, fenetre_moyenne,
                                      list_eps_sigmas, high_score, flag_aliens, flag_boucliers, list_cumul_steps, reward_str, os.path.join(MEDIA_DIR, f"Invaders_fig_{run_tag}"), max_scores=max_scores_hist)


            # Sauvegarde buffer périodique (sécurité anti-crash)
            if (episode + 1) % 100 == 0:
                trainer.save_buffer(buffer_filename)


        # Fin du programme
        self.stop_learner = True
        
        trainer.save_model(os.path.join(SCRIPT_DIR, f"invaders_{run_tag}_final.pth"))
        trainer.save_buffer(buffer_filename)
        
        # Réduction de la limite à 50 pour sauvegarder même si on arrête le processus en cours
        if episode >= 500:
            self.log_results(experiment_id, episode, list_mean_scores, config, NB_DE_DEMANDES_PAR_STEP, NB_DE_FRAMES_STEP, reward_kill, reward_mult_step)

        if record:
            recorder.stop_recording()
            recorder.ws.disconnect()
        
        self.process.terminate()

if __name__ == "__main__":
    app = InvadersApp()
    app.run()