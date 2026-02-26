"""
pacman_ai.py

Version refactorisée de pacman.py pour l'entraînement d'une IA sur Pac-Man via MAME.
Compatible avec AI_Mame.py et structurée comme invaders.py.

Code: http://cubeman.org/arcade-source/pacman.asm
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

# ==================================================================================================
# CONSTANTES & ADRESSES MÉMOIRE
# ==================================================================================================

class Memory:
    """
    Adresses mémoire du jeu Pac-Man (Midway/Namco Hardware).
    Sources et documentation :
    - http://cubeman.org/arcade-source/pacman.asm (Pac-Man Assembly Source)
    - https://github.com/mamedev/mame/blob/master/src/mame/namco/pacman_m.cpp
    """
    SCORE_10        = "4E80"  # Score : Chiffre des Dizaines (0-9)
    SCORE_100       = "4E81"  # Score : Chiffre des Centaines (0-9)
    SCORE_1000      = "4E82"  # Score : Chiffre des Milliers (0-9)
    SCORE_10000     = "4E83"  # Score : Chiffre des Dizaines de milliers (0-9)
    CREDITS         = "4E6E"  # Nombre de crédits (pièces insérées)
    PILLS_COUNT     = "4E0E"  # Compteur de pilules (utilisé pour les seuils de vitesse/fantômes)
    LIVES           = "4E14"  # Nombre de vies restantes
    PLAYER_ALIVE    = "4EAE"  # Flag statut joueur (00 = Mort/Animation, >00 = Normal)
    
    # Positions
    # Note : Le système de coordonnées est basé sur l'écran vertical (rotation 90°).
    # 4Dxx : Coordonnées LOGIQUES (Game Logic) utilisées par le CPU pour les collisions/IA.
    # 4FFx : Coordonnées MATÉRIELLES (Sprite RAM) copiées depuis 4Dxx pour l'affichage.
    # On utilise 4Dxx pour avoir la position "réelle" gérée par le moteur du jeu.
    BLINKY_X        = "4D00"  # Blinky X (Logic)
    BLINKY_Y        = "4D01"  # Blinky Y (Logic)
    PINKY_X         = "4D02"  # Pinky X
    PINKY_Y         = "4D03"
    INKY_X          = "4D04"  # Inky X
    INKY_Y          = "4D05"
    CLYDE_X         = "4D06"  # Clyde X
    CLYDE_Y         = "4D07"
    PACMAN_X        = "4D08"  # Pac-Man X (Logic)
    PACMAN_Y        = "4D09"  # Pac-Man Y (Logic)
    
    # États des fantômes (Working RAM)
    GHOST_STATE_BLINKY = "4DA7"
    GHOST_STATE_PINKY  = "4DA8"
    GHOST_STATE_INKY   = "4DA9"
    GHOST_STATE_CLYDE  = "4DAA"
    
    # Video RAM (Tilemap)
    # La mémoire vidéo commence à 0x4000.
    # L'écran est une grille de 28 colonnes x 36 lignes (total 1008 tuiles, mappé sur 1024 octets).
    VRAM_START      = "4000"  # Début de la Video RAM
    VRAM_LEN        = 1024    # Longueur (0x4000 - 0x43FF)

class GameConstants:
    ACTIONS = {
        0: "left", 
        1: "right", 
        2: "up", 
        3: "down"
    }
    VRAM_SIZE = 1024
    NUM_POSITIONS = 14 # Pacman (2) + 4 Ghosts (2*4) + 4 States

# ==================================================================================================
# GESTION DE L'INTERFACE DE JEU (MAME)
# ==================================================================================================

class PacmanInterface:
    """Gère la communication avec MAME pour Pac-Man."""
    
    def __init__(self, communicator: MameCommunicator):
        self.comm = communicator
        self.debug = 0

    def execute_action(self, action_code: int):
        """Envoie les commandes Lua pour exécuter une action."""
        act = GameConstants.ACTIONS[action_code]
        
        left  = 1 if act == "left" else 0
        right = 1 if act == "right" else 0
        up    = 1 if act == "up" else 0
        down  = 1 if act == "down" else 0
        
        if self.debug >= 3:
            print(f"[DEBUG] Action: {act} (L={left}, R={right}, U={up}, D={down})")
        
        self.comm.communicate([
            f"execute P1_Left({left})",
            f"execute P1_Right({right})",
            f"execute P1_Up({up})",
            f"execute P1_Down({down})",
        ])

    def get_state_mlp(self):
        """
        Récupère l'état sous forme de vecteur plat (VRAM + Positions).
        La VRAM (1024 octets) est essentielle car elle contient la structure du labyrinthe (murs)
        et l'emplacement des pastilles restantes, permettant au MLP de "voir" l'environnement.
        
        INFO MLP vs CNN :
        - Ici, l'IA reçoit les valeurs brutes et exactes (précision pixel).
        - MAIS elle perd la notion de 2D (elle ne sait pas qu'une case est voisine d'une autre).
        - C'est plus robuste aux erreurs de code (pas de reconstruction), mais plus lent à apprendre la géométrie.
        """
        # Lecture VRAM
        response_vram = self.comm.communicate([f"read_memory_range {Memory.VRAM_START}({Memory.VRAM_LEN})"])
        if not response_vram or not response_vram[0]:
            video_data = np.zeros(GameConstants.VRAM_SIZE, dtype=np.float32)
        else:
            try:
                raw_list = list(map(int, response_vram[0].split(",")))
                # Ajustement taille fixe 1024 pour éviter les erreurs de shape
                if len(raw_list) > GameConstants.VRAM_SIZE: raw_list = raw_list[:GameConstants.VRAM_SIZE]
                elif len(raw_list) < GameConstants.VRAM_SIZE: raw_list += [0] * (GameConstants.VRAM_SIZE - len(raw_list))
                
                raw_data = np.array(raw_list, dtype=np.int16)
                
                # Mapping sémantique pour MLP (Contraste fort)
                video_data = np.full(GameConstants.VRAM_SIZE, -1.0, dtype=np.float32) # Murs/Inaccessible = -1.0
                video_data[raw_data == 0x40] = 0.0  # Vide (Chemin) = 0.0
                video_data[raw_data == 0x10] = 1.0  # Pastille = 1.0
                video_data[raw_data == 0x14] = 1.0  # Power Pellet = 1.0
            except ValueError:
                video_data = np.zeros(GameConstants.VRAM_SIZE, dtype=np.float32)

        # Lecture Positions
        response_pos = self.comm.communicate([
            f"read_memory {Memory.PACMAN_X}", f"read_memory {Memory.PACMAN_Y}",
            f"read_memory {Memory.BLINKY_X}", f"read_memory {Memory.BLINKY_Y}",
            f"read_memory {Memory.PINKY_X}",  f"read_memory {Memory.PINKY_Y}",
            f"read_memory {Memory.INKY_X}",   f"read_memory {Memory.INKY_Y}",
            f"read_memory {Memory.CLYDE_X}",  f"read_memory {Memory.CLYDE_Y}",
            f"read_memory {Memory.GHOST_STATE_BLINKY}",
            f"read_memory {Memory.GHOST_STATE_PINKY}",
            f"read_memory {Memory.GHOST_STATE_INKY}",
            f"read_memory {Memory.GHOST_STATE_CLYDE}",
        ])
        
        if not response_pos or len(response_pos) < 14:
            positions = np.zeros(GameConstants.NUM_POSITIONS, dtype=np.float32)
        else:
            # Traitement séparé : Coordonnées (0-255) et États (0-4)
            coords = np.array(list(map(int, response_pos[:10])), dtype=np.float32) / 255.0
            
            # Traitement des états pour le MLP :
            # 0 (Normal) -> -1.0 (Danger)
            # 1 (Blue) ou 2 (Flash) -> 1.0 (Mangeable)
            # 4 (Eyes) -> 0.0 (Neutre)
            states = []
            for s_val in map(int, response_pos[10:]):
                if s_val == 0: states.append(-1.0)      # Danger
                elif s_val in [1, 2]: states.append(1.0) # Mangeable
                else: states.append(0.0)                # Yeux/Autre
            
            states_np = np.array(states, dtype=np.float32)
            positions = np.concatenate([coords, states_np])

        return np.concatenate([video_data, positions])

    def get_state_cnn(self):
        """
        Construit une image 64x64 représentant l'état du jeu.
        - Upscaling VRAM 2x (32x32 -> 64x64) : Suffisant et beaucoup plus rapide.
        - Positionnement précis des sprites.
        - Optimisé pour l'architecture 'deepmind'.
        
        Note sur les coordonnées :
        L'image VRAM est brute (Col, Row).
        Les coordonnées des sprites sont mappées sur cette grille.
        La rotation pour l'affichage est faite dans Visualizer.
        
        INFO CNN vs MLP :
        - L'IA "voit" les murs et la proximité grâce aux convolutions (vision spatiale).
        - Elle apprend plus vite les stratégies (coins, couloirs).
        - CRITIQUE : Si cette fonction dessine les sprites avec un décalage (mauvais offset), l'IA sera aveugle/suicidaire.
        """
        # 1. Lecture VRAM (Fond + Pastilles)
        response_vram = self.comm.communicate([f"read_memory_range {Memory.VRAM_START}({Memory.VRAM_LEN})"])
        if not response_vram or not response_vram[0]:
            grid = np.zeros((64, 64), dtype=np.float32)
        else:
            try:
                # VRAM est 1024 octets (32x32 tuiles).
                raw_list = list(map(int, response_vram[0].split(",")))
                
                # --- CORRECTION VRAM (Column-Major -> Row-Major) ---
                # La VRAM Namco est stockée colonne par colonne.
                # On doit la reconstruire en grille (Row, Col) correcte.
                # MURS = 0.2 (Normalisé 0-1 pour optimize_memory)
                grid_small = np.full((32, 28), 0.2, dtype=np.float32)
                
                for idx, val in enumerate(raw_list):
                    if idx >= 1024: break
                    # Mapping Hardware Namco :
                    # Index = Col * 32 + Row
                    # Les colonnes visibles sont ~2 à 29. On mappe vers 0-31.
                    r = idx % 32
                    # CORRECTION MAJEURE : Alignement Gauche->Droite (Normal)
                    # On décale de -2 pour centrer les colonnes 2-29 sur 0-27
                    c = (idx // 32) - 2
                    
                    if 0 <= r < 32 and 0 <= c < 28:
                        if val == 0x40: grid_small[r, c] = 0.4 # Vide (Gris moyen)
                        elif val == 0x10: grid_small[r, c] = 0.6 # Pastille (Clair)
                        elif val == 0x14: grid_small[r, c] = 0.9 # Power (Très clair)
                
                # --- CONTRASTE AMÉLIORÉ ---
                # Upscaling 32x32 -> 64x64 (Facteur 2)
                grid = np.kron(grid_small, np.ones((2, 2)))
            except ValueError as e:
                if self.debug >= 1: print(f"[ERROR] VRAM reshape error: {e}")
                grid = np.zeros((64, 56), dtype=np.float32)

        # 2. Lecture Positions (Sprites) pour les superposer
        response_pos = self.comm.communicate([
            f"read_memory {Memory.PACMAN_X}", f"read_memory {Memory.PACMAN_Y}",
            f"read_memory {Memory.BLINKY_X}", f"read_memory {Memory.BLINKY_Y}",
            f"read_memory {Memory.PINKY_X}",  f"read_memory {Memory.PINKY_Y}",
            f"read_memory {Memory.INKY_X}",   f"read_memory {Memory.INKY_Y}",
            f"read_memory {Memory.CLYDE_X}",  f"read_memory {Memory.CLYDE_Y}",
            f"read_memory {Memory.GHOST_STATE_BLINKY}",
            f"read_memory {Memory.GHOST_STATE_PINKY}",
            f"read_memory {Memory.GHOST_STATE_INKY}",
            f"read_memory {Memory.GHOST_STATE_CLYDE}",
        ])
        
        if response_pos and len(response_pos) >= 14:
            try:
                # Correction coordonnées (Hardware Pac-Man):
                # px (4C00) = Position Verticale (Scanline). MAME: sy = 240 - px.
                # py (4C01) = Position Horizontale. MAME: sx = py.
                # Formules calibrées pour grille 64x64 (Upscale 2x de 32x32 tiles)
                
                # AJUSTEMENT OFFSETS (Important pour l'alignement visuel F8)
                # Offset Y = -16 pixels (-2 tuiles) car on a coupé les 2 premières colonnes de la VRAM
                offset_x, offset_y = -4, -16 
                
                # Pacman (Index 0,1)
                px, py = int(response_pos[0]), int(response_pos[1])
                
                # Mapping Coordonnées Sprites -> Grille 64x64 (Centrage)
                # Les sprites font 16x16 (4x4 sur la grille). On ajoute +2 pour centrer le blob 2x2.
                # Vertical (Row): (px + offset_x) // 4
                pr = min(63, max(0, (px + offset_x) // 4))
                # Horizontal (Col): Inversion Miroir (55 - ...) pour corriger la droite/gauche
                pc = min(55, max(0, 55 - ((py + offset_y) // 4)))
                
                if self.debug >= 2:
                    print(f"[CNN] Pacman Raw: ({px},{py}) -> Grid: ({pr},{pc})")
                
                # Dessiner un "blob" de 2x2 pour Pacman
                # Pacman = 0.7 (Soi-même, bien visible)
                # Correction indices : grid[row, col] -> grid[pr, pc]
                grid[max(0, pr):min(64, pr+2), max(0, pc):min(56, pc+2)] = 0.7 

                # Fantômes (Index 2-9)
                for i in range(4):
                    fx, fy = int(response_pos[2 + i*2]), int(response_pos[3 + i*2])
                    state = int(response_pos[10 + i])
                    
                    # Détermination de la valeur sur la grille selon l'état
                    val = 0.0 # DANGER MORTEL (0.0 = Noir/Trou, distinct des murs 0.2)
                    
                    if state == 1 or state == 2: # Blue ou Flash
                        val = 1.0 # CIBLE PRIORITAIRE (Max Reward)
                    elif state >= 3: # Eaten (3) ou Eyes (4+)
                        val = 0.4 # Neutre (Invisible = Vide)

                    fr = min(63, max(0, (fx + offset_x) // 4))
                    fc = min(55, max(0, 55 - ((fy + offset_y) // 4)))
                    
                    # Dessiner un "blob" de 2x2 pour chaque fantôme
                    grid[max(0, fr):min(64, fr+2), max(0, fc):min(56, fc+2)] = val
                    
            except: pass

        # La grille est déjà orientée
        grid = np.ascontiguousarray(grid)

        return grid, 0.0

    def get_score_and_lives(self):
        """Récupère score, vies et statut."""
        response = self.comm.communicate([
            f"read_memory {Memory.SCORE_10}",
            f"read_memory {Memory.SCORE_100}",
            f"read_memory {Memory.SCORE_1000}",
            f"read_memory {Memory.LIVES}",
            f"read_memory {Memory.PLAYER_ALIVE}",
            f"read_memory {Memory.PILLS_COUNT}"
        ])
        
        if self.debug >= 3:
            print(f"[DEBUG] get_score_and_lives raw: {response}")
        
        if not response or len(response) < 6:
            return 0, 0, 0, 0

        try:
            # Logique originale demandée (BCD via hex string)
            dizaines, centaines, dizaines_de_milliers = map(int, response[:3])
            
            dizaines = int(hex(dizaines)[2:])
            centaines = int(hex(centaines)[2:])
            dizaines_de_milliers = int(hex(dizaines_de_milliers)[2:])
            
            score = dizaines + centaines * 100 + dizaines_de_milliers * 10000
            
            lives = int(response[3])
            alive_flag = int(response[4])
            pills = int(response[5])
            
            if self.debug >= 2:
                print(f"[DEBUG] Score: {score}, Lives: {lives}, Alive: {alive_flag}, Pills: {pills}")
            
            return score, lives, alive_flag, pills
        except Exception as e:
            if self.debug >= 1:
                print(f"[DEBUG] Error parsing score/lives: {e}")
            return 0, 0, 0, 0

class StateExtractor:
    """Helper pour extraire l'état selon le modèle."""
    def __init__(self, interface: PacmanInterface, model_type):
        self.interface = interface
        self.model_type = model_type

    def __call__(self):
        if self.model_type == "cnn":
            return self.interface.get_state_cnn()
        else:
            return self.interface.get_state_mlp(), 0.0

# ==================================================================================================
# VISUALISATION
# ==================================================================================================

class Visualizer:
    @staticmethod
    def create_fig(nb_parties, scores_moyens, fenetre, epsilons, rewards, nb_steps, filename="Pacman_fig", high_score=0, label_curve="Epsilon", max_scores=None):
        matplotlib.use("Agg")
        plt.close("all")
        fig, (ax1, ax3) = plt.subplots(2, 1, figsize=(12, 10), gridspec_kw={'height_ratios': [3, 1]})
        
        ax1.set_xlabel("Episodes")
        ax1.set_ylabel(label_curve, color="tab:blue")
        ax1.plot(epsilons, color="tab:blue", linestyle="--", label=label_curve)
        ax1.tick_params(axis='y', labelcolor="tab:blue")
        ax1.legend(loc="upper left")
        
        ax2 = ax1.twinx()
        ax2.set_ylabel("Score (Moyen & Max)", color="tab:red")
        ax2.plot(scores_moyens, color="tab:red", label="Score Moyen")
        
        # Ajout courbe Max Score (Rolling Max) pour voir le potentiel max
        if max_scores and len(max_scores) == len(scores_moyens):
            ax2.plot(max_scores, color="tab:green", alpha=0.3, linestyle="-", linewidth=1, label="Max Score (fenêtre)")
            
        ax2.tick_params(axis='y', labelcolor="tab:red")
        ax2.legend(loc="upper right")
        
        # Ajout des lignes de tendance (comme invaders.py)
        if len(scores_moyens) > 1:
            x_all = np.arange(len(scores_moyens))
            z = np.polyfit(x_all, scores_moyens, 1)
            p = np.poly1d(z)
            ax2.plot(x_all, p(x_all), "tab:orange", alpha=0.6)           
            pente_globale = z[0]
            # Affichage pente globale
            angle = np.arctan(pente_globale) * 180 / np.pi
            midpoint_global = len(scores_moyens) // 2
            ax2.text(midpoint_global, p(midpoint_global), f"Pente Globale: {pente_globale:.2f}", color="tab:orange", fontweight="bold", ha="center", va="bottom")
            # Pente locale
            nb_parties_pente = 1000
            n = min(nb_parties_pente, len(scores_moyens))
            x_recent = np.arange(len(scores_moyens)-n, len(scores_moyens))
            z_recent = np.polyfit(x_recent, scores_moyens[-n:], 1)
            p_recent = np.poly1d(z_recent)
            ax2.plot(x_recent, p_recent(x_recent), "tab:purple", linestyle="--")
            pente_recent = z_recent[0]
            
            midpoint = len(scores_moyens) - n // 2
            ax2.text(midpoint, p_recent(midpoint), f"Pente: {pente_recent:.2f}", color="tab:purple", fontweight="bold")

        ax1.set_title(f"Pacman Training - {nb_parties} episodes - High Score: {high_score}")

        # --- Graphique 2 : Distribution Gaussienne ---
        if len(scores_moyens) > 1:
            data = scores_moyens
            mu = np.mean(data)
            sigma = np.std(data)
            
            if sigma > 0:
                min_val, max_val = min(data), max(data)
                # Bins de 50 points
                start_bin = np.floor(min_val / 50) * 50
                end_bin = np.ceil(max_val / 50) * 50
                bins = np.arange(start_bin, end_bin + 51, 50)
                
                ax3.hist(data, bins=bins, density=True, alpha=0.6, color='green', edgecolor='black', label='Histogramme')
                x_gauss = np.linspace(min_val, max_val, 100)
                y_gauss = (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x_gauss - mu) / sigma) ** 2)
                ax3.plot(x_gauss, y_gauss, 'r-', linewidth=2, label=f'Gaussienne (μ={mu:.1f}, σ={sigma:.1f})')
                ax3.set_xlabel('Score Moyen')
                ax3.set_ylabel('Densité')
                ax3.legend()
                ax3.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(f"{filename}.png")
        plt.close()

    @staticmethod
    def afficher_frame(frame):
        plt.figure(figsize=(6, 6))
        data_to_show = frame
        title = "Frame Pacman CNN Input"
        # Si MLP (vecteur plat), on extrait la partie VRAM pour l'afficher
        if frame.ndim == 1 and frame.shape[0] >= 1024:
             data_to_show = frame[:1024].reshape(32, 32)
             # Rotation 90° vers la droite pour orienter correctement (comme le jeu vertical)
             data_to_show = np.rot90(data_to_show, k=-1)
             title = "Frame Pacman (MLP VRAM 32x32)"
        elif frame.ndim == 2:
             # La grille est maintenant en ordre normal (Gauche->Droite), plus besoin de flip !
             data_to_show = frame
        plt.imshow(data_to_show, cmap="viridis", interpolation="nearest")
        plt.title(title)
        plt.colorbar()
        filename = f"pacman_frame_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        plt.savefig(filename, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Frame sauvegardée : {filename}")

# ==================================================================================================
# APPLICATION PRINCIPALE
# ==================================================================================================

class PacmanApp:
    def __init__(self):
        self.debug = 0
        self.flag_quit = False
        self.flag_create_fig = False
        self.flag_F8_pressed = False
        self.is_normal_speed = False
        self.training_speed = 20.0
        self.slow_speed = 0.2
        
        # Web Server
        self.web_server = GraphWebServer(graph_dir=".\\", host="0.0.0.0", port=5000)
        threading.Thread(target=self.web_server.start, daemon=True).start()
        
        pygame.mixer.init()

    def log_results(self, nb_episodes, mean_scores, config, nb_mess, nb_step_frame, r_kill, r_step):
        filename = "d:\\Emulateurs\\Mame Officiel\\plugins\\resultats_pacman.txt"
        
        # Trouver le prochain index
        last_idx = 0
        if os.path.exists(filename):
            try:
                with open(filename, "r", encoding="utf-8") as f:
                    for line in f:
                        if line.strip() and line[0].isdigit():
                            parts = line.split('[')
                            if parts and parts[0].strip().isdigit():
                                idx = int(parts[0])
                                if idx > last_idx: last_idx = idx
            except Exception as e:
                print(f"Erreur lecture resultats: {e}")
        next_idx = last_idx + 1

        # Stats
        start_mean = mean_scores[0] if mean_scores else 0
        end_mean = mean_scores[-1] if mean_scores else 0
        max_mean = max(mean_scores) if mean_scores else 0
        
        # Formatage
        input_str = str(config.input_size)
        hidden_str = f"{config.hidden_size}*{config.hidden_layers}"
        
        param_line1 = (f"{next_idx}[input={input_str}][hidden={hidden_str}][output={config.output_size}]"
                       f"[gamma={config.gamma}][learning={config.learning_rate}]"
                       f"[reward_kill={r_kill}][reward_step={r_step}]")
        
        decay_val = config.epsilon_linear if config.epsilon_linear > 0 else config.epsilon_decay
        decay_type = "linear" if config.epsilon_linear > 0 else "decay"
        
        param_line2 = (f"[epsilon start={config.epsilon_start} end={config.epsilon_end} {decay_type}={decay_val}]"
                       f"[Replay_size={config.buffer_capacity}&_batch={config.batch_size}]"
                       f"[nb_mess_frame={nb_mess}][nb_step_frame={nb_step_frame}][speed={self.training_speed}]")
        
        comment = (f"=> {nb_episodes} parties. Score moyen: début={start_mean:.2f}, fin={end_mean:.2f}, max={max_mean:.2f}. "
                   f"Model: {config.model_type}, Noisy: {config.use_noisy}, Double: {config.double_dqn}, Dueling: {config.dueling}, PER: {config.prioritized_replay}")

        try:
            with open(filename, "a", encoding="utf-8") as f:
                f.write(f"\n\n{param_line1}\n{param_line2}\n{comment}")
            print(f"{Fore.CYAN}📝 Résultats sauvegardés dans {filename}{Style.RESET_ALL}")
        except Exception as e:
            print(f"{Fore.RED}❌ Erreur écriture resultats: {e}{Style.RESET_ALL}")

    def launch_mame(self):
        command = [
            "D:\\Emulateurs\\Mame Officiel\\mame.exe",
            "-window", "-resolution", "448x576",
            "-skip_gameinfo",
            "-artwork_crop",
            "-sound", "none",
            "-console",
            "-noautosave",
            "pacman",
            "-autoboot_delay", "1",
            "-autoboot_script", "D:\\Emulateurs\\Mame Sets\\MAME EXTRAs\\plugins\\PythonBridgeSocket.lua",
        ]
        # Utilisation de startupinfo pour minimiser la fenêtre console si besoin
        startupinfo = subprocess.STARTUPINFO()
        startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
        startupinfo.wShowWindow = 7 # Minimisé
        
        self.process = subprocess.Popen(command, cwd="D:\\Emulateurs\\Mame Officiel", startupinfo=startupinfo, stderr=subprocess.DEVNULL)
        time.sleep(10)

    def setup_keyboard(self, trainer, config, comm):
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
                    elif keyboard.is_pressed("f3"):
                        self.debug = 0
                        print(f"\n[F3] Debug désactivé.")
                    elif keyboard.is_pressed("f4"):
                        self.debug = (self.debug + 1) % 4
                        print(f"\n[F4] Debug level: {self.debug}")
                    elif keyboard.is_pressed("f5"):
                        self.is_normal_speed = not self.is_normal_speed
                        new_speed = self.slow_speed if self.is_normal_speed else self.training_speed
                        comm.communicate([f"execute throttle_rate({new_speed})"])
                        print(f"\n[F5] Vitesse : {'Low (' + str(self.slow_speed) + ')' if self.is_normal_speed else f'Rapide ({self.training_speed})'}")
                    elif keyboard.is_pressed("f6"):
                        self.slow_speed = round(self.slow_speed + 0.1, 1)
                        print(f"\n[F6] Vitesse lente ajustée : {self.slow_speed}")
                        if self.is_normal_speed:
                            comm.communicate([f"execute throttle_rate({self.slow_speed})"])
                    elif keyboard.is_pressed("f10"):
                        trainer.epsilon = min(1.0, trainer.epsilon + 0.1)
                        print(f"\n[F10] Epsilon augmenté : {trainer.epsilon:.3f}")
                    elif keyboard.is_pressed("f11"):
                        trainer.epsilon = max(config.epsilon_end, trainer.epsilon - 0.1)
                        print(f"\n[F11] Epsilon diminué : {trainer.epsilon:.3f}")
                    elif keyboard.is_pressed("f7") and not self.flag_create_fig:
                        print("\n[F7] Création figure demandée.")
                        self.flag_create_fig = True
                    elif keyboard.is_pressed("f8"):
                        print("\n[F8] Affichage Frame demandé.")
                        self.flag_F8_pressed = True
                    elif keyboard.is_pressed("f9"):
                        new_mode = "exploitation" if config.mode == "exploration" else "exploration"
                        print(f"\n[F9] Mode changé vers : {new_mode}")
                        config.mode = new_mode
                        trainer.set_mode(new_mode)

        keyboard.on_press(on_key_press)

    def run(self):
        # Config
        RESUME = False # Nouvelle session "Elite" (Repart de zéro avec nouvelle architecture)
        model_type = "cnn" # "cnn" ou "mlp" - MLP recommandé pour Pacman (VRAM indices)
        N = 4 # Stack size (Standard Atari pour bien capter le mouvement)
        NB_DE_FRAMES_STEP = 4 # Plus réactif pour ne pas rater les virages (Pacman va vite)
        TRAIN_EVERY_N_GLOBAL_STEPS = 4 # Entraîner tous les 4 steps pour ne pas tuer le CPU/GPU
        # Calcul du nombre de messages par step pour la synchro wait_for
        # Action(4) + State(15: 1 VRAM + 14 Pos/States) + Score(6) = 25
        NB_DE_DEMANDES_PAR_STEP = 25
        
        model_filename = f"pacman_{model_type}.pth"
        best_model_filename = f"pacman_{model_type}_best.pth"
        
        if model_type == "cnn":
            # (N, H, W)
            input_size = (N, 64, 56)
        else:
            input_size = GameConstants.VRAM_SIZE + GameConstants.NUM_POSITIONS

        # Calcul automatique du decay pour l'exploration
        # Si RESUME, on repart à 0.2 (20%) pour redonner un peu de créativité, sinon 1.0
        epsilon_start = 0.2 if RESUME else 1.0 
        # Objectif final d'exploration
        epsilon_end = 0.02 # On descend plus bas (2%) pour la fin, pour maximiser le score
        
        # Utilisation d'une décroissance linéaire basée sur les steps (plus stable que par épisode)
        target_steps_for_epsilon_end = 1_000_000 # Plus lent (1M steps) pour mieux explorer sur la durée
        epsilon_linear = (epsilon_start - epsilon_end) / target_steps_for_epsilon_end
        epsilon_decay = 0.0
        
        print(f"Configuration Epsilon: Linear Decay ({epsilon_linear:.8f}/step) pour atteindre {epsilon_end} en {target_steps_for_epsilon_end} steps.")

        use_noisy = True # Activation NoisyNet (Rainbow) - ACTIVÉ pour exploration avancée

        config = TrainingConfig(
            state_history_size=N,
            input_size=input_size,
            hidden_layers=2,
            hidden_size=256, # Standard CNN
            output_size=4,
            learning_rate=0.0001, # Réduit pour plus de stabilité et éviter l'oubli catastrophique
            gamma=0.995, # Augmenté pour viser le long terme (nettoyer le niveau)
            use_noisy=use_noisy,
            buffer_capacity=200000, # Augmenté pour stabilité (Attention: ~6Go RAM requis)
            batch_size=64, # Standard pour PER
            min_history_size=20000, # Remplissage plus rapide
            cnn_type="precise", # Architecture optimisée pour la vitesse et l'efficacité
            model_type=model_type,
            epsilon_start=epsilon_start,
            epsilon_end=epsilon_end,
            epsilon_linear=epsilon_linear,
            epsilon_decay=epsilon_decay, 
            epsilon_add=0.0,
            double_dqn=True,
            dueling=True,
            nstep=True,       # Rainbow: N-Step Returns
            nstep_n=3,        # 3 steps
            prioritized_replay=True, # Rainbow: PER
            optimize_memory=True # ✅ ACTIVÉ : Maintenant sûr grâce à la normalisation [0,1]
        )

        print(f"Configuration Input Size: {input_size} (Model: {model_type})")
        self.launch_mame()
        comm = MameCommunicator("localhost", 12346) # Port défini dans Lua pour Pacman
        game = PacmanInterface(comm)
        config.state_extractor = StateExtractor(game, model_type)
        
        trainer = DQNTrainer(config)
        self.setup_keyboard(trainer, config, comm)
        
        # Recorder
        record = False # Mettre à True pour enregistrer avec OBS
        recorder = ScreenRecorder() if record else None
        
        # Load if exists
        if RESUME:
            # Priorité au chargement du BEST model pour récupérer d'une chute de performance
            if os.path.exists(best_model_filename):
                print(f"{Fore.CYAN}♻️ RÉCUPÉRATION : Chargement du modèle BEST ({best_model_filename}) pour annuler la chute libre.{Style.RESET_ALL}")
                trainer.load_model(best_model_filename)
            elif os.path.exists(model_filename):
                trainer.load_model(model_filename)
        else:
            # Sécurité : Archivage automatique de l'ancienne session si elle existe
            if os.path.exists(model_filename) or os.path.exists(best_model_filename):
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                print(f"{Fore.YELLOW}⚠️ Archivage de la session précédente vers *_backup_{ts}...{Style.RESET_ALL}")
                if os.path.exists(model_filename): shutil.move(model_filename, f"{model_filename}_backup_{ts}")
                if os.path.exists(best_model_filename): shutil.move(best_model_filename, f"{best_model_filename}_backup_{ts}")
                if os.path.exists("pacman.buffer"): shutil.move("pacman.buffer", f"pacman.buffer_backup_{ts}")

        if RESUME and os.path.exists("pacman.buffer"):
            trainer.load_buffer("pacman.buffer")

        # Stats
        scores = deque(maxlen=100)
        mean_scores = []
        max_scores_hist = [] # Historique du score max sur la fenêtre glissante
        epsilons = []
        rewards_hist = []
        steps_hist = []
        best_mean_score = 0
        high_score = 0
        last_score = 0
        
        print("Démarrage entraînement Pacman...")
        time.sleep(5)  # Temps pour se préparer avant le lancement
        # Init MAME
        comm.communicate([
            f"write_memory {Memory.CREDITS}(1)",
            "execute P1_start(1)",
            f"execute throttle_rate({self.training_speed})", # Vitesse max
            "execute throttled(0)",
            f"frame_per_step {NB_DE_FRAMES_STEP}"
        ])

        for episode in range(10000):
            if record: recorder.start_recording()
            comm.communicate(["wait_for 0"])
            if self.flag_quit: break
            
            game.debug = self.debug
            if self.debug >= 1:
                print(f"\n[DEBUG] === Episode {episode} Start ===")
            
            # Reset
            comm.communicate([
                f"write_memory {Memory.CREDITS}(1)",
                "execute P1_start(1)"
            ])
            
            # Wait for start
            alive = 0
            wait_start_timer = time.time()
            while alive == 0:
                _, _, alive, _ = game.get_score_and_lives()
                # Si le jeu ne démarre pas après 3 secondes, on réinsère une pièce et on appuie sur Start
                if time.time() - wait_start_timer > 3.0:
                    if self.debug >= 1: print("[DEBUG] Timeout start -> Relance P1_Start...")
                    comm.communicate([f"write_memory {Memory.CREDITS}(1)", "execute P1_start(1)"])
                    wait_start_timer = time.time()
                time.sleep(0.1)
            
            # Init history
            trainer.state_history.clear()
            for _ in range(N):
                s, _ = config.state_extractor() # s est (32,32) pour CNN ou (1034,) pour MLP
                trainer.state_history.append(s)
            
            # Stack initial
            if model_type == "cnn":
                current_state_stack = np.stack(trainer.state_history, axis=0)
            else:
                current_state_stack = np.concatenate(trainer.state_history)
            
            score = 0
            step = 0
            _, lives, _, _ = game.get_score_and_lives() # Initialisation correcte des vies
            done = False
            sum_rewards = 0
            
            while not done:
                if self.flag_quit: break

                # Synchro MAME : On attend que Lua soit prêt à recevoir exactement ce nombre de commandes pour cette frame
                comm.communicate([f"wait_for {NB_DE_DEMANDES_PAR_STEP}"])

                # Action
                action = trainer.select_action(current_state_stack)
                game.execute_action(action)
                
                # Observe
                next_frame, _ = config.state_extractor()
                
                if self.flag_F8_pressed:
                    Visualizer.afficher_frame(next_frame)
                    self.flag_F8_pressed = False

                new_score, new_lives, alive, pills = game.get_score_and_lives()
                
                if self.debug >= 2:
                    print(f"[DEBUG] Step {step} | Act: {GameConstants.ACTIONS[action]} | Score: {new_score} | Lives: {new_lives} | Alive: {alive} | Pills: {pills} | Msgs: {comm.number_of_messages}")
                comm.number_of_messages = 0 # Reset compteur messages
                
                # Reward Shaping
                reward = 0
                # Récompense pour le score
                if new_score > score:
                    reward += min((new_score - score) / 10.0, 160.0) # Clip augmenté (160.0) pour valoriser les 4 fantômes (1600pts)
                
                # Pénalité de mort
                if new_lives < lives:
                    reward -= 50 
                    lives = new_lives
                
                # Pénalité de temps (légère) pour encourager l'action
                reward -= 0.01 # Réduit pour éviter le suicide tactique
                
                score = new_score
                sum_rewards += reward
                
                # Update Stack
                trainer.state_history.append(next_frame)
                if model_type == "cnn":
                    next_state_stack = np.stack(trainer.state_history, axis=0)
                else:
                    next_state_stack = np.concatenate(trainer.state_history)
                
                # Check Game Over
                if alive == 0 and lives == 0:
                    done = True
                    reward -= 10
                
                # Store & Train
                if config.nstep:
                    # Gestion N-Step (Rainbow) : Accumulation des récompenses sur N frames
                    nstep_tr = trainer.nstep_wrapper.append(current_state_stack, action, reward, done, next_state_stack)
                    if nstep_tr:
                        trainer.replay_buffer.push(*nstep_tr)
                    if done:
                        for tr in trainer.nstep_wrapper.flush():
                            trainer.replay_buffer.push(*tr)
                else:
                    trainer.replay_buffer.push(current_state_stack, action, reward, next_state_stack, done)
                
                if step % TRAIN_EVERY_N_GLOBAL_STEPS == 0:
                    trainer.train_step()
                
                current_state_stack = next_state_stack
                step += 1
                
                # Gestion mort (attente respawn)
                if alive == 0 and lives > 0:
                    # Attendre que le joueur soit de nouveau en vie
                    if self.debug >= 1: print("[DEBUG] Player died. Waiting for respawn...")
                    wait_respawn_timer = time.time()
                    while lives == 0:
                        _, lives, alive, _ = game.get_score_and_lives()
                        # Timeout de sécurité : si le respawn est trop long (>8s), c'est un Game Over
                        if time.time() - wait_respawn_timer > 8.0:
                            if self.debug >= 1: print("[DEBUG] Respawn timeout -> Game Over assumed.")
                            done = True
                            break

            if record:
                recorder.stop_recording()
                if score > last_score:
                    if last_score != 0 and os.path.exists(f"best_game_{last_score}.avi"):
                        os.remove(f"best_game_{last_score}.avi")
                    shutil.copy("output-obs.mp4", f"best_game_{score}.avi")
                    last_score = score

            # End Episode
            scores.append(score)
            high_score = max(high_score, score)
            mean = sum(scores) / len(scores)
            mean_scores.append(mean)
            max_scores_hist.append(max(scores) if scores else 0)
            
            if use_noisy:
                sigmas = trainer.dqn.get_sigma_values()
                avg_sigma = sum(sigmas.values()) / len(sigmas) if sigmas else 0
                min_sigma = min(sigmas.values()) if sigmas else 0
                max_sigma = max(sigmas.values()) if sigmas else 0
                epsilons.append(avg_sigma)
            else:
                epsilons.append(trainer.epsilon)
                min_sigma = max_sigma = 0

            rewards_hist.append(sum_rewards)
            steps_hist.append(step)
            
            # Affichage enrichi : Moyenne [Min - Max] pour voir si certaines couches explorent encore
            explo_str = f"Sigma: {epsilons[-1]:.4f} [{min_sigma:.4f}-{max_sigma:.4f}]" if use_noisy else f"Epsilon: {trainer.epsilon:.3f}"
            timestamp = datetime.now().strftime("%H:%M:%S")
            print(f"{Fore.GREEN}[{timestamp}] Episode {episode} finished. Score: {score}. Mean: {mean:.2f}. {explo_str}{Style.RESET_ALL}")
            
            # Sauvegarde Best Model
            if len(scores) >= 100 and mean > (best_mean_score + 5):
                best_mean_score = mean
                trainer.save_model(best_model_filename)
                print(f"{Fore.YELLOW}🏆 Record moyen ({best_mean_score:.2f}) ! Sauvegarde best.{Style.RESET_ALL}")
                try: pygame.mixer.Sound("c:\\Windows\\Media\\tada.wav").play()
                except: pass
            
            if episode % 10 == 0:
                trainer.save_model(model_filename)
                Visualizer.create_fig(episode, mean_scores, 100, epsilons, rewards_hist, steps_hist, "Pacman_fig", high_score, label_curve="Sigma" if use_noisy else "Epsilon", max_scores=max_scores_hist)
            
            if episode % 100 == 0:
                trainer.save_buffer("pacman.buffer")

            if self.flag_create_fig:
                Visualizer.create_fig(episode, mean_scores, 100, epsilons, rewards_hist, steps_hist, "Pacman_fig_manual", high_score, label_curve="Sigma" if use_noisy else "Epsilon", max_scores=max_scores_hist)
                self.flag_create_fig = False

        trainer.save_model(model_filename)
        
        # Sauvegarde du meilleur modèle avec stats dans le nom sur demande F2 (ou fin)
        if os.path.exists(best_model_filename):
            base_name = f"pacman_{model_type}_best_{episode}ep_HS{high_score}"
            best_with_stats = f"{base_name}.pth"
            try:
                shutil.copy(best_model_filename, best_with_stats)
                print(f"{Fore.CYAN}🏆 Meilleur modèle sauvegardé sous : {best_with_stats}{Style.RESET_ALL}")
                Visualizer.create_fig(episode, mean_scores, 100, epsilons, rewards_hist, steps_hist, base_name, high_score, label_curve="Sigma" if use_noisy else "Epsilon", max_scores=max_scores_hist)
                print(f"{Fore.CYAN}📊 Figure finale sauvegardée sous : {base_name}.png{Style.RESET_ALL}")
            except Exception as e:
                print(f"{Fore.RED}❌ Erreur copie best model: {e}{Style.RESET_ALL}")

        trainer.save_buffer("pacman.buffer")
        
        if episode >= 1000:
            self.log_results(episode, mean_scores, config, NB_DE_DEMANDES_PAR_STEP, NB_DE_FRAMES_STEP, -50, -0.1)
            
        if record:
            recorder.stop_recording()
            recorder.ws.disconnect()
            
        self.process.terminate()

if __name__ == "__main__":
    app = PacmanApp()
    app.run()
