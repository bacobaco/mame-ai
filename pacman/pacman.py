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

# Imports locaux depuis le dossier CORE
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
    PLAYER_ALIVE    = "4EAE"  # Flag statut joueur (Sûr : >0 si en jeu)
    GAME_STATE      = "4E73"  # État principal du jeu (Douteux sur cette version)
    
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
            grid = np.zeros((64, 56), dtype=np.float32)
        else:
            try:
                # VRAM est 1024 octets (32x32 tuiles).
                raw_list = list(map(int, response_vram[0].split(",")))
                
                # --- CONTRASTE AMÉLIORÉ (Sauts de 0.3 entre Danger/Mur/Chemin) ---
                # MURS = 0.3 (Gris sombre)
                grid_small = np.full((32, 28), 0.3, dtype=np.float32)
                
                for idx, val in enumerate(raw_list):
                    if idx >= 1024: break
                    r = idx % 32
                    # CORRECTION MIROIR : On applique 27 - col pour aligner sur le hardware
                    c = 27 - ((idx // 32) - 2)
                    
                    if 0 <= r < 32 and 0 <= c < 28:
                        if val == 0x40: grid_small[r, c] = 0.6 # Chemin Vide (Gris moyen)
                        elif val == 0x10: grid_small[r, c] = 1.0 # Pastille (Blanc pur)
                        elif val == 0x14: grid_small[r, c] = 1.0 # Power (Blanc pur)
                
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
                    # Danger = 0.0 (Noir)
                    val = 0.0 
                    
                    if state == 1 or state == 2: # Blue ou Flash
                        val = 1.0 # Mangeable (Cible, Blanc pur comme les pastilles)
                    elif state >= 3: # Eaten (3) ou Eyes (4+)
                        val = 0.6 # Neutre (Identique au couloir vide)

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

    def get_all_data_batched(self):
        """Récupère VRAM, Positions et Score en un seul appel socket (Gain de temps massif)."""
        cmds = [
            f"read_memory_range {Memory.VRAM_START}({Memory.VRAM_LEN})", # Index 0
            f"read_memory {Memory.PACMAN_X}", f"read_memory {Memory.PACMAN_Y}", # 1,2
            f"read_memory {Memory.BLINKY_X}", f"read_memory {Memory.BLINKY_Y}", # 3,4
            f"read_memory {Memory.PINKY_X}",  f"read_memory {Memory.PINKY_Y}", # 5,6
            f"read_memory {Memory.INKY_X}",   f"read_memory {Memory.INKY_Y}",  # 7,8
            f"read_memory {Memory.CLYDE_X}",  f"read_memory {Memory.CLYDE_Y}",  # 9,10
            f"read_memory {Memory.GHOST_STATE_BLINKY}", # 11
            f"read_memory {Memory.GHOST_STATE_PINKY}",  # 12
            f"read_memory {Memory.GHOST_STATE_INKY}",   # 13
            f"read_memory {Memory.GHOST_STATE_CLYDE}",  # 14
            f"read_memory {Memory.SCORE_10}",           # 15
            f"read_memory {Memory.SCORE_100}",          # 16
            f"read_memory {Memory.SCORE_1000}",         # 17
            f"read_memory {Memory.LIVES}",              # 18
            f"read_memory {Memory.PLAYER_ALIVE}",       # 19 (Utilisé pour alive_flag)
            f"read_memory {Memory.PILLS_COUNT}"         # 20
        ]
        return self.comm.communicate(cmds)

    def process_all_data(self, response):
        """Découpe les réponses groupées et les distribue."""
        if not response or len(response) < 21:
            return None
        
        # 1. Parsing Score/Lives (Indices 15-20)
        try:
            r_score = response[15:21]
            dizaines, centaines, dizaines_de_milliers = map(int, r_score[:3])
            dizaines = int(hex(dizaines)[2:])
            centaines = int(hex(centaines)[2:])
            dizaines_de_milliers = int(hex(dizaines_de_milliers)[2:])
            score = dizaines + centaines * 100 + dizaines_de_milliers * 10000
            lives = int(r_score[3])
            
            # Utilisation de PLAYER_ALIVE (4EAE) car GAME_STATE est instable
            player_alive = int(r_score[4])
            alive_flag = 1 if player_alive != 0 else 0
            
            pills = int(r_score[5])
        except:
            score, lives, alive_flag, pills = 0, 0, 0, 0

        # 2. Parsing CNN State
        try:
            raw_vram = list(map(int, response[0].split(",")))
            grid_small = np.full((32, 28), 0.3, dtype=np.float32)
            for idx, val in enumerate(raw_vram):
                if idx >= 1024: break
                r = idx % 32
                c = 27 - ((idx // 32) - 2)
                if 0 <= r < 32 and 0 <= c < 28:
                    if val == 0x40: grid_small[r, c] = 0.6
                    elif val == 0x10: grid_small[r, c] = 1.0
                    elif val == 0x14: grid_small[r, c] = 1.0
            grid = np.kron(grid_small, np.ones((2, 2)))

            # Sprites Positions (Indices 1-14)
            offset_x, offset_y = -4, -16 
            px, py = int(response[1]), int(response[2])
            pr = min(63, max(0, (px + offset_x) // 4))
            pc = min(55, max(0, 55 - ((py + offset_y) // 4)))
            grid[max(0, pr):min(64, pr+2), max(0, pc):min(56, pc+2)] = 0.7 

            for i in range(4):
                fx, fy = int(response[3 + i*2]), int(response[4 + i*2])
                state = int(response[11 + i])
                val = 0.0 if state < 1 else (1.0 if state <= 2 else 0.6)
                fr = min(63, max(0, (fx + offset_x) // 4))
                fc = min(55, max(0, 55 - ((fy + offset_y) // 4)))
                grid[max(0, pr):min(64, pr+2), max(0, pc):min(56, pc+2)] = 0.8
                grid[max(0, fr):min(64, fr+2), max(0, fc):min(56, fc+2)] = val
        except:
            grid = np.zeros((64, 56), dtype=np.float32)

        return grid, score, lives, alive_flag, pills

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
    def create_fig(nb_parties, scores_moyens, fenetre, epsilons, rewards, nb_steps, filename="Pacman_fig", high_score=0, label_curve="Epsilon", max_scores=None, x_axis=None):
        matplotlib.use("Agg")
        plt.close("all")
        fig, (ax1, ax3) = plt.subplots(2, 1, figsize=(12, 10), gridspec_kw={'height_ratios': [3, 1]})
        
        ax1.set_xlabel("Episodes")
        ax1.set_ylabel(label_curve, color="tab:blue")
        x_plot = x_axis if x_axis is not None else np.arange(len(epsilons))
        ax1.plot(x_plot, epsilons, color="tab:blue", linestyle="--", label=label_curve)
        ax1.tick_params(axis='y', labelcolor="tab:blue")
        ax1.legend(loc="upper left")
        
        ax2 = ax1.twinx()
        ax2.set_ylabel("Score (Moyen & Max)", color="tab:red")
        ax2.plot(x_plot, scores_moyens, color="tab:red", label="Score Moyen")
        
        # Ajout courbe Max Score (Rolling Max) pour voir le potentiel max
        if max_scores and len(max_scores) == len(scores_moyens):
            ax2.plot(x_plot, max_scores, color="tab:green", alpha=0.3, linestyle="-", linewidth=1, label="Max Score (fenêtre)")
            
        ax2.tick_params(axis='y', labelcolor="tab:red")
        ax2.legend(loc="upper right")
        
        # Ajout des lignes de tendance (comme invaders.py)
        if len(scores_moyens) > 1:
            z = np.polyfit(x_plot, scores_moyens, 1)
            p = np.poly1d(z)
            ax2.plot(x_plot, p(x_plot), "tab:orange", alpha=0.6)           
            pente_globale = z[0]
            # Affichage pente globale
            midpoint_global = x_plot[len(x_plot)//2]
            ax2.text(midpoint_global, p(midpoint_global), f"Pente Globale: {pente_globale:.2f}", color="tab:orange", fontweight="bold", ha="center", va="bottom")
            
            # Pente locale
            n = min(100, len(scores_moyens)) # On regarde les 100 derniers points enregistrés
            x_recent = x_plot[-n:]
            z_recent = np.polyfit(x_recent, scores_moyens[-n:], 1)
            p_recent = np.poly1d(z_recent)
            ax2.plot(x_recent, p_recent(x_recent), "tab:purple", linestyle="--")
            pente_recent = z_recent[0]
            
            midpoint = x_plot[-n//2]
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
        self.debug_lua = True
        self.flag_quit = False
        self.flag_create_fig = False
        self.flag_F8_pressed = False
        self.is_normal_speed = False
        self.training_speed = 5.0
        self.slow_speed = 0.2
        self.pending_commands = []
        
        # Web Server
        self.web_server = GraphWebServer(graph_dir=MEDIA_DIR, host="0.0.0.0", port=5000, auto_display_latest=True)
        
        # --- INITIALISATION RADAR TEMPS RÉEL ---
        pygame.init()
        # Fenêtre large pour afficher Radar (280px) + Vision IA (280px)
        self.radar_screen = pygame.display.set_mode((560, 330))
        pygame.display.set_caption("Pacman AI Training - Realtime Monitor (Radar | Vision IA)")
        self.clock = pygame.time.Clock()
        
        pygame.mixer.init()

    def update_radar(self, responses, next_frame):
        """Dessine le radar et la vision IA en temps réel."""
        if not responses or len(responses) < 15: return
        
        # 1. Fond
        self.radar_screen.fill((20, 20, 20))
        
        # --- PARTIE GAUCHE : RADAR LOGIQUE (32x28) ---
        # On utilise le même code que pacman_robot.py
        try:
            vram = list(map(int, responses[0].split(",")))
            # Dessin de la grille
            for idx, val in enumerate(vram):
                if idx >= 1024: break
                # Mapping Midway : Column-Major
                r = idx % 32
                c = 27 - ((idx // 32) - 2)
                if 0 <= r < 32 and 0 <= c < 28:
                    if val in [64, 16, 20, 0xBF]: # Murs/Vides
                        color = (0, 0, 80) if val == 64 else (10, 10, 10)
                        if val in [16, 20]: # Pastilles
                            pygame.draw.circle(self.radar_screen, (255, 255, 0), (c*10+5, r*10+5), 2)
                        else:
                            pygame.draw.rect(self.radar_screen, color, (c*10, r*10, 9, 9))
            
            # Sprites : Pac-Man (index 1,2) et Fantômes (index 3-10)
            # Offset Milestone : H=+2, V=+0
            # Pacman
            py, px = int(responses[1]), int(responses[2])
            pr, pc = py // 8, (27 - (px // 8) + 2) % 28
            pygame.draw.circle(self.radar_screen, (255, 255, 255), (pc*10+5, pr*10+5), 5)
            
            # Fantômes
            for i in range(4):
                gy, gx = int(responses[3+i*2]), int(responses[4+i*2])
                state = int(responses[11+i])
                gr, gc = gy // 8, (27 - (gx // 8) + 2) % 28
                color = (255, 0, 0) if state == 0 else (0, 255, 255)
                pygame.draw.rect(self.radar_screen, color, (gc*10+1, gr*10+1, 8, 8))
        except: pass

        # --- PARTIE DROITE : VISION IA (64x56) ---
        # On dessine le tenseur 'frame' tel qu'il est envoyé au cerveau
        start_x = 280
        if next_frame is not None:
            # next_frame est 64x56 normalisé [0,1]
            for r in range(64):
                for c in range(56):
                    val = int(next_frame[r, c] * 255)
                    # On upscale légèrement pour remplir l'espace (x4 pixels)
                    # row * 5 pixels car 64*5 = 320px
                    pygame.draw.rect(self.radar_screen, (val, val, val), (start_x + c*5, r*5, 5, 5))

        # Séparateur
        pygame.draw.line(self.radar_screen, (100, 100, 100), (280, 0), (280, 320), 2)
        
        pygame.display.flip()

    def log_results(self, nb_episodes, mean_scores, config, nb_mess, nb_step_frame, r_kill, r_step):
        filename = os.path.join(SCRIPT_DIR, "resultats_pacman.txt")
        
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

    def launch_obs(self):
        """Tente de lancer OBS Studio si non détecté."""
        obs_path = r"C:\Program Files\obs-studio\bin\64bit\obs64.exe"
        if os.path.exists(obs_path):
            try:
                print(f"{Fore.YELLOW}🚀 Lancement d'OBS Studio...{Style.RESET_ALL}")
                subprocess.Popen([obs_path], cwd=os.path.dirname(obs_path))
                time.sleep(8)
            except Exception as e:
                print(f"{Fore.RED}Erreur lancement OBS: {e}{Style.RESET_ALL}")

    def launch_mame(self, visible=False):
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
            "pacman",
            "-autoboot_delay", "1",
            "-autoboot_script", LUA_SCRIPT_PATH,
        ]
        # Utilisation de startupinfo pour minimiser la fenêtre console si besoin
        startupinfo = subprocess.STARTUPINFO()
        startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
        startupinfo.wShowWindow = 1 if visible else 7 # 1=Visible, 7=Minimisé
        
        self.process = subprocess.Popen(command, cwd=MAME_DIR, startupinfo=startupinfo)
        time.sleep(10)

    def setup_keyboard(self, trainer, config, comm):
        def on_key_press(event):
            def is_terminal_in_focus():
                hwnd = win32gui.GetForegroundWindow()
                title = win32gui.GetWindowText(hwnd).lower()
                # Plus flexible : VS Code, Antigravity, CMD, PowerShell ou le script lui-même
                return "code" in title or "antigravity" in title or "powershell" in title or "cmd" in title or "pacman" in title or "python" in title

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
                        self.pending_commands.append(f"execute throttle_rate({new_speed})")
                        print(f"\n[F5] Vitesse : {'Low (' + str(self.slow_speed) + ')' if self.is_normal_speed else f'Rapide ({self.training_speed})'}")
                    elif keyboard.is_pressed("f6"):
                        self.slow_speed = round(self.slow_speed + 0.1, 1)
                        print(f"\n[F6] Vitesse lente ajustée : {self.slow_speed}")
                        if self.is_normal_speed:
                            self.pending_commands.append(f"execute throttle_rate({self.slow_speed})")
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
                    elif keyboard.is_pressed("f12"):
                        self.debug_lua = not self.debug_lua
                        self.pending_commands.append(f"debug {'on' if self.debug_lua else 'off'}")
                        print(f"\n[F12] Debug Lua : {'ON' if self.debug_lua else 'OFF'}")

        keyboard.on_press(on_key_press)

    def run(self):
        # Config
        RESUME = True # Nouvelle session "Elite" (Repart de zéro avec nouvelle architecture)
        model_type = "cnn" # "cnn" ou "mlp" - MLP recommandé pour Pacman (VRAM indices)
        N = 4 # Stack size (Standard Atari pour bien capter le mouvement)
        NB_DE_FRAMES_STEP = 4 # Plus réactif pour ne pas rater les virages (Pacman va vite)
        TRAIN_EVERY_N_GLOBAL_STEPS = 4 # Entraîner tous les 4 steps pour ne pas tuer le CPU/GPU
        # Calcul du nombre de messages par step pour la synchro wait_for
        # wait_for (1) + get_all_data_batched(21) + execute_action(4) = 26 total lines
        NB_DE_DEMANDES_PAR_STEP = 25
        
        model_filename = os.path.join(SCRIPT_DIR, f"pacman_{model_type}.pth")
        best_model_filename = os.path.join(SCRIPT_DIR, "pacman_best.pth")
        
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
            hidden_size=1024,
            output_size=4,
            learning_rate=0.0000625,
            gamma=0.99, # Standard DeepMind
            use_noisy=use_noisy,
            buffer_capacity=500000, 
            batch_size=256,
            min_history_size=30000, 
            cnn_type="precise",
            model_type=model_type,
            epsilon_start=epsilon_start,
            epsilon_end=epsilon_end,
            epsilon_linear=epsilon_linear,
            epsilon_decay=epsilon_decay, 
            epsilon_add=0.0,
            double_dqn=True,
            dueling=True,
            nstep=True,       
            nstep_n=3,
            prioritized_replay=True,
            optimize_memory=True
        )

        print(f"Configuration Input Size: {input_size} (Model: {model_type})")
        
        # Recorder & OBS
        record = True # Mettre à True pour enregistrer avec OBS
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

        self.launch_mame(visible=record)
        comm = MameCommunicator("localhost", 12346) # Port défini dans Lua pour Pacman
        game = PacmanInterface(comm)
        config.state_extractor = StateExtractor(game, model_type)
        
        trainer = DQNTrainer(config)
        self.setup_keyboard(trainer, config, comm)
        
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

        buffer_filename = os.path.join(SCRIPT_DIR, "pacman.buffer")
        if RESUME and os.path.exists(buffer_filename):
            trainer.load_buffer(buffer_filename)

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
        time.sleep(3)  # Temps pour se préparer avant le lancement
        # Init MAME
        comm.communicate(["wait_for 3"])
        comm.communicate([
            f"execute throttle_rate({self.training_speed})", # Vitesse max
            "execute throttled(0)",
            f"frame_per_step {NB_DE_FRAMES_STEP}"
        ])

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
                    
        learner_thread = threading.Thread(target=learner_thread_func, daemon=True)
        learner_thread.start()
        # -----------------------------------------------

        nb_steps_total = 0

        for episode in range(90000):
            trainer.config.current_episode = episode
            if record: recorder.start_recording()
            if self.flag_quit: break
            
            game.debug = self.debug
            if self.debug >= 1:
                print(f"\n[DEBUG] === Episode {episode} Start ===")
            
            # Reset
            comm.communicate(["wait_for 2"])
            comm.communicate([
                f"write_memory {Memory.CREDITS}(1)",
                "execute P1_start(1)"
            ])
            if self.debug >= 1: print("[DEBUG] Piece insérée...")
            # Wait for start
            alive = 0
            comm.communicate(["wait_for 6"])
            while alive == 0:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT: self.flag_quit = True
                data = game.get_score_and_lives()
                if data: _, _, alive, _ = data
                if self.debug >= 1: print("[DEBUG] dans le while debut de partie: alive=",alive)

            # Init history
            pacman_loop_frame_history = deque(maxlen=N)
            for _ in range(N):
                s, _ = config.state_extractor() # s est (32,32) pour CNN ou (1034,) pour MLP
                pacman_loop_frame_history.append(s)
            
            # Stack initial
            if model_type == "cnn":
                current_state_stack = np.stack(pacman_loop_frame_history, axis=0)
            else:
                current_state_stack = np.concatenate(pacman_loop_frame_history)
            
            score = 0
            step = 0
            lives = 3
            #_, lives, _, _ = game.get_score_and_lives() # Initialisation correcte des vies
            done = False
            sum_rewards = 0

            comm.communicate([f"wait_for {NB_DE_DEMANDES_PAR_STEP}"])
            while not done:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT: self.flag_quit = True
                if self.flag_quit: break

                # Action
                action = trainer.select_action(current_state_stack)
                game.execute_action(action)
                
                # Observe (BATCH CALL - 1 seul aller-retour socket)
                responses = game.get_all_data_batched()
                batch_data = game.process_all_data(responses)
                
                if not batch_data:
                    print("⚠️ Erreur : Pas de données reçues de MAME.")
                    continue
                    
                next_frame, new_score, new_lives, alive, pills = batch_data
                
                # --- UPDATE VISUALISATION TEMPS RÉEL ---
                self.update_radar(responses, next_frame)
                
                if self.flag_F8_pressed:
                    Visualizer.afficher_frame(next_frame)
                    self.flag_F8_pressed = False
                
                if self.debug >= 2:
                    print(f"[DEBUG] Step {step} | Act: {GameConstants.ACTIONS[action]} | Score: {new_score} | Lives: {new_lives} | Alive: {alive} | Pills: {pills} | Msgs: {comm.number_of_messages}")
                comm.number_of_messages = 0 # Reset compteur messages
                
                # Reward Shaping
                reward = 0
                # --- REWARD CLIPPING : DEEPMIND STANDARD (+1/-1) ---
                reward = 0.0
                if new_score > score:
                    reward = 1.0
                
                if new_lives < lives:
                    reward = -1.0
                    lives = new_lives
                # ----------------------------------------------------
                
                score = new_score
                sum_rewards += reward
                
                # Update Stack
                pacman_loop_frame_history.append(next_frame)
                if model_type == "cnn":
                    next_state_stack = np.stack(pacman_loop_frame_history, axis=0)
                else:
                    next_state_stack = np.concatenate(pacman_loop_frame_history)
                
                # Check Game Over
                if alive == 0 and lives == 0:
                    done = True
                    reward -= 10
                
                # Store & Train
                if config.mode.lower() == "exploration":
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
                
                current_state_stack = next_state_stack
                trainer.update_epsilon()

                # --- SYNCHRONISATION ASYNCHRONE ---
                if not self.training_active and hasattr(trainer, 'replay_buffer') and trainer.replay_buffer.size >= config.min_history_size:
                    self.training_active = True
                    print(f"\n{Fore.GREEN}✅ [Actor] Buffer prêt ({config.min_history_size} éléments). Début de l'entraînement GPU asynchrone régulé !{Style.RESET_ALL}")
                
                if self.training_active and nb_steps_total % TRAIN_EVERY_N_GLOBAL_STEPS == 0:
                    self.train_queue += 1
                # ----------------------------------

                step += 1
                nb_steps_total += 1
                
                # Gestion mort (attente respawn) asynchrone sécurisée
                if alive == 0 and lives > 0:
                    if self.debug >= 1: print("[DEBUG] Mort! alive=",alive,"-lives=",lives)
                    comm.communicate(["wait_for 6"])
                    while alive == 0:
                        for event in pygame.event.get():
                            if event.type == pygame.QUIT: self.flag_quit = True
                        data = game.get_score_and_lives()
                        if data: _, lives, alive, _ = data
                        if self.debug >= 1: print("[DEBUG] boucle while (mort) alive=",alive,"lives=",lives)
                        if lives==0: break
                    comm.communicate([f"wait_for {NB_DE_DEMANDES_PAR_STEP}"])
            if record:
                if score > last_score:
                    time.sleep(2.0) # Attente pour capturer la fin de partie uniquement si record
                video_path = recorder.stop_recording()

                if video_path and os.path.exists(video_path):
                    if score > last_score:
                        # Supprimer l'ancienne meilleure vidéo si elle existe
                        if last_score != 0:
                            old_video = os.path.join(MEDIA_DIR, f"pacman_best_game_{last_score}.mp4")
                            if os.path.exists(old_video):
                                try:
                                    os.remove(old_video)
                                except OSError as e:
                                    print(f"Erreur suppression ancienne vidéo : {e}")
                        
                        # Copier la nouvelle meilleure vidéo
                        try:
                            dst_mp4 = os.path.join(MEDIA_DIR, f"pacman_best_game_{score}.mp4")
                            shutil.copy(video_path, dst_mp4)
                            print(f"📼 Nouvelle meilleure partie enregistrée : {dst_mp4}")
                            
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
                        # Suppression de la vidéo si ce n'est pas un record
                        try:
                            os.remove(video_path)
                        except Exception as e:
                            print(f"⚠️ Erreur suppression vidéo non-record : {e}")
                elif video_path: # Le chemin a été retourné mais le fichier n'existe pas
                    print(f"{Fore.YELLOW}⚠️ OBS a retourné un chemin ({video_path}) mais le fichier est introuvable.{Style.RESET_ALL}")

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
            if len(scores) >= 200 and mean > (best_mean_score + 5):
                best_mean_score = mean
                trainer.save_model(best_model_filename)
                print(f"{Fore.YELLOW}🏆 Record moyen ({best_mean_score:.2f}) ! Sauvegarde best.{Style.RESET_ALL}")
                try: pygame.mixer.Sound("c:\\Windows\\Media\\tada.wav").play()
                except: pass
            
            if episode % 10 == 0:
                trainer.save_model(model_filename)
                Visualizer.create_fig(episode, mean_scores, 100, epsilons, rewards_hist, steps_hist, os.path.join(MEDIA_DIR, "Pacman_fig"), high_score, label_curve="Sigma" if use_noisy else "Epsilon", max_scores=max_scores_hist)
            
            if episode % 100 == 0:
                trainer.save_buffer(os.path.join(SCRIPT_DIR, "pacman.buffer"))

            if self.flag_create_fig:
                Visualizer.create_fig(episode, mean_scores, 100, epsilons, rewards_hist, steps_hist, os.path.join(MEDIA_DIR, "Pacman_fig_manual"), high_score, label_curve="Sigma" if use_noisy else "Epsilon", max_scores=max_scores_hist)
                self.flag_create_fig = False

        trainer.save_model(model_filename)
        
        # Sauvegarde du meilleur modèle avec stats dans le nom sur demande F2 (ou fin)
        if os.path.exists(best_model_filename):
            base_name = f"pacman_{model_type}_best_{episode}ep_HS{high_score}"
            best_with_stats = os.path.join(SCRIPT_DIR, f"{base_name}.pth")
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
