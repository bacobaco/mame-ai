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
        Construit une grille sémantique multi-canaux native (4, 32, 28) en [0, 1] sans perte d'information :
        - Canal 0 : Labyrinthe & Pastilles (Murs=0.2, Couloir vide=0.0, Pastille=0.5, Power Pellet=1.0)
        - Canal 1 : Pac-Man uniquement (=1.0)
        - Canal 2 : Fantômes Dangereux uniquement (=1.0) -> Séparé de Pac-Man pour éviter le masquage par MaxPool !
        - Canal 3 : Fantômes Comestibles (Bleu solide=1.0, Clignotant=0.5)
        """
        response_vram = self.comm.communicate([f"read_memory_range {Memory.VRAM_START}({Memory.VRAM_LEN})"])
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
        grid = self._build_semantic_grid(
            response_vram[0] if response_vram else "",
            response_pos if response_pos and len(response_pos) >= 14 else None
        )
        return grid, 0.0

    @staticmethod
    def _build_semantic_grid(vram_str, pos_list):
        """Construit un tenseur (4, 32, 28) en [0, 1] séparant Murs/Pastilles, Pac-Man, Fantômes Mortels et Fantômes Bleus."""
        grid = np.zeros((4, 32, 28), dtype=np.float32)
        grid[0].fill(0.2) # Par défaut : Mur = 0.2
        if vram_str:
            try:
                raw_vram = list(map(int, vram_str.split(",")))
                for idx, val in enumerate(raw_vram):
                    if idx >= 1024: break
                    r = idx % 32
                    c = 27 - ((idx // 32) - 2)
                    if 0 <= r < 32 and 0 <= c < 28:
                        if val == 0x40 or val == 0:
                            grid[0, r, c] = 0.0 # Couloir vide
                        elif val == 0x10 or val == 0xBF:
                            grid[0, r, c] = 0.5 # Pastille normale
                        elif val == 0x14:
                            grid[0, r, c] = 1.0 # Power Pellet (Super pastille)
            except Exception:
                pass

        if pos_list and len(pos_list) >= 14:
            try:
                px, py = int(pos_list[0]), int(pos_list[1])
                pr = min(31, max(0, px // 8))
                pc = (27 - (py // 8) + 2) % 28
                grid[1, pr, pc] = 1.0 # Pac-Man seul sur le canal 1

                for i in range(4):
                    fx, fy = int(pos_list[2 + i*2]), int(pos_list[3 + i*2])
                    state = int(pos_list[10 + i])
                    fr = min(31, max(0, fx // 8))
                    fc = (27 - (fy // 8) + 2) % 28
                    if state == 0:
                        # Fantôme mortel seul sur le canal 2 (pleine intensité 1.0, jamais masqué par Pac-Man)
                        grid[2, fr, fc] = 1.0
                    elif state == 1:
                        # Fantôme bleu solide (comestible en sécurité) sur le canal 3
                        grid[3, fr, fc] = 1.0
                    elif state == 2:
                        # Fantôme clignotant (fin de bonus imminente !) sur le canal 3
                        grid[3, fr, fc] = max(grid[3, fr, fc], 0.5)
            except Exception:
                pass

        return np.ascontiguousarray(grid)

    @staticmethod
    def build_obs_stack(frame_history):
        """
        Assemble les 4 frames (4, 32, 28) en un tenseur (7, 32, 28) en [0, 1] :
        - Canal 0 : Murs & Pastilles actuels (t)
        - Canal 1 : Pac-Man + sa traînée de mouvement (1.0 à t, 0.75 à t-1, 0.50 à t-2, 0.25 à t-3)
        - Canaux 2..5 : Historique pur des Fantômes Dangereux (t-3, t-2, t-1, t) -> Vitesse & Direction sans masquage
        - Canal 6 : Fantômes Comestibles actuels (t) avec distinction Bleu (1.0) vs Clignotant (0.5)
        """
        frames = list(frame_history)
        pacman_trail = np.maximum.reduce([
            frames[0][1] * 0.25,
            frames[1][1] * 0.50,
            frames[2][1] * 0.75,
            frames[3][1] * 1.00,
        ])
        return np.stack([
            frames[-1][0],
            pacman_trail,
            frames[0][2],
            frames[1][2],
            frames[2][2],
            frames[3][2],
            frames[-1][3],
        ], axis=0).astype(np.float32)

    @staticmethod
    def compute_reward(new_score, score, new_lives, lives, pills, last_pills):
        """
        Calcule la récompense optimisée et l'indicateur de fin de vie (Episodic Life) :
        - Pastille (+10 pts) : +0.1 (échelle Q apprise par pacman_best.pth en Session 43)
        - Power Pellet (+50 pts) : +0.5
        - Fruits (+100..+5000 pts) : +1.0
        - Fantômes bleus (+200, +400, +800, +1600 pts) : progressif +1.0, +2.0, +3.0, +4.0
        - Level Clear : +25.0 (forte incitation à finir le niveau plutôt que mourir sur les dernières pastilles)
        - Perte d'une vie (Episodic Life) : -5.0 + life_lost=True (3 pastilles + mort reste négatif en N-Step=3 -> jamais de suicide)
        - Step sans point : 0.0 (gamma=0.99 suffit à préférer les pastilles proches sans rendre les dernières pastilles négatives)
        """
        reward = 0.0
        life_lost = False
        if new_score > score:
            diff = new_score - score
            if diff >= 1600:
                reward = 4.0
            elif diff >= 800:
                reward = 3.0
            elif diff >= 400:
                reward = 2.0
            elif diff >= 200:
                reward = 1.0
            elif diff in [100, 300, 500, 700, 1000, 2000, 3000, 5000]:
                reward = 1.0
            elif diff == 50:
                reward = 0.5
            else:
                reward = 0.1

        # Bonus Level Clear
        if pills < last_pills and last_pills > 200 and pills < 50 and new_lives == lives:
            reward += 25.0

        # Episodic Life : chaque perte de vie coupe le N-Step (done_transition=True) avec une pénalité anti-suicide
        if new_lives < lives:
            reward = -5.0
            life_lost = True

        return float(reward), life_lost

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
        """Découpe les réponses groupées et construit l'état CNN multi-canaux (3, 32, 28)."""
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
            player_alive = int(r_score[4])
            alive_flag = 1 if player_alive != 0 else 0
            pills = int(r_score[5])
        except Exception:
            score, lives, alive_flag, pills = 0, 0, 0, 0

        # 2. Parsing CNN State (3, 32, 28)
        grid = self._build_semantic_grid(response[0], response[1:15])
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
    def create_fig(nb_parties, scores_moyens, fenetre, epsilons, rewards, nb_steps, filename="Pacman_fig", high_score=0, label_curve="Epsilon", max_scores=None, x_axis=None, session_info=None):
        matplotlib.use("Agg")
        plt.close("all")
        
        # Détermine si les longueurs d'épisodes sont disponibles
        has_steps = (nb_steps is not None and len(nb_steps) > 0 and len(nb_steps) == len(scores_moyens))
        
        # Palette de couleurs Premium (Thème Sombre moderne)
        fig_bg = "#0f0f13"     # Fond de figure sombre profond
        ax_bg = "#151520"      # Fond d'axe anthracite bleuté
        text_color = "#f8fafc" # Texte blanc cassé
        grid_color = "#27273a" # Lignes de grille discrètes
        cyan_color = "#00d2fc" # Epsilon / Sigma (Bleu néon)
        rose_color = "#ff3366" # Score moyen (Rose/Rouge néon)
        green_color = "#10ac84" # Max score (Émeraude)
        orange_color = "#ff9f43" # Tendance globale (Orange chaud)
        purple_color = "#a29bfe" # Tendance locale (Lavande)
        
        # Configuration des styles par défaut de matplotlib pour cette figure
        plt.rcParams['text.color'] = text_color
        plt.rcParams['axes.labelcolor'] = text_color
        plt.rcParams['xtick.color'] = "#94a3b8"
        plt.rcParams['ytick.color'] = "#94a3b8"
        plt.rcParams['grid.color'] = grid_color
        plt.rcParams['grid.linestyle'] = '--'
        plt.rcParams['grid.alpha'] = 0.6
        
        # Création dynamique de la figure : 2 sous-graphiques si steps dispos, sinon 1 seul grand graphique
        if has_steps:
            fig, (ax1, ax3) = plt.subplots(2, 1, figsize=(13, 9.5), gridspec_kw={'height_ratios': [2.8, 1]}, facecolor=fig_bg)
        else:
            fig, ax1 = plt.subplots(figsize=(13, 7.2), facecolor=fig_bg)
            ax3 = None
            
        # Ajustement des marges
        fig.subplots_adjust(top=0.86, bottom=0.09, left=0.08, right=0.92, hspace=0.22)
        
        x_plot = x_axis if x_axis is not None else np.arange(len(epsilons))
        
        # --- GRAPH 1 : SCORES & EXPLORATION ---
        ax1.set_facecolor(ax_bg)
        ax1.set_xlabel("Épisodes", fontsize=10, fontweight="bold", labelpad=6)
        ax1.set_ylabel(label_curve, color=cyan_color, fontsize=10, fontweight="bold", labelpad=6)
        
        # Courbe Epsilon / Sigma
        ax1.plot(x_plot, epsilons, color=cyan_color, linestyle="--", linewidth=1.5, label=label_curve, alpha=0.8)
        ax1.tick_params(axis='y', labelcolor=cyan_color)
        ax1.set_zorder(1)
        
        # Axe jumeau pour les scores
        ax2 = ax1.twinx()
        ax2.set_ylabel("Score", color=rose_color, fontsize=10, fontweight="bold", labelpad=12, rotation=270)
        
        # Enveloppe de performance (Fill entre moyenne et max)
        if max_scores and len(max_scores) == len(scores_moyens):
            ax2.fill_between(x_plot, scores_moyens, max_scores, color=rose_color, alpha=0.08, label="Enveloppe Max-Moyenne")
            ax2.plot(x_plot, max_scores, color=green_color, alpha=0.4, linestyle="-", linewidth=1.0, label="Max (Fenêtre)")
            
        # Courbe Score Moyen
        ax2.plot(x_plot, scores_moyens, color=rose_color, linewidth=2.0, label="Score Moyen")
        ax2.tick_params(axis='y', labelcolor=rose_color)
        
        # Bordures et style d'axe
        for ax in [ax1, ax2]:
            ax.spines['top'].set_visible(False)
            ax.spines['left'].set_color(grid_color)
            ax.spines['right'].set_color(grid_color)
            ax.spines['bottom'].set_color(grid_color)
            
        ax1.grid(True, which='both')
        
        # Calcul des lignes de tendance
        if len(scores_moyens) > 1 and np.max(x_plot) > np.min(x_plot):
            try:
                z = np.polyfit(x_plot, scores_moyens, 1)
                p = np.poly1d(z)
                ax2.plot(x_plot, p(x_plot), color=orange_color, alpha=0.7, linestyle="-", linewidth=1.2, label=f"Tendance Globale ({z[0]:+.2f}/ep)")
                
                # Pente locale sur les 100 derniers épisodes
                n = min(100, len(scores_moyens))
                x_recent = x_plot[-n:]
                if np.max(x_recent) > np.min(x_recent):
                    z_recent = np.polyfit(x_recent, scores_moyens[-n:], 1)
                    p_recent = np.poly1d(z_recent)
                    ax2.plot(x_recent, p_recent(x_recent), color=purple_color, linestyle="--", linewidth=1.5, label=f"Tendance Locale ({z_recent[0]:+.2f}/ep)")
                    
                    # Bulle d'affichage de la pente récente
                    midpoint = x_plot[-n//2]
                    ax2.text(midpoint, p_recent(midpoint) + (np.max(scores_moyens)*0.02 + 50), f"Pente locale: {z_recent[0]:+.2f}", 
                             color=purple_color, fontweight="bold", fontsize=8, ha="center",
                             bbox=dict(boxstyle="round,pad=0.2", facecolor=ax_bg, alpha=0.85, edgecolor=purple_color, linewidth=1))
            except Exception as e:
                print(f"⚠️ Erreur calcul de tendance : {e}")
                
        # Combinaison des légendes
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax2.legend(lines1 + lines2, labels1 + labels2, loc="upper left", facecolor=ax_bg, edgecolor=grid_color, fontsize=8)
        
        ax1.set_title(f"Pac-Man RL Training - {nb_parties} Épisodes | High Score: {high_score}", 
                     fontsize=12, fontweight="bold", pad=12, color="#ffffff")
                     
        # --- GRAPH 2 : SURVIVABILITÉ (STEPS PAR ÉPISODE) ---
        if has_steps:
            ax3.set_facecolor(ax_bg)
            ax3.plot(x_plot, nb_steps, color="#6c5ce7", alpha=0.75, linewidth=1.2, label="Longueur Épisode (Steps)")
            ax3.set_xlabel("Épisodes", fontsize=10, fontweight="bold", labelpad=6)
            ax3.set_ylabel("Steps / Épisode", color="#a29bfe", fontsize=10, fontweight="bold", labelpad=6)
            ax3.tick_params(axis='y', labelcolor="#a29bfe")
            ax3.grid(True)
            
            # Tendance steps
            if len(nb_steps) > 1 and np.max(x_plot) > np.min(x_plot):
                try:
                    z_steps = np.polyfit(x_plot, nb_steps, 1)
                    p_steps = np.poly1d(z_steps)
                    ax3.plot(x_plot, p_steps(x_plot), color=orange_color, alpha=0.7, linestyle=":", linewidth=1.2, label=f"Pente: {z_steps[0]:+.1f}")
                except:
                    pass
            ax3.legend(loc="upper left", facecolor=ax_bg, edgecolor=grid_color, fontsize=8)
            
            for spine in ['top', 'left', 'right', 'bottom']:
                ax3.spines[spine].set_color(grid_color)
                
        # --- BANDEAU PREMIUM D'INFORMATIONS DE SESSION ---
        if session_info:
            fig.suptitle(session_info, fontsize=11, color="#ffffff", y=0.96, fontweight="bold",
                         bbox=dict(boxstyle="round,pad=0.4", facecolor="#1e1e2f", alpha=0.95, edgecolor="#3b3b4f", linewidth=1.5))
                         
        # Remplacement atomique de la figure
        tmp_filename = f"{filename}.png.tmp"
        plt.savefig(tmp_filename, format="png", dpi=120, facecolor=fig_bg)
        plt.close()
        
        try:
            if os.path.exists(tmp_filename):
                os.replace(tmp_filename, f"{filename}.png")
        except Exception as e:
            print(f"⚠️ Erreur lors du remplacement atomique de {filename}.png : {e}")

    @staticmethod
    def afficher_frame(frame):
        plt.figure(figsize=(6, 6))
        title = "Frame Pacman CNN Input"
        if frame.ndim == 1 and frame.shape[0] >= 1024:
            data_to_show = frame[:1024].reshape(32, 32)
            data_to_show = np.rot90(data_to_show, k=-1)
            title = "Frame Pacman (MLP VRAM 32x32)"
            plt.imshow(data_to_show, cmap="viridis", interpolation="nearest")
            plt.colorbar()
        elif frame.ndim == 3 and frame.shape[0] >= 4:
            # Affichage RGB des 4 canaux sémantiques : R=Danger+PacMan, G=Murs/Pastilles+PacMan, B=Fantômes Bleus
            rgb = np.stack([np.clip(frame[1] + frame[2], 0.0, 1.0), np.clip(frame[0] + frame[1], 0.0, 1.0), frame[3]], axis=-1)
            title = "Frame Pacman CNN Multi-Canaux (4x32x28)"
            plt.imshow(np.clip(rgb, 0.0, 1.0), interpolation="nearest")
        else:
            plt.imshow(frame, cmap="viridis", interpolation="nearest")
            plt.colorbar()
        plt.title(title)
        filename = f"pacman_frame_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        plt.savefig(filename, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Frame sauvegardée : {filename}")

# ==================================================================================================
# APPLICATION PRINCIPALE

class PacmanApp:
    def __init__(self):
        self.debug = 0
        self.debug_lua = True
        self.flag_quit = False
        self.flag_create_fig = False
        self.flag_F8_pressed = False
        self.is_normal_speed = True
        self.training_speed = 1.0
        self.slow_speed = 1.0
        self.pending_commands = []
        
        # Web Server
        self.web_server = GraphWebServer(graph_dir=MEDIA_DIR, host="0.0.0.0", port=5000, auto_display_latest=True)
        
        # --- INITIALISATION RADAR TEMPS RÉEL ---
        pygame.init()
        # Fenêtre large pour afficher Radar (280px) + Vision IA (280px)
        self.radar_screen = pygame.display.set_mode((560, 330))
        pygame.display.set_caption("Pacman AI Training - Realtime Monitor (Radar | Vision IA Multi-Canaux)")
        self.clock = pygame.time.Clock()
        
        pygame.mixer.init()

    def update_radar(self, responses, next_frame):
        """Dessine le radar logique (gauche) et la vision IA multi-canaux (droite) en temps réel."""
        if not responses or len(responses) < 15: return
        
        # 1. Fond
        self.radar_screen.fill((20, 20, 20))
        
        # --- PARTIE GAUCHE : RADAR LOGIQUE (32x28) ---
        try:
            vram = list(map(int, responses[0].split(",")))
            for idx, val in enumerate(vram):
                if idx >= 1024: break
                r = idx % 32
                c = 27 - ((idx // 32) - 2)
                if 0 <= r < 32 and 0 <= c < 28:
                    if val in [64, 16, 20, 0xBF]: # Couloirs/Pastilles
                        color = (0, 0, 80) if val == 64 else (10, 10, 10)
                        if val in [16, 20]:
                            pygame.draw.circle(self.radar_screen, (255, 255, 0), (c*10+5, r*10+5), 2 if val == 16 else 4)
                        else:
                            pygame.draw.rect(self.radar_screen, color, (c*10, r*10, 9, 9))
            
            py, px = int(responses[1]), int(responses[2])
            pr, pc = py // 8, (27 - (px // 8) + 2) % 28
            pygame.draw.circle(self.radar_screen, (255, 255, 255), (pc*10+5, pr*10+5), 5)
            
            for i in range(4):
                gy, gx = int(responses[3+i*2]), int(responses[4+i*2])
                state = int(responses[11+i])
                gr, gc = gy // 8, (27 - (gx // 8) + 2) % 28
                color = (255, 0, 0) if state == 0 else ((0, 255, 255) if state == 1 else (255, 180, 0))
                pygame.draw.rect(self.radar_screen, color, (gc*10+1, gr*10+1, 8, 8))
        except Exception:
            pass

        # --- PARTIE DROITE : VISION IA MULTI-CANAUX (4 x 32 x 28) ---
        start_x = 280
        if next_frame is not None:
            if next_frame.ndim == 3 and next_frame.shape[0] >= 4:
                for r in range(32):
                    for c in range(28):
                        ch0 = next_frame[0, r, c] # Murs (0.2), Pastilles (0.5), Power (1.0)
                        ch1 = next_frame[1, r, c] # Pacman (1.0)
                        ch2 = next_frame[2, r, c] # Fantôme mortel (1.0)
                        ch3 = next_frame[3, r, c] # Fantôme bleu (1.0), Clignotant (0.5)
                        if ch3 > 0.0:
                            col = (0, 220, 255) if ch3 > 0.7 else (255, 165, 0)
                        elif ch1 >= 0.9:
                            col = (255, 255, 0) # Pac-Man
                        elif ch2 > 0.0:
                            col = (255, 40, 40) # Fantôme dangereux
                        elif ch0 >= 0.9:
                            col = (255, 120, 255) # Power Pellet
                        elif ch0 >= 0.4:
                            col = (180, 180, 120) # Pastille
                        elif ch0 > 0.1:
                            col = (30, 40, 90) # Mur
                        else:
                            col = (12, 12, 18) # Couloir vide
                        pygame.draw.rect(self.radar_screen, col, (start_x + c*10, r*10, 9, 9))
            elif next_frame.ndim == 2:
                for r in range(next_frame.shape[0]):
                    for c in range(next_frame.shape[1]):
                        val = int(next_frame[r, c] * 255)
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

    def launch_mame(self, visible=True, video_driver="opengl"):
        # --- CONFIGURATION MAME ---
        MAME_BIN_PATH = "D:\\Emulateurs\\Mame Officiel\\mame.exe"
        MAME_DIR = os.path.dirname(MAME_BIN_PATH)
        LUA_SCRIPT_PATH = os.path.join(CORE_DIR, "PythonBridgeSocket.lua").replace('\\', '/')

        command = [
            MAME_BIN_PATH,
            "-window", "-resolution", "448x576",
            "-skip_gameinfo",
            "-artwork_crop",
            "-video", video_driver,
            "-sound", "none",
            "-nowriteconfig",
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

    def run(self, session=None, model_path=None, record_enabled=True, watch_speed=None, video_driver="opengl"):
        # Config
        RESUME = True # Nouvelle session "Elite" (Repart de zéro avec nouvelle architecture)
        model_type = "cnn" # "cnn" ou "mlp" - MLP recommandé pour Pacman (VRAM indices)
        N = 4 # Stack size (Standard Atari pour bien capter le mouvement)
        NB_DE_FRAMES_STEP = 3 # Synchronisé avec pacman_multi.py (vitesse d'apprentissage)
        TRAIN_EVERY_N_GLOBAL_STEPS = 4 # Entraîner tous les 4 steps pour ne pas tuer le CPU/GPU
        # Calcul du nombre de messages par step pour la synchro wait_for
        # wait_for (1) + get_all_data_batched(21) + execute_action(4) = 26 total lines
        NB_DE_DEMANDES_PAR_STEP = 25
        
        model_filename = os.path.join(SCRIPT_DIR, "pacman_apex.pth")
        best_model_filename = os.path.join(SCRIPT_DIR, "pacman_best.pth")
        
        if model_type == "cnn":
            # (N, H, W)
            input_size = (7, 32, 28)
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
            gamma=0.99, # Synchronisé avec pacman_multi.py (Rainbow Standard)
            use_noisy=use_noisy,
            buffer_capacity=10 if watch_speed is not None else 500000,
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
            optimize_memory=True,
            mode="exploitation"
        )

        print(f"Configuration Input Size: {input_size} (Model: {model_type})")
        
        # Recorder & OBS
        record = record_enabled
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
            else:
                recorder.ensure_pacman_target()

        self.launch_mame(visible=True, video_driver=video_driver)
        comm = MameCommunicator("localhost", 12346) # Port défini dans Lua pour Pacman
        game = PacmanInterface(comm)
        config.state_extractor = StateExtractor(game, model_type)
        
        trainer = DQNTrainer(config)
        self.setup_keyboard(trainer, config, comm)
        
        # Load if exists
        selected_model = None
        if model_path and os.path.exists(model_path):
            selected_model = model_path
        elif session is not None:
            candidates = [
                os.path.join(SCRIPT_DIR, f"pacman_best_session{session}.pth"),
                os.path.join(SCRIPT_DIR, f"pacman_best_{session}.pth"),
                os.path.join(SCRIPT_DIR, f"pacman_best_4535_s{session}.pth"),
                best_model_filename,
            ]
            for cand in candidates:
                if os.path.exists(cand):
                    selected_model = cand
                    break
            print(f"{Fore.CYAN}🎯 Session {session} demandée -> Modèle sélectionné : {selected_model}{Style.RESET_ALL}")
        elif RESUME:
            if os.path.exists(best_model_filename):
                selected_model = best_model_filename
            elif os.path.exists(model_filename):
                selected_model = model_filename

        if selected_model and os.path.exists(selected_model):
            print(f"{Fore.GREEN}🧠 Chargement du modèle : {selected_model}{Style.RESET_ALL}")
            trainer.load_model(selected_model)
        else:
            print(f"{Fore.YELLOW}⚠️ Aucun modèle pré-existant trouvé, démarrage à zéro.{Style.RESET_ALL}")

        buffer_filename = os.path.join(SCRIPT_DIR, "pacman.buffer")
        if RESUME and config.mode.lower() == "exploration" and os.path.exists(buffer_filename):
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
        if watch_speed is not None:
            comm.communicate([
                f"execute throttle_rate({float(watch_speed)})",
                "execute throttled(1)",
                f"frame_per_step {NB_DE_FRAMES_STEP}"
            ])
        else:
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
                current_state_stack = game.build_obs_stack(pacman_loop_frame_history)
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
                
                # --- REWARD SHAPING & EPISODIC LIFE (Synchronisé avec pacman_multi.py) ---
                last_pills = getattr(self, "_last_pills", pills)
                reward, life_lost = game.compute_reward(new_score, score, new_lives, lives, pills, last_pills)
                self._last_pills = pills
                if life_lost:
                    lives = new_lives
                
                score = new_score
                sum_rewards += reward
                
                # Update Stack
                pacman_loop_frame_history.append(next_frame)
                if model_type == "cnn":
                    next_state_stack = game.build_obs_stack(pacman_loop_frame_history)
                else:
                    next_state_stack = np.concatenate(pacman_loop_frame_history)
                
                # Check Game Over (Fin de la partie complète)
                if alive == 0 and lives == 0:
                    done = True
                
                # Episodic Life : done_transition est True dès qu'une vie est perdue ou Game Over
                done_transition = bool(done or life_lost)
                
                # Store & Train
                if config.mode.lower() == "exploration":
                    if config.nstep:
                        nstep_tr = trainer.nstep_wrapper.append(current_state_stack, action, reward, done_transition, next_state_stack)
                        if nstep_tr:
                            trainer.replay_buffer.push(*nstep_tr)
                        if done_transition:
                            for tr in trainer.nstep_wrapper.flush():
                                trainer.replay_buffer.push(*tr)
                    else:
                        trainer.replay_buffer.push(current_state_stack, action, reward, next_state_stack, done_transition)
                
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
                
                # Gestion mort (attente respawn + reset propre de la pile de frames)
                if (alive == 0 or life_lost) and lives > 0 and not done:
                    if self.debug >= 1: print("[DEBUG] Mort! alive=",alive,"-lives=",lives)
                    comm.communicate(["wait_for 6"])
                    while alive == 0:
                        for event in pygame.event.get():
                            if event.type == pygame.QUIT: self.flag_quit = True
                        data = game.get_score_and_lives()
                        if data: _, lives, alive, pills = data
                        if self.debug >= 1: print("[DEBUG] boucle while (mort) alive=",alive,"lives=",lives)
                        if lives == 0:
                            done = True
                            break
                    if not done:
                        self._last_pills = pills
                        pacman_loop_frame_history.clear()
                        for _ in range(N):
                            s, _ = config.state_extractor()
                            pacman_loop_frame_history.append(s)
                        if model_type == "cnn":
                            current_state_stack = game.build_obs_stack(pacman_loop_frame_history)
                        else:
                            current_state_stack = np.concatenate(pacman_loop_frame_history)
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
            
            session_info = (
                f"Pac-Man Single Actor ({model_type.upper()}) | "
                f"Arch: {config.cnn_type.upper() if model_type == 'cnn' else 'MLP'} | "
                f"LR: {config.learning_rate} | Gamma: {config.gamma} | "
                f"Buffer: {config.buffer_capacity//1000}k"
            )

            if episode % 10 == 0:
                trainer.save_model(model_filename)
                Visualizer.create_fig(episode, mean_scores, 100, epsilons, rewards_hist, steps_hist, os.path.join(MEDIA_DIR, "Pacman_fig"), high_score, label_curve="Sigma" if use_noisy else "Epsilon", max_scores=max_scores_hist, session_info=session_info)
            
            if episode % 100 == 0:
                trainer.save_buffer(os.path.join(SCRIPT_DIR, "pacman.buffer"))

            if self.flag_create_fig:
                Visualizer.create_fig(episode, mean_scores, 100, epsilons, rewards_hist, steps_hist, os.path.join(MEDIA_DIR, "Pacman_fig_manual"), high_score, label_curve="Sigma" if use_noisy else "Epsilon", max_scores=max_scores_hist, session_info=session_info)
                self.flag_create_fig = False

        trainer.save_model(model_filename)
        
        # Sauvegarde du meilleur modèle avec stats dans le nom sur demande F2 (ou fin)
        if os.path.exists(best_model_filename):
            base_name = f"pacman_{model_type}_best_{episode}ep_HS{high_score}"
            best_with_stats = os.path.join(SCRIPT_DIR, f"{base_name}.pth")
            try:
                shutil.copy(best_model_filename, best_with_stats)
                print(f"{Fore.CYAN}🏆 Meilleur modèle sauvegardé sous : {best_with_stats}{Style.RESET_ALL}")
                Visualizer.create_fig(episode, mean_scores, 100, epsilons, rewards_hist, steps_hist, base_name, high_score, label_curve="Sigma" if use_noisy else "Epsilon", max_scores=max_scores_hist, session_info=session_info)
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
    import argparse
    parser = argparse.ArgumentParser(description="Pac-Man Single Actor / Visualizer")
    parser.add_argument("--session", type=int, default=None, help="Numéro de la session à charger (ex: 43)")
    parser.add_argument("--model", type=str, default=None, help="Chemin d'un fichier de poids .pth")
    parser.add_argument("--no-record", action="store_true", help="Désactiver l'enregistrement OBS")
    parser.add_argument("--speed", type=float, default=None, help="Vitesse d'affichage MAME throttled (ex: 1.0 ou 1.5)")
    parser.add_argument("--video", type=str, default="opengl", help="Driver vidéo MAME (opengl, gdi, bgfx, auto)")
    args = parser.parse_args()

    app = PacmanApp()
    app.run(session=args.session, model_path=args.model, record_enabled=not args.no_record, watch_speed=args.speed, video_driver=args.video)
