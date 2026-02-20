"""
pacman_robot.py

Bot algorithmique déterministe pour Pac-Man sur MAME.
Utilise A*/BFS pour la navigation et la lecture mémoire directe pour l'état du jeu.

Auteur: Gemini Code Assist
"""

import os
import time
import subprocess
import numpy as np
import heapq
from collections import deque
import traceback

# Import du communicateur socket (doit être dans le même dossier)
try:
    from MameCommSocket import MameCommunicator
except ImportError:
    print("ERREUR: MameCommSocket.py manquant.")
    exit()

# ==================================================================================================
# CONSTANTES & ADRESSES MÉMOIRE (Identiques à pacman.py)
# ==================================================================================================

class Memory:
    # Adresses principales
    SCORE_10        = "4E80"  # Score : Chiffre des Dizaines (0-9)
    SCORE_100       = "4E81"  # Score : Chiffre des Centaines (0-9)
    SCORE_1000      = "4E82"  # Score : Chiffre des Milliers (0-9)
    SCORE_10000     = "4E83"  # Score : Chiffre des Dizaines de milliers (0-9)
    CREDITS         = "4E6E"  # Nombre de crédits (pièces insérées)
    LIVES           = "4E14"
    PLAYER_ALIVE    = "4EAE" # >00 = Normal
    
    # Positions (Logic)
    # Note: Sur le hardware Pacman, l'écran est tourné.
    # X (4D08) correspond souvent à la position verticale (Lignes)
    # Y (4D09) correspond à la position horizontale (Colonnes)
    BLINKY_X        = "4D00"
    BLINKY_Y        = "4D01"
    PINKY_X         = "4D02"
    PINKY_Y         = "4D03"
    INKY_X          = "4D04"
    INKY_Y          = "4D05"
    CLYDE_X         = "4D06"
    CLYDE_Y         = "4D07"
    PACMAN_X        = "4D08"
    PACMAN_Y        = "4D09"
    
    # États des fantômes
    # 0=Normal, 1=Blue, 2=Flash, 4=Eyes
    GHOST_STATE_BLINKY = "4DA7"
    GHOST_STATE_PINKY  = "4DA8"
    GHOST_STATE_INKY   = "4DA9"
    GHOST_STATE_CLYDE  = "4DAA"
    
    # Sprites (Pour les fruits)
    FRUIT_X         = "4D0A" # Sprite 5
    FRUIT_Y         = "4D0B"
    
    # VRAM (Carte)
    VRAM_START      = "4000"
    VRAM_LEN        = 1024 # 32x32 tuiles

class GameConstants:
    # Mapping des tuiles VRAM (Valeurs approximatives basées sur le hardware Namco)
    TILE_EMPTY = 0x40
    TILE_PILL  = 0x10
    TILE_POWER = 0x14
    # Tout ce qui n'est pas ci-dessus est considéré comme un mur pour la navigation
    
    # Dimensions de la grille logique (VRAM est 32x32, mais le jeu utilise ~28x31)
    GRID_W = 32
    GRID_H = 32
    TILE_SIZE = 8

# ==================================================================================================
# LOGIQUE DU ROBOT
# ==================================================================================================

class PacmanRobot:
    def __init__(self, host="127.0.0.1", port=12346):
        self.comm = MameCommunicator(host=host, port=port)
        self.grid = np.zeros((GameConstants.GRID_H, GameConstants.GRID_W), dtype=int)
        self.pacman_pos = (0, 0) # (Row, Col)
        self.fruit_pos = None
        self.last_pos = (0, 0)
        self.stuck_counter = 0
        self.raw_px = 0
        self.raw_py = 0
        self.ghosts = []
        self.score = 0
        self.lives = 0
        self.last_dir = None
        self.debug = False
        # Analyse
        self.death_history = []
        self.steps_survived = 0
        self.last_valid_state = None
        self.avoid_positions = set() # Zones à éviter basées sur l'expérience

    def read_memory(self):
        """Lit toute la mémoire nécessaire en une seule requête."""
        cmds = [
            f"read_memory_range {Memory.VRAM_START}({Memory.VRAM_LEN})", # 0
            f"read_memory {Memory.PACMAN_X}", # 1
            f"read_memory {Memory.PACMAN_Y}", # 2
            f"read_memory {Memory.BLINKY_X}", f"read_memory {Memory.BLINKY_Y}", # 3, 4
            f"read_memory {Memory.PINKY_X}",  f"read_memory {Memory.PINKY_Y}",  # 5, 6
            f"read_memory {Memory.INKY_X}",   f"read_memory {Memory.INKY_Y}",   # 7, 8
            f"read_memory {Memory.CLYDE_X}",  f"read_memory {Memory.CLYDE_Y}",  # 9, 10
            f"read_memory {Memory.GHOST_STATE_BLINKY}", # 11
            f"read_memory {Memory.GHOST_STATE_PINKY}",  # 12
            f"read_memory {Memory.GHOST_STATE_INKY}",   # 13
            f"read_memory {Memory.GHOST_STATE_CLYDE}",  # 14
            f"read_memory {Memory.LIVES}",              # 15
            f"read_memory {Memory.PLAYER_ALIVE}",       # 16
            f"read_memory {Memory.FRUIT_X}",            # 17
            f"read_memory {Memory.FRUIT_Y}",            # 18
            f"read_memory {Memory.SCORE_10}",           # 19
            f"read_memory {Memory.SCORE_100}",          # 20
            f"read_memory {Memory.SCORE_1000}",         # 21
            f"read_memory {Memory.SCORE_10000}"         # 22
        ]
        
        response = self.comm.communicate(cmds)
        if not response or len(response) < 23:
            return False

        try:
            # 1. Parsing VRAM (Carte)
            # La VRAM Namco est organisée en colonnes de 32 tuiles.
            # Adresse = Base + (Col * 32) + Row
            # On va la transformer en grille [Row][Col] pour A*
            raw_vram = list(map(int, response[0].split(",")))
            self.grid.fill(1) # 1 = Mur par défaut
            
            for idx, val in enumerate(raw_vram):
                if idx >= 1024: break
                
                # Correction Mapping VRAM (Namco Hardware: Col-Major)
                # Index = Col * 32 + Row
                # On aligne cols 2-29 (Jeu) vers 0-27 (Grille). Cols 0,1,30,31 sont hors écran.
                
                r = idx % 32
                c = 29 - (idx // 32)
                
                # Ignorer les colonnes hors écran pour éviter le débordement à gauche
                if c < 0 or c >= 28:
                    continue
                
                # Identification du type de tuile
                if val == GameConstants.TILE_EMPTY or val == GameConstants.TILE_PILL or val == GameConstants.TILE_POWER or val == 0:
                    self.grid[r, c] = 0 # 0 = Walkable
                
                # On marque les pastilles spécifiquement pour le pathfinding
                if val == GameConstants.TILE_PILL:
                    self.grid[r, c] = 2 # 2 = Pill
                elif val == GameConstants.TILE_POWER:
                    self.grid[r, c] = 3 # 3 = Power

            # 2. Parsing Positions
            # Conversion Coordonnées Pixel (0-255) -> Coordonnées Tuile (0-31)

            # Pacman
            px_raw, py_raw = int(response[1]), int(response[2])
            self.raw_px = px_raw
            self.raw_py = py_raw
            # Hypothèse: X (4D08) = Row, Y (4D09) = Col
            # Verticalement OK -> Row = px // 8
            # Horizontalement: Miroir (29 - ...) pour aligner avec la grille inversée
            p_row = px_raw // 8
            p_col = 29 - (py_raw // 8)
            
            # Snap to Grid: Si Pacman est dans un mur (imprécision), on cherche le voisin libre
            if 0 <= p_row < 32 and 0 <= p_col < 32 and self.grid[p_row, p_col] == 1:
                 found = False
                 for dr in [0, -1, 1]:
                     for dc in [0, -1, 1]:
                         nr, nc = p_row + dr, p_col + dc
                         if 0 <= nr < 32 and 0 <= nc < 32 and self.grid[nr, nc] != 1:
                             p_row, p_col = nr, nc
                             found = True
                             break
                     if found: break
            
            # Sécurité bornes
            p_row = max(0, min(31, p_row))
            p_col = max(0, min(31, p_col))
            
            self.pacman_pos = (p_row, p_col)

            # Fantômes
            self.ghosts = []
            for i in range(4):
                gx_raw = int(response[3 + i*2])
                gy_raw = int(response[4 + i*2])
                state = int(response[11 + i])
                
                g_pos = (gx_raw // 8, 29 - (gy_raw // 8))
                
                # Dangerosité
                is_dangerous = True
                if state == 1 or state == 2: is_dangerous = False # Blue/Flash
                if state >= 4: is_dangerous = False # Eyes (retourne à la base)
                
                self.ghosts.append({
                    'pos': g_pos,
                    'dangerous': is_dangerous,
                    'state': state
                })

            self.lives = int(response[15])
            self.alive = int(response[16])
            
            # Fruit (Sprite 5)
            fx_raw, fy_raw = int(response[17]), int(response[18])
            f_row = fx_raw // 8
            f_col = 29 - (fy_raw // 8)
            if 0 <= f_row < 32 and 0 <= f_col < 32 and (fx_raw > 0 or fy_raw > 0):
                self.fruit_pos = (f_row, f_col)
            else:
                self.fruit_pos = None
            
            # Score (Reconstruction depuis les digits)
            try:
                # Logique originale demandée (BCD via hex string) 
                dizaines = int(response[19])
                centaines = int(response[20])
                dizaines_de_milliers = int(response[21])
                dizaines = int(hex(dizaines)[2:])
                centaines = int(hex(centaines)[2:])
                dizaines_de_milliers = int(hex(dizaines_de_milliers)[2:])
                
                score = dizaines + centaines * 100 + dizaines_de_milliers * 10000
                self.score = score
            except: pass

            return True

        except Exception as e:
            print(f"Erreur parsing mémoire: {e}")
            return False

    def get_neighbors(self, pos, prioritize_dir=None):
        """Retourne les voisins valides (pas des murs) pour une position (r, c)."""
        r, c = pos
        moves = [(-1, 0, 'up'), (1, 0, 'down'), (0, -1, 'left'), (0, 1, 'right')]
        
        # Stabilisation: On privilégie la direction actuelle pour éviter les oscillations
        if prioritize_dir:
            for i, m in enumerate(moves):
                if m[2] == prioritize_dir:
                    moves.insert(0, moves.pop(i))
                    break
        
        valid = []
        
        for dr, dc, direction in moves:
            nr, nc = r + dr, c + dc
            
            # Correction Tunnel: Le jeu utilise les colonnes 0 à 27.
            # Wrapping horizontal pour le tunnel (Gauche <-> Droite)
            if nc < 0: nc = 27
            elif nc >= 28: nc = 0
            
            # Vérification bornes verticales
            if nr < 0: nr = GameConstants.GRID_H - 1
            if nr >= GameConstants.GRID_H: nr = 0
            
            # Vérification Mur (1 = Mur)
            if self.grid[nr, nc] != 1:
                valid.append(((nr, nc), direction))
        return valid

    def get_tunnel_rows(self):
        """Identifie les lignes qui sont des tunnels (ouvertes aux deux extrémités)."""
        rows = []
        for r in range(GameConstants.GRID_H):
            if self.grid[r, 0] != 1 and self.grid[r, 27] != 1:
                rows.append(r)
        return rows

    def bfs_find_best_move(self):
        """
        Algorithme principal de décision.
        1. Marque les zones dangereuses autour des fantômes.
        2. Cherche le chemin le plus court vers une pastille/power pellet.
        3. Si aucun chemin sûr, cherche le point le plus éloigné des fantômes.
        """
        start = self.pacman_pos
        
        # 1. Création de la carte de danger
        danger_zone = set()
        ghost_positions = []
        SAFE_DIST = 5 # Augmenté pour anticiper la fuite
        
        for ghost in self.ghosts:
            if ghost['dangerous']:
                gr, gc = ghost['pos']
                ghost_positions.append((gr, gc))
                
                # BFS local pour marquer la zone dangereuse (Radius SAFE_DIST)
                q_danger = deque([(gr, gc, 0)])
                visited_danger = {(gr, gc)}
                danger_zone.add((gr, gc))
                
                while q_danger:
                    r, c, dist = q_danger.popleft()
                    if dist >= SAFE_DIST: continue
                    
                    for n_pos, _ in self.get_neighbors((r, c)): # Pas de prio ici, c'est pour le danger
                        if n_pos not in visited_danger:
                            visited_danger.add(n_pos)
                            danger_zone.add(n_pos)
                            q_danger.append((n_pos[0], n_pos[1], dist + 1))

        # Ajout des zones apprises comme dangereuses (Mémoire des morts)
        for pos in self.avoid_positions:
            danger_zone.add(pos)

        # Si Pacman est DANS la zone de danger immédiat, mode panique
        if start in danger_zone:
            return self.escape_ghosts(start, ghost_positions)

        # 2. BFS pour trouver la pastille la plus proche (Target)
        queue = deque([(start, [])]) # (Position, Chemin [dir1, dir2...])
        visited = {start}
        
        nearest_pill_path = None
        tunnel_rows = self.get_tunnel_rows()
        
        # Optimisation: Limiter la profondeur si la grille est grande, mais 32x32 ça va.
        while queue:
            curr, path = queue.popleft()
            r, c = curr
            
            # Priorité Absolue: Fantôme Bleu (Miam !)
            for g in self.ghosts:
                if g['pos'] == curr and g['state'] == 1: # Blue
                    if path: return path[0]
                    return None # Déjà dessus
            
            # Priorité 2: Fruit (Miam !)
            if self.fruit_pos and curr == self.fruit_pos:
                if path: return path[0]
                return None
            
            # Priorité Secondaire: Pastille ou Power Pellet
            # On enregistre le chemin mais on continue de chercher un fantôme bleu
            if (self.grid[r, c] in [2, 3]) and nearest_pill_path is None:
                nearest_pill_path = path
            
            # Si on a trouvé une pastille, on continue un peu (5 cases) pour voir si un fantôme bleu est proche
            # Sinon on valide la pastille pour éviter de chercher trop longtemps
            if nearest_pill_path and len(path) > len(nearest_pill_path) + 5:
                return nearest_pill_path[0]
            
            # Exploration voisins
            # On privilégie la direction actuelle seulement au départ pour la stabilité
            neighbors = self.get_neighbors(curr, self.last_dir if curr == start else None)
            for neighbor, direction in neighbors:
                # ANTI-DEMI-TOUR TUNNEL : Interdit de faire demi-tour si on est dans le tunnel
                if curr == start and r in tunnel_rows:
                    if self.last_dir in ['left', 'right']:
                        if direction == {'left': 'right', 'right': 'left'}.get(self.last_dir):
                            continue

                if neighbor not in visited and neighbor not in danger_zone:
                    visited.add(neighbor)
                    new_path = list(path)
                    new_path.append(direction)
                    queue.append((neighbor, new_path))
        
        # Si on a trouvé une pastille et pas de fantôme bleu
        if nearest_pill_path:
            return nearest_pill_path[0]
        
        # 3. Fallback: Si aucun chemin vers une pastille n'est trouvé (bloqué par fantômes)
        # On essaie juste de survivre en s'éloignant des fantômes
        return self.escape_ghosts(start, ghost_positions)

    def escape_ghosts(self, start, ghost_positions):
        """
        Trouve le mouvement qui mène à la case la plus sûre (la plus loin des fantômes)
        en utilisant un BFS pour éviter les culs-de-sac.
        """
        # BFS pour trouver toutes les cases accessibles en N étapes
        queue = deque([(start, [])])
        visited = {start}
        max_depth = 30 # Augmenté pour détecter le tunnel de plus loin
        
        best_path = None
        max_safety_score = -float('inf')
        tunnel_rows = self.get_tunnel_rows()
        
        while queue:
            curr, path = queue.popleft()
            
            if path:
                # 1. Distance au fantôme le plus proche depuis cette case
                min_ghost_dist = float('inf')
                for g_pos in ghost_positions:
                    # Distance Manhattan avec prise en compte du Tunnel (Wrap horizontal)
                    d_row = abs(curr[0] - g_pos[0])
                    d_col = abs(curr[1] - g_pos[1])
                    d_col = min(d_col, 28 - d_col) # Distance la plus courte via les bords
                    d = d_row + d_col
                    if d < min_ghost_dist: min_ghost_dist = d
                
                # 2. Score de sécurité
                # On privilégie la distance aux fantômes.
                safety_score = min_ghost_dist
                
                # Bonus pour les intersections (plus d'options de fuite)
                if len(self.get_neighbors(curr)) > 2:
                    safety_score += 1
                
                # Bonus Tunnel (Pacman rapide, Fantômes lents)
                # Row 14 est le tunnel. On vise les entrées (cols 0-2 et 25-27)
                if curr[0] == 14 and (curr[1] <= 2 or curr[1] >= 25):
                    safety_score += 10 # Priorité absolue au tunnel en cas de danger
                
                # Malus Pastilles (Pacman ralentit en mangeant)
                # On préfère fuir par les couloirs vides (vitesse max)
                if self.grid[curr[0], curr[1]] == 2: # 2 = Pill
                    safety_score -= 2 
                
                if safety_score > max_safety_score:
                    max_safety_score = safety_score
                    best_path = path
            
            if len(path) >= max_depth:
                continue
            
            for neighbor, direction in self.get_neighbors(curr, self.last_dir if curr == start else None):
                # ANTI-DEMI-TOUR TUNNEL (Aussi en mode fuite)
                if curr == start and curr[0] in tunnel_rows:
                    if self.last_dir in ['left', 'right']:
                        if direction == {'left': 'right', 'right': 'left'}.get(self.last_dir):
                            continue

                if neighbor not in visited:
                    visited.add(neighbor)
                    new_path = list(path)
                    new_path.append(direction)
                    queue.append((neighbor, new_path))
        
        if best_path:
            return best_path[0]
            
        # Fallback: Mouvement aléatoire valide si coincé
        valid = self.get_neighbors(start, self.last_dir)
        return valid[0][1] if valid else None

    def record_death(self):
        """Enregistre les détails de la mort pour analyse."""
        if self.last_valid_state:
            print(f"\n💀 MORT DÉTECTÉE ! Score: {self.last_valid_state['score']}")
            self.death_history.append(self.last_valid_state)
            # Analyse immédiate
            pm_pos = self.last_valid_state['pacman']
            for i, g in enumerate(self.last_valid_state['ghosts']):
                g_pos = g['pos']
                dist = abs(pm_pos[0] - g_pos[0]) + abs(pm_pos[1] - g_pos[1])
                if dist <= 1:
                    print(f"   -> Collision probable avec Fantôme {i} à {g_pos} (Pacman: {pm_pos})")

    def analyze_game(self):
        """Affiche un rapport complet avant de recommencer."""
        print("\n" + "█"*60)
        print(f"📊 RAPPORT DE PARTIE - Score Final: {self.score}")
        print("█"*60)
        print(f"Steps survécus: {self.steps_survived} | Morts: {len(self.death_history)}")
        if self.death_history:
            print("\n📍 Historique des décès :")
            for idx, d in enumerate(self.death_history):
                print(f" {idx+1}. Pos: {d['pacman']} | Score: {d['score']}")
                # Apprentissage : On marque la position de la mort comme zone à éviter
                self.avoid_positions.add(d['pacman'])
                # On ajoute aussi les voisins pour créer une marge de sécurité
                for n, _ in self.get_neighbors(d['pacman']):
                    self.avoid_positions.add(n)
        
        # Limiteur de mémoire pour éviter de bloquer toute la carte
        if len(self.avoid_positions) > 60:
            self.avoid_positions.clear()

        self.death_history = []
        self.steps_survived = 0

    def execute_move(self, direction):
        """Envoie les commandes Lua."""
        
        # Mapping direction -> Lua commands
        # Note: Pacman continue d'avancer, il faut juste changer quand nécessaire.
        # Mais envoyer en continu assure que le virage est pris dès que possible.
        
        cmds = [
            "execute P1_Up(0)", "execute P1_Down(0)", 
            "execute P1_Left(0)", "execute P1_Right(0)"
        ]
        
        if direction == 'up':    cmds.append("execute P1_Up(1)")
        elif direction == 'down':  cmds.append("execute P1_Down(1)")
        elif direction == 'left':  cmds.append("execute P1_Left(1)")
        elif direction == 'right': cmds.append("execute P1_Right(1)")
        else: cmds.append("execute P1_Up(0)") # Dummy pour maintenir le compte à 5
        
        self.comm.communicate(cmds)
        if direction: self.last_dir = direction

    def print_debug_info(self):
        """Affiche les infos de débogage complètes."""
        r, c = self.pacman_pos
        print(f"\n[DEBUG] Step Info:")
        print(f"  Pacman Raw: ({self.raw_px}, {self.raw_py}) -> Tile: ({r}, {c})")
        print(f"  Grid val at Pacman: {self.grid[r, c]} (0=Walk, 1=Wall, 2=Pill)")
        
        # Neighbors
        neighbors = self.get_neighbors((r, c))
        print(f"  Neighbors: {neighbors}")
        
        # Local Grid
        print("  Local Map (7x7):")
        for dr in range(-3, 4):
            line = f"    R{r+dr:02d}: "
            for dc in range(-3, 4):
                nr, nc = r + dr, c + dc
                if 0 <= nr < 32 and 0 <= nc < 32:
                    val = self.grid[nr, nc]
                    char = '.' if val == 0 else '#' if val == 1 else 'o'
                    if (nr, nc) == (r, c): char = 'P'
                    elif any(g['pos'] == (nr, nc) for g in self.ghosts): char = 'G'
                    line += f"{char} "
                else:
                    line += "X "
            print(line)

    def print_debug_map(self):
        """Affiche une carte ASCII dans la console."""
        # os.system('cls' if os.name == 'nt' else 'clear')
        print(f"Score: {self.score} | Lives: {self.lives} | Pos: {self.pacman_pos}")
        
        display_grid = [[' ' for _ in range(GameConstants.GRID_W)] for _ in range(GameConstants.GRID_H)]
        
        # Murs et Pastilles
        for r in range(GameConstants.GRID_H):
            for c in range(GameConstants.GRID_W):
                if self.grid[r, c] == 1: display_grid[r][c] = '#'
                elif self.grid[r, c] == 2: display_grid[r][c] = '.'
                elif self.grid[r, c] == 3: display_grid[r][c] = 'O'
        
        # Fantômes
        for i, g in enumerate(self.ghosts):
            r, c = g['pos']
            if 0 <= r < GameConstants.GRID_H and 0 <= c < GameConstants.GRID_W:
                char = 'G' if g['dangerous'] else 'b'
                display_grid[r][c] = char
        
        # Pacman
        pr, pc = self.pacman_pos
        if 0 <= pr < GameConstants.GRID_H and 0 <= pc < GameConstants.GRID_W:
            display_grid[pr][pc] = 'C'
            
        # Affichage
        for row in display_grid:
            print("".join(row)) # Pas d'inversion, la grille est maintenant dans le bon sens

    def run(self):
        print("🤖 Démarrage Pacman Robot...")
        
        time.sleep(5)
        # Init MAME
        self.comm.communicate([
            f"write_memory {Memory.CREDITS}(1)",
            "execute P1_start(1)",
            f"execute throttle_rate({10})", # Vitesse max
            "execute throttled(0)",
            f"frame_per_step {1}"
        ])
        
        step = 0
        was_alive = False
        while True:
            try:
                self.comm.communicate([f"wait_for {24}"])
                # 1. Lecture État
                if not self.read_memory():
                    print("Erreur lecture mémoire. MAME est-il lancé ?")
                    time.sleep(1)
                    continue
                
                # Affichage de la carte complète au démarrage pour débogage
                if step == 0 and self.debug:
                    print("\n[DEBUG] --- FULL MAP DUMP ---")
                    self.print_debug_map()
                    print("[DEBUG] --- END MAP DUMP ---\n")
                    time.sleep(3)
                
                # Tracking état vivant pour analyse
                if self.alive != 0:
                    self.last_valid_state = {
                        "pacman": self.pacman_pos,
                        "ghosts": [g.copy() for g in self.ghosts],
                        "score": self.score
                    }
                    was_alive = True
                    self.steps_survived += 1
                elif was_alive:
                    # Transition Vivant -> Mort
                    self.record_death()
                    was_alive = False

                # Attente que le joueur soit en vie (début de partie ou respawn)
                if self.alive == 0:
                    # Si plus de vies, on réinsère une pièce
                    if self.lives == 0:
                        # ANALYSE AVANT RESTART
                        self.analyze_game()
                        
                        self.comm.communicate(["wait_for 0"])
                        print("Game Over. Restarting...")
                        self.comm.communicate([
                        f"write_memory {Memory.CREDITS}(1)"
                        ])
                        time.sleep(1)
                    continue

                # 2. Décision
                # Détection de blocage (Stuck)
                if self.pacman_pos == self.last_pos:
                    self.stuck_counter += 1
                else:
                    self.stuck_counter = 0
                    self.last_pos = self.pacman_pos
                
                if self.stuck_counter > 20: # Bloqué depuis ~0.3s
                    valid = self.get_neighbors(self.pacman_pos)
                    if valid:
                        import random
                        best_move = random.choice(valid)[1]
                        self.stuck_counter = 0
                    else:
                        best_move = self.bfs_find_best_move()
                else:
                    best_move = self.bfs_find_best_move()
                
                # 3. Action
                self.execute_move(best_move)
                
                # 4. Debug (toutes les 10 frames pour ne pas spammer)
                if self.debug:
                    self.print_debug_info()
                    if step % 10 == 0:
                        self.print_debug_map()
                    # time.sleep(0.1) # Ralentir pour lire
                
                step += 1
                # Pas de sleep nécessaire si MAME tourne vite, le socket bloquera naturellement
                # ou on peut limiter si le robot est trop rapide pour MAME
                # time.sleep(0.01) 

            except KeyboardInterrupt:
                print("Arrêt demandé.")
                break
            except Exception as e:
                print(f"Erreur boucle principale: {e}")
                traceback.print_exc()
                time.sleep(1)

# ==================================================================================================
# MAIN
# ==================================================================================================

def main():
    # Configuration MAME (Adapter les chemins)
    mame_exe = r"D:\Emulateurs\Mame Officiel\mame.exe"
    lua_script = r"D:\Emulateurs\Mame Officiel\plugins\PythonBridgeSocket.lua"
    
    cmd = [
        mame_exe, "pacman",
        "-window", "-resolution", "448x576", "-skip_gameinfo", "-artwork_crop", "-console", "-noautosave",
        "-autoboot_delay", "1",
        "-sound", "none",
        "-autoboot_script", lua_script
    ]
    
    print(f"Lancement MAME: {' '.join(cmd)}")
    
    try:
        # Lancement MAME en arrière-plan
        proc = subprocess.Popen(cmd, cwd=os.path.dirname(mame_exe))
        
        print("Attente initialisation MAME (15s)...")
        time.sleep(5)
        
        # Forcer la fenêtre au premier plan via win32gui
        try:
            import win32gui
            import win32con
            def enum_cb(hwnd, results):
                if "pac-man" in win32gui.GetWindowText(hwnd).lower()[0:7]:
                    try:
                        win32gui.ShowWindow(hwnd, win32con.SW_RESTORE)
                        win32gui.SetForegroundWindow(hwnd)
                    except Exception:
                        # Hack: Envoi de la touche ALT pour autoriser le changement de focus
                        try:
                            import win32com.client
                            shell = win32com.client.Dispatch("WScript.Shell")
                            shell.SendKeys('%')
                            win32gui.SetForegroundWindow(hwnd)
                        except: pass
            win32gui.EnumWindows(enum_cb, None)
            print("Fenêtre MAME mise au premier plan.")
        except Exception as e:
            print(f"Impossible de mettre la fenêtre au premier plan (nécessite pywin32): {e}")

        # Lancement Robot
        bot = PacmanRobot(port=12346) # Port par défaut pour Pacman dans les scripts précédents
        bot.run()
        
    except Exception as e:
        print(f"Erreur fatale: {e}")
    finally:
        if 'proc' in locals():
            proc.terminate()
            print("MAME arrêté.")

if __name__ == "__main__":
    main()
