"""
pacman_robot.py - Version "Reference Reversion" Améliorée
IA heuristique / planificatrice sans DQN avec évitement dynamique et Maximin Escape
"""

import os
import time
import sys
import subprocess
import numpy as np
import pygame
from collections import deque
from colorama import Fore, Style

# --- CONFIGURATION ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPT_DIR)
CORE_DIR = os.path.join(ROOT_DIR, "core")

if CORE_DIR not in sys.path:
    sys.path.append(CORE_DIR)

try:
    # pyrefly: ignore [missing-import]
    from MameCommSocket import MameCommunicator
except ImportError:
    print(f"{Fore.RED}ERREUR: MameCommSocket.py introuvable.{Style.RESET_ALL}")
    exit()

LUA_BRIDGE_PATH = os.path.join(CORE_DIR, "PythonBridgeSocket.lua").replace('\\', '/')

class PacmanRobot:
    def __init__(self):
        # Communication MAME
        self.comm = MameCommunicator(host="127.0.0.1", port=12346)
        self.grid = np.zeros((32, 28), dtype=int)
        self.pos = (0, 0)
        self.ghosts = []
        self.current_plan = []
        self.target_pos = None
        self.alive = 0
        self.score = 0
        self.high_score = 0
        
        pygame.init()
        # Fenêtre pygame plus grande pour un affichage plus propre (multiplié par 12)
        self.cell_size = 12
        self.screen = pygame.display.set_mode((28 * self.cell_size, 32 * self.cell_size))
        pygame.display.set_caption("Pacman AI - Robot Heuristique Premium")

    def get_all_data(self):
        """Lit la VRAM, les positions des sprites, les statuts et le score du jeu"""
        return self.comm.communicate([
            "read_memory_range 4000(1024)", # [0] VRAM (Grid)
            "read_memory 4D08", "read_memory 4D09", # [1][2] Pac-Man X, Y
            "read_memory 4D00", "read_memory 4D01", # [3][4] Blinky X, Y
            "read_memory 4D02", "read_memory 4D03", # [5][6] Pinky X, Y
            "read_memory 4D04", "read_memory 4D05", # [7][8] Inky X, Y
            "read_memory 4D06", "read_memory 4D07", # [9][10] Clyde X, Y
            "read_memory 4DA7", "read_memory 4DA8", # [11][12] Blinky state
            "read_memory 4DA9", "read_memory 4DAA", # [13][14] Inky state
            "read_memory 4E14",                     # [15] Lives (extra lives remaining)
            "read_memory 4EAE",                     # [16] Player Alive status
            "read_memory 4E80",                     # [17] Score 10s BCD
            "read_memory 4E81",                     # [18] Score 100s BCD
            "read_memory 4E82"                      # [19] Score 1000s BCD
        ])

    def get_dist(self, p1, p2):
        """Calcule la distance de Manhattan sur la grille Pac-Man en gérant le tunnel horizontal"""
        dr = abs(p1[0] - p2[0])
        # Tunnel horizontal aux colonnes 0 et 27
        dc = abs(p1[1] - p2[1])
        dc_wrapped = min(dc, 28 - dc)
        return dr + dc_wrapped

    def find_best_path(self):
        """
        BFS intelligent avec priorité dynamique :
        1) Fantômes comestibles (bleus)
        2) Super pastilles (Power Pellets) - avec priorité urgente si un fantôme est proche
        3) Pastilles normales
        Évite strictement les cases adjacentes ou à portée de tir des fantômes normaux.
        """
        dangerous_ghosts = [(gr, gc) for gr, gc, gs in self.ghosts if gs == 0]
        frightened_ghosts = [(gr, gc) for gr, gc, gs in self.ghosts if gs in [1, 2]]
        
        blue_present = len(frightened_ghosts) > 0

        # Initialisation du BFS
        init_ghost_dist = 999
        if dangerous_ghosts:
            init_ghost_dist = min(self.get_dist(self.pos, g) for g in dangerous_ghosts)
            
        q = deque([(self.pos, [], init_ghost_dist)])
        vis = {self.pos}
        candidates = []

        while q:
            (cr, cc), path, min_g_dist = q.popleft()
            
            # Si on a déjà un objectif fixé et qu'on l'a atteint
            if self.target_pos and (cr, cc) == self.target_pos:
                return path
            
            target_val = 0
            
            # 1) Fantômes comestibles (Bleus)
            for gr, gc in frightened_ghosts:
                if (cr, cc) == (gr, gc):
                    target_val = 1000000
                    break
            
            # 2) Super pastilles (Power Pellets)
            if target_val == 0 and self.grid[cr, cc] == 3:
                # Si un fantôme est à proximité, gober la super pastille devient urgent
                if dangerous_ghosts and min(self.get_dist(self.pos, g) for g in dangerous_ghosts) <= 6:
                    target_val = 500000
                else:
                    target_val = 5000
            
            # 3) Pastilles normales
            if target_val == 0 and not blue_present and self.grid[cr, cc] == 2:
                target_val = 100

            if target_val > 0:
                # Calcul du score avec pénalité de proximité du danger
                safety_penalty = 1.0
                if min_g_dist < 4:
                    safety_penalty = 0.2
                if min_g_dist <= 2:
                    safety_penalty = 0.01

                score = (target_val * safety_penalty) / (len(path)**1.1 + 1)
                candidates.append((score, path, (cr, cc)))
                
                if target_val >= 500000:
                    break  # Cible prioritaire absolue trouvée
                if len(candidates) > 20:
                    break

            if len(path) > 100:
                continue

            for dr, dc, dn in [(-1,0,'up'),(1,0,'down'),(0,-1,'left'),(0,1,'right')]:
                nr, nc = (cr+dr)%32, (cc+dc)%28
                
                # Éviter les murs
                if self.grid[nr, nc] == 1:
                    continue
                    
                # Éviter les demi-tours dans le même plan
                if path and ((path[-1] == 'up' and dn == 'down') or
                             (path[-1] == 'down' and dn == 'up') or
                             (path[-1] == 'left' and dn == 'right') or
                             (path[-1] == 'right' and dn == 'left')):
                    continue
                
                if (nr, nc) not in vis:
                    g_dist = 999
                    if dangerous_ghosts:
                        g_dist = min(self.get_dist((nr, nc), g) for g in dangerous_ghosts)
                    
                    # Règle de sécurité : interdiction absolue d'aller à distance <= 2 d'un fantôme
                    # lors de la planification normale, pour garder une marge de sécurité.
                    if g_dist > 2:
                        vis.add((nr, nc))
                        q.append(((nr, nc), path + [dn], min(min_g_dist, g_dist)))
        
        if candidates:
            candidates.sort(key=lambda x: x[0], reverse=True)
            self.target_pos = candidates[0][2]
            return candidates[0][1]
        
        # Aucun chemin sûr vers une cible : mode survie
        self.target_pos = None
        return self.find_escape_path()

    def find_escape_path(self):
        """
        BFS de survie (Maximin) : Recherche un chemin de longueur 15
        qui maximise la distance minimale aux fantômes dangereux.
        """
        dangerous_ghosts = [(gr, gc) for gr, gc, gs in self.ghosts if gs == 0]
        if not dangerous_ghosts:
            # Sans danger, avance sur n'importe quelle case libre
            for dr, dc, dn in [(-1,0,'up'),(1,0,'down'),(0,-1,'left'),(0,1,'right')]:
                nr, nc = (self.pos[0]+dr)%32, (self.pos[1]+dc)%28
                if self.grid[nr, nc] != 1:
                    return [dn]
            return []

        init_dist = min(self.get_dist(self.pos, g) for g in dangerous_ghosts)
        q = deque([(self.pos, [], init_dist)])
        
        best_path = []
        best_min_dist = -1
        best_final_dist = -1
        best_len = 0

        while q:
            (cr, cc), path, min_dist = q.popleft()
            
            final_dist = min(self.get_dist((cr, cc), g) for g in dangerous_ghosts)
            
            # Enregistrer le meilleur chemin trouvé
            if len(path) > best_len:
                best_len = len(path)
                best_min_dist = min_dist
                best_final_dist = final_dist
                best_path = path
            elif len(path) == best_len:
                if min_dist > best_min_dist:
                    best_min_dist = min_dist
                    best_final_dist = final_dist
                    best_path = path
                elif min_dist == best_min_dist and final_dist > best_final_dist:
                    best_final_dist = final_dist
                    best_path = path

            if len(path) >= 15:
                continue  # Horizon de survie max atteint

            for dr, dc, dn in [(-1,0,'up'),(1,0,'down'),(0,-1,'left'),(0,1,'right')]:
                nr, nc = (cr+dr)%32, (cc+dc)%28
                if self.grid[nr, nc] != 1:
                    # Pas de demi-tours dans le plan de fuite
                    if path and ((path[-1] == 'up' and dn == 'down') or
                                 (path[-1] == 'down' and dn == 'up') or
                                 (path[-1] == 'left' and dn == 'right') or
                                 (path[-1] == 'right' and dn == 'left')):
                        continue
                    
                    g_dist = min(self.get_dist((nr, nc), g) for g in dangerous_ghosts)
                    q.append(((nr, nc), path + [dn], min(min_dist, g_dist)))
                    
        if best_path:
            return best_path
            
        # Ultime recours : 1 pas maximisant la distance immédiate
        best_dist = -1
        best_move = None
        for dr, dc, dn in [(-1,0,'up'),(1,0,'down'),(0,-1,'left'),(0,1,'right')]:
            nr, nc = (self.pos[0]+dr)%32, (self.pos[1]+dc)%28
            if self.grid[nr, nc] != 1:
                dist = min(self.get_dist((nr, nc), g) for g in dangerous_ghosts)
                if dist > best_dist:
                    best_dist = dist
                    best_move = dn
        return [best_move] if best_move else []

    def draw_debug(self):
        self.screen.fill((5, 5, 15)) # Fond bleu très nuit
        sz = self.cell_size
        
        # Mise à jour du titre de la fenêtre avec les scores
        pygame.display.set_caption(f"Pacman AI - Score: {self.score} | High Score: {self.high_score}")
        
        # Dessin grille
        for r in range(32):
            for c in range(28):
                if self.grid[r, c] == 1:
                    # Murs bleus style arcade
                    pygame.draw.rect(self.screen, (0, 0, 150), (c*sz, r*sz, sz-1, sz-1))
                elif self.grid[r, c] == 2:
                    # Petites pastilles jaunes
                    pygame.draw.circle(self.screen, (255, 230, 100), (c*sz + sz//2, r*sz + sz//2), 2)
                elif self.grid[r, c] == 3:
                    # Super pastilles magentas clignotantes
                    t = time.time()
                    if int(t * 5) % 2 == 0:
                        pygame.draw.circle(self.screen, (255, 100, 255), (c*sz + sz//2, r*sz + sz//2), 5)
            
        # Tracé du plan d'action (ligne verte)
        c_p = self.pos
        points = [(c_p[1]*sz + sz//2, c_p[0]*sz + sz//2)]
        for mv in self.current_plan[:15]:
            nr, nc = c_p
            if mv == 'up' : nr -= 1
            elif mv == 'down' : nr += 1
            elif mv == 'left' : nc = (nc-1)%28
            elif mv == 'right' : nc = (nc+1)%28
            c_p = (nr%32, nc)
            points.append((c_p[1]*sz + sz//2, c_p[0]*sz + sz//2))
            
        if len(points) > 1:
            pygame.draw.lines(self.screen, (0, 255, 100), False, points, 2)
            
        # Point cible en rouge
        if self.target_pos:
            pygame.draw.circle(self.screen, (255, 50, 50), (self.target_pos[1]*sz + sz//2, self.target_pos[0]*sz + sz//2), 6, 2)
            
        # Dessin Pacman (Blanc)
        pygame.draw.circle(self.screen, (255, 255, 255), (self.pos[1]*sz + sz//2, self.pos[0]*sz + sz//2), 5)
        
        # Dessin des Fantômes
        for gr, gc, gs in self.ghosts:
            if gs == 0:
                color = (255, 0, 0) # Danger : Rouge
            elif gs in [1, 2]:
                color = (100, 180, 255) # Comestible : Bleu clair
            else:
                color = (180, 180, 180) # Yeux / Retour base : Gris
                
            pygame.draw.rect(self.screen, color, (gc*sz + 2, gr*sz + 2, sz - 4, sz - 4), 0, 3)
            # Petit repère blanc pour les fantômes comestibles
            if gs in [1, 2]:
                pygame.draw.rect(self.screen, (255, 255, 255), (gc*sz + 4, gr*sz + 4, sz - 8, sz - 8), 1, 1)
                
        pygame.display.flip()

    def run(self):
        """BOUCLE PRINCIPALE DE RÉFÉRENCE"""
        time.sleep(4)
        self.comm.communicate(["execute throttle_rate(5.0)", "execute throttled(1)"])
        
        print(f"{Fore.GREEN}🤖 Robot Pac-Man connecté et actif !{Style.RESET_ALL}")
        
        while True:
            # SÉQUENCE DÉMARRAGE : Crédits + Start
            print(f"{Fore.CYAN}🪙 Insertion d'un crédit...{Style.RESET_ALL}")
            self.comm.communicate(["write_memory 4E6E(1)"])
            
            # Attendre que la mémoire confirme le crédit
            credit_registered = False
            for _ in range(20): # max 1 seconde
                res = self.comm.communicate(["read_memory 4E6E"])
                if res and int(res[0]) > 0:
                    credit_registered = True
                    break
                time.sleep(0.05)
                
            time.sleep(0.2)
            
            in_game = False
            start_pressed_time = 0.0
            while not in_game:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        pygame.quit()
                        sys.exit(0)
                        
                res = self.comm.communicate(["read_memory 4E14", "read_memory 4EAE", "read_memory 4E6E"])
                if res and len(res) >= 3:
                    lives = int(res[0])
                    player_alive = int(res[1])
                    credits = int(res[2])
                    
                    if lives > 0 or player_alive > 0:
                        in_game = True
                        break
                    
                    # Si crédit enregistré, et bouton Start pas pressé trop fréquemment (toutes les 0.6s max)
                    if credits > 0 and time.time() - start_pressed_time > 0.6:
                        print(f"{Fore.CYAN}🚀 Envoi impulsion Start (1 Player)...{Style.RESET_ALL}")
                        self.comm.communicate(["execute P1_start(1)"])
                        time.sleep(0.1)
                        self.comm.communicate(["execute P1_start(0)"])
                        start_pressed_time = time.time()
                time.sleep(0.05)

            self.score = 0
            self.current_plan = []
            self.target_pos = None
            game_over = False
            
            print(f"{Fore.GREEN}🎮 Nouvelle partie démarrée !{Style.RESET_ALL}")
            
            while not game_over:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        game_over = True
                
                res = self.get_all_data()
                if not res or len(res) < 20:
                    continue
                
                lives = int(res[15])
                player_alive = int(res[16])
                
                # Récupération du score actuel (BCD)
                try:
                    diz = int(hex(int(res[17]))[2:])
                    cen = int(hex(int(res[18]))[2:])
                    mil = int(hex(int(res[19]))[2:])
                    self.score = diz + cen * 100 + mil * 10000
                except:
                    pass
                
                # Condition de fin de partie (Plus de vies en réserve et Pacman mort)
                if lives == 0 and player_alive == 0:
                    self.high_score = max(self.high_score, self.score)
                    print(f"\n{Fore.RED}💀 Game Over détecté.{Style.RESET_ALL}")
                    print(f"{Fore.YELLOW}🏆 [SCORE FINAL] {self.score} | [HIGH SCORE] {self.high_score}{Style.RESET_ALL}\n")
                    game_over = True
                    break
                
                # Gestion de la mort temporaire (Pacman mort mais il reste des vies)
                if player_alive == 0:
                    self.current_plan = []
                    self.target_pos = None
                    self.draw_debug()
                    # Relâcher toutes les touches pendant les transitions (death/READY)
                    self.comm.communicate([
                        "execute P1_Left(0)", 
                        "execute P1_Right(0)", 
                        "execute P1_Up(0)", 
                        "execute P1_Down(0)"
                    ])
                    time.sleep(0.05)
                    continue
                
                # --- TRAITEMENT VRAM ---
                self.grid.fill(1)
                vram = list(map(int, res[0].split(",")))
                for idx, val in enumerate(vram):
                    if idx >= 1024:
                        break
                    gr, gc = idx % 32, 27 - ((idx // 32) - 2)
                    if 0 <= gr < 32 and 0 <= gc < 28:
                        if val in [64, 16, 20, 0xBF]:
                            self.grid[gr, gc] = 0
                        if val == 16:
                            self.grid[gr, gc] = 2  # Pastille normale
                        elif val == 20:
                            self.grid[gr, gc] = 3  # Power Pellet (Super pastille)

                # --- POSITIONS ET ÉTATS ---
                self.pos = (min(31, max(0, int(res[1])//8)), (27 - (int(res[2])//8) + 2) % 28)
                self.ghosts = [(min(31, max(0, int(res[3+i*2])//8)), (27 - (int(res[4+i*2])//8) + 2) % 28, int(res[11+i])) for i in range(4)]
                
                # Validation de la cible (Si déjà mangée ou état changé)
                if self.target_pos:
                    target_active = False
                    if self.grid[self.target_pos] in [2, 3]:
                        target_active = True
                    for gr, gc, gs in self.ghosts:
                        if gs in [1, 2] and (gr, gc) == self.target_pos:
                            target_active = True
                    if not target_active:
                        self.target_pos = None
                        self.current_plan = []

                # Évaluation de la nécessité de replanifier
                replan = not self.current_plan
                if self.current_plan:
                    nr, nc = self.pos; m_n = self.current_plan[0]
                    if m_n == 'up': nr -= 1
                    elif m_n == 'down': nr += 1
                    elif m_n == 'left': nc = (nc-1)%28
                    elif m_n == 'right': nc = (nc+1)%28
                    
                    # Colision mur planifiée
                    if self.grid[nr%32, nc] == 1:
                        replan = True
                        
                    # Proximité immédiate d'un danger non planifié (fantôme normal apparu à distance < 3)
                    for gr, gc, gs in self.ghosts:
                        if gs == 0 and self.get_dist((nr%32, nc), (gr, gc)) < 3:
                            replan = True
                            break
                
                if replan:
                    self.current_plan = self.find_best_path()
                    
                self.draw_debug()
                
                # Exécution du mouvement
                m = self.current_plan.pop(0) if self.current_plan else None
                l, r, u, d = 0, 0, 0, 0
                if m == 'up': u = 1
                elif m == 'down': d = 1
                elif m == 'left': l = 1
                elif m == 'right': r = 1
                
                self.comm.communicate([
                    f"execute P1_Left({l})",
                    f"execute P1_Right({r})",
                    f"execute P1_Up({u})",
                    f"execute P1_Down({d})"
                ])

if __name__ == "__main__":
    b = r"D:\Emulateurs\Mame Officiel\mame.exe"
    proc = subprocess.Popen([b, "pacman", "-window", "-skip_gameinfo", "-sound", "none", "-autoboot_delay", "1", "-autoboot_script", LUA_BRIDGE_PATH], cwd=os.path.dirname(b))
    try:
        PacmanRobot().run()
    finally:
        pygame.quit()
        proc.terminate()

