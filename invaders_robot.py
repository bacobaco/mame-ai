import random
import subprocess
import time
import numpy as np
from collections import deque
import math
import itertools # Importé pour la correction potentielle de deque slice (finalement non utilisé dans la dernière version de check_game_over)
import traceback # Pour afficher les erreurs détaillées

try:
    import keyboard
    import win32gui
except ImportError as e:
    print(f"ERREUR: Module manquant ({e}). Installez avec 'pip install keyboard pywin32'")
    input("Appuyez sur Entrée pour quitter...")
    exit()

# Importer la classe MameCommunicator depuis le module fourni
# Assurez-vous que le fichier MameCommSocket.py est dans le même répertoire
# ou dans le PYTHONPATH
try:
    from MameCommSocket import MameCommunicator
except ImportError:
    print("ERREUR: Impossible d'importer MameCommunicator depuis MameCommSocket.py.")
    print("Assurez-vous que le fichier est présent et accessible.")
    input("Appuyez sur Entrée pour quitter...")
    exit()


class SpaceInvadersAgent:
    """
    Agent intelligent (amélioré et corrigé) pour jouer à Space Invaders via MAME.
    Vise les aliens, esquive les bombes, sans utiliser wait_for.
    """

    # === Constantes de Configuration ===
    # Adresses mémoire (vérifiées pour Space Invaders Rev1)
    PLAYER_POSITION = "201B"    # Position X du joueur (bord gauche)
    PLAYER_SHOT_STATUS = "2025" # 0=prêt, 1=en vol
    PLAYER_SHOT_Y = "2029"      # Ordonnée Y du tir joueur (0=haut écran interne)
    PLAYER_SHOT_X = "202A"      # Abscisse X du tir joueur
    PLAYER_OK = "2068"          # 1=joueur OK, 0=en train d'exploser
    PLAYER_ALIVE = "20E7"       # 1=joueur vivant, 0=game over imminent
    RACK_DIRECTION = "200D"     # 0=droite, 1=gauche
    REF_ALIEN_X = "200A"        # Abscisse X alien référence (coin sup gauche bloc)
    REF_ALIEN_Y = "2009"        # Ordonnée Y alien référence
    NUM_ALIENS = "2082"         # Nombre d'aliens restants
    ALIENS_MEMORY_START = "2100" # Début état aliens (55 octets, 1=vivant)
    ROL_SHOT_Y = "203D"         # Tir rolling - Y (0=inactif)
    ROL_SHOT_X = "203E"         # Tir rolling - X
    PLU_SHOT_Y = "204D"         # Tir plunger - Y
    PLU_SHOT_X = "204E"         # Tir plunger - X
    SQU_SHOT_Y = "205D"         # Tir squiggly - Y
    SQU_SHOT_X = "205E"         # Tir squiggly - X
    INVADED = "206D"            # 1=aliens ont atteint le bas
    NUM_COINS = "20EB"          # Nombre de pièces
    SAUCER_X = "208A"           # Position X de la soucoupe mystère
    SCORE = ["20F8", "20F9"]    # Score (2 octets BCD)
    SHIPS_REMAINING = "21FF"    # Vies restantes (après la vie actuelle)

    # Dimensions et Positions (pixels)
    SCREEN_WIDTH = 224
    SCREEN_HEIGHT = 256 # Hauteur interne jeu pour coordonnées Y
    PLAYER_Y_POS = 32   # Position Y approx canon joueur (depuis bas)
    PLAYER_WIDTH = 15   # Largeur approximative du joueur en pixels (Hitbox)
    ALIEN_Y_START_OFFSET = 80 # Valeur approx début aliens Y
    ALIEN_WIDTH = 16
    ALIEN_HEIGHT = 8    # Hauteur approx pour collision
    ALIEN_SPACING_X = 16
    ALIEN_SPACING_Y = 16
    NUM_ALIEN_ROWS = 5
    NUM_ALIEN_COLS = 11

    # Paramètres Physiques Estimés (À AJUSTER PRÉCISÉMENT)
    GAME_FPS = 60.0             # FPS supposé (moins pertinent sans wait_for)
    PLAYER_MISSILE_SPEED_PPS = 180.0 # Vitesse tir joueur (pixels/sec) - À MESURER
    DEFAULT_BOMB_SPEED_PPS = 100.0  # Vitesse bombes (pixels/sec) - À MESURER
    ALIEN_SPEED_FACTOR = 1.40
    ALIEN_SPEED_OFFSET = 5.74

    # Paramètres de Stratégie
    TARGET_ALIGNMENT_TOLERANCE = 3
    DANGER_THRESHOLD_Y = 200 # Augmenté pour voir les bombes plus tôt
    DANGER_EFFECTIVE_Y_MIN = PLAYER_Y_POS - 4 # Corrigé: suivre la bombe jusqu'à ce qu'elle dépasse le joueur
    DANGER_HORIZONTAL_WINDOW_BASE = 16 # Augmenté pour plus de sécurité latérale
    DANGER_TIME_FACTOR = 2
    EDGE_PENALTY_FACTOR = 0.5
    EDGE_DISTANCE_THRESHOLD = 40
    SAFE_POS_DANGER_LIMIT = 0.3
    SAFE_POS_MAX_DIST = 50
    ALIEN_EDGE_PRIORITY_FACTOR = 1.5
    ALIEN_EDGE_PROXIMITY = 32
    STARTUP_WAIT_TIMEOUT = 15.0
    ACTION_DELAY = 0.02 # Pause minimale après action (secondes)
    LOOP_DELAY = 0.01   # Pause minimale boucle attente joueur
   # NOUVEAUX PARAMÈTRES POUR L'ESQUIVE AMÉLIORÉE
    IMMINENT_DANGER_THRESHOLD_FOR_MOVE = 0.40 # Réduit pour esquiver plus tôt (était 0.65)
    MODERATE_DANGER_CONSIDER_MOVE_THRESHOLD = 0.20 # Si le danger actuel est au-dessus de ça, on envisage de bouger si un meilleur spot existe
    MIN_DISTANCE_FOR_STRATEGIC_MOVE = 8 # Distance minimale pour considérer un mouvement d'esquive "stratégique" (plus grand que l'alignement)
    SAFETY_OVER_AIMING_FACTOR = 0.7 # Si le spot le plus sûr est X% meilleur que le spot actuel, on privilégie le mouvement
    BUNKER_ZONES = [(32, 52), (80, 100), (128, 148), (176, 196)] # Zones approximatives des bunkers (X min, X max)
    BUNKER_CLEARANCE_Y = 60 # Hauteur en dessous de laquelle on tire même s'il y a un bunker (urgence)
    def __init__(self, host="127.0.0.1", port=12345):
        """Initialisation de l'agent."""
        try:
            self.comm = MameCommunicator(host=host, port=port)
            print(f"Connexion à MAME sur {host}:{port} réussie.")
        except Exception as e:
            print(f"ERREUR: Impossible de créer MameCommunicator: {e}")
            raise # Propage l'erreur pour arrêter proprement

        self.game_state = {}
        self.history = deque(maxlen=10)
        self.is_game_over = False
        self.level = 0
        self.debug = 0
        self._last_move_command = None
        self._target_alien_id = None # Pour le verrouillage de cible (Hysteresis)
        self.setup_keyboard()

    # --- Fonctions de communication ---
    def _read_memory(self, address):
        response = self.comm.communicate([f"read_memory {address}"])
        if response and response[0].isdigit(): return int(response[0])
        # print(f"DEBUG WARN: Failed read at {address}") # Décommenter pour debug
        return 0 # Retourne 0 par défaut en cas d'échec

    def _read_memory_range(self, start_address, length):
        response = self.comm.communicate([f"read_memory_range {start_address}({length})"])
        if response and response[0]:
            values_str = response[0].split(',')
            if len(values_str) == length:
                try: return [int(val) for val in values_str]
                except ValueError: return [0] * length # Erreur de conversion
        # print(f"DEBUG WARN: Failed read range at {start_address}") # Décommenter pour debug
        return [0] * length

    def _write_memory(self, address, value):
         self.comm.communicate([f"write_memory {address}({value})"])

    def _execute_command(self, command):
        self.comm.communicate([f"execute {command}"])

    def setup_keyboard(self):
        """Configure les raccourcis clavier pour le debug (Ctrl+Shift+F3/F4)."""
        def on_key_press(event):
            # Vérification focus (optionnel, inspiré de invaders.py)
            try:
                hwnd = win32gui.GetForegroundWindow()
                # title = win32gui.GetWindowText(hwnd) # Pas utilisé pour l'instant
            except: pass

            if keyboard.is_pressed("shift") and keyboard.is_pressed("ctrl"):
                if keyboard.is_pressed("f3"):
                    self.debug = 0
                    print(f"\n[F3] Debug RESET = {self.debug}")
                    time.sleep(0.2)
                elif keyboard.is_pressed("f4"):
                    self.debug = (self.debug + 1) % 4
                    print(f"\n[F4] Debug Level = {self.debug}")
                    time.sleep(0.2)
        
        keyboard.on_press(on_key_press)
        print("⌨️ Clavier configuré: Ctrl+Shift+F3 (Reset) / F4 (Cycle Debug)")

    def _send_commands(self, commands):
        # Envoi simple sans wait_for
        responses = self.comm.communicate(commands)
        if responses:
            for r in responses:
                if "ERR" in r:
                    print(f"🔴 LUA ERROR: {r}")

    # --- Fonctions de démarrage et d'état ---
    def insert_coin_and_start(self, speed=10, lua_debug="off"):
        """Insère pièce, démarre ET ATTEND joueur (sans wait_for)."""
        print("🪙 Insertion pièce et démarrage...")
        # wait_for 2 car on envoie 2 commandes (debug + throttle)
        commands_init = [f"wait_for 3", f"debug {lua_debug}", f"execute throttle_rate({speed})", "frame_per_step 3"]
        self._send_commands(commands_init)

        self._write_memory(self.NUM_COINS, 1)
        print("   Pièce insérée (1/4)") 
        print("   Appui Start (2/4)")
        self._send_commands(["execute P1_start(1)"])
        time.sleep(self.ACTION_DELAY)
        print("   Attente apparition joueur... (4/4)")
        start_wait_time = time.time()
        player_appeared = False
        while time.time() - start_wait_time < self.STARTUP_WAIT_TIMEOUT:
            self.update_game_state()
            if self.game_state.get('player_alive', 0) == 1:
                print("   Joueur détecté (player_alive == 1)!")
                player_appeared = True
                break
            time.sleep(self.LOOP_DELAY)

        if not player_appeared:
            print(f"   ERREUR: Joueur non apparu après {self.STARTUP_WAIT_TIMEOUT}s.")
            raise RuntimeError("Timeout attente démarrage partie.")

        print("🚀 Partie démarrée et joueur prêt !")
        self.level = 1
        self.update_game_state()
        return True

    def update_game_state(self):
        """Met à jour l'état complet du jeu (sans wait_for)."""
        # === Lectures mémoire groupées (Optimisation majeure) ===
        # On envoie toutes les demandes en une seule fois pour éviter la latence réseau
        read_commands = [
            f"read_memory {self.PLAYER_POSITION}",      # 0
            f"read_memory {self.PLAYER_SHOT_STATUS}",   # 1
            f"read_memory {self.PLAYER_SHOT_Y}",        # 2
            f"read_memory {self.PLAYER_SHOT_X}",        # 3
            f"read_memory {self.PLAYER_OK}",            # 4
            f"read_memory {self.PLAYER_ALIVE}",         # 5
            f"read_memory {self.REF_ALIEN_X}",          # 6
            f"read_memory {self.REF_ALIEN_Y}",          # 7
            f"read_memory {self.NUM_ALIENS}",           # 8
            f"read_memory {self.SAUCER_X}",             # 9
            f"read_memory {self.RACK_DIRECTION}",       # 10
            f"read_memory {self.ROL_SHOT_Y}",           # 11
            f"read_memory {self.ROL_SHOT_X}",           # 12
            f"read_memory {self.PLU_SHOT_Y}",           # 13
            f"read_memory {self.PLU_SHOT_X}",           # 14
            f"read_memory {self.SQU_SHOT_Y}",           # 15
            f"read_memory {self.SQU_SHOT_X}",           # 16
            f"read_memory {self.INVADED}",              # 17
            f"read_memory {self.SCORE[0]}",             # 18
            f"read_memory {self.SCORE[1]}",             # 19
            f"read_memory {self.SHIPS_REMAINING}",      # 20
            f"read_memory_range {self.ALIENS_MEMORY_START}({self.NUM_ALIEN_ROWS * self.NUM_ALIEN_COLS})" # 21
        ]
        
        # Envoi des lectures sans wait_for pour éviter les blocages
        responses = self.comm.communicate(read_commands)
        
        # Sécurité si la réponse est incomplète
        if not responses or len(responses) < len(read_commands):
            print(f"⚠️ WARN: Réponses incomplètes ({len(responses) if responses else 0}/{len(read_commands)})")
            return self.game_state

        # Parsing des réponses (conversion en int)
        # Sans wait_for, les données commencent à l'index 0
        try:
            vals = [int(r) if r.isdigit() else 0 for r in responses[0:21]]
            # Cas spécial pour la matrice (liste d'entiers)
            alien_matrix_str = responses[21]
            alien_matrix_raw = [int(x) for x in alien_matrix_str.split(',')] if alien_matrix_str else []
        except ValueError:
            return self.game_state

        # === Traitement et Stockage ===
        if vals[0] == 0 and vals[5] == 1:
             # Si le joueur est vivant mais en X=0, c'est suspect (ou il est tout à gauche)
             pass 
        gs = {}
        # Toujours essayer d'ajouter toutes les clés, même si la lecture a échoué (valeur 0)
        gs['player_x'] = vals[0] + 8
        gs['player_y'] = self.PLAYER_Y_POS
        gs['missile_status'] = vals[1]
        gs['missile_y'] = self.SCREEN_HEIGHT - vals[2] if vals[1] == 1 else -1
        gs['missile_x'] = vals[3] if vals[1] == 1 else -1
        gs['player_ok'] = vals[4]
        gs['player_alive'] = vals[5]
        gs['ref_alien_x'] = vals[6]
        gs['ref_alien_y'] = self.SCREEN_HEIGHT - vals[7]
        gs['aliens_left'] = vals[8]
        gs['saucer_x'] = vals[9]
        gs['rack_direction'] = vals[10]
        
        # Bombes
        rol_y, rol_x = vals[11], vals[12]
        plu_y, plu_x = vals[13], vals[14]
        squ_y, squ_x = vals[15], vals[16]

        if gs['aliens_left'] > 0:
            speed_per_frame = self.ALIEN_SPEED_FACTOR / (gs['aliens_left'] + self.ALIEN_SPEED_OFFSET)
            gs['alien_speed_pps'] = speed_per_frame * self.GAME_FPS * (1 if gs['rack_direction'] == 0 else -1)
        else:
            gs['alien_speed_pps'] = 0

        gs['alien_matrix'] = alien_matrix_raw
        gs['bombs'] = []
        if rol_y > 0: gs['bombs'].append({'type': 'rolling', 'x': rol_x, 'y': self.SCREEN_HEIGHT - rol_y})
        if plu_y > 0: gs['bombs'].append({'type': 'plunger', 'x': plu_x, 'y': self.SCREEN_HEIGHT - plu_y})
        if squ_y > 0: gs['bombs'].append({'type': 'squiggly', 'x': squ_x, 'y': self.SCREEN_HEIGHT - squ_y})
        
        gs['invaded'] = vals[17]
        gs['score'] = ((vals[18] & 0x0F) + ((vals[18] >> 4) * 10) +
                       (vals[19] & 0x0F) * 100 + ((vals[19] >> 4) * 1000))
        gs['ships_remaining'] = vals[20]

        self.game_state = gs
        self.history.append(dict(gs))
        return self.game_state

    # --- Fonctions de stratégie (calculs) ---
    def get_alien_positions(self):
        """Calcule position (centre X, bas Y) aliens vivants."""
        aliens = []
        # Utiliser .get() pour éviter KeyError si état incomplet
        alien_matrix = self.game_state.get('alien_matrix', [])
        base_x = self.game_state.get('ref_alien_x', 0)
        ref_y = self.game_state.get('ref_alien_y', self.SCREEN_HEIGHT) # Haut du bloc
        if not alien_matrix: return [] # Sortir si matrice vide

        base_y_bottom = ref_y - (self.NUM_ALIEN_ROWS - 1) * self.ALIEN_SPACING_Y

        for row in range(self.NUM_ALIEN_ROWS):
            for col in range(self.NUM_ALIEN_COLS):
                idx = row * self.NUM_ALIEN_COLS + col
                if idx < len(alien_matrix) and alien_matrix[idx] == 1:
                    x = base_x + col * self.ALIEN_SPACING_X + (self.ALIEN_WIDTH / 2)
                    y = base_y_bottom + row * self.ALIEN_SPACING_Y
                    mem_row = row # Supposé correct
                    if mem_row in [0, 1]: alien_type = "octopus"
                    elif mem_row in [2, 3]: alien_type = "crab"
                    else: alien_type = "squid"
                    # ID unique pour le tracking : row * 100 + col
                    aliens.append({'id': row*100+col, 'x': x, 'y': y, 'type': alien_type, 'col': col, 'row': row})
        return aliens

    def predict_alien_future_x(self, alien_x, alien_y, player_x):
        """Prédit X alien à l'impact missile (incluant temps de déplacement joueur)."""
        if self.PLAYER_MISSILE_SPEED_PPS <= 0: return alien_x
        dist_y = abs(alien_y - self.PLAYER_Y_POS)
        time_to_reach_sec = dist_y / self.PLAYER_MISSILE_SPEED_PPS
        
        # Ajout: Temps pour que le joueur s'aligne (estimation grossière)
        # On suppose une vitesse joueur de ~60 PPS
        dist_x = abs(alien_x - player_x)
        time_to_align = dist_x / 60.0 
        
        total_time = time_to_reach_sec + (time_to_align * 0.5) # On pondère le temps d'alignement
        
        alien_speed_pps = self.game_state.get('alien_speed_pps', 0)
        predicted_displacement = alien_speed_pps * total_time
        predicted_x = alien_x + predicted_displacement
        half_alien_width = self.ALIEN_WIDTH / 2
        predicted_x = max(half_alien_width, min(self.SCREEN_WIDTH - half_alien_width, predicted_x))
        return predicted_x

    def find_best_alien_target(self):
        """Détermine meilleur alien à cibler (priorité bas, bords, proche)."""
        player_x = self.game_state.get('player_x', self.SCREEN_WIDTH / 2)
        aliens = self.get_alien_positions()
        if not aliens:
            if self.debug >= 1: print("⚠️ Aucune position d'alien trouvée.")
            return None

        alien_speed_pps = self.game_state.get('alien_speed_pps', 0)
        ref_alien_x = self.game_state.get('ref_alien_x', 0)
        rack_direction = self.game_state.get('rack_direction', 0)

        target_candidates = []
        lowest_aliens_in_col = {}
        for alien in aliens:
            col = alien['col']
            if col not in lowest_aliens_in_col or alien['row'] < lowest_aliens_in_col[col]['row']:
                 lowest_aliens_in_col[col] = alien
        if not lowest_aliens_in_col: return None

        for col, alien in lowest_aliens_in_col.items():
            alien_x, alien_y = alien['x'], alien['y']
            predicted_aim_x = self.predict_alien_future_x(alien_x, alien_y, player_x)
            priority_score = alien['row'] * 50 + abs(predicted_aim_x - player_x) / 10 # Priorité massive au bas (row 0)

            # Penalité si la cible est derrière un bunker (pour éviter de tirer dedans)
            if alien_y > self.BUNKER_CLEARANCE_Y:
                for (bx_min, bx_max) in self.BUNKER_ZONES:
                    if bx_min <= predicted_aim_x <= bx_max:
                        priority_score += 25.0 # On préfère une cible dégagée
                        break

            is_moving_right = (rack_direction == 0)
            block_right_edge = ref_alien_x + (self.NUM_ALIEN_COLS -1) * self.ALIEN_SPACING_X + self.ALIEN_WIDTH
            approaching_right = is_moving_right and (block_right_edge > self.SCREEN_WIDTH - self.ALIEN_EDGE_PROXIMITY)
            approaching_left = (not is_moving_right) and (ref_alien_x < self.ALIEN_EDGE_PROXIMITY)

            # Stratégie: Tuer les colonnes extérieures en priorité pour ralentir la descente
            if col == 0 or col == self.NUM_ALIEN_COLS - 1:
                priority_score -= 5.0 # Bonus permanent pour les bords

            # Bonus Hysteresis : Si c'est la cible précédente, on la garde (évite l'oscillation)
            if self._target_alien_id is not None and alien['id'] == self._target_alien_id:
                priority_score -= 20.0 # Augmenté pour plus de stabilité (méthodique)

            target_candidates.append({'alien': alien,'predicted_aim_x': predicted_aim_x,'priority_score': priority_score})

        # Gestion de la Soucoupe Mystère (Priorité absolue si visible et sûre)
        saucer_x = self.game_state.get('saucer_x', 0)
        # La soucoupe est généralement visible entre x=20 et x=200 (approximatif)
        if 56 < saucer_x < 200:
            # On vise directement la soucoupe (elle est rapide, l'anticipation est complexe sans vecteur vitesse)
            target_candidates.append({
                'alien': {'x': saucer_x, 'y': 220, 'type': 'saucer'},
                'predicted_aim_x': saucer_x,
                'priority_score': 500.0 # Priorité très faible (après tous les aliens)
            })

        if not target_candidates: return None
        target_candidates.sort(key=lambda c: c['priority_score'])
        
        best = target_candidates[0]
        if 'id' in best['alien']: self._target_alien_id = best['alien']['id']
        return best

    # Renommer et modifier cette fonction :
    # def predict_safest_position(self):
    def get_detailed_safe_position_info(self): # Nouveau nom et retours modifiés
        """
        Calcule la position X la plus sûre (Mur de la mort) et retourne cette position,
        son niveau de danger, et le niveau de danger à la position actuelle du joueur.
        """
        player_x_raw = self.game_state.get('player_x', self.SCREEN_WIDTH / 2)
        # Assurer que player_x_raw est bien un entier pour l'indexation
        current_player_x_idx = int(min(self.SCREEN_WIDTH - 1, max(0, player_x_raw)))

        bombs = self.game_state.get('bombs', [])
        danger_map = np.zeros(self.SCREEN_WIDTH)
        
        # Si pas de bombes, la position actuelle est sûre (danger 0)
        if not bombs:
            # Retourne la position actuelle, danger 0 à cette position, et danger 0 à la position actuelle
            return current_player_x_idx, 0.0, 0.0

        for bomb in bombs:
            bomb_x, bomb_y, bomb_type = bomb['x'], bomb['y'], bomb['type']
            
            # Ignorer les bombes trop hautes ou déjà passées (sous le joueur)
            if bomb_y > self.DANGER_THRESHOLD_Y or bomb_y < self.DANGER_EFFECTIVE_Y_MIN:
                continue

            dist_y_to_player = max(1, bomb_y - self.PLAYER_Y_POS) # distance Y entre la bombe et la ligne du joueur
            
            # Si DEFAULT_BOMB_SPEED_PPS est 0 ou négatif, time_impact serait infini ou négatif.
            # On met une petite valeur pour éviter la division par zéro.
            bomb_speed = self.DEFAULT_BOMB_SPEED_PPS if self.DEFAULT_BOMB_SPEED_PPS > 0 else 1.0
            time_impact = dist_y_to_player / bomb_speed
            
            # Danger de base très élevé pour forcer la réaction
            danger_value = 100.0 * math.exp(-time_impact * self.DANGER_TIME_FACTOR)
            
            # Zone d'impact élargie (Mur de la mort)
            width = self.DANGER_HORIZONTAL_WINDOW_BASE
            if bomb_type == 'squiggly': width += 10 # Plus large car oscille
            
            min_x = max(0, int(bomb_x - width))
            max_x = min(self.SCREEN_WIDTH, int(bomb_x + width))
            
            # Application du danger uniforme sur toute la largeur (pas de gaussienne, c'est binaire : mort ou vif)
            danger_map[min_x:max_x] += danger_value

        # Si aucun danger détecté, rester sur place
        if np.sum(danger_map) == 0:
             return current_player_x_idx, 0.0, 0.0

        # --- AMÉLIORATION HITBOX ---
        # Le joueur n'est pas un point unique. Si danger_map[x] est sûr mais danger_map[x+1] est mortel,
        # le joueur mourra car son vaisseau occupe ~15 pixels.
        # On dilate le danger : le danger en X est le max du danger sur la largeur du vaisseau centrée en X.
        danger_map_expanded = np.zeros_like(danger_map)
        half_w = self.PLAYER_WIDTH // 2 + 4 # Marge supplémentaire pour la hitbox
        for i in range(len(danger_map)):
            start = max(0, i - half_w)
            end = min(len(danger_map), i + half_w + 1)
            danger_map_expanded[i] = np.max(danger_map[start:end])
        danger_map = danger_map_expanded

        # Pénalité pour les bords de l'écran
        edge = self.EDGE_DISTANCE_THRESHOLD
        danger_map_expanded[:edge] += self.EDGE_PENALTY_FACTOR
        danger_map_expanded[self.SCREEN_WIDTH-edge:] += self.EDGE_PENALTY_FACTOR

        danger_at_current_player_pos = danger_map_expanded[current_player_x_idx]

        # Trouver la position la plus sûre
        # On cherche l'index avec le danger minimum absolu
        min_danger = np.min(danger_map_expanded)
        safest_indices = np.where(danger_map_expanded == min_danger)[0]
        
        # Parmi les zones sûres, prendre la plus proche du joueur pour minimiser le déplacement
        chosen_safest_x = safest_indices[np.abs(safest_indices - current_player_x_idx).argmin()]
        
        danger_at_chosen_safest_x = danger_map_expanded[chosen_safest_x]
        
        return chosen_safest_x, danger_at_chosen_safest_x, danger_at_current_player_pos

    # --- Fonctions d'action et de contrôle ---
    def safe_move_to(self, target_x, can_fire=False):
        """Déplace et tire (sans wait_for, accès sécurisé)."""
        player_x = self.game_state.get('player_x', self.SCREEN_WIDTH / 2)
        missile_ready = self.game_state.get('missile_status', 1) == 0 # 0 = Prêt à tirer

        # 1. Calcul de la direction (Gauche/Droite/Stop)
        left = 0
        right = 0
        if abs(player_x - target_x) > self.TARGET_ALIGNMENT_TOLERANCE:
            if player_x < target_x:
                right = 1
            else:
                left = 1
        
        # 2. Calcul du Tir
        fire = 0
        if can_fire and missile_ready:
            # On tire si on est aligné ou très proche (tolérance légèrement augmentée pour le tir)
            if abs(player_x - target_x) <= self.TARGET_ALIGNMENT_TOLERANCE + 4:
                fire = 1

        if self.debug >= 2:
            print(f"🤖 MOVE: Player={player_x:.1f} Target={target_x:.1f} -> Left={left} Right={right} Fire={fire}")

        # 3. Envoi groupé des commandes (Style invaders.py : on force l'état de chaque bouton)
        commands = [
            f"execute P1_left({left})",
            f"execute P1_right({right})",
            f"execute P1_Button_1({fire})"
        ]
        
        # Pas de wait_for ici, car on est dans la fenêtre ouverte par update_game_state(close_frame=False)
        self._send_commands(commands)


    def check_game_over(self):
        """Vérifie Game Over (sans slice deque, accès sécurisé)."""
        # Vérifier si l'état existe et si le jeu a démarré
        if not self.game_state or self.level == 0 or 'player_alive' not in self.game_state:
            return False

        if self.game_state.get('invaded', 0) == 1:
            print("👾 Invasion Alien !")
            return True

        if self.game_state.get('player_alive', 1) == 0:
            history_len = len(self.history)
            # Vérifier si mort confirmée sur les 3 derniers états (si disponibles)
            if history_len >= 3:
                # Accéder aux 3 derniers éléments sans slicing
                is_confirmed_dead = True
                try:
                    # Itérer sur les indices correspondants aux 3 derniers
                    for i in range(history_len - 1, history_len - 4, -1):
                        if self.history[i].get('player_alive', 1) != 0:
                            is_confirmed_dead = False
                            break
                except IndexError:
                     # Devrait pas arriver avec la vérification history_len >= 3 mais par sécurité
                     is_confirmed_dead = (self.history[-1].get('player_alive', 1) == 0) # Vérifie juste le dernier

                if is_confirmed_dead:
                    print("💀 Joueur non vivant confirmé sur historique récent.")
                    return True
            # Si peu d'historique, considérer mort si le dernier état le dit
            elif history_len > 0 and self.history[-1].get('player_alive', 1) == 0:
                 print("💀 Joueur non vivant (état le plus récent).")
                 return True

        return False

    def wait_for_level_reset(self):
        """Attend nouveau niveau (sans wait_for)."""
        print(f"Niveau {self.level} terminé ! Attente niveau {self.level + 1}...")
        next_level = self.level + 1
        max_wait = 5.0
        start = time.time()
        reset = False

        while time.time() - start < max_wait:
            self.update_game_state()
            # Vérifier mort pendant transition (utilise check_game_over corrigé)
            if self.game_state.get('player_alive', 1) == 0 and self.check_game_over():
                 self.is_game_over = True; print("... mort pendant transition (Game Over)."); return False

            if self.game_state.get('aliens_left') == self.NUM_ALIEN_ROWS * self.NUM_ALIEN_COLS:
                print(f"Niveau {next_level} détecté. Reprise."); self.level = next_level
                time.sleep(self.ACTION_DELAY * 5); reset = True; break

            time.sleep(self.LOOP_DELAY)
            if int(time.time() - start) > int(time.time() - start - self.LOOP_DELAY):
                 print(f"   ... attente niv {next_level} ({time.time() - start:.1f}s)")

        if not reset: self.is_game_over = True; print("Timeout reset niveau."); return False
        return True

    def display_game_info(self):
        """Affiche infos jeu (accès sécurisé)."""
        if not self.game_state: return
        state = self.game_state
        # Utiliser .get() pour tous les accès pour robustesse maximale
        info = (f"Lvl:{self.level} | "
                f"Aliens:{state.get('aliens_left', '?')} | "
                f"Vies:{state.get('ships_remaining', '?')} | "
                f"Score:{state.get('score', '?')} | "
                f"Player:(X:{state.get('player_x', '?')}, Alive:{state.get('player_alive', '?')}) | "
                f"Missile:{'Ready' if state.get('missile_status', 1) == 0 else 'Flying'} ")
        print(info)
        bombs = state.get('bombs')
        if bombs: print("Bombs: " + ", ".join([f"{b.get('type','?')[:3]}({b.get('x','?')},{b.get('y','?')})" for b in bombs]))


    # --- Boucle de jeu principale ---
    def play(self, max_steps=50000):
        """Boucle principale (corrigée pour KeyError)."""
        print(f"🎮 Démarrage partie (Max {max_steps} étapes)")
        self.game_state = {}
        self.history.clear()
        self.is_game_over = False
        self.level = 0
        self._last_move_command = None

        try:
            if not self.insert_coin_and_start(speed=10): return
        except Exception as e:
            print(f"Erreur Init/Start : {e}"); return

        step = 0
        last_display_time = time.time()

        while step < max_steps and not self.is_game_over:
            # 1. MàJ État
            try: self.update_game_state()
            except Exception as e: print(f"Erreur update step {step}: {e}"); time.sleep(0.5); continue
            if not self.game_state or 'player_alive' not in self.game_state:
                print(f"WARN: État jeu incomplet step {step}. Skipping."); time.sleep(self.LOOP_DELAY); continue

            # 2. Affichage
            current_time = time.time()
            if self.debug >= 1 or current_time - last_display_time > 1.0: self.display_game_info(); last_display_time = current_time

            # 3. Vérif Fin Jeu/Niveau
            if self.check_game_over():
                self.is_game_over = True
                break
            if self.game_state.get('aliens_left', -1) == 0 and self.game_state.get('player_alive', 1) == 1:
                if not self.wait_for_level_reset(): break
                else: self._last_move_command = None; continue

            # 4. Décision Agent
            try:
                current_player_x = self.game_state.get('player_x', self.SCREEN_WIDTH / 2)
                
                safest_x_coord, danger_level_at_safest, danger_level_at_current = self.get_detailed_safe_position_info()

                perform_evasive_maneuver = False

                # Condition 1: La position actuelle est bien trop dangereuse. Esquive prioritaire.
                if danger_level_at_current > self.IMMINENT_DANGER_THRESHOLD_FOR_MOVE:
                    perform_evasive_maneuver = True
                    if self.debug >= 1 and self.game_state.get('bombs'):
                         print(f"⚠️ EVASION URGENTE! Danger: {danger_level_at_current:.2f} -> Vers {safest_x_coord}")

                # Condition 2: La position actuelle est modérément dangereuse,
                # ET la position la plus sûre est significativement meilleure,
                # ET cela implique un mouvement non trivial.
                elif danger_level_at_current > self.MODERATE_DANGER_CONSIDER_MOVE_THRESHOLD and \
                     danger_level_at_safest < danger_level_at_current and \
                     abs(safest_x_coord - current_player_x) >= self.MIN_DISTANCE_FOR_STRATEGIC_MOVE:
                    perform_evasive_maneuver = True
                    if self.debug >= 1 and self.game_state.get('bombs'):
                       print(f"🛡️ EVASION STRATEGIQUE! Danger: {danger_level_at_current:.2f} -> {danger_level_at_safest:.2f}")

                if self.debug >= 3:
                    print(f"🤔 DECISION: Danger={danger_level_at_current:.2f} Evasion={perform_evasive_maneuver}")

                if perform_evasive_maneuver:
                    self.safe_move_to(safest_x_coord, can_fire=False) # Ne pas tirer pendant l'esquive
                else:
                    # Comportement normal : viser et tirer (OPPORTUNISTE)
                    best_target = self.find_best_alien_target()
                    
                    # Tir Opportuniste : Vérifier si on est aligné avec N'IMPORTE QUEL alien maintenant
                    # même si ce n'est pas la cible principale
                    can_fire_now = False
                    
                    # Vérification si le joueur est sous un bunker
                    player_under_bunker = False
                    for (bx_min, bx_max) in self.BUNKER_ZONES:
                        if bx_min <= current_player_x <= bx_max:
                            player_under_bunker = True
                            break

                    aliens = self.get_alien_positions()
                    for alien in aliens:
                        # Si on est aligné avec un alien (tolérance large)
                        if abs(current_player_x - alien['x']) < (self.ALIEN_WIDTH / 2 + 2):
                            # Si sous bunker, on ne tire QUE si l'alien est bas (urgence)
                            if player_under_bunker and alien['y'] > self.BUNKER_CLEARANCE_Y:
                                continue
                            can_fire_now = True
                            break
                    
                    # Si danger modéré, on hésite à tirer
                    if danger_level_at_current > self.MODERATE_DANGER_CONSIDER_MOVE_THRESHOLD:
                        can_fire_now = False

                    if best_target and 'predicted_aim_x' in best_target:
                        self.safe_move_to(best_target['predicted_aim_x'], can_fire=can_fire_now)
                    else:
                        # Pas de cible valide, se déplacer vers la position la plus sûre (ou rester si déjà sûr)
                        self.safe_move_to(safest_x_coord, can_fire=can_fire_now)
            
            except Exception as e:
                 print(f"Erreur Décision step {step} : {type(e).__name__} - {e}")
                 traceback.print_exc() # Décommenter pour trace complète
                 # En cas d'erreur, on envoie des commandes neutres pour satisfaire le wait_for de Lua
                 self.safe_move_to(self.game_state.get('player_x', 0)) 
                 self._last_move_command = "stop"; time.sleep(0.1)

            # 5. Fin de Frame (Commit)
            # PLUS BESOIN DE wait_for 0 ICI !
            # Lua avancera automatiquement la frame dès qu'il aura reçu les 3 commandes de safe_move_to
            # car on a configuré le wait_for exact au début de la boucle.
            step += 1
        # --- Fin boucle while ---

        print(f"🏁 Agent terminé après {step} étapes.")
        if self.game_state: print(f"🏆 Dernier état: Score={self.game_state.get('score', 'N/A')}")
        else: print("🏆 Score final: N/A")
        self._send_commands(["execute P1_left(0)", "execute P1_right(0)"])
        self._last_move_command = "stop"

# === Fonction Main (lancement MAME et agent) ===
def main():
    """Fonction principale pour lancer MAME et l'agent."""
    print("DEBUG: Entrée dans main()")
    # --- Configuration MAME (ADAPTER LES CHEMINS !) ---
    # Chemins Windows forcés comme demandé
    mame_executable = r"D:\Emulateurs\Mame Officiel\mame.exe"
    mame_cwd = r"D:\Emulateurs\Mame Officiel"
    lua_script_path = r"D:\Emulateurs\Mame Officiel\plugins\PythonBridgeSocket.lua"

    # Vérifier si le script Lua existe
    import os
    if not os.path.exists(lua_script_path):
         print(f"ERREUR: Script Lua non trouvé à '{lua_script_path}'")
         print("Veuillez vérifier le chemin lua_script_path.")
         return
    # --- Fin Configuration MAME ---

    command = [
        mame_executable, "invaders", 
            "-window", "-resolution", "448x576", "-skip_gameinfo","-artwork_crop", "-console", "-noautosave",
        "-autoboot_delay", "1", "-autoboot_script", lua_script_path,
    ]
    process = None
    agent = None

    try:
        print("🚀 Démarrage de MAME...")
        print(f"Commande: {' '.join(command)}")
        # Utiliser le répertoire de travail spécifié si différent de None
        process_cwd = mame_cwd
        process = subprocess.Popen(command, cwd=process_cwd)

        print("⏳ Attente initiale MAME/Lua (20s)...")
        time.sleep(20) # TODO: Remplacer par connexion en boucle + timeout

        # Créer l'agent APRÈS l'attente initiale
        agent = SpaceInvadersAgent(host="127.0.0.1", port=12345) # Assurer port = script Lua

        play_count = 0
        while True:
            play_count += 1
            print(f"\n--- DÉMARRAGE PARTIE #{play_count} ---")
            try: agent.play() # Lancer une partie
            except Exception as play_error:
                 print(f"💥 Erreur fatale pendant agent.play() partie #{play_count}: {play_error}")
                 traceback.print_exc()
            print(f"--- FIN PARTIE #{play_count} ---")

            # Vérifier si MAME est toujours actif
            if process.poll() is not None: print("MAME s'est arrêté."); break

            print("\n" + "="*30 + "\nPréparation suivante (1s)... Ctrl+C pour quitter.\n" + "="*30 + "\n")
            time.sleep(1)

    except FileNotFoundError: print(f"Erreur: MAME non trouvé ('{mame_executable}') ou CWD incorrect.")
    except ConnectionRefusedError: print("Erreur: Connexion refusée (MAME/Lua non prêt ou port/hôte incorrect?)")
    except KeyboardInterrupt: print("\n🛑 Arrêt demandé.")
    except Exception as e:
        print(f"\n💥 Erreur inattendue dans main: {e}")
        traceback.print_exc()
    finally:
        # Nettoyage
        if process and process.poll() is None:
            try:
                print("🧹 Arrêt de MAME..."); process.terminate()
                process.wait(timeout=5); print("✅ MAME arrêté.")
            except subprocess.TimeoutExpired:
                print("⚠️ Forçage MAME..."); process.kill(); print("✅ MAME forcé.")
            except Exception as e_term: print(f"Erreur arrêt MAME: {e_term}")
        else: print("ℹ️ MAME non démarré ou déjà arrêté.")
        print("Script terminé.")


if __name__ == "__main__":
    print("DEBUG: Script exécuté directement.")
    main()