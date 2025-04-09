import random
import subprocess
import time
import numpy as np
from collections import deque

# Importer la classe MameCommunicator depuis le module fourni
from MameCommSocket import MameCommunicator

class SpaceInvadersAgent:
    """
    Agent intelligent pour jouer parfaitement à Space Invaders via MAME.
    Version optimisée pour cibler uniquement les aliens (ignore la soucoupe).
    """
    
    def __init__(self, host="127.0.0.1", port=12346):
        """
        Initialisation de l'agent Space Invaders.
        
        Args:
            host: Adresse IP de MAME (par défaut: localhost)
            port: Port de communication (par défaut: 12346)
        """
        # Connexion à MAME via le socket
        self.comm = MameCommunicator(host=host, port=port)
        
        # Dimensions de l'écran
        self.MAX_Y = 256
        self.MAX_X = 224
        
        # Initialiser les adresses mémoire du jeu
        self._init_memory_addresses()
        
        # État du jeu et historique
        self.game_state = {}
        self.history = deque(maxlen=100)
        self.is_game_over = False
        
    def _init_memory_addresses(self):
        """Initialise toutes les adresses mémoire importantes du jeu."""
        # Joueur
        self.PLAYER_POSITION = "201B"   # Position horizontale (X) du joueur
        self.PLAYER_SHOT_STATUS = "2025" # État du tir du joueur (0=prêt, 1=en vol)
        self.PLAYER_SHOT_Y = "2029"     # Ordonnée du tir du joueur
        self.PLAYER_SHOT_X = "202A"     # Abscisse du tir du joueur
        self.PLAYER_OK = "2068"         # 1=joueur OK, 0=en train d'exploser
        self.PLAYER_ALIVE = "20E7"      # 1=joueur vivant, 0=mort (dernière vie)
        
        # Aliens et position de référence
        self.RACK_DIRECTION = "200D"     # 	Value 0 if rack is moving right or 1 if rack is moving left
        self.REF_ALIEN_DX = "2008"       # Reference alien delta Xr
        self.REF_ALIEN_DY = "2007"       # Reference alien delta Yr
        self.REF_ALIEN_Y = "2009"       # Ordonnée de l'alien de référence
        self.REF_ALIEN_X = "200A"       # Abscisse de l'alien de référence
        self.NUM_ALIENS = "2082"        # Nombre d'aliens restants
        self.ALIENS_MEMORY_START = "2100" # Début de la mémoire des aliens (55 positions)
        
        # Tirs d'aliens (3 types)
        self.ROL_SHOT_Y = "203D"        # Tir rolling (oscillant) - Y
        self.ROL_SHOT_X = "203E"        # Tir rolling (oscillant) - X
        self.PLU_SHOT_Y = "204D"        # Tir plunger (droit) - Y
        self.PLU_SHOT_X = "204E"        # Tir plunger (droit) - X
        self.SQU_SHOT_Y = "205D"        # Tir squiggly (sinusoïdal) - Y
        self.SQU_SHOT_X = "205E"        # Tir squiggly (sinusoïdal) - X
        
        # Soucoupe et autres états (on utilise l'adresse mais on l'ignore)
        self.INVADED = "206D"           # 1=aliens ont atteint le bas
        
        # Score, vies et état du jeu
        self.NUM_COINS = "20EB"         # Nombre de pièces insérées
        self.SCORE = ["20F8", "20F9"]   # Score (2 octets)
        self.SHIPS_REMAINING = "21FF"   # Nombre de vies restantes

    def insert_coin_and_start(self, vitesse_de_jeu=10, lua_debug="off"):
        """
        Insère une pièce et démarre la partie.
        
        Args:
            vitesse_de_jeu: Vitesse d'exécution de MAME (1-100)
            lua_debug: Active/désactive le debug Lua ("on"/"off")
        """
        print("🪙 Insertion d'une pièce et démarrage de la partie...")
        
        # Configuration initiale
        self.comm.communicate([f"debug {lua_debug}"])
        self.comm.communicate([f"execute throttle_rate({vitesse_de_jeu})"])
        
        # Insérer une pièce et démarrer
        self.comm.communicate(["write_memory 20EB(1)"])
        self.comm.communicate(["execute P1_start(1)"])
        print("🚀 Partie démarrée !")
        

    def read_memory(self, address):
        """
        Lit une valeur mémoire à l'adresse spécifiée.
        
        Args:
            address: Adresse mémoire en hexadécimal (string)
            
        Returns:
            int: Valeur lue (0-255)
        """
        response = self.comm.communicate([f"read_memory {address}"])
        if response and response[0].isdigit():
            return int(response[0])
        return 0
        
    def read_memory_range(self, start_address, length):
        """
        Lit une plage de mémoire à partir de l'adresse spécifiée.
        
        Args:
            start_address: Adresse de début en hexadécimal (string)
            length: Nombre d'octets à lire
            
        Returns:
            list: Liste des valeurs lues
        """
        response = self.comm.communicate([f"read_memory_range {start_address}({length})"])
        if response and response[0]:
            return [int(val) for val in response[0].split(',') if val.isdigit()]
        return [0] * length
    
    def update_game_state(self):
        """
        Met à jour l'état complet du jeu en lisant la mémoire.
        
        Returns:
            dict: État actuel du jeu
        """
        # Demander une série de lectures mémoire
        self.comm.communicate([f"wait_for 23"])
        
        # === Lecture des positions et états ===
        
        # Position centrale du joueur d'après le code
        # 061B: 3A 1B 20 LD A,(playerXr) ; Player's X coordinate
        # 061E: C6 08 ADD A,$08 ; Center of player
        self.game_state['player_x'] = self.read_memory(self.PLAYER_POSITION) + 8
        
        # Référence des aliens (coin supérieur gauche du bloc) 
        self.game_state['aliens_x'] = self.read_memory(self.REF_ALIEN_X)
        self.game_state['aliens_y'] = self.read_memory(self.REF_ALIEN_Y)
        self.game_state['aliens_dx'] = self.read_memory(self.REF_ALIEN_DX)
        self.game_state['aliens_dy'] = self.read_memory(self.REF_ALIEN_DY)
        self.game_state['rack_direction'] = self.read_memory(self.RACK_DIRECTION) # 0=droite, 1=gauche
        
        # Nombre d'aliens restants
        self.game_state['aliens_left'] = self.read_memory(self.NUM_ALIENS)
        # vitesse déplacement par frame des aliens v=1.40/(x+5.74) => (0.023->0.2) en 55 aliens
        self.game_state["alien_x_vitesse"] = int( 1.40/(self.game_state['aliens_left']+5.74)*60)*(-1 if self.game_state['rack_direction'] else 1)
        # Tir du joueur
        self.game_state['missile_status'] = self.read_memory(self.PLAYER_SHOT_STATUS)
        self.game_state['missile_x'] = self.read_memory(self.PLAYER_SHOT_X)
        self.game_state['missile_y'] = self.MAX_Y - self.read_memory(self.PLAYER_SHOT_Y)
        
        # === Bombes des aliens (3 types) ===
        self.game_state['bombs'] = []
        
        # Tir de type "rolling" (oscillant)
        rol_x = self.read_memory(self.ROL_SHOT_X)
        rol_y = self.read_memory(self.ROL_SHOT_Y)
        if rol_y >0:
            self.game_state['bombs'].append(('rolling', rol_x, rol_y))
            
        # Tir de type "plunger" (droit)
        plu_x = self.read_memory(self.PLU_SHOT_X)
        plu_y = self.read_memory(self.PLU_SHOT_Y)
        if plu_y >0:
            self.game_state['bombs'].append(('plunger', plu_x, plu_y))
            
        # Tir de type "squiggly" (sinusoïdal)
        squ_x = self.read_memory(self.SQU_SHOT_X)
        squ_y = self.read_memory(self.SQU_SHOT_Y)
        if squ_y >0:
            self.game_state['bombs'].append(('squiggly', squ_x, squ_y))
        
        # === États du joueur ===
        self.game_state['player_ok'] = self.read_memory(self.PLAYER_OK)
        self.game_state['player_alive'] = self.read_memory(self.PLAYER_ALIVE)
        self.game_state['ships_remaining'] = self.read_memory(self.SHIPS_REMAINING)
        
        self.game_state['invaded'] = self.read_memory(self.INVADED)
        
        # === Score (2 octets) ===
        score_low = self.read_memory(self.SCORE[0])
        score_high = self.read_memory(self.SCORE[1])
        self.game_state['score'] = (
            (score_low & 0x0F) + 
            ((score_low >> 4) * 10) + 
            (score_high & 0x0F) * 100 + 
            ((score_high >> 4) * 1000)
        )
        
        # === Structure des aliens ===
        alien_data = self.read_memory_range(self.ALIENS_MEMORY_START, 55)
        self.game_state['alien_matrix'] = alien_data
        
        # Ajouter l'état actuel à l'historique
        self.history.append(dict(self.game_state))
        
        # Terminer la séquence de lecture
        self.comm.communicate([f"wait_for 1"])
        
        return self.game_state
    
    def get_alien_positions(self):
        """
        Calcule la position de chaque alien vivant.
        
        Returns:
            list: Liste de tuples (x, y, type, col, row) pour chaque alien
        """
        aliens = []
        alien_matrix = self.game_state['alien_matrix']
        
        # Positions de référence
        base_x = self.game_state['aliens_x']
        base_y = self.game_state['aliens_y']
        
        # Espacement entre les aliens
        x_spacing = 16
        y_spacing = 16
        
        for row in range(5):
            for col in range(11):
                idx = row * 11 + col
                # Dans la mémoire, 1 = alien vivant, 0 = alien mort (dans cette version)
                if idx < len(alien_matrix) and alien_matrix[idx] == 1:
                    x = base_x + col * x_spacing + 8  # Centre de l'alien
                    y = self.MAX_Y - (base_y - row * y_spacing)
                    
                    # Type d'alien détermine les points gagnés
                    if row == 4:
                        alien_type = "squid"     # 30 points
                    elif row in [2, 3]:
                        alien_type = "crab"      # 20 points
                    else:  # row in [0, 1]
                        alien_type = "octopus"   # 10 points
                    
                    aliens.append((x, y, alien_type, col, row))
        
        return aliens
    
    def predict_safest_position(self):
        """
        Calcule la position la plus sûre pour éviter les bombes.
        
        Returns:
            int: Position X la plus sûre
        """
        bombs = self.game_state['bombs']
        player_x = self.game_state['player_x']
        
        # Si pas de bombes, maintenir la position actuelle
        if not bombs:
            return player_x
            
        # Créer une carte de danger pour chaque position horizontale
        danger_map = np.ones(self.MAX_X) * 0.1  # Danger de base
        
        for bomb_type, bomb_x, bomb_y in bombs:
                
            # Plus la bombe est proche, plus elle est dangereuse
            vertical_danger = max(0, 1 - (bomb_y / 100))
            
            # Différents types de bombes ont différentes zones de danger
            if bomb_type == 'rolling':
                danger_width = 30     # Large zone (oscillante)
                variance = 40
            elif bomb_type == 'plunger':
                danger_width = 10     # Zone étroite (descente verticale)
                variance = 20
            elif bomb_type == 'squiggly':
                danger_width = 40     # Très large zone (sinusoïdale)
                variance = 64
            
            # Appliquer une distribution gaussienne de danger
            for x in range(max(0, bomb_x-danger_width), min(self.MAX_X, bomb_x+danger_width)):
                horizontal_distance = abs(x - bomb_x)
                horizontal_danger = np.exp(-(horizontal_distance**2) / (2 * variance))
                danger_map[x] += vertical_danger * horizontal_danger * 2.0
        
        # Pénaliser les bords (moins d'options d'échappement)
        danger_map[:15] += 0.3
        danger_map[-15:] += 0.3
        
        # Trouver les positions les moins dangereuses
        safest_positions = np.argsort(danger_map)
        
        # Préférer les positions proches actuelles pour éviter de grands déplacements
        for pos in safest_positions:
            if danger_map[pos] < 0.5 and abs(pos - player_x) < 50:
                return pos
                
        # Si aucune position proche n'est sûre, prendre la plus sûre globalement
        return safest_positions[0]
    
    def find_best_alien_target(self):
        """
        Détermine le meilleur alien à cibler.
        
        Stratégie: Prioriser les aliens les plus bas par colonne,
        avec préférence pour ceux au-dessus du joueur.
        
        Returns:
            Position X du meilleur alien à cibler, ou None si aucun
        """
        player_x = self.game_state['player_x']
        
        # Obtenir les positions des aliens
        aliens = self.get_alien_positions()
        if not aliens:
            return None
        
        # Trouver les aliens les plus bas dans chaque colonne
        lowest_aliens = []
        alien_columns = {}
        
        for alien in aliens:
            x, y, alien_type, col, row = alien
            if col not in alien_columns or row < alien_columns[col][1]:
                alien_columns[col] = (x, row, alien_type)
        
        for col, (x, row, alien_type) in alien_columns.items():
            lowest_aliens.append((x, row, alien_type, col))
        
        # Trier les aliens:
        # 1. Priorité aux aliens les plus proches du sol (rangées 0-1)
        # 2. À hauteur égale, priorité à ceux plus proches du joueur
        lowest_aliens.sort(key=lambda a: (a[1], abs(a[0] - player_x)))
        
        if lowest_aliens:
            # Retourne la position X de l'alien prioritaire
            return lowest_aliens[0][0]
        
        return 0
    
    def safe_move_to(self, target_x):
        """
        Déplace le joueur vers une position cible de manière sécurisée.
        
        Args:
            target_x: Position horizontale cible
        """
        player_x = self.game_state['player_x']
        
        # Si déjà à la position cible (ou très proche)
        if abs(player_x - target_x) < 4:
            # Arrêter tout mouvement
            self.comm.communicate(["execute P1_left(0)", "execute P1_right(0)"])
            self.fire()
            return
        
        # Déterminer la direction du mouvement
        if player_x < target_x:
            # Aller à droite
            self.comm.communicate(["execute P1_left(0)", "execute P1_right(1)"])
        else:
            # Aller à gauche
            self.comm.communicate(["execute P1_right(0)", "execute P1_left(1)"])
    
    def fire(self):
        """Tire le missile du joueur."""
        self.comm.communicate(["execute P1_Button_1(1)"])
        self.comm.communicate(["execute P1_Button_1(0)"])
    
    def check_game_over(self):
        """
        Vérifie si la partie est terminée.
        
        Returns:
            bool: True si la partie est terminée
        """
        return (
            # Plus d'aliens et score stable
            (len(self.history) >= 2 and 
             self.history[-1]['score'] == self.history[-2]['score'] and
             self.game_state['aliens_left'] == 0)
            or
            # Joueur mort (dernière vie perdue)
            self.game_state['player_alive'] == 0
        )
    
    def display_game_info(self):
        """Affiche les informations du jeu en cours."""
        print(f"👾 Aliens: {self.game_state['aliens_left']} | "
              f"👾 vitesse des aliens: {self.game_state['alien_x_vitesse']} | "
              f"🚀 Vies: {self.game_state['ships_remaining']} | "
              f"🎯 Score: {self.game_state['score']} | "
              f"🎖️ Level: {self.level} | "
              f"🎮 Joueur: {self.game_state['player_x']}")
        
        # Informations sur les bombes
        if self.game_state['bombs']:
            bomb_info = ", ".join([f"{t[:3]} ({x},{y})" for t, x, y in self.game_state['bombs']])
            print(f"💣 Bombes: {bomb_info}")
    
    def play(self, max_steps=10000):
        """
        Joue une partie complète de Space Invaders en ciblant uniquement les aliens.
        
        Args:
            max_steps: Nombre maximum d'étapes avant arrêt
        """
        print("🎮 Démarrage de l'agent IA Space Invaders (Mode: aliens uniquement)")
        self.insert_coin_and_start(10, "off")
        
        step = 0
        last_display_time = 0
        last_score = 0
        self.level = 0
        best_target_x=last_best_target_x=vitesse_alien=0
        self.old_alien_x = 56 # Position initiale des aliens
        
        while step < max_steps and not self.is_game_over:
            
            # Mettre à jour l'état du jeu
            self.update_game_state()
            if self.game_state['aliens_left'] == 0:
                self.level+=1
                        # Attendre que tous les aliens soient initialisés
                max_attempts = 100
                for _ in range(max_attempts):
                    self.update_game_state()
                    aliens = self.game_state['aliens_left']
                    if aliens == 55:
                        break
                    time.sleep(0.1)
            step += 1
            current_score = self.game_state['score']
            
            # Affichage périodique
            current_time = time.time()
            if current_time - last_display_time > 1:
                self.display_game_info()
                last_display_time = current_time
            
            # Vérifier si la partie est terminée
            if self.check_game_over():
                self.is_game_over = True
                print("🎮 Partie terminée!")
                break

                        
            # Suivi simple du score
            if current_score > last_score:
                last_score = current_score
            
            immediate_danger = False
            for bomb_type, bomb_x, bomb_y in self.game_state['bombs']:
                # Bombe proche du joueur et en dessous d'une certaine hauteur
                if bomb_y < 80 and bomb_y > 22 and abs(bomb_x - self.game_state['player_x']) < 16:
                    immediate_danger = True
                    break
            
            # 2. Si danger immédiat, priorité à l'esquive
            if immediate_danger:
                safest_x = self.predict_safest_position()
                self.safe_move_to(safest_x)
            else:
                best_target_x = self.find_best_alien_target()
                best_target_x+=self.game_state["alien_x_vitesse"]
                self.safe_move_to(best_target_x)                
        
        print(f"🏆 Score final: {self.game_state['score']}")
        print(f"🎮 Agent terminé après {step} étapes")
        print("🎮 Fin de partie")
        
        # Réinitialiser pour la prochaine partie
        self.comm.communicate([f"wait_for 0"])
        self.is_game_over = False


def main():
    """Fonction principale pour lancer MAME et l'agent."""
    # Configuration et lancement de MAME
    command = [
        "E:\\Emulateurs\\Mame Officiel\\mame.exe",
        "-artwork_crop",
        "-console",
        "-noautosave",
        "invaders",
        "-autoboot_delay",
        "1",
        "-autoboot_script",
        "E:\\Emulateurs\\Mame Sets\\MAME EXTRAs\\plugins\\PythonBridgeSocket_12346.lua",
    ]
    
    # Lancer MAME
    print("Démarrage de MAME...")
    process = subprocess.Popen(command, cwd="E:\\Emulateurs\\Mame Officiel")
    
    # Attendre que MAME soit prêt
    print("Attente de l'initialisation de MAME...")
    time.sleep(22)
    
    # Créer et démarrer l'agent
    agent = SpaceInvadersAgent()
    
    try:
        # Jouer en boucle
        while True:
            agent.play()
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nArrêt demandé par l'utilisateur")
    finally:
        # Nettoyer à la fin
        try:
            process.terminate()
            print("MAME arrêté.")
        except:
            pass


if __name__ == "__main__":
    main()