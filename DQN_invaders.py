from datetime import datetime
import time, keyboard, random
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from MameCommSocket import MameCommunicator

# Initialisation socket
comm = MameCommunicator("localhost", 12345)

# Constantes
MAX_PLAYER_POS = 255
MAX_SAUCER_POS = 255
MAX_BOMB_POS_X = 255
MAX_BOMB_POS_Y = 255
ACTION_DELAY = 0.01
debug = False
flag_tir = True

# Adresses du jeu space invaders
# https://www.computerarcheology.com/Arcade/SpaceInvaders/RAMUse.html
numCoins = "20EB"
P1ScorL = "20F8"
P1ScorM = "20F9"
shotSync = "2080"  # Les 3 tirs sont synchronisés avec le minuteur GO-2. Ceci est copié à partir du minuteur dans la boucle de jeuDQN

alienShotYr = "207B"  # Delta Y du tir alien
alienShotXr = "207C"  # Delta X du tir alien
refAlienYr="2009" #Reference alien Yr coordinate
refAlienXr="200A"#Reference alien Xr coordinate
rolShotYr = "203D"  # Game object 2: Alien rolling-shot (targets player specifically)
rolShotXr = "203E"
squShotYr = "205D"  # Game object 4: squiggly shot position
squShotXr = "205E"
pluShotYr = "204D"  # the plunger-shot (object 3) position
pluSHotXr = "204E"

playerAlienDead = "2100"  # 2100:2136		Player 1 alien ship indicators (0=dead) 11*5 = 55

saucerDeltaX = "208A"  # 208A saucerPriPicMSB Mystery ship print descriptor [oscille entre 41 et 224 quand il n'est pas à l'écran hors de porté]
playerXr = "201B"  # 	Descripteur de sprite du joueur ... position MSB
plyrShotStatus = "2025"  # 0 if available, 1 if just initiated, 2 moving normally, 3 hit something besides alien, 5 if alien explosion is in progress, 4 if alien has exploded (remove from active duty)
obj1CoorYr = "2029"  # 	Player shot Y coordinate
obj1CoorXr = "202A"  # 	Player shot X coordinate
p1ShipsRem = "21FF"  # 		Ships remaining after current dies
gameMode = "20EF"
playerAlive = (
    "2015"  # Player is alive [FF=alive]. Toggles between 0 and 1 for blow-up images.
)
player1Alive = "20E7"  # 1 if player is alive, 0 if dead (after last man)

# Liste des actions du jeu demandées au réseau de neurone
actions = { 0: "left",1: "right",2: "stop",3:"tir",4:"tir-left",5:"tir-right"}

import os


class DQN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
        self.chemin = "./dqn_invaders_model.pth"

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def sauvegarder_modele(self):
        torch.save(self.state_dict(), self.chemin)

    def charger_modele(self):
        if os.path.exists(self.chemin):
            self.load_state_dict(torch.load(self.chemin))
            self.eval()
        else:
            print(
                f"Le fichier {self.chemin} n'existe pas. Impossible de charger le modèle."
            )


def copier_poids(dqn_source, dqn_cible):
    dqn_cible.load_state_dict(dqn_source.state_dict())


def choisir_action(dqn, etat, epsilon):
    if random.random() < epsilon:
        return random.randint(0, len(actions) - 1)
    q_values = dqn(torch.tensor(etat, dtype=torch.float32))
    return torch.argmax(q_values).item()


def executer_action(action):
    if actions[action] == "left":
        response = comm.communicate(["execute P1_left(1)", "execute P1_right(0)","execute P1_Button_1(0)"])
    elif actions[action] == "right":
        response = comm.communicate(["execute P1_left(0)", "execute P1_right(1)","execute P1_Button_1(0)"])
    elif actions[action] == "stop":
        response = comm.communicate(["execute P1_left(0)", "execute P1_right(0)","execute P1_Button_1(0)"])
    elif actions[action] == "tir":
        response = comm.communicate(["execute P1_left(0)", "execute P1_right(0)","execute P1_Button_1(1)"])
    elif actions[action] == "tir-left":
        response = comm.communicate(["execute P1_left(1)", "execute P1_right(0)","execute P1_Button_1(1)"])
    elif actions[action] == "tir-right":
        response = comm.communicate(["execute P1_left(0)", "execute P1_right(1)","execute P1_Button_1(1)"])



def get_reward():
    response = comm.communicate([f"read_memory {P1ScorL}", f"read_memory {P1ScorM}"])
    if debug:
        print(f"score={response}")
    P1ScorL_v, P1ScorM_v = list(map(int, response))

    return (P1ScorL_v >> 4) * 10 + (P1ScorM_v & 0x0F) * 100 + ((P1ScorM_v) >> 4) * 1000


def get_state():  # sourcery skip: inline-immediately-returned-variable
    # Lire les informations de la mémoire du jeu position du player et soucoupe
    response = comm.communicate(
        [
            f"read_memory {playerXr}",
            f"read_memory {obj1CoorXr}",
            f"read_memory {obj1CoorYr}",
            f"read_memory {saucerDeltaX}",
            f"read_memory {rolShotXr}",
            f"read_memory {rolShotYr}",
            f"read_memory {pluSHotXr}",
            f"read_memory {pluShotYr}",
            f"read_memory {squShotXr}",
            f"read_memory {squShotYr}",
#            f"read_memory {refAlienXr}",
#            f"read_memory {refAlienYr}",
            
        ]
    )
    player_pos, psx, psy, saucer_pos, rx, ry, px, py, sx, sy = list(map(int, response))
    state = np.array(
        [
            player_pos,
            psx,
            psy,
            saucer_pos,
            rx,
            ry,
            px,
            py,
            sx,
            sy,
#            rax,
#            ray,
            *extract_alien_coordinates(),
        ]
    )
    return state


def extract_alien_coordinates():
    alien_coordinates = []
    messages = ["read_memory 2009", "read_memory 200A"]
    for index in range(55):
        # Check if the alien is alive
        alive_flag_addr = hex(0x2100 + index)[2:]
        messages.append(f"read_memory {alive_flag_addr}")
    response = comm.communicate(messages)
    ref_alien_x, ref_alien_y = list(map(int, response[:2]))
    for index in range(55):
        # Calculating alien coordinates
        row = index // 11
        col = index % 11
        # Calculate the alien coordinates based on the reference alien
        alien_x = ref_alien_x + col * 16
        alien_y = ref_alien_y + row * 16
        if int(response[index + 2]) == 1:
            # Get the reference alien coordinates
            alien_coordinates.append((alien_x, alien_y))
        else:
            # The alien is dead, set the coordinates to (0, 0)
            alien_coordinates.append((0, 0))
    return [element for couple in alien_coordinates for element in couple]


def main():
    global debug
    f2_pressed = False
    # Configuration
    # Taille de l'entrée du réseau de neurones (à adapter en fonction de vos besoins)
    input_size = 10+55*2  # 3*2 positions des bombes 1 position joueur et 1 pos soucoupe et 1*2 positions tir joueur et 1*2 pos AlienRef
    # input_size+= 55*2 #2*55 position des aliens
    hidden_size = 10  # Taille de la couche cachée
    output_size = (
        6  # Taille de la sortie du réseau de neurones (nombre d'actions possibles)
    )

    learning_rate = 0.01  # 0.01
    gamma = 0.99  # 0.99
    num_episodes = 10000  # nb de partie quasi infini ;-)
    epsilon_start = 1
    epsilon_end = 0
    epsilon_decay = 0.99999
    epsilon = epsilon_start
    dqn = DQN(input_size, hidden_size, output_size)
    target_dqn = DQN(input_size, hidden_size, output_size)
    optimizer = optim.Adam(dqn.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()
    dqn.charger_modele()
    copier_poids(dqn, target_dqn)

    # nombre d'info ou commandes à exécuter par frame:
    # 2 pour connaitre l'état du joueur
    # 2 pour le score
    # 1 pour le bouton tire
    # 1 pour l'action
    # input_size pour les positions, entrées du DQN
    # et 1 pour le message "wait_for" lui même !
    NB_DE_DEMANDES_PAR_FRAME = str(10+55 +2+ 2+2+3) #nb entrees+2refalien(uniquement dans le cas de toutes les pos des aliens)+2 alien_alive/dead+2 message pour le score+3 messages-actions
    sum_of_reward = mean_score = mean_score_old = 0
    # VITESSE DU JEU==================
    vitesse_de_jeu = 5
    # VITESSE DU JEU==================
    response = comm.communicate(
        [
            f"write_memory {numCoins}(1)",
            "execute P1_start(1)",
            f"execute throttle_rate({vitesse_de_jeu})",
            "execute throttled(0)",
        ]
    )
    for episode in range(num_episodes):
        step = 0
        if f2_pressed:
            print("Sortie prématurée de la boucle 'for'.")
            break
        reward = 0
        player_life_end = 0
        player_alive = 0
        response = comm.communicate([f"write_memory {numCoins}(1)"])
        while player_alive == 0:
            player_alive = int(comm.communicate([f"read_memory {player1Alive}"])[0])
        state = get_state()  # Remplacer par l'état initial du jeu
        while player_alive == 1:
            while step == 0:
                time.sleep(5 / vitesse_de_jeu)  # attendre 5 secondes au début
                step += 1
                response = comm.communicate([f"wait_for {NB_DE_DEMANDES_PAR_FRAME}"])
            # Vérifier si la touche "esc" est pressée
            if keyboard.is_pressed("f2"):
                print("Touche 'F2' détectée. Sortie prématurée de la boucle.")
                f2_pressed = True
                break
            if keyboard.is_pressed('ctrl') and keyboard.is_pressed('f4'):
                debug=False
            elif  keyboard.is_pressed('f4'):
                debug=True
        
                
            # Choisir une action en fonction de l'état actuel et de la stratégie epsilon-greedy
            action = choisir_action(dqn, state, epsilon)
            # Exécuter l'action dans MAME et obtenir la récompense et l'état suivant
            executer_action(action)
            # Remplacer les lignes suivantes par les commandes MameCommunicator appropriées
            next_state = (
                get_state()
            )  # Remplacer par le nouvel état obtenu après l'action
            response = comm.communicate(
                [
                    f"read_memory {playerAlive}",
                    f"read_memory {player1Alive}",
                ]
            )
            player_life_end, player_alive = list(
                map(int, response),
            )
            reward = get_reward()  # Remplacer par la récompense obtenue après l'action
            # Mettre à jour le réseau de neurones
            q_values = dqn(torch.tensor(next_state, dtype=torch.float32))
            target_q_values = q_values.clone().detach()
            _, best_action = torch.max(
                dqn(torch.tensor(next_state, dtype=torch.float32)), 0
            )
            if player_life_end < 255:
                response = comm.communicate(["wait_for 0"])
                target_q_values[action] = reward - 100
                time.sleep(3 / vitesse_de_jeu)  # attendre 2 seconde
                response = comm.communicate([f"wait_for {NB_DE_DEMANDES_PAR_FRAME}"])
            else:
                target_q_values[action] = (
                    reward
                    + gamma
                    * target_dqn(torch.tensor(next_state, dtype=torch.float32))[
                        best_action
                    ]
                )
            optimizer.zero_grad()
            loss = criterion(q_values, target_q_values)
            loss.backward()
            optimizer.step()
            state = next_state
            epsilon = max(epsilon * epsilon_decay, epsilon_end)
            step += 1
            # Insérer ici le code pour sauvegarder le modèle, afficher les statistiques, etc.
            if debug:
                print(
                    f"action={actions[action]:<10}episode={episode:<10}player_kill={player_life_end:<10}player_end_game={player_alive:<10}reward={reward:<10}nb_mess_step={comm.number_of_messages:<10}\nstate={state}"
                )
            comm.number_of_messages=0
        response = comm.communicate(["wait_for 0"])
        if (episode + 1) % 10 == 0:
            print("Sauvegarde du modèle...")
            dqn.sauvegarder_modele()
            copier_poids(dqn, target_dqn)
        comm.communicate([f'draw_text(25,1,"Game number: {episode+1} - mean score={mean_score} - ba(c)o 2013")'])
        _d = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        sum_of_reward += reward
        mean_score_old = mean_score
        mean_score = round(sum_of_reward / (episode + 1), 2)
        if mean_score_old > mean_score:
            epsilon += 0.001
        print(
            f"Episodes N°{episode+1} [{_d}][ε={epsilon}][score={reward}][score moyen={mean_score}]"
        )
    dqn.sauvegarder_modele()


if __name__ == "__main__":
    main()
