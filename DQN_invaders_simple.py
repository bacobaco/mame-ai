from datetime import datetime
import time, keyboard, random, pygame, os

pygame.mixer.init()
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from MameCommSocket import MameCommunicator
from collections import deque


# Initialisation socket
comm = MameCommunicator("localhost", 12345)

# Constantes
MAX_PLAYER_POS = 255
MAX_SAUCER_POS = 255
MAX_BOMB_POS_X = 255
MAX_BOMB_POS_Y = 255
ACTION_DELAY = 0.01
debug = 0
flag_tir = True

# Adresses du jeu space invaders
# https://www.computerarcheology.com/Arcade/SpaceInvaders/RAMUse.html
numCoins = "20EB"
P1ScorL = "20F8"
P1ScorM = "20F9"
shotSync = "2080"  # Les 3 tirs sont synchronisés avec le minuteur GO-2. Ceci est copié à partir du minuteur dans la boucle de jeuDQN

alienShotYr = "207B"  # Delta Y du tir alien
alienShotXr = "207C"  # Delta X du tir alien
refAlienYr = "2009"  # Reference alien Yr coordinate
refAlienXr = "200A"  # Reference alien Xr coordinate
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
p1ShipsRem = "21FF"  # 	Ships remaining after current dies
gameMode = "20EF"
playerAlive = (
    "2015"  # Player is alive [FF=alive]. Toggles between 0 and 1 for blow-up images.
)
player1Alive = "20E7"  # 1 if player is alive, 0 if dead (after last man)

# Liste des actions du jeu demandées au réseau de neurone
actions = {
    0: "left",
    1: "right",
}  # , 2: "stop", 3: "tir", 4: "tir-left", 5: "tir-right"}


class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        state = np.expand_dims(state, 0)
        next_state = np.expand_dims(next_state, 0)

        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        state, action, reward, next_state, done = zip(
            *random.sample(self.buffer, batch_size)
        )
        return np.concatenate(state), action, reward, np.concatenate(next_state), done

    def __len__(self):
        return len(self.buffer)


class DQN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, n=1):
        super(DQN, self).__init__()
        self.n = n
        self.fc_hidden = []
        self.fc_input = nn.Linear(input_size, hidden_size)
        self.fc_hidden.extend(
            nn.Linear(hidden_size, hidden_size) for _ in range(self.n)
        )
        self.fc_output = nn.Linear(hidden_size, output_size)
        self.chemin = "./dqn_invaders_model_simple.pth"

    def forward(self, x):
        x = torch.relu(self.fc_input(x))
        for i in range(self.n):
            x = torch.relu(self.fc_hidden[i](x))
        x = self.fc_output(x)
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
    global flag_tir
    if actions[action] == "left":
        response = comm.communicate(["execute P1_left(1)", "execute P1_right(0)"])
    elif actions[action] == "right":
        response = comm.communicate(["execute P1_left(0)", "execute P1_right(1)"])
    if flag_tir:
        comm.communicate(["execute P1_Button_1(1)"])
    else:
        comm.communicate(["execute P1_Button_1(0)"])
    flag_tir = not flag_tir


def get_score():
    response = comm.communicate(
        [
            f"read_memory {P1ScorL}",
            f"read_memory {P1ScorM}",
            f"read_memory {plyrShotStatus}",
        ]
    )
    P1ScorL_v, P1ScorM_v, P1ShotStatus = list(map(int, response))
    if debug == 1:
        print(f"==>SHOT STATUS={P1ShotStatus}")
    # ShotStatus	0 if available, 1 if just initiated, 2 moving normally, 3 hit something besides alien, 5 if alien explosion is in progress, 4 if alien has exploded (remove from active duty)
    return (
        (P1ScorL_v >> 4) * 10
        + (P1ScorM_v & 0x0F) * 100
        + ((P1ScorM_v) >> 4) * 1000
        + (P1ShotStatus == 5)  # alien is explosing
        + (P1ShotStatus == 4) * 5  # alien has exploded
        # - (P1ShotStatus == 3) #hit shield or nothing
    )


def get_state():
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
            f"read_memory {refAlienXr}",
            f"read_memory {refAlienYr}",
        ]
    )
    player_pos, psx, psy, saucer_pos, rx, ry, px, py, sx, sy, ax, ay = list(
        map(int, response)
    )
    return np.array(
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
            ax,
            ay,
            # *extract_alien_coordinates(),
            # *extract_shield_info(),
        ]
    )


# non utilisé uniquement position de l'alien en bas à gauche
def extract_alien_coordinates():
    alien_coordinates = []
    messages = [f"read_memory {refAlienYr}", f"read_memory {refAlienXr}"]
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


# non utilisé
def extract_shield_info():
    messages = []
    for index in range(
        0xB0
    ):  # 2142:21F1	Player 1 shields remembered between rounds 44 bytes * 4 shields ($B0 bytes)
        shield_info_addr = hex(0x2142 + index)[2:]
        messages.append(f"read_memory {shield_info_addr}")
    return list(map(int, comm.communicate(messages)))


def main():
    # Utilisation de plusieurs états en entrée du réseau permettant d'avoir un historique
    # cette historique refléterait la dynamique des positions (vitesse)?
    N = 1
    global debug
    ############################################
    # Configuration réseaux et hyperparameter  #
    ############################################
    # Taille de l'entrée du réseau de neurones (à adapter en fonction de vos besoins)
    # 3*2 positions des bombes 1 position joueur et 1 pos soucoupe et 2 positions tir joueur + 2 pour ref_alien
    input_size = (10 + 2) * N
    nb_hidden_layer = 1
    hidden_size = 500
    output_size = 2  # nb actions
    # multiple state
    state_history = deque(maxlen=N)  # crée un deque avec une longueur maximale de 5
    learning_rate = 0.001  # 0.01
    gamma = 0.999  # 0.99
    num_episodes = 10000  # nb de partie quasi infini ;-)
    epsilon_start = 1
    epsilon_end = 0
    epsilon_decay = 0.99999  # 0.99999
    epsilon = epsilon_start
    reward_alive = +1
    reward_kill = -2000
    reward_mult_step = +0.01
    dqn = DQN(input_size, hidden_size, output_size, nb_hidden_layer)
    target_dqn = DQN(input_size, hidden_size, output_size, nb_hidden_layer)
    optimizer = optim.Adam(dqn.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()
    dqn.charger_modele()
    copier_poids(dqn, target_dqn)

    buffer_capacity = 2000  # autour de 1 episodes
    batch_size = 100  # autour d'un 1/20 episodes
    replay_buffer = ReplayBuffer(buffer_capacity)

    # nombre d'info ou commandes à exécuter par frame:
    # 2 pour connaître l'état du joueur
    # 3 pour calcul score 2 pour score et un pour status shot
    # 2 pour l'action (left ou right plus tir)
    # récupération des actions: 10 + 2 alien ref
    # et 1 pour le message "wait_for" lui même !
    NB_DE_DEMANDES_PAR_FRAME = str(
        2 + 3 + 2 + 10 + 2
    ) 

    # ============================ VITESSE DU JEU ======================================================
    vitesse_de_jeu = 3
    NB_DE_FRAMES_STEP = 1  # combien de frame pour un step
    # ============================ VITESSE DU JEU ======================================================

    sum_of_score = mean_score = mean_score_old = 0
    f2_pressed = False
    response = comm.communicate(
        [
            f"write_memory {numCoins}(1)",
            "execute P1_start(1)",
            f"execute throttle_rate({vitesse_de_jeu})",
            "execute throttled(0)",
            f"frame_per_step {NB_DE_FRAMES_STEP}",
        ]
    )
    print(
        f"[input={input_size//N}*{N}={input_size}]"
        f"[hidden={hidden_size}*{nb_hidden_layer}#{hidden_size**nb_hidden_layer}]"
        f"[output={output_size}]"
        f"[gamma={gamma}][learning={learning_rate}]"
        f"[reward_kill={reward_kill}][reward_alive={reward_alive}][reward_step={reward_mult_step}]"
        f"[epsilon start, end, decay={epsilon},{epsilon_end},{epsilon_decay}]"
        f"[Replay_size={buffer_capacity}&_samples={batch_size}]"
        f"[nb_mess_frame={NB_DE_DEMANDES_PAR_FRAME}]"
        f"[nb_step_frame={NB_DE_FRAMES_STEP}][speed={vitesse_de_jeu}]"
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
        # Remplit le deque avec des vecteurs d'état non nuls avec le même état de départ...
        for _ in range(N):
            state_history.append(get_state())
        state = np.concatenate(state_history)
        while player_alive == 1:
            while step == 0:
                time.sleep(5 / vitesse_de_jeu)  # attendre 5 secondes au début
                step += 1
                comm.communicate([f"wait_for {NB_DE_DEMANDES_PAR_FRAME}"])
            # Vérifier si la touche "esc" est pressée
            if keyboard.is_pressed("f2"):
                print("Touche 'F2' détectée. Sortie prématurée de la boucle.")
                f2_pressed = True
                break
            if keyboard.is_pressed("ctrl") and keyboard.is_pressed("f4"):
                debug -= debug > 0
                print(debug)
                time.sleep(0.1)
            elif keyboard.is_pressed("f4"):
                debug += debug < 2
                print(debug)
                time.sleep(0.1)
            player_life_end, player_alive = list(
                map(
                    int,
                    comm.communicate(
                        [
                            f"read_memory {playerAlive}",
                            f"read_memory {player1Alive}",
                        ]
                    ),
                )
            )

            # Choisir une action en fonction de l'état actuel et de la stratégie epsilon-greedy
            action = choisir_action(dqn, state, epsilon)
            # Exécuter l'action dans MAME et obtenir la récompense et l'état suivant
            executer_action(action)
            score = get_score()
            reward = round(score + step * reward_mult_step) + (
                reward_kill if player_life_end < 255 else reward_alive
            )  # step lower reward
            # Mettre à jour le réseau de neurones
            state_history.append(get_state())  # utilisation de plusieurs états
            next_state = np.concatenate(state_history)  # un seul vecteur aplatit
            done = player_alive == 0
            replay_buffer.push(state, action, reward, next_state, done)
            state = next_state
            epsilon = max(epsilon * epsilon_decay, epsilon_end)
            step += 1
            if len(replay_buffer) > batch_size:
                (
                    state_batch,
                    action_batch,
                    reward_batch,
                    next_state_batch,
                    done_batch,
                ) = replay_buffer.sample(batch_size)
                if debug == 2:
                    print(
                        state_batch,
                        action_batch,
                        reward_batch,
                        next_state_batch,
                        done_batch,
                    )
                # Convert to tensors
                state_batch = torch.tensor(state_batch, dtype=torch.float32)
                action_batch = torch.tensor(action_batch, dtype=torch.int64)
                reward_batch = torch.tensor(reward_batch, dtype=torch.float32)
                next_state_batch = torch.tensor(next_state_batch, dtype=torch.float32)
                done_batch = torch.tensor(done_batch, dtype=torch.float32)

                # Current Q Values
                curr_q_values = (
                    dqn(state_batch).gather(1, action_batch.unsqueeze(-1)).squeeze(-1)
                )
                # Calculate target Q Values
                next_q_values = target_dqn(next_state_batch).max(1)[0].detach()
                # Compute target of Q values
                target_q_values = (
                    reward_batch + gamma * (1 - done_batch) * next_q_values
                )
                # Compute loss
                loss = criterion(curr_q_values, target_q_values)
                # Optimize the model
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            if player_life_end < 255:  # fin du joueur
                comm.communicate(["wait_for 0"])
                time.sleep(3 / vitesse_de_jeu)  # attendre 3 secondes si vitesse normal
                comm.communicate([f"wait_for {NB_DE_DEMANDES_PAR_FRAME}"])
            # Insérer ici le code pour sauvegarder le modèle, afficher les statistiques, etc.
            if debug == 1:
                print(
                    f"action={actions[action]:<9}episode={episode:<5d}player_kill={player_life_end:<4d}player_end_game={player_alive:<2d}reward={reward:<5d}nb_mess_step={comm.number_of_messages:<4d}nb_step={step:<5d}score={score:<5d}"
                )
            comm.number_of_messages = 0
        # fin de partie
        response = comm.communicate(["wait_for 0"])
        if (episode + 1) % 10 == 0:
            print("Sauvegarde du modèle...")
            dqn.sauvegarder_modele()
            copier_poids(dqn, target_dqn)
        comm.communicate(
            [
                f'draw_text(25,1,"Game number: {episode+1:04d} - mean score={mean_score:04.0f} - ba(c)o 2013")'
            ]
        )
        _d = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        sum_of_score += score
        mean_score_old = mean_score
        mean_score = round(sum_of_score / (episode + 1), 2)
        if mean_score_old > mean_score:
            epsilon += 0.001
        print(
            f"N°{episode+1} [{_d}][nb steps={step:4d}][ε={epsilon:.4f}][reward={reward:3d}][score={score:3d}][score moyen={mean_score:.2f}]"
        )
        if mean_score == mean_score_old:
            pygame.mixer.Sound("c:\\Windows\\Media\\tada.wav").play()
    dqn.sauvegarder_modele()


if __name__ == "__main__":
    main()
