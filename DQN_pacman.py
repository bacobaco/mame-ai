from datetime import datetime
import shutil
import time, keyboard, random, pygame, os

pygame.mixer.init()  # pour le son au cas où mean_score est stable
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
from MameCommSocket import MameCommunicator
from ScreenRecorder import ScreenRecorder


# Initialisation socket port unique défini dans PythonBridgeSocket.lua
comm = MameCommunicator("localhost", 12346)

# Constantes
MAX_PLAYER_POS = 255.0
debug = 0

# Adresses du jeu Pac-Man
# https://datacrystal.romhacking.net/wiki/Pac-Man:RAM_map
CurrentScoreUnites = "0070"
CurrentScoreDizaines = "0071"
CurrentScoreCentaines = "0072"
CurrentScoreMilliers = "0073"
CurrentScoreDizainesDeMilliers = "0074"
CurrentScoreCentainesDeMilliers = "0075"

CurrentLives = "0067"
RemainingPellets = "006A"

PacManCoordX = "001A"

# Liste des actions du jeu demandées au réseau de neurone
actions = {
    0: "left",
    1: "right",
    2: "up",
    3: "down",
}


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
        self.fc_input = nn.Linear(input_size, hidden_size)
        self.fc_hidden = nn.ModuleList(
            [nn.Linear(hidden_size, hidden_size) for _ in range(self.n)]
        )
        self.fc_output = nn.Linear(hidden_size, output_size)
        self.chemin = "./dqn_pacman.pth"

    def forward(self, x):
        x = torch.relu(self.fc_input(x))
        for layer in self.fc_hidden:
            x = torch.relu(layer(x))
        x = self.fc_output(x)
        return x

    def sauvegarder_modele(self):
        torch.save(self.state_dict(), self.chemin)

    def charger_modele(self):
        if os.path.exists(self.chemin):
            self.load_state_dict(torch.load(self.chemin))
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
    # go to see mame command for pac-man in PythonBridgeSocket.lua
    if actions[action] == "left":
        response = comm.communicate(
            [
                "execute P1_Left(1)",
                "execute P1_Right(0)",
                "execute P1_Up(0)",
                "execute P1_Down(0)",
            ]
        )
    elif actions[action] == "right":
        response = comm.communicate(
            [
                "execute P1_Left(0)",
                "execute P1_Right(1)",
                "execute P1_Up(0)",
                "execute P1_Down(0)",
            ]
        )
    elif actions[action] == "up":
        response = comm.communicate(
            [
                "execute P1_Left(0)",
                "execute P1_Right(0)",
                "execute P1_Up(1)",
                "execute P1_Down(0)",
            ]
        )
    elif actions[action] == "down":
        response = comm.communicate(
            [
                "execute P1_Left(0)",
                "execute P1_Right(0)",
                "execute P1_Up(0)",
                "execute P1_Down(1)",
            ]
        )


def get_score():
    response = comm.communicate(
        [
            f"read_memory {CurrentScoreUnites}",
            f"read_memory {CurrentScoreDizaines}",
            f"read_memory {CurrentScoreCentaines}",
            f"read_memory {CurrentScoreMilliers}",
            f"read_memory {CurrentScoreDizainesDeMilliers}",
            f"read_memory {CurrentScoreCentainesDeMilliers}",
        ]
    )
    print(list(map(int, response)))

    return ()


def get_state():
    # Lire les informations de la mémoire du jeu position du player et soucoupe
    response = comm.communicate(
        [
            f"read_memory {PacManCoordX}",
        ]
    )
    player_pos = list(map(int, response))
    return np.array(
        [
            player_pos,
        ]
    )


def nb_parameters(e, H, n, s):
    # Poids et biais de la première couche cachée
    parameters = e * n + n
    # Poids et biais des couches cachées intermédiaires
    if H > 1:
        parameters += (H - 1) * (n * n + n)
    # Poids et biais de la couche de sortie
    parameters += n * s + s
    return parameters


def create_fig(nb_parties, scores_moyens, fenetre, epsilons):
    import matplotlib.pyplot as plt

    print("===> Création du graphe f(épisodes)= scores_moyens")
    fig, ax1 = plt.subplots()

    color = "tab:blue"
    # L'axe des y de ax1 représentera epsilon
    ax1.set_xlabel("Episodes/Parties")
    ax1.set_ylabel("Epsilon", color=color)
    ax1.plot(epsilons, color=color, linestyle="dashed")
    ax1.tick_params(axis="y", labelcolor=color)
    # Ajout d'une grille pour l'axe des y de ax1
    ax1.grid(True, which="both", axis="both", linestyle="-", linewidth=0.5)
    # Création d'un deuxième axe des y pour l'évolution score moyen
    ax2 = ax1.twinx()

    color = "tab:red"
    # L'axe des y de ax2 représentera le score moyen
    ax2.set_ylabel(
        f"Score moyen (pour {fenetre} parties)", color=color, rotation=270, labelpad=10
    )
    ax2.plot(scores_moyens, color=color)
    ax2.tick_params(axis="y", labelcolor=color)
    # Ajout d'une grille pour l'axe des y de ax2
    ax2.grid(True, which="both", axis="both", linestyle="-", linewidth=0.5)

    # Calcul des coefficients de la ligne de tendance
    coefficients = np.polyfit(range(len(scores_moyens)), scores_moyens, 1)
    pente = coefficients[0]  # Pente de la ligne de tendance
    # Génération des valeurs y pour la ligne de tendance
    trendline = np.poly1d(coefficients)
    # Tracé de la ligne de tendance
    ax2.plot(trendline(range(len(scores_moyens))), color="tab:orange")
    # Calcul de l'angle de la ligne de tendance en degrés
    angle = np.arctan(pente) * 180 / np.pi
    midpoint = len(scores_moyens) / 2
    ax2.text(
        midpoint,
        trendline(midpoint) + trendline(midpoint) * 0.01,
        f"Pente = {pente:.2f}",
        color="tab:orange",
        ha="center",
        rotation=angle,
        rotation_mode="anchor",
        transform_rotates_text=True,
    )

    fig.tight_layout()

    plt.title(f"Evolution du score moyen et d'Epsilon sur {str(nb_parties)} épisodes")

    # Sauvegarde du graphique dans un fichier
    plt.savefig("evolution_score_moyen_et_epsilon.png", dpi=300, bbox_inches="tight")

    return pente


def main():
    # Utilisation de plusieurs états en entrée du réseau permettant d'avoir un historique
    # cette historique refléterait la dynamique des positions (vitesse)?
    N = 1
    global debug
    ############################################
    # Configuration réseaux et hyperparameter  #
    ############################################
    # Taille de l'entrée du réseau de neurones (à adapter en fonction de vos besoins)
    # position pacman X
    input_size = (1) * N
    nb_hidden_layer = 1
    hidden_size = 10
    output_size = 4  # nb actions
    # multiple state
    state_history = deque(maxlen=N)  # crée un deque avec une longueur maximale de 5
    learning_rate = 0.001  # 0.01
    gamma = 0.9999  # 0.99
    num_episodes = 10000  # nb de partie quasi infini ;-)
    epsilon_start = 0.99
    epsilon_end = 0.0
    epsilon_decay = 0.99999  # 0.99999
    epsilon_add = 0.005  # ajoute en fin de partie un peu d'epsilon si score inférieur à la moyenne
    epsilon = epsilon_start
    reward_alive = 0  # 1
    reward_kill = -5000
    reward_mult_step = 0.0  # 0.01

    buffer_capacity = 10000  # taille des step cumulés
    batch_size = 1000  # taille des steps pris au hasard dans le buffer_capacity pour modif le model
    replay_buffer = ReplayBuffer(buffer_capacity)

    # nombre d'info ou commandes à exécuter par frame:
    # 2 pour connaître l'état du joueur
    # 3 pour calcul score 2 pour score et un pour status shot
    # 3 pour l'action (left ou right ou stop ou 1 tir)
    # récupération des états: 10+1 pos alienY
    NB_DE_DEMANDES_PAR_FRAME = str(2 + 3 + 3 + 10 + 1)

    # ============================ VITESSE DU JEU ======================================================
    vitesse_de_jeu = 1
    NB_DE_FRAMES_STEP = 4  # combien de frame pour un step
    # ==================================================================================================

    ##########Créer lee deux réseaux de neurones
    dqn = DQN(input_size, hidden_size, output_size, nb_hidden_layer)
    target_dqn = DQN(input_size, hidden_size, output_size, nb_hidden_layer)
    optimizer = optim.Adam(dqn.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()
    dqn.charger_modele()
    copier_poids(dqn, target_dqn)

    record = False  # Utilise OBS socket server
    if record:
        recorder = ScreenRecorder()
    fenetre_du_calcul_de_la_moyenne = 100
    collection_of_score = deque(
        maxlen=fenetre_du_calcul_de_la_moyenne
    )  # permet de ne prendre en compte que 100 scores dans la moyenne
    list_of_mean_scores = []
    list_of_epsilons = []
    mean_score = mean_score_old = last_score = 0
    f2_pressed = False
    response = comm.communicate(
        [
            "execute P1_start(1)",
            f"execute throttle_rate({vitesse_de_jeu})",
            "execute throttled(0)",
            f"frame_per_step {NB_DE_FRAMES_STEP}",
        ]
    )
    print(
        f"[input={input_size//N}*{N}={input_size}]"
        f"[hidden={hidden_size}*{nb_hidden_layer}#{nb_parameters(input_size,nb_hidden_layer,hidden_size,output_size)}]"
        f"[output={output_size}]"
        f"[gamma={gamma}][learning={learning_rate}]"
        f"[reward_kill={reward_kill}][reward_alive={reward_alive}][reward_step={reward_mult_step}]"
        f"[epsilon start, end, decay, add={epsilon},{epsilon_end},{epsilon_decay},{epsilon_add}]"
        f"[Replay_size={buffer_capacity}&_samples={batch_size}]"
        f"[nb_mess_frame={NB_DE_DEMANDES_PAR_FRAME}]"
        f"[nb_step_frame={NB_DE_FRAMES_STEP}][speed={vitesse_de_jeu}]"
    )
    for episode in range(num_episodes):
        if record:
            recorder.start_recording()
        step = reward = sum_rewards = player_life_end = player_alive = score = 0
        flag_create_fig = False
        # the "F2" key to exit the loop prematurely
        # the "F3" key to reset the debug level.
        # "F4" key to increase the debug level.
        # "F7" key create fig of scores/episodes.
        if f2_pressed:
            print("Sortie prématurée de la boucle 'for'.")
            break
        response = comm.communicate(["execute Coin_1(1)"])
        while player_alive == 0:
            player_alive = int(comm.communicate([f"read_memory {CurrentLives}"])[0])
        # Remplit le deque avec des vecteurs d'état non nuls avec le même état de départ...
        for _ in range(N):
            state_history.append(get_state())
        state = np.concatenate(state_history)
        while player_alive >= 1:
            while step == 0:
                time.sleep(5 / vitesse_de_jeu)  # attendre 5 secondes au début
                step += 1
                comm.communicate([f"wait_for {NB_DE_DEMANDES_PAR_FRAME}"])
            # Vérifier si la touche "esc" est pressée
            if keyboard.is_pressed("f2"):
                print("Touche 'F2' détectée. Sortie prématurée de la boucle.")
                f2_pressed = True
                break
            elif keyboard.is_pressed("f3") and debug > 0:
                debug = 0
                print(f"debug={debug}")
            elif keyboard.is_pressed("f4"):
                debug += debug < 3
                print(f"debug={debug}")
                time.sleep(0.2)
            elif keyboard.is_pressed("f7") and not flag_create_fig:
                print("<=== Demande création d'une figure des scores/episodes")
                flag_create_fig = True
            player_life_end, player_alive = list(
                map(
                    int,
                    comm.communicate(
                        [
                            f"read_memory {CurrentLives}",
                            f"read_memory {CurrentLives}",
                        ]
                    ),
                )
            )
            # Choisir une action en fonction de l'état actuel et de la stratégie epsilon-greedy
            action = choisir_action(dqn, state, epsilon)
            # Exécuter l'action dans MAME et obtenir la récompense et l'état suivant
            executer_action(action)
            _last_score = score
            score = get_score()
            reward = (
                round(score + step * reward_mult_step)
                + (reward_kill if player_life_end < 255 else reward_alive)
                - _last_score
            )
            # Mettre à jour le réseau de neurones
            state_history.append(
                get_state()
            )  # ajouter un état dans la collection history actuellement pas utilisé capacity=1
            next_state = np.concatenate(state_history)  # un seul vecteur aplatit
            done = player_alive == 0  # done = fin d'épisode
            if debug >= 2:
                print(f"state={state}\nnext ={next_state}")
            reward += (
                next_state[-1] - state[-1]
            ) * 20  # pénalité lorsque l'alien ref Y diminue (descente des aliens)
            sum_rewards += reward
            replay_buffer.push(state, action, reward, next_state, done)
            state = next_state
            epsilon = max(
                epsilon * epsilon_decay,
                epsilon_end,
            )
            step += 1
            if len(replay_buffer) >= batch_size:
                (
                    state_batch,
                    action_batch,
                    reward_batch,
                    next_state_batch,
                    done_batch,
                ) = replay_buffer.sample(batch_size)
                if debug >= 3:
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
            if debug >= 1:
                print(
                    f"action={actions[action]:<8}episode={episode:<5d}player_kill={player_life_end:<4d}player_end_game={player_alive:<2d}reward={reward:<5d}all rewards={sum_rewards:<6d}nb_mess_step={comm.number_of_messages:<4d}nb_step={step:<5d}score={score:<5d}"
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
        if record:
            if score > last_score:
                time.sleep(1)  # on garde le "game over" de la fin de partie
            recorder.stop_recording()
            time.sleep(
                0.55
            )  # attente pour laisser le temps à obs d'arrêter l'enregistrement...
            if score > last_score:
                time.sleep(
                    2
                )  # visiblement obs prends son temps pour écraser l'ancien fichier...
                shutil.copy("output-obs.mp4", "best_game.avi")
                last_score = score
        if flag_create_fig:
            _pente = create_fig(
                episode,
                list_of_mean_scores,
                fenetre_du_calcul_de_la_moyenne,
                list_of_epsilons,
            )
            if _pente < 0.1 and episode > 200:
                break
            time.sleep(0.2)
        collection_of_score.append(score)
        mean_score_old = mean_score
        mean_score = round(sum(collection_of_score) / len(collection_of_score), 2)
        list_of_mean_scores.append(mean_score)
        list_of_epsilons.append(epsilon)
        if mean_score_old > mean_score:
            epsilon += epsilon_add
        if mean_score == mean_score_old:
            pygame.mixer.Sound("c:\\Windows\\Media\\tada.wav").play()
        _d = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(
            f"N°{episode+1} [{_d}][nb steps={step:4d}][ε={epsilon:.4f}][rewards={sum_rewards:4d}][score={score:3d}][score moyen={mean_score:.2f}]"
        )
    dqn.sauvegarder_modele()
    time.sleep(5)
    if record:
        print(recorder.stop_recording())
        time.sleep(5)
        print(recorder.ws.disconnect())


if __name__ == "__main__":
    main()
