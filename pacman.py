from datetime import datetime
import shutil
import time, keyboard, random, pygame, os, math, psutil,win32gui

pygame.mixer.init()  # pour le son au cas où mean_score est stable
from colorama import Fore, Style, Back
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque, Counter
from MameCommSocket import MameCommunicator
from ScreenRecorder import ScreenRecorder

import subprocess
from torch.utils.tensorboard import SummaryWriter

# Nouvelle importation des classes depuis AI_Mame.py
from AI_Mame import TrainingConfig, ReplayBuffer, DQN, DQNTrainer

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Utilisation de l'appareil : {device}")

# shortcut = 'E:\\Emulateurs\\Mame Officiel\\mame.exe - puckman - comm python socket.lnk'
# process = subprocess.Popen(f'cmd.exe /c start "" "{shortcut}"')
command = [
    "E:\\Emulateurs\\Mame Officiel\\mame.exe",
    "-artwork_crop",
    "-console",
    "-noautosave",
    "pacman",
    "-autoboot_delay",
    "1",
    "-autoboot_script",
    "E:\\Emulateurs\\Mame Sets\\MAME EXTRAs\\plugins\\PythonBridgeSocket.lua",
]
process = subprocess.Popen(command, cwd="E:\\Emulateurs\\Mame Officiel")
time.sleep(10)

# Initialisation socket port unique défini dans PythonBridgeSocket.lua
comm = MameCommunicator("localhost", 12346)
time.sleep(5)

# Adresses du jeu Pac-Man
# Adresses mémoires trouvées via le debugger de MAME
AdScoreDizaines = "4E80"  
AdScoreCentaines = "4E81"  
AdScoreMilliers = "4E82"  
AdCredits = "4E6E"
AdPillsAccumulator = "4E0E"  
AdNbPlayerLive = "4E14"  
AdPlayerAlive = "4EAE"  
AdPacManPosX = "4C1A"  
AdPacManPosY = "4C1B"  
AdBlinkyPosX = "4C12"  
AdBlinkyPosY = "4C13"
AdPinkyPosX = "4C14"  
AdPinkyPosY = "4C15"
AdInkyPosX = "4C16"  
AdInkyPosY = "4C17"
AdClydePosX = "4C18"  
AdClydePosY = "4C19"

# La mémoire vidéo se trouve entre $6000 et $63FF (1024 octets)
AdVideoRamStart = "4000"
AdVideoRamEnd = "43FF"
VideoRamLong = 1024

# Liste des actions du jeu demandées au réseau de neurones
actions = {0: "left", 1: "right", 2: "up", 3: "down"}

def executer_action(action):
    if actions[action] == "left":
        comm.communicate([
            "execute P1_Left(1)",
            "execute P1_Right(0)",
            "execute P1_Up(0)",
            "execute P1_Down(0)",
        ])
    elif actions[action] == "right":
        comm.communicate([
            "execute P1_Left(0)",
            "execute P1_Right(1)",
            "execute P1_Up(0)",
            "execute P1_Down(0)",
        ])
    elif actions[action] == "up":
        comm.communicate([
            "execute P1_Left(0)",
            "execute P1_Right(0)",
            "execute P1_Up(1)",
            "execute P1_Down(0)",
        ])
    elif actions[action] == "down":
        comm.communicate([
            "execute P1_Left(0)",
            "execute P1_Right(0)",
            "execute P1_Up(0)",
            "execute P1_Down(1)",
        ])


def get_score():
    response = comm.communicate([
        f"read_memory {AdScoreDizaines}",
        f"read_memory {AdScoreCentaines}",
        f"read_memory {AdScoreMilliers}",
    ])
    dizaines, centaines, dizaines_de_milliers = map(int, response)
    dizaines = int(hex(dizaines)[2:])
    centaines = int(hex(centaines)[2:])
    dizaines_de_milliers = int(hex(dizaines_de_milliers)[2:])
    return dizaines + centaines * 100 + dizaines_de_milliers * 10000

def get_state():
    # Lecture de la mémoire vidéo (0x4000 - 0x43FF) en un seul appel
    # On utilise la commande "read_memory_range startAddr(hex)(lengthDecimal)"
    response = comm.communicate([f"read_memory_range {AdVideoRamStart}({VideoRamLong})"])
    # La réponse est un tableau de 1024 valeurs, séparées par des virgules dans une seule chaîne
    video_data = np.array(list(map(int, response[0].split(","))))

    # Lecture des positions : Pac-Man et les 4 fantômes
    positions_response = comm.communicate([
        f"read_memory {AdPacManPosX}",
        f"read_memory {AdPacManPosY}",
        f"read_memory {AdBlinkyPosX}",
        f"read_memory {AdBlinkyPosY}",
        f"read_memory {AdPinkyPosX}",
        f"read_memory {AdPinkyPosY}",
        f"read_memory {AdInkyPosX}",
        f"read_memory {AdInkyPosY}",
        f"read_memory {AdClydePosX}",
        f"read_memory {AdClydePosY}",
    ])
    positions = np.array(list(map(int, positions_response)))

    # Concaténation de la vidéo et des positions pour constituer l'état complet
    state = np.concatenate([video_data, positions])
    return state

def nb_parameters(e, H, n, s):
    parameters = e * n + n
    if H > 1:
        parameters += (H - 1) * (n * n + n)
    parameters += n * s + s
    return parameters

def create_fig(nb_parties, scores_moyens, fenetre, epsilons, rewards, nb_steps, nb_frames_per_step, filename="Pacman_fig"):
    import matplotlib.pyplot as plt
    if nb_parties < 10:
        print(f"{Fore.RED}Attention ! Création du graphe impossible: le nb de partie est inférieur à 10 !")
        return 0
    else:
        print(f"===> Création du graphe f(épisode)= score moyen (fenêtre de calcul pour {fenetre} parties)")
        scores_moyens[:10] = [sum(scores_moyens[:10]) / 10] * 10
    plt.close("all")
    fig, ax1 = plt.subplots(figsize=(20, 5))
    ax1.set_xlabel("Episodes/Parties")
    color = "tab:blue"
    ax1.set_ylabel("Epsilon en  %", color=color)
    ax1.plot(list(map(lambda x: int(x * 100), epsilons)), color=color, linestyle="--", zorder=10)
    ax1.tick_params(axis="y", labelcolor=color)
    ax1.set_yticks(ax1.get_yticks())
    ax1.set_yticklabels(map(str, ax1.get_yticks()), rotation=50)
    ax1.grid(True, which="both", axis="both", linestyle="-", linewidth=0.1, color=color)
    ax2 = ax1.twinx()
    color = "tab:red"
    ax2.set_ylabel(f"Score moyen (pour {fenetre} parties)", color=color, rotation=270, labelpad=10)
    ax2.plot(scores_moyens, color=color, linestyle="-", zorder=20)
    ax2.tick_params(axis="y", labelcolor=color)
    ax2.set_yticks(ax2.get_yticks())
    ax2.set_yticklabels(map(str, ax2.get_yticks()), rotation=50)
    ax2.grid(True, which="both", axis="both", linestyle="-", linewidth=0.5, color=color)
    min_score = min(scores_moyens)
    max_score = max(scores_moyens)
    min_score_idx = scores_moyens.index(min_score)
    max_score_idx = scores_moyens.index(max_score)
    ax2.annotate(f"Min: {min_score}", xy=(min_score_idx, min_score), xytext=(-10, -10),
                 textcoords="offset points", ha="center", va="bottom",
                 bbox=dict(boxstyle="round,pad=0.5", fc="yellow", alpha=0.5),
                 arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0"))
    ax2.annotate(f"Max: {max_score}", xy=(max_score_idx, max_score), xytext=(10, 10),
                 textcoords="offset points", ha="center", va="top",
                 bbox=dict(boxstyle="round,pad=0.5", fc="yellow", alpha=0.5),
                 arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0"))
    ax3 = ax1.twinx()
    ax3.spines["right"].set_position(("outward", 45))
    color = "tab:green"
    ax3.set_ylabel("Rewards", color=color, rotation=270, labelpad=10)
    ax3.plot(rewards, color=color, linestyle=":", linewidth=0.5, zorder=5)
    ax3.tick_params(axis="y", labelcolor=color)
    ax3.set_yticks(ax3.get_yticks())
    ax3.set_yticklabels(map(str, ax3.get_yticks()), rotation=50)
    ax3.grid(True, which="both", axis="both", linestyle="-.", linewidth=0.3, color=color)
    ax4 = ax1.twinx()
    ax4.spines["right"].set_position(("outward", 100))
    color = "tab:purple"
    ax4.set_ylabel(f"Nb Steps (nb {nb_frames_per_step} frames/step)", color=color, rotation=270, labelpad=10)
    ax4.plot(nb_steps, color=color, linewidth=0.5, zorder=1)
    ax4.tick_params(axis="y", labelcolor=color)
    ax4.set_yticks(ax4.get_yticks())
    ax4.set_yticklabels(map(str, ax4.get_yticks()), rotation=50)
    ax4.grid(True, which="both", axis="both", linestyle="-", linewidth=0.3, color=color)
    x_min, x_max = ax2.get_xlim()
    coefficients = np.polyfit(range(len(scores_moyens)), scores_moyens, 1)
    pente = coefficients[0]
    trendline = np.poly1d(coefficients)
    trendline_x = np.linspace(x_min, x_max, len(scores_moyens))
    ax2.plot(trendline_x, trendline(trendline_x), color="tab:orange")
    angle = np.arctan(pente) * 180 / np.pi
    midpoint = len(scores_moyens) / 2
    ax2.text(midpoint, trendline(midpoint) + trendline(midpoint) * 0.001, f"Pente = {pente:.2f}",
             color="tab:orange", ha="center", weight="bold",
             bbox=dict(facecolor="white", alpha=0.5, edgecolor="none"),
             rotation=angle, rotation_mode="anchor", transform_rotates_text=True)
    fig.tight_layout()
    plt.title(f"Evolution du score moyen et d'Epsilon sur {str(nb_parties)} épisodes")
    plt.savefig(filename+"_"+str(nb_parties)+"_"+datetime.now().strftime("%Y%m%d%H%M")+".png", dpi=300, bbox_inches="tight")
    return pente

def main():
    global f2_pressed, flag_create_fig, debug
    debug=0
    f2_pressed = flag_create_fig= False
    vitesse_de_jeu = 10
    NB_DE_FRAMES_STEP = 5  # nombre de frames par step
    N = 5
    num_episodes = 10000
    ############################################
    # Configuration réseaux et hyperparamètres   #
    ############################################
    # On utilise désormais la mémoire vidéo (1024 valeurs) + 10 valeurs pour les positions (Pac-Man et 4 fantômes)
    input_size = (VideoRamLong + 10) * N
    nb_hidden_layer = 2
    hidden_size = 128
    output_size = 4  # 4 actions possibles : left, right, up, down
    state_history = deque(maxlen=N)
    learning_rate = 0.001
    gamma = 0.9999
    bExploration = True
    epsilon_start = 0.5
    epsilon_end = 0.01
    epsilon_decay = 0.99 ** (1.0 / NB_DE_FRAMES_STEP)
    epsilon_add = 0.0000
    reward_kill = -5000
    reward_mult_step = 0.05*NB_DE_FRAMES_STEP
    buffer_capacity = 200000 if bExploration else 1
    batch_size = 256 if bExploration else 1

    # Nouvelle création de la configuration et du trainer à partir de AI_Mame.py
    NB_DE_DEMANDES_PAR_FRAME = str(1+10+3+4+2)
    config = TrainingConfig(
        input_size=input_size,
        hidden_size=hidden_size,
        output_size=output_size,
        hidden_layers=nb_hidden_layer,
        learning_rate=learning_rate,
        gamma=gamma,
        epsilon_start=epsilon_start,
        epsilon_end=epsilon_end,
        epsilon_decay=epsilon_decay,
        epsilon_add=epsilon_add,
        buffer_capacity=buffer_capacity,
        batch_size=batch_size,
        state_history_size=N,
        double_dqn=True  # Active le Double DQN
    )
    trainer = DQNTrainer(config)
    
    writer = SummaryWriter('runs/experiment_1')
    record = False  # Utilise OBS socket server
    if record:
        recorder = ScreenRecorder()
    fenetre_du_calcul_de_la_moyenne = 100
    collection_of_score = deque(maxlen=fenetre_du_calcul_de_la_moyenne)
    list_of_mean_scores, list_of_epsilons, list_of_rewards, list_of_nb_steps = [], [], [], []
    mean_score = mean_score_old = last_score = 0

    def is_terminal_in_focus():
        hwnd = win32gui.GetForegroundWindow()
        title = win32gui.GetWindowText(hwnd)
        return title[-20:] == "- Visual Studio Code"
    def on_key_press(event):
        global f2_pressed, flag_create_fig, debug
        if is_terminal_in_focus():
            # Code à exécuter lorsque la touche est pressée et que le terminal est en focus
            if keyboard.is_pressed("shift") and keyboard.is_pressed("ctrl") and keyboard.is_pressed("f2"):
                print("Touche 'F2' détectée. Sortie prématurée de la boucle.")
                f2_pressed = True
            elif keyboard.is_pressed("shift") and keyboard.is_pressed("ctrl") and keyboard.is_pressed("f3") and debug > 0:
                debug = 0
                print(f"debug={debug}")
            elif keyboard.is_pressed("shift") and keyboard.is_pressed("ctrl") and keyboard.is_pressed("f4"):
                debug += debug < 3
                print(f"debug={debug}")
                time.sleep(0.2)
            elif keyboard.is_pressed("shift") and keyboard.is_pressed("ctrl") and keyboard.is_pressed("f7") and not flag_create_fig:
                print("<=== Demande création d'une figure des scores/episodes")
                flag_create_fig = True
    keyboard.on_press(on_key_press)
    time.sleep(5 / vitesse_de_jeu)  # on attend un peu avant de commencer le jeu
    response = comm.communicate([
        "execute P1_start(1)",
        f"execute throttle_rate({vitesse_de_jeu})",
        "execute throttled(0)",
        f"frame_per_step {NB_DE_FRAMES_STEP}",
    ])
    print(
        Style.BRIGHT + Fore.RED + f"[input={input_size//N}*{N}={input_size}]"
        f"[hidden={hidden_size}*{nb_hidden_layer}#{nb_parameters(input_size, nb_hidden_layer, hidden_size, output_size)}]"
        f"[output={output_size}]"
        f"[gamma={gamma}][learning={learning_rate}]"
        f"[reward_kill={reward_kill}][reward_step={reward_mult_step}]",
        end="\n",
    )
    if bExploration:
        print(f"[epsilon start={epsilon_start:.2f} end={epsilon_end:.2f} decay={epsilon_decay:.5f} add={epsilon_add:.2f}]", end="")
    else:
        print(["epsilon=MODE EXPLOITATION"], end="")
    print(f"[Replay_size={buffer_capacity}&_samples={batch_size}]"
          f"[nb_mess_frame={NB_DE_DEMANDES_PAR_FRAME}]"
          f"[nb_step_frame={NB_DE_FRAMES_STEP}][speed={vitesse_de_jeu}]" + Style.RESET_ALL)
    for episode in range(num_episodes): 
        if record:
            recorder.start_recording()
        step = reward = sum_rewards = player_nb_vies = player_alive = score = 0
        if f2_pressed:
            print("Sortie prématurée de la boucle 'for'.")
            break
        response = comm.communicate([f"write_memory {AdCredits}(1)"])
        while player_alive == 0:
            player_alive = int(comm.communicate([f"read_memory {AdPlayerAlive}"])[0])
        player_nb_vies = 3
        flag_in_game = False
        comm.communicate([f"wait_for {(1+10)*N}"])
        for _ in range(N):
            state_history.append(get_state())
            trainer.state_history.append(get_state())  # Mise à jour de l'historique dans le trainer
        state = np.concatenate(state_history)
        while player_nb_vies > 0:
            if not flag_in_game:
                comm.communicate([f"wait_for {NB_DE_DEMANDES_PAR_FRAME}"])
                flag_in_game = True

            player_nb_vies, player_alive = list(map(
                int,
                comm.communicate([f"read_memory {AdNbPlayerLive}", f"read_memory {AdPlayerAlive}"])
            ))
            # Utilisation de la méthode select_action du trainer au lieu de choisir_action
            action = trainer.select_action(state)
            if debug >= 2: print(f"Action choisi={action}")
            executer_action(action)
            _last_score = score
            score = get_score()
            reward = round((score - _last_score) ** 2 + (reward_kill if player_alive == 0 else step * reward_mult_step))
            _sign = np.sign(reward)
            reward = round(_sign * (abs(reward) ** 0.5))
            state_history.append(get_state())
            trainer.state_history.append(get_state())  # Mise à jour de l'historique dans le trainer
            next_state = np.concatenate(state_history)
            done = (player_nb_vies == 1 and player_alive == 0)
            if debug >= 2:
                print(f"state={state}\nnext ={next_state}")
            sum_rewards += reward
            # Utilisation du replay_buffer du trainer
            trainer.replay_buffer.push(state, action, reward, next_state, done)
            state = next_state
            step += 1
            # Remplacement du bloc de training manuel par la méthode train_step du trainer
            loss = trainer.train_step()
            if loss is not None:
                writer.add_scalar('Loss/train', loss, step)
            while player_alive == 0 and player_nb_vies > 0:
                if flag_in_game:
                    response = comm.communicate(["wait_for 2"])
                    flag_in_game = False
                player_nb_vies, player_alive = list(map(
                    int,
                    comm.communicate([f"read_memory {AdNbPlayerLive}", f"read_memory {AdPlayerAlive}"])
                ))
            if debug >= 1:
                print(
                    Fore.BLUE + f"action={actions[action]:<8}episode={episode:<5d}player dans le jeu={player_alive:<2d}nb_vies={player_nb_vies:<2d}"
                    f" reward={reward:<6d}all rewards={sum_rewards:<7d}nb_mess_step={comm.number_of_messages:<4d}nb_step={step:<5d}score={score:<5d}"
                    + Style.RESET_ALL
                )
            comm.number_of_messages = 0
        response = comm.communicate(["wait_for 1"])
        if (episode + 1) % 10 == 0:
            print("Sauvegarde du modèle...")
            # Sauvegarde via torch.save sur le modèle du trainer
            torch.save(trainer.dqn.state_dict(), "./dqn_pacman.pth")
        if (episode + 1) % 100 == 0:  # Mise à jour complète tous les 100 épisodes
            print("Synchronisation complète du réseau cible...")
            DQNTrainer.copy_weights(trainer.dqn, trainer.target_dqn)
        comm.communicate([f'draw_text(25,1,"Game number: {episode+1:04d} - mean score={mean_score:04.0f} - ba(c)o 2023")'])
        if record:
            if score > last_score:
                time.sleep(1)
            recorder.stop_recording()
            time.sleep(0.55)
            if score > last_score:
                time.sleep(2)
                shutil.copy("output-obs.mp4", "best_game.avi")
                last_score = score
        if flag_create_fig:
            _pente = create_fig(
                episode,
                list_of_mean_scores,
                fenetre_du_calcul_de_la_moyenne,
                list_of_epsilons,
                list_of_rewards,
                list_of_nb_steps,
                NB_DE_FRAMES_STEP,
            )
            flag_create_fig = False
            time.sleep(0.2)
        collection_of_score.append(score)
        mean_score_old = mean_score
        mean_score = round(sum(collection_of_score) / len(collection_of_score), 2)
        list_of_mean_scores.append(mean_score)
        list_of_epsilons.append(trainer.epsilon)
        list_of_rewards.append(sum_rewards)
        list_of_nb_steps.append(step)
        # Mise à jour de epsilon via le trainer
        trainer.update_epsilon(mean_score, mean_score_old)
        if mean_score == mean_score_old:
            pygame.mixer.Sound("c:\\Windows\\Media\\tada.wav").play()
        _d = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(
            Fore.GREEN + f"N°{episode+1} [{_d}][nb steps={step:4d}][ε={trainer.epsilon:.4f}][rewards={sum_rewards:4d}][score={score:3d}][score moyen={mean_score:.2f}]"
            + Fore.RESET
        )
    torch.save(trainer.dqn.state_dict(), "./dqn_pacman.pth")
    time.sleep(5)
    if record:
        print(recorder.stop_recording())
        time.sleep(5)
        print(recorder.ws.disconnect())
    print(create_fig(
        episode,
        list_of_mean_scores,
        fenetre_du_calcul_de_la_moyenne,
        list_of_epsilons,
        list_of_rewards,
        list_of_nb_steps,
        NB_DE_FRAMES_STEP,
        filename="PACMAN_FINAL",
    ))
    print(process.terminate())
    print(
        Style.BRIGHT + Fore.RED + f"[input={input_size//N}*{N}={input_size}]"
        f"[hidden={hidden_size}*{nb_hidden_layer}#{nb_parameters(input_size, nb_hidden_layer, hidden_size, output_size)}]"
        f"[output={output_size}]"
        f"[gamma={gamma}][learning={learning_rate}]"
        f"[reward_kill={reward_kill}][reward_step={reward_mult_step}]",
        end="\n",
    )
    if bExploration:
        print(f"[epsilon start={epsilon_start:.2f} end={epsilon_end:.2f} decay={epsilon_decay:.5f} add={epsilon_add:.2f}]", end="")
    else:
        print(["epsilon=MODE EXPLOITATION"], end="")
    print(f"[Replay_size={buffer_capacity}&_samples={batch_size}]"
          f"[nb_mess_frame={NB_DE_DEMANDES_PAR_FRAME}]"
          f"[nb_step_frame={NB_DE_FRAMES_STEP}][speed={vitesse_de_jeu}]" + Style.RESET_ALL)


if __name__ == "__main__":
    main()
