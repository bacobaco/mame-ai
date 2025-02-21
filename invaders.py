# file: Invaders_ChatGPT.py : Reinforcement Learning for Space Invaders
from datetime import datetime
import shutil
import time, keyboard, random, pygame, os, psutil, win32gui

from colorama import Fore, Style, Back
pygame.mixer.init()  # pour le son au cas où mean_score est stable
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
from MameCommSocket import MameCommunicator
from ScreenRecorder import ScreenRecorder
import subprocess
import base64
import zlib

# Importer les classes de ai_mame.py
from AI_Mame import TrainingConfig, ReplayBuffer, DQN, DQNTrainer

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Utilisation de l'appareil : {device}")

# Lancement de MAME
command = [
    "E:\\Emulateurs\\Mame Officiel\\mame.exe",
    "-artwork_crop",
    "-console",
    "-noautosave",
    "invaders",
    "-autoboot_delay",
    "1",
    "-autoboot_script",
    "E:\\Emulateurs\\Mame Sets\\MAME EXTRAs\\plugins\\PythonBridgeSocket.lua",
]
process = subprocess.Popen(command, cwd="E:\\Emulateurs\\Mame Officiel")
time.sleep(12)

# Initialisation socket
comm = MameCommunicator("localhost", 12345)

# Constantes
MAX_PLAYER_POS = 255.0
MAX_SAUCER_POS = 255.0
MAX_BOMB_POS_X = 255.0
MAX_BOMB_POS_Y = 255.0
ACTION_DELAY = 0.01
debug = 0
flag_tir = True
NB_DECOUPAGES = 1

# Adresses du jeu Space Invaders: https://computerarcheology.com/Arcade/SpaceInvaders/RAMUse.html 
numCoins = "20EB"
P1ScorL = "20F8"
P1ScorM = "20F9"
shotSync = "2080"
alienShotYr = "207B"
alienShotXr = "207C"
refAlienYr = "2009"
refAlienXr = "200A"
rolShotYr = "203D"
rolShotXr = "203E"
squShotYr = "205D"
squShotXr = "205E"
pluShotYr = "204D"
pluSHotXr = "204E"
playerAlienDead = "2100"
saucerDeltaX = "208A"
playerXr = "201B"
plyrShotStatus = "2025"
obj1CoorYr = "2029"
obj1CoorXr = "202A"
p1ShipsRem = "21FF"
gameMode = "20EF"
playerAlive = "2015"
player1Alive = "20E7"

# Liste des actions
actions = {
    0: "left",
    1: "right",
    2: "stop",
    3: "tir"
}

def toggle_debug_lua():
    global debug_lua
    debug_lua = not debug_lua
    if debug_lua:
        comm.communicate(["debug on"])
        print("Debug mode activé (debug=1)")
    else:
        comm.communicate(["debug off"])
        print("Debug mode désactivé (debug=0)")


# Fonctions spécifiques au jeu (inchangées)
def executer_action(action):
    global flag_tir
    if actions[action] == "left":
        comm.communicate(["execute P1_left(1)", "execute P1_right(0)"])
    elif actions[action] == "right":
        comm.communicate(["execute P1_left(0)", "execute P1_right(1)"])
    elif actions[action] == "stop":
        comm.communicate(["execute P1_left(0)", "execute P1_right(0)"])
    if flag_tir and actions[action] == "tir":
        comm.communicate(["execute P1_Button_1(1)","execute P1_Button_1(1)","execute P1_Button_1(1)"])
        flag_tir = False
    else:
        comm.communicate(["execute P1_Button_1(0)"])
        flag_tir = True

def get_score():
    response = comm.communicate(
        [
            f"read_memory {P1ScorL}",
            f"read_memory {P1ScorM}",
            f"read_memory {plyrShotStatus}",
        ]
    )
    P1ScorL_v, P1ScorM_v, P1ShotStatus = list(map(int, response))
    if debug >= 3:
        print(f"==>SHOT STATUS={P1ShotStatus}")
    return (
        (P1ScorL_v >> 4) * 10
        + (P1ScorM_v & 0x0F) * 100
        + ((P1ScorM_v) >> 4) * 1000
    )
def get_state():
    messages = [
        f"read_memory {alienShotYr}",
        f"read_memory {alienShotXr}",
        f"read_memory {rolShotYr}",
        f"read_memory {rolShotXr}",
        f"read_memory {squShotYr}",
        f"read_memory {squShotXr}",
        f"read_memory {pluShotYr}",
        f"read_memory {pluSHotXr}",
        f"read_memory {saucerDeltaX}",
        f"read_memory {playerXr}",
        f"read_memory {plyrShotStatus}",
        f"read_memory {refAlienYr}",
        f"read_memory {refAlienXr}",
        "read_memory_range 2100(55)",      # Lecture des flags des aliens
        "read_memory_range 2142(176)"      # Lecture complète des infos des boucliers
    ]
    response = comm.communicate(messages)
    
    # Les 13 premiers messages (indices 0 à 12) correspondent aux lectures individuelles
    data = list(map(int, response[:13]))
    
    # Le 14ème message contient les flags des aliens sous forme de chaîne séparée par des virgules
    alien_flags = list(map(int, response[13].split(",")))
    
    # Le 15ème message contient l'ensemble des données des boucliers (176 octets)
    shield_data = list(map(int, response[14].split(",")))
    
    # Extraction des références pour le positionnement des aliens
    refAlienY = data[11]  # Valeur lue à l'adresse refAlienYr
    refAlienX = data[12]  # Valeur lue à l'adresse refAlienXr
    flags = data[:11]     # Les 11 premiers flags individuels
    
    # Calcul des positions des aliens à partir des flags
    alien_positions = []
    for i, flag in enumerate(alien_flags):
        row = i // 11
        col = i % 11
        if flag == 1:
            x = refAlienX + col * 16
            y = refAlienY + row * 16
        else:
            x = 0
            y = 0
        alien_positions.append(x)
        alien_positions.append(y)
    
    # Concaténation des informations pour constituer l'état complet :
    # - flags individuels (premiers 11 octets)
    # - positions des aliens (calculées à partir des flags alien_flags)
    # - données complètes des boucliers (176 octets)
    return np.array(flags + alien_positions + shield_data, dtype=np.float32)
def get_state_FULL_FRAME():
    video_bytes_list = []
    for col in range(224):
        col_start = 0x2400 + col * 32 + 2
        length_to_read = 26
        cmd = f"read_memory_range {hex(col_start)[2:].upper()}({length_to_read})"
        response = comm.communicate([cmd])
        if not response or len(response[0]) == 0:
            video_bytes_list.extend([0]*length_to_read)
        else:
            chunk = list(map(int, response[0].split(',')))
            video_bytes_list.extend(chunk)
    video_bytes = np.array(video_bytes_list, dtype=np.uint8)
    screen = np.zeros((208, 224), dtype=np.uint8)
    idx = 0
    for col in range(224):
        for byte_i in range(26):
            byte_val = video_bytes[idx]
            idx += 1
            for bit in range(8):
                pixel = (byte_val >> bit) & 1
                row = byte_i * 8 + bit
                screen[row, col] = pixel
    factor = 4
    new_rows = screen.shape[0] // factor
    new_cols = screen.shape[1] // factor
    screen_downsampled = (
        screen.reshape(new_rows, factor, new_cols, factor)
              .max(axis=3)
              .max(axis=1)
    )
    flat_pixels = screen_downsampled.flatten()
    packed_size = len(flat_pixels) // 8
    packed_data = np.zeros(packed_size, dtype=np.uint8)
    for i, pix in enumerate(flat_pixels):
        byte_index = i // 8
        bit_index = i % 8
        if pix == 1:
            packed_data[byte_index] |= (1 << bit_index)
    return packed_data
def default_state_extractor(config: TrainingConfig):
    """
    Extrait l'état du jeu en fonction du type de modèle.
    Pour un CNN, utilise get_state_FULL_FRAME, décompresse l'image et la reformate en (channels, height, width).
    Pour un MLP, utilise get_state pour obtenir le vecteur d'état.
    """
    if config.model_type.lower() == "cnn":
        # On attend que config.input_size soit un tuple (channels, height, width)
        channels, height, width = config.input_size
        total_bits = channels * height * width
        # Récupère la frame complète sous forme packée
        packed = get_state_FULL_FRAME()
        # Décompresse en un tableau 1D de bits
        unpacked = unpack_1bit_to_1darray(packed, total_bits=total_bits)
        # Reformate en image (channels, height, width)
        state = unpacked.reshape(channels, height, width)
        return state.astype(np.float32)
    else:
        return get_state()

def save_frame(frame, filename):
    np.save(filename, frame)
    print(f"Frame sauvegardée dans {filename}.")

def load_frame(filename):
    frame = np.load(filename)
    print(f"Frame chargée depuis {filename}.")
    return frame

def unpack_1bit_to_1darray(packed_data, total_bits=2912):
    unpacked = np.zeros(total_bits, dtype=np.uint8)
    for i in range(total_bits):
        byte_index = i // 8
        bit_index = i % 8
        if packed_data[byte_index] & (1 << bit_index):
            unpacked[i] = 1
    return unpacked
def afficher_frame(frame_flat=None):
    from matplotlib import pyplot as plt
    if frame_flat is None:
        frame_flat = get_state_FULL_FRAME() #c'est forcement une image
    unpacked = unpack_1bit_to_1darray(frame_flat, total_bits=52*56)
    frame_img = unpacked.reshape(52, 56)
    plt.figure(figsize=(6, 6))
    plt.imshow(frame_img, cmap="gray", interpolation="nearest")
    plt.title("Frame Space Invaders (sans 16 lignes bas + 32 lignes haut)")
    plt.axis("off")
    plt.show()

def nb_parameters(e, H, n, s):
    parameters = e * n + n
    if H > 1:
        parameters += (H - 1) * (n * n + n)
    parameters += n * s + s
    return parameters

def create_fig(nb_parties, scores_moyens, fenetre, epsilons, filename="Invaders_fig"):
    import matplotlib.pyplot as plt
    print("===> Création du graphe f(épisodes)= scores_moyens")
    fig, ax1 = plt.subplots()
    color = "tab:blue"
    ax1.set_xlabel("Episodes/Parties")
    ax1.set_ylabel("Epsilon", color=color)
    ax1.plot(epsilons, color=color, linestyle="dashed")
    ax1.tick_params(axis="y", labelcolor=color)
    ax1.grid(True, which="both", axis="both", linestyle="-", linewidth=0.5)
    ax2 = ax1.twinx()
    color = "tab:red"
    ax2.set_ylabel(
        f"Score moyen (pour {fenetre} parties)", color=color, rotation=270, labelpad=10
    )
    ax2.plot(scores_moyens, color=color)
    ax2.tick_params(axis="y", labelcolor=color)
    ax2.grid(True, which="both", axis="both", linestyle="-", linewidth=0.5)
    coefficients = np.polyfit(range(len(scores_moyens)), scores_moyens, 1)
    pente = coefficients[0]
    trendline = np.poly1d(coefficients)
    ax2.plot(trendline(range(len(scores_moyens))), color="tab:orange")
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
    plt.title(f"Invaders AI: Score moyen et d'epsilon sur {str(nb_parties)} épisodes")
    plt.savefig(filename+"_"+str(nb_parties)+"_"+datetime.now().strftime("%Y%m%d%H%M")+".png", dpi=300, bbox_inches="tight")
    return pente

def main():
    global f2_pressed, flag_create_fig, debug
    debug=0
    f2_pressed = flag_create_fig= False
    NB_DE_DEMANDES_PAR_FRAME = str(15+2+3+3)
    vitesse_de_jeu = 10
    NB_DE_FRAMES_STEP = 5
    # Configuration réseaux et hyperparamètres
    # Utilisation d'états multiples pour capter la dynamique
    N = 2
    frame_size = 110+11+176  # Taille d'une frame (valeurs extraites du jeu)
    num_episodes = 5000
    reward_alive = 1
    reward_kill = -5000
    reward_mult_step = 0.001

    # Création de la configuration avec TrainingConfig
    config = TrainingConfig(
    state_history_size=N,
    input_size=frame_size * N,  # pour le CNN, on attend un tuple (channels, height, width)
    hidden_size=192,
    output_size=4,
    hidden_layers=2,
    learning_rate=0.005,
    gamma=0.9999,
    epsilon_start=1,
    epsilon_end=0.01,
    epsilon_decay=0.999999,
    epsilon_add=0.0001,
    buffer_capacity=100000,
    batch_size=128,
    double_dqn=True,         # ou False selon vos tests
    model_type="mlp",        # ici on indique que le modèle souhaité "cnn" ou "mlp"
    state_extractor=None     # nous allons l'initialiser juste après
    )
    if config.state_extractor is None:
        # Injection de la fonction d'extraction d'état par défaut
        config.state_extractor = lambda: default_state_extractor(config)
    def on_key_press(event):
        global f2_pressed, flag_create_fig, debug
        def is_terminal_in_focus():
            hwnd = win32gui.GetForegroundWindow()
            title = win32gui.GetWindowText(hwnd)
            return title[-20:] == "- Visual Studio Code"
        if is_terminal_in_focus():
            # Code à exécuter lorsque la touche est pressée et que le terminal est en focus
            if keyboard.is_pressed("shift") and keyboard.is_pressed("ctrl") and keyboard.is_pressed("f2"):
                print("Touche 'F2' détectée. Sortie prématurée. Graphique final créé")
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
            elif config.model_type=="cnn" and keyboard.is_pressed("ctrl") and keyboard.is_pressed("f8"):
                print("<=== Affichage d'une frame")
                afficher_frame()
            elif keyboard.is_pressed("ctrl") and keyboard.is_pressed("f11"):
                toggle_debug_lua()
                time.sleep(0.2)
    keyboard.on_press(on_key_press)
    
    # Instanciation de DQNTrainer
    trainer = DQNTrainer(config)
    trainer.load_model("./invaders.pth")

    # Initialisations
    record = False
    if record:
        recorder = ScreenRecorder()
    fenetre_du_calcul_de_la_moyenne = 100
    collection_of_score = deque(maxlen=fenetre_du_calcul_de_la_moyenne)
    list_of_mean_scores = []
    list_of_epsilons = []
    mean_score = mean_score_old = last_score = 0

    # Crée une frame vide (composée uniquement de zéros)
    empty_frame = np.zeros(frame_size, dtype=np.uint8)

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
        f"[input={config.input_size//config.state_history_size}*{config.state_history_size}={config.input_size}]"
        f"[hidden={config.hidden_size}*{config.hidden_layers}#{nb_parameters(config.input_size, config.hidden_layers, config.hidden_size, config.output_size)}]"
        f"[output={config.output_size}]"
        f"[gamma={config.gamma}][learning={config.learning_rate}]"
        f"[epsilon start, end, decay, add={config.epsilon_start},{config.epsilon_end},{config.epsilon_decay},{config.epsilon_add}]"
        f"[Replay_size={config.buffer_capacity}&_samples={config.batch_size}]"
        f"[model_type={config.model_type}][double_dqn={config.double_dqn}]"
        f"[nb_mess_frame={NB_DE_DEMANDES_PAR_FRAME}]"
        f"[nb_step_frame={NB_DE_FRAMES_STEP}][speed={vitesse_de_jeu}]"
    )


    for episode in range(num_episodes):
        if record:
            recorder.start_recording()
        step = reward = sum_rewards = player_life_end = player_alive = score = 0
        if f2_pressed:
            print("Sortie prématurée de la boucle 'for'.")
            break
        response = comm.communicate([f"write_memory {numCoins}(1)"])
        time.sleep(1)
        while player_alive == 0:
            player_alive = int(comm.communicate([f"read_memory {player1Alive}"])[0])

        # Initialisation de l'historique avec N copies de la frame initiale
        for _ in range(N):
            trainer.state_history.append(empty_frame.copy())
        state = np.concatenate(trainer.state_history)

        while player_alive == 1:
            while step == 0:
                time.sleep(1 / vitesse_de_jeu)
                step += 1
                comm.communicate([f"wait_for {NB_DE_DEMANDES_PAR_FRAME}"])
            ###Bonne Emplacement pour sauvegarder la frame initiale
            #frame_temp=get_state()
            #save_frame(frame_temp, "frame_initiale.npy")
            ############################################
            player_life_end, player_alive = list(
                map(int, comm.communicate([f"read_memory {playerAlive}", f"read_memory {player1Alive}"]))
            )
            action = trainer.select_action(state)
            executer_action(action)
            _last_score = score
            score = get_score()
            reward = round((score- _last_score)*10+step * reward_mult_step + (reward_alive if player_life_end == 255 else reward_kill))
            # Accumulation des récompenses
            sum_rewards += reward  
            current_state = get_state()
            next_state = trainer.update_state_history(current_state)
            done = player_alive == 0
            trainer.replay_buffer.push(state, action, reward, next_state, done)
            state = next_state
            trainer.epsilon = max(min(trainer.epsilon * config.epsilon_decay, config.epsilon_start), config.epsilon_end)
            step += 1

            if len(trainer.replay_buffer) >= config.batch_size:
                trainer.train_step()

            if player_life_end < 255:
                comm.communicate(["wait_for 0"])
                time.sleep(3 / vitesse_de_jeu)
                comm.communicate([f"wait_for {NB_DE_DEMANDES_PAR_FRAME}"])
            if debug >= 1:
                print(
                    f"action={actions[action]:<8}episode={episode:<5d}player_kill={player_life_end:<4d}"
                    f"player_end_game={player_alive:<2d}reward={reward:<5d}"
                    f"all rewards={sum_rewards:<6d}nb_mess_step={comm.number_of_messages:<4d}"
                    f"nb_step={step:<5d}score={score:<5d}"
                )
            comm.number_of_messages = 0

        response = comm.communicate(["wait_for 0"])
        if (episode + 1) % 10 == 0:
            print("Sauvegarde du modèle...")
            trainer.save_model("./dqn_invaders_model_simple.pth")

        comm.communicate(
            [f'draw_text(25,1,"Game number: {episode+1:04d} - mean score={mean_score:04.0f} - ba(c)o 2023")']
        )
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
                episode, list_of_mean_scores, fenetre_du_calcul_de_la_moyenne, list_of_epsilons
            )
            flag_create_fig=False
            time.sleep(0.2)
        collection_of_score.append(score)
        mean_score_old = mean_score
        mean_score = round(sum(collection_of_score) / len(collection_of_score), 2)
        list_of_mean_scores.append(mean_score)
        list_of_epsilons.append(trainer.epsilon)
        if mean_score_old > mean_score:
            trainer.epsilon += config.epsilon_add
        if mean_score == mean_score_old:
            pygame.mixer.Sound("c:\\Windows\\Media\\tada.wav").play()
        _d = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(
            f"N°{episode+1} [{_d}][nb steps={step:4d}][ε={trainer.epsilon:.4f}][rewards={sum_rewards:4d}]"
            f"[score={score:3d}][score moyen={mean_score:.2f}]"
        )

    trainer.save_model("./dqn_invaders_model_simple.pth")
    time.sleep(5)
    if record:
        print(recorder.stop_recording())
        time.sleep(5)
        print(recorder.ws.disconnect())
    print(create_fig(
        episode, list_of_mean_scores, fenetre_du_calcul_de_la_moyenne, list_of_epsilons,
        filename="Final_Invaders_figure"
    ))
    time.sleep(0.2)
    print(process.terminate())

if __name__ == "__main__":
    main()