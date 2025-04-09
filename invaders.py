# file: Invaders_ChatGPT.py : Reinforcement Learning for Space Invaders
from datetime import datetime
import shutil
import time, keyboard, random, pygame, os, psutil, win32gui
import cProfile
import pstats
from colorama import Fore, Style, Back

pygame.mixer.init()  # pour le son au cas où mean_score est stable
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
from MameCommSocket import MameCommunicator
from ScreenRecorder import ScreenRecorder
from matplotlib import pyplot as plt
import matplotlib
import subprocess
import base64
import zlib

# Importer les classes de ai_mame.py
from AI_Mame import TrainingConfig, DQNTrainer, GraphWebServer

import threading

web_server = GraphWebServer(
    graph_dir=".\\", host="0.0.0.0", port=5000, auto_display_latest=True
)
threading.Thread(target=web_server.start, daemon=True).start()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Utilisation de l'appareil : {device}")

# Lancement de MAME
desactiver_video_son = False  # Change à False si tu veux garder la vidéo et le son
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
# Ajouter '-video none' et '-nosound' si désactivé
if desactiver_video_son:
    command.extend(["-video", "none", "-sound", "none"])
# Lancer MAME
process = subprocess.Popen(command, cwd="E:\\Emulateurs\\Mame Officiel")
time.sleep(15)

# Initialisation socket
comm = MameCommunicator("localhost", 12345)

# Constantes
MAX_PLAYER_POS = 255.0
MAX_SAUCER_POS = 255.0
MAX_BOMB_POS_X = 255.0
MAX_BOMB_POS_Y = 255.0
ACTION_DELAY = 0.01
debug = 0
NB_DECOUPAGES = 1

# Adresses du jeu Space Invaders: https://computerarcheology.com/Arcade/SpaceInvaders/RAMUse.html
numCoins = "20EB"  # Nombre de pièces insérées
P1ScorL = "20F8"  # Partie basse du score du joueur 1
P1ScorM = "20F9"  # Partie haute (ou médiane) du score du joueur 1
shotSync = "2080"  # Synchronisation des tirs
numAliens = "2082"  # Nombre d'aliens restants
alienShotYr = "207B"  # Ordonnée du tir d'alien
alienShotXr = "207C"  # Abscisse du tir d'alien
refAlienYr = "2009"  # Ordonnée de l'alien de référence
refAlienXr = "200A"  # Abscisse de l'alien de référence
rolShotYr = "203D"  # Ordonnée du tir de l'alien type "rol"
rolShotXr = "203E"  # Abscisse du tir de l'alien type "rol"
squShotYr = "205D"  # Ordonnée du tir de l'alien type "squ" (ex. Squid)
squShotXr = "205E"  # Abscisse du tir de l'alien type "squ"
pluShotYr = "204D"  # Ordonnée du tir de l'alien type "plu"
pluSHotXr = "204E"  # Abscisse du tir de l'alien type "plu"
playerAlienDead = "2100"  # Flag indiquant la mort du joueur ou d'un alien
saucerDeltaX = "208A"  # Variation horizontale (delta X) de la soucoupe
saucerActive = "2084"  # Etat d'activation de la soucoupe
playerXr = "201B"  # Position horizontale (X) du joueur
plyrShotStatus = "2025"  # Etat du tir du joueur (en vol, prêt, etc.)
obj1CoorYr = "2029"  # Player shot Y coordinate
obj1CoorXr = "202A"  # Player shot X coordinate
p1ShipsRem = "21FF"  # Nombre de vies restantes pour le joueur 1
gameMode = "20EF"  # Mode de jeu actif (par exemple, début, jeu en cours, etc.)
invaded = "206D"  #	Set to 1 when player blows up because rack has reached bottom
playerAlive = "2015"  # Player is alive (FF=alive). Toggles between 0 and 1 for blow-up images.
player1Alive = "20E7"  # 1 if player is alive, 0 if dead (after last man)
playerOK = "2068"  # 	1 means OK, 0 means blowing up

def toggle_debug_lua(debug_lua):
    if debug_lua:
        comm.communicate(["debug on"])
        print("Debug mode activé (debug=1)")
    else:
        comm.communicate(["debug off"])
        print("Debug mode désactivé (debug=0)")
    return not debug_lua     

actions = {
    0: ("left", False),
    1: ("left", True),
    2: ("rght", False),
    3: ("rght", True),
    4: ("stop", False),
    5: ("stop", True),
}
def executer_action(action):
    direction, tirer = actions[action]
    comm.communicate([
        f"execute P1_left({int(direction=='left')})",
        f"execute P1_right({int(direction=='rght')})",
        f"execute P1_Button_1({int(tirer)})"
    ])


def get_score():
    response = comm.communicate(
        [
            f"read_memory {P1ScorL}",
            f"read_memory {P1ScorM}",
        ]
    )
    P1ScorL_v, P1ScorM_v = list(map(int, response))
    return (P1ScorL_v >> 4) * 10 + (P1ScorM_v & 0x0F) * 100 + ((P1ScorM_v) >> 4) * 1000
def get_state(flag_coord_aliens=True, flag_boucliers=False, mult_reward_state=0.0,colonnes_deja_detruites=[False]*11):
    messages = [
        f"read_memory {saucerDeltaX}", #prévoir plutôt p1ShipsRem (à tester avec reward_kill=0)
        f"read_memory {rolShotYr}",
        f"read_memory {rolShotXr}",
        f"read_memory {squShotYr}",
        f"read_memory {squShotXr}",
        f"read_memory {pluShotYr}",
        f"read_memory {pluSHotXr}",
        f"read_memory {numAliens}",
        f"read_memory {playerXr}",
        f"read_memory {obj1CoorXr}",
        f"read_memory {obj1CoorYr}",
        f"read_memory {refAlienYr}",
        f"read_memory {refAlienXr}",
    ]

    if flag_coord_aliens:
        messages.append("read_memory_range 2100(55)")

    response = comm.communicate(messages)
    values = list(map(int, response[:13]))
    flags = values[:11]
    # --- Normalisation individuelle des 11 flags système
    # flags = values[:11]  → on les copie pour éviter de modifier "values"
    flags_normalized = [
        values[0] / 255.0,  # saucerDeltaX MAIS /3.0 si p1ShipsRem
        values[1] / 255.0,  # rolShotYr
        values[2] / 255.0,  # rolShotXr
        values[3] / 255.0,  # squShotYr
        values[4] / 255.0,  # squShotXr
        values[5] / 255.0,  # pluShotYr
        values[6] / 255.0,  # pluShotXr
        values[7] / 55.0,   # numAliens (max 55)
        values[8] / 255.0,  # playerXr
        values[9] / 255.0,  # obj1CoorXr
        values[10] / 255.0, # obj1CoorYr
    ]
    refAlienYr_val = values[11]
    refAlienXr_val = values[12]
    penalty_descente = 0.0

    if flag_coord_aliens:
        alien_flags = list(map(int, response[13].split(",")))

        nb_aliens_par_colonne = []
        rewards_colonne_detruite=0.0
        for col in range(11):
            count = sum(alien_flags[row * 11 + col] for row in range(5))
            nb_aliens_par_colonne.append(count / 5.0)  # Normalisation [0,1]
            if count == 0 and not colonnes_deja_detruites[col]:
                rewards_colonne_detruite += 1000.0
                colonnes_deja_detruites[col] = True

        # Résumé spatial
        positions = [(col, row) for row in range(5) for col in range(11) if alien_flags[row * 11 + col] == 1]
        if positions:
            xs, ys = zip(*positions)
            mean_x = sum(xs) / len(xs)
            mean_y = sum(ys) / len(ys)
            min_x, max_x = min(xs), max(xs)
            min_y, max_y = min(ys), max(ys)
            max_y_pixel = refAlienYr_val + max_y * 16
        else:
            mean_x = mean_y = min_x = max_x = min_y = max_y = max_y_pixel = 0

        # Vecteur d’état complet (31 entrées)
        state_vector = (
            flags_normalized +
            [mean_x / 10.0, mean_y / 4.0, min_x / 10.0, max_x / 10.0, min_y / 4.0, max_y / 4.0] +
            nb_aliens_par_colonne +
            [refAlienXr_val / 255.0, refAlienYr_val / 255.0]
        )
        penalty_descente = ((max_y_pixel - 184)/10+rewards_colonne_detruite ) *  mult_reward_state
    else:
        # Mode "léger" : 13 valeurs seulement (pas d’analyse alien)
        penalty_descente = (refAlienYr_val-120) *  mult_reward_state # 120 hauteur max des aliens puis 112, 104,...,40,32,24,16,8
        state_vector = flags_normalized + [refAlienXr_val / 255.0, refAlienYr_val / 255.0]

    # Normalisation [-1, 1]
    state_vector = np.array(state_vector, dtype=np.float32)
    state_vector = (state_vector - 0.5) * 2.0
    
    if flag_boucliers:
        start_hl = 0x2806
        boucliers = []
        spacing = 0x05A0  # 22 lignes (0x2C0) + 23 lignes (0x2E0) = 1440 octets

        for i in range(4):
            hl = start_hl + i * spacing
            response = comm.communicate([f"read_memory_range {hl:04X}(704)"])  # 22 lignes x 32
            raw = list(map(int, response[0].split(",")))

            bouclier_bytes = []
            for j in range(22):
                base = j * 32
                bouclier_bytes.append(raw[base])       # octet 1 (colonne 6)
                bouclier_bytes.append(raw[base + 1])   # octet 2 (colonne 7)

            ratio = sum(bouclier_bytes) / (255.0 * 44)
            boucliers.append(ratio)

        state_vector = np.concatenate((state_vector, np.array(boucliers, dtype=np.float32)))

    return state_vector, penalty_descente

def get_state_full_screen(factor_div=2):
    """
    Extraction intelligente de la frame Space Invaders.
    Garde uniquement les lignes contenant aliens, boucliers, tirs.
    Supprime joueur et score dès la lecture mémoire (gain perf).
    
    🎯 Résumé spatial horizontal (X, colonnes)
            Zone	        Colonnes   Garde ?
            Bords inutiles	0-15	    ❌ Non
            Zone utile	    16-208	    ✅ Oui
            Bords inutiles	209-224	    ❌ Non

     📏 Résumé spatial sur les 32 octets par colonne (de bas en haut)
            Octets	    Pixels	Contenu
            0-2	        0-23	Zone sous joueur (inutile) ❌
            3-5	        24-47	Joueur + tirs bas ✅ (❌ ???)
            6-9	        48-79	Boucliers ✅ (à garder si tu veux les impacts)
            10-24	    80-199	Aliens + tirs ✅ INDISPENSABLE
            25-27	    200-223	Soucoupe / Score ❌
            28-31	    224-255	Hors écran ❌
    """
    response = comm.communicate(["read_memory_range 2400(7168)"])
    if not response:
        raise ValueError("Aucune réponse reçue de Lua.")

    raw_str = response[0]
    all_bytes = np.array(list(map(int, raw_str.split(","))), dtype=np.uint8)

    if all_bytes.size != 7168:
        raise ValueError(f"Expected 7168 bytes, got {all_bytes.size}")
    
    # All_bytes: toujours reshapeé en 224 colonnes de 32 octets verticaux
    columns = all_bytes.reshape(224, 32)

    # ➕ Horizontal crop : colonnes 16 à 208 (192 colonnes utiles)
    columns = columns[16:208, :]  # ← (224, 32)

    # ➕ Vertical crop : octets 6 à 24 inclus → 176 lignes (22 lignes de 32 octets)
    cropped_columns = columns[:, 3:25]  # ← (192, 22 octets) 

    # ➕ Bits (verticaux) → puis transpose
    image = np.unpackbits(cropped_columns, axis=1, bitorder="little")  # (192, 176 pts)

    # ✅ Downscale si demandé
    if factor_div > 1:
        h, w = image.shape
        h_ = h // factor_div
        w_ = w // factor_div
        image = image[:h_ * factor_div, :w_ * factor_div]
        image = image.reshape(h_, factor_div, w_, factor_div).max(axis=(1, 3))
        
    return image.astype(np.float32)  # dans [0.0, 1.0]
    #return (image.astype(np.float32) - 0.5) * 2.0  # dans [-1, +1]

def save_frame(frame, filename):
    np.save(filename, frame)
    print(f"Frame sauvegardée dans {filename}.")

def load_frame(filename):
    frame = np.load(filename)
    print(f"Frame chargée depuis {filename}.")
    return frame

def afficher_frame(frame=None, factor_div=4):
    if frame is None:
        frame = get_state_full_screen(factor_div)  # Obtient directement une image 2D (192, 176)

    # Appliquer uniquement un flip horizontal
    frame_img = np.rot90(frame, 1) # pour imshow => frame_img doit avoir une forme de (hauteur, largeur).
    #frame_img = np.fliplr(frame_img)

    plt.figure(figsize=(6, 6))
    plt.imshow(frame_img, cmap="gray", interpolation="nearest")
    plt.title("Frame Space Invaders (rot. 90° + flip horizontal)")
    plt.axis("off")
    #plt.show()
    # Générer un nom de fichier avec la date et l'heure au format YYYYMMDD_HHMMSS
    filename = f"frame(shrink factor={factor_div})_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close()

def afficher_get_state():
    state, _ = get_state()
    print("🧠 État (get_state) avec labels (valeurs dénormalisées) :")

    # Dénormalisation (l’inverse de (x - 0.5)*2)
    state_denorm = (state / 2.0 + 0.5)

    labels = [
        ("saucerDeltaX", 255.0),
        ("rolShotYr", 255.0),
        ("rolShotXr", 255.0),
        ("squShotYr", 255.0),
        ("squShotXr", 255.0),
        ("pluShotYr", 255.0),
        ("pluShotXr", 255.0),
        ("numAliens", 55.0),
        ("playerXr", 255.0),
        ("obj1CoorXr", 255.0),
        ("obj1CoorYr", 255.0),

        # Résumé spatial aliens
        ("mean_x", 10.0),
        ("mean_y", 4.0),
        ("min_x", 10.0),
        ("max_x", 10.0),
        ("min_y", 4.0),
        ("max_y", 4.0),
    ]

    # Ajout des colonnes aliens (déjà normalisées [0-1])
    for i in range(11):
        labels.append((f"aliens_col_{i}", 1.0))  # Normalisé directement

    # Positions ref aliens
    labels += [
        ("refAlienXr", 255.0),
        ("refAlienYr", 255.0),
    ]

    for i, (label, maxval) in enumerate(labels):
        if i >= len(state_denorm):
            print(f"{label:<16}: ---")
        else:
            value = state_denorm[i] * maxval
            print(f"{label:<16}: {value:.2f}")

    # === Affichage du nb d'Aliens dans chaque colonne ===
    if len(state) >= 28:
        lrf = state[17:28]  # flags système (11) + résumé aliens (6) = 17
        print("\n📊 lowest_row_flags (colonne la plus basse active) :")
        print("   " + " ".join([f"{(i+1):<3d}" for i in range(11)]))
        print("   " + " ".join([f"{int(i*5):<3d}" for i in lrf]))
    else:
        print("⚠️ Vecteur trop court pour lowest_row_flags.")

def nb_parameters(input_size, num_hidden_layers, hidden_size, output_size, state_history_size=1, model_type="mlp"):
    def conv2d_output_size(size, kernel, stride):
    # Formule modifiée pour arrondir vers le haut en cas de non-divisibilité
        return (size - kernel + stride) // stride
    if isinstance(input_size, tuple):  # Cas CNN
        channels, height, width = input_size
        
        # Calcul de la taille de sortie après chaque couche convolutionnelle
        convw = conv2d_output_size(conv2d_output_size(conv2d_output_size(width, 8, 4), 4, 2), 3, 2)
        convh = conv2d_output_size(conv2d_output_size(conv2d_output_size(height, 8, 4), 4, 2), 3, 2)
        linear_input_size = convw * convh * 128  # 128 filtres en sortie de conv3

        # Calcul du nombre de paramètres pour les convolutions
        conv1_params = 32 * (channels * 8 * 8) + 32            # conv1: (in_channels * kernel_size^2) * out_channels + out_channels
        conv2_params = 64 * (32 * 4 * 4) + 64                  # conv2
        conv3_params = 128 * (64 * 3 * 3) + 128                # conv3
        conv_params = conv1_params + conv2_params + conv3_params

        # Paramètres des couches entièrement connectées (dense)
        fc1_params = linear_input_size * hidden_size + hidden_size
        fc2_params = hidden_size * output_size + output_size
        dense_params = fc1_params + fc2_params

        return conv_params + dense_params

    else:  # Cas MLP
        total_input = input_size * state_history_size
        total_parameters = total_input * hidden_size + hidden_size
        if num_hidden_layers > 1:
            total_parameters += (num_hidden_layers - 1) * (hidden_size * hidden_size + hidden_size)
        total_parameters += hidden_size * output_size + output_size
        return total_parameters

def create_fig(
    trainer,NB_DE_FRAMES_STEP,
    nb_parties,
    scores_moyens,
    fenetre,
    epsilons,
    high_score,flag_aliens,flag_boucliers,
    steps_cumules,
    reward_str="",
    filename="Invaders_fig",
    nb_parties_pente=1000
):
    print("===> Création du graphe f(épisodes)= scores_moyens")
    matplotlib.use("Agg")  # ✅ Désactive l'affichage de la fenêtre
    fig, ax1 = plt.subplots()
    # Courbe epsilon (gauche)
    ax1.set_ylabel("Sigma FC1" if trainer.config.use_noisy else "Epsilon", color="tab:blue", fontsize=7, labelpad=0,fontweight="bold", zorder=10)
    ax1.yaxis.set_label_coords(0, 0.5)
    ax1.plot(epsilons, color="tab:blue", linestyle="dashed")# Lignes horizontales légères pour chaque graduation d'epsilon
    for ytick in ax1.get_yticks():
        ax1.axhline(y=ytick, color='gray', linestyle='--', linewidth=0.5, alpha=0.5, zorder=0)
    ax1.tick_params(axis="y", labelcolor="tab:blue", labelsize=7)
    for label in ax1.get_yticklabels():
        label.set_rotation(90)
    # ➕ Ajouter les labels epsilon à droite à l'intérieur du graphe (sans rotation)
    for ytick in ax1.get_yticks():
        ax1.text(
            1.0, ytick, f"{ytick:.2f}",
            transform=ax1.get_yaxis_transform(which='grid'),
            ha='right',
            va='center',
            fontsize=6,
            color='tab:blue',
            alpha=0.6,
            zorder=5,
            clip_on=True,
        )

        
    # Score moyen (droite)
    ax2 = ax1.twinx()
    ax2.set_ylabel("Score moyen sur 100 épisodes/parties", color="tab:red", rotation=270, labelpad=0,fontsize=8,fontweight="bold", zorder=10)
    ax2.yaxis.set_label_coords(1.02, 0.5)
    ax2.plot(scores_moyens, color="tab:red")
    ax2.tick_params(axis="y", labelcolor="tab:red", labelsize=8)
    # Steps cumulés (2e axe gauche)
    ax3 = ax1.twinx()
    ax3.spines["left"].set_position(("axes", -0.04))  # 🔹 décalage vers la gauche
    ax3.spines["left"].set_visible(True)
    ax3.yaxis.set_label_position("left")
    ax3.yaxis.set_ticks_position("left")
    steps_k = [s / 1000 for s in steps_cumules]
    ax3.plot(steps_k, color="tab:green", linestyle=":", label="Steps cumulés (k)")
    ax3.set_ylabel("Steps cumulés (k)", color="tab:green", fontsize=7, labelpad=0,fontweight="bold", zorder=10)
    ax3.yaxis.set_label_coords(-0.04, 0.5)
    ax3.tick_params(axis="y", labelcolor="tab:green", labelsize=7)
        # Ajoute un 'k' à chaque tick de l'axe steps (ax3)
    def thousands_formatter(x, pos):
        return f"{int(x)}k"
    ax3.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(thousands_formatter))


    # Trendline sur toutes les parties (orange)
    x_all = np.arange(len(scores_moyens))
    y_all = np.array(scores_moyens)
    coeff_all = np.polyfit(x_all, y_all, 1)
    trend_all = np.poly1d(coeff_all)
    pente_all = coeff_all[0]
    ax2.plot(x_all, trend_all(x_all), color="tab:orange")

    # Annotation pente globale
    angle_all = np.arctan(pente_all) * 180 / np.pi
    midpoint_all = len(scores_moyens) // 2
    ax2.text(midpoint_all, trend_all(midpoint_all) + 0.01, f"Pente globale = {pente_all:.2f}",
            color="tab:orange", ha="center", weight="bold",
            bbox=dict(facecolor="white", alpha=0.5, edgecolor="none"),
            rotation=angle_all, rotation_mode="anchor", transform_rotates_text=True)

    # Trendline sur les 1000 dernières parties (violet)
    n = min(nb_parties_pente, len(scores_moyens))
    x_recent = np.arange(-n, 0)
    y_recent = np.array(scores_moyens[-n:])
    coeff_recent = np.polyfit(x_recent, y_recent, 1)
    trend_recent = np.poly1d(coeff_recent)
    pente_recent = coeff_recent[0]
    x_recent_plot = np.arange(len(scores_moyens) - n, len(scores_moyens))
    ax2.plot(x_recent_plot, trend_recent(x_recent), color="tab:purple", linestyle="--")

    # Annotation pente locale
    angle_recent = np.arctan(pente_recent) * 180 / np.pi
    midpoint_recent = len(scores_moyens) - n // 2
    ax2.text(midpoint_recent, trend_recent(-n // 2) + 0.01, f"Pente locale = {pente_recent:.2f}",
            color="tab:purple", ha="center", weight="bold",
            bbox=dict(facecolor="white", alpha=0.5, edgecolor="none"),
            rotation=angle_recent, rotation_mode="anchor", transform_rotates_text=True)
    if trainer.config.use_noisy:
        sigma_vals = trainer.dqn.get_sigma_values()
        sigma_str = ", ".join([f"{name} = {val:.6f}" for name, val in sigma_vals.items()])
        exploration_str = f"[sigma {sigma_str}]"
    # Préparation d'un résumé des paramètres de config
    config_str = (
        (f"Config: Modèle={trainer.config.model_type}, DoubleDQ={trainer.config.double_dqn}, NB_FRAMES_STEP={NB_DE_FRAMES_STEP}\n")+
        (f"N={trainer.config.state_history_size}, input_size={trainer.config.input_size}, output_size={trainer.config.output_size}, ")+
        (f"avec positions des aliens?={flag_aliens}, avec boucliers?={flag_boucliers}\n" if trainer.config.model_type.lower() == "mlp" else "")+
        (f"hidden_layers={trainer.config.hidden_layers}, hidden_size={trainer.config.hidden_size}\n")+
        (f"Buffer Capacity={trainer.config.buffer_capacity}, Size of batch={trainer.config.batch_size}, Prioritized?={trainer.config.prioritized_replay} \n")+
        (f"learning={trainer.config.learning_rate}, gamma={trainer.config.gamma}, target_update_freq={trainer.config.target_update_freq} \n")+
        (f"epsilon=({trainer.config.epsilon_start}->{trainer.config.epsilon_end}), linear?={trainer.config.epsilon_decay==0}, epsilon_add={trainer.config.epsilon_add}" if not trainer.config.use_noisy else f"NoisyNet={exploration_str}\n")+
        (f"{reward_str}")
    )
    # Affichage en petit texte (fontsize=6) dans le coin supérieur gauche
    ax1.text(
        0.01,
        0.99,
        config_str,
        transform=ax1.transAxes,
        fontsize=6,
        va="top",
        ha="left",
        color="gray",
    )

    fig.tight_layout()

    plt.title(
        f"Invaders AI: Score moyen des {fenetre} derniers episodes sur {str(nb_parties)} - HiSc: {high_score}pts"
    )
    plt.savefig(filename + ".png", dpi=300, bbox_inches="tight")
    # plt.savefig(filename+"_"+str(nb_parties)+"_"+datetime.now().strftime("%Y%m%d%H%M")+".png", dpi=300, bbox_inches="tight")
    plt.close()  # 🔥 Évite que plt.show() l’affiche plus tard
    return pente_recent

# --- Classe pickle-safe pour encapsuler l'extraction d'état ---
class StateExtractor:
    def __init__(self, model_type, flag_aliens, flag_boucliers, factor_div_frame,mult_reward_state,colonnes_deja_detruites):
        self.model_type = model_type
        self.flag_aliens = flag_aliens
        self.flag_boucliers = flag_boucliers
        self.factor_div_frame = factor_div_frame
        self.mult_reward_state = mult_reward_state
        self.colonnes_deja_detruites = colonnes_deja_detruites

    def __call__(self):
        if self.model_type.lower() == "cnn":
            frame = get_state_full_screen(factor_div=self.factor_div_frame)
            assert frame.dtype == np.float32, f"frame should be float32 but is {frame.dtype}"
            if debug>1:print(f"🧪 CNN frame stats → min={frame.min():.4f}, max={frame.max():.4f}, mean={frame.mean():.4f}")
            return frame, 0.0
        else:
            return get_state(flag_coord_aliens=self.flag_aliens, flag_boucliers=self.flag_boucliers,mult_reward_state=self.mult_reward_state,colonnes_deja_detruites=self.colonnes_deja_detruites)
def main():
    global debug
    # ✅ Définir les flags pour éviter l'erreur si 'cnn'
    flag_aliens = False
    flag_boucliers = False
    debug = 0
    debug_lua = True # Debug mode activé dès l'appuie de F11
    vitesse_de_jeu = 15
    flag_F8_pressed=flag_F11_pressed=flag_quit = flag_create_fig = False
    # N = state_history_size => N=2 capture la vitesse et N>=3 capture la dynamique (accélération/sens)
    N = 4
    # 🎮 Space Invaders (invaders ROM) → 60 frames/seconde
    NB_DE_FRAMES_STEP = 4 # 4 on a 15 steps par secondes, 5 correspond à 12 steps par secondes, 6=10 steps par secondes
    # Création de la configuration avec TrainingConfig
    model_type = "cnn"  # mlp ou cnn (pour les full_2d_frame)
    if model_type.lower() == "cnn":
        cnn_type = "deepmind"  # precise ou default, precise ou deepmind
        full_frame_2D = (192, 176) # on utilise que 192 sur 224 pts (Largeur) et 176 sur 256 pts (Hauteur)
        factor_div_frame = 1
        input_size = (N,) + tuple(x // factor_div_frame for x in full_frame_2D)
        # 224 appels (224*26=5824 data) pour obtenir une full_frame ou 15 appels (13+55+176 data) pour get_state "mlp"
        # 2 appels pour obtenir une player mort ou partie finie
        # 3 appels pour chaque action
        # 2 appels pour le score
        NB_DE_DEMANDES_PAR_STEP = str(1 + 2 + 3 + 2)
    else:
        factor_div_frame, cnn_type= None, None # valeurs non utilisées car pas utilisées pour MLP
        flag_aliens = False  # comporte 110 coordonnées => pour 6 entrées coordonnées (mean/min/max)
        # Flag_boucliers (alpha)  4 appels à partir de 0x2460 => 4 inputs en + (somme des octets de chaque bouclier)
        flag_boucliers = False
        ############## len(state_vector) = 11 (flags) + flag_aliens(6 (summary) + 11 (lowest row)) + 2 (ref pos) = 30
        input_size = 11 + (6 + 11) * flag_aliens + 4 * flag_boucliers  + 2
        # 11 infos + 2 refAliens + aliens flags + 4 boucliers + 3 actions + 2 score + 2 fin de partie et joueur
        NB_DE_DEMANDES_PAR_STEP = str((11 + 2) + flag_aliens + 4 * flag_boucliers + 3 + 2 + 2)

    # === Rewards ===
    reward_clipping_deepmind=True  ## ATTENTION !! réduit les rewards à 3 valeurs [-1,0,1], si on utilise que le score (CNN/DeepMind) alors [0,1]
    reward_aliens_mult = 1 # Multiplicateur de la récompense si un alien est tue
    reward_kill = -1*reward_aliens_mult # Perte de points si le joueur meurt (pas pour la dernier vie)
    reward_alive = 0.00*NB_DE_FRAMES_STEP # Ajout d'une récompense à chaque step si le joueur est vivant
    reward_mult_step = -0.00000*NB_DE_FRAMES_STEP # Multiplicateur de la récompense par step (pour ne pas être attentiste)
    mult_reward_state = 0.000 # multiplicateur d'un reward spécifique pour get_state (cf code): aliens approchant du joueur,...
    reward_end_of_game = -000.0 # en fin de partie
    colonnes_deja_detruites = [False] * 11 # Utiliser (mlp) pour savoir si on colonne d'aliens vient juste être détruite (True=déjà détruite)
    # === Exploration ===
    use_noisy = True
    epsilon_start = 1
    epsilon_end = 0.1
    target_steps = 1000_000
    epsilon_linear = (epsilon_start-epsilon_end)/target_steps
    epsilon_decay = (epsilon_end / epsilon_start) ** (1 / target_steps) if epsilon_linear==0 else 0
    epsilon_add = 0.00001*NB_DE_FRAMES_STEP if not use_noisy else 0.0 # ajoute à epsilon si pente<0 et mean_old_score>mean_new_score

    config = TrainingConfig(
        state_history_size=N,  # Nombre de frames consécutives à conserver dans l'historique
        # (recommandé pour Invaders : 1 à 5, ici typiquement 2)
        input_size=input_size,  # Taille de l'entrée pour un CNN (N,L,H) après 3 convolutions 3200 entrées si (104,112)         
        # Taille totale de l'entrée pour le MLP
        # (pour Invaders, typiquement entre 100 et 300 ;
        # pour un CNN, fournir un tuple (channels, height, width))
        hidden_layers=1,  # Nombre de couches cachées dans le réseau
        # (recommandé : 1 à 3, ici 2 est courant)
        hidden_size=512,  # Nombre de neurones par couche cachée (512 pour Atari)
        # (recommandé : 64 à 512, 192 est souvent un bon compromis)
        output_size=6,  # Nombre de sorties (actions possibles)
        # (fixe pour Invaders : 6 actions)
        learning_rate=0.00025,  #0.00025 pour DeepMind => Taux d'apprentissage pour l'optimiseur Adam
        # (recommandé : entre 0.0001 et 0.01 pour un bon compromis)
        gamma=0.99,  # Facteur de discount pour valoriser les récompenses futures
        # (recommandé : entre 0.9 et 0.9999, ici très élevé pour privilégier l'avenir)
        use_noisy=use_noisy,
        epsilon_start=epsilon_start,
        epsilon_end=epsilon_end,
        epsilon_linear=epsilon_linear,
        epsilon_decay=epsilon_decay,
        epsilon_add=epsilon_add,
        buffer_capacity=10_000,  # Capacité maximale du replay buffer pour cnn 10 000 vaut 4 Go de GPU RAM !
        # (recommandé : de 10 000 à 1 000 000, ici 100 000 est courant)
        batch_size=32,  # Taille du lot d'échantillons pour l'entraînement
        # (recommandé : entre 32 et 256, ici 128)
        prioritized_replay=True,  # Activation du replay buffer prioritaire 
        target_update_freq=3000,  #soit 10 episodes x nbsteps/episodes (~1000) ou soit batch_size*10
        double_dqn=True,  # Activation du Double DQN (True pour réduire l'overestimation des Q-valeurs)
        model_type=model_type,  # Type de modèle : "cnn" pour réseaux convolutionnels, "mlp" pour perceptron multicouche
        # (pour Invaders, un MLP sur l'état vectoriel est souvent utilisé)
        cnn_type=cnn_type, # test d'autre convolution (autre valeur ou si non défini triple convolution)
        state_extractor=None,  # Fonction d'extraction de l'état (sera initialisée par défaut ultérieurement)
        mode="exploration",  # "exploration" par défaut; passez à "exploitation" en inference
    )

    # --- Dans main(), juste après création de config et des flags ---
    config.state_extractor = StateExtractor(model_type, flag_aliens, flag_boucliers, factor_div_frame,mult_reward_state,colonnes_deja_detruites)

    def on_key_press(event):
        nonlocal flag_F8_pressed,flag_F11_pressed,flag_quit,flag_create_fig
        global debug
        def is_terminal_in_focus():
            hwnd = win32gui.GetForegroundWindow()
            title = win32gui.GetWindowText(hwnd)
            return title[-20:] == "- Visual Studio Code"

        if is_terminal_in_focus():
            # Code à exécuter lorsque la touche est pressée et que le terminal est en focus
            if (
                keyboard.is_pressed("shift")
                and keyboard.is_pressed("ctrl")
                and keyboard.is_pressed("f2")
            ):
                print("\nTouche 'F2' détectée. Sortie prématurée. Graphique final créé\n")
                flag_quit = True
            elif (
                keyboard.is_pressed("shift")
                and keyboard.is_pressed("ctrl")
                and keyboard.is_pressed("f3")
                and debug > 0
            ):
                debug = 0
                print(f"\n====>F3 (RESET DEBUG)====>debug={debug}\n")
            elif (
                keyboard.is_pressed("shift")
                and keyboard.is_pressed("ctrl")
                and keyboard.is_pressed("f4")
            ):
                debug += debug < 3
                print(f"\n====>F4====>debug={debug}\n")
            elif (
                keyboard.is_pressed("shift")
                and keyboard.is_pressed("ctrl")
                and keyboard.is_pressed("f5")
            ):
                trainer.epsilon+=0.01
                print("\n====>F5====> Augmentation d'epsilon de 0.01 à {:.3f}\n".format(trainer.epsilon))
            elif (
                keyboard.is_pressed("shift")
                and keyboard.is_pressed("ctrl")
                and keyboard.is_pressed("f6")
            ):
                trainer.epsilon-=0.01
                print("\nF6 <=== Diminution d'epsilon de 0.01 à {:.3f}\n".format(trainer.epsilon))
            elif (
                keyboard.is_pressed("shift")
                and keyboard.is_pressed("ctrl")
                and keyboard.is_pressed("f7")
                and not flag_create_fig
            ):
                print("\n====>F7====> Demande création d'une figure des scores/episodes\n")
                flag_create_fig = True
            elif (
                keyboard.is_pressed("shift")
                and keyboard.is_pressed("ctrl")
                and keyboard.is_pressed("f8")
            ):
                flag_F8_pressed=True
                print("\n=====>F8\n")
            elif (
                config.mode == "exploration"
                and keyboard.is_pressed("shift")
                and keyboard.is_pressed("ctrl")
                and keyboard.is_pressed("f9")
            ):
                print("\n====>F9====> Exploration =====> Exploitation")
                config.mode = "exploitation"
            elif (
                config.mode == "exploitation"
                and keyboard.is_pressed("shift")
                and keyboard.is_pressed("ctrl")
                and keyboard.is_pressed("f9")
            ):
                print("\n=====>F9====> Exploitation =====>  Exploration")
                config.mode = "exploration"
            elif (
                keyboard.is_pressed("shift")
                and keyboard.is_pressed("ctrl") 
                and keyboard.is_pressed("f11")
                ):
                flag_F11_pressed = True
                print("\n=====>F11\n")
    keyboard.on_press(on_key_press)

    # Instanciation de DQNTrainer
    trainer = DQNTrainer(config)
    trainer.load_model("./invaders.pth")
    trainer.load_buffer("./invaders.buffer")
    if config.use_noisy:
        # 🔧 Reset sigma des couches NoisyLinear à init=0.022 (sigma0=0.5, fc1=512 neurones et sigma_init=sigma0/sqrt(fc1)) d'après  
        # https://arxiv.org/pdf/1706.10295 (page 6)
        init = 0.022  # sigma0 / sqrt(fc1)
        for name, module in trainer.dqn.named_modules():
            if hasattr(module, "sigma_weight") and hasattr(module, "sigma_bias"):
                with torch.no_grad():
                    module.sigma_weight.data.fill_(init)
                    module.sigma_bias.data.fill_(init)
                print(f"🔁 Reset sigma de {name} à {init}")


    # Initialisations video OBS
    record = False
    if record:
        if desactiver_video_son:
            print ("!!! La video de Mame n'est pas activée ('video none') donc pas de recorder video !!!")
            record=False
        else:
            recorder = ScreenRecorder()

    fenetre_du_calcul_de_la_moyenne = 100
    collection_of_score = deque(maxlen=fenetre_du_calcul_de_la_moyenne)
    list_of_mean_scores = []
    list_of_epsilons = []
    list_of_cumulated_steps = []
    mean_score = mean_score_old = last_score = high_score = 0

    # Déterminer la taille correcte de la frame vide en fonction du modèle
    if config.model_type.lower() == "cnn":
        empty_frame = np.zeros(
            (config.input_size[1], config.input_size[2]), dtype=np.float32
        )  # ( H, W)
    else:
        empty_frame = np.zeros(
            (config.input_size,), dtype=np.float32
        )  # Format MLP (1D vecteur)

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
        (f"[*** Mode EXPLOITATION ***]\n" if config.mode == "exploitation" else f"")+
        (f"[input={config.input_size}*{config.state_history_size}={config.input_size*config.state_history_size}]"
        if config.model_type.lower() == "mlp" else f"[input={config.input_size}]")+
        (f"[hidden={config.hidden_size}*{config.hidden_layers}#{nb_parameters(config.input_size, config.hidden_layers, config.hidden_size, config.output_size)}]")+
        (f"[output={config.output_size}][gamma={config.gamma}][learning={config.learning_rate}]\n")+
        (f"[epsilon start, end, linear?, add={config.epsilon_start},{config.epsilon_end},{config.epsilon_decay==0},{config.epsilon_add}]\n"
        if not config.use_noisy else f"[NoisyNet]\n")+
        f"[Replay=capacity,batch_size,prioritized_replay={config.buffer_capacity},{config.batch_size},{config.prioritized_replay}]\n"
        f"[model_type={config.model_type}][double_dqn={config.double_dqn}]"
        f"[nb_mess_frame={NB_DE_DEMANDES_PAR_STEP}]"
        f"[nb_step_frame={NB_DE_FRAMES_STEP}][speed={vitesse_de_jeu}]"
    )
    nb_steps = 0  # Compte le nombre d'actions effectuées
    num_episodes = 10000
    for episode in range(num_episodes):
        if record:
            recorder.start_recording()
        step = reward = sum_rewards = NewGameStarting = PlayerIsOK = NotEndOfGame = score = 0
        if flag_quit:
            print("Sortie prématurée de la boucle 'for'.")
            break

        comm.communicate([f"wait_for 1"])
        comm.communicate([f"write_memory {numCoins}(1)"])
        # Initialiser l'historique avec N copies de frames vides
        trainer.state_history.extend(
            [empty_frame.copy() for _ in range(config.state_history_size)]
        )
        assert (
            len(trainer.state_history) == config.state_history_size
        ), f"Erreur : state_history a une taille incorrecte {len(trainer.state_history)}, attendu {config.state_history_size}."
        # Générer l'état initial en empilant les frames
        if config.model_type.lower() == "cnn":
            state = np.stack(trainer.state_history, axis=0)  # (N, H, W)
        else:
            # 🔍 Vérifier la taille des `states` avant stockage
            state = np.concatenate(
                trainer.state_history, axis=0
            )  # ✅ Concatène `N` états en un vecteur unique
            assert (
                state.shape[0] == config.input_size * config.state_history_size
            ), f"Erreur : state.shape={state.shape}, attendu={config.input_size}. Vérifie input_size dans TrainingConfig."

        while NotEndOfGame == 0:
            NotEndOfGame = int(comm.communicate([f"read_memory {player1Alive}"])[0])
        while PlayerIsOK == 0:
            PlayerIsOK = int(comm.communicate([f"read_memory {playerOK}"])[0])      
        while NewGameStarting != 55:
            NewGameStarting = int(comm.communicate([f"read_memory {numAliens}"])[0])    
        comm.communicate([f"wait_for {NB_DE_DEMANDES_PAR_STEP}"])
        start_steps_time = time.time()  # 🔥 Début de la mesure d'un step
        nb_vies=3
        while NotEndOfGame == 1:
            action = trainer.select_action(state)
            executer_action(action)
            _last_score = score
            score = get_score()
            current_state0, reward_state = config.state_extractor()
            next_state = trainer.update_state_history(current_state0)
            PlayerIsOK, NotEndOfGame = list(map(int,comm.communicate([f"read_memory {playerOK}", f"read_memory {player1Alive}"]),))   
            if PlayerIsOK == 1:
                reward = reward_alive
            elif nb_vies > 1: # Possibilité de différencier derniere vie...
                reward = reward_kill 
            else:
                reward = reward_kill+reward_end_of_game
            reward += (
                ((score - _last_score) * reward_aliens_mult) + 
                (step * reward_mult_step) +
                (reward_state)
                )
            # 🎯 Clipping DeepMind
            if reward_clipping_deepmind: 
                reward = 1 if reward > 0 else -1 if reward < 0 else 0  # → -1, 0, +1 uniquement (DeepMind)
            # Accumulation des récompenses
            sum_rewards += reward
            done = NotEndOfGame == 0 or PlayerIsOK == 0
            trainer.replay_buffer.push(state, action, reward, next_state, done)
            state = next_state
            trainer.update_epsilon()
            step += 1
            if len(trainer.replay_buffer) >= config.batch_size:
                trainer.train_step()
            if debug >= 1:
                elapsed_steps_time = (time.time() - start_steps_time)
                direction, tirer = actions[action]
                action_str = f"{direction}{'+tir' if tirer else ''}"
                print(
                    f"⏱️={(elapsed_steps_time / step)*1000.0:2.2f}ms "
                    f"N°={episode:<5d}action={action_str:<8} PlayerIsOK={PlayerIsOK:<2d}NotEndOfGame={NotEndOfGame:<2d}"
                    f"reward={reward:<6.3f}score={score:<5d} sum_rewards={sum_rewards:<6.0f}"
                    f"nb_mess_step={comm.number_of_messages:<4d}nb_step={step:<5d}"
                )
                if debug >= 2:
                    if config.model_type.lower() == "cnn":
                        print(f"🧠 state.shape = {state.shape} | min={state.min():.2f}, max={state.max():.2f}, mean={state.mean():.2f}")
                    else:
                        print(f"state={state}")
            comm.number_of_messages = 0   
            if flag_F11_pressed:
                debug_lua=toggle_debug_lua(debug_lua)
                flag_F11_pressed=False
                print(f" ==============  toggle_debug_lua({not debug_lua}) <====> F11")
            if flag_F8_pressed:
                if config.model_type == "cnn":
                    print("====>F8====> Affichage d'une frame")
                    afficher_frame(current_state0)  
                elif config.model_type == "mlp":
                    print("====>F8====> Affichage de l'état MLP get_state()")
                    afficher_get_state()
                flag_F8_pressed=False
            if PlayerIsOK == 0:
                comm.communicate(["wait_for 2"])        
                while PlayerIsOK == 0 and NotEndOfGame == 1:
                    PlayerIsOK, NotEndOfGame = list(map(int,comm.communicate([f"read_memory {playerOK}", f"read_memory {player1Alive}"]),))
                if PlayerIsOK == 1: 
                    nb_vies-=1
                    comm.communicate([f"wait_for {NB_DE_DEMANDES_PAR_STEP}"])

        response = comm.communicate(["wait_for 0"])
        reward_str = f"Rewards: alive={reward_alive:.2f}, step×={reward_mult_step:.6f}, Xpoints/alien={reward_aliens_mult}, kill={reward_kill:3.0f},\n"
        reward_str+= f"         reward_state={mult_reward_state}, rewardend_of_game={reward_end_of_game}, reward_clipping_deepmind={reward_clipping_deepmind}"
        if (episode + 1) % 10 == 0:
            print("Sauvegarde du modèle...")
            trainer.save_model("./invaders.pth")
            _pente = create_fig(
                trainer,NB_DE_FRAMES_STEP,
                episode,
                list_of_mean_scores,
                fenetre_du_calcul_de_la_moyenne,
                list_of_epsilons,
                high_score,flag_aliens,flag_boucliers,
                list_of_cumulated_steps,
                reward_str=reward_str,  # 👈 ajout ici
                filename="Invaders_fig",
            )
            if _pente < 0 and episode > 100 and not use_noisy: trainer.epsilon+=config.epsilon_add if trainer.epsilon < trainer.config.epsilon_start else 0.0
        comm.communicate([f'draw_text(25,1,"Game number: {episode+1:04d} - mean score={mean_score:04.0f} - ba(c)o 2023")'])
        if record:
            recorder.stop_recording()
            time.sleep(1)
            if score > last_score:
                if last_score != 0: os.remove(f"best_game_{last_score}.avi")
                time.sleep(1)
                shutil.copy("output-obs.mp4", f"best_game_{score}.avi")
                time.sleep(1)
                last_score = score
        if flag_create_fig:
            _pente = create_fig(
                trainer,NB_DE_FRAMES_STEP,
                episode,
                list_of_mean_scores,
                fenetre_du_calcul_de_la_moyenne,
                list_of_epsilons,
                high_score,flag_aliens,flag_boucliers,
                list_of_cumulated_steps,
                reward_str=reward_str,  # 👈 ajout ici
                filename="Invaders_fig_ask",
            )
            flag_create_fig = False
            time.sleep(0.2)
        collection_of_score.append(score)
        if use_noisy:
            sigmas = trainer.dqn.get_sigma_values()
            list_of_epsilons.append(sigmas['fc1'])
        else:
            list_of_epsilons.append(trainer.epsilon)
        list_of_cumulated_steps.append(nb_steps)
        mean_score_old = mean_score
        mean_score = round(sum(collection_of_score) / len(collection_of_score), 2)
        list_of_mean_scores.append(mean_score)
        if score > high_score:
            high_score = (
                score  # Mise à jour du high score si un meilleur score est atteint
            )
        if mean_score_old > mean_score:
            trainer.epsilon += config.epsilon_add
        if mean_score == mean_score_old:
            pygame.mixer.Sound("c:\\Windows\\Media\\tada.wav").play()
        _d = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        nb_steps+=step
        if config.use_noisy:
            sigma_vals = trainer.dqn.get_sigma_values()
            sigma_str = ", ".join([f"{name} = {val:.6f}" for name, val in sigma_vals.items()])
            exploration_str = f"[sigma {sigma_str}]"
        else:
            exploration_str = f"[ε={trainer.epsilon:.4f}]" if config.mode != "exploitation" else "[*** Mode EXPLOITATION ***]"

        print(
            f"N°{episode+1} [{_d}][steps,all={step:4d},{str(nb_steps//1000)+'k':>5}]"
            + exploration_str
            + f"[rewards={sum_rewards:5.0f}]"
            + f"[score={score:3d}][score moyen={mean_score:3.0f}]"
        )
    final_titre = (
        (f"[EXPLOITATION]_" if config.mode == "exploitation" else f"")+
        f"{config.model_type}_double={config.double_dqn}_N={config.state_history_size}_i={config.input_size}"
        f"_hl={config.hidden_layers},{config.hidden_size}"
        f"_batch={config.buffer_capacity},{config.batch_size}"
        f"_l={config.learning_rate}_g={config.gamma}"
        f"{f'_e={config.epsilon_start},{config.epsilon_end},{config.epsilon_decay==0}' if not config.use_noisy else '[NoisyNet]'}"
        f"_nb={episode}_ms={mean_score:.0f}_hs={high_score}"
    )
    trainer.save_model(f"./invaders_{final_titre}.pth")
    trainer.save_buffer("./invaders.buffer")
    time.sleep(2)
    print(
        create_fig(
            trainer,NB_DE_FRAMES_STEP,
            episode,
            list_of_mean_scores,
            fenetre_du_calcul_de_la_moyenne,
            list_of_epsilons,
            high_score,flag_aliens,flag_boucliers,
            list_of_cumulated_steps,
            reward_str=reward_str,  # 👈 ajout ici
            filename=f"Final_Invaders_{final_titre}",
        )
    )
    time.sleep(2)
    if record:
        print(recorder.stop_recording())
        time.sleep(2)
        print(recorder.ws.disconnect())
        time.sleep(2)

    print(process.terminate())


if __name__ == "__main__":
    if False:
        profiler = cProfile.Profile()
        profiler.enable()
        main()
        profiler.disable()
        stats = pstats.Stats(profiler).sort_stats("cumtime")
        stats.print_stats()
    else:
        main()
