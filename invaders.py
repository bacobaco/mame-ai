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
from dreamerv2 import DreamerTrainer  # ajouté pour modèle Dreamer
import threading

web_server = GraphWebServer(
    graph_dir=".\\", host="0.0.0.0", port=5000, auto_display_latest=True
)
threading.Thread(target=web_server.start, daemon=True).start()


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
pluShotXr = "204E"  # Abscisse du tir de l'alien type "plu"
playerAlienDead = "2100"  # Flag indiquant la mort du joueur ou d'un alien
saucerXr = "208A"  # Coordonnées X de la soucoupe (de $29=41 à $E0=224)
saucerActive = "2084"  # Etat d'activation de la soucoupe
playerXr = "201B"  # Position horizontale (X) du joueur
plyrShotStatus = "2025" # 0 if available, 1 if just initiated, 2 moving normally, 3 hit something besides alien, 
                        # 5 if alien explosion is in progress, 4 if alien has exploded (remove from active duty)
obj1CoorYr = "2029"  # Player shot Y coordinate
obj1CoorXr = "202A"  # Player shot X coordinate
p1ShipsRem = "21FF"  # Nombre de vies restantes pour le joueur 1
gameMode = "20EF"  # Mode de jeu actif (par exemple, début, jeu en cours, etc.)
invaded = "206D"  #	Set to 1 when player blows up because rack has reached bottom
playerAlive = "2015"  # Player is alive (FF=alive). Toggles between 0 and 1 for blow-up images.
player1Alive = "20E7"  # 1 if player is alive, 0 if dead (after last man)
playerOK = "2068"  # 	1 means OK, 0 means blowing up

def toggle_debug_lua(debug_lua_current_state):
    if debug_lua_current_state: # Si True, on veut désactiver
        comm.communicate(["debug off"])
        print("Debug LUA mode désactivé")
        return False
    else: # Si False, on veut activer
        comm.communicate(["debug on"])
        print("Debug LUA mode activé")
        return True   
actions = {
    0: ("left", False), 1: ("left", True),
    2: ("rght", False), 3: ("rght", True),
    4: ("stop", False), 5: ("stop", True),
}
def executer_action(action):
    direction, tirer = actions[action]
    comm.communicate([
        f"execute P1_left({int(direction=='left')})",
        f"execute P1_right({int(direction=='rght')})",
        f"execute P1_Button_1({int(tirer)})"
    ])

def get_score():
    response = comm.communicate([f"read_memory {P1ScorL}", f"read_memory {P1ScorM}"])
    if not response or len(response) < 2: return 0 # Gestion d'erreur
    P1ScorL_v, P1ScorM_v = list(map(int, response))
    return (P1ScorL_v >> 4) * 10 + (P1ScorM_v & 0x0F) * 100 + ((P1ScorM_v) >> 4) * 1000

def get_state(flag_coord_aliens=True, flag_boucliers=False, mult_reward_state=0.0, colonnes_deja_detruites_input=None): # Renommé pour éviter conflit
    if colonnes_deja_detruites_input is None:
        colonnes_deja_detruites_input = [False] * 11 # Initialisation par défaut si non fourni

    messages = [
        f"read_memory {saucerXr}", f"read_memory {rolShotYr}", f"read_memory {rolShotXr}",
        f"read_memory {squShotYr}", f"read_memory {squShotXr}", f"read_memory {pluShotYr}",
        f"read_memory {pluShotXr}", f"read_memory {numAliens}", f"read_memory {playerXr}",
        f"read_memory {obj1CoorXr}", f"read_memory {obj1CoorYr}", f"read_memory {refAlienYr}",
        f"read_memory {refAlienXr}",
    ]
    if flag_coord_aliens: messages.append("read_memory_range 2100(55)") # 55 octets à partir de 2100

    response = comm.communicate(messages)
    if not response or len(response) < 13: return np.zeros(11), 0.0 # Gestion d'erreur, retourne un état par défaut

    values = list(map(int, response[:13]))
    flags_normalized = [
        (values[0] - 41) / (224 - 41) if (224 - 41) != 0 else 0.0, # saucerXr
        values[1] / 223.0, values[2] / 223.0, values[3] / 223.0, values[4] / 223.0,
        values[5] / 223.0, values[6] / 223.0, values[7] / 55.0 if values[7] != 0 else 0.0, # numAliens
        values[8] / 223.0, values[9] / 223.0, values[10] / 223.0,
    ]
    refAlienYr_val = values[11]
    refAlienXr_val = values[12]
    penalty_descente = 0.0
    rewards_colonne_detruite_total = 0.0 # Renommé pour éviter conflit avec variable locale

    if flag_coord_aliens:
        if len(response) < 14: return np.array(flags_normalized), 0.0 # Gestion d'erreur
        alien_flags_str = response[13]
        try:
            alien_flags = list(map(int, alien_flags_str.split(",")))
            if len(alien_flags) != 55: # 5 lignes * 11 colonnes
                print(f"Warning: Expected 55 alien flags, got {len(alien_flags)}. Using zeros.")
                alien_flags = [0] * 55
        except ValueError:
            print(f"Warning: Could not parse alien flags: {alien_flags_str}. Using zeros.")
            alien_flags = [0] * 55


        nb_aliens_par_colonne = []
        for col_idx in range(11): # Renommé col en col_idx
            count = sum(alien_flags[row * 11 + col_idx] for row in range(5))
            nb_aliens_par_colonne.append(count / 5.0)
            if count == 0 and not colonnes_deja_detruites_input[col_idx]:
                rewards_colonne_detruite_total += 1000.0
                colonnes_deja_detruites_input[col_idx] = True

        positions = [(col_idx, row) for row in range(5) for col_idx in range(11) if alien_flags[row * 11 + col_idx] == 1]
        if positions:
            xs, ys = zip(*positions)
            mean_x, mean_y = sum(xs) / len(xs), sum(ys) / len(ys)
            min_x, max_x, min_y, max_y = min(xs), max(xs), min(ys), max(ys)
            max_y_pixel = refAlienYr_val + max_y * 16
        else:
            mean_x = mean_y = min_x = max_x = min_y = max_y = max_y_pixel = 0

        state_vector_list = ( # Renommé en state_vector_list pour éviter confusion
            flags_normalized +
            [mean_x / 10.0, mean_y / 4.0, min_x / 10.0, max_x / 10.0, min_y / 4.0, max_y / 4.0] +
            nb_aliens_par_colonne +
            [refAlienXr_val / 255.0, refAlienYr_val / 255.0]
        )
        penalty_descente = ((max_y_pixel - 184) / 10 + rewards_colonne_detruite_total) * mult_reward_state
    else:
        penalty_descente = (refAlienYr_val - 120) * mult_reward_state
        state_vector_list = flags_normalized # + [refAlienXr_val / 255.0, refAlienYr_val / 255.0] # Commenté car non utilisé si flag_aliens=False

    state_vector_np = np.array(state_vector_list, dtype=np.float32) # Renommé
    state_vector_np = (state_vector_np - 0.5) * 2.0

    if flag_boucliers:
        # ... (logique des boucliers, s'assurer qu'elle retourne un np.array)
        # Pour l'instant, je vais simuler pour éviter des erreurs si non implémenté complètement
        boucliers_np = np.zeros(4, dtype=np.float32) # Exemple
        state_vector_np = np.concatenate((state_vector_np, boucliers_np))


    return state_vector_np, penalty_descente
def old_get_state(flag_coord_aliens=True, flag_boucliers=False, mult_reward_state=0.0,colonnes_deja_detruites=[False]*11):
    messages = [
        f"read_memory {saucerXr}", #prévoir plutôt p1ShipsRem (à tester avec reward_kill=0)
        f"read_memory {rolShotYr}",
        f"read_memory {rolShotXr}",
        f"read_memory {squShotYr}",
        f"read_memory {squShotXr}",
        f"read_memory {pluShotYr}",
        f"read_memory {pluShotXr}",
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
        (values[0] - 41) / (224 - 41),  # saucerXr, X absolu de la soucoupe [41,224]
        values[1] / 223.0,              # rolShotYr, Y tir alien "rol"
        values[2] / 223.0,              # rolShotXr, X tir alien "rol"
        values[3] / 223.0,              # squShotYr, Y tir alien "squ"
        values[4] / 223.0,              # squShotXr, X tir alien "squ"
        values[5] / 223.0,              # pluShotYr, Y tir alien "plu"
        values[6] / 223.0,              # pluShotXr, X tir alien "plu"
        values[7] / 55.0,               # numAliens
        values[8] / 223.0,              # playerXr, X joueur
        values[9] / 223.0,              # obj1CoorXr, X tir joueur
        values[10] / 223.0,             # obj1CoorYr, Y tir joueur
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
        state_vector = flags_normalized #+ [refAlienXr_val / 255.0, refAlienYr_val / 255.0]

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

def afficher_get_state():
    state, _ = get_state()
    print("🧠 État (get_state) avec labels (valeurs dénormalisées) :")

    # Dénormalisation (l’inverse de (x - 0.5)*2)
    state_denorm = (state / 2.0 + 0.5)

    labels = [
        ("saucerXr", 224.0),
        ("rolShotYr", 223.0),
        ("rolShotXr", 223.0),
        ("squShotYr", 223.0),
        ("squShotXr", 223.0),
        ("pluShotYr", 223.0),
        ("pluShotXr", 223.0),
        ("numAliens", 55.0),
        ("playerXr", 223.0),
        ("obj1CoorXr", 223.0),
        ("obj1CoorYr", 223.0),

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
        ("refAlienXr", 223.0),
        ("refAlienYr", 223.0),
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

def afficher_frame(frame=None, factor_div=1):
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
    fenetre_lissage,
    epsilons_or_sigmas,
    high_score,flag_aliens,flag_boucliers,
    steps_cumules,
    reward_str="",
    filename="Invaders_fig",
    nb_parties_pente=1000
):
    matplotlib.use("Agg")  # ✅ Désactive l'affichage de la fenêtre
    fig, ax1 = plt.subplots(figsize=(12, 8), constrained_layout=True) # Par exemple 12x8 pouces ou même plus grand si nécessaire : figsize=(15, 10)
    # --- Axe Sigma/Epsilon (ax1, gauche) ---
    ax1.set_xlabel("Nombre d'épisodes", fontsize=10)
    ax1.set_ylabel("Sigma FC-In (bleu)/FC-Out (orange)" if trainer.config.use_noisy else "Epsilon", color="tab:blue", fontsize=7, labelpad=0,fontweight="bold", zorder=10)
    ax1.yaxis.set_label_coords(0, 0.5)
    if trainer.config.use_noisy:
        ax1.plot(epsilons_or_sigmas[0], color="tab:blue", linestyle="dashed",lw=0.8)# Lignes horizontales légères FC1
        ax1.plot(epsilons_or_sigmas[1], color="tab:orange", linestyle="dashdot",lw=0.8)# Lignes horizontales légères FC2 
    else:
        ax1.plot(epsilons_or_sigmas[0], color="tab:blue", linestyle="dashed")  

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
        (f"Config: Modèle={trainer.config.model_type}, DoubleDQ={trainer.config.double_dqn}, Dueling={trainer.config.dueling}, Nstep={trainer.config.nstep}/{trainer.config.nstep_n if trainer.config.nstep else chr(8)}, NB_FRAMES_STEP={NB_DE_FRAMES_STEP}\n")+
        (f"N={trainer.config.state_history_size}, input_size={trainer.config.input_size}, output_size={trainer.config.output_size}, ")+
        (f"avec positions des aliens?={flag_aliens}, avec boucliers?={flag_boucliers}\n" if trainer.config.model_type.lower() == "mlp" else "")+
        (f"hidden_layers={trainer.config.hidden_layers}, hidden_size={trainer.config.hidden_size}\n")+
        (f"Buffer Capacity={trainer.config.buffer_capacity}, Size of batch={trainer.config.batch_size}, Prioritized?={trainer.config.prioritized_replay} \n")+
        (f"learning={trainer.config.learning_rate}, gamma={trainer.config.gamma}, target_update_freq={trainer.config.target_update_freq} \n")+
        (f"epsilon=({trainer.config.epsilon_start}->{trainer.config.epsilon_end}), linear?={trainer.config.epsilon_decay==0}, epsilon_add={trainer.config.epsilon_add}\n" if not trainer.config.use_noisy else f"NoisyNet={exploration_str}\n")+
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

    # ➕ Affichage des dernières valeurs de chaque courbe sous forme de "bulle"
    last_x = len(scores_moyens) - 1

    # 1. Dernier score moyen (rouge, ax2)
    last_score = scores_moyens[-1]
    ax2.annotate(
        f"{last_score:.1f}",
        xy=(last_x, last_score),
        xytext=(last_x, last_score + 10),
        fontsize=8,
        color="tab:red",
        ha="right",
        arrowprops=dict(arrowstyle="->", color="tab:red", lw=1),
        bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="tab:red", lw=0.5, alpha=0.7),
        zorder=15
    )
    # 🔺 Bulle sur le score moyen maximal (sommet de la courbe rouge)
    max_idx = np.argmax(scores_moyens)
    max_score = scores_moyens[max_idx]
    ax2.annotate(
        f"Max {max_score:.1f}",
        xy=(max_idx, max_score),
        xytext=(max_idx, max_score + 20),
        fontsize=8,
        color="tab:red",
        ha="center",
        arrowprops=dict(arrowstyle="->", color="tab:red", lw=1),
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="tab:red", lw=1, alpha=0.8),
        zorder=20
    )
    # 2. Dernier epsilon ou sigma (bleu ou orange, ax1)
    if trainer.config.use_noisy:
        if epsilons_or_sigmas[0]:
            sigma_fc1_last = epsilons_or_sigmas[0][-1]
            ax1.annotate(
                f"{sigma_fc1_last:.3f}",
                xy=(last_x, sigma_fc1_last),
                xytext=(last_x - 50, sigma_fc1_last + 0.02),
                fontsize=7,
                color="tab:blue",
                ha="center",
                arrowprops=dict(arrowstyle="->", color="tab:blue", lw=0.8),
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="tab:blue", lw=0.5, alpha=0.6),
                zorder=15
            )
        if epsilons_or_sigmas[1]:
            sigma_fc2_last = epsilons_or_sigmas[1][-1]
            ax1.annotate(
                f"{sigma_fc2_last:.3f}",
                xy=(last_x, sigma_fc2_last),
                xytext=(last_x - 50, sigma_fc2_last - 0.05),
                fontsize=7,
                color="tab:orange",
                ha="center",
                arrowprops=dict(arrowstyle="->", color="tab:orange", lw=0.8),
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="tab:orange", lw=0.5, alpha=0.6),
                zorder=15
            )
    else:
        eps_last = epsilons_or_sigmas[0][-1]
        ax1.annotate(
            f"{eps_last:.3f}",
            xy=(last_x, eps_last),
            xytext=(last_x - 50, eps_last + 0.02),
            fontsize=7,
            color="tab:blue",
            ha="center",
            arrowprops=dict(arrowstyle="->", color="tab:blue", lw=0.8),
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="tab:blue", lw=0.5, alpha=0.6),
            zorder=15
        )

    # 3. Derniers steps cumulés (vert, ax3)
    last_step_k = steps_cumules[-1] / 1000
    ax3.annotate(
        f"{last_step_k:.1f}k",
        xy=(last_x, last_step_k),
        xytext=(last_x - 80, last_step_k + 10),
        fontsize=7,
        color="tab:green",
        ha="center",
        arrowprops=dict(arrowstyle="->", color="tab:green", lw=0.8),
        bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="tab:green", lw=0.5, alpha=0.6),
        zorder=15
    )
    #fig.tight_layout()

    plt.title(
        f"Invaders AI: Score moyen des {fenetre_lissage} derniers episodes sur {str(nb_parties)} - HiSc: {high_score}pts"
    )
    plt.savefig(filename + ".png", dpi=300)
    # plt.savefig(filename+"_"+str(nb_parties)+"_"+datetime.now().strftime("%Y%m%d%H%M")+".png", dpi=300, bbox_inches="tight")
    plt.close()  # 🔥 Évite que plt.show() l’affiche plus tard
    return pente_recent

class StateExtractor:
    def __init__(self, model_type, flag_aliens, flag_boucliers, factor_div_frame, mult_reward_state, colonnes_deja_detruites_ref): # Renommé
        self.model_type = model_type
        self.flag_aliens = flag_aliens
        self.flag_boucliers = flag_boucliers
        self.factor_div_frame = factor_div_frame
        self.mult_reward_state = mult_reward_state
        # Conserver une référence à la liste mutable pour que les modifications soient visibles
        self.colonnes_deja_detruites_ref = colonnes_deja_detruites_ref

    def __call__(self):
        if self.model_type in ("cnn", "dreamer"):
            frame = get_state_full_screen(factor_div=self.factor_div_frame)
            # assert frame.dtype == np.float32, f"frame should be float32 but is {frame.dtype}" # Déjà float32
            # if debug>1:print(f"🧪 CNN frame stats → min={frame.min():.4f}, max={frame.max():.4f}, mean={frame.mean():.4f}")
            return frame, 0.0 # Pas de reward_state pour CNN/Dreamer ici
        else: # MLP
            return get_state(
                flag_coord_aliens=self.flag_aliens,
                flag_boucliers=self.flag_boucliers,
                mult_reward_state=self.mult_reward_state,
                colonnes_deja_detruites_input=self.colonnes_deja_detruites_ref # Passer la référence
            )
def main():
    global debug
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
    model_type = "mlp"  # mlp ou cnn (pour les full_2d_frame) ou "dreamer" solution DreamerV2 lié à dreamerv2.py
    if model_type.lower() == "cnn":
        cnn_type = "deepmind"  # precise ou default, precise ou deepmind
        full_frame_2D = (192, 176) # on utilise que 192 sur 224 pts (Largeur) et 176 sur 256 pts (Hauteur)
        factor_div_frame = 2
        input_size = (N,) + tuple(x // factor_div_frame for x in full_frame_2D)
        # 224 appels (224*26=5824 data) pour obtenir une full_frame ou 15 appels (13+55+176 data) pour get_state "mlp"
        # 2 appels pour obtenir une player mort ou partie finie
        # 3 appels pour chaque action
        # 2 appels pour le score
        NB_DE_DEMANDES_PAR_STEP = str(1 + 2 + 3 + 2)
        TRAIN_EVERY_N_GLOBAL_STEPS = 2
    elif model_type.lower() == "mlp":
        factor_div_frame, cnn_type= None, None # valeurs non utilisées car pas utilisées pour MLP
        flag_aliens = False # comporte 110 coordonnées => pour 6 entrées coordonnées (mean/min/max)
        # Flag_boucliers (alpha)  4 appels à partir de 0x2460 => 4 inputs en + (somme des octets de chaque bouclier)
        flag_boucliers = False
        ############## len(state_vector) = 11 (flags) + flag_aliens(6 (summary) + 11 (lowest row)) + 2 (ref pos)* = 30
        input_size = 11 + (6 + 11) * flag_aliens + 4 * flag_boucliers  #+ 2
        # 11 infos + 2 refAliens + aliens flags + 4 boucliers + 3 actions + 2 score + 2 fin de partie et joueur
        NB_DE_DEMANDES_PAR_STEP = str((11 + 2) + flag_aliens + 4 * flag_boucliers + 3 + 2 + 2)
        TRAIN_EVERY_N_GLOBAL_STEPS = 1
    elif model_type.lower() == "dreamer":
        factor_div_frame = 2
        cnn_type = None
        full_frame_2D = (192//factor_div_frame, 176//factor_div_frame) # on utilise que 192 sur 224 pts (Largeur) et 176 sur 256 pts (Hauteur) => (96, 176)
        input_size = (N,) + full_frame_2D
        NB_DE_DEMANDES_PAR_STEP = str(1 + 2 + 3 + 2)
        TRAIN_EVERY_N_GLOBAL_STEPS = 10
    else:
        raise ValueError(f"Modèle {model_type} non pris en charge.")
    # === Rewards ===
    reward_clipping_deepmind=True  ## ATTENTION !! réduit les rewards à 3 valeurs [-1,0,1], si on utilise que le score (CNN/DeepMind) alors [0,1]
    reward_aliens_mult = 1 # Multiplicateur de la récompense si un alien est tue
    reward_kill = -00*reward_aliens_mult # Perte de points si le joueur meurt (pas pour la dernier vie)
    reward_alive = 0.000*NB_DE_FRAMES_STEP # Ajout d'une récompense à chaque step si le joueur est vivant
    reward_mult_step = -0.0000*NB_DE_FRAMES_STEP # Multiplicateur de la récompense par step (pour ne pas être attentiste)
    mult_reward_state = 0.00 # multiplicateur d'un reward spécifique pour get_state (cf code): aliens approchant du joueur,...
    reward_end_of_game = -0 # en fin de partie
    colonnes_deja_detruites = [False] * 11 # Utiliser (mlp) pour savoir si on colonne d'aliens vient juste être détruite (True=déjà détruite)
    # === Exploration ===
    use_noisy = True if model_type.lower() != "dreamer" else False 
    rainbow_eval = 250_000 # Nombre de steps avant de commencer les évaluations (250 000 pour Rainbow)
    rainbow_eval_pourcent = 3/TRAIN_EVERY_N_GLOBAL_STEPS # Pourcentage de rainbow_eval  pour évaluer le modèle
    epsilon_start = 1 if not use_noisy else 0.001
    epsilon_end = 0.0
    target_steps_for_epsilon_end = 1000_000
    epsilon_linear =  (epsilon_start-epsilon_end)/target_steps_for_epsilon_end
    epsilon_decay = (epsilon_end / epsilon_start) ** (1 / target_steps_for_epsilon_end) if epsilon_linear==0 else 0
    epsilon_add = ((epsilon_start-epsilon_end)/target_steps_for_epsilon_end)*NB_DE_FRAMES_STEP*100 if not use_noisy else 0.0 # ajoute à epsilon si pente<0 et mean_old_score>mean_new_score

    config = TrainingConfig(
        state_history_size=N,  # Nombre de frames consécutives à conserver dans l'historique
        # (recommandé pour Invaders : 1 à 5, ici typiquement 2)
        input_size=input_size,  # Taille de l'entrée pour un CNN (N,L,H) après 3 convolutions 3200 entrées si (104,112)         
        # Taille totale de l'entrée pour le MLP
        # (pour Invaders, typiquement entre 100 et 300 ;
        # pour un CNN, fournir un tuple (channels, height, width))
        hidden_layers=2,  # Nombre de couches cachées dans le réseau
        # (recommandé : 1 à 3, ici 2 est courant)
        hidden_size=512,  # Nombre de neurones par couche cachée (512x1 pour Atari en cnn/rainbow, 128x2 ou 256x2 proposé par chatGPT)
        # (recommandé : 64 à 512, 192 est souvent un bon compromis)
        output_size=6,  # Nombre de sorties (actions possibles)
        # (fixe pour Invaders : 6 actions)
        learning_rate=0.0000625,  #0.0000625 pour rainbow (0.001 pour space invaders) et 0.00025 pour DeepMind => Taux d'apprentissage pour l'optimiseur Adam
        # (recommandé : entre 0.0001 et 0.01 pour un bon compromis)
        gamma=0.995,  # Facteur de discount pour valoriser les récompenses futures 0.99 pour deepmind
        # (recommandé : entre 0.9 et 0.9999, ici très élevé pour privilégier l'avenir)
        use_noisy=use_noisy,
        rainbow_eval=rainbow_eval,  # Nombre de steps avant de commencer les evaluations (250 000 pour Rainbow)
        rainbow_eval_pourcent=rainbow_eval_pourcent,  # Pourcentage de rainbow_eval_steps pour évaluer le modèle
        epsilon_start=epsilon_start,
        epsilon_end=epsilon_end,
        epsilon_linear=epsilon_linear,
        epsilon_decay=epsilon_decay,
        epsilon_add=epsilon_add,
        buffer_capacity=2000_000,  # Capacité maximale du replay buffer pour cnn/dreamerv2 10 000 vaut 4 Go de GPU RAM minimum !
        # (recommandé mlp: de 10 000 à 1 000 000, ici 100 000 est courant)
        batch_size=64,  # Taille du lot d'échantillons pour l'entraînement
        # (recommandé : entre 32 et 256, ici 128)
        min_history_size=20000,  # Nombre d'échantillons minimum dans le replay buffer avant de commencer l'entraînement (80k rainbow)
        prioritized_replay=True,  # Activation du replay buffer prioritaire 
        target_update_freq=10000,  #32000 pour rainbow soit 10 episodes x nbsteps/episodes (~1000) ou soit batch_size*10
        double_dqn=True,  # Activation du Double DQN (True pour réduire l'overestimation des Q-valeurs)
        dueling=True,
        nstep= True,   # ← option nstep activable
        nstep_n= 10,      # ← valeur par défaut (3 ou 5) 3 pour rainbow et mais 5 si jeu déterministe
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
                print("\n=====>F8====> Afficher Frame CNN ou get_state MLP\n")
            elif (
                config.mode == "exploration"
                and keyboard.is_pressed("shift")
                and keyboard.is_pressed("ctrl")
                and keyboard.is_pressed("f9")
            ):
                print("\n====>F9====> Exploration =====> Exploitation")
                trainer.config.mode = "exploitation"
                trainer.set_mode(trainer.config.mode)
            elif (
                config.mode == "exploitation"
                and keyboard.is_pressed("shift")
                and keyboard.is_pressed("ctrl")
                and keyboard.is_pressed("f9")
            ):
                print("\n=====>F9====> Exploitation =====>  Exploration")
                trainer.config.mode = "exploration"
                trainer.set_mode(trainer.config.mode)
            elif (
                keyboard.is_pressed("shift")
                and keyboard.is_pressed("ctrl") 
                and keyboard.is_pressed("f11")
                ):
                flag_F11_pressed = True
                print("\n=====>====> Debug Lua\n")
    keyboard.on_press(on_key_press)

    # Instanciation de DQNTrainer
    if config.model_type.lower() == "dreamer": 
        trainer = DreamerTrainer(config)
    else:
        trainer = DQNTrainer(config)
    print(f"Utilisation de l'appareil : {trainer.device}")
    trainer.load_model("./invaders.pth")
    _flag=trainer.load_buffer("./invaders.buffer")
    if _flag==-1:
        pass

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
    list_of_epsilons_or_sigmas = [[],[]]
    list_of_cumulated_steps = []
    mean_score = mean_score_old = last_score = high_score = 0

    # Déterminer la taille correcte de la frame vide en fonction du modèle
    if config.model_type.lower() in ("cnn", "dreamer"):
        empty_frame = np.zeros(
            (config.input_size[1], config.input_size[2]), dtype=np.float32
        )  # (H, W)
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
        (f"[model_type={config.model_type}][double_dqn={config.double_dqn}][dueling={config.dueling}][nstep={config.nstep}/{config.nstep_n if config.nstep else chr(8)}]\n")+
        (f"[*** Mode EXPLOITATION ***]\n" if trainer.config.mode == "exploitation" else f"")+
        (f"[input={config.input_size}*{config.state_history_size}={config.input_size*config.state_history_size}]"
        if config.model_type.lower() == "mlp" else f"[input={config.input_size}]")+
        (f"[hidden={config.hidden_size}*{config.hidden_layers}#{nb_parameters(config.input_size, config.hidden_layers, config.hidden_size, config.output_size)}]")+
        (f"[output={config.output_size}][gamma={config.gamma}][learning={config.learning_rate}]\n")+
        (f"[epsilon start, end, linear?, add={config.epsilon_start},{config.epsilon_end},{config.epsilon_decay==0},{config.epsilon_add}]\n"
        if not config.use_noisy else f"[NoisyNet]\n")+
        f"[Replay=capacity,batch_size,prioritized_replay={config.buffer_capacity},{config.batch_size},{config.prioritized_replay}]\n"
        f"[nb_mess_frame={NB_DE_DEMANDES_PAR_STEP}]"
        f"[nb_step_frame={NB_DE_FRAMES_STEP}][speed={vitesse_de_jeu}]"
    )
    nb_steps_total = 0  # Compte le nombre d'actions de step avec state effectuées mais pas forcément de train_step
    num_episodes = 99999
    for episode in range(num_episodes):
        trainer.config.current_episode = episode
        if record:
            recorder.start_recording()
        step = reward = sum_rewards = NewGameStarting = PlayerIsOK = NotEndOfGame = score = 0
        colonnes_deja_detruites = [False] * 11 # 11 colonnes des aliens non détruite en début d'episodes
        if flag_quit:
            print("Sortie prématurée de la boucle 'for'.")
            break

        comm.communicate([f"wait_for 1"])
        comm.communicate([f"write_memory {numCoins}(1)"])
        
        # Vider l'historique des frames brutes (géré par DQNTrainer ou ici si Dreamer)
        if hasattr(trainer, 'state_history') and isinstance(trainer.state_history, deque):
             trainer.state_history.clear()
        else: # Si DreamerTrainer n'a pas cet attribut, on le crée temporairement pour la logique ici
             local_state_history = deque(maxlen=config.state_history_size)
        for _ in range(config.state_history_size):
            # `config.state_extractor()` retourne (frame_brute, reward_supplementaire_de_state)
            current_single_frame, _ = config.state_extractor()
            if hasattr(trainer, 'state_history') and isinstance(trainer.state_history, deque):
                trainer.state_history.append(current_single_frame)
            else:
                local_state_history.append(current_single_frame)
        # Construire la première pile d'observations (o_0)
        if hasattr(trainer, 'state_history') and isinstance(trainer.state_history, deque):
            initial_observation_stack = np.stack(trainer.state_history, axis=0)
        else:
            initial_observation_stack = np.stack(local_state_history, axis=0)
        # `initial_observation_stack` est (N, H, W)

        # ---- Initialisation spécifique à DreamerV2 ----
        if config.model_type.lower() == "dreamer":
            trainer.encode_state_initial(initial_observation_stack)
            prev_action_value = 0 # Action fictive pour le premier pas du RSSM (ex: "ne rien faire")


        while NotEndOfGame == 0:
            NotEndOfGame = int(comm.communicate([f"read_memory {player1Alive}"])[0])
        while PlayerIsOK == 0:
            PlayerIsOK = int(comm.communicate([f"read_memory {playerOK}"])[0])      
        while NewGameStarting != 55:
            NewGameStarting = int(comm.communicate([f"read_memory {numAliens}"])[0])    
        comm.communicate([f"wait_for {NB_DE_DEMANDES_PAR_STEP}"])
        start_steps_time = time.time()  # 🔥 Début de la mesure d'un step
        nb_vies=3
        current_loop_observation_stack = initial_observation_stack # o_t
        # Déplacer la création de `local_state_history_for_invaders_loop` ici si nécessaire
        # Cette deque est pour la logique de `invaders.py` pour empiler les frames individuelles
        # reçues de `config.state_extractor()` pour former la pile `(N,H,W)`.
        invaders_loop_frame_history = deque(maxlen=config.state_history_size)
        for frame_in_stack in initial_observation_stack: # Pré-remplir avec les frames initiales
            invaders_loop_frame_history.append(frame_in_stack)
        while NotEndOfGame == 1:
            # 1. Sélection d'action
            if config.model_type.lower() == "dreamer":
                # `current_loop_observation_stack` est o_t (N,H,W)
                # `prev_action_value` est a_{t-1}
                action = trainer.dreamer_step(current_loop_observation_stack, prev_action_value)
            else: # CNN / MLP
                # `current_loop_observation_stack` est l'état pour DQN (peut être (N,H,W) ou vecteur concaténé)
                # La logique existante pour CNN/MLP devrait fonctionner si `current_loop_observation_stack` est bien formaté.
                # Pour CNN, `state` est `np.stack(trainer.state_history, axis=0)`
                # Pour MLP, `state` est `np.concatenate(trainer.state_history, axis=0)`
                # Assurons-nous que `current_loop_observation_stack` est correct pour DQN.
                # La logique originale était:
                # current_state0, reward_state = config.state_extractor()
                # trainer.state_history.append(current_state0)
                # state_for_dqn = np.stack(trainer.state_history, axis=0) # ou concatenate pour MLP
                # action = trainer.select_action(state_for_dqn)
                # CE QUI EST FAIT CI-DESSOUS JUSTE AVANT L'APPEL A `select_action` DANS LA BOUCLE ORIGINALE.
                # Donc, `current_loop_observation_stack` est bien o_t pour Dreamer.
                # Pour DQN, nous allons reconstruire l'état comme avant.
                
                # Pour DQN:
                # La logique originale de `invaders.py` pour CNN/MLP:
                # current_single_frame_dqn, reward_state_dqn = config.state_extractor()
                # trainer.state_history.append(current_single_frame_dqn)
                # if config.model_type.lower() == "cnn":
                #     state_for_dqn_action = np.stack(trainer.state_history, axis=0)
                # else: # MLP
                #     state_for_dqn_action = np.concatenate(trainer.state_history, axis=0)
                # action = trainer.select_action(state_for_dqn_action)
                # Cette logique est recréée plus bas si ce n'est pas Dreamer.
                # Pour l'instant, `current_loop_observation_stack` est l'observation pour Dreamer.
                # La variable `state` dans votre code original était `current_loop_observation_stack`
                action = trainer.select_action(current_loop_observation_stack if config.model_type.lower() == "cnn" else np.concatenate(list(invaders_loop_frame_history),axis=0))

            # 2. Exécuter l'action
            executer_action(action)
            # `action` est a_t

            # 3. Obtenir la nouvelle observation brute, la récompense, et 'done'
            # `next_single_frame` est l'image brute o'_{t+1}
            next_single_frame, reward_state_component = config.state_extractor()
            _last_score = score
            score = get_score()
            PlayerIsOK, NotEndOfGame = list(map(int, comm.communicate([
                f"read_memory {playerOK}",
                f"read_memory {player1Alive}"]),
            ))

            # Calcul de la récompense r_t
            if PlayerIsOK == 1:
                reward = reward_alive
            elif nb_vies > 1: # Le joueur est mort mais a encore des vies
                reward = reward_kill
            else: # Le joueur est mort et c'est la dernière vie (fin de partie)
                reward = reward_kill + reward_end_of_game

            reward += (
                ((score - _last_score) * reward_aliens_mult) +
                (step * reward_mult_step) + # step est le compteur de pas dans l'épisode
                reward_state_component
            )
            if reward_clipping_deepmind:
                reward = np.sign(reward) # Clip reward à -1, 0, 1

            sum_rewards += reward
            done = PlayerIsOK == 0

            # 4. Construire la pile d'observations suivante (o_{t+1})
            invaders_loop_frame_history.append(next_single_frame)
            next_observation_stack = np.stack(invaders_loop_frame_history, axis=0) # (N,H,W)
            # `next_observation_stack` est o_{t+1}

            # 5. Stocker la transition
            if config.model_type.lower() == "dreamer":
                # `current_loop_observation_stack` est o_t
                # `action` est a_t
                # `reward` est r_t
                # `next_observation_stack` est o_{t+1}
                # `done` est d_t
                trainer.store_transition(current_loop_observation_stack, action, reward, next_observation_stack, False)
            elif config.mode.lower() == "exploration": # Pour CNN/MLP
                # La logique originale pour DQN:
                # state_for_buffer = current_loop_observation_stack (ou sa version MLP)
                # next_state_for_buffer = next_observation_stack (ou sa version MLP)
                if config.model_type.lower() == "cnn":
                    state_for_buffer_dqn = current_loop_observation_stack
                else: # MLP
                    # Utiliser l'état MLP tel qu'il était au moment de la sélection d'action
                    state_for_buffer_dqn = np.concatenate(list(invaders_loop_frame_history), axis=0)
                next_state_for_buffer_dqn = next_observation_stack if config.model_type.lower() == "cnn" else np.concatenate(list(invaders_loop_frame_history), axis=0)

                if trainer.config.nstep:
                    nstep_tr = trainer.nstep_wrapper.append(state_for_buffer_dqn, action, reward, done, next_state_for_buffer_dqn)
                    if nstep_tr:
                        trainer.replay_buffer.push(*nstep_tr)
                    if done: # S'assurer que `done` est bien d_t ici.
                        for tr_flush in trainer.nstep_wrapper.flush():
                            trainer.replay_buffer.push(*tr_flush)
                else:
                    trainer.replay_buffer.push(state_for_buffer_dqn, action, reward, next_state_for_buffer_dqn, done)

            # 6. Mettre à jour l'état pour le prochain pas de la boucle
            current_loop_observation_stack = next_observation_stack # o_t devient o_{t+1}
            if config.model_type.lower() == "dreamer":
                prev_action_value = action # a_{t-1} devient a_t
            else: # CNN/MLP
                trainer.update_epsilon() # Géré par DQNTrainer

            # 7. Apprentissage
            if nb_steps_total % TRAIN_EVERY_N_GLOBAL_STEPS == 0:
                trainer.train_step()
            step += 1
            nb_steps_total +=1
            
            # 7.bis. Affichage de la frame ou debug ou appui de touche
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
            comm.number_of_messages = 0   
            if flag_F11_pressed:
                debug_lua=toggle_debug_lua(debug_lua)
                flag_F11_pressed=False
                print(f" ==============  toggle_debug_lua({not debug_lua}) <====> F11")
            if flag_F8_pressed:
                if config.model_type == "cnn" or config.model_type == "dreamer":
                    print("====>F8====> Affichage d'une frame (la plus récente observée)")
                    afficher_frame(next_single_frame, factor_div=factor_div_frame)
                elif config.model_type == "mlp":
                    print("====>F8====> Affichage de l'état MLP get_state()")
                    afficher_get_state()
                flag_F8_pressed=False
            # 8. Gestion de la fin de vie du joueur et debug et appuies de touches
            if PlayerIsOK == 0:
                comm.communicate(["wait_for 2"])        
                while PlayerIsOK == 0 and NotEndOfGame == 1:
                    PlayerIsOK, NotEndOfGame = list(map(int,comm.communicate([f"read_memory {playerOK}", f"read_memory {player1Alive}"]),))
                if PlayerIsOK == 1: 
                    nb_vies-=1
                    comm.communicate([f"wait_for {NB_DE_DEMANDES_PAR_STEP}"])
                if NotEndOfGame == 0 and config.model_type.lower() == "dreamer":
                    trainer.store_transition(current_loop_observation_stack, action, reward, next_observation_stack, True)

        response = comm.communicate(["wait_for 0"])
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
                list_of_epsilons_or_sigmas,
                high_score,flag_aliens,flag_boucliers,
                list_of_cumulated_steps,
                reward_str=reward_str,  # 👈 ajout ici
                filename="Invaders_fig_ask",
            )
            flag_create_fig = False
            time.sleep(0.2)
        if use_noisy:
            sigmas = trainer.dqn.get_sigma_values()
            # 1️⃣ Récupère la hidden layer
            if trainer.config.model_type.lower() == "cnn":
                sigma_hidden1 = sigmas.get("fc1", 0.0)
            else:
                sigma_hidden1 = sigmas.get("hidden_modules.0.0", 0.0)

            # 2️⃣ Récupère la couche de sortie (output)
            if trainer.config.dueling:
                # En dueling, la sortie = advantage_head (actions)
                sigma_output = sigmas.get("advantage_head", 0.0)
            else:
                # Sinon c'est output_layer
                sigma_output = sigmas.get("output_layer", 0.0)

            # 3️⃣ Stockage pour les courbes
            list_of_epsilons_or_sigmas[0].append(sigma_hidden1)
            list_of_epsilons_or_sigmas[1].append(sigma_output)
        else:
            list_of_epsilons_or_sigmas[0].append(trainer.epsilon)

        list_of_cumulated_steps.append(nb_steps_total)
        collection_of_score.append(score)
        mean_score_old = mean_score
        mean_score = round(sum(collection_of_score) / len(collection_of_score), 2)
        list_of_mean_scores.append(mean_score)
        if score > high_score: high_score = score  # Mise à jour du high score si un meilleur score est atteint
        if not (config.model_type.lower() == "dreamer" or config.use_noisy): # Epsilon update pour DQN non-bruyant
            if mean_score_old > mean_score:
                 trainer.epsilon += config.epsilon_add if trainer.epsilon < config.epsilon_start else 0.0
            if trainer.epsilon < 0.001 and trainer.config.mode == "exploration":
                trainer.config.mode = "exploitation"
                trainer.set_mode(trainer.config.mode)
                print("====================================> Mode EXPLOITATION car epsilon < 0.001 <====")

        if mean_score == mean_score_old:
            pygame.mixer.Sound("c:\\Windows\\Media\\tada.wav").play()
        _d = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        #nb_steps_total+=step
        if config.use_noisy:
            sigma_vals = trainer.dqn.get_sigma_values()
            sigma_str = ", ".join([f"{name} = {val:.6f}" for name, val in sigma_vals.items()])
            exploration_str = f"[sigma {sigma_str}]"
        else:
            exploration_str = f"[ε={trainer.epsilon:.4f}]"
        exploration_str = exploration_str  if config.mode != "exploitation" else "[*** Mode EXPLOITATION ***]"
        exploration_str = exploration_str if config.model_type != "dreamer" else "[Mode DREAMER.V2]"
        print(
            f"N°{episode+1} [{_d}][steps_ep,all={step:4d},{str(nb_steps_total//1000)+'k':>5}]"
            + (
                f"[Buffer={trainer.replay_buffer.size/1000:.0f}k/{trainer.config.buffer_capacity/1000:.0f}k]"
                if hasattr(trainer, "replay_buffer")
                else ""
            )
            + exploration_str
            + f"[rewards={sum_rewards:5.0f}]"
            + f"[score={score:3d}][score moyen={mean_score:3.0f}]"
        )
        reward_str = f"Rewards: alive={reward_alive:.2f}, step×={reward_mult_step:.6f}, Xpoints/alien={reward_aliens_mult}, kill={reward_kill:3.0f},\n"
        reward_str+= f"         reward_state={mult_reward_state}, rewardend_of_game={reward_end_of_game}, reward_clipping_deepmind={reward_clipping_deepmind}"
        if (episode + 1) % 10 == 0:
            print(f"Sauvegarde du modèle \"invaders.pth\" et création du graphique pour le {episode + 1}ème épisode")
            trainer.save_model("./invaders.pth")
            _pente = create_fig(
                trainer,NB_DE_FRAMES_STEP,
                episode,
                list_of_mean_scores,
                fenetre_du_calcul_de_la_moyenne,
                list_of_epsilons_or_sigmas,
                high_score,flag_aliens,flag_boucliers,
                list_of_cumulated_steps,
                reward_str=reward_str,  # 👈 ajout ici
                filename="Invaders_fig",
            )
        comm.communicate([f'draw_text(25,1,"Game number: {episode+1:04d} - mean score={mean_score:04.0f} - ba(c)o 2025")'])        
        
    final_titre = (
        f"{config.model_type}_{'DDQN_' if config.double_dqn else 'DQN'}{'_Dueling' if config.dueling else ''}{'_Prioritized' if config.prioritized_replay else ''}{'_nstep' if config.nstep else ''}"
        f"_N={config.state_history_size}{config.state_history_size}_i={config.input_size}_o={config.output_size}"
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
            list_of_epsilons_or_sigmas,
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
