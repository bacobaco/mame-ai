"""
invaders_multi.py
"""
import os
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = '1'
import time, subprocess, sys, psutil, socket, queue
import threading
import numpy as np
import torch
import torch.multiprocessing as mp
from collections import deque
from datetime import datetime
from colorama import Fore, Style

# ==================================================================================================
# CONFIGURATION DES CHEMINS (RELATIFS)
# ==================================================================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPT_DIR)
CORE_DIR = os.path.join(ROOT_DIR, "core")
MEDIA_DIR = os.path.join(ROOT_DIR, "media")

# Assurer la présence des dossiers
os.makedirs(MEDIA_DIR, exist_ok=True)

# Ajout du dossier 'core' au sys.path pour les imports locaux
if CORE_DIR not in sys.path:
    sys.path.append(CORE_DIR)

# Imports locaux depuis le dossier CORE
from MameCommSocket import MameCommunicator
from AI_Mame import TrainingConfig, DQNTrainer, GraphWebServer, NStepTransitionWrapper, TeeLogger
from invaders import Memory, GameConstants, InvadersInterface, StateExtractor

# --- CONFIGURATION LOGIQUE ---
NUM_ACTORS = 4
ACTOR_DEVICE = "cuda:0"  # "cpu" ou "cuda:0"
BASE_PORT = 12345
MAME_PATH = "D:\\Emulateurs\\Mame Officiel"
LUA_BRIDGE_PATH = os.path.join(CORE_DIR, "PythonBridgeSocket.lua").replace('\\', '/')
MODEL_FILENAME = os.path.join(SCRIPT_DIR, "invaders_apex.pth")
BUFFER_FILENAME = os.path.join(SCRIPT_DIR, "invaders_multi.buffer")
RESULTS_FILENAME = os.path.join(SCRIPT_DIR, "resultats_invaders.txt")

def kill_existing_mame():
    print(f"{Fore.YELLOW}🧹 Arrêt des processus MAME existants...{Style.RESET_ALL}")
    for proc in psutil.process_iter(['pid', 'name']):
        try:
            if proc.info['name'] and 'mame' in proc.info['name'].lower():
                proc.terminate()
        except: pass
    time.sleep(1)

def launch_mame_instance(port):
    command = [
        os.path.join(MAME_PATH, "mame.exe"),
        "-window", "-resolution", "448x576",
        "-skip_gameinfo", "-artwork_crop",
        "-video", "none", "-sound", "none", "-nothrottle",
        "-console", "-noautosave", "invaders",
        "-autoboot_delay", "1",
        "-autoboot_script", LUA_BRIDGE_PATH,
    ]
    env_vars = os.environ.copy()
    env_vars["MAME_SOCKET_PORT"] = str(port)
    startupinfo = subprocess.STARTUPINFO()
    startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
    startupinfo.wShowWindow = 7
    return subprocess.Popen(command, cwd=MAME_PATH, startupinfo=startupinfo, env=env_vars, 
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

def actor_process(actor_id, port, config, transition_queue, score_queue, shared_weights, device_type):
    """
    Processus autonome qui gère un acteur (Jeu MAME + Inférence)
    """
    os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = '1'
    torch.set_num_threads(1)
    
    time.sleep(actor_id * 2) 
    print(f"🎮 Démarrage Acteur {actor_id} => Port {port} (Hardware: {device_type.upper()})")
    
    config.device = device_type
    
    # CORRECTION CRITIQUE (RAM) : Les acteurs ne font pas de backpropagation (entraînement) !
    # Inutile de créer un ReplayBuffer de 100 000 images. Sinon : 8 Acteurs x 7 Go = 56 Go de RAM = Crash Windows
    config.buffer_capacity = 10
    config.prioritized_replay = False # Gain de temps CPU
    
    trainer = DQNTrainer(config, silent=True)
    
    # L'epsilon est déjà pré-attribué par le manager dans config.epsilon_start
    trainer.epsilon = config.epsilon_start 
    
    comm = MameCommunicator("127.0.0.1", port, deferred_accept=True)
    process = launch_mame_instance(port)
    comm.accept_connection()
    
    game = InvadersInterface(comm)
    colonnes_deja_detruites = [False] * 11
    
    local_extractor = StateExtractor(game, config.model_type, False, False, 2, 0.01, colonnes_deja_detruites)
    actor_nstep = NStepTransitionWrapper(config.nstep_n, config.gamma) if config.nstep else None
    
    comm.communicate([
        f"write_memory {Memory.NUM_COINS}(1)",
        "execute P1_start(1)",
        f"execute throttle_rate(10.0)",
        "execute throttled(0)",
        "frame_per_step 3",
    ])
    
    _N = config.state_history_size
    next_weights_update = 0.0 # Force la récupération de invaders_best.pth dès la TUTE PREMIERE image !
    
    try:
        while True:
            score = 0
            for i in range(11): colonnes_deja_detruites[i] = False
            
            comm.communicate([f"write_memory {Memory.NUM_COINS}(1)"])
            local_state_history = deque(maxlen=_N)
            for _ in range(_N):
                frame, _ = local_extractor()
                local_state_history.append(frame)
            
            initial_obs_stack = np.stack(local_state_history, axis=0)
            
            NotEndOfGame = 0; PlayerIsOK = 0; NewGameStarting = 0
            while NotEndOfGame == 0:
                res = comm.communicate([f"read_memory {Memory.PLAYER_1_ALIVE}"])[0]
                NotEndOfGame = int(res) if res else 0
                time.sleep(0.01)
            while PlayerIsOK == 0:
                res = comm.communicate([f"read_memory {Memory.PLAYER_OK}"])[0]
                PlayerIsOK = int(res) if res else 0
                time.sleep(0.01)
            while NewGameStarting != 55:
                res = comm.communicate([f"read_memory {Memory.NUM_ALIENS}"])[0]
                NewGameStarting = int(res) if res else 0
                time.sleep(0.01)

            current_loop_obs_stack = initial_obs_stack 
            invaders_loop_frame_history = deque(maxlen=_N)
            for f in initial_obs_stack: invaders_loop_frame_history.append(f)

            while NotEndOfGame == 1:
                if time.time() > next_weights_update:
                    if 'model' in shared_weights:
                        try:
                            trainer.dqn.load_state_dict(shared_weights['model'])
                            trainer.dqn.reset_noise()
                        except: pass
                    next_weights_update = time.time() + 2.0
                
                action = trainer.select_action(current_loop_obs_stack)
                
                # --- ÉTAPE ATOMIQUE ---
                _last_score = score
                next_frame, reward_state_comp, score, PlayerIsOK, NotEndOfGame, lives = game.get_complete_step(action, factor_div=2, mult_reward_state=0.01)
                
                if next_frame is None: break # Sécurité socket

                if NotEndOfGame == 0:
                    reward = -100.0
                    if PlayerIsOK == 1: reward -= 400.0
                    done = True
                elif PlayerIsOK == 1:
                    reward = 0.0; done = False
                elif lives > 0:
                    reward = -15.0; done = False
                else:
                    reward = -120.0; done = True

                reward += ((score - _last_score) * 2.0) - 0.005 + reward_state_comp

                invaders_loop_frame_history.append(next_frame)
                next_obs_stack = np.stack(invaders_loop_frame_history, axis=0)

                def safe_put(item):
                    try:
                        transition_queue.put_nowait(item)
                    except queue.Full:
                        pass # Si le Learner GPU est trop lent, on jette la frame pour ne pas figer MAME

                if config.nstep:
                    nstep_tr = actor_nstep.append(current_loop_obs_stack, action, reward, done, next_obs_stack)
                    if nstep_tr: safe_put((nstep_tr[0], nstep_tr[1], float(nstep_tr[2]), nstep_tr[3], bool(nstep_tr[4])))
                    if done:
                        for tr in actor_nstep.flush(): safe_put((tr[0], tr[1], float(tr[2]), tr[3], bool(tr[4])))
                else:
                    safe_put((current_loop_obs_stack, action, float(reward), next_obs_stack, bool(done)))

                current_loop_obs_stack = next_obs_stack
                
                if PlayerIsOK == 0:
                    comm.communicate(["wait_for 5"])
                    while PlayerIsOK == 0 and NotEndOfGame == 1:
                        score, PlayerIsOK, NotEndOfGame, lives = game.get_score_and_status(score)

            comm.communicate(["wait_for 1"])
            score_queue.put((actor_id, score)) 
            trainer.dqn.reset_noise() 

    except Exception as e:
        print(f"💀 Erreur interne Acteur {actor_id}: {e}")
        process.terminate()
    except KeyboardInterrupt:
        process.terminate()

class ApeXManager:
    def __init__(self):
        logs_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
        self.web_server = GraphWebServer(graph_dir=logs_dir, host="0.0.0.0", port=5000, auto_display_latest=True)
        threading.Thread(target=self.web_server.start, daemon=True).start()
        self.global_steps = 0
        self.global_episodes = 0
        self.experiment_id = self.get_next_experiment_id()

    def get_next_experiment_id(self):
        """Lit le fichier de résultats pour déterminer le prochain ID d'expérience."""
        last_idx = 0
        if os.path.exists(RESULTS_FILENAME):
            try:
                with open(RESULTS_FILENAME, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line: continue
                        idx = -1
                        # Format: "[31][model..." ou "30[input..."
                        if line.startswith('['):
                            parts = line.split(']')
                            if parts:
                                val = parts[0][1:].strip()
                                if val.isdigit(): idx = int(val)
                        elif line[0].isdigit():
                            parts = line.split('[')
                            if parts[0].strip().isdigit(): idx = int(parts[0].strip())
                        if idx > last_idx: last_idx = idx
            except: pass
        return last_idx + 1

    def run(self):
        config = TrainingConfig(
            state_history_size=4, input_size=(4, 96, 100),
            hidden_layers=1, hidden_size=1024, output_size=6,
            learning_rate=0.0000625, gamma=0.999, # LR plus bas pour Noisy (standard Rainbow)
            use_noisy=True, rainbow_eval=250_000, rainbow_eval_pourcent=2,
            epsilon_start=0.0, epsilon_end=0.0, epsilon_linear=0.0,
            epsilon_decay=0, epsilon_add=0,
            buffer_capacity=100_000, batch_size=256, min_history_size=20000,
            prioritized_replay=True, target_update_freq=5000,
            double_dqn=True, dueling=True, nstep=True, nstep_n=5,
            model_type="cnn", cnn_type="precise", mode="exploration", optimize_memory=True
        )

        trainer = DQNTrainer(config)
        
        # Priorité à la progression en cours (apex), sinon au meilleur modèle (best)
        model_to_load = None
        if os.path.exists(MODEL_FILENAME):
            model_to_load = MODEL_FILENAME
        elif os.path.exists(os.path.join(SCRIPT_DIR, "invaders_best.pth")):
            model_to_load = os.path.join(SCRIPT_DIR, "invaders_best.pth")

        if model_to_load:
            try:
                trainer.load_model(model_to_load)
                print(f"{Fore.CYAN}♻️ Reprise de l'entraînement depuis : {model_to_load}{Style.RESET_ALL}")
            except Exception as e:
                print(f"{Fore.RED}⚠️ Impossible de charger {model_to_load}. On repart de ZÉRO !{Style.RESET_ALL}")
            
        manager = mp.Manager()
        transition_queue = mp.Queue(maxsize=50000)
        score_queue = mp.Queue()
        shared_weights = manager.dict()
        
        shared_weights['model'] = {k: v.cpu() for k, v in trainer.dqn.state_dict().items()}
        
        def find_free_port(start_port):
            while True:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                    try:
                        s.bind(('127.0.0.1', start_port))
                        return start_port
                    except OSError:
                        start_port += 1

        # --- EXPLORATION (Distribution si Epsilon, Zéro si Noisy) ---
        epsilons = []
        if not config.use_noisy:
            epsilon_min = 0.01; epsilon_max = 0.40
            for i in range(NUM_ACTORS):
                e = epsilon_min + (epsilon_max - epsilon_min) * (i / max(1, NUM_ACTORS - 1))**2
                epsilons.append(e)
            explo_range_str = f"[{min(epsilons):.2f} -> {max(epsilons):.2f}]"
            print(f"\n{Fore.GREEN}🚀 Lancement de {NUM_ACTORS} Acteurs avec distribution Dynamique :")
            print(f"   👉 Epsilon Range: {explo_range_str}{Style.RESET_ALL}")
        else:
            epsilons = [0.0] * NUM_ACTORS
            explo_range_str = "NoisyNet"
            print(f"\n{Fore.MAGENTA}🚀 Lancement de {NUM_ACTORS} Acteurs en mode NOISY-NET :")
            print(f"   👉 Exploration intelligente gérée par les poids du réseau.{Style.RESET_ALL}")

        processes = []
        curr_port = BASE_PORT
        try:
            for i in range(NUM_ACTORS):
                curr_port = find_free_port(curr_port)
                actor_config = config.clone()
                actor_config.epsilon_start = epsilons[i]
                
                # --- SILENCE POUR LES ACTEURS ---
                p = mp.Process(target=actor_process, args=(i, curr_port, actor_config, transition_queue, score_queue, shared_weights, ACTOR_DEVICE))
                p.daemon = True
                p.start()
                processes.append(p)
                time.sleep(1.0)
                curr_port += 1

            print(f"\n{Fore.MAGENTA}🧠 [Learner] Cerveau Prêt. En attente de {config.min_history_size} steps...{Style.RESET_ALL}")
            collection_score = deque(maxlen=100)
            best_mean_score = 650.0
            
            last_log_time = time.time()
            last_sync = time.time()
            last_saved_episode = -1
            
            last_steps = 0
            last_fill_log = 0
            global_max_score = 0
            start_time = time.time()
            while True:
                # 1. Vidage de la file (Dynamique pour garantir la fraîcheur)
                q_sz = transition_queue.qsize() if sys.platform != "darwin" else 0
                max_ingest = 1000 if q_sz < 5000 else 5000 # On vide plus vite si ça sature
                
                num_ingested = 0
                while not transition_queue.empty() and num_ingested < max_ingest:
                    try:
                        s, a, r, ns, d = transition_queue.get_nowait()
                        trainer.replay_buffer.push(s, a, r, ns, d)
                        num_ingested += 1
                        self.global_steps += 1
                        # Log de remplissage tous les 2000 steps pendant la phase d'attente
                        if trainer.replay_buffer.size < config.min_history_size:
                            if trainer.replay_buffer.size // 2000 > last_fill_log:
                                last_fill_log = trainer.replay_buffer.size // 2000
                                print(f"{Fore.CYAN}📥 Remplissage Buffer: {trainer.replay_buffer.size}/{config.min_history_size}...{Style.RESET_ALL}")
                    except: break
                        
                # 2. Avaler les scores
                while not score_queue.empty():
                    try:
                        actor_id, new_score = score_queue.get_nowait()
                        collection_score.append(new_score)
                        if new_score > global_max_score:
                            global_max_score = new_score
                        self.global_episodes += 1
                        
                        # Sauvegarde toutes les 100 épisodes (fréquence réduite pour plus de fluidité)
                        if self.global_episodes > 0 and (self.global_episodes % 100) == 0 and self.global_episodes != last_saved_episode:
                            last_saved_episode = self.global_episodes
                            trainer.save_model(MODEL_FILENAME)
                            mean_score = sum(collection_score) / len(collection_score) if collection_score else 0
                            if mean_score > (best_mean_score + 10):
                                best_mean_score = mean_score
                                trainer.save_model(os.path.join(SCRIPT_DIR, "invaders_best.pth"))
                                print(f"{Fore.YELLOW}🏆 [RECORD] Nouveau record de moyenne : {best_mean_score:.1f} !{Style.RESET_ALL}")
                    except:
                        break

                # 3. Entraînement au fil de l'eau (Interleaved)
                buf_len = len(trainer.replay_buffer)
                if buf_len >= config.min_history_size:
                    # On s'entraîne, mais on limite à 10 pas max par itération pour ne pas bloquer les logs
                    train_count = 0
                    while last_steps < self.global_steps and train_count < 10:
                        trainer.train_step()
                        last_steps += 1
                        train_count += 1
                else:
                    time.sleep(0.01)
                        
                # 4. Log périodique indépendant (Toutes les 5 secondes)
                if time.time() - last_log_time > 5.0:
                    mean_score = sum(collection_score) / len(collection_score) if collection_score else 0
                    q_sz = transition_queue.qsize() if sys.platform != "darwin" else 0
                    elapsed = time.time() - start_time
                    h = int(elapsed // 3600)
                    m = int((elapsed % 3600) // 60)
                    s = int(elapsed % 60)
                    time_str = f"{h:02d}h{m:02d}m{s:02d}s"
                    
                    explo_str = f"Explo: {explo_range_str}"
                    if config.use_noisy:
                        sigmas = trainer.dqn.get_sigma_values()
                        explo_val = sum(sigmas.values()) / len(sigmas) if sigmas else 0
                        explo_str = f"Sigma: {explo_val:.4f}"
                    
                    print(f"[{time_str}] Eps: {self.global_episodes:4d} | Pas Globaux: {self.global_steps/1000:5.1f}k | "
                          f"Mean: {mean_score:4.1f} | HiScore: {global_max_score} | Buffer: {buf_len/1000:4.1f}k | Q: {q_sz} | {explo_str}")
                    last_log_time = time.time()
                    
                    # Sync des poids
                    shared_weights['model'] = {k: v.cpu() for k, v in trainer.dqn.state_dict().items()}

        except KeyboardInterrupt:
            print(f"\n{Fore.YELLOW}🛑 Interruption détectée ! Sauvegarde des résultats finaux...{Style.RESET_ALL}")
            duration = (time.time() - start_time) / 3600
            mean_score = sum(collection_score) / len(collection_score) if collection_score else 0
            
            try:
                with open(RESULTS_FILENAME, "a", encoding="utf-8") as f:
                    f.write(f"\n\n{res_str}\n")
                print(f"\n{Fore.GREEN}✅ Résultats finaux sauvegardés élégamment dans resultats_invaders.txt{Style.RESET_ALL}")
            except Exception as e:
                print(f"\n{Fore.RED}❌ Erreur écriture résultats : {e}{Style.RESET_ALL}")
            
            sys.exit(0)

if __name__ == "__main__":
    mp.freeze_support()
    logger = TeeLogger() # On active le logger seulement ici pour ne pas saturer les fils
    app = ApeXManager()
    app.run()
