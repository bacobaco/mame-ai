import os, time, subprocess, sys, psutil, socket, queue
import threading
import numpy as np
import torch
import torch.multiprocessing as mp
from collections import deque
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
from AI_Mame import TrainingConfig, DQNTrainer, GraphWebServer, NStepTransitionWrapper
from pacman import Memory, GameConstants, PacmanInterface, StateExtractor, Visualizer

# --- CONFIGURATION MULTI ---
NUM_ACTORS = 6 
BASE_PORT = 12347 
MAME_PATH = "D:\\Emulateurs\\Mame Officiel"
LUA_BRIDGE_PATH = os.path.join(CORE_DIR, "PythonBridgeSocket.lua").replace('\\', '/')
MODEL_FILENAME = os.path.join(SCRIPT_DIR, "pacman_cnn_multi.pth")
BEST_MODEL_FILENAME = os.path.join(SCRIPT_DIR, "pacman_cnn_best.pth")
BUFFER_FILENAME = os.path.join(SCRIPT_DIR, "pacman_multi.buffer")
RESULTS_FILENAME = os.path.join(SCRIPT_DIR, "resultats_pacman.txt")

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
        "-console", "-noautosave", "pacman",
        "-autoboot_delay", "1",
        "-autoboot_script", LUA_BRIDGE_PATH,
    ]
    env_vars = os.environ.copy()
    env_vars["MAME_SOCKET_PORT"] = str(port)
    startupinfo = subprocess.STARTUPINFO()
    startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
    startupinfo.wShowWindow = 7 
    return subprocess.Popen(command, cwd=MAME_PATH, startupinfo=startupinfo, env=env_vars)

def actor_process(actor_id, port, config, transition_queue, score_queue, shared_weights):
    """Processus Acteur : Joue au jeu et envoie les transitions au Learner."""
    torch.set_num_threads(1) 
    time.sleep(actor_id * 2) 
    # Utilisation du GPU pour les acteurs (si la VRAM le permet)
    config.device = "cuda:0" if torch.cuda.is_available() else "cpu"
    config.buffer_capacity = 10 
    
    trainer = DQNTrainer(config)
    trainer.optimizer = None 
    
    comm = MameCommunicator("127.0.0.1", port, deferred_accept=True)
    process = launch_mame_instance(port)
    comm.accept_connection()
    
    game = PacmanInterface(comm)
    local_extractor = StateExtractor(game, config.model_type)
    actor_nstep = NStepTransitionWrapper(config.nstep_n, config.gamma) if config.nstep else None
    
    # Init MAME (Optimisé après boot)
    time.sleep(5) 
    comm.communicate(["wait_for 5"])
    comm.communicate([
        f"write_memory {Memory.CREDITS}(1)",
        "execute P1_start(1)",
        "execute throttle_rate(20.0)", 
        "execute throttled(0)",
        "frame_per_step 4",
    ])
    
    _N = config.state_history_size
    next_weights_update = 0.0 
    NB_DE_DEMANDES_PAR_STEP = 25 
    
    try:
        while True:
            # Episode Loop
            score = 0
            comm.communicate(["wait_for 2"]) 
            # LOUP DÉBUSQUÉ : Il faut ré-insérer un crédit ET appuyer sur Start à chaque nouvel épisode !
            comm.communicate([
                f"write_memory {Memory.CREDITS}(1)",
                "execute P1_start(1)"
            ])
            
            # Attente lancement
            alive = 0
            while alive == 0:
                comm.communicate(["wait_for 6"])
                data = game.get_score_and_lives()
                if data: _, _, alive, _ = data
                if alive == 0: time.sleep(0.01)
            
            # Init history
            local_state_history = deque(maxlen=_N)
            for _ in range(_N):
                frame, _ = local_extractor()
                local_state_history.append(frame)
            
            current_loop_obs_stack = np.stack(local_state_history, axis=0) # Sera (4, 64, 56)
            lives = 3
            done = False
            comm.communicate([f"wait_for {NB_DE_DEMANDES_PAR_STEP}"])

            while not done:
                # Synchro des poids plus fréquente (3s au lieu de 5s) pour une politique plus fraîche
                if time.time() > next_weights_update:
                    if 'model' in shared_weights:
                        try:
                            trainer.dqn.load_state_dict(shared_weights['model'])
                            trainer.dqn.reset_noise()
                        except: pass
                    next_weights_update = time.time() + 3.0
                
                action = trainer.select_action(current_loop_obs_stack)
                game.execute_action(action)
                responses = game.get_all_data_batched()
                batch_data = game.process_all_data(responses)
                
                if not batch_data: continue
                next_frame, new_score, new_lives, alive, pills = batch_data
                
                # Reward Scaling parité avec pacman.py
                reward = min((new_score - score) / 10.0, 160.0)
                if new_lives < lives:
                    reward -= 50
                    lives = new_lives
                reward -= 0.01 
                
                score = new_score
                local_state_history.append(next_frame)
                next_obs_stack = np.stack(local_state_history, axis=0)

                if alive == 0 and lives == 0:
                    done = True
                    reward -= 10

                def safe_put(item):
                    try:
                        transition_queue.put_nowait(item)
                    except queue.Full:
                        pass # Evite de figer l'acteur

                if config.nstep:
                    nstep_tr = actor_nstep.append(current_loop_obs_stack, action, reward, done, next_obs_stack)
                    if nstep_tr: safe_put(nstep_tr)
                    if done:
                        for tr in actor_nstep.flush(): safe_put(tr)
                else:
                    safe_put((current_loop_obs_stack, action, float(reward), next_obs_stack, bool(done)))

                current_loop_obs_stack = next_obs_stack

                if alive == 0 and not done:
                    while alive == 0:
                        comm.communicate(["wait_for 6"])
                        data = game.get_score_and_lives()
                        if data: _, lives, alive, _ = data
                        if lives == 0: break
                    comm.communicate([f"wait_for {NB_DE_DEMANDES_PAR_STEP}"])

            score_queue.put(score)
            trainer.dqn.reset_noise()

    except Exception as e:
        print(f"Erreur Acteur {actor_id}: {e}")
        process.terminate()

class PacmanApeX:
    """Orchestrateur central (Learner)"""
    def __init__(self):
        logs_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
        self.web_server = GraphWebServer(graph_dir=logs_dir, host="0.0.0.0", port=5000, auto_display_latest=True)
        threading.Thread(target=self.web_server.start, daemon=True).start()
        self.global_steps = 0
        self.global_episodes = 0
        self.mean_scores_history = []
        self.sigma_history = []
        self.max_score = 0
        self.experiment_id = self.get_next_experiment_id()

    def get_next_experiment_id(self):
        """Lit le fichier de résultats pour déterminer le prochain ID d'expérience spécifique à Pacman."""
        last_idx = 0
        if os.path.exists(RESULTS_FILENAME):
            try:
                with open(RESULTS_FILENAME, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line: continue
                        idx = -1
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
            state_history_size=4, input_size=(4, 64, 56),
            hidden_size=256, output_size=4, hidden_layers=2,
            learning_rate=0.0001, gamma=0.995,
            use_noisy=True, epsilon_start=0.6, epsilon_end=0.02, epsilon_linear=0.000001,
            epsilon_decay=0.0, epsilon_add=0.0,
            buffer_capacity=250_000, batch_size=128, min_history_size=30000,
            prioritized_replay=True, target_update_freq=10000,
            double_dqn=True, dueling=True, nstep=True, nstep_n=3,
            model_type="cnn", cnn_type="precise", mode="exploration", optimize_memory=True
        )

        kill_existing_mame()
        trainer = DQNTrainer(config)
        if os.path.exists(BEST_MODEL_FILENAME):
            trainer.load_model(BEST_MODEL_FILENAME)

        manager = mp.Manager()
        transition_queue = mp.Queue(maxsize=100000)
        score_queue = mp.Queue()
        shared_weights = manager.dict()
        shared_weights['model'] = {k: v.cpu() for k, v in trainer.dqn.state_dict().items()}

        processes = []
        for i in range(NUM_ACTORS):
            p = mp.Process(target=actor_process, args=(i, BASE_PORT + i, config, transition_queue, score_queue, shared_weights))
            p.daemon = True
            processes.append(p)
            p.start()

        time.sleep(NUM_ACTORS * 4)  
        print(f"\n{Fore.MAGENTA}🧠 [Learner] Démarrage de l'entraînement GPU...{Style.RESET_ALL}")
        collection_score = deque(maxlen=100)
        best_mean_score = 2500.0
        start_time = time.time()
        last_weight_sync = time.time()
        last_log_time = time.time()
        last_steps = 0
        try:
            while True:
                # 1. Ingestion via get_nowait (Non-bloquant) - Limite augmentée à 10k pour vider Q plus vite
                ingested = 0
                while not transition_queue.empty() and ingested < 10000:
                    try:
                        s, a, r, ns, d = transition_queue.get_nowait()
                        trainer.replay_buffer.push(s, a, r, ns, d)
                        self.global_steps += 1
                        ingested += 1
                    except: break
                
                # 2. Ingestion des scores
                while not score_queue.empty():
                    sc = score_queue.get_nowait()
                    collection_score.append(sc)
                    self.global_episodes += 1
                    if sc > self.max_score: self.max_score = sc

                # 3. Entraînement 
                if len(trainer.replay_buffer) >= config.min_history_size:
                    diff = self.global_steps - last_steps
                    if diff >= 4:
                        # Retour au ratio 1:4 plus sage pour laisser le Learner respirer
                        iters = min(diff // 4, 128) 
                        for _ in range(iters):
                            trainer.train_step()
                        last_steps = self.global_steps # CORRECTION : Empêche la dette infinie
                    
                    if time.time() - last_weight_sync > 3.0:
                        shared_weights['model'] = {k: v.cpu() for k, v in trainer.dqn.state_dict().items()}
                        last_weight_sync = time.time()

                # LOGS TOUTES LES 5 SECONDES (Sortis du bloc training pour voir progression remplissage buffer)
                if time.time() - last_log_time > 5.0:
                    mean_sc = np.mean(collection_score) if collection_score else 0
                    q_sz = transition_queue.qsize()
                    
                    if config.use_noisy:
                        sigmas = trainer.dqn.get_sigma_values()
                        explo_val = sum(sigmas.values()) / len(sigmas) if sigmas else 0
                        explo_str = f"Sigma: {Fore.BLUE}{explo_val:.4f}"
                        label_curve = "Sigma"
                    else:
                        explo_val = trainer.epsilon
                        explo_str = f"Eps: {Fore.BLUE}{explo_val:.4f}"
                        label_curve = "Epsilon"
                    
                    elapsed = time.time() - start_time
                    h, m, s = int(elapsed // 3600), int((elapsed % 3600) // 60), int(elapsed % 60)
                    time_str = f"{h:02d}h{m:02d}m{s:02d}s"
                    
                    buf_per = (len(trainer.replay_buffer) / config.min_history_size) * 100
                    buf_str = f"Buf: {len(trainer.replay_buffer)/1000:.1f}k ({buf_per:.0f}%)" if buf_per < 100 else f"Steps: {self.global_steps/1000:6.1f}k"
                    
                    print(f"{Fore.CYAN}[{time_str}] {Fore.WHITE}Eps: {Fore.YELLOW}{self.global_episodes:4d} {Fore.WHITE}| "
                          f"{Fore.GREEN}{buf_str} {Fore.WHITE}| "
                          f"Mean: {Fore.RED}{mean_sc:6.1f} {Fore.WHITE}| "
                          f"HiScore: {Fore.CYAN}{int(self.max_score)} {Fore.WHITE}| "
                          f"Q: {Fore.MAGENTA}{q_sz:5d} {Fore.WHITE}| "
                          f"{explo_str}{Style.RESET_ALL}")
                    
                    self.mean_scores_history.append(mean_sc)
                    self.sigma_history.append(explo_val)
                    Visualizer.create_fig(self.global_episodes, self.mean_scores_history, 100, 
                                          self.sigma_history, [], [], os.path.join(MEDIA_DIR, "Pacman_fig"), 
                                          self.max_score, label_curve=label_curve)
                    
                    last_log_time = time.time()
                    
                    # Mise à jour du scheduler de Learning Rate
                    trainer.update_learning_rate(mean_sc)
                    
                    if mean_sc > best_mean_score and len(collection_score) >= 20:
                        best_mean_score = mean_sc
                        trainer.save_model(os.path.join(SCRIPT_DIR, "pacman_cnn_best.pth"))

                time.sleep(0.01)

        except KeyboardInterrupt:
            print(f"\n{Fore.RED}🛑 Arrêt intercepté. Sauvegarde...{Style.RESET_ALL}")
            trainer.save_model(MODEL_FILENAME)
            trainer.save_buffer(BUFFER_FILENAME)
            kill_existing_mame()
            
            # --- Enregistrement propre dans le fichier de résultats ---
            mean_sc = np.mean(collection_score) if collection_score else 0
            res_str = (f"[{self.experiment_id}][Pac-Man Multi={NUM_ACTORS}][input={config.input_size}]"
                       f"[fc={config.hidden_size}][gamma={config.gamma}][batch={config.batch_size}][nstep={config.nstep_n}]"
                       f"[lr={config.learning_rate}][target_upd={config.target_update_freq}][cnn={config.cnn_type}]"
                       f"\n=> {self.global_episodes} parties jouées. Score moyen final: {mean_sc:.2f}, Max absolu: {self.max_score:.2f}")
            try:
                with open(RESULTS_FILENAME, "a", encoding="utf-8") as f:
                    f.write(f"\n\n{res_str}\n")
                print(f"\n{Fore.GREEN}✅ Résultats finaux sauvegardés dans resultats_pacman.txt{Style.RESET_ALL}")
            except Exception as e:
                print(f"\n{Fore.RED}❌ Erreur écriture résultats : {e}{Style.RESET_ALL}")
            
            sys.exit(0)

if __name__ == "__main__":
    mp.freeze_support()
    app = PacmanApeX()
    app.run()
