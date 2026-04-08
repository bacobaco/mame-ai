import os, time, subprocess, sys, psutil, socket, queue
import threading
from datetime import datetime
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
from AI_Mame import TrainingConfig, DQNTrainer, GraphWebServer, NStepTransitionWrapper, TeeLogger
from pacman import Memory, GameConstants, PacmanInterface, StateExtractor, Visualizer

# --- CONFIGURATION MULTI ---
NUM_ACTORS = 4
ACTOR_DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
BASE_PORT = 12347 
MAME_PATH = "D:\\Emulateurs\\Mame Officiel"
LUA_BRIDGE_PATH = os.path.join(CORE_DIR, "PythonBridgeSocket.lua").replace('\\', '/')
MODEL_FILENAME = os.path.join(SCRIPT_DIR, "pacman_apex.pth")
BEST_MODEL_FILENAME = os.path.join(SCRIPT_DIR, "pacman_best.pth")
BUFFER_FILENAME = os.path.join(SCRIPT_DIR, "pacman_multi.buffer")
RESULTS_FILENAME = os.path.join(SCRIPT_DIR, "resultats_pacman.txt")

def find_free_port(start_port):
    while True:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(('127.0.0.1', start_port))
                return start_port
            except OSError:
                start_port += 1

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
    
    # Init MAME
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
            score = 0
            comm.communicate(["wait_for 2"]) 
            comm.communicate([
                f"write_memory {Memory.CREDITS}(1)",
                "execute P1_start(1)"
            ])
            
            alive = 0
            while alive == 0:
                comm.communicate(["wait_for 6"])
                data = game.get_score_and_lives()
                if data: _, _, alive, _ = data
                if alive == 0: time.sleep(0.01)
            
            local_state_history = deque(maxlen=_N)
            for _ in range(_N):
                frame, _ = local_extractor()
                local_state_history.append(frame)
            
            current_loop_obs_stack = np.stack(local_state_history, axis=0)
            lives = 3
            done = False
            comm.communicate([f"wait_for {NB_DE_DEMANDES_PAR_STEP}"])

            while not done:
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
                    try: transition_queue.put_nowait(item)
                    except queue.Full: pass

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
        TeeLogger(os.path.join(SCRIPT_DIR, "logs")) # Activation du logging centralisé
        self.web_server = GraphWebServer(graph_dir=os.path.join(SCRIPT_DIR, "logs"), host="0.0.0.0", port=5000, auto_display_latest=True)
        threading.Thread(target=self.web_server.start, daemon=True).start()
        self.global_steps = 0
        self.global_episodes = 0
        self.mean_scores_history = []
        self.sigma_history = []
        self.episodes_history = [] # Nouvel historique pour l'axe X réel
        self.max_score = 0
        self.experiment_id = self.get_next_experiment_id()

    def _archive_result(self, res_str, config):
        """Sauvegarde les résultats en évitant les doublons pour la session actuelle."""
        lines = []
        if os.path.exists(RESULTS_FILENAME):
            try:
                with open(RESULTS_FILENAME, "r", encoding="utf-8") as f:
                    lines = f.readlines()
            except: pass

        # Nettoyage : on enlève les anciennes lignes de la session en cours
        tag = f"[{self.experiment_id}]"
        new_lines = [l for l in lines if not l.strip().startswith(tag)]
        
        # On ajoute la nouvelle ligne
        new_lines.append(f"{res_str}\n")
        
        # Écriture atomique (via fichier temporaire)
        tmp_file = RESULTS_FILENAME + ".tmp"
        try:
            with open(tmp_file, "w", encoding="utf-8") as f:
                f.writelines(new_lines)
            os.replace(tmp_file, RESULTS_FILENAME)
        except Exception as e:
            print(f"Erreur archivage : {e}")

    def get_next_experiment_id(self):
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
            input_size=(4, 64, 56), state_history_size=4,
            hidden_layers=2, hidden_size=1024, output_size=4, # Rainbow Config
            learning_rate=0.0000625, gamma=0.999, # Rainbow Standard
            use_noisy=True, epsilon_start=0.0, epsilon_end=0.0, epsilon_linear=0.0,
            epsilon_decay=0.0, epsilon_add=0.0,
            buffer_capacity=200_000, batch_size=256, min_history_size=30000,
            prioritized_replay=True, target_update_freq=5000,
            double_dqn=True, dueling=True, nstep=True, nstep_n=5,
            model_type="cnn", cnn_type="precise", mode="exploration", optimize_memory=True
        )

        kill_existing_mame()
        trainer = DQNTrainer(config)
        
        # Priorité au chargement
        model_to_load = None
        if os.path.exists(MODEL_FILENAME): model_to_load = MODEL_FILENAME
        elif os.path.exists(BEST_MODEL_FILENAME): model_to_load = BEST_MODEL_FILENAME
        
        if model_to_load:
            try:
                trainer.load_model(model_to_load)
                print(f"{Fore.CYAN}♻️ Reprise de l'entraînement depuis : {model_to_load}{Style.RESET_ALL}")
            except: pass

        # --- CHARGEMENT DU BUFFER ---
        if os.path.exists(BUFFER_FILENAME):
            trainer.load_buffer(BUFFER_FILENAME)


        manager = mp.Manager()
        transition_queue = mp.Queue(maxsize=100000)
        score_queue = mp.Queue()
        shared_weights = manager.dict()
        shared_weights['model'] = {k: v.cpu() for k, v in trainer.dqn.state_dict().items()}

        processes = []
        for i in range(NUM_ACTORS):
            port = find_free_port(BASE_PORT + i)
            p = mp.Process(target=actor_process, args=(i, port, config, transition_queue, score_queue, shared_weights))
            p.daemon = True
            processes.append(p)
            p.start()

        time.sleep(NUM_ACTORS * 4)  
        print(f"\n{Fore.MAGENTA}🧠 [Learner] Démarrage de l'entraînement Ape-X ({ACTOR_DEVICE})...{Style.RESET_ALL}")
        collection_score = deque(maxlen=100)
        best_mean_score = 1500.0
        start_time = time.time()
        last_weight_sync = time.time()
        last_log_time = time.time()
        last_steps = 0
        last_fill_log = -1
        last_results_save = 0 # Tracker pour les 1000 épisodes

        try:
            while True:
                ingested = 0
                while not transition_queue.empty() and ingested < 10000:
                    try:
                        tr = transition_queue.get_nowait()
                        trainer.replay_buffer.push(*tr)
                        self.global_steps += 1
                        ingested += 1
                        
                        if trainer.replay_buffer.size < config.min_history_size:
                            fill_mil = trainer.replay_buffer.size // 2000
                            if fill_mil > last_fill_log:
                                last_fill_log = fill_mil
                                print(f"{Fore.CYAN}📥 Remplissage Buffer: {trainer.replay_buffer.size}/{config.min_history_size}...{Style.RESET_ALL}")
                    except: break
                
                while not score_queue.empty():
                    sc = score_queue.get_nowait()
                    collection_score.append(sc)
                    self.global_episodes += 1
                    if sc > self.max_score: self.max_score = sc

                if len(trainer.replay_buffer) >= config.min_history_size:
                    diff = self.global_steps - last_steps
                    if diff >= 4:
                        iters = min(diff // 4, 64) 
                        for _ in range(iters): trainer.train_step()
                        last_steps = self.global_steps
                    
                    if time.time() - last_weight_sync > 3.0:
                        shared_weights['model'] = {k: v.cpu() for k, v in trainer.dqn.state_dict().items()}
                        last_weight_sync = time.time()

                if time.time() - last_log_time > 5.0:
                    mean_sc = np.mean(collection_score) if collection_score else 0
                    q_sz = transition_queue.qsize() if sys.platform != "darwin" else 0
                    
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
                    self.episodes_history.append(self.global_episodes)
                    
                    Visualizer.create_fig(self.global_episodes, self.mean_scores_history, 100, 
                                          self.sigma_history, [], [], os.path.join(MEDIA_DIR, "Pacman_fig"), 
                                          self.max_score, label_curve=label_curve, x_axis=self.episodes_history)
                    
                    last_log_time = time.time()
                    trainer.update_learning_rate(mean_sc)
                    
                    if mean_sc > (best_mean_score + 10.0) and len(collection_score) >= 50:
                        best_mean_score = mean_sc
                        trainer.save_model(BEST_MODEL_FILENAME)
                        print(f"{Fore.YELLOW}🏆 [RECORD] Nouveau record de moyenne : {best_mean_score:.1f} !{Style.RESET_ALL}")

                    # --- SAUVEGARDE PÉRIODIQUE (TOUS LES 1000 ÉPISODES) ---
                    if self.global_episodes >= last_results_save + 1000:
                        last_results_save = (self.global_episodes // 1000) * 1000
                        now_str = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
                        res_periodic = (f"[{self.experiment_id}][{now_str}] Pac-Man Ape-X | LR={config.learning_rate} | H={config.hidden_size} | "
                                        f"Eps: {self.global_episodes} | Mean: {mean_sc:.2f} | Max: {self.max_score:.2f} | "
                                        f"Exp: {self.global_steps/1e6:.2f}M steps")
                        self._archive_result(res_periodic, config)
                        print(f"{Fore.GREEN}📝 Session {self.experiment_id} mise à jour (Eps {self.global_episodes}){Style.RESET_ALL}")

                time.sleep(0.01)

        except KeyboardInterrupt:
            print(f"\n{Fore.RED}🛑 Arrêt...{Style.RESET_ALL}")
            trainer.save_model(MODEL_FILENAME)
            trainer.save_buffer(BUFFER_FILENAME)
            kill_existing_mame()
            
            mean_sc = np.mean(collection_score) if collection_score else 0
            now_str = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
            res_str = (f"[{self.experiment_id}][{now_str}][FIN SESSION] Pac-Man Ape-X | LR={config.learning_rate} | H={config.hidden_size} | "
                       f"Eps: {self.global_episodes} | Mean: {mean_sc:.2f} | Max: {self.max_score:.2f} | Exp: {self.global_steps/1e6:.2f}M steps")
            self._archive_result(res_str, config)
            sys.exit(0)

if __name__ == "__main__":
    mp.freeze_support()
    app = PacmanApeX()
    app.run()
