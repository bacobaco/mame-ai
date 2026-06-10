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
    
    trainer = DQNTrainer(config, silent=True)
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
        "frame_per_step 3",
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
            pills_history = deque(maxlen=2)
            lives = 3
            done = False
            last_pills = 0
            step_count = 0
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
                step_count += 1
                
                # --- REWARD SHAPING COMPOSITE ---
                reward = 0.0
                if new_score > score:
                    diff = new_score - score
                    if diff >= 200:   # Fantôme mangé
                        reward = 2.0  # Récompense accrue pour guider la chasse des fantômes bleus
                    else:
                        reward = 1.0  # Manger une pastille normale/super/fruit
                
                # Détection bonus de fin de niveau (Level Clear)
                if pills < last_pills and last_pills > 200 and new_lives == lives:
                    reward = 10.0  # Bonus de fin de niveau conséquent
                
                if new_lives < lives:
                    reward = -1.0 # Mort = -1
                    lives = new_lives
                # ------------------------------------------------
                
                score = new_score
                last_pills = pills
                pills_history.append(pills)
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

            score_queue.put((score, step_count))
            trainer.dqn.reset_noise()

    except Exception as e:
        print(f"Erreur Acteur {actor_id}: {e}")
        process.terminate()

class PacmanApeX:
    """Orchestrateur central (Learner)"""
    def __init__(self):
        self.global_steps = 0
        self.global_episodes = 0
        self.best_mean_recorded = 300.0 # Seuil bas pour commencer à sauvegarder rapidement les premiers progrès
        self.mean_scores_history = []
        self.mean_steps_history = []
        self.sigma_history = []
        self.episodes_history = [] # Nouvel historique pour l'axe X réel
        self.max_score = 0
        self.raw_scores_loaded = []
        self.raw_steps_loaded = []
        
        # Argument parsing
        import argparse
        parser = argparse.ArgumentParser()
        parser.add_argument('--session', type=int, default=None, help='Session ID to continue')
        args, _ = parser.parse_known_args()
        
        if args.session is not None:
            self.experiment_id = args.session
            self.resume_session_stats(args.session)
        else:
            self.experiment_id = self.get_next_experiment_id()

        # Activation du logging centralisé par session (mode append si reprise)
        TeeLogger(os.path.join(SCRIPT_DIR, "logs"), session_id=self.experiment_id)
        
        self.web_server = GraphWebServer(graph_dir=MEDIA_DIR, host="0.0.0.0", port=5000, auto_display_latest=True)
        threading.Thread(target=self.web_server.start, daemon=True).start()

    def save_episode_to_csv(self, episode, score, steps, cumulative_steps, sigma, q_size):
        csv_file = os.path.join(SCRIPT_DIR, "logs", f"session_{self.experiment_id}_scores.csv")
        write_header = not os.path.exists(csv_file)
        try:
            with open(csv_file, "a", encoding="utf-8") as f:
                if write_header:
                    f.write("episode,score,steps,cumulative_steps,sigma,q_size\n")
                f.write(f"{episode},{score},{steps},{cumulative_steps},{sigma:.6f},{q_size}\n")
        except Exception as e:
            print(f"Erreur écriture CSV : {e}")

    def _parse_log_file(self, log_file):
        """Parse le fichier log texte et retourne des listes (episodes, means, mean_steps, sigmas)."""
        import re
        log_episodes, log_means, log_mean_steps, log_sigmas = [], [], [], []
        if not os.path.exists(log_file):
            return log_episodes, log_means, log_mean_steps, log_sigmas
        try:
            ansi_escape = re.compile(r'\x1b\[[0-9;]*[mGKF]')
            with open(log_file, "r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    clean_line = ansi_escape.sub('', line)
                    if "Eps:" not in clean_line or "Mean:" not in clean_line:
                        continue
                    e_m = re.search(r"Eps:\s*(\d+)", clean_line)
                    m_m = re.search(r"Mean:\s*([-+]?\d*\.?\d+)", clean_line)
                    mst_m = re.search(r"MeanSteps:\s*([-+]?\d*\.?\d+)", clean_line)
                    s_m = re.search(r"Sigma:\s*([-+]?\d*\.?\d+)", clean_line)
                    if e_m and m_m:
                        ep = int(e_m.group(1))
                        mean_val = float(m_m.group(1))
                        mean_steps = float(mst_m.group(1)) if mst_m else 500.0
                        sigma_val = float(s_m.group(1)) if s_m else 0.01
                        # Eviter les doublons consecutifs
                        if not log_episodes or log_episodes[-1] != ep:
                            log_episodes.append(ep)
                            log_means.append(mean_val)
                            log_mean_steps.append(mean_steps)
                            log_sigmas.append(sigma_val)
        except Exception as e:
            print(f"{Fore.YELLOW}[WARNING] Erreur lecture log : {e}{Style.RESET_ALL}")
        return log_episodes, log_means, log_mean_steps, log_sigmas

    def _parse_csv_file(self, csv_file):
        """Parse le fichier CSV et retourne des listes (episodes, scores, steps, sigmas)."""
        csv_episodes, csv_scores, csv_steps, csv_sigmas = [], [], [], []
        if not os.path.exists(csv_file):
            return csv_episodes, csv_scores, csv_steps, csv_sigmas
        try:
            with open(csv_file, "r", encoding="utf-8") as f:
                header = f.readline()
                for line in f:
                    line = line.strip()
                    if not line: continue
                    parts = line.split(',')
                    if len(parts) >= 5:
                        try:
                            csv_episodes.append(int(parts[0]))
                            csv_scores.append(float(parts[1]))
                            csv_steps.append(float(parts[2]))
                            csv_sigmas.append(float(parts[4]))
                        except ValueError:
                            continue
        except Exception as e:
            print(f"{Fore.YELLOW}[WARNING] Erreur lecture CSV : {e}{Style.RESET_ALL}")
        return csv_episodes, csv_scores, csv_steps, csv_sigmas

    def resume_session_stats(self, session_id):
        if not os.path.exists(RESULTS_FILENAME):
            print(f"{Fore.YELLOW}Fichier de resultats introuvable ({RESULTS_FILENAME}). Les stats demarreront a zero.{Style.RESET_ALL}")
            return
        try:
            found = False
            with open(RESULTS_FILENAME, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line: continue
                    tag = f"[{session_id}]"
                    if line.startswith(tag):
                        found = True
                        parts = line.split('|')
                        for part in parts:
                            part = part.strip()
                            if part.startswith("Eps:"):
                                val = part.split(':')[1].strip()
                                self.global_episodes = int(val)
                            elif part.startswith("BMean:"):
                                val = part.split(':')[1].strip()
                                self.best_mean_recorded = float(val)
                            elif part.startswith("Max:"):
                                val = part.split(':')[1].strip()
                                self.max_score = float(val)
                            elif part.startswith("Exp:"):
                                val_str = part.split(':')[1].strip().split()[0]
                                if val_str.endswith('M'):
                                    self.global_steps = int(float(val_str[:-1]) * 1_000_000)
                                elif val_str.endswith('k'):
                                    self.global_steps = int(float(val_str[:-1]) * 1_000)
                                else:
                                    self.global_steps = int(float(val_str))
            if found:
                print(f"{Fore.GREEN}[REPRISE] STATS RESTORED for Session {session_id}: Eps={self.global_episodes}, Steps={self.global_steps}, BMean={self.best_mean_recorded}, MaxScore={self.max_score}{Style.RESET_ALL}")
                
                csv_file = os.path.join(SCRIPT_DIR, "logs", f"session_{session_id}_scores.csv")
                log_file = os.path.join(SCRIPT_DIR, "logs", f"training_log_session_{session_id}.txt")
                
                # --- RECONSTITUTION COMPLETE DE L'HISTORIQUE ---
                # 1) Parser le log texte (contient les moyennes glissantes pour TOUS les episodes)
                log_eps, log_means, log_msteps, log_sigmas = self._parse_log_file(log_file)
                # 2) Parser le CSV (contient les scores bruts individuels des episodes recents)
                csv_eps, csv_scores, csv_steps, csv_sigmas = self._parse_csv_file(csv_file)
                
                csv_start_ep = csv_eps[0] if csv_eps else float('inf')
                
                # PARTIE 1 : Historique depuis le LOG (episodes avant le CSV)
                log_count = 0
                for i, ep in enumerate(log_eps):
                    if ep < csv_start_ep:
                        self.episodes_history.append(ep)
                        self.mean_scores_history.append(log_means[i])
                        self.mean_steps_history.append(log_msteps[i])
                        self.sigma_history.append(log_sigmas[i])
                        log_count += 1
                
                # PARTIE 2 : Historique depuis le CSV (episodes avec scores bruts)
                csv_count = 0
                if csv_eps:
                    self.raw_scores_loaded = csv_scores
                    self.raw_steps_loaded = csv_steps
                    for i in range(len(csv_eps)):
                        start_idx = max(0, i - 199)
                        mean_sc = float(np.mean(csv_scores[start_idx:i+1]))
                        mean_st = float(np.mean(csv_steps[start_idx:i+1]))
                        self.episodes_history.append(csv_eps[i])
                        self.mean_scores_history.append(mean_sc)
                        self.mean_steps_history.append(mean_st)
                        self.sigma_history.append(csv_sigmas[i])
                        csv_count += 1
                
                total = log_count + csv_count
                print(f"{Fore.GREEN}[REPRISE] Historique complet restaure : {total} points ({log_count} depuis log + {csv_count} depuis CSV){Style.RESET_ALL}")
                
            else:
                print(f"{Fore.YELLOW}[WARNING] Session {session_id} introuvable dans {RESULTS_FILENAME}. Demarrage de la session {session_id} a zero.{Style.RESET_ALL}")
        except Exception as e:
            print(f"{Fore.RED}[ERREUR] Erreur lors de la recuperation des stats de la session : {e}{Style.RESET_ALL}")

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
            learning_rate=0.0000625, gamma=0.99, # Standard DM / Rainbow
            use_noisy=True, epsilon_start=0.01, epsilon_end=0.01, epsilon_linear=0.0,
            epsilon_decay=0.0, epsilon_add=0.0,
            buffer_capacity=500_000, batch_size=256, min_history_size=30000,
            prioritized_replay=True, target_update_freq=2500,
            double_dqn=True, dueling=True, nstep=True, nstep_n=3,
            model_type="cnn", cnn_type="precise", mode="exploration", optimize_memory=True
        )

        kill_existing_mame()
        trainer = DQNTrainer(config)
        
        # Priorité au chargement
        model_to_load = None
        if os.path.exists(BEST_MODEL_FILENAME): model_to_load = BEST_MODEL_FILENAME
        elif os.path.exists(MODEL_FILENAME): model_to_load = MODEL_FILENAME
        
        if model_to_load:
            try:
                trainer.load_model(model_to_load)
                print(f"{Fore.CYAN}[REPRISE] Reprise de l'entraînement depuis : {model_to_load}{Style.RESET_ALL}")
            except: pass

        # --- CHARGEMENT DU BUFFER ---
        if os.path.exists(BUFFER_FILENAME):
            trainer.load_buffer(BUFFER_FILENAME)


        manager = mp.Manager()
        transition_queue = mp.Queue(maxsize=30000) # Réduit de 100k à 30k pour éviter d'étouffer Windows
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
        print(f"\n{Fore.MAGENTA}[Learner] Démarrage de l'entraînement Ape-X ({ACTOR_DEVICE})...{Style.RESET_ALL}")
        collection_score = deque(maxlen=200) # Augmenté à 200 pour matcher la condition de record
        collection_steps = deque(maxlen=200)
        last_record_episode = self.global_episodes 
        start_time = time.time()
        last_weight_sync = time.time()
        last_log_time = time.time()
        last_steps = self.global_steps
        last_fill_log = -1
        last_results_save = (self.global_episodes // 1000) * 1000 # Tracker pour les 1000 épisodes

        if self.global_episodes > 0:
            if not self.mean_scores_history:
                self.mean_scores_history.append(self.best_mean_recorded)
                self.mean_steps_history.append(500.0)
                self.sigma_history.append(0.01)
                self.episodes_history.append(self.global_episodes)
            
            # Remplir collection_score/steps pour stabiliser la moyenne glissante
            if self.raw_scores_loaded:
                for sc in self.raw_scores_loaded[-200:]:
                    collection_score.append(sc)
                for st in self.raw_steps_loaded[-200:]:
                    collection_steps.append(st)
            else:
                last_mean = self.mean_scores_history[-1] if self.mean_scores_history else self.best_mean_recorded
                last_step = self.mean_steps_history[-1] if self.mean_steps_history else 500.0
                for _ in range(200):
                    collection_score.append(last_mean)
                    collection_steps.append(last_step)


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
                                print(f"{Fore.CYAN}[INFO] Remplissage Buffer: {trainer.replay_buffer.size}/{config.min_history_size}...{Style.RESET_ALL}")
                    except: break
                
                while not score_queue.empty():
                    try:
                        sc, st = score_queue.get_nowait()
                        collection_score.append(sc)
                        collection_steps.append(st)
                        self.global_episodes += 1
                        if sc > self.max_score: self.max_score = sc
                        
                        # Récupérer la valeur de bruit actuelle
                        if config.use_noisy:
                            sigmas = trainer.dqn.get_sigma_values()
                            explo_val = sum(sigmas.values()) / len(sigmas) if sigmas else 0.0
                        else:
                            explo_val = trainer.epsilon
                            
                        # Sauvegarder dans le CSV haute résolution
                        q_sz = transition_queue.qsize() if sys.platform != "darwin" else 0
                        self.save_episode_to_csv(self.global_episodes, sc, st, self.global_steps, explo_val, q_sz)
                        
                        # Mettre à jour l'historique en mémoire épisode par épisode
                        mean_sc = np.mean(collection_score) if collection_score else 0.0
                        mean_st = np.mean(collection_steps) if collection_steps else 0.0
                        
                        self.mean_scores_history.append(mean_sc)
                        self.mean_steps_history.append(mean_st)
                        self.sigma_history.append(explo_val)
                        self.episodes_history.append(self.global_episodes)
                    except queue.Empty:
                        break
                    except Exception as e:
                        print(f"Erreur traitement score : {e}")

                if len(trainer.replay_buffer) >= config.min_history_size:
                    diff = self.global_steps - last_steps
                    if diff >= 4:
                        # On remet le plafond de sécurité pour éviter de bloquer le script
                        iters = min(diff // 4, 128) 
                        for _ in range(iters): trainer.train_step()
                        last_steps = self.global_steps
                    
                    if time.time() - last_weight_sync > 3.0:
                        shared_weights['model'] = {k: v.cpu() for k, v in trainer.dqn.state_dict().items()}
                        last_weight_sync = time.time()

                if time.time() - last_log_time > 5.0:
                    mean_sc = np.mean(collection_score) if collection_score else 0
                    mean_st = np.mean(collection_steps) if collection_steps else 0
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
                          f"MeanSteps: {Fore.MAGENTA}{mean_st:5.1f} {Fore.WHITE}| "
                          f"HiScore: {Fore.CYAN}{int(self.max_score)} {Fore.WHITE}| "
                          f"Q: {Fore.MAGENTA}{q_sz:5d} {Fore.WHITE}| "
                          f"{explo_str}{Style.RESET_ALL}")
                    
                    # L'historique est désormais mis à jour en temps réel à chaque épisode dans la boucle score_queue
                    
                    session_info = (
                        f"SESSION {self.experiment_id} | Ape-X Multi-Actor (4 Actors) | "
                        f"CNN: {config.cnn_type.upper()} | LR: {config.learning_rate} | "
                        f"Gamma: {config.gamma} | Buffer: {config.buffer_capacity//1000}k"
                    )
                    
                    Visualizer.create_fig(self.global_episodes, self.mean_scores_history, 100, 
                                          self.sigma_history, [], self.mean_steps_history, os.path.join(MEDIA_DIR, f"Pacman_fig_v{self.experiment_id}"), 
                                          self.max_score, label_curve=label_curve, x_axis=self.episodes_history, session_info=session_info)
                    
                    last_log_time = time.time()
                    trainer.update_learning_rate(mean_sc)
                    
                    if mean_sc > (self.best_mean_recorded + 10.0) and len(collection_score) >= 200:
                        self.best_mean_recorded = mean_sc
                        last_record_episode = self.global_episodes
                        trainer.save_model(BEST_MODEL_FILENAME)
                        print(f"{Fore.YELLOW}[RECORD] Nouveau record de moyenne : {self.best_mean_recorded:.1f} !{Style.RESET_ALL}")

                    # --- PLATEAU RESET (RELANCER L'EXPLORATION SANS DÉTRUIRE LES SIGMAS) ---
                    if self.global_episodes - last_record_episode > 1500:
                        print(f"\n{Fore.YELLOW}[PLATEAU] Plateau détecté @ Ep {self.global_episodes}. Reset noise (sans toucher sigma).{Style.RESET_ALL}")
                        trainer.dqn.reset_noise()  # Re-sample epsilon noise, préserve les sigma appris
                        last_record_episode = self.global_episodes

                    # --- SAUVEGARDE PÉRIODIQUE (TOUS LES 1000 ÉPISODES) ---
                    if self.global_episodes >= last_results_save + 1000:
                        last_results_save = (self.global_episodes // 1000) * 1000
                        now_str = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
                        res_periodic = (f"[{self.experiment_id}][{now_str}] Pac-Man Ape-X | "
                                        f"LR={config.learning_rate} | H={config.hidden_size} | Gamma={config.gamma} | CNN={config.cnn_type} | "
                                        f"Eps: {self.global_episodes} | Mean: {mean_sc:.2f} | BMean: {self.best_mean_recorded:.2f} | Max: {self.max_score:.2f} | "
                                        f"Exp: {self.global_steps/1e6:.2f}M steps")
                        self._archive_result(res_periodic, config)
                        trainer.save_model(MODEL_FILENAME)
                        trainer.save_buffer(BUFFER_FILENAME)
                        print(f"{Fore.GREEN}[SAUVEGARDE] Session {self.experiment_id} mise à jour & sauvegardée (Eps {self.global_episodes}){Style.RESET_ALL}")

                # --- DETECTION DE L'ARRET GRACIEUX ---
                if os.path.exists("stop.flag") or os.path.exists(os.path.join(SCRIPT_DIR, "stop.flag")):
                    print(f"\n{Fore.RED}[ARRET DETECTE] Fichier stop.flag trouvé. Sauvegarde et arrêt gracieux...{Style.RESET_ALL}")
                    try:
                        if os.path.exists("stop.flag"): os.remove("stop.flag")
                        if os.path.exists(os.path.join(SCRIPT_DIR, "stop.flag")): os.remove(os.path.join(SCRIPT_DIR, "stop.flag"))
                    except: pass
                    
                    trainer.save_model(MODEL_FILENAME)
                    trainer.save_buffer(BUFFER_FILENAME)
                    kill_existing_mame()
                    
                    mean_sc = np.mean(collection_score) if collection_score else 0
                    now_str = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
                    res_str = (f"[{self.experiment_id}][{now_str}][FIN SESSION] Pac-Man Ape-X | "
                               f"LR={config.learning_rate} | H={config.hidden_size} | Gamma={config.gamma} | CNN={config.cnn_type} | "
                               f"Eps: {self.global_episodes} | Mean: {mean_sc:.2f} | BMean: {self.best_mean_recorded:.2f} | Max: {self.max_score:.2f} | Exp: {self.global_steps/1e6:.2f}M steps")
                    self._archive_result(res_str, config)
                    print(f"{Fore.GREEN}[OK] Session {self.experiment_id} arrêtée proprement.{Style.RESET_ALL}")
                    sys.exit(0)

                time.sleep(0.01)

        except KeyboardInterrupt:
            print(f"\n{Fore.RED}[ARRET] Arrêt...{Style.RESET_ALL}")
            trainer.save_model(MODEL_FILENAME)
            trainer.save_buffer(BUFFER_FILENAME)
            kill_existing_mame()
            
            mean_sc = np.mean(collection_score) if collection_score else 0
            now_str = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
            res_str = (f"[{self.experiment_id}][{now_str}][FIN SESSION] Pac-Man Ape-X | "
                       f"LR={config.learning_rate} | H={config.hidden_size} | Gamma={config.gamma} | CNN={config.cnn_type} | "
                       f"Eps: {self.global_episodes} | Mean: {mean_sc:.2f} | BMean: {self.best_mean_recorded:.2f} | Max: {self.max_score:.2f} | Exp: {self.global_steps/1e6:.2f}M steps")
            self._archive_result(res_str, config)
            sys.exit(0)

if __name__ == "__main__":
    mp.freeze_support()
    app = PacmanApeX()
    app.run()
