import os
import sys
import time
import subprocess
import argparse
from collections import deque
import numpy as np
import torch

ROOT_DIR = r"C:\Users\baco\Documents\Dev\MameAI"
PACMAN_DIR = os.path.join(ROOT_DIR, "pacman")
CORE_DIR = os.path.join(ROOT_DIR, "core")
for d in (CORE_DIR, PACMAN_DIR):
    if d not in sys.path:
        sys.path.insert(0, d)

from MameCommSocket import MameCommunicator
from pacman import Memory, PacmanInterface, StateExtractor
from pacman_30k import (
    PixelGrandmasterCNN,
    MazeSensor11,
    build_subtile_vec,
    resolve_prev_motion,
    find_free_port,
    MAME_BIN_PATH,
    LUA_SCRIPT_PATH,
)

DEFAULT_MODEL = r"C:\Users\baco\Documents\Dev\MameAI\pacman\pacman_30k.pth"


def run_single_episode(model, device, ep_index, wid_equiv, speed):
    port = find_free_port(12346 + ep_index)
    comm = MameCommunicator("127.0.0.1", port, deferred_accept=True, recv_timeout=60)

    # "-sound", "none" est indispensable : "-sound auto" synchronise MAME sur l'audio Windows et fausse le timing
    cmd = [
        MAME_BIN_PATH,
        "-window", "-resolution", "672x864",
        "-skip_gameinfo", "-artwork_crop",
        "-video", "opengl",
        "-sound", "none",
        "-console", "-noautosave", "-nowriteconfig",
        "pacman",
        "-autoboot_delay", "1",
        "-autoboot_script", LUA_SCRIPT_PATH,
    ]
    env = os.environ.copy()
    env["MAME_SOCKET_PORT"] = str(port)
    proc = subprocess.Popen(cmd, cwd=os.path.dirname(MAME_BIN_PATH), env=env)

    score = 0
    clears = 0
    try:
        comm.accept_connection()
        game = PacmanInterface(comm)
        extractor = StateExtractor(game, "cnn")
        sensor = MazeSensor11()
        time.sleep(3.5)

        comm.communicate([
            "wait_for 5",
            f"write_memory {Memory.CREDITS}(1)",
            "execute P1_start(1)",
            "execute throttle_rate(20.0)",
            "execute throttled(0)",
            "frame_per_step 1",
        ])

        alive = 0
        while alive == 0:
            comm.communicate(["wait_for 6"])
            d = game.get_score_and_lives()
            if d:
                _, _, alive, _ = d

        comm.communicate(["wait_for 25"])

        # Reproduction exacte du décalage d'ouverture des workers dans det_eval_worker
        shift_steps = (wid_equiv * 3) % 15
        if wid_equiv > 0 and shift_steps > 0:
            init_act = 1 if (wid_equiv % 2 == 1) else 0
            for _ in range(shift_steps):
                game.execute_action(init_act)

        hist = deque(maxlen=4)
        for _ in range(4):
            f, _ = extractor()
            hist.append(f)
        obs7 = game.build_obs_stack(hist)

        pos_hist = deque(maxlen=12)
        blue_elapsed, steps, lives, score, clears, last_dots = 0, 0, 3, 0, 0, 244
        resp = game.get_all_data_batched()

        frame_delay = (1.0 / (60.0 * speed)) if (speed is not None and speed > 0) else 0.0

        print(f"\n🎮 Début Partie {ep_index + 1} (profil Worker {wid_equiv}, shift={shift_steps}, speed={speed}x)...")

        while steps < 60000:
            t_start = time.perf_counter()
            bd = game.process_all_data(resp) if resp else None
            if not bd:
                resp = game.get_all_data_batched()
                continue

            nf, new_score, new_lives, alive, _ = bd
            vram_list = list(map(int, resp[0].split(",")))
            pos_list = resp[1:15]
            cur_dots = sum(1 for v in vram_list[:1024] if v in (0x10, 0x14))

            if last_dots <= 5 and cur_dots >= 200 and new_lives >= lives:
                clears += 1
                pos_hist.clear()
                blue_elapsed = 0
                print(f"  🏆 NIVEAU {clears} TERMINÉ ! Score={new_score} | Vies={new_lives}")
                wc = 0
                while alive == 0 and wc < 400:
                    comm.communicate(["wait_for 6"])
                    dc = game.get_score_and_lives()
                    if dc:
                        new_score, new_lives, alive, _ = dc
                    wc += 1
                comm.communicate(["wait_for 25"])
                hist.clear()
                for _ in range(4):
                    f, _ = extractor()
                    hist.append(f)
                obs7 = game.build_obs_stack(hist)
                resp = game.get_all_data_batched()
                last_dots, lives, score = 244, new_lives, new_score
                continue

            if new_lives > lives:
                print(f"  ⭐ VIE BONUS ! Score={new_score} | Vies={new_lives}")

            last_dots, score, lives = cur_dots, new_score, new_lives

            if alive == 0:
                if lives == 0:
                    break
                print(f"  💀 Vie perdue (Niveau {clears + 1}, Score={score}, Vies restantes={lives})")
                while alive == 0:
                    comm.communicate(["wait_for 6"])
                    d = game.get_score_and_lives()
                    if d:
                        _, lives, alive, _ = d
                    if lives == 0:
                        break
                if lives == 0:
                    break
                comm.communicate(["wait_for 25"])
                hist.clear()
                for _ in range(4):
                    f, _ = extractor()
                    hist.append(f)
                obs7 = game.build_obs_stack(hist)
                pos_hist.clear()
                blue_elapsed = 0
                resp = game.get_all_data_batched()
                continue

            blue_elapsed = (blue_elapsed + 1) if any(int(pos_list[10 + i]) == 1 for i in range(4)) else 0
            pos_hist.append([int(x) for x in pos_list[:10]])
            prev4 = resolve_prev_motion(pos_hist, base_lag=5)
            obs12 = sensor.encode_11ch(obs7, vram_list, pos_list, prev4_pos=prev4)
            sub_vec = build_subtile_vec(
                pos_list, cur_dots, prev4_pos=prev4, level_num=clears + 1, blue_elapsed=blue_elapsed
            )

            with torch.no_grad():
                logits = model(
                    torch.from_numpy(obs12).float().unsqueeze(0).to(device),
                    torch.from_numpy(sub_vec).float().unsqueeze(0).to(device),
                )[0]
            m_act = int(logits.argmax().item())

            game.execute_action(m_act)
            resp = game.get_all_data_batched()
            bd2 = game.process_all_data(resp) if resp else None
            if bd2:
                hist.append(bd2[0])
                obs7 = game.build_obs_stack(hist)
            steps += 1

            if frame_delay > 0:
                elapsed = time.perf_counter() - t_start
                if elapsed < frame_delay:
                    time.sleep(frame_delay - elapsed)

        print(f"🏁 Fin Partie {ep_index + 1} : Score={score} | Niveaux terminés={clears}")
        return score, clears
    finally:
        comm.close()
        try:
            proc.kill()
        except Exception:
            pass


def watch_game(model_path=DEFAULT_MODEL, episodes=6, speed=0.0, target_score=28000):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = PixelGrandmasterCNN().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=False))
    model.eval()
    print(f"✅ Champion chargé : {model_path} ({device})")

    # Worker 2 testé à froid (ep 1) puis à chaud (ep 2), suivi des profils 4, 3, 5, 1
    candidate_workers = [2, 2, 4, 3, 5, 1]
    for ep in range(episodes):
        wid = candidate_workers[ep % len(candidate_workers)]
        score, clears = run_single_episode(model, device, ep, wid, speed)
        if target_score > 0 and score >= target_score:
            print(f"\n🎯 Objectif atteint ({score} pts >= {target_score}) ! Arrêt de la démo.")
            break


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--episodes", type=int, default=6)
    parser.add_argument("--speed", type=float, default=0.0, help="0 = cadence naturelle, 1.0 = 60 FPS, 2.0 = 120 FPS")
    parser.add_argument("--target", type=int, default=28000, help="Arrête dès qu'une partie atteint ce score")
    args = parser.parse_args()
    watch_game(model_path=args.model, episodes=args.episodes, speed=args.speed, target_score=args.target)