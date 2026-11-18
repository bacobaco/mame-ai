"""
pacman_30k.py
Pixel-Aware Grandmaster CNN player & visualizer for Pac-Man (MAME 60Hz).
Loads a trained .pth save file (default: pacman/pacman_30k.pth) and plays live
with MAME + Pygame Radar & AI Multi-Channel Vision monitor.
"""

import os
import sys
import time
import socket
import subprocess
import argparse
from collections import deque
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pygame
from colorama import Fore, Style

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPT_DIR)
CORE_DIR = os.path.join(ROOT_DIR, "core")
if CORE_DIR not in sys.path:
    sys.path.append(CORE_DIR)
if SCRIPT_DIR not in sys.path:
    sys.path.append(SCRIPT_DIR)

from MameCommSocket import MameCommunicator
from pacman import Memory, PacmanInterface, StateExtractor

MAME_BIN_PATH = r"D:\Emulateurs\Mame Officiel\mame.exe"
LUA_SCRIPT_PATH = os.path.join(CORE_DIR, "PythonBridgeSocket.lua").replace("\\", "/")
DEFAULT_MODEL_PATH = os.path.join(SCRIPT_DIR, "pacman_30k.pth")


class MazeSensor11:
    """
    Pure spatial sensor encoder: converts raw MAME VRAM + sprite RAM into 4 wall-aware
    BFS distance fields (danger ghosts, remaining dots, power pellets, edible blue ghosts)
    stacked onto the 7-channel CNN frame -> (11, 32, 28) tensor. Zero action logic.
    """
    DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    def __init__(self):
        self.walkable = None
        self.neighbors = {}

    def init_maze(self, vram_list):
        walk = np.zeros((32, 28), dtype=bool)
        for idx, val in enumerate(vram_list):
            if idx >= 1024:
                break
            r = idx % 32
            c = 27 - ((idx // 32) - 2)
            if 0 <= r < 32 and 0 <= c < 28 and val in (0x10, 0x14, 0x40):
                walk[r, c] = True
        walk[15, :] = True
        walk[13:18, 10:18] = False
        walk[12:19, 9] = True
        walk[12:19, 18] = True
        walk[12, 9:19] = True
        walk[18, 9:19] = True
        self.walkable = walk
        self.neighbors = {}
        for r in range(32):
            for c in range(28):
                if not walk[r, c]:
                    continue
                nbrs = []
                for dr, dc in self.DIRS:
                    nr, nc = r + dr, (c + dc) % 28
                    if 0 <= nr < 32 and walk[nr, nc]:
                        nbrs.append((nr, nc))
                self.neighbors[(r, c)] = nbrs

    def _snap(self, r: int, c: int):
        if self.walkable[r, c]:
            return (r, c)
        if 13 <= r <= 16 and 11 <= c <= 16:
            return (12, 13 if c <= 13 else 14)
        best, bd = (12, 13), 999
        for rr, cc in self.neighbors:
            d = abs(rr - r) + min(abs(cc - c), 28 - abs(cc - c))
            if d < bd:
                bd, best = d, (rr, cc)
        return best

    def _bfs_field(self, sources, max_dist: int) -> np.ndarray:
        field = np.zeros((32, 28), dtype=np.float32)
        if not sources:
            return field
        dist = np.full((32, 28), 99, dtype=np.int16)
        q = deque()
        for sr, sc in sources:
            sr, sc = self._snap(sr, sc)
            if dist[sr, sc] > 0:
                dist[sr, sc] = 0
                field[sr, sc] = 1.0
                q.append((sr, sc))
        while q:
            r, c = q.popleft()
            nd = int(dist[r, c]) + 1
            if nd >= max_dist:
                continue
            val = 1.0 / (1.0 + 0.22 * float(nd))
            for nr, nc in self.neighbors[(r, c)]:
                if dist[nr, nc] > nd:
                    dist[nr, nc] = nd
                    field[nr, nc] = val
                    q.append((nr, nc))
        return field

    def _directed_danger_field(self, pos_list, prev4_pos, max_dist: int = 16) -> np.ndarray:
        field = np.zeros((32, 28), dtype=np.float32)
        best = np.full((32, 28), 99, dtype=np.int16)
        q = deque()
        seen_state = {}
        for i in range(4):
            gs = int(pos_list[10 + i])
            if gs not in (0, 2):
                continue
            gx, gy = int(pos_list[2 + i * 2]), int(pos_list[3 + i * 2])
            gr = min(31, max(0, gx // 8))
            gc = (27 - (gy // 8) + 2) % 28
            in_house = (13 <= gr <= 16 and 11 <= gc <= 16)
            sr, sc = (12, 13 if gc <= 13 else 14) if in_house else self._snap(gr, gc)
            gdr, gdc = 0, 0
            if not in_house and prev4_pos is not None:
                gvx = gx - int(prev4_pos[2 + i * 2])
                gvy = gy - int(prev4_pos[3 + i * 2])
                if 1 <= abs(gvx) + abs(gvy) <= 16:
                    if abs(gvx) >= abs(gvy):
                        gdr, gdc = (1 if gvx > 0 else -1), 0
                    else:
                        gdr, gdc = 0, (-1 if gvy > 0 else 1)
            best[sr, sc] = 0
            field[sr, sc] = 1.0
            key = (sr, sc, gdr, gdc)
            if seen_state.get(key, 99) > 0:
                seen_state[key] = 0
                q.append((sr, sc, gdr, gdc, 0))

        inv = 1.0 / float(max_dist)
        while q:
            r, c, dr, dc, d = q.popleft()
            nd = d + 1
            if nd >= max_dist:
                continue
            nbrs = self.neighbors[(r, c)]
            can_rev = (len(nbrs) <= 1) or (dr == 0 and dc == 0)
            for nr, nc in nbrs:
                sdr = nr - r
                sdc = nc - c
                if sdc > 1:
                    sdc = -1
                elif sdc < -1:
                    sdc = 1
                if not can_rev and (sdr, sdc) == (-dr, -dc):
                    continue
                if nd < best[nr, nc]:
                    best[nr, nc] = nd
                    field[nr, nc] = max(0.0, 1.0 - nd * inv)
                k2 = (nr, nc, sdr, sdc)
                if seen_state.get(k2, 99) > nd:
                    seen_state[k2] = nd
                    q.append((nr, nc, sdr, sdc, nd))
        return field, best

    def _safe_reach_field(self, pos_list, ghost_best: np.ndarray, max_dist: int = 18) -> np.ndarray:
        """Pure spatial reachability field: highlights tiles Pac-Man can safely reach strictly ahead of dangerous ghosts."""
        field = np.zeros((32, 28), dtype=np.float32)
        px, py = int(pos_list[0]), int(pos_list[1])
        pr = min(31, max(0, px // 8))
        pc = (27 - (py // 8) + 2) % 28
        sr, sc = self._snap(pr, pc)
        dist = np.full((32, 28), 99, dtype=np.int16)
        dist[sr, sc] = 0
        field[sr, sc] = 1.0
        q = deque([(sr, sc)])
        inv = 1.0 / float(max_dist)
        while q:
            r, c = q.popleft()
            nd = int(dist[r, c]) + 1
            if nd >= max_dist:
                continue
            for nr, nc in self.neighbors[(r, c)]:
                lead = int(ghost_best[nr, nc]) - nd
                if dist[nr, nc] > nd and lead >= 3:
                    dist[nr, nc] = nd
                    reach_score = max(0.15, 1.0 - nd * inv)
                    margin_scale = min(1.0, (lead - 2) / 5.0)
                    field[nr, nc] = reach_score * (0.35 + 0.65 * margin_scale)
                    q.append((nr, nc))
        return field

    def encode_11ch(self, obs7: np.ndarray, vram_list, pos_list, prev4_pos=None) -> np.ndarray:
        if self.walkable is None:
            self.init_maze(vram_list)
        dot_srcs = []
        pellet_srcs = []
        for idx, val in enumerate(vram_list):
            if idx >= 1024:
                break
            if val == 0x10 or val == 0x14:
                r = idx % 32
                c = 27 - ((idx // 32) - 2)
                if 0 <= r < 32 and 0 <= c < 28:
                    if val == 0x10:
                        dot_srcs.append((r, c))
                    else:
                        pellet_srcs.append((r, c))
        blue_srcs = []
        for i in range(4):
            gx, gy = int(pos_list[2 + i * 2]), int(pos_list[3 + i * 2])
            gs = int(pos_list[10 + i])
            gr = min(31, max(0, gx // 8))
            gc = (27 - (gy // 8) + 2) % 28
            if gs == 1:
                blue_srcs.append((gr, gc))

        danger_field, ghost_best = self._directed_danger_field(pos_list, prev4_pos, 16)
        safe_field = self._safe_reach_field(pos_list, ghost_best, 22)
        extra = np.stack([
            danger_field,
            self._bfs_field(dot_srcs, 64),
            self._bfs_field(pellet_srcs, 48),
            self._bfs_field(blue_srcs, 32),
            safe_field,
        ], axis=0)
        return np.concatenate([obs7, extra], axis=0).astype(np.float32)


class DilatedResBlock(nn.Module):
    def __init__(self, channels: int, dilation: int = 1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=dilation, dilation=dilation, bias=False),
            nn.GroupNorm(8, channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, dilation=1, bias=False),
            nn.GroupNorm(8, channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.gelu(x + self.block(x))


class PixelGrandmasterCNN(nn.Module):
    """
    12-Channel Translation-Invariant Center-Aligned Egocentric ResNet + Directed Ghost & Safe-Reach Sensors.
    - Full 32x28 dilated spatial trunk (C=48, d=1,2,4,1) preserves exact 1-tile corridors.
    - Keeps exact (5,5) center alignment on the 11x11 egocentric patch (3x3 immediate + 5x5 symmetric cross).
    - Uses purely relative/wrapped ghost vectors, sub-tile alignment, velocities, and regularized heads.
    """
    def __init__(self, num_actions: int = 4, in_channels: int = 12):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 48, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8, 48),
            nn.GELU(),
        )
        self.res_tower = nn.Sequential(
            DilatedResBlock(48, dilation=1),
            DilatedResBlock(48, dilation=2),
            DilatedResBlock(48, dilation=4),
            DilatedResBlock(48, dilation=2),
        )
        self.ego_conv = nn.Sequential(
            nn.Conv2d(48, 64, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8, 64),
            nn.GELU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8, 64),
            nn.GELU(),
        )
        self.ego_cross_proj = nn.Conv2d(64, 12, kernel_size=1)
        self.ego_fc = nn.Sequential(
            nn.Linear(64 * 9 + 12 * 25 + 12 * 20, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.15),
        )
        self.global_conv = nn.Sequential(
            nn.Conv2d(48, 16, kernel_size=1),
            nn.GELU(),
        )
        self.global_fc = nn.Sequential(
            nn.Linear(16 * 4 * 4 * 2, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(0.20),
        )
        # 4 rel_gx + 4 rel_gy + 4 rel_dist + 19 invariant floats = 31 floats
        self.pix_mlp = nn.Sequential(
            nn.Linear(31, 192),
            nn.LayerNorm(192),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(192, 192),
            nn.LayerNorm(192),
            nn.GELU(),
        )
        self.head = nn.Sequential(
            nn.Linear(256 + 128 + 192, 384),
            nn.LayerNorm(384),
            nn.GELU(),
            nn.Dropout(0.20),
            nn.Linear(384, 192),
            nn.LayerNorm(192),
            nn.GELU(),
            nn.Dropout(0.10),
            nn.Linear(192, num_actions),
        )

    def _extract_ego_11x11(self, h: torch.Tensor, sub_vec: torch.Tensor) -> torch.Tensor:
        bsz = h.size(0)
        h_wrap = torch.cat([h[:, :, :, -5:], h, h[:, :, :, :5]], dim=3)
        h_pad = F.pad(h_wrap, (0, 0, 5, 5), mode="constant", value=0.0)

        px = torch.round(sub_vec[:, 0] * 255.0).long()
        py = torch.round(sub_vec[:, 1] * 255.0).long()
        pr = torch.clamp(px // 8, 0, 31)
        pc = (27 - (py // 8) + 2) % 28

        dr = torch.arange(11, device=h.device).view(1, 11, 1)
        dc = torch.arange(11, device=h.device).view(1, 1, 11)
        r_idx = (pr.view(bsz, 1, 1) + dr).clamp(0, 41)
        c_idx = (pc.view(bsz, 1, 1) + dc).clamp(0, 37)
        b_idx = torch.arange(bsz, device=h.device).view(bsz, 1, 1)

        patches = h_pad.permute(0, 2, 3, 1)[b_idx, r_idx, c_idx]
        return patches.permute(0, 3, 1, 2).contiguous()

    def forward(self, obs_grid: torch.Tensor, sub_vec: torch.Tensor) -> torch.Tensor:
        bsz = obs_grid.size(0)
        h = self.res_tower(self.stem(obs_grid))
        ego = self.ego_conv(self._extract_ego_11x11(h, sub_vec))
        center_3x3 = ego[:, :, 4:7, 4:7].reshape(bsz, -1)
        ego_proj = self.ego_cross_proj(ego)
        cross_5x5 = ego_proj[:, :, 1:10:2, 1:10:2].reshape(bsz, -1)
        rays = torch.cat([
            ego_proj[:, :, 5, :5].reshape(bsz, -1),
            ego_proj[:, :, 5, 6:].reshape(bsz, -1),
            ego_proj[:, :, :5, 5].reshape(bsz, -1),
            ego_proj[:, :, 6:, 5].reshape(bsz, -1),
        ], dim=1)
        feat_ego = self.ego_fc(torch.cat([center_3x3, cross_5x5, rays], dim=1))

        g_map = self.global_conv(h)
        g_avg = F.adaptive_avg_pool2d(g_map, (4, 4)).reshape(bsz, -1)
        g_max = F.adaptive_max_pool2d(g_map, (4, 4)).reshape(bsz, -1)
        feat_glob = self.global_fc(torch.cat([g_avg, g_max], dim=1))

        rel_gx = sub_vec[:, 2:10:2] - sub_vec[:, 0:1]
        raw_dy = sub_vec[:, 3:10:2] - sub_vec[:, 1:2]
        wrap_y = 224.0 / 255.0
        rel_gy = torch.remainder(raw_dy + 0.5 * wrap_y, wrap_y) - 0.5 * wrap_y
        rel_dist = (rel_gx.abs() + rel_gy.abs()).clamp(0.0, 2.0)
        pix_in = torch.cat([rel_gx, rel_gy, rel_dist, sub_vec[:, 10:29]], dim=1)
        feat_pix = self.pix_mlp(pix_in)

        return self.head(torch.cat([feat_ego, feat_glob, feat_pix], dim=1))


def resolve_prev_motion(pos_hist, base_lag: int = 5):
    """Find most recent non-zero displacement anchor per sprite so dot-eat pauses retain direction."""
    if len(pos_hist) < 2:
        return None
    cur = pos_hist[-1]
    base_idx = max(0, len(pos_hist) - base_lag)
    eff = list(pos_hist[base_idx])
    for j in range(5):
        ix, iy = 2 * j, 2 * j + 1
        d0 = abs(cur[ix] - eff[ix]) + abs(cur[iy] - eff[iy])
        if d0 == 0 or d0 > 16:
            for k in range(len(pos_hist) - 2, -1, -1):
                cand = pos_hist[k]
                dk = abs(cur[ix] - cand[ix]) + abs(cur[iy] - cand[iy])
                if 1 <= dk <= 16:
                    eff[ix], eff[iy] = cand[ix], cand[iy]
                    break
    return eff


def build_subtile_vec(pos_list, cur_dots: int, prev4_pos=None, level_num: int = 1, blue_elapsed: int = 0) -> np.ndarray:
    cur_coords_int = np.array(list(map(int, pos_list[:10])), dtype=np.float32)
    px, py = int(cur_coords_int[0]), int(cur_coords_int[1])
    coords = cur_coords_int / 255.0
    subtile = np.array([(px % 8) / 7.0, (py % 8) / 7.0], dtype=np.float32)
    gstates = np.array([int(x) / 4.0 for x in pos_list[10:14]], dtype=np.float32)
    dots_norm = np.array([cur_dots / 244.0], dtype=np.float32)
    if prev4_pos is not None:
        diff = cur_coords_int - np.asarray(prev4_pos[:10], dtype=np.float32)
        diff = np.where(np.abs(diff) > 16.0, 0.0, diff)
        vel = np.clip(diff / 6.0, -1.0, 1.0).astype(np.float32)
    else:
        vel = np.zeros(10, dtype=np.float32)
    meta = np.array([min(level_num, 12) / 12.0, min(blue_elapsed, 360) / 360.0], dtype=np.float32)
    return np.concatenate([coords, subtile, gstates, dots_norm, vel, meta])


def find_free_port(start: int = 12346) -> int:
    while True:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("127.0.0.1", start))
                return start
            except OSError:
                start += 1


def draw_monitor(screen, responses, obs_frame, score, level, lives, clears):
    if not responses or len(responses) < 15:
        return
    screen.fill((15, 15, 22))
    # Left: logical radar
    try:
        vram = list(map(int, responses[0].split(",")))
        for idx, val in enumerate(vram):
            if idx >= 1024:
                break
            r = idx % 32
            c = 27 - ((idx // 32) - 2)
            if 0 <= r < 32 and 0 <= c < 28:
                if val in (64, 16, 20, 0xBF):
                    if val == 16:
                        pygame.draw.circle(screen, (255, 230, 100), (c * 10 + 5, r * 10 + 5), 2)
                    elif val == 20:
                        pygame.draw.circle(screen, (255, 100, 255), (c * 10 + 5, r * 10 + 5), 4)
                    else:
                        pygame.draw.rect(screen, (10, 10, 20), (c * 10, r * 10, 9, 9))
                else:
                    pygame.draw.rect(screen, (0, 35, 130), (c * 10, r * 10, 9, 9))

        px, py = int(responses[1]), int(responses[2])
        pr, pc = min(31, max(0, px // 8)), (27 - (py // 8) + 2) % 28
        pygame.draw.circle(screen, (255, 255, 0), (pc * 10 + 5, pr * 10 + 5), 5)
        for i in range(4):
            gx, gy = int(responses[3 + i * 2]), int(responses[4 + i * 2])
            st = int(responses[11 + i])
            gr, gc = min(31, max(0, gx // 8)), (27 - (gy // 8) + 2) % 28
            col = (255, 40, 40) if st in (0, 2) else ((0, 220, 255) if st == 1 else (160, 160, 160))
            pygame.draw.rect(screen, col, (gc * 10 + 1, gr * 10 + 1, 8, 8))
    except Exception:
        pass

    # Right: AI multi-channel vision
    sx = 280
    if obs_frame is not None and obs_frame.ndim == 3 and obs_frame.shape[0] >= 4:
        for r in range(32):
            for c in range(28):
                ch0, ch1, ch2, ch3 = obs_frame[0, r, c], obs_frame[1, r, c], obs_frame[2, r, c], obs_frame[3, r, c]
                if ch3 > 0.0:
                    col = (0, 220, 255) if ch3 > 0.7 else (255, 165, 0)
                elif ch1 >= 0.9:
                    col = (255, 255, 0)
                elif ch2 > 0.0:
                    col = (255, 40, 40)
                elif ch0 >= 0.9:
                    col = (255, 120, 255)
                elif ch0 >= 0.4:
                    col = (180, 180, 120)
                elif ch0 > 0.1:
                    col = (30, 40, 90)
                else:
                    col = (12, 12, 18)
                pygame.draw.rect(screen, col, (sx + c * 10, r * 10, 9, 9))

    pygame.draw.line(screen, (100, 100, 120), (280, 0), (280, 320), 2)
    pygame.display.set_caption(f"Pac-Man 30K AI | Score: {score} | Level: {level} (Clears: {clears}) | Lives: {lives}")
    pygame.display.flip()


def play(model_path=DEFAULT_MODEL_PATH, speed=1.5, video_driver="opengl", headless=False, episodes=1):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = PixelGrandmasterCNN().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=False))
    model.eval()
    sensor = MazeSensor11()
    print(f"{Fore.GREEN}✅ Modèle 30K chargé depuis {model_path} ({device}){Style.RESET_ALL}", flush=True)

    screen = None
    if not headless:
        pygame.init()
        screen = pygame.display.set_mode((560, 325))
        pygame.display.set_caption("Pac-Man 30K AI - Initialisation...")

    port = find_free_port(12346 if not headless else 12750)
    comm = MameCommunicator("127.0.0.1", port, deferred_accept=True, recv_timeout=60)

    cmd = [
        MAME_BIN_PATH,
        "-window", "-resolution", "448x576",
        "-skip_gameinfo", "-artwork_crop",
        "-video", "none" if headless else video_driver,
        "-sound", "none",
        "-console", "-noautosave", "-nowriteconfig",
        "pacman",
        "-autoboot_delay", "1",
        "-autoboot_script", LUA_SCRIPT_PATH,
    ]
    if headless or speed is None or speed <= 0:
        cmd.append("-nothrottle")

    env = os.environ.copy()
    env["MAME_SOCKET_PORT"] = str(port)
    si = subprocess.STARTUPINFO()
    si.dwFlags |= subprocess.STARTF_USESHOWWINDOW
    si.wShowWindow = 7 if headless else 1
    proc = subprocess.Popen(cmd, cwd=os.path.dirname(MAME_BIN_PATH), startupinfo=si, env=env)

    try:
        comm.accept_connection()
        game = PacmanInterface(comm)
        extractor = StateExtractor(game, "cnn")
        time.sleep(4)
        comm.communicate(["wait_for 5"])
        if headless or speed is None or speed <= 0:
            comm.communicate([
                f"write_memory {Memory.CREDITS}(1)",
                "execute P1_start(1)",
                "execute throttle_rate(20.0)",
                "execute throttled(0)",
                "frame_per_step 1",
            ])
        else:
            comm.communicate([
                f"write_memory {Memory.CREDITS}(1)",
                "execute P1_start(1)",
                f"execute throttle_rate({float(speed)})",
                "execute throttled(1)",
                "frame_per_step 1",
            ])

        scores = []
        clears_all = []
        for ep in range(episodes):
            comm.communicate(["wait_for 2"])
            comm.communicate([f"write_memory {Memory.CREDITS}(1)", "execute P1_start(1)"])
            alive = 0
            while alive == 0:
                if screen:
                    for ev in pygame.event.get():
                        if ev.type == pygame.QUIT:
                            return scores, clears_all
                comm.communicate(["wait_for 6"])
                data = game.get_score_and_lives()
                if data:
                    _, _, alive, _ = data
                if alive == 0:
                    time.sleep(0.01)

            comm.communicate(["wait_for 25"])
            hist = deque(maxlen=4)
            for _ in range(4):
                f, _ = extractor()
                hist.append(f)
            obs = game.build_obs_stack(hist)

            lives = 3
            score = 0
            clears = 0
            steps = 0
            last_dots = 244
            blue_elapsed = 0
            pos_hist = deque(maxlen=12)
            done = False
            resp = game.get_all_data_batched()

            while not done and steps < 100000:
                if screen:
                    for ev in pygame.event.get():
                        if ev.type == pygame.QUIT:
                            return scores, clears_all

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
                    print(f"  🏆 [Ep {ep+1}] NIVEAU {clears} TERMINÉ ! Score={new_score} | Steps={steps}", flush=True)
                last_dots = cur_dots

                if new_lives < lives:
                    print(f"  💥 Vie perdue (Niveau {clears+1}, Score={new_score}, Pastilles restantes={cur_dots})", flush=True)
                elif new_lives > lives:
                    print(f"  ⭐ Vie bonus obtenue ! (Score={new_score}, Vies={new_lives})", flush=True)
                lives = new_lives
                score = new_score

                if alive == 0 and lives == 0:
                    done = True
                    break
                if alive == 0:
                    print(f"  💀 Vie perdue en cours (Niveau {clears+1}, Score={score}, Pastilles={cur_dots})", flush=True)
                    while alive == 0:
                        if screen:
                            for ev in pygame.event.get():
                                if ev.type == pygame.QUIT:
                                    return scores, clears_all
                        comm.communicate(["wait_for 6"])
                        data = game.get_score_and_lives()
                        if data:
                            _, lives, alive, _ = data
                        if lives == 0:
                            done = True
                            break
                    if not done:
                        comm.communicate(["wait_for 25"])
                        hist.clear()
                        pos_hist.clear()
                        blue_elapsed = 0
                        for _ in range(4):
                            f, _ = extractor()
                            hist.append(f)
                        obs = game.build_obs_stack(hist)
                        resp = game.get_all_data_batched()
                    continue

                if any(int(pos_list[10 + i]) == 1 for i in range(4)):
                    blue_elapsed += 1
                else:
                    blue_elapsed = 0
                cur_p10 = [int(x) for x in pos_list[:10]]
                pos_hist.append(cur_p10)
                prev4 = resolve_prev_motion(pos_hist, base_lag=5)
                obs11 = sensor.encode_11ch(obs, vram_list, pos_list, prev4_pos=prev4)
                sub_vec = build_subtile_vec(pos_list, cur_dots, prev4_pos=prev4, level_num=clears + 1, blue_elapsed=blue_elapsed)
                with torch.no_grad():
                    g_t = torch.from_numpy(obs11).float().unsqueeze(0).to(device)
                    v_t = torch.from_numpy(sub_vec).float().unsqueeze(0).to(device)
                    action = int(model(g_t, v_t).argmax(dim=1).item())

                if screen and steps % 2 == 0:
                    draw_monitor(screen, resp, nf, score, clears + 1, lives, clears)

                game.execute_action(action)
                resp = game.get_all_data_batched()
                bd2 = game.process_all_data(resp) if resp else None
                if bd2:
                    hist.append(bd2[0])
                    obs = game.build_obs_stack(hist)
                steps += 1

            scores.append(score)
            clears_all.append(clears)
            print(f"🎮 Fin Partie {ep+1} : Score={score} | Niveaux terminés={clears} | Steps={steps}", flush=True)
        return scores, clears_all
    finally:
        comm.close()
        try:
            proc.kill()
        except Exception:
            pass
        if not headless:
            pygame.quit()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pac-Man 30K Grandmaster AI Player")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL_PATH, help="Chemin du fichier .pth")
    parser.add_argument("--speed", type=float, default=1.5, help="Vitesse d'affichage MAME (0 = max)")
    parser.add_argument("--video", type=str, default="opengl", help="Driver vidéo MAME (opengl, gdi)")
    parser.add_argument("--headless", action="store_true", help="Sans fenêtre graphique")
    parser.add_argument("--episodes", type=int, default=1, help="Nombre de parties à jouer")
    args = parser.parse_args()
    play(model_path=args.model, speed=args.speed, video_driver=args.video, headless=args.headless, episodes=args.episodes)
