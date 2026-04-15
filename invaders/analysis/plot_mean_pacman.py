import re
import matplotlib.pyplot as plt
import numpy as np
import os
import time
import matplotlib
from datetime import datetime, timedelta
from matplotlib.ticker import FuncFormatter
from pathlib import Path

matplotlib.use('Agg') # Mode non-interactif

# --- CONFIGURATION DES CHEMINS ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPT_DIR)
MEDIA_DIR = os.path.join(ROOT_DIR, "media")
os.makedirs(MEDIA_DIR, exist_ok=True)

# --- SÉLECTION DYNAMIQUE DU LOG ---
log_dir = os.path.join(ROOT_DIR, "pacman", "logs")

def strip_ansi(text):
    ansi_escape = re.compile(r'\x1b\[[0-9;]*[mGKF]')
    return ansi_escape.sub('', text)

def generate_plot(eps, means, sigmas, hiscores, global_steps_list, qs, start_dt, now_dt, save_path, is_zoom=False):
    if len(eps) < 2: return
    
    fig, ax1 = plt.subplots(figsize=(14, 8))
    
    # --- AXE Y1 (Scores Moyens) ---
    ax1.plot(eps, means, label='Score (Moyenne 100)', color='black', alpha=0.5, linewidth=1.0)
    
    # Projection Polynomiale (si assez de points)
    proj_dist = min(2000, max(500, int((eps[-1]-eps[0]) * 0.2) if is_zoom else int(eps[-1] * 0.5)))
    future_eps = np.linspace(eps[0], eps[-1] + proj_dist, 300)

    if len(means) > 50:
        try:
            z3 = np.polyfit(eps, means, 3)
            p3 = np.poly1d(z3)
            y_proj = p3(future_eps)
            if np.max(y_proj) < 100000 and np.min(y_proj) > -1000:
                ax1.plot(future_eps, y_proj, label='Projection (Poly d-3)', color='#d62728', linewidth=1.5, alpha=0.9)
        except: pass
    
    ax1.set_xlim(left=eps[0] if is_zoom else 0, right=eps[-1] + proj_dist)
    # AXE Y : Dynamique (S'adapte au score min avec une petite marge)
    if len(means) > 0:
        y_min = min(means)
        y_max = max(means)
        margin = (y_max - y_min) * 0.05 if y_max > y_min else 50
        ax1.set_ylim(max(0, y_min - margin), y_max + margin * 3) # Plus de place en haut pour les cadres

        # --- ÉTIQUETTE MAX MEAN ---
        idx_max = np.argmax(means)
        ax1.annotate(f'Max Mean: {means[idx_max]:.0f}', 
                     xy=(eps[idx_max], means[idx_max]), 
                     xytext=(0, 10), textcoords='offset points',
                     fontsize=10, color='red', fontweight='bold', ha='center',
                     bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.7, edgecolor='red'))

    ax1.set_xlabel('Épisode (Steps Session)', fontsize=12)
    ax1.set_ylabel('Mean Score', color='black', fontsize=12)

    def steps_formatter(x, pos):
        if len(eps) > 1:
            try:
                s = np.interp(x, eps, global_steps_list)
                return f"{int(x)}\n({s/1e6:.2f}M)"
            except: return str(int(x))
        return str(int(x))
    ax1.xaxis.set_major_formatter(FuncFormatter(steps_formatter))

    # --- AXE Y2 (Saturation Q) ---
    ax2 = ax1.twinx()
    ax2.scatter(eps, qs, c=qs, cmap='jet', s=3, alpha=0.3, label='Queue Q')
    ax2.set_ylabel('Saturation Queue (Q)', color='black', fontsize=10)
    if len(qs) > 0:
        q_min, q_max = min(qs), max(qs)
        q_margin = (q_max - q_min) * 0.1 if q_max > q_min else 500
        ax2.set_ylim(max(0, q_min - q_margin), q_max + q_margin * 2)

    # --- AXE Y3 (HiScore - BLEU) ---
    ax3 = ax1.twinx()
    ax3.spines['right'].set_position(('axes', 0.95))
    ax3.set_frame_on(True)
    ax3.patch.set_visible(False)
    for sp in ax3.spines.values(): sp.set_visible(False)
    ax3.spines['right'].set_visible(True)
    ax3.spines['right'].set_color('blue')
    ax3.plot(eps, hiscores, label='HiScore (Max)', color='blue', alpha=0.6, linewidth=1.0, linestyle='--')
    ax3.set_ylabel('HiScore (Max)', color='blue', fontsize=10, labelpad=-35)
    ax3.tick_params(axis='y', colors='blue', labelsize=8, pad=-20)

    # --- ÉTIQUETTE HISCORE FINAL ---
    if len(hiscores) > 0:
        ax3.annotate(f'Record: {hiscores[-1]}', 
                     xy=(eps[-1], hiscores[-1]), 
                     xytext=(-50, 10), textcoords='offset points',
                     fontsize=10, color='blue', fontweight='bold',
                     bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.7, edgecolor='blue'))

    if len(hiscores) > 0:

        h_min, h_max = min(hiscores), max(hiscores)
        h_margin = (h_max - h_min) * 0.1 if h_max > h_min else 500
        ax3.set_ylim(max(0, h_min - h_margin), h_max + h_margin * 2)

    # --- AXE Y4 (Sigma - ORANGE) ---
    ax4 = ax1.twinx()
    ax4.spines['right'].set_position(('axes', 0.88))
    ax4.set_frame_on(True)
    ax4.patch.set_visible(False)
    for sp in ax4.spines.values(): sp.set_visible(False)
    ax4.spines['right'].set_visible(True)
    ax4.spines['right'].set_color('orange')
    ax4.plot(eps, sigmas, label='Sigma', color='orange', alpha=0.4, linewidth=0.8)
    if len(sigmas) > 0:
        s_min, s_max = min(sigmas), max(sigmas)
        s_margin = (s_max - s_min) * 0.1 if s_max > s_min else 0.001
        ax4.set_ylim(max(0, s_min - s_margin), s_max + s_margin * 2)
    ax4.tick_params(axis='y', left=False, right=True, direction='in', labelsize=7, colors='orange')

    # --- AXE X TOP (TIMELINE) ---
    total_duration_h_real = (now_dt - start_dt).total_seconds() / 3600.0
    ep_speed = eps[-1] / total_duration_h_real if total_duration_h_real > 0 else 800.0
    sec_ax = ax1.secondary_xaxis('top', functions=(lambda x: x / ep_speed, lambda x: x * ep_speed))
    sec_ax.xaxis.set_major_formatter(FuncFormatter(lambda x, pos: f"{int(x)}h\n({(start_dt + timedelta(hours=x)).strftime('%d/%m %H:%M')})"))
    sec_ax.set_xlabel('Timeline Session', fontsize=10, color='green', labelpad=-32)
    sec_ax.tick_params(axis='x', colors='green', labelsize=8)

    # TENDANCES LOCALES & GLOBALES (ÉTENDUES)
    if len(means) > 10:
        # Globale (session entière) - Étendue sur tout le graph
        z_g = np.polyfit(eps, means, 1)
        ax1.plot(future_eps, np.poly1d(z_g)(future_eps), color='magenta', linestyle='--', linewidth=1, alpha=0.6, label=f'Globale (Session) | Pente: {z_g[0]:+.4f}')
        
        # Locale (1000 derniers) - Étendue vers le futur
        n_s = min(1000, len(means))
        z_s = np.polyfit(eps[-n_s:], means[-n_s:], 1)
        future_eps_local = np.linspace(eps[-n_s], future_eps[-1], 200)
        ax1.plot(future_eps_local, np.poly1d(z_s)(future_eps_local), color='green', linewidth=2, alpha=0.8, label=f'Locale ({n_s} eps) | Pente: {z_s[0]:+.4f}')

    # PRÉDICTIONS
    try:
        z_g = np.polyfit(eps, means, 1)
        if z_g[0] > 0:
            ep_10000 = (10000 - z_g[1]) / z_g[0]
            h_10000 = (ep_10000 - eps[-1]) / ep_speed
            text_pred = (f"=== SESSION DU {start_dt.strftime('%d/%m')} ===\n"
                         f" >> Objectif 10000 : Épisode {int(ep_10000)} (~{h_10000:.1f}h)\n"
                         f" -- Expérience : {global_steps_list[-1]/1e6:.1f} Millions")
            ax1.text(0.02, 0.94, text_pred, transform=ax1.transAxes, fontsize=9, fontweight='bold',
                     verticalalignment='top', bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.9, edgecolor='#d62728', linewidth=2.0))
    except: pass

    ax1.legend(loc='lower left', bbox_to_anchor=(0.02, 0.02), fontsize=8, fancybox=True, shadow=True)
    ax1.grid(True, alpha=0.2)
    ax1.text(0.02, 0.99, f"COCKPIT PACMAN - {Path(save_path).name}", transform=ax1.transAxes, fontsize=10, fontweight='bold', verticalalignment='top')
    
    plt.tight_layout(pad=1.0)
    plt.savefig(save_path, dpi=150)
    plt.close('all')

last_log_used = None
while True:
    now_dt = datetime.now()
    if not os.path.exists(log_dir):
        time.sleep(10)
        continue

    all_logs = sorted([f for f in os.listdir(log_dir) if f.startswith("training_log_") and f.endswith(".txt")])
    if all_logs:
        target_log = max(all_logs, key=lambda f: os.path.getmtime(os.path.join(log_dir, f)))
    else:
        target_log = None
        
    if not target_log:
        print(f"❌ Aucun log trouvé !")
        time.sleep(10)
        continue

    filepath = os.path.join(log_dir, target_log)
    if target_log != last_log_used:
        print(f"🚀 Analyse du log : {target_log} (Dernière modification)")
        last_log_used = target_log

    match = re.search(r"(\d{8}_\d{6})", target_log)
    start_dt = datetime.strptime(match.group(1), "%Y%m%d_%H%M%S") if match else now_dt

    eps, means, sigmas_list, hiscores_list, global_steps_list, qs = [], [], [], [], [], []
    curr_ep, curr_data = None, {}
    
    try:
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                clean_line = strip_ansi(line)
                e_m = re.search(r"Eps:\s*(\d+)", clean_line)
                p_m = re.search(r"(?:Pas Globaux|Steps|Buf):\s*([-+]?\d*\.\d+|\d+)k", clean_line)
                m_m = re.search(r"Mean:\s*([-+]?\d*\.\d+|\d+)", clean_line)
                s_m = re.search(r"Sigma:\s*([-+]?\d*\.\d+|\d+)", clean_line)
                q_m = re.search(r"Q:\s*(\d+)", clean_line)
                h_m = re.search(r"HiScore:\s*(\d+)", clean_line)
                
                if e_m:
                    new_ep = int(e_m.group(1))
                    if new_ep != curr_ep and curr_ep is not None:
                        if 'mean' in curr_data:
                            eps.append(curr_ep)
                            means.append(curr_data['mean'])
                            global_steps_list.append(curr_data.get('steps', 0))
                            sigmas_list.append(curr_data.get('sigma', 0))
                            qs.append(curr_data.get('q', 0))
                            hiscores_list.append(curr_data.get('hiscore', curr_data['mean']))
                    curr_ep = new_ep
                
                if p_m: curr_data['steps'] = float(p_m.group(1)) * 1000
                if m_m: curr_data['mean'] = float(m_m.group(1))
                if s_m: curr_data['sigma'] = float(s_m.group(1))
                if q_m: curr_data['q'] = int(q_m.group(1))
                if h_m: curr_data['hiscore'] = int(h_m.group(1))

        if curr_ep is not None and 'mean' in curr_data:
            eps.append(curr_ep)
            means.append(curr_data['mean'])
            global_steps_list.append(curr_data.get('steps', 0))
            sigmas_list.append(curr_data.get('sigma', 0))
            qs.append(curr_data.get('q', 0))
            hiscores_list.append(curr_data.get('hiscore', curr_data['mean']))

        if len(eps) >= 2:
            output_path = os.path.join(MEDIA_DIR, "pacman_mean_graph.png")
            generate_plot(eps, means, sigmas_list, hiscores_list, global_steps_list, qs, start_dt, now_dt, output_path)
            
            # Zoom : On se base sur les 4000 derniers numéros d'épisodes (indépendant de la fréquence du log)
            target_ep = eps[-1]
            zoom_indices = [i for i, ep in enumerate(eps) if ep >= target_ep - 4000]
            if len(zoom_indices) > 1:
                idx = zoom_indices[0]
                output_path_zoom = os.path.join(MEDIA_DIR, "pacman_mean_graph_zoom.png")
                generate_plot(eps[idx:], means[idx:], sigmas_list[idx:], hiscores_list[idx:], global_steps_list[idx:], qs[idx:], start_dt, now_dt, output_path_zoom, is_zoom=True)
            
            print(f"[{datetime.now().strftime('%H:%M:%S')}] ✅ Dashboard PACMAN OK | {len(eps)} Lignes | Eps: {eps[-1]}")

    except Exception as e:
        print(f"❌ Erreur lecture : {e}")

    time.sleep(30)
