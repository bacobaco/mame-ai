import re
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import matplotlib
import matplotlib.patheffects
from datetime import datetime, timedelta
from matplotlib.ticker import FuncFormatter
from pathlib import Path

# Fix Windows encoding issues for print
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

# Mode non-interactif
matplotlib.use('Agg')

# Configuration du style "Premium Dark"
plt.style.use('dark_background')
matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial'] # Déjà présent sur la plupart des systèmes

def strip_ansi(text):
    ansi_escape = re.compile(r'\x1b\[[0-9;]*[mGKF]')
    return ansi_escape.sub('', text)

def smooth(y, box_pts):
    if len(y) < box_pts: return y
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth

def generate_oneshot_plot(filepath, output_path):
    print(f"Generation du graphique Premium pour : {Path(filepath).name}")
    
    # Parsing du log
    eps, means, sigmas, hiscores, steps, qs = [], [], [], [], [], []
    curr_ep, curr_data = None, {}
    
    # Détermination de la date de début via le nom du fichier
    match = re.search(r"(\d{8}_\d{6})", Path(filepath).name)
    start_dt = datetime.strptime(match.group(1), "%Y%m%d_%H%M%S") if match else datetime.now()
    
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
                            steps.append(curr_data.get('steps', 0))
                            sigmas.append(curr_data.get('sigma', 0))
                            qs.append(curr_data.get('q', 0))
                            hiscores.append(curr_data.get('hiscore', curr_data['mean']))
                    curr_ep = new_ep
                
                if p_m: curr_data['steps'] = float(p_m.group(1)) * 1000
                if m_m: curr_data['mean'] = float(m_m.group(1))
                if s_m: curr_data['sigma'] = float(s_m.group(1))
                if q_m: curr_data['q'] = int(q_m.group(1))
                if h_m: curr_data['hiscore'] = int(h_m.group(1))

        # Dernier point
        if curr_ep is not None and 'mean' in curr_data:
            eps.append(curr_ep)
            means.append(curr_data['mean'])
            steps.append(curr_data.get('steps', 0))
            sigmas.append(curr_data.get('sigma', 0))
            qs.append(curr_data.get('q', 0))
            hiscores.append(curr_data.get('hiscore', curr_data['mean']))

    except Exception as e:
        print(f"Erreur lecture : {e}")
        return

    if len(eps) < 5:
        print("Pas assez de données pour générer un graphique.")
        return

    # --- CRÉATION DU GRAPHIQUE ---
    fig, ax1 = plt.subplots(figsize=(16, 10), facecolor='#0B0E14')
    ax1.set_facecolor('#0B0E14')
    
    # Couloirs de couleur
    ax1.axhspan(0, 1000, color='red', alpha=0.05)
    ax1.axhspan(1000, 3000, color='orange', alpha=0.05)
    ax1.axhspan(3000, 6000, color='yellow', alpha=0.05)
    ax1.axhspan(6000, 10000, color='green', alpha=0.05)

    # Courbe Mean (Brute + Lissée)
    ax1.plot(eps, means, color='#FFFFFF', alpha=0.15, linewidth=1, label='Mean Score (Raw)')
    means_smoothed = smooth(means, min(50, len(means)//10))
    ax1.plot(eps, means_smoothed, color='#00FFCC', linewidth=2.5, label='Mean Score (Smoothed)', 
             solid_capstyle='round')

    # HiScore
    ax1.plot(eps, hiscores, color='#3399FF', alpha=0.8, linewidth=1.5, linestyle='--', label='High Score')

    # Annotation Max
    idx_max = np.argmax(means)
    ax1.annotate(f'PEAK: {means[idx_max]:.0f}', 
                 xy=(eps[idx_max], means[idx_max]), 
                 xytext=(0, 20), textcoords='offset points',
                 fontsize=12, color='#00FFCC', fontweight='bold', ha='center',
                 bbox=dict(boxstyle='round4,pad=0.5', facecolor='#0B0E14', alpha=0.8, edgecolor='#00FFCC'))

    # Axe Sigma
    ax_sig = ax1.twinx()
    ax_sig.plot(eps, sigmas, color='#FF9900', alpha=0.3, linewidth=1, label='Sigma (Exploration)')
    ax_sig.set_ylabel('Sigma Value', color='#FF9900', fontsize=12)
    ax_sig.tick_params(axis='y', colors='#FF9900')

    # Titres et étiquettes
    plt.title(f"PAC-MAN APE-X | PREMIUM PERFORMANCE ANALYSIS\n{Path(filepath).name}", 
              fontsize=18, color='white', fontweight='bold', pad=30)
    
    ax1.set_xlabel('Episodes', fontsize=14, color='white')
    ax1.set_ylabel('Score / Points', fontsize=14, color='white')
    ax1.grid(True, which='both', color='#333333', linestyle=':', alpha=0.5)
    
    # Légende
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax_sig.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc='upper left', frameon=True, facecolor='#1A1C23', edgecolor='#333333')

    # Footer avec info session
    footer_text = (f"Session Started: {start_dt.strftime('%Y-%m-%d %H:%M:%S')} | "
                   f"Max Score: {max(hiscores)} | "
                   f"Final Mean: {means[-1]:.1f} | "
                   f"Total Experience: {steps[-1]/1e6:.2f}M steps")
    plt.figtext(0.5, 0.02, footer_text, ha="center", fontsize=10, bbox={"facecolor":"#1A1C23", "alpha":0.5, "pad":5})

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(output_path, dpi=200, facecolor='#0B0E14')
    plt.close()
    print(f"Graphique genere avec succes : {output_path}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python plot_oneshot_premium.py <path_to_log>")
    else:
        log_path = sys.argv[1]
        out_name = Path(log_path).stem + "_premium.png"
        out_path = os.path.join(os.path.dirname(log_path), out_name)
        generate_oneshot_plot(log_path, out_path)
