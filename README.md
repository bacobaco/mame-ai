# 🕹️ Mame-AI: Ape-X Deep Reinforcement Learning for Arcade Games

Mame-AI is an artificial intelligence training framework designed for classic arcade games emulated via MAME. It uses state-of-the-art architectures (Ape-X, Rainbow DQN) for efficient distributed learning.

![Aesthetics](https://img.shields.io/badge/Aesthetics-Premium-blueviolet)
![Framework](https://img.shields.io/badge/Framework-PyTorch-orange)
![Games](https://img.shields.io/badge/Games-PacMan%20%7C%20Invaders-yellow)
![Version](https://img.shields.io/badge/Version-2.1-green)

---

## 🇺🇸 English Version

### 🚀 Key Features (v2.1 Update)
- **Distributed Ape-X Architecture**: Multiple actors (`NUM_ACTORS`) collect experiences in parallel, sent to a centralized GPU-based `Learner`.
- **Rainbow DQN**: NoisyNets, Prioritized Replay Buffer, Double DQN, Dueling Networks, and N-Step Learning (n=5).
- **[NEW] 3-Layer Precise CNN**: Upgraded CNN architecture (32, 64, 128 filters) with GroupNorm for superior spatial feature detection in complex maze environments.
- **[NEW] Sigma Burst Mechanism**: Automatic exploration trigger that resets NoisyNet noise if performance stagnation is detected (2000 episodes without new record).
- **[NEW] Reward Shaping (Level Focus)**: Implemented Level Clear bonuses (+50) to incentivize long-term strategy over simple survival.
- **High-Performance MAME Bridge**: Socket-based communication between Python and MAME (via Lua script).
- **Optimized Queue Handling**: Balanced multi-process communication tuned for Windows to avoid lock contention (30k queue / 100k buffer).

### 🛠️ Installation
1. **Prerequisites**: Python 3.8+, MAME installed, and ROMs (`pacman`, `invaders`).
2. **Environment Setup**:
   ```bash
   git clone https://github.com/bacobaco/mame-ai.git
   cd mame-ai
   pip install -r core/requirements.txt
   ```
3. **Paths**: Update `MAME_PATH` in `pacman/pacman_multi.py`.

### 🎮 Usage
- **Train Pac-Man (Multi-Agent)**: `python pacman/pacman_multi.py`
- **Monitoring Dashboard**: `python analysis/plot_mean_pacman.py`

---

## 🇫🇷 Version Française

### 🚀 Fonctionnalités Clés (v2.1)
- **Architecture Ape-X Distribuée** : Collecte parallèle massive par plusieurs acteurs.
- **Rainbow DQN** : NoisyNets, PER, Double DQN, Dueling & N-Step.
- **[NOUVEAU] CNN Precise 3-Couches** : Architecture étendue (128 filtres) avec GroupNorm pour une perception spatiale ultra-fine.
- **[NOUVEAU] Sigma Burst** : Mécanisme d'exploration forcée en cas de stagnation (reset du bruit après 2000 épisodes sans record).
- **[NOUVEAU] Reward Shaping** : Bonus de fin de niveau (+50) pour pousser l'IA à "nettoyer" le tableau plutôt que fuir.
- **Optimisation Windows** : Gestion de queue (30k) et buffer (100k) optimisée pour éviter les ralentissements du système.

### 🛠️ Installation
1. **Prérequis** : Python 3.8+, MAME, ROMs (`pacman`, `invaders`).
2. **Setup** :
   ```bash
   git clone https://github.com/bacobaco/mame-ai.git
   cd mame-ai
   pip install -r core/requirements.txt
   ```

### 🎮 Utilisation
- **Entraîner Pac-Man** : `python pacman/pacman_multi.py`
- **Dashboard de Monitoring** : `python analysis/plot_mean_pacman.py`

---

## 📁 Project Structure
- `core/`: AI Logic, DQN, Buffer, Socket Communication.
- `pacman/`: Pac-Man specific interface and scripts.
- `invaders/`: Space Invaders scripts.
- `analysis/`: Monitoring and graphing scripts.
- `media/`: Graphs and frame captures.

## 📝 License
Free for educational and AI research purposes.
