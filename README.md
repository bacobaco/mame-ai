# 🕹️ Mame-AI: Ape-X Deep Reinforcement Learning for Arcade Games

Mame-AI is an artificial intelligence training framework designed for classic arcade games emulated via MAME. It uses state-of-the-art architectures (Ape-X, Rainbow DQN) for efficient distributed learning.

![Aesthetics](https://img.shields.io/badge/Aesthetics-Premium-blueviolet)
![Framework](https://img.shields.io/badge/Framework-PyTorch-orange)
![Games](https://img.shields.io/badge/Games-PacMan%20%7C%20Invaders-yellow)

---

## 🇺🇸 English Version

### 🚀 Key Features
- **Distributed Ape-X Architecture**: Multiple actors (`NUM_ACTORS`) collect experiences in parallel, sent to a centralized GPU-based `Learner`.
- **Rainbow DQN**: Includes NoisyNets (exploration), Prioritized Replay Buffer, Double DQN, Dueling Networks, and N-Step Learning.
- **High-Performance MAME Bridge**: Socket-based communication between Python and MAME (via Lua script) for ultra-fast memory reading and command injection.
- **Dynamic Dashboard**: Real-time performance analysis (moving averages, polynomial projections, records, queue saturation).
- **Precise CNN Extraction**: Direct mapping of VRAM (Video RAM) and hardware sprites into tensors for the neural network.

### 🛠️ Installation
1. **Prerequisites**: Python 3.8+, MAME installed, and ROMs (`pacman`, `invaders`) in your MAME folder.
2. **Environment Setup**:
   ```bash
   git clone https://github.com/bacobaco/mame-ai.git
   cd mame-ai
   pip install -r core/requirements.txt
   ```
3. **Paths**: Update `MAME_PATH` in the launch scripts (e.g., `pacman/pacman_multi.py`).

### 🎮 Usage
- **Train Pac-Man (Multi-Agent)**: `python pacman/pacman_multi.py`
- **Monitoring Dashboard**: `python analysis/plot_mean_pacman.py`

---

## 🇫🇷 Version Française

### 🚀 Fonctionnalités Clés
- **Architecture Ape-X Distribuée** : Utilisation de plusieurs acteurs collectant des expériences en parallèle, envoyées à un `Learner` centralisé sur GPU.
- **Rainbow DQN** : NoisyNets, Prioritized Replay, Double DQN, Dueling Networks et N-Step Learning.
- **Pont MAME Haute Performance** : Communication Socket (Lua) ultra-rapide.
- **Dashboard Dynamique** : Analyse en temps réel des performances et projections.

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
- **Dashboard** : `python analysis/plot_mean_pacman.py`

---

## 📁 Project Structure
- `core/`: AI Logic, DQN, Buffer, Socket Communication.
- `pacman/`: Pac-Man specific interface and scripts.
- `invaders/`: Space Invaders scripts.
- `analysis/`: Monitoring and graphing scripts.
- `media/`: High-resolution graphs and frame captures.

## 📝 License
Free for educational and AI research purposes.
