# 🕹️ Mame-AI: Ape-X Deep Reinforcement Learning for Arcade Games

Mame-AI is an artificial intelligence training framework designed for classic arcade games emulated via MAME. It uses state-of-the-art architectures (Ape-X, Rainbow DQN) for efficient distributed learning.

![Aesthetics](https://img.shields.io/badge/Aesthetics-Premium-blueviolet)
![Framework](https://img.shields.io/badge/Framework-PyTorch-orange)
![Games](https://img.shields.io/badge/Games-PacMan%20%7C%20Invaders-yellow)
![Version](https://img.shields.io/badge/Version-2.2-green)

---

## 🇺🇸 English Version

### 🚀 Key Features (v2.2 Update)
- **Distributed Ape-X Architecture**: Multiple actors (`NUM_ACTORS`) collect experiences in parallel, sent to a centralized GPU-based `Learner`.
- **Rainbow DQN**: NoisyNets, Prioritized Replay Buffer, Double DQN, Dueling Networks, and N-Step Learning (n=5).
- **3-Layer Precise CNN**: Standardized CNN architecture (32, 64, 128 filters) with GroupNorm for stable spatial representation in maze environments.
- **Sigma Burst Mechanism**: Automatic exploration trigger that resets NoisyNet noise if performance stagnation is detected (1500 episodes without new record).
- **Reward Shaping (Level Focus)**: Implemented Level Clear bonuses to incentivize long-term strategy over simple survival.
- **[NEW] RAM Optimization (500k Buffer)**: Capped transition buffer size to 500,000 transitions to guarantee error-free, fast serialization on 32GB RAM systems.
- **[NEW] Maze Vision Correction**: Restored optimal grid contrast mappings (walls=0.2, corridors=0.4, dots=0.6, power=0.9, vulnerable ghosts=1.0, eaten=0.4) to eliminate blind spots.
- **High-Performance MAME Bridge**: Socket-based communication between Python and MAME (via Lua script).
- **Optimized Queue Handling**: Balanced multi-process communication tuned for Windows to avoid lock contention (30k queue / 500k buffer).

### 🛠️ Installation
1. **Prerequisites**: Python 3.8+, MAME installed, and ROMs (`pacman`, `invaders`).
2. **Environment Setup**:
   ```bash
   git clone https://github.com/bacobaco/mame-ai.git
   cd mame-ai
   pip install -r core/requirements.txt
   ```
3. **Paths**: Update `MAME_PATH` in `pacman/pacman_multi.py`.

### 🎮 Usage (Pac-Man & Space Invaders)

#### 🟡 Pac-Man
- **Single Agent (Standard DQN)**:
  ```bash
  python pacman/pacman.py
  ```
- **Multi-Agent (Ape-X Distributed)**:
  ```bash
  python pacman/pacman_multi.py
  ```
  *Note: Launches multiple MAME actors (configured in script) reporting to a central Learner.*

#### 👾 Space Invaders
- **Single Agent (Standard DQN)**:
  ```bash
  python invaders/invaders.py
  ```
- **Multi-Agent (Ape-X Distributed)**:
  ```bash
  python invaders/invaders_multi.py
  ```

---

## 🇫🇷 Version Française

### 🚀 Fonctionnalités Clés (v2.2)
- **Architecture Ape-X Distribuée** : Collecte parallèle massive par plusieurs acteurs pour un entraînement accéléré.
- **Rainbow DQN** : Intégration complète de NoisyNets (exploration), PER (priorités), Double DQN (stabilité), Dueling (avantage) & N-Step (anticipation).
- **CNN Precise 3-Couches** : Architecture étendue (128 filtres) avec GroupNorm pour une perception spatiale ultra-fine des labyrinthes.
- **Sigma Burst** : Mécanisme d'exploration forcée automatique en cas de stagnation prolongée (1500 épisodes sans record).
- **Reward Shaping** : Système de récompenses affiné pour Pac-Man (Bonus niveau) et Invaders (priorité soucoupes).
- **[NOUVEAU] Optimisation RAM (500k Buffer)** : Limite du buffer à 500 000 transitions pour garantir des sauvegardes rapides et sans erreur de mémoire (`MemoryError`) sur les configurations à 32 Go.
- **[NOUVEAU] Correction Contraste Labyrinthe** : Alignement de la vision de la grille de jeu pour dissocier clairement les fantômes vulnérables des pastilles de score.
- **Optimisation Windows** : Gestion de flux de données optimisée pour éviter les ralentissements système.

### 🛠️ Installation
1. **Prérequis** : Python 3.8+, MAME, ROMs (`pacman`, `invaders`).
2. **Setup** :
   ```bash
   git clone https://github.com/bacobaco/mame-ai.git
   cd mame-ai
   pip install -r core/requirements.txt
   ```

### 🎮 Utilisation

#### 🟡 Pac-Man
- **Agent Solo (DQN Standard)** : `python pacman/pacman.py`
- **Multi-Agent (Ape-X Distribué)** : `python pacman/pacman_multi.py`

#### 👾 Space Invaders
- **Agent Solo (DQN Standard)** : `python invaders/invaders.py`
- **Multi-Agent (Ape-X Distribué)** : `python invaders/invaders_multi.py`

---

## 📁 Project Structure
- `core/`: Logique IA commune, DQN, Buffer PER, Communication Socket MAME.
- `pacman/`: Interface spécifique, scripts d'entraînement et assets pour Pac-Man.
- `invaders/`: Interface spécifique et scripts pour Space Invaders.
- `analysis/`: Scripts de monitoring et génération de graphiques de performance.
- `media/`: Captures d'écran, vidéos des meilleurs runs et logs graphiques.

## 📝 License
Free for educational and AI research purposes.
