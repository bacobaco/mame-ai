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
  # Or resume a specific session:
  python pacman/pacman_multi.py --session 38
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

### ⌨️ Runtime Controls & Shortcuts

#### 🕹️ Live Keyboard Controls (Single-Agent: `pacman.py` & `invaders.py`)
> [!NOTE]
> Shortcuts require holding **`Ctrl + Shift + [Function Key]`** while the terminal or game window has focus.

| Shortcut | Description | Action |
| :--- | :--- | :--- |
| **`Ctrl + Shift + F2`** | **Exit & Save** | Triggers graceful quit, saves final model checkpoint (`.pth`) and buffer (`.buffer`). |
| **`Ctrl + Shift + F3`** | **Disable Debug** | Resets debug output level to 0. |
| **`Ctrl + Shift + F4`** | **Debug Level** | Cycles/increments debug verbosity (0 to 3). |
| **`Ctrl + Shift + F5`** | **Speed Toggle** | Toggles MAME throttle between slow/normal viewing speed and maximum training speed. |
| **`Ctrl + Shift + F6`** | **Speed Adjust** | Increments the slow viewing speed (+0.1). |
| **`Ctrl + Shift + F7`** | **Generate Figure** | Instantly exports an updated training performance graph to `media/`. |
| **`Ctrl + Shift + F8`** | **Inspect Frame** | Saves and displays the current frame/state tensor as seen by the AI (CNN/VRAM vision). |
| **`Ctrl + Shift + F9`** | **Mode Toggle** | Switches between `exploration` (training) and `exploitation` (deterministic evaluation). |
| **`Ctrl + Shift + F10`**| **Increase Epsilon** | Increases exploration rate $\epsilon$ (+0.1, up to 1.0). |
| **`Ctrl + Shift + F11`**| **Decrease Epsilon** | Decreases exploration rate $\epsilon$ (-0.1, down to $\epsilon_{end}$). |
| **`Ctrl + Shift + F12`**| **Lua Debug** | Toggles verbose Lua socket bridge logs (`debug on/off`). |

#### 🛑 Graceful Shutdown & Multi-Actor Controls (`pacman_multi.py`)
- **`stop.flag` Mechanism**: 
  Create an empty file named `stop.flag` at the project root or in `pacman/` (e.g. `New-Item stop.flag` in PowerShell or `touch stop.flag`).
  - Automatically caught by the main learner loop.
  - Safely saves the neural network (`pacman_apex.pth`), replay buffer (`pacman_multi.buffer`), and archives session results in `resultats_pacman.txt`.
  - Terminates MAME instances and cleans up the flag file automatically.
- **`Ctrl + C` (KeyboardInterrupt)**: Emergency fallback that also saves model and buffer before exiting.

#### 🌐 Real-Time Web Monitoring
- While training, a live web dashboard runs automatically on port 5000:
  - **Local**: `http://127.0.0.1:5000/`
  - **LAN / Mobile**: `http://<YOUR_LOCAL_IP>:5000/`
  - Automatically auto-refreshes the latest performance graph every 30 seconds.


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
- **Agent Solo (DQN Standard)** :
  ```bash
  python pacman/pacman.py
  ```
- **Multi-Agent (Ape-X Distribué)** :
  ```bash
  python pacman/pacman_multi.py
  # Ou pour reprendre une session existante :
  python pacman/pacman_multi.py --session 38
  ```

#### 👾 Space Invaders
- **Agent Solo (DQN Standard)** :
  ```bash
  python invaders/invaders.py
  ```
- **Multi-Agent (Ape-X Distribué)** :
  ```bash
  python invaders/invaders_multi.py
  ```

### ⌨️ Raccourcis Clavier & Contrôles d'Entraînement

#### 🕹️ Raccourcis en Direct (Mode Solo : `pacman.py` & `invaders.py`)
> [!NOTE]
> Les raccourcis s'activent en maintenant **`Ctrl + Shift + [Touche Fonction]`** pendant que la console ou la fenêtre du jeu est active.

| Raccourci | Fonction | Description |
| :--- | :--- | :--- |
| **`Ctrl + Shift + F2`** | **Quitter & Sauvegarder** | Déclenche un arrêt propre, sauvegarde le modèle final (`.pth`) et le buffer (`.buffer`). |
| **`Ctrl + Shift + F3`** | **Désactiver Debug** | Remet le niveau d'affichage de debug à 0. |
| **`Ctrl + Shift + F4`** | **Niveau Debug** | Fait défiler/augmente la verbosité du debug (niveaux 0 à 3). |
| **`Ctrl + Shift + F5`** | **Bascule Vitesse** | Bascule la vitesse MAME entre affichage fluide/lent et vitesse maximale d'entraînement. |
| **`Ctrl + Shift + F6`** | **Ajuster Vitesse Lente**| Augmente par palier la vitesse lente (+0.1). |
| **`Ctrl + Shift + F7`** | **Générer Graphique** | Exporte immédiatement une nouvelle figure de performance dans le dossier `media/`. |
| **`Ctrl + Shift + F8`** | **Inspecter la Frame** | Sauvegarde et affiche l'état/tenseur actuel perçu par l'IA (vision CNN / VRAM). |
| **`Ctrl + Shift + F9`** | **Bascule Mode** | Alterne entre `exploration` (apprentissage avec bruit) et `exploitation` (inférence pure). |
| **`Ctrl + Shift + F10`**| **Augmenter Epsilon** | Augmente le taux d'exploration $\epsilon$ (+0.1, max 1.0). |
| **`Ctrl + Shift + F11`**| **Diminuer Epsilon** | Réduit le taux d'exploration $\epsilon$ (-0.1, min $\epsilon_{end}$). |
| **`Ctrl + Shift + F12`**| **Debug Socket Lua** | Active/désactive les logs détaillés du pont Lua (`debug on/off`). |

#### 🛑 Arrêt Gracieux & Contrôles Multi-Acteurs (`pacman_multi.py`)
- **Mécanisme `stop.flag`** :
  Créez un fichier vide nommé exactement **`stop.flag`** à la racine du projet ou dans le dossier `pacman/` (ex : `New-Item stop.flag` dans PowerShell ou `touch stop.flag`).
  - Détecté automatiquement par la boucle principale du Learner.
  - Sauvegarde en toute sécurité le réseau de neurones (`pacman_apex.pth`), le buffer de rejeu (`pacman_multi.buffer`), et archive les résultats dans `resultats_pacman.txt`.
  - Ferme les instances MAME et supprime automatiquement le fichier `stop.flag`.
- **`Ctrl + C` (Interruption clavier)** : Arrêt de secours qui sauvegarde également les poids et le buffer avant de quitter.

#### 🌐 Tableau de Bord Web en Temps Réel
- Lors de l'entraînement, un serveur Flask est automatiquement lancé sur le port 5000 :
  - **Localement** : `http://127.0.0.1:5000/`
  - **Réseau Local / Mobile** : `http://<VOTRE_IP_LOCALE>:5000/`
  - Rafraîchit automatiquement le dernier graphique de progression toutes les 30 secondes.

---

---

## 🏆 Pac-Man 30K Grandmaster (Pure Neural AI — ~28k–32k pts)

### 🧠 Architecture & Principe (`pacman/pacman_30k.py` & `pacman/pacman_30k.pth`)
- **Inférence 100 % neuronale (`frame_per_step 1`, 60 Hz)** : aucun script ni masque d'action au runtime — chaque frame passe uniquement dans `PixelGrandmasterCNN(...).argmax()`.
- **Capteur spatial 12 canaux (`MazeSensor11`)** :
  - 7 canaux temporels MAME (`build_obs_stack` : murs/pastilles, traînée Pac-Man, 4 frames fantômes dangereux, fantômes bleus).
  - 5 champs continus (`danger_field`, BFS hyperbolique `1/(1 + 0.22*d)` vers les pastilles `64`, super-pastilles `48`, fantômes bleus `32`, et champ `safe_reach` avec marge de sécurité `lead >= 3` sur `22` tuiles).
- **Tête hybride (`PixelGrandmasterCNN`)** : tronc ResNet dilaté (`48` canaux), patch égocentrique `11x11` centré sur Pac-Man + 4 rayons directionnels (`L, R, U, D`), contexte global `4x4`, et `31` scalaires invariants (alignement sous-tuile, vitesses relatives, états des fantômes, niveau, timer bleu).
- **Entraînement (`SoftQ-V3`)** : 24 époques sur `708 007` frames sans pauses (parties teacher `28k–34.5k` + DAgger on-policy + Elite Self-Play `28.5k–33.3k` + `NegDeath`), combinant Cross-Entropy, distillation Soft-Q 4 actions (`tau=14`), barrière/marge anti-pièges (`unsafe4`) et anti-murs (`wall4`), avec pondération renforcée aux carrefours dangereux et en fin de niveau (`<= 18` pastilles).
- **Scores observés en évaluation déterministe (6 profils)** : `[3230, 4180, 27230, 27180, 32080, 32280]` (`max=32 280` pts, `7` niveaux complétés, validation inédite `98.26%`).

### ▶️ Rejouer / Observer une partie à ~28 000+ pts (`pacman/watch_30k.py`)
Utilisez toujours l'environnement virtuel du projet (`.\venv\Scripts\python.exe`) depuis la racine `MameAI` :

```powershell
# Regarder en temps réel (60 FPS) et s'arrêter dès qu'une partie atteint >= 28 000 pts :
.\venv\Scripts\python.exe pacman\watch_30k.py --model pacman\pacman_30k.pth --speed 1.0 --target 28000

# En vitesse x2 ou vitesse max (0) :
.\venv\Scripts\python.exe pacman\watch_30k.py --model pacman\pacman_30k.pth --speed 2.0 --target 28000
.\venv\Scripts\python.exe pacman\watch_30k.py --model pacman\pacman_30k.pth --speed 0 --target 28000
```

Vous pouvez aussi lancer le lecteur standard avec vue radar/capteurs Pygame :
```powershell
.\venv\Scripts\python.exe pacman\pacman_30k.py --model pacman\pacman_30k.pth --speed 1.0
```

### ⚙️ Pourquoi `watch_30k.py` reproduit fidèlement les parties ~28k–32k
1. **Timing MAME fidèle (`-sound none`)** : désactiver le son évite la synchronisation audio Windows et conserve la cadence exacte à 60 Hz (`frame_per_step 1`).
2. **Capture propre après respawn (`wait_for 25` puis 4 frames)** : après chaque démarrage, passage de niveau ou perte de vie, l'historique des 4 frames est rempli *après* l'animation d'attente afin de ne pas injecter de frames figées dans le CNN.
3. **Décalage d'ouverture déterministe (`candidate_workers = [2, 2, 4, 3, 5, 1]`)** : reproduit le décalage initial exact `shift_steps = (wid * 3) % 15` (`0` = gauche, `1` = droite) utilisé lors des évaluations déterministes (les profils `2, 3, 4, 5` atteignent régulièrement `27k–32k` en pur `argmax`).

---

## 📁 Project Structure
- `core/`: Logique IA commune, DQN, Buffer PER, Communication Socket MAME.
- `pacman/`: Interface spécifique, scripts d'entraînement et assets pour Pac-Man.
- `invaders/`: Interface spécifique et scripts pour Space Invaders.
- `analysis/`: Scripts de monitoring et génération de graphiques de performance.
- `media/`: Captures d'écran, vidéos des meilleurs runs et logs graphiques.

## 📝 License
Free for educational and AI research purposes.
