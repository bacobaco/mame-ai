# 🕹️ MAME AI Training Framework

[![Français](https://img.shields.io/badge/Langue-Français-blue)](#français) [![English](https://img.shields.io/badge/Language-English-red)](#english)

---

<a name="français"></a>
# 🇫🇷 Français

Ce projet est un framework complet permettant d'entraîner des agents d'Intelligence Artificielle (Reinforcement Learning) sur des jeux d'arcade classiques (Space Invaders, Pac-Man) via l'émulateur **MAME**.

Il utilise une architecture **Client-Serveur** où Python (le cerveau) communique avec MAME (le corps) via un socket TCP local, permettant un contrôle frame-par-frame et une lecture directe de la mémoire RAM du jeu.

---

## 🚀 Fonctionnement & Architecture

Le système repose sur trois composants principaux :

1.  **MAME & Lua Script (`PythonBridgeSocket.lua`)** :
    *   MAME exécute le jeu.
    *   Un script Lua intégré agit comme serveur. Il expose les adresses mémoire (RAM) et écoute les commandes d'input.
    *   Il synchronise l'émulation avec l'IA via un mécanisme de `wait_for` (attente de commandes).

2.  **Interface de Communication (`MameCommSocket.py`)** :
    *   Gère la connexion TCP brute entre Python et Lua.
    *   Envoie des commandes (ex: `execute P1_Button_1(1)`) et reçoit les états mémoire.

3.  **Cerveau IA (`AI_Mame.py` & Scripts de Jeu)** :
    *   Implémente des algorithmes de **Deep Reinforcement Learning** (Rainbow DQN, DreamerV2).
    *   **`invaders.py` / `pacman.py`** : Wrappers spécifiques à chaque jeu qui définissent les récompenses (rewards), extraient l'état (pixels ou RAM) et gèrent la boucle d'entraînement.

```mermaid
graph LR
    A[MAME Emulator] -- RAM & Video --> B(Lua Script Server)
    B -- TCP Socket --> C(Python Client)
    C -- Actions (Joy/Btn) --> B
    C -- PyTorch Model --> D[Neural Network]
```

---

## ✨ Fonctionnalités Clés

*   **Algorithmes Avancés (Rainbow DQN)** :
    *   **Double DQN** & **Dueling DQN** pour une meilleure estimation des valeurs.
    *   **Noisy Nets** pour une exploration dynamique (remplace Epsilon-Greedy).
    *   **Prioritized Experience Replay (PER)** pour apprendre des moments importants.
    *   **N-Step Learning** pour une vision à plus long terme.
*   **Support Multi-Architectures** :
    *   **CNN (Convolutional Neural Network)** : L'IA "voit" l'écran (pixels bruts ou redimensionnés).
    *   **MLP (Multi-Layer Perceptron)** : L'IA lit directement la RAM (positions X/Y, états).
*   **Outils de Visualisation** :
    *   Serveur Web intégré (Flask) pour suivre les courbes de score en temps réel.
    *   Génération de graphiques `.png` automatiques.
    *   Enregistrement vidéo des meilleures parties.

---

## 📂 Structure du Projet

| Fichier | Description |
| :--- | :--- |
| `AI_Mame.py` | **Cœur de l'IA**. Contient les classes `DQNTrainer`, `DQNModel` (PyTorch), et le `ReplayBuffer`. |
| `invaders.py` | Script principal pour **Space Invaders**. Gère les rewards spécifiques (tuer alien, éviter bombe). |
| `pacman.py` | Script principal pour **Pac-Man**. Gère la lecture de la VRAM (labyrinthe) et des sprites. |
| `invaders_robot.py` | Un bot algorithmique (non-IA) pour Space Invaders, basé sur des règles logiques. |
| `MameCommSocket.py` | Gère le protocole de communication bas niveau avec Lua. |
| `ScreenRecorder.py` | Utilitaire pour capturer l'écran de jeu. |
| `dreamerv2.py` | Implémentation expérimentale de l'algo DreamerV2 (Model-Based RL). |

---

## 🛠️ Installation et Configuration

### Pré-requis
*   Python 3.8+
*   Bibliothèques : `torch`, `numpy`, `matplotlib`, `flask`, `keyboard`, `pygame`, `pywin32`.
*   **MAME** installé avec les ROMs nécessaires (`invaders`, `pacman`).

### Configuration des Chemins
⚠️ **Important** : Les scripts Python contiennent des chemins absolus qu'il faut adapter à votre machine.
Ouvrez `invaders.py` ou `pacman.py` et modifiez la méthode `launch_mame` :

```python
command = [
    r"C:\Chemin\Vers\Votre\mame.exe", # <--- Modifier ici
    "-autoboot_script", r"C:\Chemin\Vers\Plugins\PythonBridgeSocket.lua", # <--- Et ici
    ...
]
```

---

## 🎮 Utilisation

### Lancer un entraînement
Exécutez simplement le script correspondant au jeu :

```bash
python invaders.py
# ou
python pacman.py
```

### Raccourcis Clavier (Pendant l'entraînement)
Le focus doit être sur la fenêtre du terminal/console pour que les touches fonctionnent.

| Touche | Action |
| :--- | :--- |
| **F2** | 🛑 Arrêt propre et sauvegarde du modèle. |
| **F3** | 🐞 Désactiver le mode Debug (console moins verbeuse). |
| **F4** | 🐛 Changer le niveau de Debug (0-3). |
| **F5** | ⏩ Augmenter la vitesse d'émulation (Throttle). |
| **F6** | ⏪ Réduire la vitesse d'émulation. |
| **F7** | 📊 Générer manuellement le graphique des scores. |
| **F8** | 👁️ Afficher ce que l'IA "voit" (Input Frame/State). |
| **F9** | 🔄 Basculer entre mode **Exploration** (Apprentissage) et **Exploitation** (Jeu pur). |
| **F10/F11** | 🎛️ Ajuster manuellement le taux d'exploration (Epsilon) (si NoisyNet inactif). |

---

## 📊 Suivi des Résultats
*   Les logs sont affichés dans la console.
*   Un serveur web local est lancé sur `http://localhost:5000` pour voir les graphiques d'évolution.
*   Les modèles (`.pth`) et les buffers (`.buffer`) sont sauvegardés automatiquement à la racine.

---

<a name="english"></a>
# 🇬🇧 English

This project is a complete framework for training Artificial Intelligence (Reinforcement Learning) agents on classic arcade games (Space Invaders, Pac-Man) via the **MAME** emulator.

It uses a **Client-Server** architecture where Python (the brain) communicates with MAME (the body) via a local TCP socket, allowing frame-by-frame control and direct reading of the game's RAM.

---

## 🚀 Operation & Architecture

The system relies on three main components:

1.  **MAME & Lua Script (`PythonBridgeSocket.lua`)**:
    *   MAME runs the game.
    *   An embedded Lua script acts as a server. It exposes memory addresses (RAM) and listens for input commands.
    *   It synchronizes emulation with the AI via a `wait_for` mechanism (waiting for commands).

2.  **Communication Interface (`MameCommSocket.py`)**:
    *   Handles the raw TCP connection between Python and Lua.
    *   Sends commands (e.g., `execute P1_Button_1(1)`) and receives memory states.

3.  **AI Brain (`AI_Mame.py` & Game Scripts)**:
    *   Implements **Deep Reinforcement Learning** algorithms (Rainbow DQN, DreamerV2).
    *   **`invaders.py` / `pacman.py`**: Game-specific wrappers that define rewards, extract state (pixels or RAM), and manage the training loop.

```mermaid
graph LR
    A[MAME Emulator] -- RAM & Video --> B(Lua Script Server)
    B -- TCP Socket --> C(Python Client)
    C -- Actions (Joy/Btn) --> B
    C -- PyTorch Model --> D[Neural Network]
```

---

## ✨ Key Features

*   **Advanced Algorithms (Rainbow DQN)**:
    *   **Double DQN** & **Dueling DQN** for better value estimation.
    *   **Noisy Nets** for dynamic exploration (replaces Epsilon-Greedy).
    *   **Prioritized Experience Replay (PER)** to learn from important moments.
    *   **N-Step Learning** for longer-term vision.
*   **Multi-Architecture Support**:
    *   **CNN (Convolutional Neural Network)**: The AI "sees" the screen (raw or resized pixels).
    *   **MLP (Multi-Layer Perceptron)**: The AI reads RAM directly (X/Y positions, states).
*   **Visualization Tools**:
    *   Integrated Web Server (Flask) to track score curves in real-time.
    *   Automatic `.png` graph generation.
    *   Video recording of best games.

---

## 📂 Project Structure

| File | Description |
| :--- | :--- |
| `AI_Mame.py` | **AI Core**. Contains `DQNTrainer`, `DQNModel` (PyTorch), and `ReplayBuffer` classes. |
| `invaders.py` | Main script for **Space Invaders**. Handles specific rewards (kill alien, avoid bomb). |
| `pacman.py` | Main script for **Pac-Man**. Handles VRAM reading (maze) and sprites. |
| `invaders_robot.py` | Algorithmic bot (non-AI) for Space Invaders, based on logic rules. |
| `MameCommSocket.py` | Handles low-level communication protocol with Lua. |
| `ScreenRecorder.py` | Utility to capture game screen. |
| `dreamerv2.py` | Experimental implementation of DreamerV2 algo (Model-Based RL). |

---

## 🛠️ Installation and Configuration

### Prerequisites
*   Python 3.8+
*   Libraries: `torch`, `numpy`, `matplotlib`, `flask`, `keyboard`, `pygame`, `pywin32`.
*   **MAME** installed with necessary ROMs (`invaders`, `pacman`).

### Path Configuration
⚠️ **Important**: Python scripts contain absolute paths that must be adapted to your machine.
Open `invaders.py` or `pacman.py` and modify the `launch_mame` method:

```python
command = [
    r"C:\Path\To\Your\mame.exe", # <--- Modify here
    "-autoboot_script", r"C:\Path\To\Plugins\PythonBridgeSocket.lua", # <--- And here
    ...
]
```

---

## 🎮 Usage

### Start Training
Simply run the script corresponding to the game:

```bash
python invaders.py
# or
python pacman.py
```

### Keyboard Shortcuts (During Training)
Focus must be on the terminal/console window for keys to work.

| Key | Action |
| :--- | :--- |
| **F2** | 🛑 Clean stop and model save. |
| **F3** | 🐞 Disable Debug mode (less verbose console). |
| **F4** | 🐛 Change Debug level (0-3). |
| **F5** | ⏩ Increase emulation speed (Throttle). |
| **F6** | ⏪ Decrease emulation speed. |
| **F7** | 📊 Manually generate score graph. |
| **F8** | 👁️ Display what AI "sees" (Input Frame/State). |
| **F9** | 🔄 Toggle between **Exploration** (Learning) and **Exploitation** (Pure play) modes. |
| **F10/F11** | 🎛️ Manually adjust exploration rate (Epsilon) (if NoisyNet inactive). |

---

## 📊 Results Tracking
*   Logs are displayed in the console.
*   A local web server is launched on `http://localhost:5000` to view evolution graphs.
*   Models (`.pth`) and buffers (`.buffer`) are saved automatically at the root.
