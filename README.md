# 🕹️ Mame-AI: Ape-X Deep Reinforcement Learning for Arcade Games

Mame-AI est un framework d'entraînement d'intelligence artificielle conçu pour les jeux d'arcade classiques émulés via MAME. Il utilise des architectures de pointe (Ape-X, Rainbow DQN) pour permettre un apprentissage distribué et performant.

![Aesthetics](https://img.shields.io/badge/Aesthetics-Premium-blueviolet)
![Framework](https://img.shields.io/badge/Framework-PyTorch-orange)
![Games](https://img.shields.io/badge/Games-PacMan%20%7C%20Invaders-yellow)

---

## 🚀 Fonctionnalités Clés

- **Architecture Ape-X Distribuée** : Utilisation de plusieurs acteurs (`NUM_ACTORS`) collectant des expériences en parallèle, envoyées à un `Learner` centralisé sur GPU.
- **Rainbow DQN** : Intégration de NoisyNets (exploration), Prioritized Replay Buffer, Double DQN, Dueling Networks et N-Step Learning.
- **Pont MAME Haute Performance** : Communication via Socket entre Python et MAME (script Lua) pour une lecture mémoire et une injection de commandes ultra-rapide.
- **Dashboard Dynamique** : Analyse en temps réel des performances (moyennes mobiles, projections polynomiales, records, saturation des queues).
- **Extraction CNN Précise** : Mapping direct de la VRAM (Vidéo RAM) et des Sprites matériels en tenseurs pour le réseau de neurones.

---

## 🛠️ Installation

### 1. Prérequis
- Python 3.8+
- [MAME](https://www.mamedev.org/) (installé et configuré)
- Les ROMs correspondantes (`pacman`, `invaders`) dans votre dossier MAME.

### 2. Configuration de l'environnement
Clonez le dépôt et installez les dépendances :
```bash
git clone https://github.com/bacobaco/mame-ai.git
cd mame-ai
pip install -r core/requirements.txt
```

### 3. Ajustement des chemins
Modifiez les variables `MAME_PATH` dans les scripts de lancement (`pacman/pacman_multi.py` par exemple) pour pointer vers votre exécutable MAME.

---

## 🎮 Utilisation

### Entraîner Pac-Man (Multi-Agent)
Pour lancer l'orchestrateur Ape-X avec 4 acteurs en parallèle :
```bash
python pacman/pacman_multi.py
```

### Lancer le Dashboard de Monitoring
Pour visualiser les progrès en temps réel (graphiques, prédictions, records) :
```bash
python analysis/plot_mean_pacman.py
```
Le dashboard génère des graphiques haute résolution dans le dossier `media/`.

---

## 📁 Structure du Projet

- `core/` : Cœur logique (DQN, Buffer, Communication Socket, API MAME).
- `pacman/` : Scripts spécifiques à Pac-Man (Interface mémoire, Training, Logic).
- `invaders/` : Scripts spécifiques à Space Invaders.
- `analysis/` : Scripts de monitoring et de génération de graphiques.
- `media/` : Visuels, graphiques et sauvegardes de frames.

---

## 🧠 Spécifications Techniques

- **Entrées** : Stack de frames (64x56) extraites directement de la VRAM.
- **Modèle** : CNN "Precise" avec couches Dueling et Noisy Linear.
- **Optimiseur** : Adam avec un Learning Rate dynamique ajustable.
- **Persistence** : Sauvegarde automatique du modèle (`.pth`) et du Replay Buffer (`.buffer`) toutes les sessions pour une reprise sans perte.

---

## 📝 Licence
Projet libre d'utilisation à des fins éducatives et de recherche en IA.
