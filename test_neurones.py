import torch
import matplotlib.pyplot as plt
import numpy as np
from collections import deque
from AI_Mame import TrainingConfig, DQNTrainer, DQNModel
from invaders import StateExtractor
import matplotlib
matplotlib.use("TkAgg")

# --- Configuration du modèle (identique à ton entraînement)
config = TrainingConfig(
    state_history_size=2,
    input_size=(2,96,88),
    output_size=6,
    hidden_layers=1,
    hidden_size=512,
    learning_rate=0.00025,
    gamma=0.99,
    use_noisy=True,
    epsilon_start=1.0,
    epsilon_end=0.1,
    epsilon_linear=0,
    epsilon_decay=(0.1 / 1.0) ** (1 / 1_000_000),
    epsilon_add=0.0005,
    buffer_capacity=20_000,
    batch_size=32,
    device="cuda" if torch.cuda.is_available() else "cpu",
    double_dqn=True,
    prioritized_replay=True,
    model_type="cnn",
    cnn_type="deepmind",
    state_extractor=StateExtractor("cnn", True, False, None, True,[False]*11),
    mode="exploration"
)

# --- Initialisation du modèle + buffer
trainer = DQNTrainer(config)
trainer.load_model("invaders.pth")
trainer.load_buffer("invaders.buffer")

# --- Extraction d'un vrai batch depuis le buffer
sample_size = min(128, len(trainer.replay_buffer))
states, *_ = trainer.replay_buffer.sample(sample_size)
input_tensor = states.view(sample_size, *config.input_size).to(config.device)

# --- Hook d'observation des activations + stats poids fc1
model = trainer.dqn
model_layers = list(model.encoder) + [model.fc1]
if hasattr(model, 'bn_fc1'):
    model_layers.append(model.bn_fc1)
if hasattr(model, 'relu_fc1'):
    model_layers.append(model.relu_fc1)
model_layers.append(model.fc2)

for i, layer in enumerate(model_layers):
    print(f"Layer {i}: {layer}")

# Affichage stats poids fc1
print("\n📊 Poids fc1:")

if hasattr(model, "fc1"):
    print("📊 Couche fc1:")
    if hasattr(model.fc1, "mu_weight"):
        print(" - Moyenne mu_weight:", model.fc1.mu_weight.mean().item())
        print(" - Écart-type sigma_weight:", model.fc1.sigma_weight.std().item())
        try:
            poids_bruites = model.fc1.mu_weight + model.fc1.sigma_weight * model.fc1.epsilon_weight
            print(" - Moyenne poids bruités:", poids_bruites.mean().item())
        except AttributeError:
            print(" - ⚠️ Bruit non échantillonné : appelle forward() d'abord pour générer epsilon.")
    elif hasattr(model.fc1, "weight"):
        print(" - Moyenne poids:", model.fc1.weight.mean().item())
    else:
        print(" - ⚠️ Aucun attribut reconnu dans fc1.")

if hasattr(model, "fc2"):
    print("📊 Couche fc2:")
    if hasattr(model.fc2, "mu_weight"):
        print(" - Moyenne mu_weight:", model.fc2.mu_weight.mean().item())
        print(" - Écart-type sigma_weight:", model.fc2.sigma_weight.std().item())
        try:
            poids_bruites = model.fc2.mu_weight + model.fc2.sigma_weight * model.fc2.epsilon_weight
            print(" - Moyenne poids bruités:", poids_bruites.mean().item())
        except AttributeError:
            print(" - ⚠️ Bruit non échantillonné : appelle forward() d'abord pour générer epsilon.")
    elif hasattr(model.fc2, "weight"):
        print(" - Moyenne poids:", model.fc2.weight.mean().item())
    else:
        print(" - ⚠️ Aucun attribut reconnu dans fc2.")

activations = []

def hook_fn(name):
    def hook(module, input, output):
        activations.append((name, output.detach().cpu()))
    return hook

hooks = []
hooks.append(model.fc1.register_forward_hook(hook_fn("fc1")))
if hasattr(model, 'bn_fc1'):
    hooks.append(model.bn_fc1.register_forward_hook(hook_fn("bn_fc1")))
if hasattr(model, 'relu_fc1'):
    hooks.append(model.relu_fc1.register_forward_hook(hook_fn("relu_fc1")))
hooks.append(model.fc2.register_forward_hook(hook_fn("fc2")))

# --- Forward pass
with torch.no_grad():
    model(input_tensor)

# --- Analyse et visualisation
for name, act in activations:
    morts = (act == 0).all(dim=0).sum().item()
    total = act.shape[1]
    pourcentage_morts = 100 * morts / total
    moyenne = act.mean().item()
    stddev = act.std().item()
    sparsity = (act < 1e-5).float().mean().item() * 100

    print(f"\n🧠 {name}")
    print(f" - Neurones morts : {morts}/{total} ({pourcentage_morts:.2f}%)")
    print(f" - Moyenne activation : {moyenne:.4f}")
    print(f" - Écart-type : {stddev:.4f}")
    print(f" - Sparsité (valeurs < 1e-5) : {sparsity:.1f}%")

    plt.figure(figsize=(6, 3))
    plt.hist(act.flatten().numpy(), bins=50, color='steelblue')
    plt.title(f"Histogramme des activations - {name}")
    plt.xlabel("Activation")
    plt.ylabel("Fréquence")
    plt.grid(True)

plt.tight_layout()
plt.show()

# Nettoyage
for h in hooks:
    h.remove()