import torch
import matplotlib.pyplot as plt
from AI_Mame import TrainingConfig, DQNTrainer, NoisyLinear
import numpy as np

config = TrainingConfig(
    state_history_size=3,
    input_size=13,
    output_size=6,
    hidden_layers=2,
    hidden_size=128,
    learning_rate=0.001,
    gamma=0.9999,
    epsilon_start=1.0,
    epsilon_end=0.1,
    epsilon_linear=(1-0.1) / 1_000_000,
    epsilon_decay=(0.1 / 1.0) ** (1 / 1_000_000),
    epsilon_add=0.0005,
    use_noisy=True,
    buffer_capacity=500000,
    batch_size=64,
    double_dqn=True,
    dueling=True,
    prioritized_replay=False,
    model_type="mlp",  # ou "cnn"
    cnn_type="deepmind",
    state_extractor=None,
    device="cuda" if torch.cuda.is_available() else "cpu",
    mode="exploration"
)

trainer = DQNTrainer(config)
trainer.load_model("invaders_mlp_double=True_N=3_i=13_hl=2,128_batch=1000000,32_l=6.25e-05_g=0.99[NoisyNet]_nb=16426_ms=618_hs=1240.pth")
model = trainer.dqn.eval()

# Dummy batch
if config.model_type == "mlp":
    input_tensor = torch.randn(128, config.input_size * config.state_history_size).to(config.device)
else:
    input_tensor = torch.randn(128, *config.input_size).to(config.device)

# Hook des activations
activations = {}
hooks = []

def hook_fn(name):
    def hook(_, __, output):
        activations[name] = output.detach().cpu()
    return hook

# Enregistrement des hooks selon le type de modèle
if config.model_type == "mlp":
    for i, layer in enumerate(model.mlp_layers):
        name = f"mlp_fc{i+1}"
        hooks.append(layer.register_forward_hook(hook_fn(name)))
elif config.model_type == "cnn":
    if hasattr(model, "fc1"):
        hooks.append(model.fc1.register_forward_hook(hook_fn("fc1")))
else:
    raise ValueError("Type de modèle non pris en charge")

# Forward pass
with torch.no_grad():
    model(input_tensor)

# Analyse des activations
for name, act in activations.items():
    morts = (act == 0).all(dim=0).sum().item()
    total = act.shape[1]
    avg = act.mean().item()
    std = act.std().item()
    sparsity = (act < 1e-5).float().mean().item() * 100

    print(f"\n🧠 Activations de {name}")
    print(f" - Neurones morts : {morts}/{total} ({100 * morts / total:.2f}%)")
    print(f" - Moyenne : {avg:.4f} | Écart-type : {std:.4f}")
    print(f" - Sparsité (<1e-5) : {sparsity:.1f}%")

    if morts / total > 0.5:
        print(f" ⚠️ Plus de 50% des neurones sont morts dans {name}")
    elif morts == 0:
        print(f" ✅ Aucun neurone mort")

    if sparsity > 80:
        print(f" 🔸 Sparsité très forte : activations quasi nulles")
    elif sparsity < 10:
        print(f" 🔹 Faible sparsité : couche très active")

    # Histogramme
    plt.figure(figsize=(6, 3))
    plt.hist(act.flatten().numpy(), bins=50, color='steelblue')
    plt.title(f"Histogramme des activations - {name}")
    plt.xlabel("Activation")
    plt.ylabel("Fréquence")
    plt.grid(True)

plt.tight_layout()
plt.show()

# Nettoyage des hooks
for h in hooks:
    h.remove()
