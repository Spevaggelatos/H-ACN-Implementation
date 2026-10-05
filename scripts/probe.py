import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.cm as cm

def run_offline_probes(model, test_loader, device, task_type):
    """Trains and evaluates linear probes on the intermediate representations."""
    model.eval()
    layer_hiddens = {i: [] for i in range(8)}
    root_labels = []

    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x = batch_x.to(device)
            outputs = model(batch_x)

            for i, h in enumerate(outputs["hidden_states"]):
                target_repr = h[:, 0, :] if task_type == 'classification' else h[:, -1, :]
                layer_hiddens[i].append(target_repr.cpu())
            root_labels.append(batch_y.cpu())

    root_labels = torch.cat(root_labels, dim=0).to(device)
    for i in range(8):
        layer_hiddens[i] = torch.cat(layer_hiddens[i], dim=0).to(device)

    # Train / Validation Split
    N = root_labels.size(0)
    probe_val_frac = 0.2
    n_val = max(1, int(N * probe_val_frac))

    g = torch.Generator(device=device).manual_seed(42)
    perm = torch.randperm(N, generator=g, device=device)
    val_idx, train_idx = perm[:n_val], perm[n_val:]

    probe_accuracies = []

    # Train and evaluation of probes
    for layer_idx in range(8):
        X = layer_hiddens[layer_idx]
        y = root_labels

        # Split the data of the specific layer
        X_train, y_train = X[train_idx], y[train_idx]
        X_val, y_val = X[val_idx], y[val_idx]

        probe = nn.Sequential(
            nn.Linear(128, 256), nn.GELU(), nn.Linear(256, 2)
        ).to(device)

        probe_opt = optim.AdamW(probe.parameters(), lr=0.002, weight_decay=1e-4)
        probe_criterion = nn.CrossEntropyLoss()

        # Train on train split specifically
        probe.train()
        for _ in range(200):
            probe_opt.zero_grad()
            loss = probe_criterion(probe(X_train), y_train)
            loss.backward()
            probe_opt.step()

        # Eval specifically on VALIDATION SPLIT
        probe.eval()
        with torch.no_grad():
            preds = probe(X_val).argmax(dim=-1)
            acc = (preds == y_val).float().mean().item()
            probe_accuracies.append(acc * 100)

    return probe_accuracies

def plot_encoding_front_timeline(run_name, results_timeline):
    """Plots the accuracy of probes over different layers and epochs."""
    plt.figure(figsize=(10, 6))
    layers = np.arange(1, 9)

    # Take the data from the specific epoch and classify them for every epoch
    epochs_recorded = sorted(results_timeline[run_name].keys())
    max_epoch = max(epochs_recorded)

    colormap = cm.get_cmap('viridis')

    for epoch in epochs_recorded:
        accuracies = results_timeline[run_name][epoch]

        # Calculate color: As closer as we are to max epoch, intense color
        color_intensity = 0.3 + 0.7 * (epoch / max_epoch)
        color = colormap(color_intensity)

        plt.plot(layers, accuracies, marker='o', linestyle='-',
                 markersize=7, linewidth=2, color=color,
                 label=f'Epoch {epoch}')

    plt.title(f'Encoding Front Timeline: {run_name}', fontsize=14, pad=15, fontweight='bold')
    plt.xlabel('Transformer Layer', fontsize=12, fontweight='bold')
    plt.ylabel('Root-MLP Accuracy (%)', fontsize=12, fontweight='bold')
    plt.ylim(0, 105)
    plt.xticks(layers)
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.legend(title="Training Progress", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig('encoding_front.png') # Saving instead of displaying for scripts
