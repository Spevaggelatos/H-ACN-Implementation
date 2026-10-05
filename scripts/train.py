import torch
import torch.nn as nn
import torch.optim as optim
import math
import numpy as np
import matplotlib.pyplot as plt
from dataset import get_dataloaders
from model import HACN_Model
from probe import run_offline_probes, plot_encoding_front_timeline

# System Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Experiment Configuration
run_name = "FFW damper = 2"
probe_every_epochs = 24

batch_size = 64
learning_rate = 1e-3
epochs = 96
mode = 'pure_damper'
task_type = 'ntp'

# RHM Configuration
L = 3
s = 2
m = 7
v = 14
n_classes = 2
seed = 42

seq_len = s ** L
vocab_size = v

# Get Dataloaders
train_loader, test_loader, _ = get_dataloaders(
    num_samples=20000, L=L, s=s, m=m, v=vocab_size, num_classes=n_classes, 
    seed=seed, batch_size=batch_size
)

# Initialize global results dictionary
experiment_timeline_results = {run_name: {}}

# Initialize Model and Loss
model = HACN_Model(
    vocab_size=vocab_size, seq_len=seq_len, num_layers=8, 
    mode=mode, task_type=task_type
).to(device)

criterion = nn.CrossEntropyLoss()

# Optimizer and Scheduler
optimizer = optim.AdamW(model.parameters(), lr=learning_rate, betas=(0.9, 0.95), weight_decay=0.01)
total_steps = epochs * len(train_loader)
warmup_steps = int(0.05 * total_steps)

def lr_lambda(step):
    """Learning rate scheduler with warmup and cosine decay."""
    if step < warmup_steps:
        return float(step) / float(max(1, warmup_steps))
    progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
    return 0.5 * (1.0 + math.cos(math.pi * progress))

scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

# Metrics tracking
accuracy_history = []
alphas_history = []

print(f"\n--- Starting {task_type.upper()} Training for {mode.upper()} mode ---")

for epoch in range(epochs):
    model.train()
    total_loss = 0.0
    
    for batch_x, batch_y in train_loader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)

        optimizer.zero_grad()
        outputs = model(batch_x)

        # Main Task Loss Calculation
        if task_type == 'classification':
            loss = criterion(outputs["main_logits"], batch_y)
        elif task_type == 'ntp':
            targets = torch.cat([batch_x[:, 1:], batch_y.unsqueeze(1)], dim=1).contiguous()
            loss = criterion(outputs["main_logits"].view(-1, vocab_size), targets.view(-1))

        # Backward pass updates main parameters
        loss.backward() 

        # Gradient Clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        optimizer.step()
        scheduler.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)

    # Evaluation Phase
    model.eval()
    main_correct, main_total = 0, 0

    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            outputs = model(batch_x)

            # Main Task Accuracy
            if task_type == 'classification':
                m_preds = torch.argmax(outputs["main_logits"], dim=-1)
                main_correct += (m_preds == batch_y).sum().item()
                main_total += batch_y.size(0)
            elif task_type == 'ntp':
                shifted_logits = outputs["main_logits"][:, :-1, :].contiguous()
                shifted_labels = batch_x[:, 1:].contiguous()
                m_preds = torch.argmax(shifted_logits, dim=-1)
                main_correct += (m_preds == shifted_labels).sum().item()
                main_total += shifted_labels.numel()

    main_acc = main_correct / main_total
    accuracy_history.append(main_acc)

    _, ffw_alphas = model.get_alphas()
    alphas_history.append(ffw_alphas)

    print(f"Epoch {epoch+1} | Loss: {avg_loss:.4f} | Main ({task_type}) Acc: {main_acc*100:.1f}%")

    # Periodic Offline Probing
    if task_type == 'ntp':
        if (epoch + 1) % probe_every_epochs == 0 or (epoch + 1) == epochs:
            print(f"  --> Running Periodic Offline Probes (Epoch {epoch+1})...")
            p_accs = run_offline_probes(model, test_loader, device, task_type)
            experiment_timeline_results[run_name][epoch + 1] = p_accs
            
            probes_str = " | ".join([f"L{i+1}: {acc:.1f}%" for i, acc in enumerate(p_accs)])
            print(f"      Probes: {probes_str}")

if mode in ["h-acn", "hybrid"]:
    formatted_alphas = " | ".join([f"{a:.3f}" for a in alphas_history[-1]])
    print(f"\nFinal FFW Alphas: {formatted_alphas}")

# Save the final probing plot
if task_type == 'ntp':
    plot_encoding_front_timeline(run_name, experiment_timeline_results)
    print("Encoding front plot saved to encoding_front.png")
