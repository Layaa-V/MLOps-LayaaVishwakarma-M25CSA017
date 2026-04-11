"""
Q1 Step 5: Optuna hyperparameter search for LoRA on ViT-S / CIFAR-100
Searches rank, alpha keeping dropout fixed at 0.1.
"""

import os
import argparse
import wandb
import optuna
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from peft import LoraConfig, get_peft_model, TaskType
from transformers import ViTForImageClassification
from tqdm import tqdm

DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_CLASSES = 100
EPOCHS_OPT  = 5          # fewer epochs during search to save time
BATCH_SIZE  = 64
LR          = 1e-3
DATA_DIR    = "./data"
WEIGHTS_DIR = "./weights"
os.makedirs(WEIGHTS_DIR, exist_ok=True)


def get_loaders(batch_size=BATCH_SIZE, subset_frac=1.0):
    mean = (0.5071, 0.4867, 0.4408)
    std  = (0.2675, 0.2565, 0.2761)
    train_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    val_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    train_ds = datasets.CIFAR100(DATA_DIR, train=True,  download=True, transform=train_tf)
    val_ds   = datasets.CIFAR100(DATA_DIR, train=False, download=True, transform=val_tf)

    if subset_frac < 1.0:
        n = int(len(train_ds) * subset_frac)
        train_ds = Subset(train_ds, list(range(n)))

    return (DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=4, pin_memory=True),
            DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True))


def build_model(rank, alpha, dropout):
    base = ViTForImageClassification.from_pretrained(
        "WinKawaks/vit-small-patch16-224",
        num_labels=NUM_CLASSES,
        ignore_mismatched_sizes=True,
    )
    cfg = LoraConfig(
        r=rank, lora_alpha=alpha, lora_dropout=dropout,
        target_modules=["query", "key", "value"],
        bias="none",
    )
    model = get_peft_model(base, cfg)
    for name, param in model.named_parameters():
        if "classifier" in name:
            param.requires_grad = True
    return model


def train_and_eval(model, train_loader, val_loader, epochs=EPOCHS_OPT):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                            lr=LR, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    scaler    = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())
    model.to(DEVICE)
    best_val = 0.0

    for epoch in range(1, epochs + 1):
        model.train()
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                loss = criterion(model(imgs).logits, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        scheduler.step()

        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
                preds = model(imgs).logits.argmax(dim=1)
                correct += (preds == labels).sum().item()
                total   += labels.size(0)
        val_acc = correct / total
        best_val = max(best_val, val_acc)

    return best_val


def objective(trial):
    rank    = trial.suggest_categorical("rank",  [2, 4, 8, 16])
    alpha   = trial.suggest_categorical("alpha", [2, 4, 8, 16, 32])
    dropout = trial.suggest_float("dropout", 0.0, 0.3, step=0.05)

    train_loader, val_loader = get_loaders(subset_frac=0.5)  # 50% data for speed
    model   = build_model(rank, alpha, dropout)
    val_acc = train_and_eval(model, train_loader, val_loader)

    wandb.log({"rank": rank, "alpha": alpha, "dropout": dropout, "val_accuracy": val_acc})
    return val_acc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_trials",  type=int, default=20)
    parser.add_argument("--wandb_key", type=str, default=None)
    args = parser.parse_args()

    if args.wandb_key:
        wandb.login(key=args.wandb_key)
    wandb.init(project="DLOps-Ass5-Q1-Optuna", name="optuna_search")

    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=args.n_trials, show_progress_bar=True)

    best = study.best_trial
    print(f"\nBest trial: val_acc={best.value:.4f}")
    print(f"  rank={best.params['rank']}, alpha={best.params['alpha']}, dropout={best.params['dropout']}")

    wandb.log({"best_val_accuracy": best.value, **best.params})
    wandb.finish()

    # ── Retrain best config for full epochs and save ───────────────────────
    print("\nRetraining best config for full epochs …")
    wandb.init(project="DLOps-Ass5-Q1", name="best_optuna_model")
    train_loader, val_loader = get_loaders()
    model   = build_model(best.params["rank"], best.params["alpha"], best.params["dropout"])
    val_acc = train_and_eval(model, train_loader, val_loader, epochs=10)
    print(f"Full-train val accuracy: {val_acc:.4f}")
    ckpt = os.path.join(WEIGHTS_DIR, "best_optuna_model.pt")
    torch.save(model.state_dict(), ckpt)
    print(f"Saved → {ckpt}")
    wandb.log({"final_val_accuracy": val_acc})
    wandb.finish()


if __name__ == "__main__":
    main()
