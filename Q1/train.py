# -*- coding: utf-8 -*-
"""
Q1: ViT-S Fine-tuning on CIFAR-100 with and without LoRA
- Finetune classification head only (no LoRA baseline)
- Finetune with LoRA (PEFT) for various ranks and alphas
- Logs to WandB, saves best model weights
"""

import os
import argparse
import wandb
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from transformers import ViTForImageClassification, ViTConfig
from peft import LoraConfig, get_peft_model, TaskType
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# ── Config ────────────────────────────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_CLASSES = 100
EPOCHS = 10
BATCH_SIZE = 64
LR = 1e-3
DATA_DIR = "./data"
WEIGHTS_DIR = "./weights"
os.makedirs(WEIGHTS_DIR, exist_ok=True)


# ── Data ──────────────────────────────────────────────────────────────────────
def get_loaders(batch_size=BATCH_SIZE):
    mean = (0.5071, 0.4867, 0.4408)
    std  = (0.2675, 0.2565, 0.2761)
    train_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(224, padding=16),
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
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    return train_loader, val_loader


# ── Model helpers ─────────────────────────────────────────────────────────────
def load_vit_baseline():
    """ViT-S pretrained on ImageNet, only classification head trainable."""
    model = ViTForImageClassification.from_pretrained(
        "WinKawaks/vit-small-patch16-224",
        num_labels=NUM_CLASSES,
        ignore_mismatched_sizes=True,
    )
    # Freeze everything except the head
    for name, param in model.named_parameters():
        if "classifier" not in name:
            param.requires_grad = False
    return model


def load_vit_lora(rank, alpha, dropout, target_modules=None):
    """ViT-S with LoRA applied to Q, K, V attention projections."""
    if target_modules is None:
        target_modules = ["query", "key", "value"]

    base = ViTForImageClassification.from_pretrained(
        "WinKawaks/vit-small-patch16-224",
        num_labels=NUM_CLASSES,
        ignore_mismatched_sizes=True,
    )
    lora_cfg = LoraConfig(
        r=rank,
        lora_alpha=alpha,
        lora_dropout=dropout,
        target_modules=target_modules,
        bias="none",
    )
    model = get_peft_model(base, lora_cfg)
    # Ensure classification head is also trainable
    for name, param in model.named_parameters():
        if "classifier" in name:
            param.requires_grad = True
    return model


def count_trainable(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ── Training / evaluation ──────────────────────────────────────────────────────
def train_epoch(model, loader, optimizer, criterion, scaler):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    lora_grad_norms = []

    for imgs, labels in tqdm(loader, leave=False):
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
            out = model(imgs).logits
            loss = criterion(out, labels)
        scaler.scale(loss).backward()

        # Track gradient norms on LoRA weights
        for name, param in model.named_parameters():
            if "lora" in name and param.grad is not None:
                lora_grad_norms.append(param.grad.norm().item())

        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item() * labels.size(0)
        preds = out.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total   += labels.size(0)

    avg_grad_norm = float(np.mean(lora_grad_norms)) if lora_grad_norms else 0.0
    return total_loss / total, correct / total, avg_grad_norm


@torch.no_grad()
def eval_epoch(model, loader, criterion):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []

    for imgs, labels in tqdm(loader, leave=False):
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        out  = model(imgs).logits
        loss = criterion(out, labels)
        total_loss += loss.item() * labels.size(0)
        preds = out.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total   += labels.size(0)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    return total_loss / total, correct / total, np.array(all_preds), np.array(all_labels)


def class_accuracy_histogram(all_preds, all_labels, run_name):
    """Per-class test accuracy bar chart logged to WandB."""
    class_correct = np.zeros(NUM_CLASSES)
    class_total   = np.zeros(NUM_CLASSES)
    for p, l in zip(all_preds, all_labels):
        class_total[l]   += 1
        class_correct[l] += int(p == l)
    class_acc = class_correct / (class_total + 1e-9)

    fig, ax = plt.subplots(figsize=(20, 5))
    ax.bar(range(NUM_CLASSES), class_acc)
    ax.set_xlabel("Class ID")
    ax.set_ylabel("Accuracy")
    ax.set_title(f"Per-class Test Accuracy – {run_name}")
    plt.tight_layout()
    return fig


# ── Main training loop ─────────────────────────────────────────────────────────
def run_experiment(model, run_name, lora_params=None, save_best=False):
    """Train model, log to WandB, return best val accuracy."""
    wandb.init(
        project="DLOps-Ass5-Q1",
        name=run_name,
        config=lora_params or {"mode": "baseline"},
        reinit=True,
    )

    train_loader, val_loader = get_loaders()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=LR, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    scaler    = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())

    model.to(DEVICE)
    best_val_acc, best_epoch = 0.0, 0
    print(f"\n{'='*60}")
    print(f"  Run: {run_name}  |  Trainable params: {count_trainable(model):,}")
    print(f"{'='*60}")

    for epoch in range(1, EPOCHS + 1):
        tr_loss, tr_acc, grad_norm = train_epoch(model, train_loader, optimizer, criterion, scaler)
        vl_loss, vl_acc, _, _     = eval_epoch(model, val_loader, criterion)
        scheduler.step()

        print(f"Epoch {epoch:2d}/{EPOCHS} | "
              f"tr_loss={tr_loss:.4f} tr_acc={tr_acc:.4f} | "
              f"vl_loss={vl_loss:.4f} vl_acc={vl_acc:.4f} | "
              f"lora_grad_norm={grad_norm:.4f}")

        log = {
            "epoch": epoch,
            "train/loss": tr_loss, "train/accuracy": tr_acc,
            "val/loss":   vl_loss, "val/accuracy":   vl_acc,
            "lora_grad_norm": grad_norm,
        }
        wandb.log(log)

        if vl_acc > best_val_acc:
            best_val_acc = vl_acc
            best_epoch   = epoch
            if save_best:
                ckpt = os.path.join(WEIGHTS_DIR, f"{run_name}_best.pt")
                torch.save(model.state_dict(), ckpt)
                print(f"  ✓ Saved best model → {ckpt}")

    # Per-class accuracy on test set
    _, test_acc, all_preds, all_labels = eval_epoch(model, val_loader, criterion)
    fig = class_accuracy_histogram(all_preds, all_labels, run_name)
    wandb.log({"class_accuracy_histogram": wandb.Image(fig)})
    plt.close(fig)

    print(f"  Best val acc: {best_val_acc:.4f} at epoch {best_epoch}")
    wandb.log({"best_val_accuracy": best_val_acc, "test_accuracy": test_acc,
               "trainable_params": count_trainable(model)})
    wandb.finish()
    return best_val_acc, test_acc


# ── CLI ────────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="Q1: ViT-S CIFAR-100 LoRA experiments")
    p.add_argument("--mode", choices=["baseline", "lora", "all"], default="all",
                   help="Which experiments to run")
    p.add_argument("--rank",    type=int,   default=None, help="Single LoRA rank (used with --mode lora)")
    p.add_argument("--alpha",   type=int,   default=None, help="Single LoRA alpha (used with --mode lora)")
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--wandb_key", type=str, default=None, help="WandB API key")
    return p.parse_args()


def main():
    args = parse_args()
    if args.wandb_key:
        wandb.login(key=args.wandb_key)

    results = []  # (run_name, rank, alpha, dropout, lora, test_acc, trainable)

    # ── Baseline ─────────────────────────────────────────────────────────────
    if args.mode in ("baseline", "all"):
        model = load_vit_baseline()
        val_acc, test_acc = run_experiment(model, "baseline_no_lora", save_best=True)
        results.append(("baseline", "—", "—", "—", "No", test_acc, count_trainable(model)))

    # ── LoRA grid ────────────────────────────────────────────────────────────
    if args.mode in ("lora", "all"):
        ranks   = [args.rank]   if args.rank  else [2, 4, 8]
        alphas  = [args.alpha]  if args.alpha else [2, 4, 8]
        dropout = args.dropout

        exp_no = 1
        for rank in ranks:
            for alpha in alphas:
                run_name = f"lora_r{rank}_a{alpha}_d{dropout}"
                print(f"\nExperiment {exp_no}: rank={rank}, alpha={alpha}, dropout={dropout}")
                model = load_vit_lora(rank, alpha, dropout)
                lora_params = {"rank": rank, "alpha": alpha, "dropout": dropout,
                               "experiment_no": exp_no}
                val_acc, test_acc = run_experiment(
                    model, run_name, lora_params=lora_params,
                    save_best=(exp_no == 1)  # save first to have something; optuna picks best
                )
                results.append((run_name, rank, alpha, dropout, "Yes", test_acc, count_trainable(model)))
                exp_no += 1

    # ── Summary table ─────────────────────────────────────────────────────────
    print("\n\n" + "="*80)
    print(f"{'LoRA':^12}{'Rank':^8}{'Alpha':^8}{'Dropout':^10}{'Test Acc':^12}{'Trainable Params':^20}")
    print("="*80)
    for row in results:
        print(f"{row[4]:^12}{str(row[1]):^8}{str(row[2]):^8}{str(row[3]):^10}{row[5]:^12.4f}{row[6]:^20,}")


if __name__ == "__main__":
    main()
