"""
Q1 Step 7 [Optional]: ViT-S with partial model freeze + LoRA on frozen part + trainable head.

Strategy:
  - Freeze the first half of ViT encoder blocks
  - Keep second half trainable (fine-tuned)
  - Apply LoRA to the frozen attention weights (Q, K, V)
  - Classification head is always trainable
"""

import os
import argparse
import wandb
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from transformers import ViTForImageClassification
from peft import LoraConfig, get_peft_model, TaskType
from tqdm import tqdm

DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_CLASSES = 100
EPOCHS      = 10
BATCH_SIZE  = 64
LR          = 1e-3
DATA_DIR    = "./data"
WEIGHTS_DIR = "./weights"
os.makedirs(WEIGHTS_DIR, exist_ok=True)


def get_loaders():
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
    tr = datasets.CIFAR100(DATA_DIR, train=True,  download=True, transform=train_tf)
    vl = datasets.CIFAR100(DATA_DIR, train=False, download=True, transform=val_tf)
    return (DataLoader(tr, batch_size=BATCH_SIZE, shuffle=True,  num_workers=4, pin_memory=True),
            DataLoader(vl, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True))


def build_partial_freeze_lora(rank, alpha, dropout, freeze_ratio=0.5):
    """
    Load ViT-S, freeze first `freeze_ratio` of encoder layers,
    apply LoRA to the frozen layers' Q/K/V weights.
    The second half of layers + classifier head remain fully trainable.
    """
    model = ViTForImageClassification.from_pretrained(
        "WinKawaks/vit-small-patch16-224",
        num_labels=NUM_CLASSES,
        ignore_mismatched_sizes=True,
    )
    n_layers   = len(model.vit.encoder.layer)
    freeze_up  = int(n_layers * freeze_ratio)
    print(f"ViT has {n_layers} encoder layers. Freezing first {freeze_up}.")

    # Freeze all parameters first
    for param in model.parameters():
        param.requires_grad = False

    # Unfreeze the second half + classifier
    for i in range(freeze_up, n_layers):
        for param in model.vit.encoder.layer[i].parameters():
            param.requires_grad = True
    for param in model.classifier.parameters():
        param.requires_grad = True

    # Apply LoRA only to the frozen (first half) Q, K, V weights
    # We specify target modules with layer indices
    frozen_targets = [
        f"vit.encoder.layer.{i}.attention.attention.{proj}"
        for i in range(freeze_up)
        for proj in ["query", "key", "value"]
    ]

    cfg = LoraConfig(
        r=rank, lora_alpha=alpha, lora_dropout=dropout,
        target_modules=frozen_targets,
        bias="none",
    )
    model = get_peft_model(model, cfg)

    # Ensure classifier head stays trainable after PEFT wrapping
    for name, param in model.named_parameters():
        if "classifier" in name:
            param.requires_grad = True

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {trainable:,}")
    return model


@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    correct, total = 0, 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        preds = model(imgs).logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total   += labels.size(0)
    return correct / total


def train(model, run_name, rank, alpha, dropout):
    wandb.init(
        project="DLOps-Ass5-Q1-Optional",
        name=run_name,
        config={"rank": rank, "alpha": alpha, "dropout": dropout},
        reinit=True,
    )
    train_loader, val_loader = get_loaders()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                            lr=LR, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    scaler    = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())
    model.to(DEVICE)

    best_val = 0.0
    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss, correct, total = 0.0, 0, 0
        for imgs, labels in tqdm(train_loader, desc=f"Ep{epoch}", leave=False):
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                out  = model(imgs).logits
                loss = criterion(out, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item() * labels.size(0)
            correct    += (out.argmax(1) == labels).sum().item()
            total      += labels.size(0)
        scheduler.step()

        tr_acc = correct / total
        vl_acc = evaluate(model, val_loader)
        print(f"Ep{epoch:2d} | tr_acc={tr_acc:.4f} | vl_acc={vl_acc:.4f}")
        wandb.log({"epoch": epoch, "train_acc": tr_acc, "val_acc": vl_acc})
        best_val = max(best_val, vl_acc)

    print(f"Best val acc: {best_val:.4f}")
    torch.save(model.state_dict(), os.path.join(WEIGHTS_DIR, f"{run_name}.pt"))
    wandb.log({"best_val_acc": best_val})
    wandb.finish()
    return best_val


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rank",        type=int,   default=4)
    parser.add_argument("--alpha",       type=int,   default=8)
    parser.add_argument("--dropout",     type=float, default=0.1)
    parser.add_argument("--freeze_ratio",type=float, default=0.5,
                        help="Fraction of ViT encoder layers to freeze (default: 0.5)")
    parser.add_argument("--wandb_key",   default=None)
    args = parser.parse_args()

    if args.wandb_key:
        wandb.login(key=args.wandb_key)

    run_name = f"partial_freeze_lora_r{args.rank}_a{args.alpha}"
    model    = build_partial_freeze_lora(args.rank, args.alpha, args.dropout, args.freeze_ratio)
    acc      = train(model, run_name, args.rank, args.alpha, args.dropout)
    print(f"\nFinal best val acc: {acc:.4f}")


if __name__ == "__main__":
    main()
