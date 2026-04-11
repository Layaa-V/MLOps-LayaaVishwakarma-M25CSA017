"""
Q1 Testing script: load a saved checkpoint and evaluate on CIFAR-100 test set.
Prints overall accuracy and per-class accuracy table.
"""

Ïimport os
import argparse
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from transformers import ViTForImageClassification
from peft import LoraConfig, get_peft_model, TaskType

DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_CLASSES = 100
DATA_DIR    = "./data"


def get_val_loader(batch_size=64):
    mean = (0.5071, 0.4867, 0.4408)
    std  = (0.2675, 0.2565, 0.2761)
    val_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    ds = datasets.CIFAR100(DATA_DIR, train=False, download=True, transform=val_tf)
    return DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=4)


def load_model(ckpt_path, use_lora=False, rank=4, alpha=8, dropout=0.1):
    base = ViTForImageClassification.from_pretrained(
        "WinKawaks/vit-small-patch16-224",
        num_labels=NUM_CLASSES,
        ignore_mismatched_sizes=True,
    )
    if use_lora:
        cfg = LoraConfig(
            r=rank, lora_alpha=alpha, lora_dropout=dropout,
            target_modules=["query", "key", "value"],
            bias="none",
        )
        model = get_peft_model(base, cfg)
    else:
        model = base
    model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE))
    return model.to(DEVICE)


@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    correct, total = 0, 0
    preds_all, labels_all = [], []
    for imgs, labels in loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        preds = model(imgs).logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total   += labels.size(0)
        preds_all.extend(preds.cpu().numpy())
        labels_all.extend(labels.cpu().numpy())
    return correct / total, np.array(preds_all), np.array(labels_all)


def class_histogram(preds, labels, save_path="class_accuracy.png"):
    class_correct = np.zeros(NUM_CLASSES)
    class_total   = np.zeros(NUM_CLASSES)
    for p, l in zip(preds, labels):
        class_total[l]   += 1
        class_correct[l] += int(p == l)
    class_acc = class_correct / (class_total + 1e-9)
    fig, ax = plt.subplots(figsize=(20, 5))
    ax.bar(range(NUM_CLASSES), class_acc)
    ax.set_xlabel("Class ID")
    ax.set_ylabel("Accuracy")
    ax.set_title("Per-class Test Accuracy")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"Histogram saved → {save_path}")
    plt.close()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt",    required=True, help="Path to .pt checkpoint")
    p.add_argument("--lora",    action="store_true")
    p.add_argument("--rank",    type=int,   default=4)
    p.add_argument("--alpha",   type=int,   default=8)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--hist",    default="class_accuracy.png")
    args = p.parse_args()

    loader = get_val_loader()
    model  = load_model(args.ckpt, args.lora, args.rank, args.alpha, args.dropout)
    acc, preds, labels = evaluate(model, loader)
    print(f"\nOverall Test Accuracy: {acc*100:.2f}%")
    class_histogram(preds, labels, args.hist)


if __name__ == "__main__":
    main()
