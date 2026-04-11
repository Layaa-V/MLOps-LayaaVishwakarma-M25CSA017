"""
Q2(i): FGSM Attack – From Scratch vs IBM ART
- Train ResNet18 from scratch on CIFAR-10 (≥ 72% accuracy)
- Implement FGSM manually
- Implement FGSM via IBM ART
- Compare clean vs adversarial accuracy, visualise samples on WandB
"""

import os
import argparse
import wandb
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from tqdm import tqdm

# IBM ART
from art.attacks.evasion import FastGradientMethod
from art.estimators.classification import PyTorchClassifier
from art.defences.preprocessor import FeatureSqueezing   # not used here, kept for ref

DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_CLASSES = 10
DATA_DIR    = "./data"
WEIGHTS_DIR = "./weights"
os.makedirs(WEIGHTS_DIR, exist_ok=True)

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD  = (0.2023, 0.1994, 0.2010)
CIFAR10_CLASSES = ["airplane","automobile","bird","cat","deer",
                   "dog","frog","horse","ship","truck"]


# ── Data ──────────────────────────────────────────────────────────────────────
def get_loaders(batch_size=128):
    train_tf = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])
    test_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])
    train_ds = datasets.CIFAR10(DATA_DIR, train=True,  download=True, transform=train_tf)
    test_ds  = datasets.CIFAR10(DATA_DIR, train=False, download=True, transform=test_tf)
    return (DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=4, pin_memory=True),
            DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True))


# ── Model ─────────────────────────────────────────────────────────────────────
def build_resnet18():
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
    return model


# ── Training ──────────────────────────────────────────────────────────────────
def train_clean(model, train_loader, val_loader, epochs=50):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    scaler    = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())
    model.to(DEVICE)

    best_acc, best_ckpt = 0.0, os.path.join(WEIGHTS_DIR, "resnet18_cifar10_best.pt")
    for epoch in range(1, epochs + 1):
        model.train()
        for imgs, labels in tqdm(train_loader, desc=f"Ep{epoch}", leave=False):
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                loss = criterion(model(imgs), labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        scheduler.step()

        acc = evaluate_clean(model, val_loader)
        print(f"Epoch {epoch:3d} | val_acc={acc:.4f}")
        wandb.log({"epoch": epoch, "val_acc_clean": acc})
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), best_ckpt)
    print(f"Best clean test accuracy: {best_acc:.4f}")
    return best_ckpt


@torch.no_grad()
def evaluate_clean(model, loader):
    model.eval()
    correct, total = 0, 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        preds = model(imgs).argmax(dim=1)
        correct += (preds == labels).sum().item()
        total   += labels.size(0)
    return correct / total


# ── FGSM from scratch ─────────────────────────────────────────────────────────
def fgsm_scratch(model, imgs, labels, epsilon, criterion):
    """Returns adversarial examples using FGSM implemented manually."""
    imgs   = imgs.clone().requires_grad_(True)
    output = model(imgs)
    loss   = criterion(output, labels)
    model.zero_grad()
    loss.backward()
    adv = imgs + epsilon * imgs.grad.sign()
    # Clip to valid normalised range
    for c in range(3):
        adv[:, c] = adv[:, c].clamp(
            (0 - CIFAR10_MEAN[c]) / CIFAR10_STD[c],
            (1 - CIFAR10_MEAN[c]) / CIFAR10_STD[c],
        )
    return adv.detach()


def eval_fgsm_scratch(model, loader, epsilon):
    model.eval()
    criterion = nn.CrossEntropyLoss()
    correct, total = 0, 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        adv   = fgsm_scratch(model, imgs, labels, epsilon, criterion)
        preds = model(adv).argmax(dim=1)
        correct += (preds == labels).sum().item()
        total   += labels.size(0)
    return correct / total


# ── FGSM via IBM ART ──────────────────────────────────────────────────────────
def build_art_classifier(model):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)   # dummy; not used for inference
    art_clf   = PyTorchClassifier(
        model=model,
        loss=criterion,
        optimizer=optimizer,
        input_shape=(3, 32, 32),
        nb_classes=NUM_CLASSES,
        clip_values=(
            min((0 - m) / s for m, s in zip(CIFAR10_MEAN, CIFAR10_STD)),
            max((1 - m) / s for m, s in zip(CIFAR10_MEAN, CIFAR10_STD)),
        ),
        device_type="gpu" if torch.cuda.is_available() else "cpu",
    )
    return art_clf


def eval_fgsm_art(art_clf, loader, epsilon):
    attack = FastGradientMethod(estimator=art_clf, eps=epsilon, batch_size=128)
    correct, total = 0, 0
    for imgs, labels in loader:
        x_np = imgs.numpy()
        adv_np = attack.generate(x=x_np)
        adv_t  = torch.tensor(adv_np).to(DEVICE)
        labels = labels.to(DEVICE)
        with torch.no_grad():
            preds = art_clf.model(adv_t).argmax(dim=1)
        correct += (preds == labels).sum().item()
        total   += labels.size(0)
    return correct / total


# ── Visualisation ─────────────────────────────────────────────────────────────
def denormalise(t):
    mean = torch.tensor(CIFAR10_MEAN).view(3, 1, 1)
    std  = torch.tensor(CIFAR10_STD).view(3, 1, 1)
    return (t * std + mean).clamp(0, 1)


def save_comparison_images(model, art_clf, loader, epsilon, n=10):
    model.eval()
    criterion = nn.CrossEntropyLoss()
    imgs, labels = next(iter(loader))
    imgs, labels = imgs[:n].to(DEVICE), labels[:n].to(DEVICE)

    adv_scratch = fgsm_scratch(model, imgs, labels, epsilon, criterion)
    adv_art_np  = FastGradientMethod(estimator=art_clf, eps=epsilon).generate(imgs.cpu().numpy())
    adv_art     = torch.tensor(adv_art_np).to(DEVICE)

    wandb_imgs = []
    fig, axes = plt.subplots(n, 3, figsize=(9, 3 * n))
    for i in range(n):
        orig = denormalise(imgs[i].cpu()).permute(1, 2, 0).numpy()
        scr  = denormalise(adv_scratch[i].cpu()).permute(1, 2, 0).numpy()
        art_ = denormalise(adv_art[i].cpu()).permute(1, 2, 0).numpy()
        axes[i, 0].imshow(orig);  axes[i, 0].set_title("Original");       axes[i, 0].axis("off")
        axes[i, 1].imshow(scr);   axes[i, 1].set_title("FGSM Scratch");   axes[i, 1].axis("off")
        axes[i, 2].imshow(art_);  axes[i, 2].set_title("FGSM ART");       axes[i, 2].axis("off")

        wandb_imgs.append(wandb.Image(orig,  caption=f"Original – {CIFAR10_CLASSES[labels[i]]}"))
        wandb_imgs.append(wandb.Image(scr,   caption=f"FGSM Scratch – {CIFAR10_CLASSES[labels[i]]}"))
        wandb_imgs.append(wandb.Image(art_,  caption=f"FGSM ART – {CIFAR10_CLASSES[labels[i]]}"))

    plt.tight_layout()
    plt.savefig("fgsm_comparison.png", dpi=100)
    plt.close()
    wandb.log({"FGSM_samples": wandb_imgs})
    print("FGSM comparison grid saved → fgsm_comparison.png")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs",    type=int,   default=50)
    parser.add_argument("--train",     action="store_true", help="Train from scratch (skip if weights exist)")
    parser.add_argument("--ckpt",      default=os.path.join(WEIGHTS_DIR, "resnet18_cifar10_best.pt"))
    parser.add_argument("--wandb_key", default=None)
    args = parser.parse_args()

    if args.wandb_key:
        wandb.login(key=args.wandb_key)
    wandb.init(project="DLOps-Ass5-Q2i", name="fgsm_comparison")

    train_loader, test_loader = get_loaders()
    model = build_resnet18()

    # ── Train ──────────────────────────────────────────────────────────────
    if args.train or not os.path.exists(args.ckpt):
        print("Training ResNet18 on CIFAR-10 …")
        ckpt = train_clean(model, train_loader, test_loader, epochs=args.epochs)
    else:
        ckpt = args.ckpt

    model.load_state_dict(torch.load(ckpt, map_location=DEVICE))
    model.to(DEVICE).eval()

    clean_acc = evaluate_clean(model, test_loader)
    print(f"Clean test accuracy: {clean_acc*100:.2f}%")
    assert clean_acc >= 0.72, f"Model did not reach 72% (got {clean_acc:.4f}). Train longer."

    art_clf = build_art_classifier(model)
    epsilons = [0.01, 0.02, 0.05, 0.1, 0.2]

    print("\nEpsilon | Scratch Acc | ART Acc")
    print("-" * 40)
    rows = []
    for eps in epsilons:
        acc_scr = eval_fgsm_scratch(model, test_loader, eps)
        acc_art = eval_fgsm_art(art_clf, test_loader, eps)
        print(f"  {eps:.3f}  |   {acc_scr:.4f}   |  {acc_art:.4f}")
        wandb.log({"epsilon": eps, "acc_fgsm_scratch": acc_scr, "acc_fgsm_art": acc_art})
        rows.append((eps, acc_scr, acc_art))

    # Comparison plot
    epsi_list   = [r[0] for r in rows]
    acc_s_list  = [r[1] for r in rows]
    acc_a_list  = [r[2] for r in rows]
    fig, ax = plt.subplots()
    ax.plot(epsi_list, acc_s_list, "o-", label="FGSM Scratch")
    ax.plot(epsi_list, acc_a_list, "s-", label="FGSM ART")
    ax.axhline(clean_acc, linestyle="--", color="grey", label="Clean")
    ax.set_xlabel("Epsilon")
    ax.set_ylabel("Accuracy")
    ax.set_title("Perturbation Strength vs Accuracy")
    ax.legend()
    plt.tight_layout()
    plt.savefig("fgsm_eps_vs_acc.png", dpi=150)
    wandb.log({"eps_vs_acc": wandb.Image("fgsm_eps_vs_acc.png")})

    # Visual samples at eps=0.05
    save_comparison_images(model, art_clf, test_loader, epsilon=0.05)
    wandb.log({"clean_accuracy": clean_acc})
    wandb.finish()


if __name__ == "__main__":
    main()
