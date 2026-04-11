"""
Q2(ii): Adversarial Detection Model
- ResNet-34 binary classifier: clean vs adversarial
- Attack (a): PGD via IBM ART
- Attack (b): BIM via IBM ART
- Logs 10 samples per attack type to WandB
"""

import os
import argparse
import wandb
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset, ConcatDataset
from torchvision import datasets, transforms, models
from tqdm import tqdm

# IBM ART
from art.attacks.evasion import ProjectedGradientDescent, BasicIterativeMethod
from art.estimators.classification import PyTorchClassifier

DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_CLASSES = 10   # CIFAR-10 source classes
DATA_DIR    = "./data"
WEIGHTS_DIR = "./weights"
os.makedirs(WEIGHTS_DIR, exist_ok=True)

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD  = (0.2023, 0.1994, 0.2010)
CIFAR10_CLASSES = ["airplane","automobile","bird","cat","deer",
                   "dog","frog","horse","ship","truck"]


# ── Helpers ───────────────────────────────────────────────────────────────────
def get_raw_data(batch_size=128):
    tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])
    train_ds = datasets.CIFAR10(DATA_DIR, train=True,  download=True, transform=tf)
    test_ds  = datasets.CIFAR10(DATA_DIR, train=False, download=True, transform=tf)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=False, num_workers=4)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=4)
    return train_loader, test_loader


def loader_to_numpy(loader):
    X_list, Y_list = [], []
    for imgs, labels in loader:
        X_list.append(imgs.numpy())
        Y_list.append(labels.numpy())
    return np.concatenate(X_list), np.concatenate(Y_list)


def build_victim_classifier(ckpt_path):
    """Load the ResNet18 trained in Q2(i) as the victim model for generating attacks."""
    victim = models.resnet18(weights=None)
    victim.fc = nn.Linear(victim.fc.in_features, NUM_CLASSES)
    victim.load_state_dict(torch.load(ckpt_path, map_location=DEVICE))
    victim.eval().to(DEVICE)

    art_clf = PyTorchClassifier(
        model=victim,
        loss=nn.CrossEntropyLoss(),
        optimizer=optim.SGD(victim.parameters(), lr=0.01),
        input_shape=(3, 32, 32),
        nb_classes=NUM_CLASSES,
        clip_values=(
            min((0 - m) / s for m, s in zip(CIFAR10_MEAN, CIFAR10_STD)),
            max((1 - m) / s for m, s in zip(CIFAR10_MEAN, CIFAR10_STD)),
        ),
        device_type="gpu" if torch.cuda.is_available() else "cpu",
    )
    return art_clf


def generate_adversarial(art_clf, X_np, attack_name, eps=0.03):
    """Generate adversarial examples using PGD or BIM via IBM ART."""
    if attack_name == "pgd":
        attack = ProjectedGradientDescent(
            estimator=art_clf, eps=eps, eps_step=eps / 10,
            max_iter=40, targeted=False, batch_size=128,
        )
    elif attack_name == "bim":
        attack = BasicIterativeMethod(
            estimator=art_clf, eps=eps, eps_step=eps / 10,
            max_iter=40, batch_size=128,
        )
    else:
        raise ValueError(f"Unknown attack: {attack_name}")
    print(f"Generating {attack_name.upper()} adversarial examples …")
    return attack.generate(x=X_np)


# ── Detector model ────────────────────────────────────────────────────────────
def build_detector():
    """ResNet-34 binary classifier (clean=0, adversarial=1)."""
    model = models.resnet34(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 2)
    return model


def make_detector_dataset(X_clean_np, X_adv_np):
    """Build a balanced dataset of clean (label=0) and adversarial (label=1) images."""
    n = min(len(X_clean_np), len(X_adv_np))
    X = np.concatenate([X_clean_np[:n], X_adv_np[:n]])
    Y = np.array([0] * n + [1] * n, dtype=np.int64)
    # Shuffle
    idx = np.random.permutation(len(X))
    X, Y = X[idx], Y[idx]
    ds = TensorDataset(torch.tensor(X, dtype=torch.float32),
                       torch.tensor(Y, dtype=torch.long))
    return ds


def train_detector(model, train_ds, val_ds, epochs=20, run_name="detector"):
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True,  num_workers=4)
    val_loader   = DataLoader(val_ds,   batch_size=64, shuffle=False, num_workers=4)
    criterion    = nn.CrossEntropyLoss()
    optimizer    = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler    = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    scaler       = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())
    model.to(DEVICE)

    best_acc, ckpt_path = 0.0, os.path.join(WEIGHTS_DIR, f"{run_name}_best.pt")
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss, correct, total = 0.0, 0, 0
        for X, Y in tqdm(train_loader, desc=f"[{run_name}] Ep{epoch}", leave=False):
            X, Y = X.to(DEVICE), Y.to(DEVICE)
            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                out  = model(X)
                loss = criterion(out, Y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item() * Y.size(0)
            correct    += (out.argmax(1) == Y).sum().item()
            total      += Y.size(0)
        scheduler.step()

        val_acc = eval_detector(model, val_loader)
        tr_acc  = correct / total
        print(f"Epoch {epoch:2d} | train_acc={tr_acc:.4f} | val_acc={val_acc:.4f}")
        wandb.log({f"{run_name}/epoch": epoch,
                   f"{run_name}/train_acc": tr_acc,
                   f"{run_name}/val_acc": val_acc})

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), ckpt_path)

    print(f"[{run_name}] Best detection accuracy: {best_acc:.4f}")
    assert best_acc >= 0.70, (
        f"Detection accuracy {best_acc:.4f} < 70%. Consider more epochs or data."
    )
    return best_acc


@torch.no_grad()
def eval_detector(model, loader):
    model.eval()
    correct, total = 0, 0
    for X, Y in loader:
        X, Y = X.to(DEVICE), Y.to(DEVICE)
        preds = model(X).argmax(dim=1)
        correct += (preds == Y).sum().item()
        total   += Y.size(0)
    return correct / total


# ── WandB sample logging ──────────────────────────────────────────────────────
def log_samples_wandb(X_clean_np, X_adv_np, attack_name, n=10):
    mean = np.array(CIFAR10_MEAN).reshape(3, 1, 1)
    std  = np.array(CIFAR10_STD).reshape(3, 1, 1)

    def denorm(x):
        return np.clip(x * std + mean, 0, 1).transpose(1, 2, 0)

    imgs = []
    for i in range(n):
        imgs.append(wandb.Image(denorm(X_clean_np[i]), caption=f"Clean #{i}"))
        imgs.append(wandb.Image(denorm(X_adv_np[i]),   caption=f"{attack_name.upper()} Adv #{i}"))
    wandb.log({f"{attack_name}_samples": imgs})


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--victim_ckpt", required=True,
                        help="Path to Q2(i) ResNet18 checkpoint used to generate attacks")
    parser.add_argument("--eps",         type=float, default=0.03)
    parser.add_argument("--epochs",      type=int,   default=20)
    parser.add_argument("--wandb_key",   default=None)
    args = parser.parse_args()

    if args.wandb_key:
        wandb.login(key=args.wandb_key)
    wandb.init(project="DLOps-Ass5-Q2ii", name="adversarial_detection")

    # ── Load CIFAR-10 as numpy ─────────────────────────────────────────────
    print("Loading CIFAR-10 …")
    train_loader, test_loader = get_raw_data()
    X_train, _  = loader_to_numpy(train_loader)
    X_test, _   = loader_to_numpy(test_loader)

    art_clf = build_victim_classifier(args.victim_ckpt)

    results = {}
    for attack_name in ["pgd", "bim"]:
        print(f"\n{'='*60}")
        print(f"  Attack: {attack_name.upper()}")
        print(f"{'='*60}")

        X_train_adv = generate_adversarial(art_clf, X_train, attack_name, args.eps)
        X_test_adv  = generate_adversarial(art_clf, X_test,  attack_name, args.eps)

        # Log 10 samples to WandB
        log_samples_wandb(X_test, X_test_adv, attack_name)

        # Build detector datasets
        train_ds = make_detector_dataset(X_train, X_train_adv)
        n_val    = int(len(train_ds) * 0.15)
        n_trn    = len(train_ds) - n_val
        train_split, val_split = torch.utils.data.random_split(train_ds, [n_trn, n_val])
        test_ds  = make_detector_dataset(X_test, X_test_adv)
        test_loader_det = DataLoader(test_ds, batch_size=64, shuffle=False)

        # Train detector
        detector = build_detector()
        run_name = f"detector_{attack_name}"
        best_acc = train_detector(detector, train_split, val_split,
                                  epochs=args.epochs, run_name=run_name)

        # Final test accuracy
        detector.load_state_dict(
            torch.load(os.path.join(WEIGHTS_DIR, f"{run_name}_best.pt"), map_location=DEVICE)
        )
        test_acc = eval_detector(detector.to(DEVICE), test_loader_det)
        print(f"[{attack_name.upper()}] Test detection accuracy: {test_acc:.4f}")
        wandb.log({f"{attack_name}_test_detection_acc": test_acc})
        results[attack_name] = test_acc

    print("\n\n=== Summary ===")
    for k, v in results.items():
        print(f"  {k.upper()} Detection Accuracy: {v*100:.2f}%")

    wandb.finish()


if __name__ == "__main__":
    main()
