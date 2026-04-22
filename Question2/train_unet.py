import os, glob, random, json
import numpy as np
import cv2
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from tqdm import tqdm


seed = 10
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_CLASSES = 23
IMG_W, IMG_H = 128, 96          # (width, height) for cv2.resize
EPOCHS     = 15
BATCH_SIZE = 8
LR         = 1e-3
SAVE_DIR   = "Question2"
os.makedirs(SAVE_DIR, exist_ok=True)
print(f"Training on: {DEVICE}")


class CityscapesDataset(Dataset):
    def __init__(self, image_paths, mask_paths):
        self.image_paths = image_paths
        self.mask_paths  = mask_paths

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Read Image
        img = cv2.imread(self.image_paths[idx])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (IMG_W, IMG_H), interpolation=cv2.INTER_NEAREST)
        img = img.astype(np.float32) / 255.0

        # Read Mask
        mask = cv2.imread(self.mask_paths[idx])
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
        mask = cv2.resize(mask, (IMG_W, IMG_H), interpolation=cv2.INTER_NEAREST)
        mask = np.max(mask, axis=-1)          # (H, W) with values 0-255
        # Map to class indices 0-22
        mask = (mask / 255.0 * (NUM_CLASSES - 1)).astype(np.int64)
        mask = np.clip(mask, 0, NUM_CLASSES - 1)

        img  = torch.from_numpy(img).permute(2, 0, 1)   # (C, H, W)
        mask = torch.from_numpy(mask).long()              # (H, W)
        return img, mask


def get_data_loaders():
    img_paths  = sorted(glob.glob("data/CameraRGB/*.png"))
    mask_paths = sorted(glob.glob("data/CameraMask/*.png"))

    # Fallback extensions
    if not img_paths:
        img_paths  = sorted(glob.glob("data/CameraRGB/*.jpg"))
        mask_paths = sorted(glob.glob("data/CameraMask/*.jpg"))

    assert len(img_paths) > 0, "No images found – check data/ directory"
    assert len(img_paths) == len(mask_paths), "Image/mask count mismatch"

    tr_imgs, te_imgs, tr_masks, te_masks = train_test_split(
        img_paths, mask_paths, test_size=0.2, random_state=SEED)

    train_ds = CityscapesDataset(tr_imgs, tr_masks)
    test_ds  = CityscapesDataset(te_imgs, te_masks)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=2, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

    print(f"Train: {len(train_ds)} | Test: {len(test_ds)}")
    return train_loader, test_loader, te_imgs, te_masks


# ─────────────────────────────────────────
# 2.  UNet MODEL
# ─────────────────────────────────────────
class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
        )
    def forward(self, x): return self.net(x)


class UNet(nn.Module):
    def __init__(self, in_channels=3, num_classes=NUM_CLASSES, features=[64, 128, 256, 512]):
        super().__init__()
        self.downs, self.ups = nn.ModuleList(), nn.ModuleList()
        self.pool = nn.MaxPool2d(2, 2)

        # Encoder
        ch = in_channels
        for f in features:
            self.downs.append(DoubleConv(ch, f))
            ch = f

        self.bottleneck = DoubleConv(features[-1], features[-1] * 2)

        # Decoder
        for f in reversed(features):
            self.ups.append(nn.ConvTranspose2d(f * 2, f, 2, 2))
            self.ups.append(DoubleConv(f * 2, f))

        self.final = nn.Conv2d(features[0], num_classes, 1)

    def forward(self, x):
        skips = []
        for down in self.downs:
            x = down(x); skips.append(x); x = self.pool(x)
        x = self.bottleneck(x)
        skips = skips[::-1]
        for i in range(0, len(self.ups), 2):
            x = self.ups[i](x)
            s = skips[i // 2]
            if x.shape != s.shape:
                x = nn.functional.interpolate(x, size=s.shape[2:])
            x = torch.cat([s, x], dim=1)
            x = self.ups[i + 1](x)
        return self.final(x)


# ─────────────────────────────────────────
# 3.  METRICS
# ─────────────────────────────────────────
def compute_miou_mdice(preds, masks, num_classes=NUM_CLASSES):
    """preds: (B,H,W) int64 logits argmax, masks: (B,H,W) int64"""
    iou_list, dice_list = [], []
    preds = preds.cpu().numpy().flatten()
    masks = masks.cpu().numpy().flatten()
    for c in range(num_classes):
        p = (preds == c); g = (masks == c)
        inter = (p & g).sum()
        union = (p | g).sum()
        iou_list.append(inter / (union + 1e-8) if union > 0 else float('nan'))
        dice_list.append(2 * inter / (p.sum() + g.sum() + 1e-8) if (p.sum() + g.sum()) > 0 else float('nan'))
    miou  = np.nanmean(iou_list)
    mdice = np.nanmean(dice_list)
    return miou, mdice


# ─────────────────────────────────────────
# 4.  TRAIN / EVAL LOOPS
# ─────────────────────────────────────────
def train_one_epoch(model, loader, criterion, optimizer):
    model.train()
    total_loss, total_miou, total_mdice = 0, 0, 0
    for imgs, masks in tqdm(loader, leave=False, desc="Train"):
        imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
        optimizer.zero_grad()
        preds = model(imgs)
        loss  = criterion(preds, masks)
        loss.backward(); optimizer.step()
        total_loss  += loss.item()
        pred_cls     = preds.argmax(1)
        m, d         = compute_miou_mdice(pred_cls, masks)
        total_miou  += m; total_mdice += d
    n = len(loader)
    return total_loss / n, total_miou / n, total_mdice / n


@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    total_loss, total_miou, total_mdice = 0, 0, 0
    criterion = nn.CrossEntropyLoss()
    for imgs, masks in tqdm(loader, leave=False, desc="Eval"):
        imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
        preds = model(imgs)
        loss  = criterion(preds, masks)
        total_loss  += loss.item()
        pred_cls     = preds.argmax(1)
        m, d         = compute_miou_mdice(pred_cls, masks)
        total_miou  += m; total_mdice += d
    n = len(loader)
    return total_loss / n, total_miou / n, total_mdice / n


# ─────────────────────────────────────────
# 5.  MAIN TRAINING LOOP
# ─────────────────────────────────────────
def main():
    train_loader, test_loader, te_imgs, te_masks = get_data_loaders()

    model     = UNet().to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    history = {"loss": [], "miou": [], "mdice": []}

    best_miou = 0
    for epoch in range(1, EPOCHS + 1):
        tr_loss, tr_miou, tr_dice = train_one_epoch(model, train_loader, criterion, optimizer)
        scheduler.step()
        history["loss"].append(tr_loss)
        history["miou"].append(tr_miou)
        history["mdice"].append(tr_dice)
        print(f"Epoch {epoch:02d}/{EPOCHS} | Loss: {tr_loss:.4f} | mIoU: {tr_miou:.4f} | mDice: {tr_dice:.4f}")
        if tr_miou > best_miou:
            best_miou = tr_miou
            torch.save(model.state_dict(), f"{SAVE_DIR}/best_unet.pth")

    # ── Final Test Evaluation ──────────────────
    model.load_state_dict(torch.load(f"{SAVE_DIR}/best_unet.pth", map_location=DEVICE))
    _, test_miou, test_mdice = evaluate(model, test_loader)
    print(f"\n{'='*50}")
    print(f"TEST  mIoU : {test_miou:.4f}")
    print(f"TEST  mDice: {test_mdice:.4f}")
    print(f"{'='*50}")

    # Save metrics JSON
    results = {
        "history": history,
        "test_miou":  float(test_miou),
        "test_mdice": float(test_mdice),
        "test_image_paths": te_imgs[:20],
        "test_mask_paths":  te_masks[:20],
    }
    with open(f"{SAVE_DIR}/metrics.json", "w") as f:
        json.dump(results, f, indent=2)

    # ── Plots ──────────────────────────────────
    epochs_range = range(1, EPOCHS + 1)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].plot(epochs_range, history["loss"],  "b-o", markersize=4)
    axes[0].set_title("Training Loss");  axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss")
    axes[0].grid(True)

    axes[1].plot(epochs_range, history["miou"],  "g-o", markersize=4)
    axes[1].axhline(y=test_miou,  color="r", linestyle="--", label=f"Test mIoU={test_miou:.4f}")
    axes[1].set_title("mIoU");   axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("mIoU")
    axes[1].legend(); axes[1].grid(True)

    axes[2].plot(epochs_range, history["mdice"], "m-o", markersize=4)
    axes[2].axhline(y=test_mdice, color="r", linestyle="--", label=f"Test mDice={test_mdice:.4f}")
    axes[2].set_title("mDice"); axes[2].set_xlabel("Epoch"); axes[2].set_ylabel("mDice")
    axes[2].legend(); axes[2].grid(True)

    plt.suptitle("UNet Cityscapes Segmentation – Training Curves", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(f"{SAVE_DIR}/training_plots.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nPlots saved to {SAVE_DIR}/training_plots.png")
    print(f"Model saved to {SAVE_DIR}/best_unet.pth")
    print(f"Metrics saved to {SAVE_DIR}/metrics.json")


if __name__ == "__main__":
    main()
