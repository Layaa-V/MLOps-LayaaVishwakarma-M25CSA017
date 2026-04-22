import os, json
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import streamlit as st
from PIL import Image

NUM_CLASSES = 23
IMG_W, IMG_H = 128, 96
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

PALETTE = np.array([
    [128,  64, 128], [244,  35, 232], [ 70,  70,  70], [102, 102, 156],
    [190, 153, 153], [153, 153, 153], [250, 170,  30], [220, 220,   0],
    [107, 142,  35], [152, 251, 152], [ 70, 130, 180], [220,  20,  60],
    [255,   0,   0], [  0,   0, 142], [  0,   0,  70], [  0,  60, 100],
    [  0,  80, 100], [  0,   0, 230], [119,  11,  32], [ 50, 205,  50],
    [255, 165,   0], [138,  43, 226], [  0, 255, 255],
], dtype=np.uint8)

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
        ch = in_channels
        for f in features:
            self.downs.append(DoubleConv(ch, f)); ch = f
        self.bottleneck = DoubleConv(features[-1], features[-1] * 2)
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
                x = F.interpolate(x, size=s.shape[2:])
            x = torch.cat([s, x], dim=1)
            x = self.ups[i + 1](x)
        return self.final(x)

@st.cache(allow_output_mutation=True)
def load_model():
    model = UNet().to(DEVICE)
    model.load_state_dict(torch.load("best_unet.pth", map_location=DEVICE))
    model.eval()
    return model

def mask_to_rgb(mask_array):
    rgb = np.zeros((*mask_array.shape, 3), dtype=np.uint8)
    for c in range(NUM_CLASSES):
        rgb[mask_array == c] = PALETTE[c]
    return rgb

def preprocess_uploaded(pil_img):
    arr = np.array(pil_img.convert("RGB"))
    arr = cv2.resize(arr, (IMG_W, IMG_H), interpolation=cv2.INTER_NEAREST)
    arr = arr.astype(np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)

def preprocess_mask_for_display(pil_mask):
    arr = np.array(pil_mask.convert("RGB"))
    arr = cv2.resize(arr, (IMG_W, IMG_H), interpolation=cv2.INTER_NEAREST)
    arr_cls = np.clip((np.max(arr, axis=-1) / 255.0 * (NUM_CLASSES - 1)).astype(np.int64), 0, NUM_CLASSES - 1)
    return mask_to_rgb(arr_cls)

@torch.no_grad()
def predict(model, tensor):
    logits = model(tensor.to(DEVICE))
    return mask_to_rgb(logits.argmax(1)[0].cpu().numpy())


page = st.sidebar.radio("Navigation", ["Page 1: Training Results", "Page 2: Model Inference"])


if page == "Page 1: Training Results":
    st.title("Training Results")
    
    with open("metrics.json") as f:
        metrics = json.load(f)
    st.write(f"**Test mIoU:** {metrics['test_miou']:.4f}")
    st.write(f"**Test mDice:** {metrics['test_mdice']:.4f}")
    
    st.image("training_plots.png", caption="Training Curves (Loss, mIoU, mDice)")


elif page == "Page 2: Model Inference":
    st.title("Model Inference")
    model = load_model()

    imgs = st.file_uploader("Upload 4 Input Images", accept_multiple_files=True)

    if imgs and len(imgs) == 4:
        for i in range(4):
            pil_img = Image.open(imgs[i])
            filename = imgs[i].name
            
            # Look up the corresponding mask on the server using the uploaded filename
            mask_path = os.path.join("data", "CameraMask", filename)
            
            if not os.path.exists(mask_path):
                st.error(f"⚠️ Could not find the ground truth mask for '{filename}' at {mask_path}")
                continue

            pil_mask = Image.open(mask_path)
            
            # Preprocess and Predict
            inp_tensor = preprocess_uploaded(pil_img)
            gt_rgb = preprocess_mask_for_display(pil_mask)
            pred_rgb = predict(model, inp_tensor)

            # Display Output
            st.write(f"### Image Set {i+1}: `{filename}`")
            c1, c2, c3 = st.columns(3)
            c1.image(pil_img, caption="Input Image", use_column_width=True)
            c2.image(cv2.resize(gt_rgb, (IMG_W*3, IMG_H*3), interpolation=cv2.INTER_NEAREST), caption="Ground Truth", use_column_width=True)
            c3.image(cv2.resize(pred_rgb, (IMG_W*3, IMG_H*3), interpolation=cv2.INTER_NEAREST), caption="Predicted Mask", use_column_width=True)
            st.markdown("---")
