"""
setup.py
--------
What this does:
  Reads the Kaggle fashion dataset, picks only Apparel/clothing items,
  takes 10,000 samples, runs each product image through the CLIP model
  to generate a 512-number "fingerprint" (embedding) for each image,
  saves those fingerprints and product info to disk.

Why this matters:
  These fingerprints are what the search engine compares against.
  When a user types "blue jeans", CLIP converts that text into a
  512-number fingerprint too, then finds the most similar image fingerprints.
  This step only runs ONCE — it is offline preparation.

Outputs:
  embeddings_a.pt   (10000 x 512 tensor — one fingerprint per product)
  embeddings_b.pt   (copy of above — Variant B starts identical to A)
  metadata.json     (product names, categories)
  images/0.jpg ...  (resized product photos for the frontend)

Time: 25–40 min on CPU  /  8–12 min on Mac M2 GPU

Usage:
  python setup.py --data_dir /path/to/fashion-dataset
"""

import os, json, argparse, time, shutil
import torch
import pandas as pd
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

parser = argparse.ArgumentParser()
parser.add_argument("--data_dir",    required=True,
                    help="Path to extracted Kaggle dataset folder (contains styles.csv and images/)")
parser.add_argument("--max_samples", type=int, default=10000)
args = parser.parse_args()

os.makedirs("images", exist_ok=True)

# ------------------------------------------------------------------
# Step 1: Read the CSV file that describes every product
# ------------------------------------------------------------------
styles_path = os.path.join(args.data_dir, "styles.csv")
print(f"Reading {styles_path} ...")
df = pd.read_csv(styles_path, on_bad_lines="skip")
print(f"  Total products in CSV: {len(df)}")

# Keep only Apparel (clothing) — removes shoes, bags, accessories etc.
df = df[df["masterCategory"] == "Apparel"].copy()
print(f"  After keeping only Apparel: {len(df)} products")

# ------------------------------------------------------------------
# Step 2: Drop products whose image file is missing on disk
# ------------------------------------------------------------------
images_dir = os.path.join(args.data_dir, "images")
df["img_path"] = df["id"].astype(str).apply(
    lambda x: os.path.join(images_dir, f"{x}.jpg"))
df = df[df["img_path"].apply(os.path.exists)].copy()
print(f"  After removing missing images: {len(df)} products")

# Take 10,000 samples (same samples every time thanks to random_state=42)
if len(df) > args.max_samples:
    df = df.sample(args.max_samples, random_state=42).reset_index(drop=True)
    print(f"  Sampled down to: {args.max_samples} products")

# ------------------------------------------------------------------
# Step 3: Load the CLIP model
# CLIP is a model from OpenAI that understands both images and text.
# It can convert an image or a sentence into a list of 512 numbers
# that capture its "meaning". Similar things get similar numbers.
# ------------------------------------------------------------------
device = ("mps"  if torch.backends.mps.is_available()  else
          "cuda" if torch.cuda.is_available()           else "cpu")
print(f"\nUsing device: {device}")
print("Loading CLIP model (downloads ~600MB first time)...")
model     = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
model.eval()
print("CLIP model ready.\n")

# ------------------------------------------------------------------
# Step 4: Process each product image
# For every product: open image → run through CLIP → save the 512 numbers
# ------------------------------------------------------------------
print("Generating embeddings (this is the slow part)...")
embeddings = []
metadata   = []
failed     = 0
start      = time.time()

for loop_idx, (_, row) in enumerate(df.iterrows()):
    try:
        img = Image.open(row["img_path"]).convert("RGB")
    except Exception as e:
        print(f"  WARNING: Skipping {row['img_path']}: {e}")
        failed += 1
        continue

    # Save a small copy of the image for the frontend to display
    img.resize((224, 224)).save(f"images/{loop_idx}.jpg", "JPEG", quality=85)

    # Run the image through CLIP to get its 512-number fingerprint
    inputs = processor(images=img, return_tensors="pt").to(device)
    with torch.no_grad():
        feat = model.get_image_features(**inputs)
        
        # --- FOOLPROOF TENSOR EXTRACTION ---
        if not isinstance(feat, torch.Tensor):
            if hasattr(feat, "image_embeds") and feat.image_embeds is not None:
                feat = feat.image_embeds
            elif hasattr(feat, "pooler_output") and feat.pooler_output is not None:
                feat = feat.pooler_output
            elif isinstance(feat, tuple):
                feat = feat[0]
                
        # Just in case your specific transformers version skipped the projection
        if feat.shape[-1] != 512:
            feat = model.visual_projection(feat)
        # -----------------------------------
                
    feat = feat / feat.norm(dim=-1, keepdim=True)
    # Normalize so cosine similarity works correctly (values between -1 and 1)
    feat = feat / feat.norm(dim=-1, keepdim=True)
    embeddings.append(feat.cpu())

    metadata.append({
        "id":           str(row["id"]),
        "name":         str(row.get("productDisplayName", "")),
        "category":     str(row.get("masterCategory", "")),
        "sub_category": str(row.get("subCategory", "")),
        "image_path":   f"{loop_idx}.jpg"
    })

    # Print progress every 500 items
    if (loop_idx + 1) % 500 == 0:
        elapsed = time.time() - start
        rate    = (loop_idx + 1) / elapsed
        eta     = (len(df) - loop_idx - 1) / rate
        print(f"  [{loop_idx+1:5d}/{len(df)}]  "
              f"{rate:.1f} items/sec  "
              f"ETA: {eta/60:.1f} min")

# ------------------------------------------------------------------
# Step 5: Save everything to disk
# ------------------------------------------------------------------
emb_tensor = torch.cat(embeddings)   # shape: (10000, 512)
torch.save(emb_tensor, "embeddings_a.pt")
shutil.copy("embeddings_a.pt", "embeddings_b.pt")  # B starts identical to A

with open("metadata.json", "w") as f:
    json.dump(metadata, f, indent=2)

elapsed_total = (time.time() - start) / 60
print(f"""
Done in {elapsed_total:.1f} minutes!
  Products processed : {len(metadata)}
  Products failed    : {failed}
  embeddings_a.pt    : shape {emb_tensor.shape}  (10000 products × 512 numbers)
  embeddings_b.pt    : copy of above (Variant B starts same as A)
  metadata.json      : product names and categories
  images/            : {len(metadata)} product photos

Next step:  python finetune.py --use_synthetic_feedback
""")
