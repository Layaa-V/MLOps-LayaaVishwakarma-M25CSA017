"""
setup_variant_b.py  (FIXED)
----------------------------
WHAT WAS WRONG:
  The previous version looked for images at paths like images/9888.jpg,
  images/9889.jpg etc — sequential index numbers stored in metadata["image_path"].
  But these files did not exist because setup.py saves images using the
  product's original Kaggle ID as the filename (e.g. images/15970.jpg),
  NOT the sequential loop index.
  Result: ALL 10,000 images failed → embeddings_b.pt was entirely zero vectors
  → Variant B gave completely wrong results.

THE FIX:
  This script tries FOUR different ways to find each product's image,
  in order of most likely to least likely:
    1. metadata["image_path"]           e.g. "15970.jpg"  (Kaggle ID filename)
    2. images/{metadata["id"]}.jpg      product's original ID
    3. images/{loop_index}.jpg          sequential index (setup.py loop counter)
    4. Fallback: search the images/ folder for any file containing the product ID

  At least one of these will work for your setup.
  The script prints exactly which method succeeded for the first 5 items
  so you can verify it is finding the right files.

USAGE:
  python3 setup_variant_b.py

  After it finishes (25-40 min CPU, 8-12 min GPU):
    docker compose down
    docker compose up --build -d

  Then verify: curl http://localhost:8000/health
  Should show: "variant_b_ready": true
"""

import os
import json
import argparse
import time
import shutil
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

parser = argparse.ArgumentParser()
parser.add_argument("--images_dir", type=str, default="backend/images",
                    help="Folder containing your product images")
parser.add_argument("--upload_gcs", action="store_true")
parser.add_argument("--bucket",     type=str, default="")
args = parser.parse_args()

IMAGES_DIR         = args.images_dir
OUTPUT_FILE        = "embeddings_b.pt"
FASHION_CLIP_MODEL = "patrickjohncyh/fashion-clip"

print("\n" + "="*60)
print("  setup_variant_b.py — FashionCLIP Embeddings (FIXED)")
print("="*60 + "\n")

# ── Diagnose the images folder first ──────────────────────────────
print(f"Checking images folder: {IMAGES_DIR}/")
if not os.path.isdir(IMAGES_DIR):
    print(f"ERROR: '{IMAGES_DIR}' folder not found!")
    print("Make sure you are running this from your project root directory")
    print("and that setup.py has been run to generate the images folder.")
    exit(1)

all_image_files = os.listdir(IMAGES_DIR)
image_count = len(all_image_files)
print(f"  Images found in folder: {image_count}")
if image_count == 0:
    print("ERROR: No images found! Run setup.py first.")
    exit(1)

# Show first 5 filenames so user can see what naming convention is used
sample = sorted(all_image_files)[:5]
print(f"  Sample filenames: {sample}")

# Build a set of all existing filenames for fast lookup
existing_files = set(all_image_files)

# ── Load metadata ──────────────────────────────────────────────────
with open("metadata.json") as f:
    metadata = json.load(f)
print(f"  Products in metadata.json: {len(metadata)}")
print()

# ── Figure out the image naming convention ─────────────────────────
# Check the first few metadata entries to figure out which lookup works
print("Detecting image naming convention...")
convention_found = None

for test_idx, test_item in enumerate(metadata[:10]):
    # Try all four conventions
    candidates = [
        test_item.get("image_path", ""),                  # e.g. "15970.jpg"
        f"{test_item.get('id', '')}.jpg",                 # e.g. "15970.jpg"
        f"{test_idx}.jpg",                                # e.g. "0.jpg"
        os.path.basename(test_item.get("image_path", "")),# basename only
    ]
    for c in candidates:
        if c and c in existing_files:
            if convention_found is None:
                convention_found = "direct"
            break

if convention_found is None:
    # Try sequential index more broadly
    if "0.jpg" in existing_files:
        convention_found = "sequential"
        print("  Convention: sequential index (0.jpg, 1.jpg, ...)")
    else:
        print("  Could not auto-detect convention. Will try all four methods per image.")

print()

def find_image_path(idx: int, item: dict) -> str:
    """
    Try four methods to find the image file for this product.
    Returns the full path if found, empty string if not found.
    """
    # Method 1: image_path stored in metadata (could be "15970.jpg" or "0.jpg")
    img_path_field = item.get("image_path", "")
    if img_path_field:
        full = os.path.join(IMAGES_DIR, img_path_field)
        if os.path.exists(full):
            return full
        # Also try just the basename in case it contains a subfolder
        base = os.path.basename(img_path_field)
        full_base = os.path.join(IMAGES_DIR, base)
        if os.path.exists(full_base):
            return full_base

    # Method 2: product ID as filename (Kaggle dataset IDs)
    product_id = str(item.get("id", ""))
    if product_id:
        full = os.path.join(IMAGES_DIR, f"{product_id}.jpg")
        if os.path.exists(full):
            return full

    # Method 3: sequential loop index (0.jpg, 1.jpg, ...)
    full = os.path.join(IMAGES_DIR, f"{idx}.jpg")
    if os.path.exists(full):
        return full

    # Method 4: search for any file whose name contains the product ID
    if product_id:
        for fname in existing_files:
            if product_id in fname:
                return os.path.join(IMAGES_DIR, fname)

    return ""   # not found

# ── Verify the detection works on first 5 items ──────────────────
print("Verifying image detection on first 5 products:")
success_count = 0
for i in range(min(5, len(metadata))):
    path = find_image_path(i, metadata[i])
    status = "✓ FOUND" if path else "✗ NOT FOUND"
    print(f"  [{i}] {metadata[i].get('name','')[:40]}")
    print(f"       {status}: {path or '(none of the 4 methods worked)'}")
    if path:
        success_count += 1

if success_count == 0:
    print("\nERROR: Could not find images for ANY of the first 5 products.")
    print("Please check:")
    print(f"  1. Your images are in: {os.path.abspath(IMAGES_DIR)}/")
    print(f"  2. Image filenames match product IDs or sequential numbers")
    print(f"  3. Sample files in your images/ folder: {sample[:10]}")
    print(f"  4. Sample metadata ids: {[m.get('id') for m in metadata[:5]]}")
    print(f"  5. Sample metadata image_paths: {[m.get('image_path') for m in metadata[:5]]}")
    exit(1)

print(f"\nImage detection working: {success_count}/5 found. Proceeding...\n")

# ── Load FashionCLIP ──────────────────────────────────────────────
device = ("mps"  if torch.backends.mps.is_available() else
          "cuda" if torch.cuda.is_available()          else "cpu")
print(f"Device: {device}")
print(f"Loading FashionCLIP ({FASHION_CLIP_MODEL})...")
print("(First run downloads ~600MB — subsequent runs use cache)\n")

model     = CLIPModel.from_pretrained(FASHION_CLIP_MODEL).to(device)
processor = CLIPProcessor.from_pretrained(FASHION_CLIP_MODEL)
model.eval()
print("FashionCLIP loaded.\n")

# ── Generate embeddings ───────────────────────────────────────────
print(f"Generating FashionCLIP embeddings for {len(metadata)} products...")
print("This is the slow part. Progress shown every 500 items.\n")

embeddings   = []
failed_zero  = []   # products where zero vector was used (image missing)
found_count  = 0
start        = time.time()

for idx, item in enumerate(metadata):
    img_path = find_image_path(idx, item)

    if not img_path:
        # Image not found — use zero vector as placeholder
        # This keeps the tensor indices aligned with metadata indices
        embeddings.append(torch.zeros(1, 512))
        failed_zero.append(idx)
        continue

    try:
        img    = Image.open(img_path).convert("RGB")
        inputs = processor(images=img, return_tensors="pt").to(device)
        with torch.no_grad():
            feat = model.get_image_features(**inputs)

        # Robustly extract tensor (handles all return types)
        if not isinstance(feat, torch.Tensor):
            if hasattr(feat, 'image_embeds') and feat.image_embeds is not None:
                feat = feat.image_embeds
            elif hasattr(feat, 'pooler_output') and feat.pooler_output is not None:
                feat = feat.pooler_output
            elif isinstance(feat, (tuple, list)):
                feat = feat[0]

        feat = feat / feat.norm(dim=-1, keepdim=True)
        embeddings.append(feat.cpu())
        found_count += 1

    except Exception as e:
        print(f"  ERROR processing {img_path}: {e}")
        embeddings.append(torch.zeros(1, 512))
        failed_zero.append(idx)

    if (idx + 1) % 500 == 0:
        elapsed = time.time() - start
        rate    = found_count / max(elapsed, 0.001)
        eta     = (len(metadata) - idx - 1) / max(rate, 0.001)
        pct_ok  = found_count / (idx + 1) * 100
        print(f"  [{idx+1:5d}/{len(metadata)}]  "
              f"found={found_count}  failed={len(failed_zero)}  "
              f"({pct_ok:.0f}% OK)  ETA: {eta/60:.1f} min")

emb_tensor = torch.cat(embeddings)   # shape: (N, 512)

elapsed_total = (time.time() - start) / 60
success_pct   = found_count / len(metadata) * 100

print(f"\n" + "="*60)
print(f"  Embedding generation complete in {elapsed_total:.1f} min")
print(f"  Successful: {found_count}/{len(metadata)} ({success_pct:.1f}%)")
print(f"  Failed (zero vector): {len(failed_zero)}")
print(f"  Output shape: {emb_tensor.shape}")
print("="*60)

# Warn if too many failed
if success_pct < 50:
    print(f"\nWARNING: Only {success_pct:.0f}% of images were found.")
    print("embeddings_b.pt will have many zero vectors which will give wrong results.")
    print("First 5 failed indices:", failed_zero[:5])
    print("These metadata entries have image_path:", [metadata[i].get('image_path') for i in failed_zero[:5]])
    print("Files actually in images/:", sorted(existing_files)[:10])
    print("\nPlease fix the image path mismatch before using embeddings_b.pt.")
elif success_pct < 90:
    print(f"\nNote: {len(failed_zero)} images not found — these products will have zero vectors.")
    print("Results will still work well for the {success_pct:.0f}% that succeeded.")
else:
    print(f"\nExcellent! {success_pct:.1f}% success rate. embeddings_b.pt is ready.")

# ── Save ──────────────────────────────────────────────────────────
torch.save(emb_tensor, OUTPUT_FILE)
print(f"\nSaved: {OUTPUT_FILE}")

os.makedirs("backend", exist_ok=True)
shutil.copy(OUTPUT_FILE, f"backend/{OUTPUT_FILE}")
print(f"Copied to: backend/{OUTPUT_FILE}")

# ── Optional GCS upload ───────────────────────────────────────────
if args.upload_gcs and args.bucket:
    try:
        from google.cloud import storage
        storage.Client().bucket(args.bucket).blob(
            "data/embeddings_b.pt").upload_from_filename(OUTPUT_FILE)
        print(f"Uploaded: gs://{args.bucket}/data/embeddings_b.pt")
    except Exception as e:
        print(f"GCS upload failed: {e}")

if success_pct >= 50:
    print(f"""
Next steps:
  1. Rebuild Docker:
       docker compose down
       docker compose up --build -d

  2. Verify Variant B is working:
       curl http://localhost:8000/health
       (should show "variant_b_ready": true)

  3. Run the diagnose endpoint:
       curl http://localhost:8000/admin/diagnose
       (should show "ALL OK" verdict)
""")