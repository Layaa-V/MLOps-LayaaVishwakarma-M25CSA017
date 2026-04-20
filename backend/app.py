from flask import Flask, request, jsonify, Response
import torch
import torch.nn as nn
from transformers import CLIPProcessor, CLIPModel
import json, os, random, time, base64, io
from PIL import Image
from datetime import datetime
from collections import Counter
import wandb
import logging

from prometheus_client import (
    Counter as PCounter,
    Histogram,
    Gauge,
    generate_latest,
    CONTENT_TYPE_LATEST
)

app = Flask(__name__)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    handlers=[logging.FileHandler("app.log"), logging.StreamHandler()]
)
log = logging.getLogger(__name__)

# ── Prometheus metrics ────────────────────────────────────────────────────────
SEARCH_TOTAL       = PCounter("fashion_search_total",
                               "Total search requests", ["variant", "search_type"])
SEARCH_LATENCY     = Histogram("fashion_search_latency_seconds",
                                "Search latency", ["variant", "search_type"],
                                buckets=[.05, .1, .25, .5, 1, 2, 5])
SIMILARITY_GAUGE   = Gauge("fashion_avg_similarity",
                            "Rolling avg similarity", ["variant"])
FEEDBACK_TOTAL     = PCounter("fashion_feedback_total",
                               "Feedback clicks", ["variant", "rating"])
ACTIVE_SEARCHES    = Gauge("fashion_active_searches", "Searches in progress")
MODEL_LOADED       = Gauge("fashion_model_finetuned",
                            "1 = FashionCLIP loaded for Variant B")
IMAGE_SEARCH_TOTAL = PCounter("fashion_image_search_total",
                               "Total image search requests")

# ── Config ────────────────────────────────────────────────────────────────────
WANDB_PROJECT = os.getenv("WANDB_PROJECT", "fashion-discovery-engine")
GCS_BUCKET    = os.getenv("GCS_BUCKET", "fashion-engine-layaa")
FEEDBACK_LOG  = os.getenv("FEEDBACK_LOG",  "feedback_log.json")
IMAGES_DIR    = os.getenv("IMAGES_DIR",    "images")

IMAGE_CACHE: dict[int, str] = {}

METRICS = {
    "total_searches":       0,
    "variant_a_count":      0,    "variant_b_count":      0,
    "variant_a_total_sim":  0.0,  "variant_b_total_sim":  0.0,
    "variant_a_total_lat":  0.0,  "variant_b_total_lat":  0.0,
    "total_image_searches": 0,
    "total_feedback":       0,
    "positive_feedback":    0,    "negative_feedback":    0,
    "variant_a_positive":   0,    "variant_b_positive":   0,
    "variant_a_negative":   0,    "variant_b_negative":   0,
    "search_history":       []
}
wandb_search_step   = 0
wandb_feedback_step = 0

MODEL_A_NAME = "openai/clip-vit-base-patch32"
MODEL_B_NAME = "patrickjohncyh/fashion-clip"


def extract_tensor(feat, feature_type: str = "text") -> torch.Tensor:
    # Case 1: Already a plain tensor — most common case
    if isinstance(feat, torch.Tensor):
        return feat

    # Case 2: text_embeds (the projected CLIP text features)
    if hasattr(feat, 'text_embeds') and feat.text_embeds is not None:
        log.debug(f"extract_tensor: using .text_embeds")
        return feat.text_embeds

    # Case 3: image_embeds (the projected CLIP image features)
    if hasattr(feat, 'image_embeds') and feat.image_embeds is not None:
        log.debug(f"extract_tensor: using .image_embeds")
        return feat.image_embeds

    # Case 4: pooler_output (CLS token, still good for similarity)
    if hasattr(feat, 'pooler_output') and feat.pooler_output is not None:
        log.debug(f"extract_tensor: using .pooler_output")
        return feat.pooler_output

    # Case 5: last_hidden_state — mean pool over sequence dimension
    if hasattr(feat, 'last_hidden_state') and feat.last_hidden_state is not None:
        log.debug(f"extract_tensor: mean-pooling last_hidden_state")
        return feat.last_hidden_state.mean(dim=1)

    # Case 6: tuple or list — take first element (common for older models)
    if isinstance(feat, (tuple, list)):
        log.debug(f"extract_tensor: taking first element of tuple/list")
        first = feat[0]
        if isinstance(first, torch.Tensor):
            return first

    # Case 7: Nothing worked — raise a clear error
    raise TypeError(
        f"Cannot extract tensor from CLIP output of type {type(feat)}. "
        f"Attributes: {[a for a in dir(feat) if not a.startswith('_')]}. "
        f"Please report this to fix extract_tensor()."
    )


log.info("Fashion Discovery Engine v5.1 — Two-Model A/B (Fixed)")
log.info(f"  Variant A: {MODEL_A_NAME}")
log.info(f"  Variant B: {MODEL_B_NAME}")


wandb.init(
    project = WANDB_PROJECT,
    name    = f"serving_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
    tags    = ["serving", "ab-test", "fashion-clip"],
    reinit  = True
)

# Load Variant A (raw CLIP)
log.info("Loading Variant A — raw CLIP...")
model_a     = CLIPModel.from_pretrained(MODEL_A_NAME)
processor_a = CLIPProcessor.from_pretrained(MODEL_A_NAME)
model_a.eval()
log.info("Variant A ready.")

# Load Variant B (FashionCLIP) — with graceful fallback
log.info("Loading Variant B — FashionCLIP...")
try:
    model_b     = CLIPModel.from_pretrained(MODEL_B_NAME)
    processor_b = CLIPProcessor.from_pretrained(MODEL_B_NAME)
    model_b.eval()
    variant_b_model_loaded = True
    log.info("Variant B (FashionCLIP) ready.")

   
    try:
        _test_inputs = processor_b(text=["blue dress"], return_tensors="pt", padding=True)
        with torch.no_grad():
            _test_feat = model_b.get_text_features(**_test_inputs)
        _test_tensor = extract_tensor(_test_feat, "text")
        _test_normed = _test_tensor / _test_tensor.norm(dim=-1, keepdim=True)
        assert _test_normed.shape[-1] == 512, \
            f"Expected 512-dim, got {_test_normed.shape[-1]}"
        assert not torch.isnan(_test_normed).any(), "NaN in FashionCLIP output!"
        log.info(f"FashionCLIP encoding sanity check PASSED. "
                 f"Output shape: {_test_normed.shape}, "
                 f"Return type: {type(_test_feat).__name__}")
        MODEL_LOADED.set(1)
    except Exception as sanity_err:
        log.error(f"FashionCLIP FAILED sanity check: {sanity_err}")
        log.error("Falling back to raw CLIP for Variant B.")
        model_b     = model_a
        processor_b = processor_a
        variant_b_model_loaded = False
        MODEL_LOADED.set(0)

except Exception as load_err:
    log.warning(f"FashionCLIP failed to load: {load_err}")
    log.warning("Variant B will use raw CLIP as fallback.")
    model_b     = model_a
    processor_b = processor_a
    variant_b_model_loaded = False
    MODEL_LOADED.set(0)

log.info("Loading product embeddings...")
emb_a = torch.load("embeddings_a.pt", map_location="cpu")


with open("metadata.json") as f:
    metadata = json.load(f)

emb_b_valid = False
if os.path.exists("embeddings_b.pt") and variant_b_model_loaded:
    emb_b_candidate = torch.load("embeddings_b.pt", map_location="cpu")
    if emb_b_candidate.shape[0] == len(metadata):
        emb_b = emb_b_candidate
        emb_b_valid = True
        log.info(f"Loaded embeddings_b.pt: shape={emb_b.shape} "
                 f"(FashionCLIP image vectors — correct space for Variant B)")
    else:
        log.error(f"embeddings_b.pt has {emb_b_candidate.shape[0]} rows "
                  f"but metadata has {len(metadata)} items. "
                  f"Re-run setup_variant_b.py to regenerate it.")
        emb_b = emb_a
        model_b     = model_a
        processor_b = processor_a
        variant_b_model_loaded = False
        MODEL_LOADED.set(0)
elif not os.path.exists("embeddings_b.pt"):
    log.warning("embeddings_b.pt NOT FOUND.")
    log.warning("Run: python3 setup_variant_b.py")
    log.warning("Until then, Variant B = Variant A (both raw CLIP).")
    emb_b = emb_a
    model_b     = model_a
    processor_b = processor_a
    variant_b_model_loaded = False
    MODEL_LOADED.set(0)
else:
    
    log.warning("FashionCLIP not loaded — not using embeddings_b.pt to avoid space mismatch.")
    emb_b = emb_a

log.info(f"Products indexed: {len(metadata)}")
log.info(f"Variant A: model={MODEL_A_NAME}  embs=embeddings_a.pt")
log.info(f"Variant B: model={MODEL_B_NAME if variant_b_model_loaded else MODEL_A_NAME}  "
         f"embs={'embeddings_b.pt' if emb_b_valid else 'embeddings_a.pt (fallback)'}")
log.info("=" * 58)

if GCS_BUCKET:
    try:
        from google.cloud import storage
        blob = storage.Client().bucket(GCS_BUCKET).blob("data/feedback_log.json")
        if blob.exists():
            blob.download_to_filename(FEEDBACK_LOG)
            log.info("Loaded feedback from GCS")
    except Exception as e:
        log.warning(f"GCS load failed: {e}")


def get_image_b64(idx: int) -> str:
    if idx in IMAGE_CACHE:
        return IMAGE_CACHE[idx]
    path = os.path.join(IMAGES_DIR, f"{idx}.jpg")
    if os.path.exists(path):
        with open(path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode()
        IMAGE_CACHE[idx] = b64
        return b64
    return ""


def encode_text_variant_a(query: str) -> torch.Tensor:
    """Encode text with raw CLIP (Variant A)."""
    inputs = processor_a(text=[query], return_tensors="pt", padding=True)
    with torch.no_grad():
        feat = model_a.get_text_features(**inputs)

    if hasattr(feat, 'pooler_output') and feat.pooler_output is not None:
        feat = feat.pooler_output
    elif hasattr(feat, 'text_embeds') and feat.text_embeds is not None:
        feat = feat.text_embeds
    elif not isinstance(feat, torch.Tensor):
        feat = extract_tensor(feat, "text")

    return feat / feat.norm(dim=-1, keepdim=True)


def encode_text_variant_b(query: str) -> torch.Tensor:
    """Encode text with FashionCLIP (Variant B)."""
    inputs = processor_b(text=[query], return_tensors="pt", padding=True)
    with torch.no_grad():
        feat = model_b.get_text_features(**inputs)

    if hasattr(feat, 'pooler_output') and feat.pooler_output is not None:
        feat = feat.pooler_output
    elif hasattr(feat, 'text_embeds') and feat.text_embeds is not None:
        feat = feat.text_embeds
    elif not isinstance(feat, torch.Tensor):
        feat = extract_tensor(feat, "text")

    return feat / feat.norm(dim=-1, keepdim=True)
    

def encode_image_variant_a(pil_img: Image.Image) -> torch.Tensor:
    """Encode uploaded image with raw CLIP (Variant A)."""
    inputs = processor_a(images=pil_img, return_tensors="pt")
    with torch.no_grad():
        feat = model_a.get_image_features(**inputs)

    if hasattr(feat, 'pooler_output') and feat.pooler_output is not None:
        feat = feat.pooler_output
    elif hasattr(feat, 'image_embeds') and feat.image_embeds is not None:
        feat = feat.image_embeds
    elif not isinstance(feat, torch.Tensor):
        feat = extract_tensor(feat, "image")

    return feat / feat.norm(dim=-1, keepdim=True)


def encode_image_variant_b(pil_img: Image.Image) -> torch.Tensor:
    """Encode uploaded image with FashionCLIP (Variant B)."""
    inputs = processor_b(images=pil_img, return_tensors="pt")
    with torch.no_grad():
        feat = model_b.get_image_features(**inputs)

    if hasattr(feat, 'pooler_output') and feat.pooler_output is not None:
        feat = feat.pooler_output
    elif hasattr(feat, 'image_embeds') and feat.image_embeds is not None:
        feat = feat.image_embeds
    elif not isinstance(feat, torch.Tensor):
        feat = extract_tensor(feat, "image")

    return feat / feat.norm(dim=-1, keepdim=True)

def encode_image_for_search(pil_img: Image.Image) -> torch.Tensor:
    """Encode uploaded image with raw CLIP for image search."""
    inputs = processor_a(images=pil_img, return_tensors="pt")
    with torch.no_grad():
        feat = model_a.get_image_features(**inputs)

    # Preserved from your working v5 — extended with extract_tensor fallback
    if hasattr(feat, 'pooler_output') and feat.pooler_output is not None:
        feat = feat.pooler_output
    elif hasattr(feat, 'image_embeds') and feat.image_embeds is not None:
        feat = feat.image_embeds
    elif not isinstance(feat, torch.Tensor):
        feat = extract_tensor(feat, "image")

    return feat / feat.norm(dim=-1, keepdim=True)


def run_search(feat: torch.Tensor, embs: torch.Tensor, top_k: int = 3):
    sims = torch.matmul(feat, embs.T).squeeze(0)
    vals, idxs = torch.topk(sims, top_k)
    return idxs.tolist(), vals.tolist()


def build_result(idx: int, score: float) -> dict:
    item = dict(metadata[idx])
    item["similarity"] = round(float(score), 4)
    item["image_b64"]  = get_image_b64(idx)
    return item


def save_feedback_to_gcs():
    if not GCS_BUCKET:
        return
    try:
        from google.cloud import storage
        storage.Client().bucket(GCS_BUCKET).blob(
            "data/feedback_log.json").upload_from_filename(FEEDBACK_LOG)
    except Exception as e:
        log.warning(f"GCS upload failed: {e}")


#endpoints
@app.route("/health")
def health():
    return jsonify({
        "status":                  "healthy",
        "indexed_items":           len(metadata),
        "variant_a_model":         MODEL_A_NAME,
        "variant_b_model":         MODEL_B_NAME if variant_b_model_loaded
                                   else f"{MODEL_A_NAME} (fallback)",
        "fashion_clip_loaded":     variant_b_model_loaded,
        "embeddings_b_valid":      emb_b_valid,
        "variant_b_ready":         variant_b_model_loaded and emb_b_valid,
    })


@app.route("/metrics")
def metrics():
    """Prometheus scrape endpoint."""
    return Response(generate_latest(), mimetype=CONTENT_TYPE_LATEST)


@app.route("/admin/diagnose")
def admin_diagnose():
    """
    Diagnoses whether FashionCLIP encoding is working correctly.
    Call this if Variant B shows similarity as ? or gives wrong results.
    Returns the exact return type from each model so you know what is happening.
    """
    results = {}

    # Test Variant A
    try:
        inp = processor_a(text=["blue dress"], return_tensors="pt", padding=True)
        with torch.no_grad():
            raw_a = model_a.get_text_features(**inp)
        tensor_a = encode_text_variant_a("blue dress")
        results["variant_a"] = {
            "status":        "ok",
            "raw_return_type": type(raw_a).__name__,
            "tensor_shape":  list(tensor_a.shape),
            "has_nan":       bool(torch.isnan(tensor_a).any()),
            "norm":          round(float(tensor_a.norm().item()), 4),
            "model":         MODEL_A_NAME,
        }
    except Exception as e:
        results["variant_a"] = {"status": "error", "error": str(e)}

    # Test Variant B
    try:
        inp = processor_b(text=["blue dress"], return_tensors="pt", padding=True)
        with torch.no_grad():
            raw_b = model_b.get_text_features(**inp)
        tensor_b = encode_text_variant_b("blue dress")
        results["variant_b"] = {
            "status":          "ok",
            "raw_return_type": type(raw_b).__name__,
            "tensor_shape":    list(tensor_b.shape),
            "has_nan":         bool(torch.isnan(tensor_b).any()),
            "norm":            round(float(tensor_b.norm().item()), 4),
            "model":           MODEL_B_NAME if variant_b_model_loaded else MODEL_A_NAME,
            "fashion_clip_loaded": variant_b_model_loaded,
        }
        # Also check if the two embeddings are the same (indicates fallback is active)
        sim = float((tensor_a * tensor_b).sum().item())
        results["variant_b"]["similarity_to_a"] = round(sim, 4)
        results["variant_b"]["are_same_model"] = (sim > 0.9999)
    except Exception as e:
        results["variant_b"] = {"status": "error", "error": str(e)}

    # Check embeddings
    results["embeddings"] = {
        "emb_a_shape":      list(emb_a.shape),
        "emb_b_shape":      list(emb_b.shape),
        "emb_b_valid":      emb_b_valid,
        "emb_b_is_copy_of_a": (emb_a is emb_b),
    }

    # Overall verdict
    a_ok = results["variant_a"].get("status") == "ok"
    b_ok = results["variant_b"].get("status") == "ok"
    b_has_nan = results.get("variant_b", {}).get("has_nan", True)
    b_same_as_a = results.get("variant_b", {}).get("are_same_model", True)

    if a_ok and b_ok and not b_has_nan and not b_same_as_a and emb_b_valid:
        verdict = "ALL OK — FashionCLIP is working correctly as Variant B"
    elif b_same_as_a:
        verdict = ("Variant B is using the same model as A (fallback active). "
                   "Check that setup_variant_b.py ran successfully and "
                   "embeddings_b.pt was generated correctly.")
    elif b_has_nan:
        verdict = ("NaN detected in Variant B output — tensor extraction failed. "
                   "The extract_tensor() function could not handle the return type. "
                   "Check raw_return_type in variant_b above.")
    elif not emb_b_valid:
        verdict = ("embeddings_b.pt is missing or has wrong shape. "
                   "Run: python3 setup_variant_b.py")
    else:
        verdict = "Unknown issue — check individual results above"

    results["verdict"] = verdict
    return jsonify(results)


@app.route("/search")
def search():
    global wandb_search_step
    query = request.args.get("query", "").strip()
    if not query:
        return jsonify({"error": "Please provide a search query"}), 400

    variant = "A" if random.random() < 0.5 else "B"
    ACTIVE_SEARCHES.inc()
    start = time.time()

    if variant == "A":
        feat       = encode_text_variant_a(query)
        embs       = emb_a
        model_desc = f"Raw CLIP ({MODEL_A_NAME})"
    else:
        feat       = encode_text_variant_b(query)
        embs       = emb_b
        model_desc = (f"FashionCLIP ({MODEL_B_NAME})" if variant_b_model_loaded
                      else f"Raw CLIP fallback ({MODEL_A_NAME})")

    result_indices, top_scores = run_search(feat, embs)

    latency_s  = time.time() - start
    latency_ms = round(latency_s * 1000, 2)
    avg_sim    = round(sum(top_scores) / 3, 4)

    SEARCH_TOTAL.labels(variant=variant, search_type="text").inc()
    SEARCH_LATENCY.labels(variant=variant, search_type="text").observe(latency_s)

    v = variant.lower()
    METRICS["total_searches"]         += 1
    METRICS[f"variant_{v}_count"]     += 1
    METRICS[f"variant_{v}_total_sim"] += avg_sim
    METRICS[f"variant_{v}_total_lat"] += latency_ms
    METRICS["search_history"].append(avg_sim)
    if len(METRICS["search_history"]) > 200:
        METRICS["search_history"].pop(0)
    count = METRICS[f"variant_{v}_count"]
    SIMILARITY_GAUGE.labels(variant=variant).set(
        METRICS[f"variant_{v}_total_sim"] / count)
    ACTIVE_SEARCHES.dec()

    wandb_search_step += 1
    wandb.log({
        "search/step":                        wandb_search_step,
        f"search/variant_{v}_avg_similarity": avg_sim,
        f"search/variant_{v}_latency_ms":     latency_ms,
    })

    log.info(f"TEXT-SEARCH | '{query}' | {variant} | sim={avg_sim} | {latency_ms}ms")

    results = [build_result(i, s) for i, s in zip(result_indices, top_scores)]
    return jsonify({
        "query":       query,
        "variant":     variant,
        "search_type": "text",
        "results":     results,
        "scores":      top_scores,
        "latency_ms":  latency_ms,
        "model_info":  {
            "A": f"Raw CLIP ({MODEL_A_NAME})",
            "B": (f"FashionCLIP ({MODEL_B_NAME})" if variant_b_model_loaded
                  else f"Raw CLIP fallback ({MODEL_A_NAME})")
        }
    })


@app.route("/image-search", methods=["POST"])
def image_search():
    global wandb_search_step
    ACTIVE_SEARCHES.inc()
    start = time.time()

    data      = request.get_json(force=True)
    image_b64 = data.get("image_b64", "").strip()
    if not image_b64:
        ACTIVE_SEARCHES.dec()
        return jsonify({"error": "image_b64 is required"}), 400

    try:
        img_bytes = base64.b64decode(image_b64)
        pil_img   = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    except Exception as e:
        ACTIVE_SEARCHES.dec()
        return jsonify({"error": f"Could not decode image: {e}"}), 400

    # Randomly select Variant A or B
    variant = "A" if random.random() < 0.5 else "B"

    try:
        if variant == "A":
            img_feat   = encode_image_variant_a(pil_img)
            embs       = emb_a
            model_desc = f"Raw CLIP ({MODEL_A_NAME})"
        else:
            img_feat   = encode_image_variant_b(pil_img)
            embs       = emb_b
            model_desc = (f"FashionCLIP ({MODEL_B_NAME})" if variant_b_model_loaded
                          else f"Raw CLIP fallback ({MODEL_A_NAME})")
    except Exception as e:
        ACTIVE_SEARCHES.dec()
        return jsonify({"error": f"CLIP encoding failed: {e}"}), 500

    result_indices, top_scores = run_search(img_feat, embs)

    latency_s  = time.time() - start
    latency_ms = round(latency_s * 1000, 2)
    avg_sim    = round(sum(top_scores) / 3, 4)

    IMAGE_SEARCH_TOTAL.inc()
    SEARCH_TOTAL.labels(variant=variant, search_type="image").inc()
    SEARCH_LATENCY.labels(variant=variant, search_type="image").observe(latency_s)
    
    # Track metrics for A/B dashboard
    v = variant.lower()
    METRICS["total_image_searches"]   += 1
    METRICS["total_searches"]         += 1
    METRICS[f"variant_{v}_count"]     += 1
    METRICS[f"variant_{v}_total_sim"] += avg_sim
    METRICS[f"variant_{v}_total_lat"] += latency_ms
    METRICS["search_history"].append(avg_sim)
    if len(METRICS["search_history"]) > 200:
        METRICS["search_history"].pop(0)

    ACTIVE_SEARCHES.dec()

    wandb_search_step += 1
    wandb.log({
        "search/step":                        wandb_search_step,
        f"search/image_variant_{v}_latency_ms": latency_ms,
        f"search/image_variant_{v}_avg_sim":    avg_sim,
    })

    log.info(f"IMAGE-SEARCH | {variant} | sim={avg_sim} | {latency_ms}ms")
    
    return jsonify({
        "search_type": "image",
        "variant":     variant,
        "results":     [build_result(i, s) for i, s in zip(result_indices, top_scores)],
        "scores":      top_scores,
        "latency_ms":  latency_ms,
        "model_info":  {
            "A": f"Raw CLIP ({MODEL_A_NAME})",
            "B": (f"FashionCLIP ({MODEL_B_NAME})" if variant_b_model_loaded
                  else f"Raw CLIP fallback ({MODEL_A_NAME})")
        }
    })

@app.route("/feedback", methods=["POST"])
def feedback():
    global wandb_feedback_step
    data    = request.get_json(force=True)
    query   = data.get("query",       "")
    rname   = data.get("result_name", "")
    rating  = data.get("rating",      0)
    variant = data.get("variant",     "?")

    if rating not in (1, -1):
        return jsonify({"error": "rating must be 1 or -1"}), 400

    record = {"timestamp": datetime.now().isoformat(), "query": query,
              "result_name": rname, "rating": rating, "variant": variant}

    log_data = []
    if os.path.exists(FEEDBACK_LOG):
        with open(FEEDBACK_LOG) as f:
            try:    log_data = json.load(f)
            except: log_data = []
    log_data.append(record)
    with open(FEEDBACK_LOG, "w") as f:
        json.dump(log_data, f, indent=2)
    save_feedback_to_gcs()

    FEEDBACK_TOTAL.labels(variant=variant,
                           rating="positive" if rating==1 else "negative").inc()
    METRICS["total_feedback"] += 1
    if rating == 1:
        METRICS["positive_feedback"]                   += 1
        METRICS[f"variant_{variant.lower()}_positive"] += 1
    else:
        METRICS["negative_feedback"]                   += 1
        METRICS[f"variant_{variant.lower()}_negative"] += 1

    wandb_feedback_step += 1
    tf, pos = METRICS["total_feedback"], METRICS["positive_feedback"]
    wandb.log({
        "feedback/step":             wandb_feedback_step,
        "feedback/satisfaction_pct": round(pos/tf*100, 1) if tf else 0,
    })

    log.info(f"FEEDBACK | '{query}' | {variant} | {'👍' if rating==1 else '👎'}")
    return jsonify({"status": "logged", "total_feedback": len(log_data)})


@app.route("/admin/metrics")
def admin_metrics():
    a, b   = METRICS["variant_a_count"], METRICS["variant_b_count"]
    t      = METRICS["total_searches"]
    pos    = METRICS["positive_feedback"]
    neg    = METRICS["negative_feedback"]
    tf     = METRICS["total_feedback"]

    avg_sim_a = round(METRICS["variant_a_total_sim"] / a, 4) if a else 0.0
    avg_sim_b = round(METRICS["variant_b_total_sim"] / b, 4) if b else 0.0
    avg_lat_a = round(METRICS["variant_a_total_lat"] / a, 1) if a else 0.0
    avg_lat_b = round(METRICS["variant_b_total_lat"] / b, 1) if b else 0.0

    a_pos, b_pos = METRICS["variant_a_positive"], METRICS["variant_b_positive"]
    a_neg, b_neg = METRICS["variant_a_negative"], METRICS["variant_b_negative"]
    a_sat = round(a_pos/(a_pos+a_neg)*100, 1) if (a_pos+a_neg) else 0.0
    b_sat = round(b_pos/(b_pos+b_neg)*100, 1) if (b_pos+b_neg) else 0.0

    winner_sim = ("A (Raw CLIP)" if avg_sim_a >= avg_sim_b else "B (FashionCLIP)") \
                 if (a and b) else "Need more searches"
    winner_sat = ("A (Raw CLIP)" if a_sat >= b_sat else "B (FashionCLIP)") \
                 if (a_pos+a_neg > 0 and b_pos+b_neg > 0) else "Need more feedback"

    return jsonify({
        "total_searches": t, "total_image_searches": METRICS["total_image_searches"],
        "variant_a_count": a, "variant_b_count": b,
        "variant_a_pct": round(a/t*100,1) if t else 0,
        "variant_b_pct": round(b/t*100,1) if t else 0,
        "variant_a_avg_similarity": avg_sim_a, "variant_b_avg_similarity": avg_sim_b,
        "variant_a_avg_latency_ms": avg_lat_a, "variant_b_avg_latency_ms": avg_lat_b,
        "variant_a_satisfaction_pct": a_sat,   "variant_b_satisfaction_pct": b_sat,
        "winner_by_similarity": winner_sim, "winner_by_satisfaction": winner_sat,
        "total_feedback": tf, "positive_feedback": pos, "negative_feedback": neg,
        "overall_satisfaction_pct": round(pos/tf*100,1) if tf else 0,
        "model_info": {
            "variant_a": f"Raw CLIP ({MODEL_A_NAME})",
            "variant_b": (f"FashionCLIP ({MODEL_B_NAME})" if variant_b_model_loaded
                          else f"Raw CLIP fallback ({MODEL_A_NAME})")
        }
    })


@app.route("/admin/evaluate")
def admin_evaluate():
    eval_file = "eval_queries.json"
    if not os.path.exists(eval_file):
        return jsonify({"error": "eval_queries.json not found"}), 404
    with open(eval_file) as f:
        eqs = json.load(f)

    hits_a, hits_b, rows_a, rows_b = 0, 0, [], []
    for item in eqs:
        kw = item.get("expected_name_contains", "").lower()
        for encode_fn, embs, hv, rv in [
            (encode_text_variant_a, emb_a, "a", rows_a),
            (encode_text_variant_b, emb_b, "b", rows_b),
        ]:
            feat   = encode_fn(item["query"])
            sims   = torch.matmul(feat, embs.T).squeeze(0)
            _, top = torch.topk(sims, 3)
            names  = [metadata[i]["name"].lower() for i in top.tolist()]
            hit    = any(kw in n for n in names)
            if hit:
                if hv == "a": hits_a += 1
                else:         hits_b += 1
            rv.append({"query": item["query"], "expected": kw,
                       "top3": [metadata[i]["name"] for i in top.tolist()],
                       "hit":  hit})

    p3_a = round(hits_a/len(eqs), 3) if eqs else 0.0
    p3_b = round(hits_b/len(eqs), 3) if eqs else 0.0
    wandb.log({"eval/p3_A": p3_a, "eval/p3_B": p3_b})
    return jsonify({
        "variant_a_precision_at_3": p3_a, "variant_b_precision_at_3": p3_b,
        "winner": "A" if p3_a >= p3_b else "B",
        "total_queries": len(eqs),
        "breakdown_a": rows_a, "breakdown_b": rows_b
    })


@app.route("/admin/feedback-analysis")
def admin_feedback_analysis():
    if not os.path.exists(FEEDBACK_LOG):
        return jsonify({"worst_queries": [], "total": 0})
    with open(FEEDBACK_LOG) as f:
        log_data = json.load(f)
    bad = Counter(e["query"] for e in log_data if e["rating"] == -1)
    return jsonify({
        "total": len(log_data), "worst_queries": bad.most_common(10),
        "positive_count": sum(1 for e in log_data if e["rating"]==1),
        "negative_count": sum(1 for e in log_data if e["rating"]==-1)
    })


@app.route("/")
def home():
    return (f"Fashion Discovery Engine v5.1<br>"
            f"Variant A: {MODEL_A_NAME}<br>"
            f"Variant B: {MODEL_B_NAME} "
            f"({'loaded' if variant_b_model_loaded else 'FALLBACK to A'})<br>"
            f"embeddings_b valid: {emb_b_valid}<br>"
            "Endpoints: /search /image-search /feedback /metrics /health<br>"
            "Admin: /admin/metrics /admin/evaluate /admin/diagnose "
            "/admin/feedback-analysis")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=False)
