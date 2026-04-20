import streamlit as st
import requests
import base64
import io
import os
from PIL import Image

API_URL = os.getenv("BACKEND_URL", "http://localhost:8000")

st.set_page_config(
    page_title="Fashion Discovery Engine",
    page_icon="👗",
    layout="wide"
)

st.markdown("""
<style>
.badge-A  { background:#dbeafe; color:#1e40af; padding:2px 10px;
            border-radius:12px; font-size:.8em; font-weight:600; }
.badge-B  { background:#dcfce7; color:#166534; padding:2px 10px;
            border-radius:12px; font-size:.8em; font-weight:600; }
.badge-img{ background:#fef3c7; color:#92400e; padding:2px 10px;
            border-radius:12px; font-size:.8em; font-weight:600; }
.latency  { background:#f3f4f6; color:#374151; padding:2px 8px;
            border-radius:10px; font-size:.78em; }
img.pimg  { border-radius:8px; width:100%; object-fit:cover; max-height:220px; }
.ph       { width:100%; height:160px; background:#f0f0f0; border-radius:8px;
            display:flex; align-items:center; justify-content:center;
            color:#aaa; font-size:.85rem; }
.rname    { font-weight:600; font-size:.95rem; margin-top:4px; }
.rmeta    { color:#666; font-size:.82rem; }
</style>
""", unsafe_allow_html=True)


# ── Shared helpers ─────────────────────────────────────────────────────────────

def show_product_image(b64: str, caption: str = ""):
    """Display a product image from base64. Shows placeholder if missing."""
    if b64:
        st.markdown(
            f'<img class="pimg" src="data:image/jpeg;base64,{b64}" alt="{caption}"/>',
            unsafe_allow_html=True
        )
    else:
        st.markdown('<div class="ph">No image available</div>',
                    unsafe_allow_html=True)


def send_feedback(query: str, result_name: str, rating: int, variant: str):
    """Send thumbs up/down to the Flask backend. Silently ignores errors."""
    try:
        requests.post(
            f"{API_URL}/feedback",
            json={"query": query, "result_name": result_name,
                  "rating": rating, "variant": variant},
            timeout=3
        )
    except Exception:
        pass


def render_result_columns(results: list, scores: list,
                           query_label: str, variant: str):
    """
    Renders top-3 results side by side in equal columns.
    """
    cols = st.columns(len(results))
    for col, item, score in zip(cols, results, scores):
        with col:
            show_product_image(item.get("image_b64", ""), item.get("name", ""))
            idx = results.index(item)
            st.markdown(
                f'<div class="rname">#{idx+1}  {item["name"]}</div>'
                f'<div class="rmeta">'
                f'📂 {item.get("category", "—")} · {item.get("sub_category", "")}<br>'
                f'🎯 Similarity: {round(score, 3) if score else "?"}'
                f'</div>',
                unsafe_allow_html=True
            )
            c1, c2 = st.columns(2)
            with c1:
                if st.button("👍", key=f"up_{idx}_{item['name'][:10]}_{query_label[:8]}"):
                    send_feedback(query_label, item["name"], 1, variant)
                    st.success("Thanks!")
            with c2:
                if st.button("👎", key=f"dn_{idx}_{item['name'][:10]}_{query_label[:8]}"):
                    send_feedback(query_label, item["name"], -1, variant)
                    st.warning("Noted!")


def pil_image_to_b64(pil_img: Image.Image) -> str:
    buffer = io.BytesIO()
    pil_img.save(buffer, format="JPEG", quality=90)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def backend_is_up() -> bool:
    try:
        r = requests.get(f"{API_URL}/health", timeout=3)
        return r.ok
    except Exception:
        return False


st.title("👗 Fashion Discovery Engine")
st.caption(
    "Find fashion by description or by uploading a photo · "
    "A/B: Raw CLIP vs Fine-tuned · "
    "Monitored on W&B + Grafana"
)

# Initialize Session States to prevent results from vanishing
if "text_results" not in st.session_state:
    st.session_state.text_results = None
if "image_results" not in st.session_state:
    st.session_state.image_results = None

tab1, tab2 = st.tabs(["🔍 Text Search", "📷 Image Search"])



with tab1:
    st.markdown("Type a clothing description. The engine compares your words against 10,000 product images.")

    q1 = st.text_input(
        "What are you looking for?",
        placeholder="e.g. white sneakers  /  navy blazer  /  floral summer dress",
        key="q1"
    )

    if st.button("Search 🔍", key="btn_text"):
        if not q1.strip():
            st.warning("Please type something to search for.")
        elif not backend_is_up():
            st.error("Backend is not running. Start Flask on port 8000 first.")
        else:
            with st.spinner("Searching 10,000 products..."):
                try:
                    r = requests.get(f"{API_URL}/search", params={"query": q1.strip()}, timeout=30)
                    if r.ok:
                        # Save the results to session memory
                        st.session_state.text_results = r.json()
                    else:
                        st.error(f"Error: {r.json().get('error', 'Unknown error')}")
                except requests.exceptions.ConnectionError:
                    st.error("Cannot connect to backend.")

    # Render results from session memory (outside the button logic so it persists)
    if st.session_state.text_results:
        data = st.session_state.text_results
        variant = data.get("variant", "?")
        mi = data.get("model_info", {})
        lat = data.get("latency_ms", "?")

        st.markdown(
            f'Results for **"{data["query"]}"** &nbsp;'
            f'<span class="badge-{variant}">Variant {variant}</span>&nbsp;'
            f'<span class="latency">{lat} ms</span>',
            unsafe_allow_html=True
        )
        st.caption(f"Model: {mi.get(variant, '')}")

        render_result_columns(
            data.get("results", []),
            data.get("scores", []),
            query_label=data["query"],
            variant=variant
        )


with tab2:
    st.markdown("Upload any clothing photo. The engine finds the 3 most visually similar products.")

    uploaded_file = st.file_uploader(
        "Upload a clothing photo",
        type=["jpg", "jpeg", "png", "webp"],
        key="img_upload"
    )

    # Clear image results if user uploads a new image
    if uploaded_file is None and st.session_state.image_results is not None:
        st.session_state.image_results = None

    if uploaded_file is not None:
        uploaded_pil = Image.open(uploaded_file).convert("RGB")

        col_preview, col_spacer = st.columns([1, 2])
        with col_preview:
            st.markdown("**Your uploaded image:**")
            st.image(uploaded_pil, width=220)

        st.divider()

        if st.button("🔍 Find Similar Products", key="btn_image"):
            if not backend_is_up():
                st.error("Backend is not running.")
            else:
                with st.spinner("Encoding your image with CLIP and searching..."):
                    try:
                        img_b64 = pil_image_to_b64(uploaded_pil)
                        r = requests.post(f"{API_URL}/image-search", json={"image_b64": img_b64}, timeout=30)

                        if r.ok:
                            # Save to session memory
                            st.session_state.image_results = r.json()
                        else:
                            st.error(f"Search failed: {r.json().get('error', 'Unknown error')}")
                    except requests.exceptions.ConnectionError:
                        st.error("Cannot connect to backend.")

        # Render image results from session memory
        # Render image results from session memory
        if st.session_state.image_results:
            data = st.session_state.image_results
            lat = data.get("latency_ms", "?")
            variant = data.get("variant", "?")
            mi = data.get("model_info", {})

            st.markdown(
                f'Visually similar products &nbsp;'
                f'<span class="badge-{variant}">Variant {variant}</span>&nbsp;'
                f'<span class="latency">{lat} ms</span>',
                unsafe_allow_html=True
            )
            st.caption(f"Model: {mi.get(variant, '')}")
            
            render_result_columns(
                data.get("results", []),
                data.get("scores", []),
                query_label="image_search",
                variant=variant
            )

st.divider()
st.caption("Fashion Discovery Engine · CLIP · Streamlit · Grafana · Docker")