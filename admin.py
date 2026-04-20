
#admin.py  —  Owner Monitoring Dashboard for my first trial version
#Run separately on port 8502. 


import streamlit as st
import requests
import json
import os
import time

API_URL = os.getenv("BACKEND_URL", "http://localhost:8000")

st.set_page_config(
    page_title="Admin — Fashion Engine",
    page_icon="🛠️",
    layout="wide"
)

st.title("🛠️ Owner Monitoring Dashboard")
st.caption("Private page — do not share this URL with customers.")

col_r, col_a, _ = st.columns([2, 2, 6])
with col_r:
    auto = st.toggle("Auto-refresh every 15s", value=False)
with col_a:
    if st.button("🔄 Refresh now"):
        st.rerun()

if auto:
    time.sleep(15)
    st.rerun()

# ── Fetch live metrics from Flask ─────────────────────────────────
try:
    m  = requests.get(f"{API_URL}/admin/metrics", timeout=4).json()
    ok = True
except Exception:
    st.error("⚠️  Cannot reach Flask backend. Make sure it is running.")
    ok = False

if ok:
    st.divider()

    # ── Top-level numbers ─────────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Searches",       m["total_searches"])
    c2.metric("Total Feedback",       m["total_feedback"])
    c3.metric("Overall Satisfaction", f"{m['overall_satisfaction_pct']}%")
    c4.metric("👍 / 👎",
              f"{m['positive_feedback']} / {m['negative_feedback']}")

    st.divider()

    # ── A/B Comparison ────────────────────────────────────────────
    st.subheader("🔬 A/B Model Comparison")

    mi = m.get("model_info", {})
    st.info(
        f"**Variant A:** {mi.get('variant_a', 'Raw CLIP')}  \n"
        f"**Variant B:** {mi.get('variant_b', 'Fine-tuned CLIP')}"
    )

    col_a, col_b, col_w = st.columns(3)

    with col_a:
        st.markdown("#### Variant A — Raw CLIP")
        st.metric("Requests served",  m["variant_a_count"])
        st.metric("Traffic share",    f"{m['variant_a_pct']}%")
        st.metric("Avg similarity",   m["variant_a_avg_similarity"])
        st.metric("Avg latency",      f"{m['variant_a_avg_latency_ms']} ms")
        st.metric("Satisfaction",     f"{m['variant_a_satisfaction_pct']}%")

    with col_b:
        st.markdown("#### Variant B — Fine-tuned CLIP")
        st.metric("Requests served",  m["variant_b_count"])
        st.metric("Traffic share",    f"{m['variant_b_pct']}%")
        st.metric("Avg similarity",   m["variant_b_avg_similarity"])
        st.metric("Avg latency",      f"{m['variant_b_avg_latency_ms']} ms")
        st.metric("Satisfaction",     f"{m['variant_b_satisfaction_pct']}%")

    with col_w:
        st.markdown("#### Winner So Far")
        sim_w = m.get("winner_by_similarity", "—")
        sat_w = m.get("winner_by_satisfaction", "—")

        if "Need more" in str(sim_w):
            st.info(f"📊 {sim_w}")
        else:
            st.success(f"🏆 By similarity: **{sim_w}**")
            st.success(f"🏆 By satisfaction: **{sat_w}**")

        # Visual similarity bar
        a_sim = m["variant_a_avg_similarity"]
        b_sim = m["variant_b_avg_similarity"]
        if a_sim or b_sim:
            best = max(a_sim, b_sim) or 1
            st.markdown(f"`A` {'█' * int(a_sim/best*15)} {a_sim:.4f}")
            st.markdown(f"`B` {'█' * int(b_sim/best*15)} {b_sim:.4f}")

    st.caption(
        "💡 Higher avg similarity = model finds closer semantic matches.  \n"
        "Once you have 50+ searches, higher satisfaction rate is the cleaner signal.  \n"
        "After fine-tuning and redeploying, watch Variant B's similarity climb."
    )

    st.divider()

    # ── Best Hyperparameters ──────────────────────────────────────
    st.subheader("⚙️ Best Hyperparameters (found by Ray Tune + Optuna)")
    hp_path = "backend/configs/best_hparams.json"
    if os.path.exists(hp_path):
        with open(hp_path) as f:
            hp = json.load(f)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Learning rate",  f"{hp.get('lr','—'):.2e}")
        c2.metric("Hidden dim",     hp.get("hidden_dim", "—"))
        c3.metric("Epochs",         hp.get("epochs", "—"))
        c4.metric("Dropout",        f"{hp.get('dropout','—'):.2f}")
        st.caption("These were chosen by Optuna's TPE algorithm across 8 trials.")
    else:
        st.info("Run finetune.py first to generate best_hparams.json.")

    st.divider()

    # ── Precision@3 Evaluation ────────────────────────────────────
    st.subheader("🔬 Precision@3 Evaluation (both variants)")
    st.markdown(
        "Tests both variants against 15 known queries. "
        "Checks if the expected product type appears in the top 3 results."
    )
    if st.button("▶ Run Evaluation"):
        with st.spinner("Running 15 queries on both A and B..."):
            try:
                ev = requests.get(f"{API_URL}/admin/evaluate", timeout=120).json()
                if "error" in ev:
                    st.error(ev["error"])
                else:
                    e1, e2, e3 = st.columns(3)
                    e1.metric("Variant A  P@3", ev["variant_a_precision_at_3"])
                    e2.metric("Variant B  P@3", ev["variant_b_precision_at_3"])
                    e3.metric("Winner",         ev["winner"])

                    with st.expander("Variant A — detailed results"):
                        for row in ev["breakdown_a"]:
                            icon = "✅" if row["hit"] else "❌"
                            st.write(f"{icon} **{row['query']}** — expected: `{row['expected']}`")
                            st.caption("Top 3: " + "  |  ".join(row["top3"]))

                    with st.expander("Variant B — detailed results"):
                        for row in ev["breakdown_b"]:
                            icon = "✅" if row["hit"] else "❌"
                            st.write(f"{icon} **{row['query']}** — expected: `{row['expected']}`")
                            st.caption("Top 3: " + "  |  ".join(row["top3"]))
            except Exception as e:
                st.error(str(e))

    st.divider()

    # ── Worst Queries ─────────────────────────────────────────────
    st.subheader("📋 Queries with Most Negative Feedback")
    if st.button("Show worst-performing queries"):
        try:
            fa = requests.get(f"{API_URL}/admin/feedback-analysis", timeout=4).json()
            if fa["worst_queries"]:
                for q, count in fa["worst_queries"]:
                    st.markdown(f"- `{q}` — **{count}** thumbs-down(s)")
                st.caption(
                    "These are candidates for your next fine-tuning run. "
                    "The model struggles with these queries — add them to "
                    "eval_queries.json to track improvement."
                )
            else:
                st.info("No negative feedback yet.")
        except Exception as e:
            st.error(str(e))

    st.divider()

    # ── W&B and Grafana links ─────────────────────────────────────
    st.subheader("📊 External Dashboards")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**W&B** — training curves, sweep comparison, feedback over time")
        st.code("https://wandb.ai → project: fashion-discovery-engine")
    with col2:
        st.markdown("**Grafana** — live server health, latency, search rate")
        st.code("http://localhost:3000  (admin / admin)")

st.divider()
st.caption("Admin Dashboard — Fashion Discovery Engine")
