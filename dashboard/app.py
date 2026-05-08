"""
app.py — Vision Feature Selection & Neural Evolution Platform
Main Streamlit dashboard entry point.

Run with:  streamlit run dashboard/app.py
"""

import streamlit as st
import pandas as pd
from PIL import Image
import io
import base64

# ─── Page Config (must be first Streamlit call) ───────────────────────────────

st.set_page_config(
    page_title="Vision Evolution Platform",
    layout="wide",
    page_icon="🧠",
    initial_sidebar_state="expanded",
)

# ─── Local Imports ─────────────────────────────────────────────────────────────

from utils import (
    predict_image, load_results, load_ga_log,
    compute_model_summary, dataframe_to_csv_bytes,
)
from charts import (
    plot_accuracy_comparison, plot_ga_convergence,
    plot_confidence_distribution,
    plot_experiment_metrics_overview, plot_feature_selection,
)

# ─── Global CSS ───────────────────────────────────────────────────────────────

st.markdown("""
<style>
/* ── Import fonts ── */
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Syne:wght@400;600;700;800&display=swap');

/* ── Root palette ── */
:root {
    --bg:        #0A0F1E;
    --surface:   #111827;
    --surface2:  #1E293B;
    --border:    #1E3A5F;
    --cyan:      #22D3EE;
    --purple:    #A78BFA;
    --sky:       #38BDF8;
    --emerald:   #34D399;
    --amber:     #F59E0B;
    --text:      #E2E8F0;
    --muted:     #94A3B8;
}

/* ── Base ── */
html, body, [data-testid="stAppViewContainer"] {
    background-color: var(--bg) !important;
    color: var(--text);
    font-family: 'Syne', sans-serif;
}

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0D1B2A 0%, #0A0F1E 100%) !important;
    border-right: 1px solid var(--border) !important;
}

/* ── Sidebar nav buttons ── */
[data-testid="stSidebar"] .stButton > button {
    width: 100%;
    text-align: left;
    background: transparent;
    border: 1px solid transparent;
    color: var(--muted);
    border-radius: 8px;
    padding: 10px 14px;
    font-family: 'Space Mono', monospace;
    font-size: 13px;
    transition: all 0.2s ease;
    margin-bottom: 2px;
}
[data-testid="stSidebar"] .stButton > button:hover {
    background: rgba(34,211,238,0.08) !important;
    border-color: var(--cyan) !important;
    color: var(--cyan) !important;
}

/* ── Cards ── */
.vep-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 22px 26px;
    margin-bottom: 16px;
    position: relative;
    overflow: hidden;
}
.vep-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, var(--cyan), var(--purple));
}

/* ── Metric cards ── */
.metric-card {
    background: var(--surface2);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 18px 20px;
    text-align: center;
    transition: transform 0.2s, box-shadow 0.2s;
}
.metric-card:hover {
    transform: translateY(-3px);
    box-shadow: 0 8px 30px rgba(34,211,238,0.12);
}
.metric-value {
    font-size: 2rem;
    font-weight: 800;
    font-family: 'Space Mono', monospace;
    background: linear-gradient(135deg, var(--cyan), var(--purple));
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.metric-label {
    font-size: 0.78rem;
    color: var(--muted);
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin-top: 4px;
}

/* ── Section headings ── */
.section-title {
    font-family: 'Syne', sans-serif;
    font-weight: 800;
    font-size: 1.5rem;
    color: var(--text);
    margin-bottom: 4px;
}
.section-sub {
    color: var(--muted);
    font-size: 0.88rem;
    margin-bottom: 20px;
}

/* ── Hero banner ── */
.hero-banner {
    background: linear-gradient(135deg, #0D1B2A 0%, #0A1628 50%, #0D0A1E 100%);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 48px 40px;
    text-align: center;
    margin-bottom: 28px;
    position: relative;
    overflow: hidden;
}
.hero-banner::after {
    content: '';
    position: absolute;
    inset: 0;
    background: radial-gradient(ellipse at 50% 0%, rgba(34,211,238,0.07) 0%, transparent 70%);
    pointer-events: none;
}
.hero-title {
    font-family: 'Syne', sans-serif;
    font-weight: 800;
    font-size: 2.6rem;
    line-height: 1.15;
    background: linear-gradient(90deg, var(--cyan) 0%, var(--sky) 45%, var(--purple) 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 14px;
}
.hero-sub {
    color: var(--muted);
    font-size: 1.05rem;
    max-width: 640px;
    margin: 0 auto 24px;
    line-height: 1.6;
}

/* ── Tag badges ── */
.badge {
    display: inline-block;
    background: rgba(34,211,238,0.1);
    border: 1px solid var(--cyan);
    color: var(--cyan);
    border-radius: 20px;
    padding: 3px 12px;
    font-size: 0.75rem;
    font-family: 'Space Mono', monospace;
    margin: 3px 4px;
}
.badge-purple {
    background: rgba(167,139,250,0.1);
    border-color: var(--purple);
    color: var(--purple);
}
.badge-sky {
    background: rgba(56,189,248,0.1);
    border-color: var(--sky);
    color: var(--sky);
}

/* ── Divider ── */
.vep-divider {
    border: none;
    border-top: 1px solid var(--border);
    margin: 24px 0;
}

/* ── Workflow step ── */
.workflow-step {
    display: flex;
    align-items: flex-start;
    gap: 16px;
    margin-bottom: 16px;
}
.step-num {
    min-width: 36px;
    height: 36px;
    background: linear-gradient(135deg, var(--cyan), var(--purple));
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    font-family: 'Space Mono', monospace;
    font-weight: 700;
    font-size: 0.85rem;
    color: var(--bg);
    flex-shrink: 0;
}
.step-content { padding-top: 4px; }
.step-title { font-weight: 700; color: var(--text); font-size: 0.95rem; }
.step-desc { color: var(--muted); font-size: 0.83rem; margin-top: 2px; }

/* ── Prediction result box ── */
.pred-result {
    background: linear-gradient(135deg, rgba(34,211,238,0.05), rgba(167,139,250,0.05));
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 28px;
    margin-top: 16px;
}
.pred-class {
    font-family: 'Space Mono', monospace;
    font-size: 2.2rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    background: linear-gradient(90deg, var(--cyan), var(--purple));
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

/* ── Streamlit overrides ── */
[data-testid="stMetric"] {
    background: var(--surface2) !important;
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
    padding: 14px 18px !important;
}
[data-testid="stMetricValue"] {
    font-family: 'Space Mono', monospace !important;
    color: var(--cyan) !important;
}
[data-testid="stMetricLabel"] {
    color: var(--muted) !important;
}
[data-testid="stTabs"] [data-baseweb="tab"] {
    font-family: 'Syne', sans-serif !important;
    font-weight: 600 !important;
}
[data-testid="stDataFrameContainer"] {
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
}
</style>
""", unsafe_allow_html=True)

# ─── Session State ─────────────────────────────────────────────────────────────

if "page" not in st.session_state:
    st.session_state.page = "Home"

# ─── Sidebar ──────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding: 16px 0 24px;'>
        <div style='font-size:2.4rem; margin-bottom:6px;'>🧠</div>
        <div style='font-family:"Syne",sans-serif; font-weight:800; font-size:1rem;
                    background:linear-gradient(90deg,#22D3EE,#A78BFA);
                    -webkit-background-clip:text; -webkit-text-fill-color:transparent;'>
            Vision Evolution<br>Platform
        </div>
        <div style='color:#475569; font-size:0.72rem; font-family:"Space Mono",monospace;
                    margin-top:6px; letter-spacing:0.08em;'>v1.0 · RESEARCH DEMO</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div style='color:#334155; font-size:0.72rem; text-transform:uppercase;"
                "letter-spacing:0.12em; padding:0 4px 8px; font-family:\"Space Mono\",monospace;'>"
                "Navigation</div>", unsafe_allow_html=True)

    nav_items = [
        ("🏠", "Home",                "Platform overview & workflow"),
        ("🔍", "Prediction",          "Upload & classify images"),
        ("📊", "Dashboard Analytics", "Interactive experiment charts"),
        ("📋", "Experiment Logs",     "Raw data & CSV exports"),
    ]

    for icon, label, tooltip in nav_items:
        if st.button(f"{icon}  {label}", key=f"nav_{label}", help=tooltip,
                     use_container_width=True):
            st.session_state.page = label

    st.markdown("<hr style='border-color:#1E293B; margin:20px 0;'>", unsafe_allow_html=True)

    # Status indicators
    st.markdown("""
    <div style='font-size:0.72rem; color:#475569; font-family:"Space Mono",monospace;
                text-transform:uppercase; letter-spacing:0.1em; margin-bottom:8px;'>System Status</div>
    """, unsafe_allow_html=True)
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("<span style='color:#34D399; font-size:0.78rem;'>● Dashboard</span>", unsafe_allow_html=True)
    with col_b:
        st.markdown("<span style='color:#F59E0B; font-size:0.78rem;'>● API Demo</span>", unsafe_allow_html=True)

    st.markdown("<div style='height:20px;'></div>", unsafe_allow_html=True)
    st.markdown("""
    <div style='color:#334155; font-size:0.68rem; font-family:"Space Mono",monospace; text-align:center;
                line-height:1.6;'>
        Computer Vision · Evolutionary Algorithms<br>Cloud Computing · Deep Learning
    </div>""", unsafe_allow_html=True)

# ─── Page Router ──────────────────────────────────────────────────────────────

page = st.session_state.page

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — HOME
# ══════════════════════════════════════════════════════════════════════════════

if page == "Home":

    # Hero
    st.markdown("""
    <div class="hero-banner">
        <div class="hero-title">Vision Feature Selection<br>&amp; Neural Evolution Platform</div>
        <div class="hero-sub">
            A research platform that combines Genetic Algorithm-driven feature selection with
            Convolutional Neural Networks to achieve state-of-the-art image classification
            accuracy — deployed on scalable cloud infrastructure.
        </div>
        <div>
            <span class="badge">🔬 Computer Vision</span>
            <span class="badge badge-purple">🧬 Evolutionary Algorithms</span>
            <span class="badge badge-sky">☁️ Cloud Computing</span>
            <span class="badge">🤖 Deep Learning</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Summary metrics row
    results_df = load_results()
    summary   = compute_model_summary(results_df)

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Best Accuracy",     f"{summary['best_accuracy']:.2%}")
    with c2:
        st.metric("Baseline Average",  f"{summary['avg_baseline_acc']:.2%}")
    with c3:
        st.metric("GA Optimized Avg",  f"{summary['avg_optimized_acc']:.2%}")
    with c4:
        st.metric("Improvement",       f"+{summary['improvement_pct']:.1f}%")

    st.markdown("<div style='height:28px;'></div>", unsafe_allow_html=True)

    # Two-column layout: workflow + courses
    col_wf, col_tech = st.columns([3, 2], gap="large")

    with col_wf:
        st.markdown("<div class='vep-card'>", unsafe_allow_html=True)
        st.markdown("<div class='section-title'>🔄 Platform Workflow</div>"
                    "<div class='section-sub'>End-to-end pipeline from raw image to optimized prediction</div>",
                    unsafe_allow_html=True)
        steps = [
            ("01", "Image Ingestion",         "Upload JPEG/PNG images through the cloud-connected frontend dashboard."),
            ("02", "Feature Extraction",      "CNN backbone extracts spatial feature maps from raw pixel data."),
            ("03", "GA Feature Selection",    "Genetic Algorithm evolves a binary mask, selecting the most discriminative features."),
            ("04", "Model Training",          "Selected features train a compact optimized classifier, reducing overfitting."),
            ("05", "Cloud Inference",         "Trained model is deployed as a REST API on cloud infrastructure."),
            ("06", "Result Analytics",        "Predictions, confidence scores, and experiment metrics are visualized in the dashboard."),
        ]
        for num, title, desc in steps:
            st.markdown(f"""
            <div class="workflow-step">
                <div class="step-num">{num}</div>
                <div class="step-content">
                    <div class="step-title">{title}</div>
                    <div class="step-desc">{desc}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with col_tech:
        st.markdown("<div class='vep-card'>", unsafe_allow_html=True)
        st.markdown("<div class='section-title'>📚 Course Modules</div>"
                    "<div class='section-sub'>Academic disciplines integrated in this platform</div>",
                    unsafe_allow_html=True)

        courses = [
            ("👁️", "Computer Vision",          "#22D3EE",
             "CNN architectures, feature maps, image preprocessing, transfer learning."),
            ("🧬", "Evolutionary Algorithms",  "#A78BFA",
             "Genetic algorithms, selection, crossover, mutation, fitness evaluation."),
            ("☁️", "Cloud Computing",           "#38BDF8",
             "REST API deployment, scalable inference, containerization & monitoring."),
        ]
        for icon, name, color, desc in courses:
            st.markdown(f"""
            <div style='border:1px solid {color}30; border-radius:10px; padding:16px 18px;
                        margin-bottom:12px; background:rgba(255,255,255,0.02);'>
                <div style='color:{color}; font-weight:700; font-size:0.95rem; margin-bottom:6px;'>
                    {icon} {name}
                </div>
                <div style='color:#94A3B8; font-size:0.82rem; line-height:1.5;'>{desc}</div>
            </div>
            """, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

        # Tech stack
        st.markdown("<div class='vep-card'>", unsafe_allow_html=True)
        st.markdown("<div class='section-title' style='font-size:1.1rem;'>🛠️ Tech Stack</div>",
                    unsafe_allow_html=True)
        techs = ["Python 3.10", "TensorFlow/Keras", "DEAP (GA)", "Streamlit",
                 "Plotly", "FastAPI", "Docker", "GCP / AWS"]
        badges_html = "".join(
            f"<span class='badge {'badge-purple' if i%2 else ''}'>{t}</span>"
            for i, t in enumerate(techs)
        )
        st.markdown(badges_html, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # Quick-start CTA
    st.markdown("""
    <div style='background:linear-gradient(90deg,rgba(34,211,238,0.06),rgba(167,139,250,0.06));
                border:1px solid #1E3A5F; border-radius:12px; padding:20px 28px;
                display:flex; align-items:center; gap:20px; margin-top:8px;'>
        <div style='font-size:2rem;'>⚡</div>
        <div>
            <div style='font-weight:700; color:#E2E8F0; margin-bottom:4px;'>Ready to classify?</div>
            <div style='color:#94A3B8; font-size:0.86rem;'>
                Navigate to <strong style='color:#22D3EE;'>Prediction</strong> to upload an image and get
                real-time AI predictions, or visit <strong style='color:#A78BFA;'>Dashboard Analytics</strong>
                to explore experiment results.
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — PREDICTION
# ══════════════════════════════════════════════════════════════════════════════

elif page == "Prediction":

    st.markdown("<div class='section-title'>🔍 Image Prediction</div>"
                "<div class='section-sub'>Upload an image and classify it using the deployed GA-optimized CNN</div>",
                unsafe_allow_html=True)

    col_upload, col_result = st.columns([1, 1], gap="large")

    with col_upload:
        st.markdown("<div class='vep-card'>", unsafe_allow_html=True)
        st.markdown("#### 📁 Image Upload", unsafe_allow_html=False)

        uploaded = st.file_uploader(
            "Choose an image file",
            type=["jpg", "jpeg", "png"],
            help="Supported: JPEG, PNG — max 10 MB",
        )

        if uploaded:
            try:
                img = Image.open(uploaded)
                st.image(img, caption=f"{uploaded.name}  |  {img.size[0]}×{img.size[1]}px",
                         use_container_width=True)

                # File info
                size_kb = len(uploaded.getvalue()) / 1024
                st.markdown(f"""
                <div style='display:flex; gap:12px; margin-top:8px; flex-wrap:wrap;'>
                    <span class='badge'>📄 {uploaded.name}</span>
                    <span class='badge badge-sky'>{img.format or uploaded.type}</span>
                    <span class='badge badge-purple'>{size_kb:.1f} KB</span>
                </div>
                """, unsafe_allow_html=True)
            except Exception as e:
                st.error(f"Cannot read image: {e}")
                uploaded = None

        st.markdown("</div>", unsafe_allow_html=True)

        # Predict button
        if uploaded:
            st.markdown("<div style='height:8px;'></div>", unsafe_allow_html=True)
            predict_clicked = st.button("🚀  Run Prediction", type="primary",
                                        use_container_width=True)
        else:
            predict_clicked = False
            st.info("⬆️ Upload an image above to enable prediction.")

    with col_result:
        st.markdown("<div class='vep-card'>", unsafe_allow_html=True)
        st.markdown("#### 🎯 Prediction Result", unsafe_allow_html=False)

        if predict_clicked and uploaded:
            with st.spinner("Sending to cloud API…"):
                uploaded.seek(0)
                result = predict_image(uploaded)

            if "error" in result:
                st.error(f"❌ Prediction failed: {result['error']}")
            else:
                pred       = result.get("prediction", "Unknown")
                confidence = float(result.get("confidence", 0.0))
                model_used = result.get("model", "Unknown")

                st.success("✅ Prediction received successfully!")

                st.markdown(f"""
                <div class="pred-result">
                    <div style='color:#94A3B8; font-size:0.75rem; font-family:"Space Mono",monospace;
                                text-transform:uppercase; letter-spacing:0.1em; margin-bottom:4px;'>
                        Predicted Class
                    </div>
                    <div class="pred-class">{pred}</div>
                </div>
                """, unsafe_allow_html=True)

                st.markdown("<div style='height:16px;'></div>", unsafe_allow_html=True)

                m1, m2 = st.columns(2)
                with m1:
                    st.metric("Confidence", f"{confidence:.1%}")
                with m2:
                    st.metric("Model", model_used)

                st.markdown("**Confidence Score**")
                st.progress(confidence)

                if confidence >= 0.90:
                    st.success(f"High confidence prediction ({confidence:.1%})")
                elif confidence >= 0.75:
                    st.warning(f"Moderate confidence ({confidence:.1%}) — consider verifying")
                else:
                    st.error(f"Low confidence ({confidence:.1%}) — result may be unreliable")

        else:
            st.markdown("""
            <div style='text-align:center; padding:50px 20px; color:#334155;'>
                <div style='font-size:3rem; margin-bottom:12px;'>🎯</div>
                <div style='font-family:"Space Mono",monospace; font-size:0.85rem;'>
                    Prediction will appear here<br>after you click "Run Prediction"
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)

    # API info expander
    with st.expander("🔧 API Configuration & Debug Info"):
        from utils import API_URL
        st.code(f"Endpoint: {API_URL}", language="text")
        st.markdown("""
        **Expected response schema:**
        ```json
        {
          "prediction": "cat",
          "confidence": 0.97,
          "model": "GA Optimized CNN"
        }
        ```
        > **Demo mode**: When `API_URL` is set to `YOUR_API_URL`, the platform returns
        > mock predictions locally so the dashboard remains fully explorable without a
        > live backend.
        """)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — DASHBOARD ANALYTICS
# ══════════════════════════════════════════════════════════════════════════════



elif page == "Dashboard Analytics":

    st.markdown("<div class='section-title'>📊 Dashboard Analytics</div>"
                "<div class='section-sub'>Interactive experiment metrics and Genetic Algorithm visualizations</div>",
                unsafe_allow_html=True)

    results_df = load_results()
    ga_df      = load_ga_log()
    summary    = compute_model_summary(results_df)

    # Top metrics strip
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1: st.metric("Experiments",      summary["total_experiments"])
    with c2: st.metric("Best Accuracy",    f"{summary['best_accuracy']:.2%}")
    with c3: st.metric("Best Model",       summary["best_model"])
    with c4: st.metric("GA Improvement",   f"+{summary['improvement_pct']:.1f}%")
    with c5: st.metric("GA Generations",   len(ga_df) if not ga_df.empty else 0)

    st.markdown("<div style='height:20px;'></div>", unsafe_allow_html=True)

    # Tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "🏆 Model Comparison",
        "🧬 GA Convergence",
        "⚡ Runtime & Distribution",
        "🕸️ Metrics Radar",
    ])

    with tab1:
        st.markdown("#### Baseline CNN vs GA Optimized — Accuracy per Experiment")

        # CLEAN DATA FIRST (VERY IMPORTANT)
        results_df["model"] = results_df["model"].astype(str).str.strip()

        st.write("Models found:", results_df["model"].unique())

        fig = plot_accuracy_comparison(results_df)
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

        if not results_df.empty:

            st.markdown("<div style='height:8px;'></div>", unsafe_allow_html=True)
            c_a, c_b = st.columns(2)

            with c_a:
                baseline = results_df[
                    results_df["model"].str.contains("Baseline", na=False)
                ]["accuracy"]

                st.markdown("**Baseline CNN — Accuracy Stats**")
                st.dataframe(baseline.describe().to_frame(), use_container_width=True)

            with c_b:
                optimized = results_df[
                    results_df["model"].str.contains("GA", na=False)
                ]["accuracy"]

                st.markdown("**GA Optimized CNN — Accuracy Stats**")
                st.dataframe(optimized.describe().to_frame(), use_container_width=True)
    with tab2:
        st.markdown("#### Genetic Algorithm Convergence")
        col_l, col_r = st.columns([3, 1])
        with col_l:
            fig = plot_ga_convergence(ga_df)
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
        with col_r:
            if not ga_df.empty:
                final = ga_df.iloc[-1]
                st.metric("Final Best Fitness",   f"{final['best_fitness']:.4f}")
                st.metric("Final Avg Fitness",    f"{final['avg_fitness']:.4f}")
                st.metric("Features Selected",    int(final.get("selected_features", 0)))
                st.metric("Final Diversity",      f"{final.get('diversity_score', 0):.4f}")

        st.markdown("#### Feature Selection & Diversity Over Generations")
        fig2 = plot_feature_selection(ga_df)
        st.plotly_chart(fig2, use_container_width=True, config={"displayModeBar": False})

    with tab3:
        c_l, c_r = st.columns(2, gap="medium")
        
        with c_r:
            st.markdown("#### Accuracy Score Distribution")
            fig = plot_confidence_distribution(results_df)
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    with tab4:
        st.markdown("#### Multi-Metric Radar Overview")
        fig = plot_experiment_metrics_overview(results_df)
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

        if not results_df.empty:
            st.markdown("**Mean Metrics per Model**")
            metrics_cols = [c for c in ["accuracy", "f1_macro", "f1_weighted"]
                        if c in results_df.columns]
            pivot = results_df.groupby("model")[metrics_cols].mean().round(4)
            st.dataframe(pivot, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — EXPERIMENT LOGS
# ══════════════════════════════════════════════════════════════════════════════

elif page == "Experiment Logs":

    st.markdown("<div class='section-title'>📋 Experiment Logs</div>"
                "<div class='section-sub'>Raw data tables, metrics summaries, and CSV exports</div>",
                unsafe_allow_html=True)

    results_df = load_results()
    ga_df      = load_ga_log()

    log_tab1, log_tab2 = st.tabs(["📈 Evaluation Results", "🧬 GA Evolution Log"])

    # ── Evaluation Results ──────────────────────────────────────────────────

    with log_tab1:
        st.markdown("#### Evaluation Results — All Experiments")

        if results_df.empty:
            st.warning("No results data found. Ensure `evaluation/results.csv` exists.")
        else:
            # Filter controls
            fc1, fc2, fc3 = st.columns([2, 2, 2])
            with fc1:
                model_filter = st.multiselect(
                    "Filter by Model",
                    options=results_df["model"].unique().tolist(),
                    default=results_df["model"].unique().tolist(),
                )
            with fc2:
                min_acc = st.slider("Min Accuracy", 0.0, 1.0, 0.0, 0.01)
            with fc3:
                sort_col = st.selectbox("Sort by", ["experiment_id", "accuracy", "f1_score", "runtime_seconds"])

            filtered = results_df[
                (results_df["model"].isin(model_filter)) &
                (results_df["accuracy"] >= min_acc)
            ].sort_values(sort_col, ascending=False)

            st.markdown(f"<div style='color:#94A3B8; font-size:0.82rem; margin-bottom:6px;'>"
                        f"Showing {len(filtered)} of {len(results_df)} records</div>",
                        unsafe_allow_html=True)

            st.dataframe(
                filtered.reset_index(drop=True),
                use_container_width=True,
                height=400,
            )

            col_dl1, col_dl2 = st.columns([1, 4])
            with col_dl1:
                st.download_button(
                    label="⬇️ Download CSV",
                    data=dataframe_to_csv_bytes(filtered),
                    file_name="evaluation_results_filtered.csv",
                    mime="text/csv",
                    use_container_width=True,
                )

        # Summary stats expander
        with st.expander("📊 Summary Statistics"):
            if not results_df.empty:
                st.dataframe(results_df.describe().round(4), use_container_width=True)

    # ── GA Evolution Log ────────────────────────────────────────────────────

    with log_tab2:
        st.markdown("#### Genetic Algorithm Evolution Log")

        if ga_df.empty:
            st.warning("No GA log data found. Ensure `ga_log.csv` exists.")
        else:
            g1, g2, g3 = st.columns([2, 2, 2])
            with g1:
                gen_range = st.slider(
                    "Generation Range",
                    int(ga_df["generation"].min()),
                    int(ga_df["generation"].max()),
                    (int(ga_df["generation"].min()), int(ga_df["generation"].max())),
                )
            with g2:
                min_fitness = st.slider("Min Best Fitness", 0.0, 1.0, 0.0, 0.01)
            with g3:
                ga_sort = st.selectbox("Sort by", ["generation", "best_fitness", "avg_fitness", "selected_features"])

            ga_filtered = ga_df[
                (ga_df["generation"] >= gen_range[0]) &
                (ga_df["generation"] <= gen_range[1]) &
                (ga_df["best_fitness"] >= min_fitness)
            ].sort_values(ga_sort, ascending=False)

            st.markdown(f"<div style='color:#94A3B8; font-size:0.82rem; margin-bottom:6px;'>"
                        f"Showing {len(ga_filtered)} generations</div>",
                        unsafe_allow_html=True)

            st.dataframe(
                ga_filtered.reset_index(drop=True),
                use_container_width=True,
                height=400,
            )

            col_ga1, _ = st.columns([1, 4])
            with col_ga1:
                st.download_button(
                    label="⬇️ Download CSV",
                    data=dataframe_to_csv_bytes(ga_filtered),
                    file_name="ga_log_filtered.csv",
                    mime="text/csv",
                    use_container_width=True,
                )

        with st.expander("📊 GA Summary Statistics"):
            if not ga_df.empty:
                st.dataframe(ga_df.describe().round(4), use_container_width=True)

    # ── Deployment Guide ────────────────────────────────────────────────────

    st.markdown("<hr class='vep-divider'>", unsafe_allow_html=True)
    with st.expander("🚀 Deployment Instructions — Streamlit Cloud"):
        st.markdown("""
        ### Deploying to Streamlit Cloud

        **1. Prepare your repository**
        ```
        your-repo/
        ├── dashboard/
        │   ├── app.py
        │   ├── utils.py
        │   ├── charts.py
        │   ├── evaluation/results.csv
        │   ├── ga_log.csv
        │   └── requirements.txt
        ```

        **2. Push to GitHub**
        ```bash
        git init && git add . && git commit -m "init"
        git remote add origin https://github.com/USERNAME/REPO.git
        git push -u origin main
        ```

        **3. Deploy on [share.streamlit.io](https://share.streamlit.io)**
        - Connect your GitHub account
        - Select repository & branch
        - Set **Main file path** to `dashboard/app.py`
        - Click **Deploy**

        **4. Set your API URL**
        - In Streamlit Cloud → App Settings → Secrets:
        ```toml
        [api]
        url = "https://your-cloud-api.example.com/predict"
        ```
        - Update `utils.py` to read from `st.secrets["api"]["url"]`

        **5. Local development**
        ```bash
        cd dashboard
        pip install -r requirements.txt
        streamlit run app.py
        ```
        """)
