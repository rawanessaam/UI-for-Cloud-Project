"""
charts.py — Plotly chart factory for the Vision Evolution Platform dashboard.
All chart functions return a Plotly Figure object ready for st.plotly_chart().
"""

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np

# ─── Shared Theme ─────────────────────────────────────────────────────────────

_PALETTE = {
    "baseline":  "#38BDF8",   # sky blue
    "optimized": "#A78BFA",   # purple
    "accent":    "#22D3EE",   # cyan
    "warning":   "#F59E0B",   # amber
    "success":   "#34D399",   # emerald
    "bg":        "#0F172A",   # deep navy
    "surface":   "#1E293B",   # slate
    "border":    "#334155",   # subtle border
    "text":      "#E2E8F0",   # light slate
}

_LAYOUT_BASE = dict(
    paper_bgcolor=_PALETTE["bg"],
    plot_bgcolor=_PALETTE["surface"],
    font=dict(family="'JetBrains Mono', monospace", color=_PALETTE["text"], size=12),
    margin=dict(l=40, r=30, t=50, b=40),
    legend=dict(
        bgcolor="rgba(30,41,59,0.8)",
        bordercolor=_PALETTE["border"],
        borderwidth=1,
    ),
)


# ─── 1. Baseline vs Optimised Accuracy ────────────────────────────────────────

def plot_accuracy_comparison(results: pd.DataFrame) -> go.Figure:
    """
    Grouped bar chart comparing Baseline CNN vs GA Optimized CNN accuracy
    across experiments.
    """
    if results.empty:
        return _empty_figure("No experiment data available")

    df = results[["experiment_id", "model", "accuracy"]].copy()
    df["experiment_id"] = df["experiment_id"].astype(str)

    fig = px.bar(
        df,
        x="experiment_id",
        y="accuracy",
        color="model",
        barmode="group",
        color_discrete_map={
            "Baseline CNN":       _PALETTE["baseline"],
            "GA Optimized CNN":   _PALETTE["optimized"],
        },
        labels={"experiment_id": "Experiment #", "accuracy": "Accuracy", "model": "Model"},
        title="Baseline CNN vs GA Optimized CNN — Accuracy per Experiment",
    )

    fig.update_traces(marker_line_width=0, opacity=0.92)
    fig.update_yaxes(range=[0.6, 1.0], tickformat=".0%", gridcolor=_PALETTE["border"])
    fig.update_xaxes(gridcolor="rgba(0,0,0,0)")
    fig.update_layout(**_LAYOUT_BASE)

    # Reference line at baseline mean
    baseline_mean = results[results["model"] == "Baseline CNN"]["accuracy"].mean()
    if not np.isnan(baseline_mean):
        fig.add_hline(
            y=baseline_mean, line_dash="dot",
            line_color=_PALETTE["baseline"],
            annotation_text=f"Baseline avg {baseline_mean:.2%}",
            annotation_font_color=_PALETTE["baseline"],
        )

    return fig


# ─── 2. GA Convergence Graph ───────────────────────────────────────────────────

def plot_ga_convergence(ga_log: pd.DataFrame) -> go.Figure:
    """
    Line chart showing best / average / worst fitness over GA generations,
    with a shaded band for the fitness range.
    """
    if ga_log.empty:
        return _empty_figure("No GA log data available")

    fig = go.Figure()

    # Fitness range shading
    fig.add_trace(go.Scatter(
        x=pd.concat([ga_log["generation"], ga_log["generation"][::-1]]),
        y=pd.concat([ga_log["best_fitness"], ga_log["worst_fitness"][::-1]]),
        fill="toself",
        fillcolor="rgba(167,139,250,0.12)",
        line=dict(color="rgba(0,0,0,0)"),
        name="Fitness Range",
        showlegend=True,
    ))

    # Worst fitness
    fig.add_trace(go.Scatter(
        x=ga_log["generation"], y=ga_log["worst_fitness"],
        mode="lines",
        line=dict(color=_PALETTE["warning"], width=1, dash="dot"),
        name="Worst Fitness",
    ))

    # Average fitness
    fig.add_trace(go.Scatter(
        x=ga_log["generation"], y=ga_log["avg_fitness"],
        mode="lines",
        line=dict(color=_PALETTE["accent"], width=2),
        name="Avg Fitness",
    ))

    # Best fitness (hero line)
    fig.add_trace(go.Scatter(
        x=ga_log["generation"], y=ga_log["best_fitness"],
        mode="lines+markers",
        marker=dict(size=5, color=_PALETTE["optimized"]),
        line=dict(color=_PALETTE["optimized"], width=3),
        name="Best Fitness",
    ))

    fig.update_yaxes(
        range=[0.5, 1.0], tickformat=".2f",
        gridcolor=_PALETTE["border"], title="Fitness Score",
    )
    fig.update_xaxes(gridcolor="rgba(0,0,0,0)", title="Generation")
    fig.update_layout(title="GA Convergence — Fitness Over Generations", **_LAYOUT_BASE)

    return fig


# ─── 3. Runtime Comparison ────────────────────────────────────────────────────

def plot_runtime_comparison(results: pd.DataFrame) -> go.Figure:
    """
    Horizontal box plot comparing training runtime (seconds) per model type.
    """
    if results.empty:
        return _empty_figure("No runtime data available")

    fig = go.Figure()

    for model, color in [
        ("Baseline CNN",     _PALETTE["baseline"]),
        ("GA Optimized CNN", _PALETTE["optimized"]),
    ]:
        subset = results[results["model"] == model]["runtime_seconds"].dropna()
        if subset.empty:
            continue
        fig.add_trace(go.Box(
            x=subset,
            name=model,
            marker_color=color,
            line_color=color,
            fillcolor=color.replace(")", ",0.25)").replace("rgb", "rgba") if color.startswith("rgb") else color + "40",
            boxmean="sd",
            orientation="h",
        ))

    fig.update_xaxes(title="Runtime (seconds)", gridcolor=_PALETTE["border"])
    fig.update_yaxes(gridcolor="rgba(0,0,0,0)")
    fig.update_layout(title="Training Runtime Distribution by Model", **_LAYOUT_BASE)

    return fig


# ─── 4. Confidence Distribution ───────────────────────────────────────────────

def plot_confidence_distribution(results: pd.DataFrame) -> go.Figure:
    """
    Overlapping histogram of model accuracy scores to visualise confidence spread.
    If an 'f1_score' column exists it is overlaid as a KDE-style line.
    """
    if results.empty:
        return _empty_figure("No data to plot")

    fig = go.Figure()

    for model, color in [
        ("Baseline CNN",     _PALETTE["baseline"]),
        ("GA Optimized CNN", _PALETTE["optimized"]),
    ]:
        subset = results[results["model"] == model]["accuracy"].dropna()
        if subset.empty:
            continue
        fig.add_trace(go.Histogram(
            x=subset,
            name=model,
            nbinsx=12,
            marker_color=color,
            opacity=0.72,
        ))

    fig.update_layout(
        barmode="overlay",
        title="Accuracy Score Distribution by Model",
        xaxis=dict(title="Accuracy", tickformat=".2f", gridcolor=_PALETTE["border"]),
        yaxis=dict(title="Count", gridcolor=_PALETTE["border"]),
        **_LAYOUT_BASE,
    )

    return fig


# ─── 5. Experiment Metrics Overview (radar) ───────────────────────────────────

def plot_experiment_metrics_overview(results: pd.DataFrame) -> go.Figure:
    """
    Radar / spider chart comparing mean metric values across the two model types.
    Metrics: accuracy, precision, recall, f1_score (normalised 0-1).
    """
    metrics = ["accuracy", "precision", "recall", "f1_score"]
    available = [m for m in metrics if m in results.columns]

    if results.empty or not available:
        return _empty_figure("No metrics data available")

    fig = go.Figure()

    for model, color in [
        ("Baseline CNN",     _PALETTE["baseline"]),
        ("GA Optimized CNN", _PALETTE["optimized"]),
    ]:
        subset = results[results["model"] == model][available].mean().values.tolist()
        # Close the radar loop
        fig.add_trace(go.Scatterpolar(
            r=subset + [subset[0]],
            theta=[m.replace("_", " ").title() for m in available] + [available[0].replace("_", " ").title()],
            fill="toself",
            fillcolor=color + "33",
            line=dict(color=color, width=2),
            name=model,
        ))

    fig.update_layout(
        title="Experiment Metrics Radar — Mean Values",
        polar=dict(
            radialaxis=dict(
                visible=True, range=[0.6, 1.0],
                tickformat=".0%",
                gridcolor=_PALETTE["border"],
                linecolor=_PALETTE["border"],
            ),
            angularaxis=dict(gridcolor=_PALETTE["border"]),
            bgcolor=_PALETTE["surface"],
        ),
        **_LAYOUT_BASE,
    )

    return fig


# ─── 6. Feature Selection Over GA Generations ─────────────────────────────────

def plot_feature_selection(ga_log: pd.DataFrame) -> go.Figure:
    """
    Dual-axis line chart: selected features (left) and diversity score (right)
    as the GA evolves.
    """
    if ga_log.empty or "selected_features" not in ga_log.columns:
        return _empty_figure("No feature selection data available")

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig.add_trace(go.Scatter(
        x=ga_log["generation"], y=ga_log["selected_features"],
        mode="lines+markers",
        marker=dict(size=5, color=_PALETTE["accent"]),
        line=dict(color=_PALETTE["accent"], width=2),
        name="Selected Features",
    ), secondary_y=False)

    if "diversity_score" in ga_log.columns:
        fig.add_trace(go.Scatter(
            x=ga_log["generation"], y=ga_log["diversity_score"],
            mode="lines",
            line=dict(color=_PALETTE["warning"], width=2, dash="dash"),
            name="Population Diversity",
        ), secondary_y=True)

    fig.update_xaxes(title_text="Generation", gridcolor="rgba(0,0,0,0)")
    fig.update_yaxes(title_text="Selected Features", gridcolor=_PALETTE["border"], secondary_y=False)
    fig.update_yaxes(title_text="Diversity Score", gridcolor="rgba(0,0,0,0)", secondary_y=True)
    fig.update_layout(title="Feature Selection & Population Diversity Over Generations", **_LAYOUT_BASE)

    return fig


# ─── Internal Helpers ─────────────────────────────────────────────────────────

def _empty_figure(message: str) -> go.Figure:
    """Return a styled empty chart with a centred message."""
    fig = go.Figure()
    fig.add_annotation(
        text=message,
        xref="paper", yref="paper",
        x=0.5, y=0.5,
        showarrow=False,
        font=dict(size=16, color=_PALETTE["text"]),
    )
    fig.update_layout(**_LAYOUT_BASE)
    return fig
