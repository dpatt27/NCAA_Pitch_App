"""
app.py  —  NCAA D1 Pitch Intelligence App
Run with:  streamlit run app.py
"""

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from utils import FEATURES, PITCH_TYPE_MAP

# ── Page config ──────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Pitch Intelligence",
    page_icon="⚾",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ───────────────────────────────────────────────────────────────

st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=Inter:wght@300;400;600;700&display=swap');

  html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

  /* Sidebar */
  [data-testid="stSidebar"] {
    background: #0d1117;
    border-right: 1px solid #21262d;
  }
  [data-testid="stSidebar"] * { color: #e6edf3 !important; }

  /* Main background */
  .main { background: #010409; }
  .block-container { padding-top: 2rem; }

  /* Metric cards */
  .metric-card {
    background: #0d1117;
    border: 1px solid #21262d;
    border-radius: 8px;
    padding: 1.1rem 1.4rem;
    text-align: center;
  }
  .metric-label {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.65rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #8b949e;
    margin-bottom: 0.3rem;
  }
  .metric-value {
    font-size: 1.8rem;
    font-weight: 700;
    color: #e6edf3;
    line-height: 1;
  }
  .metric-sub {
    font-size: 0.75rem;
    color: #8b949e;
    margin-top: 0.2rem;
  }

  /* Section headers */
  .section-header {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.85rem;
    font-weight: 600;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #79c0ff;
    border-bottom: 2px solid #388bfd;
    padding-bottom: 0.5rem;
    margin-bottom: 1.1rem;
  }

  /* Page titles */
  h1 { font-size: 2.4rem !important; color: #ffffff !important; font-weight: 800 !important; letter-spacing: -0.02em !important; }
  h2 { font-size: 1.6rem !important; color: #e6edf3 !important; font-weight: 700 !important; }
  h3 { font-size: 1.2rem !important; color: #c9d1d9 !important; font-weight: 600 !important; }

  /* Pitch type badge */
  .pitch-badge {
    display: inline-block;
    background: #1f2937;
    border: 1px solid #374151;
    border-radius: 4px;
    padding: 2px 8px;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.72rem;
    color: #93c5fd;
    margin: 2px;
  }

  /* Gauge container */
  .gauge-wrap { text-align: center; padding: 1rem 0; }

  /* Dataframe styling */
  [data-testid="stDataFrame"] { border: 1px solid #21262d; border-radius: 8px; }



  /* Selectbox / slider labels */
  label { color: #8b949e !important; font-size: 0.8rem !important; }
</style>
""", unsafe_allow_html=True)

# ── Auto-train on first deploy ───────────────────────────────────────────────

import os
import subprocess

if not os.path.exists("model.pkl"):
    with st.spinner("Training models on first launch… this takes ~30 seconds."):
        import sys
        script_dir = os.path.dirname(os.path.abspath(__file__))
        subprocess.run([sys.executable, os.path.join(script_dir, "model.py")],
                       check=True, cwd=script_dir)

# ── Load artifacts ────────────────────────────────────────────────────────────

@st.cache_resource
def load_artifacts():
    model          = joblib.load("model.pkl")
    scaler         = joblib.load("scaler.pkl")
    kmeans         = joblib.load("kmeans.pkl")
    kmeans_scaler  = joblib.load("kmeans_scaler.pkl")
    df             = joblib.load("clean_data.pkl")
    pitcher_summary= joblib.load("pitcher_summary.pkl")
    return model, scaler, kmeans, kmeans_scaler, df, pitcher_summary

model, scaler, kmeans, kmeans_scaler, df, pitcher_summary = load_artifacts()

# Reverse pitch type map for display
PITCH_TYPE_LABELS = {v: k for k, v in PITCH_TYPE_MAP.items()}

PITCH_COLORS = {
    "Fastball":  "#ff4444",  # vivid red
    "FourSeam":  "#ff4444",  # vivid red
    "Sinker":    "#ff8c00",  # dark orange
    "Cutter":    "#ffd700",  # gold
    "Slider":    "#00e676",  # bright green
    "Sweeper":   "#00bcd4",  # cyan
    "Slurve":    "#40c4ff",  # light blue
    "Curveball": "#2979ff",  # vivid blue
    "Gyro":      "#d500f9",  # vivid purple
    "Reaper":    "#aa00ff",  # deep violet
    "ChangeUp":  "#ff4081",  # hot pink
    "Splitter":  "#ffab40",  # amber
}

CLUSTER_LABELS = {
    0: "Finesse Arms",
    1: "Mid-Tier Starters",
    2: "Control Specialists",
    3: "Power Pitchers",
}

KMEANS_FEATURES = [
    "avg_velo", "avg_spin", "avg_ivb", "avg_hb",
    "avg_extension", "n_pitch_types", "overall_whiff_rate",
]

# ── Sidebar nav ───────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## ⚾ Pitch Intelligence")
    st.markdown("<div style='font-size:0.75rem;color:#8b949e;margin-bottom:1.5rem'>NCAA D1 · 2026 Season</div>", unsafe_allow_html=True)

    page = st.radio(
        "Navigate",
        ["Pitcher Dashboard", "Pitch Quality Scorer", "Arsenal Comparison"],
        label_visibility="collapsed",
    )

    st.markdown("---")
    st.markdown("<div style='font-size:0.7rem;color:#8b949e'>Model: XGBoost · AUC 0.775<br>Data: 25,010 pitches · 189 pitchers</div>", unsafe_allow_html=True)


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  VIEW 1 — PITCHER DASHBOARD                                              ║
# ╚══════════════════════════════════════════════════════════════════════════╝

if page == "Pitcher Dashboard":

    col_title, col_select = st.columns([2, 2])
    with col_title:
        st.markdown("# Pitcher Dashboard")
    with col_select:
        pitcher_names = sorted(df["Pitcher"].dropna().unique())
        selected = st.selectbox("Select pitcher", pitcher_names, label_visibility="collapsed")

    st.markdown("---")

    p_df    = df[df["Pitcher"] == selected]
    p_swings = p_df[p_df["is_swing"] == True]
    p_sum   = pitcher_summary[pitcher_summary["Pitcher"] == selected]

    # ── Top metrics row ──
    hand      = p_df["PitcherThrows"].iloc[0] if len(p_df) else "—"
    team      = p_df["PitcherTeam"].iloc[0] if len(p_df) else "—"
    avg_velo  = p_df["RelSpeed"].mean()
    whiff_rt  = p_swings["whiff"].mean() * 100 if len(p_swings) else 0
    n_pitches = len(p_df)
    n_types   = p_df["TaggedPitchType"].nunique()
    cluster   = int(p_sum["cluster"].iloc[0]) if len(p_sum) else -1
    cluster_label = CLUSTER_LABELS.get(cluster, "—")

    c1, c2, c3, c4, c5 = st.columns(5)
    for col, label, value, sub in [
        (c1, "Team",         team,                  f"{hand}HP"),
        (c2, "Avg Velo",     f"{avg_velo:.1f}",     "mph"),
        (c3, "Whiff Rate",   f"{whiff_rt:.1f}%",    "on swings"),
        (c4, "Pitches",      f"{n_pitches:,}",      f"{n_types} pitch types"),
        (c5, "Profile",      cluster_label,         f"Cluster {cluster}"),
    ]:
        col.markdown(f"""
        <div class="metric-card">
          <div class="metric-label">{label}</div>
          <div class="metric-value">{value}</div>
          <div class="metric-sub">{sub}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Arsenal table + usage pie ──
    st.markdown('<div class="section-header">Arsenal Breakdown</div>', unsafe_allow_html=True)

    arsenal = (
        p_swings.groupby("TaggedPitchType")
        .agg(
            Pitches    = ("whiff", "count"),
            Velo       = ("RelSpeed", "mean"),
            Spin       = ("SpinRate", "mean"),
            IVB        = ("InducedVertBreak", "mean"),
            HB         = ("HorzBreak", "mean"),
            WhiffPct   = ("whiff", "mean"),
        )
        .reset_index()
        .rename(columns={"TaggedPitchType": "Pitch Type"})
    )
    total = arsenal["Pitches"].sum()
    arsenal["Usage%"] = (arsenal["Pitches"] / total * 100).round(1)
    arsenal["Velo"]   = arsenal["Velo"].round(1)
    arsenal["Spin"]   = arsenal["Spin"].round(0).astype(int)
    arsenal["IVB"]    = arsenal["IVB"].round(1)
    arsenal["HB"]     = arsenal["HB"].round(1)
    arsenal["Whiff%"] = (arsenal["WhiffPct"] * 100).round(1)
    arsenal = arsenal.drop(columns=["WhiffPct"]).sort_values("Usage%", ascending=False)

    left, right = st.columns([3, 2])

    with left:
        st.dataframe(
            arsenal[["Pitch Type","Usage%","Velo","Spin","IVB","HB","Whiff%"]],
            use_container_width=True,
            hide_index=True,
        )

    with right:
        fig_pie = px.pie(
            arsenal, names="Pitch Type", values="Usage%",
            color="Pitch Type",
            color_discrete_map=PITCH_COLORS,
            hole=0.55,
        )
        fig_pie.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font_color="#8b949e",
            showlegend=True,
            legend=dict(font=dict(size=11)),
            margin=dict(t=10, b=10, l=10, r=10),
            height=260,
        )
        fig_pie.update_traces(textinfo="none")
        st.plotly_chart(fig_pie, use_container_width=True)

    # ── Zone heat map ──
    outcome_filter = st.radio(
        "Color by", ["Whiff", "All swings"], horizontal=True, label_visibility="collapsed"
    )

    if outcome_filter == "Whiff":
        zone_df = p_swings[p_swings["whiff"] == 1]
        color_label = "Whiff"
    else:
        zone_df = p_swings.copy()
        zone_df["Whiff"] = zone_df["whiff"].map({1: "Whiff", 0: "Contact/Foul"})
        color_label = "Whiff"

    fig_zone = px.scatter(
        zone_df,
        x="PlateLocSide", y="PlateLocHeight",
        color="TaggedPitchType",
        color_discrete_map=PITCH_COLORS,
        opacity=0.7,
        labels={"PlateLocSide": "Horizontal Location (ft)", "PlateLocHeight": "Height (ft)", "TaggedPitchType": "Pitch Type"},
    )

    # ── Strike zone — outer border ──
    fig_zone.add_shape(type="rect", x0=-0.8333, x1=0.8333, y0=1.5, y1=3.5,
                       line=dict(color="#ffffff", width=2.5),
                       fillcolor="rgba(255,255,255,0.03)")

    # Inner 3×3 grid lines (horizontal)
    for y in [1.5 + (3.5 - 1.5) / 3, 1.5 + 2 * (3.5 - 1.5) / 3]:
        fig_zone.add_shape(type="line", x0=-0.8333, x1=0.8333, y0=y, y1=y,
                           line=dict(color="rgba(255,255,255,0.25)", width=1))

    # Inner 3×3 grid lines (vertical)
    for x in [-0.8333 + (1.6666) / 3, -0.8333 + 2 * (1.6666) / 3]:
        fig_zone.add_shape(type="line", x0=x, x1=x, y0=1.5, y1=3.5,
                           line=dict(color="rgba(255,255,255,0.25)", width=1))

    fig_zone.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#0d1117",
        font_color="#8b949e",
        xaxis=dict(gridcolor="#21262d", range=[-2.5, 2.5], zeroline=False, scaleanchor="y", scaleratio=1),
        yaxis=dict(gridcolor="#21262d", range=[0, 5], zeroline=False),
        margin=dict(t=10, b=40, l=40, r=10),
        height=460,
        legend=dict(font=dict(size=11)),
    )

    # ── Movement profile ──
    fig_mov = px.scatter(
        p_df.dropna(subset=["HorzBreak","InducedVertBreak"]),
        x="HorzBreak", y="InducedVertBreak",
        color="TaggedPitchType",
        color_discrete_map=PITCH_COLORS,
        opacity=0.7,
        labels={"HorzBreak": "Horizontal Break (in)", "InducedVertBreak": "Induced Vert Break (in)", "TaggedPitchType": "Pitch Type"},
    )
    fig_mov.add_hline(y=0, line_color="#444d56", line_width=1)
    fig_mov.add_vline(x=0, line_color="#444d56", line_width=1)
    fig_mov.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#0d1117",
        font_color="#8b949e",
        xaxis=dict(gridcolor="#21262d", zeroline=False),
        yaxis=dict(gridcolor="#21262d", zeroline=False),
        margin=dict(t=10, b=40, l=40, r=10),
        height=460,
        legend=dict(font=dict(size=11)),
    )

    # ── Side by side ──
    plot_col1, plot_col2 = st.columns([1, 1])
    with plot_col1:
        st.markdown('<div class="section-header">Pitch Location by Outcome</div>', unsafe_allow_html=True)
        st.plotly_chart(fig_zone, use_container_width=True)
    with plot_col2:
        st.markdown('<div class="section-header">Movement Profile</div>', unsafe_allow_html=True)
        st.plotly_chart(fig_mov, use_container_width=True)


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  VIEW 2 — PITCH QUALITY SCORER                                           ║
# ╚══════════════════════════════════════════════════════════════════════════╝

elif page == "Pitch Quality Scorer":

    st.markdown("# Pitch Quality Scorer")
    st.markdown("<div style='color:#8b949e;margin-bottom:1.5rem'>Adjust pitch metrics below to get a predicted whiff probability from the trained XGBoost model.</div>", unsafe_allow_html=True)
    st.markdown("---")

    left, right = st.columns([1, 1])

    with left:
        st.markdown('<div class="section-header">Pitch Characteristics</div>', unsafe_allow_html=True)

        pitch_type = st.selectbox("Pitch Type", list(PITCH_TYPE_MAP.keys()))
        hand       = st.radio("Pitcher Hand", ["Right", "Left"], horizontal=True)

        velo      = st.slider("Velocity (mph)",            60.0, 100.0, 88.0, 0.1)
        zone_spd  = st.slider("Zone Speed (mph)",          55.0,  95.0, 83.0, 0.1)
        spin      = st.slider("Spin Rate (rpm)",          1000,   3500,  2200,  10)
        spin_axis = st.slider("Spin Axis (degrees)",         0,    360,   200,   1)

        st.markdown('<div class="section-header" style="margin-top:1rem">Movement</div>', unsafe_allow_html=True)

        ivb  = st.slider("Induced Vert Break (in)", -20.0, 25.0, 12.0, 0.1)
        hb   = st.slider("Horizontal Break (in)",   -20.0, 20.0,  5.0, 0.1)
        vb   = st.slider("Vert Break (in)",         -20.0, 25.0,  8.0, 0.1)

        st.markdown('<div class="section-header" style="margin-top:1rem">Release & Approach</div>', unsafe_allow_html=True)

        ext       = st.slider("Extension (ft)",              4.0,  7.5,  6.2, 0.1)
        rel_h     = st.slider("Release Height (ft)",         4.5,  7.0,  5.8, 0.1)
        rel_side  = st.slider("Release Side (ft)",          -3.0,  3.0,  1.5 if hand == "Right" else -1.5, 0.1)
        horz_rel  = st.slider("Horiz Release Angle (deg)",  -5.0,  5.0, -1.0, 0.1)
        vert_rel  = st.slider("Vert Release Angle (deg)",   -5.0,  5.0, -3.0, 0.1)
        vert_appr = st.slider("Vert Approach Angle (deg)", -10.0,  0.0, -5.0, 0.1)
        horz_appr = st.slider("Horiz Approach Angle (deg)", -5.0,  5.0, -1.0, 0.1)

    with right:
        st.markdown('<div class="section-header">Predicted Whiff Probability</div>', unsafe_allow_html=True)

        # Build feature vector — must match FEATURES order in utils.py
        pt_encoded = PITCH_TYPE_MAP.get(pitch_type, 0)
        X_input = pd.DataFrame([{
            "RelSpeed": velo, "SpinRate": spin, "SpinAxis": spin_axis,
            "ZoneSpeed": zone_spd, "InducedVertBreak": ivb, "HorzBreak": hb,
            "VertBreak": vb, "Extension": ext, "RelHeight": rel_h,
            "RelSide": rel_side, "HorzRelAngle": horz_rel, "VertRelAngle": vert_rel,
            "VertApprAngle": vert_appr, "HorzApprAngle": horz_appr,
            "pitch_type_encoded": pt_encoded,
        }])
        X_scaled = scaler.transform(X_input)
        whiff_prob = model.predict_proba(X_scaled)[0][1]
        whiff_pct  = whiff_prob * 100

        # Color thresholds
        if whiff_pct >= 45:
            gauge_color = "#22c55e"
            rating = "Elite"
        elif whiff_pct >= 30:
            gauge_color = "#eab308"
            rating = "Above Average"
        elif whiff_pct >= 20:
            gauge_color = "#f97316"
            rating = "Average"
        else:
            gauge_color = "#ef4444"
            rating = "Below Average"

        # Gauge chart
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=whiff_pct,
            number={"suffix": "%", "font": {"size": 42, "color": "#e6edf3", "family": "IBM Plex Mono"}},
            gauge={
                "axis":      {"range": [0, 80], "tickcolor": "#8b949e", "tickfont": {"color": "#8b949e"}},
                "bar":       {"color": gauge_color, "thickness": 0.25},
                "bgcolor":   "#0d1117",
                "bordercolor": "#21262d",
                "steps": [
                    {"range": [0,  20], "color": "#1a0a0a"},
                    {"range": [20, 30], "color": "#1a1200"},
                    {"range": [30, 45], "color": "#0a1a0a"},
                    {"range": [45, 80], "color": "#0d1a0d"},
                ],
                "threshold": {"line": {"color": gauge_color, "width": 3}, "value": whiff_pct},
            },
        ))
        fig_gauge.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            font_color="#8b949e",
            height=280,
            margin=dict(t=30, b=0, l=30, r=30),
        )
        st.plotly_chart(fig_gauge, use_container_width=True)

        st.markdown(f"""
        <div style="text-align:center;margin-top:-1rem">
          <div style="font-size:1.5rem;font-weight:700;color:{gauge_color}">{rating}</div>
          <div style="font-size:0.8rem;color:#8b949e;margin-top:0.3rem">NCAA D1 average: ~23.5% on swings</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="section-header">Feature Summary</div>', unsafe_allow_html=True)

        summary_df = pd.DataFrame({
            "Metric": ["Velocity","Zone Speed","Spin Rate","Spin Axis","IVB","HB","VB","Extension","Rel Height","Rel Side"],
            "Value":  [f"{velo} mph", f"{zone_spd} mph", f"{spin} rpm", f"{spin_axis}°",
                       f"{ivb} in", f"{hb} in", f"{vb} in", f"{ext} ft", f"{rel_h} ft", f"{rel_side} ft"],
        })
        st.dataframe(summary_df, use_container_width=True, hide_index=True)

        # Compare to dataset average for this pitch type
        st.markdown('<div class="section-header" style="margin-top:1rem">vs. NCAA D1 Avg ({})'.format(pitch_type) + '</div>', unsafe_allow_html=True)
        comp = df[df["TaggedPitchType"] == pitch_type][["RelSpeed","SpinRate","InducedVertBreak","HorzBreak","VertBreak","Extension"]].mean()
        if not comp.isna().all():
            cdf = pd.DataFrame({
                "Metric": ["Velo","Spin","IVB","HB","VB","Extension"],
                "Yours":  [velo, spin, ivb, hb, vb, ext],
                "D1 Avg": [
                    round(comp["RelSpeed"],1), round(comp["SpinRate"],0),
                    round(comp["InducedVertBreak"],1), round(comp["HorzBreak"],1),
                    round(comp["VertBreak"],1), round(comp["Extension"],1),
                ],
            })
            cdf["Δ"] = (cdf["Yours"] - cdf["D1 Avg"]).round(1)
            st.dataframe(cdf, use_container_width=True, hide_index=True)


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  VIEW 3 — ARSENAL COMPARISON                                             ║
# ╚══════════════════════════════════════════════════════════════════════════╝

elif page == "Arsenal Comparison":

    st.markdown("# Arsenal Comparison")
    st.markdown("<div style='color:#8b949e;margin-bottom:1.5rem'>KMeans clustering of 189 pitchers by arsenal profile. Select a cluster to explore who's in it.</div>", unsafe_allow_html=True)
    st.markdown("---")

    ps = pitcher_summary.copy()
    ps["Cluster Name"] = ps["cluster"].map(CLUSTER_LABELS)
    ps["overall_whiff_rate"] = (ps["overall_whiff_rate"] * 100).round(1)

    # ── Scatter plot ──
    st.markdown('<div class="section-header">Velocity vs. Whiff Rate by Cluster</div>', unsafe_allow_html=True)

    fig_scatter = px.scatter(
        ps,
        x="avg_velo",
        y="overall_whiff_rate",
        color="Cluster Name",
        size="total_pitches",
        hover_name="Pitcher",
        hover_data={"avg_spin": True, "n_pitch_types": True, "avg_velo": ":.1f", "overall_whiff_rate": ":.1f"},
        color_discrete_sequence=["#3b82f6","#22c55e","#f97316","#a855f7"],
        labels={
            "avg_velo": "Avg Velocity (mph)",
            "overall_whiff_rate": "Whiff Rate (%)",
            "Cluster Name": "Cluster",
        },
        size_max=20,
    )
    fig_scatter.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#0d1117",
        font_color="#8b949e",
        xaxis=dict(gridcolor="#21262d", zeroline=False),
        yaxis=dict(gridcolor="#21262d", zeroline=False),
        height=420,
        margin=dict(t=10, b=40, l=40, r=10),
        legend=dict(font=dict(size=11)),
    )
    st.plotly_chart(fig_scatter, use_container_width=True)

    # ── Cluster summaries ──
    st.markdown('<div class="section-header">Cluster Averages</div>', unsafe_allow_html=True)

    cluster_agg = (
        ps.groupby("Cluster Name")
        .agg(
            Pitchers      = ("Pitcher", "count"),
            Avg_Velo      = ("avg_velo", "mean"),
            Avg_Spin      = ("avg_spin", "mean"),
            Avg_IVB       = ("avg_ivb", "mean"),
            Avg_HB        = ("avg_hb", "mean"),
            Whiff_Rate    = ("overall_whiff_rate", "mean"),
            Pitch_Types   = ("n_pitch_types", "mean"),
        )
        .round(1)
        .reset_index()
    )
    st.dataframe(cluster_agg, use_container_width=True, hide_index=True)

    # ── Filter by cluster ──
    st.markdown('<div class="section-header" style="margin-top:1.5rem">Explore a Cluster</div>', unsafe_allow_html=True)

    selected_cluster = st.selectbox(
        "Cluster",
        options=list(CLUSTER_LABELS.values()),
        label_visibility="collapsed",
    )

    cluster_id = {v: k for k, v in CLUSTER_LABELS.items()}[selected_cluster]
    cluster_pitchers = ps[ps["cluster"] == cluster_id].sort_values("overall_whiff_rate", ascending=False)

    st.markdown(f"**{len(cluster_pitchers)} pitchers** in *{selected_cluster}*")

    display_cols = {
        "Pitcher": "Pitcher",
        "avg_velo": "Velo",
        "avg_spin": "Spin",
        "avg_ivb": "IVB",
        "avg_hb": "HB",
        "n_pitch_types": "Pitch Types",
        "overall_whiff_rate": "Whiff%",
        "total_pitches": "Pitches",
    }
    cluster_display = cluster_pitchers[list(display_cols.keys())].rename(columns=display_cols)

    st.dataframe(
        cluster_display.reset_index(drop=True),
        use_container_width=True,
        hide_index=True,
    )

    # ── Radar chart comparing cluster averages ──
    st.markdown('<div class="section-header" style="margin-top:1.5rem">Cluster Radar</div>', unsafe_allow_html=True)

    radar_metrics = ["avg_velo","avg_spin","avg_ivb","overall_whiff_rate","n_pitch_types"]
    radar_labels  = ["Velocity","Spin","Vert Break","Whiff%","Pitch Types"]

    radar_data = ps.groupby("Cluster Name")[radar_metrics].mean()
    # Normalise 0-1 per metric
    radar_norm = (radar_data - radar_data.min()) / (radar_data.max() - radar_data.min() + 1e-9)

    fig_radar = go.Figure()
    colors      = ["#3b82f6", "#22c55e", "#f97316", "#a855f7"]
    fill_colors = [
        "rgba(59,130,246,0.15)",
        "rgba(34,197,94,0.15)",
        "rgba(249,115,22,0.15)",
        "rgba(168,85,247,0.15)",
    ]

    for i, (cname, row) in enumerate(radar_norm.iterrows()):
        vals = row[radar_metrics].tolist()
        vals += [vals[0]]  # close loop
        fig_radar.add_trace(go.Scatterpolar(
            r=vals,
            theta=radar_labels + [radar_labels[0]],
            fill="toself",
            name=cname,
            line_color=colors[i],
            fillcolor=fill_colors[i],
            opacity=0.85,
        ))

    fig_radar.update_layout(
        polar=dict(
            bgcolor="#0d1117",
            radialaxis=dict(visible=True, range=[0, 1], gridcolor="#21262d", tickfont=dict(color="#8b949e")),
            angularaxis=dict(gridcolor="#21262d", tickfont=dict(color="#c9d1d9")),
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        font_color="#8b949e",
        showlegend=True,
        legend=dict(font=dict(size=11)),
        height=420,
        margin=dict(t=20, b=20, l=40, r=40),
    )
    st.plotly_chart(fig_radar, use_container_width=True)
