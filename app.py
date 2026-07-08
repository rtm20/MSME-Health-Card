"""
MSME Financial Health Card - IDBI Innovate 2026 (Problem Statement 3)

An AI/ML-driven MSME Financial Health Card that aggregates alternate data
(GST, UPI, Account Aggregator, EPFO), computes a transparent multidimensional
health score, estimates default risk with an ML challenger model, visualises
strengths & risks, recommends improvements, and emits an OCEN/ULI-ready payload
to onboard credit-invisible (NTC/NTB) MSMEs.

Run locally:   streamlit run app.py
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.data_generator import generate_msme_pool
from src.model import explain, predict_pd, train_model
from src.recommendations import build_ocen_payload, recommend
from src.scoring import PILLAR_WEIGHTS, score_msme, score_pool

# ---------------------------------------------------------------------------
# Page config & light styling
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="MSME Financial Health Card | IDBI Innovate 2026",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

GREEN = "#00A651"
BAND_COLORS = {
    "AAA": "#00A651",
    "AA": "#4CAF50",
    "A": "#FBC02D",
    "B": "#FB8C00",
    "C": "#E53935",
}

st.markdown(
    """
    <style>
      .block-container {padding-top: 1.6rem;}
      div[data-testid="stMetricValue"] {font-size: 1.6rem;}
      .hcard {border-radius: 16px; padding: 18px 22px; background: #161B22;
              border: 1px solid #23303d;}
      .pill {display:inline-block; padding:3px 12px; border-radius: 999px;
             font-weight:600; font-size:0.8rem;}
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Cached data + model
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_scored_pool(n: int = 1500, seed: int = 42) -> pd.DataFrame:
    pool = generate_msme_pool(n=n, seed=seed)
    return score_pool(pool)


@st.cache_resource(show_spinner=False)
def load_model(n: int = 1500, seed: int = 42):
    pool = generate_msme_pool(n=n, seed=seed)
    return train_model(pool, seed=seed)


with st.spinner("Aggregating alternate data & training risk engine..."):
    scored = load_scored_pool()
    bundle = load_model()


# ---------------------------------------------------------------------------
# Reusable plotting helpers
# ---------------------------------------------------------------------------
def gauge(fhs: float, band: str) -> go.Figure:
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=fhs,
            number={"suffix": " / 100", "font": {"size": 30}},
            gauge={
                "axis": {"range": [0, 100], "tickwidth": 1},
                "bar": {"color": BAND_COLORS.get(band, GREEN), "thickness": 0.3},
                "steps": [
                    {"range": [0, 35], "color": "#3a1d1d"},
                    {"range": [35, 50], "color": "#3a2f1d"},
                    {"range": [50, 65], "color": "#33361d"},
                    {"range": [65, 80], "color": "#1d3327"},
                    {"range": [80, 100], "color": "#123a26"},
                ],
                "threshold": {
                    "line": {"color": "white", "width": 3},
                    "thickness": 0.8,
                    "value": fhs,
                },
            },
        )
    )
    fig.update_layout(height=260, margin=dict(l=20, r=20, t=20, b=10))
    return fig


def radar(pillars) -> go.Figure:
    names = [p.name for p in pillars]
    vals = [p.score for p in pillars]
    fig = go.Figure(
        go.Scatterpolar(
            r=vals + [vals[0]],
            theta=names + [names[0]],
            fill="toself",
            line=dict(color=GREEN),
            fillcolor="rgba(0,166,81,0.25)",
        )
    )
    fig.update_layout(
        polar=dict(radialaxis=dict(range=[0, 100], showticklabels=True)),
        height=340,
        margin=dict(l=40, r=40, t=30, b=30),
        showlegend=False,
    )
    return fig


def contrib_bar(exp_df: pd.DataFrame, top: int = 8) -> go.Figure:
    d = exp_df.head(top).iloc[::-1]
    colors = ["#E53935" if c >= 0 else GREEN for c in d["contribution"]]
    fig = go.Figure(
        go.Bar(
            x=d["contribution"],
            y=d["label"],
            orientation="h",
            marker_color=colors,
        )
    )
    fig.update_layout(
        height=320,
        margin=dict(l=10, r=10, t=10, b=10),
        xaxis_title="Impact on default risk  (red = increases, green = reduces)",
    )
    return fig


def band_pill(band: str, label: str) -> str:
    c = BAND_COLORS.get(band, GREEN)
    return (
        f"<span class='pill' style='background:{c}22;color:{c};border:1px solid {c}'>"
        f"Band {band} &middot; {label}</span>"
    )


def inr(x: float) -> str:
    if x >= 1e7:
        return f"Rs {x / 1e7:.2f} Cr"
    if x >= 1e5:
        return f"Rs {x / 1e5:.2f} L"
    return f"Rs {x / 1e3:.0f} K"


# ---------------------------------------------------------------------------
# Sidebar navigation
# ---------------------------------------------------------------------------
st.sidebar.title("🏦 MSME Health Card")
st.sidebar.caption("IDBI Innovate 2026 · Problem Statement 3")
page = st.sidebar.radio(
    "Navigate",
    [
        "🏠 Overview",
        "💳 Financial Health Card",
        "🧪 What-if Simulator",
        "📊 Portfolio & Inclusion",
        "🔌 OCEN / ULI API",
        "ℹ️ Methodology",
    ],
)
st.sidebar.divider()
st.sidebar.metric("Model AUC", bundle.auc)
st.sidebar.metric("Model KS", bundle.ks)
st.sidebar.caption(f"Engine: {bundle.backend} · portfolio: {len(scored):,} MSMEs")
st.sidebar.caption("Synthetic data only — no real customer information.")


# ===========================================================================
# PAGE: Overview
# ===========================================================================
if page.startswith("🏠"):
    st.title("AI-Driven MSME Financial Health Card")
    st.markdown(
        "A unified assessment framework that turns **alternate data** "
        "(GST · UPI · Account Aggregator · EPFO) into a transparent, "
        "multidimensional **Financial Health Score** — so banks can safely "
        "underwrite **New-to-Credit / New-to-Bank** MSMEs that lack formal "
        "financial documents."
    )

    ntc = int(scored["is_ntc"].sum())
    approvable = int((scored["fhs"] >= 50).sum())
    credit_invisible_approvable = int(((scored["is_ntc"]) & (scored["fhs"] >= 50)).sum())

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("MSMEs assessed", f"{len(scored):,}")
    c2.metric("New-to-Credit in pool", f"{ntc:,}", f"{ntc / len(scored) * 100:.0f}%")
    c3.metric("Underwritable (FHS ≥ 50)", f"{approvable:,}",
              f"{approvable / len(scored) * 100:.0f}%")
    c4.metric("Credit-invisible unlocked", f"{credit_invisible_approvable:,}",
              "onboarded via alt-data")

    st.divider()
    left, right = st.columns([1.15, 1])
    with left:
        st.subheader("Financial Health Score distribution")
        fig = go.Figure()
        for band, col in BAND_COLORS.items():
            sub = scored[scored["band"] == band]
            fig.add_trace(
                go.Histogram(x=sub["fhs"], name=f"Band {band}", marker_color=col,
                             xbins=dict(size=5))
            )
        fig.update_layout(barmode="stack", height=340, bargap=0.05,
                          margin=dict(l=10, r=10, t=10, b=10),
                          xaxis_title="Financial Health Score", yaxis_title="MSMEs")
        st.plotly_chart(fig, use_container_width=True)
    with right:
        st.subheader("The problem we solve")
        st.markdown(
            "- Traditional MSME underwriting needs **audited financials** many "
            "NTC/NTB firms simply don't have.\n"
            "- Rich alternate data (**GST, UPI, AA, EPFO**) already exists but is "
            "**fragmented** and unused.\n"
            "- Result: **high rejection rates**, missed viable borrowers, slow "
            "financial inclusion.\n\n"
            "**Our answer:** a single, explainable Health Card that fuses these "
            "signals, is **ULI/OCEN-ready**, and supports **near real-time** "
            "re-scoring on fresh consented data."
        )
        st.info("Pick a sample MSME on the **Financial Health Card** page to see "
                "an end-to-end assessment.")


# ===========================================================================
# PAGE: Financial Health Card
# ===========================================================================
elif page.startswith("💳"):
    st.title("💳 Financial Health Card")

    fcol1, fcol2, fcol3 = st.columns([1.4, 1, 1])
    only_ntc = fcol1.toggle("Show only New-to-Credit MSMEs", value=False)
    view = scored[scored["is_ntc"]] if only_ntc else scored
    sector_filter = fcol2.selectbox("Sector", ["All"] + sorted(view["sector"].unique()))
    if sector_filter != "All":
        view = view[view["sector"] == sector_filter]

    label_map = {
        f"{r.business_name} · {r.msme_id} · {r.sector} · FHS {r.fhs}": r.msme_id
        for r in view.itertuples()
    }
    if not label_map:
        st.warning("No MSMEs match the current filter.")
        st.stop()
    picked_label = fcol3.selectbox("Select MSME", list(label_map.keys()))
    msme_id = label_map[picked_label]

    row = scored[scored["msme_id"] == msme_id].iloc[0]
    result = score_msme(row)
    ml_pd = predict_pd(bundle, row)
    exp_df = explain(bundle, row)

    # Header card
    st.markdown(
        f"### {row['business_name']} &nbsp; {band_pill(result.band, result.band_label)}",
        unsafe_allow_html=True,
    )
    st.caption(
        f"{row['msme_id']} · {row['sector']} · {row['city']} · {row['scale']} · "
        f"Vintage {int(row['vintage_months'])} mo · "
        f"{'🆕 New-to-Credit' if row['is_ntc'] else 'Has credit history'}"
    )

    g1, g2, g3 = st.columns([1, 1, 1.1])
    with g1:
        st.plotly_chart(gauge(result.fhs, result.band), use_container_width=True)
    with g2:
        st.metric("Rule-based PD (12m)", f"{result.pd_estimate * 100:.1f}%")
        st.metric("ML challenger PD", f"{ml_pd * 100:.1f}%")
        st.metric("Indicative credit limit", inr(result.suggested_limit))
    with g3:
        st.plotly_chart(radar(result.pillars), use_container_width=True)

    st.divider()
    st.subheader("Pillar breakdown")
    pcols = st.columns(3)
    for i, p in enumerate(result.pillars):
        with pcols[i % 3]:
            st.markdown(f"**{p.name}**  ·  weight {int(p.weight * 100)}%")
            st.progress(min(int(p.score), 100), text=f"{p.score:.0f}/100")
            with st.expander("Why this score"):
                for k, v in p.drivers.items():
                    st.markdown(f"- **{k}:** {v}")

    st.divider()
    ecol, rcol = st.columns([1.1, 1])
    with ecol:
        st.subheader("🔍 ML explainability — what drives this risk")
        st.plotly_chart(contrib_bar(exp_df), use_container_width=True)
    with rcol:
        st.subheader("📈 How to improve the score")
        for rec in recommend(result):
            st.markdown(
                f"**{rec['pillar']}** — now {rec['current_score']:.0f}/100 "
                f"(potential +{rec['potential_fhs_uplift']:.1f} FHS)"
            )
            for a in rec["actions"]:
                st.markdown(f"- {a}")


# ===========================================================================
# PAGE: What-if Simulator
# ===========================================================================
elif page.startswith("🧪"):
    st.title("🧪 What-if Simulator")
    st.caption(
        "Adjust a credit-invisible MSME's alternate-data signals and watch the "
        "Financial Health Score recompute live — great for RM coaching and "
        "'how do I qualify' conversations."
    )

    base = dict(scored.iloc[0])
    c1, c2, c3 = st.columns(3)
    base["vintage_months"] = c1.slider("Business vintage (months)", 3, 240, 18)
    base["annual_turnover"] = c1.slider("Annual turnover (Rs L)", 5, 2500, 80) * 1e5
    base["turnover_growth"] = c1.slider("Turnover growth YoY", -0.4, 0.9, 0.1, 0.01)
    base["gst_registered"] = c1.toggle("GST registered", value=True)
    base["gst_filing_regularity"] = c2.slider("GST filing regularity", 0.0, 1.0, 0.7, 0.05)
    base["inflow_outflow_ratio"] = c2.slider("Inflow / outflow ratio", 0.6, 1.8, 1.05, 0.01)
    base["balance_volatility"] = c2.slider("Balance volatility", 0.02, 0.95, 0.3, 0.01)
    base["obligation_ratio"] = c2.slider("Obligation ratio (EMI/inflow)", 0.0, 0.85, 0.3, 0.01)
    base["upi_monthly_txns"] = c3.slider("UPI txns / month", 5, 1500, 180)
    base["upi_counterparties"] = c3.slider("Unique UPI counterparties", 3, 500, 60)
    base["digital_adoption"] = c3.slider("Digital adoption index", 0.05, 1.0, 0.6, 0.05)
    base["cheque_bounces"] = c3.slider("Cheque bounces (12m)", 0, 8, 1)

    # Derive dependent fields consistently.
    base["avg_monthly_inflow"] = base["annual_turnover"] / 12 * 0.8
    base["avg_monthly_outflow"] = base["avg_monthly_inflow"] / base["inflow_outflow_ratio"]
    base["avg_balance"] = base["avg_monthly_inflow"] * 0.4
    base["existing_emi"] = base["avg_monthly_inflow"] * base["obligation_ratio"]
    base["negative_balance_days"] = int(base["balance_volatility"] * 12)
    base["seasonality"] = 0.2
    base["scale"] = "Micro" if base["annual_turnover"] < 2.5e7 else "Small"
    base["epfo_registered"] = base["annual_turnover"] > 5e7
    base["epfo_regularity"] = 0.8 if base["epfo_registered"] else 0.0
    base["employee_count"] = int(base["annual_turnover"] / 1.2e6)

    result = score_msme(base)
    ml_pd = predict_pd(bundle, base)

    st.divider()
    s1, s2, s3 = st.columns([1, 1, 1.1])
    with s1:
        st.plotly_chart(gauge(result.fhs, result.band), use_container_width=True)
        st.markdown(band_pill(result.band, result.band_label), unsafe_allow_html=True)
    with s2:
        st.metric("Financial Health Score", f"{result.fhs:.1f}")
        st.metric("ML PD (12m)", f"{ml_pd * 100:.1f}%")
        st.metric("Indicative limit", inr(result.suggested_limit))
    with s3:
        st.plotly_chart(radar(result.pillars), use_container_width=True)


# ===========================================================================
# PAGE: Portfolio & Inclusion
# ===========================================================================
elif page.startswith("📊"):
    st.title("📊 Portfolio & Financial Inclusion")

    c1, c2, c3 = st.columns(3)
    with c1:
        st.subheader("Risk band mix")
        band_counts = scored["band"].value_counts().reindex(
            ["AAA", "AA", "A", "B", "C"]).fillna(0)
        fig = go.Figure(
            go.Bar(x=band_counts.index, y=band_counts.values,
                   marker_color=[BAND_COLORS[b] for b in band_counts.index])
        )
        fig.update_layout(height=300, margin=dict(l=10, r=10, t=10, b=10),
                          yaxis_title="MSMEs")
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        st.subheader("Inclusion: NTC vs rest")
        ntc_appr = scored[scored["is_ntc"]]["fhs"].ge(50).mean() * 100
        rest_appr = scored[~scored["is_ntc"]]["fhs"].ge(50).mean() * 100
        fig = go.Figure(
            go.Bar(x=["New-to-Credit", "Existing"], y=[ntc_appr, rest_appr],
                   marker_color=[GREEN, "#4C8BF5"])
        )
        fig.update_layout(height=300, margin=dict(l=10, r=10, t=10, b=10),
                          yaxis_title="% underwritable (FHS ≥ 50)")
        st.plotly_chart(fig, use_container_width=True)
    with c3:
        st.subheader("Health vs default risk")
        fig = go.Figure(
            go.Scatter(x=scored["fhs"], y=scored["pd_estimate"] * 100,
                       mode="markers",
                       marker=dict(color=scored["fhs"], colorscale="RdYlGn",
                                   size=6, opacity=0.6))
        )
        fig.update_layout(height=300, margin=dict(l=10, r=10, t=10, b=10),
                          xaxis_title="Financial Health Score",
                          yaxis_title="PD (12m) %")
        st.plotly_chart(fig, use_container_width=True)

    st.divider()
    st.subheader("Portfolio ledger")
    show_cols = [
        "msme_id", "business_name", "sector", "scale", "is_ntc",
        "annual_turnover", "fhs", "band", "pd_estimate", "suggested_limit",
    ]
    st.dataframe(
        scored[show_cols].sort_values("fhs", ascending=False),
        use_container_width=True,
        height=380,
        column_config={
            "annual_turnover": st.column_config.NumberColumn("Turnover", format="₹%d"),
            "suggested_limit": st.column_config.NumberColumn("Limit", format="₹%d"),
            "pd_estimate": st.column_config.NumberColumn("PD", format="%.3f"),
            "fhs": st.column_config.ProgressColumn("FHS", min_value=0, max_value=100),
        },
    )
    st.download_button(
        "⬇️ Download scored portfolio (CSV)",
        scored[show_cols].to_csv(index=False).encode(),
        "msme_health_portfolio.csv",
        "text/csv",
    )


# ===========================================================================
# PAGE: OCEN / ULI API
# ===========================================================================
elif page.startswith("🔌"):
    st.title("🔌 OCEN / ULI-ready Health Card API")
    st.markdown(
        "The Health Card is emitted as a **lender-agnostic JSON enrichment "
        "block** so it can plug into **Account Aggregator + ULI/OCEN** rails "
        "instead of living in a silo. This is exactly the payload a Loan "
        "Service Provider would attach to a credit application."
    )
    pick = st.selectbox(
        "MSME",
        scored["msme_id"] + " · " + scored["business_name"],
    )
    msme_id = pick.split(" · ")[0]
    row = scored[scored["msme_id"] == msme_id].iloc[0]
    result = score_msme(row)
    ml_pd = predict_pd(bundle, row)
    payload = build_ocen_payload(dict(row), result, ml_pd)
    st.json(payload)
    st.download_button(
        "⬇️ Download OCEN payload (JSON)",
        json.dumps(payload, indent=2),
        f"{msme_id}_health_card.json",
        "application/json",
    )


# ===========================================================================
# PAGE: Methodology
# ===========================================================================
else:
    st.title("ℹ️ Methodology & Scoring Framework")
    st.markdown(
        "The **Financial Health Score (FHS)** is a transparent weighted sum of "
        "six pillars, each scored 0–100 from alternate data. Transparency is "
        "deliberate — lending decisions must be explainable and auditable."
    )
    wdf = pd.DataFrame(
        {"Pillar": list(PILLAR_WEIGHTS.keys()),
         "Weight": [f"{int(w * 100)}%" for w in PILLAR_WEIGHTS.values()]}
    )
    c1, c2 = st.columns([1, 1.3])
    with c1:
        st.table(wdf)
    with c2:
        st.markdown(
            "**Data sources fused**\n"
            "- **GST** — filing regularity, turnover, growth\n"
            "- **UPI** — transaction velocity, counterparty diversity\n"
            "- **Account Aggregator** — inflow/outflow, balance stability, bounces, EMIs\n"
            "- **EPFO** — workforce & statutory (PF) regularity\n\n"
            "**Dual-engine risk**\n"
            "- *Glass-box*: transparent pillar score → FHS → band → PD\n"
            f"- *ML challenger*: gradient-boosted model (AUC **{bundle.auc}**, "
            f"KS **{bundle.ks}**) with per-applicant SHAP explainability\n\n"
            "**Interoperability**: OCEN/ULI-ready JSON, DEPA-AA consent framing, "
            "near real-time re-scoring on fresh consented data."
        )
    st.divider()
    st.subheader("Score → Band → Action")
    st.table(pd.DataFrame({
        "Band": ["AAA", "AA", "A", "B", "C"],
        "FHS range": ["80–100", "65–79", "50–64", "35–49", "0–34"],
        "Interpretation": ["Excellent - low risk", "Good - prime lending",
                            "Moderate - monitor", "Elevated risk", "High risk"],
        "Suggested action": ["Straight-through approve", "Approve",
                             "Approve with conditions", "Manual review / secured",
                             "Decline / nurture"],
    }))
    st.caption(
        "Disclaimer: all data is synthetically generated for the IDBI Innovate "
        "2026 hackathon. No real customer data is used; PD and limits are "
        "indicative and not a lending decision."
    )
