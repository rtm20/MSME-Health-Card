# MSME Financial Health Card 🏦

**IDBI Innovate 2026 — Problem Statement 3: Financial Health Score**

An AI/ML-driven **MSME Financial Health Card** that aggregates alternate data
(**GST · UPI · Account Aggregator · EPFO**), computes a transparent
**multidimensional financial health score**, estimates 12-month default risk
with an ML challenger model, visualises strengths & risks, recommends
improvements, and emits an **OCEN/ULI-ready** payload — so banks can safely
onboard **credit-invisible (New-to-Credit / New-to-Bank) MSMEs** and improve
portfolio quality.

> Live demo: _add your Streamlit Cloud URL here after deploying_
> Repo: _add your GitHub URL here_

---

## Why this wins the brief

| Expected outcome (PS3) | How this solution delivers |
|---|---|
| Aggregate alternate data (GST, UPI, AA, EPFO) | 20+ engineered signals across all four sources |
| Multidimensional financial health score | 6-pillar glass-box score (0–100) → rating band |
| Visualise strengths & risks | Gauge, radar, pillar breakdown, portfolio charts |
| Integrate with ULI/OCEN/AA | Lender-agnostic OCEN/ULI JSON + DEPA-AA consent framing |
| Near real-time credit assessment | Instant scoring + live what-if re-scoring |
| Expand onboarding of credit-invisible MSMEs | Explicit NTC/NTB inclusion view + unlocked-borrower metric |
| Improve portfolio quality | ML default model (AUC ≈ 0.78, KS ≈ 0.49) with explainability |

**Differentiators:** dual-engine risk (transparent score **+** ML challenger),
per-applicant SHAP explainability (regulator-friendly), and true
interoperability (OCEN/ULI payload) instead of a siloed score.

---

## Features

- **Financial Health Card** — per-MSME gauge, band, PD, indicative credit
  limit, pillar radar, and driver-level "why this score".
- **ML explainability** — XGBoost with native SHAP contributions showing what
  raises/reduces each borrower's default risk.
- **What-if simulator** — move alternate-data sliders and watch the score
  recompute live (great for RM coaching / "how do I qualify").
- **Portfolio & inclusion view** — risk-band mix, NTC-vs-existing
  underwritability, health-vs-PD scatter, downloadable scored ledger.
- **OCEN / ULI API** — copy/download the health card as a lender-agnostic JSON
  enrichment block.
- **Methodology page** — full transparency of pillar weights and score→action
  mapping.

---

## Architecture

```
Alternate data (synthetic)        Scoring & risk                 Delivery
------------------------          ----------------------         --------------------
GST  · UPI · AA · EPFO   ─┐       6-pillar glass-box score ─┐    Streamlit UI (cards,
                          ├──►     → FHS (0-100) → band     ├──► radar, portfolio)
data_generator.py         │       scoring.py                │
                          │                                 │    OCEN/ULI JSON payload
                          └──►     XGBoost PD + SHAP  ───────┘    recommendations.py
                                  model.py
```

- `src/data_generator.py` — reproducible synthetic MSME pool with alt-data + labels
- `src/scoring.py` — transparent 6-pillar Financial Health Score engine
- `src/model.py` — XGBoost default-risk model + per-instance SHAP explainability
- `src/recommendations.py` — improvement actions + OCEN/ULI payload builder
- `app.py` — Streamlit application (6 views)

---

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

Open http://localhost:8501

---

## Deploy for free (Streamlit Community Cloud)

1. Push this folder to a **public GitHub repo**.
2. Go to **https://share.streamlit.io** → sign in with GitHub → **Create app**.
3. Pick your repo, branch `main`, main file `app.py`.
4. Click **Deploy**. You get a public URL like
   `https://<your-app>.streamlit.app` in ~2 minutes.
5. Paste that URL as your **Project Deployment Link** in the submission.

_`runtime.txt` pins Python 3.11 and `requirements.txt` pins Streamlit 1.39 so
the build is deterministic._

**Backup host (Hugging Face Spaces):** create a Space → SDK **Streamlit** →
push the same files → auto-deploys.

---

## Data & compliance note

All data is **synthetically generated** for the hackathon — **no real customer
information is used**. PD estimates and credit limits are **indicative** and are
not a lending decision. In production the same pipeline consumes real signals
via **Account Aggregator consent (DEPA-AA)** and plugs into **ULI/OCEN** rails.

## Tech stack

Python · Streamlit · Plotly · scikit-learn · XGBoost · pandas · NumPy
