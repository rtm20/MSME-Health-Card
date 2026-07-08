"""
Synthetic MSME alternate-data generator for the IDBI Innovate 2026 -
MSME Financial Health Card.

Generates a realistic pool of Micro/Small/Medium enterprises with alternate
data signals that mirror India's digital-lending stack:

  * GST      - filing regularity, turnover, turnover growth
  * UPI      - transaction velocity, counterparty diversity, digital footprint
  * AA       - bank account inflow/outflow, balance stability, bounces, EMIs
  * EPFO     - employee base, statutory (PF) filing regularity

A latent "true risk" is used to synthesise a 12-month default label so that a
supervised model can be trained for demo purposes. No real customer data is
used - everything here is procedurally generated and reproducible via a seed.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

SECTORS = [
    "Retail Trade",
    "Manufacturing",
    "Textiles",
    "Food & Beverage",
    "IT Services",
    "Logistics",
    "Agri-Processing",
    "Healthcare Services",
    "Construction",
    "Hospitality",
]

CITIES = [
    "Mumbai", "Pune", "Ahmedabad", "Surat", "Indore", "Coimbatore",
    "Ludhiana", "Jaipur", "Hyderabad", "Kochi", "Lucknow", "Nagpur",
]

# Deterministic-ish name parts for readable demo profiles.
_NAME_PREFIX = [
    "Shakti", "Sunrise", "Kaveri", "Anand", "Prime", "Deccan", "Ganga",
    "Sri Balaji", "Krishna", "Metro", "Nova", "Everest", "Sagar", "Lotus",
]
_NAME_SUFFIX = [
    "Traders", "Enterprises", "Industries", "Textiles", "Foods", "Logistics",
    "Solutions", "Agro", "Fabricators", "Distributors", "Exports",
]


def _band_from_scale(scale: str) -> tuple[float, float]:
    """Annual turnover band (in INR) roughly aligned to MSME classification."""
    return {
        "Micro": (5e5, 2.5e7),      # up to ~2.5 Cr
        "Small": (2.5e7, 1e8),      # 2.5 - 10 Cr
        "Medium": (1e8, 2.5e8),     # 10 - 25 Cr
    }[scale]


def generate_msme_pool(n: int = 2000, seed: int = 42) -> pd.DataFrame:
    """Generate ``n`` synthetic MSMEs with alternate-data features + labels."""
    rng = np.random.default_rng(seed)

    scale = rng.choice(["Micro", "Small", "Medium"], size=n, p=[0.65, 0.28, 0.07])
    sector = rng.choice(SECTORS, size=n)
    city = rng.choice(CITIES, size=n)

    # Business vintage in months (credit-invisible NTC firms are younger).
    vintage_months = np.clip(rng.gamma(shape=2.2, scale=18, size=n), 2, 300).astype(int)

    # New-to-Credit (NTC) / New-to-Bank (NTB): younger + no bureau history.
    ntc = (vintage_months < 24) & (rng.random(n) < 0.7)
    ntb = ntc & (rng.random(n) < 0.5)

    # ---- Turnover (GST) ----
    turnover = np.empty(n)
    for s in ["Micro", "Small", "Medium"]:
        lo, hi = _band_from_scale(s)
        mask = scale == s
        turnover[mask] = rng.uniform(lo, hi, size=mask.sum())
    annual_turnover = np.round(turnover, -3)

    # GST filing regularity 0..1 (fraction of last 12 returns filed on time).
    gst_months_eligible = np.minimum(vintage_months, 12)
    gst_filing_regularity = np.clip(rng.beta(6, 2, size=n), 0, 1)
    # Some firms are not GST-registered yet (very micro / informal).
    gst_registered = rng.random(n) > (0.35 * (scale == "Micro"))
    gst_filing_regularity = np.where(gst_registered, gst_filing_regularity, 0.0)

    # Turnover growth YoY (-30% .. +60%).
    turnover_growth = np.clip(rng.normal(0.08, 0.22, size=n), -0.4, 0.9)

    # ---- AA / bank account signals ----
    avg_monthly_inflow = annual_turnover / 12 * rng.uniform(0.55, 0.95, size=n)
    inflow_outflow_ratio = np.clip(rng.normal(1.08, 0.18, size=n), 0.6, 1.8)
    avg_monthly_outflow = avg_monthly_inflow / inflow_outflow_ratio
    avg_balance = avg_monthly_inflow * rng.uniform(0.15, 0.9, size=n)
    balance_volatility = np.clip(rng.beta(2, 5, size=n), 0.02, 0.95)  # cv of balance
    negative_balance_days = rng.poisson(lam=np.clip(6 * balance_volatility, 0, 20))
    cheque_bounces = rng.poisson(lam=np.clip(1.5 * balance_volatility, 0, 8))

    # Existing obligations (EMI) as a share of monthly inflow.
    obligation_ratio = np.clip(rng.beta(2, 4, size=n), 0, 0.85)
    existing_emi = avg_monthly_inflow * obligation_ratio

    # ---- UPI / digital footprint ----
    upi_monthly_txns = np.clip(
        rng.normal(180, 90, size=n) * (1 + 0.4 * (scale != "Micro")), 5, 2000
    ).astype(int)
    upi_counterparties = np.clip(
        (upi_monthly_txns * rng.uniform(0.15, 0.5, size=n)), 3, 600
    ).astype(int)
    digital_adoption = np.clip(rng.beta(3, 2, size=n), 0.05, 1.0)

    # ---- EPFO ----
    employee_count = np.clip(
        (annual_turnover / 1.2e6) * rng.uniform(0.4, 1.6, size=n), 0, 250
    ).astype(int)
    epfo_registered = employee_count >= 10
    epfo_regularity = np.where(
        epfo_registered, np.clip(rng.beta(5, 2, size=n), 0, 1), 0.0
    )

    # Revenue seasonality (coefficient of variation of monthly revenue).
    seasonality = np.clip(rng.beta(2, 6, size=n), 0.03, 0.8)

    df = pd.DataFrame(
        {
            "msme_id": [f"MSME{100000 + i}" for i in range(n)],
            "business_name": [
                f"{rng.choice(_NAME_PREFIX)} {rng.choice(_NAME_SUFFIX)}"
                for _ in range(n)
            ],
            "sector": sector,
            "city": city,
            "scale": scale,
            "vintage_months": vintage_months,
            "is_ntc": ntc,
            "is_ntb": ntb,
            "gst_registered": gst_registered,
            "annual_turnover": annual_turnover,
            "turnover_growth": np.round(turnover_growth, 3),
            "gst_filing_regularity": np.round(gst_filing_regularity, 3),
            "avg_monthly_inflow": np.round(avg_monthly_inflow, 0),
            "avg_monthly_outflow": np.round(avg_monthly_outflow, 0),
            "inflow_outflow_ratio": np.round(inflow_outflow_ratio, 3),
            "avg_balance": np.round(avg_balance, 0),
            "balance_volatility": np.round(balance_volatility, 3),
            "negative_balance_days": negative_balance_days,
            "cheque_bounces": cheque_bounces,
            "existing_emi": np.round(existing_emi, 0),
            "obligation_ratio": np.round(obligation_ratio, 3),
            "upi_monthly_txns": upi_monthly_txns,
            "upi_counterparties": upi_counterparties,
            "digital_adoption": np.round(digital_adoption, 3),
            "employee_count": employee_count,
            "epfo_registered": epfo_registered,
            "epfo_regularity": np.round(epfo_regularity, 3),
            "seasonality": np.round(seasonality, 3),
        }
    )

    df["default_12m"] = _synthesise_default_label(df, rng)
    return df


def _synthesise_default_label(df: pd.DataFrame, rng: np.random.Generator) -> np.ndarray:
    """Create a plausible 12-month default flag from a latent risk score.

    The label is intentionally driven by the same economic signals a credit
    analyst would care about, plus noise, so a model can learn a realistic
    (but synthetic) relationship.
    """
    z = (
        -1.9
        + 3.4 * df["obligation_ratio"]
        + 2.6 * df["balance_volatility"]
        + 0.18 * df["cheque_bounces"]
        + 0.09 * df["negative_balance_days"]
        - 2.3 * df["gst_filing_regularity"]
        - 1.6 * df["turnover_growth"]
        - 1.2 * df["digital_adoption"]
        - 0.010 * np.sqrt(df["vintage_months"])
        + 1.1 * (df["inflow_outflow_ratio"] < 1.0).astype(float)
        + 0.6 * df["is_ntc"].astype(float)
        + rng.normal(0, 0.32, size=len(df))
    )
    prob = 1 / (1 + np.exp(-z))
    return (rng.random(len(df)) < prob).astype(int)


if __name__ == "__main__":  # quick smoke test
    pool = generate_msme_pool(1000)
    print(pool.head())
    print("Default rate:", pool["default_12m"].mean().round(3))
