"""
Multidimensional MSME Financial Health scoring engine.

This is a fully transparent (glass-box) model. Every pillar sub-score is a
deterministic function of alternate-data signals, which makes the resulting
Financial Health Score explainable and auditable - a hard requirement for
regulated lending decisions.

Composite Financial Health Score (FHS) = weighted sum of 6 pillars, each 0-100:

    1. Cashflow & Liquidity        25%
    2. GST & Sales Health          20%
    3. Repayment & Obligations     20%
    4. Business Stability          15%
    5. Digital Footprint           10%
    6. Compliance & Statutory      10%

FHS maps to a rating band and an indicative Probability of Default (PD), and
drives an indicative credit-limit recommendation.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

PILLAR_WEIGHTS = {
    "Cashflow & Liquidity": 0.25,
    "GST & Sales Health": 0.20,
    "Repayment & Obligations": 0.20,
    "Business Stability": 0.15,
    "Digital Footprint": 0.10,
    "Compliance & Statutory": 0.10,
}


def _clip100(x: float) -> float:
    return float(np.clip(x, 0, 100))


def _scale(value: float, lo: float, hi: float, invert: bool = False) -> float:
    """Linearly map ``value`` from [lo, hi] to [0, 100] (optionally inverted)."""
    if hi == lo:
        return 50.0
    frac = (value - lo) / (hi - lo)
    frac = float(np.clip(frac, 0, 1))
    return _clip100((1 - frac if invert else frac) * 100)


@dataclass
class PillarResult:
    name: str
    score: float
    weight: float
    drivers: dict[str, str] = field(default_factory=dict)

    @property
    def weighted(self) -> float:
        return self.score * self.weight


@dataclass
class HealthResult:
    fhs: float
    band: str
    band_label: str
    pd_estimate: float
    suggested_limit: float
    pillars: list[PillarResult]

    def pillar_dict(self) -> dict[str, float]:
        return {p.name: round(p.score, 1) for p in self.pillars}


# ---------------------------------------------------------------------------
# Pillar computations
# ---------------------------------------------------------------------------
def _cashflow_pillar(r: pd.Series) -> PillarResult:
    liquidity = _scale(r["inflow_outflow_ratio"], 0.85, 1.4)
    buffer = _scale(r["avg_balance"] / max(r["avg_monthly_outflow"], 1), 0.05, 0.8)
    stability = _scale(r["balance_volatility"], 0.05, 0.7, invert=True)
    overdraft = _scale(r["negative_balance_days"], 0, 15, invert=True)
    score = 0.35 * liquidity + 0.25 * buffer + 0.25 * stability + 0.15 * overdraft
    return PillarResult(
        "Cashflow & Liquidity",
        _clip100(score),
        PILLAR_WEIGHTS["Cashflow & Liquidity"],
        {
            "Inflow/Outflow ratio": f"{r['inflow_outflow_ratio']:.2f}",
            "Balance buffer (months)": f"{r['avg_balance'] / max(r['avg_monthly_outflow'], 1):.2f}",
            "Balance volatility": f"{r['balance_volatility']:.2f}",
            "Negative-balance days": f"{int(r['negative_balance_days'])}",
        },
    )


def _gst_pillar(r: pd.Series) -> PillarResult:
    if not r["gst_registered"]:
        return PillarResult(
            "GST & Sales Health",
            25.0,
            PILLAR_WEIGHTS["GST & Sales Health"],
            {"GST registered": "No - not yet registered"},
        )
    regularity = r["gst_filing_regularity"] * 100
    growth = _scale(r["turnover_growth"], -0.2, 0.4)
    scale_score = _scale(np.log10(max(r["annual_turnover"], 1)), 5.7, 8.3)
    score = 0.5 * regularity + 0.3 * growth + 0.2 * scale_score
    return PillarResult(
        "GST & Sales Health",
        _clip100(score),
        PILLAR_WEIGHTS["GST & Sales Health"],
        {
            "GST filing regularity": f"{r['gst_filing_regularity'] * 100:.0f}%",
            "Turnover growth (YoY)": f"{r['turnover_growth'] * 100:+.0f}%",
            "Annual turnover": f"Rs {r['annual_turnover'] / 1e5:.1f} L",
        },
    )


def _repayment_pillar(r: pd.Series) -> PillarResult:
    obligation = _scale(r["obligation_ratio"], 0.05, 0.7, invert=True)
    bounces = _scale(r["cheque_bounces"], 0, 6, invert=True)
    surplus = (r["avg_monthly_inflow"] - r["avg_monthly_outflow"] - r["existing_emi"])
    coverage = _scale(surplus / max(r["avg_monthly_inflow"], 1), -0.1, 0.35)
    score = 0.45 * obligation + 0.3 * coverage + 0.25 * bounces
    return PillarResult(
        "Repayment & Obligations",
        _clip100(score),
        PILLAR_WEIGHTS["Repayment & Obligations"],
        {
            "Obligation ratio (EMI/inflow)": f"{r['obligation_ratio'] * 100:.0f}%",
            "Free monthly surplus": f"Rs {surplus / 1e3:.0f} K",
            "Cheque bounces (12m)": f"{int(r['cheque_bounces'])}",
        },
    )


def _stability_pillar(r: pd.Series) -> PillarResult:
    vintage = _scale(r["vintage_months"], 6, 120)
    seasonality = _scale(r["seasonality"], 0.05, 0.6, invert=True)
    scale_bonus = {"Micro": 40, "Small": 70, "Medium": 90}[r["scale"]]
    score = 0.5 * vintage + 0.3 * seasonality + 0.2 * scale_bonus
    return PillarResult(
        "Business Stability",
        _clip100(score),
        PILLAR_WEIGHTS["Business Stability"],
        {
            "Business vintage": f"{int(r['vintage_months'])} months",
            "Revenue seasonality": f"{r['seasonality']:.2f} (lower is steadier)",
            "Enterprise scale": r["scale"],
        },
    )


def _digital_pillar(r: pd.Series) -> PillarResult:
    velocity = _scale(r["upi_monthly_txns"], 20, 500)
    diversity = _scale(r["upi_counterparties"], 5, 200)
    adoption = r["digital_adoption"] * 100
    score = 0.4 * velocity + 0.3 * diversity + 0.3 * adoption
    return PillarResult(
        "Digital Footprint",
        _clip100(score),
        PILLAR_WEIGHTS["Digital Footprint"],
        {
            "UPI txns / month": f"{int(r['upi_monthly_txns'])}",
            "Unique counterparties": f"{int(r['upi_counterparties'])}",
            "Digital adoption index": f"{r['digital_adoption']:.2f}",
        },
    )


def _compliance_pillar(r: pd.Series) -> PillarResult:
    if r["epfo_registered"]:
        epfo = r["epfo_regularity"] * 100
    else:
        # Small firms below the EPFO threshold are not penalised heavily.
        epfo = 60.0
    gst_flag = 70 if r["gst_registered"] else 30
    score = 0.6 * epfo + 0.4 * gst_flag
    return PillarResult(
        "Compliance & Statutory",
        _clip100(score),
        PILLAR_WEIGHTS["Compliance & Statutory"],
        {
            "EPFO registered": "Yes" if r["epfo_registered"] else "No / below threshold",
            "EPFO filing regularity": f"{r['epfo_regularity'] * 100:.0f}%"
            if r["epfo_registered"] else "N/A",
            "Employees": f"{int(r['employee_count'])}",
        },
    )


_PILLAR_FUNCS = [
    _cashflow_pillar,
    _gst_pillar,
    _repayment_pillar,
    _stability_pillar,
    _digital_pillar,
    _compliance_pillar,
]


def _band_for(fhs: float) -> tuple[str, str]:
    if fhs >= 80:
        return "AAA", "Excellent - Low Risk"
    if fhs >= 65:
        return "AA", "Good - Prime Lending"
    if fhs >= 50:
        return "A", "Moderate - Monitor"
    if fhs >= 35:
        return "B", "Elevated Risk"
    return "C", "High Risk"


def _pd_from_fhs(fhs: float) -> float:
    """Map health score to an indicative 12-month PD (monotonic, 1%..45%)."""
    z = (fhs - 55) / 12.0
    pd_val = 0.45 / (1 + np.exp(z))
    return float(np.clip(pd_val, 0.005, 0.5))


def _suggested_limit(r: pd.Series, fhs: float, pd_val: float) -> float:
    """Indicative unsecured working-capital limit.

    Anchored on assessed monthly cashflow surplus, scaled by health and
    discounted by PD. Deliberately conservative for a bank context.
    """
    surplus = max(r["avg_monthly_inflow"] - r["avg_monthly_outflow"] - r["existing_emi"], 0)
    base = surplus * 6  # ~6 months of free cashflow
    health_mult = 0.4 + 1.1 * (fhs / 100)
    risk_disc = 1 - pd_val
    limit = base * health_mult * risk_disc
    # Round to nearest 10k, cap for micro prudence.
    return float(np.round(min(limit, 5e7) / 1e4) * 1e4)


def score_msme(row: pd.Series | dict) -> HealthResult:
    """Compute the full Financial Health Card for a single MSME record."""
    r = pd.Series(row) if isinstance(row, dict) else row
    pillars = [f(r) for f in _PILLAR_FUNCS]
    fhs = _clip100(sum(p.weighted for p in pillars))
    band, band_label = _band_for(fhs)
    pd_val = _pd_from_fhs(fhs)
    limit = _suggested_limit(r, fhs, pd_val)
    return HealthResult(
        fhs=round(fhs, 1),
        band=band,
        band_label=band_label,
        pd_estimate=round(pd_val, 4),
        suggested_limit=limit,
        pillars=pillars,
    )


def score_pool(df: pd.DataFrame) -> pd.DataFrame:
    """Vectorised-ish scoring for a whole pool (adds FHS + band + PD columns)."""
    results = df.apply(lambda r: score_msme(r), axis=1)
    out = df.copy()
    out["fhs"] = [res.fhs for res in results]
    out["band"] = [res.band for res in results]
    out["pd_estimate"] = [res.pd_estimate for res in results]
    out["suggested_limit"] = [res.suggested_limit for res in results]
    for pname in PILLAR_WEIGHTS:
        out[pname] = [res.pillar_dict()[pname] for res in results]
    return out
