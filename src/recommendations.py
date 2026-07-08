"""
Recommendation engine + OCEN/ULI-style interoperability payloads.

Two responsibilities:

1. ``recommend`` - turn the weakest pillars into concrete, MSME-friendly
   actions that would lift the Financial Health Score (the "how do I improve"
   answer a relationship manager or the MSME owner needs).

2. ``build_ocen_payload`` - emit the health card as a lender-agnostic JSON
   object shaped like an OCEN / ULI "loan-application enrichment" block, so the
   score can plug into Account Aggregator + ULI rails instead of living in a
   silo.
"""
from __future__ import annotations

from datetime import datetime, timezone

from .scoring import HealthResult

_PILLAR_ADVICE = {
    "Cashflow & Liquidity": [
        "Maintain a minimum current-account buffer of ~1 month of outflow to cut negative-balance days.",
        "Route more receivables through the primary account to strengthen inflow/outflow ratio.",
    ],
    "GST & Sales Health": [
        "File pending GSTR-1/3B returns on time - filing regularity is the single biggest sales-health lever.",
        "Consolidate sales onto GST invoices to make turnover growth verifiable.",
    ],
    "Repayment & Obligations": [
        "Reduce EMI-to-inflow (obligation ratio) below 40% before seeking fresh credit.",
        "Clear any cheque-bounce history and keep auto-debit accounts funded.",
    ],
    "Business Stability": [
        "Longer operating history improves this pillar automatically; smoothing seasonal dips helps sooner.",
        "Diversify revenue across months/customers to lower seasonality.",
    ],
    "Digital Footprint": [
        "Increase UPI acceptance at point-of-sale to raise transaction velocity.",
        "Broaden the customer base (more unique UPI counterparties) to signal demand diversity.",
    ],
    "Compliance & Statutory": [
        "Keep EPFO contributions regular for the registered workforce.",
        "Complete GST registration if turnover has crossed the threshold.",
    ],
}


def recommend(result: HealthResult, top_n: int = 3) -> list[dict]:
    """Return prioritised improvement actions for the weakest pillars."""
    ranked = sorted(result.pillars, key=lambda p: p.score)
    recs: list[dict] = []
    for pillar in ranked[:top_n]:
        advice = _PILLAR_ADVICE.get(pillar.name, [])
        # Estimated FHS uplift if this pillar reaches 75.
        headroom = max(0.0, 75 - pillar.score) * pillar.weight
        recs.append(
            {
                "pillar": pillar.name,
                "current_score": round(pillar.score, 1),
                "potential_fhs_uplift": round(headroom, 1),
                "actions": advice,
            }
        )
    return recs


def build_ocen_payload(msme: dict, result: HealthResult, ml_pd: float | None = None) -> dict:
    """Shape the health card as an OCEN/ULI-compatible enrichment block."""
    return {
        "schemaVersion": "1.0",
        "issuer": "IDBI-MSME-FinancialHealthCard",
        "generatedAt": datetime.now(timezone.utc).isoformat(),
        "borrower": {
            "entityId": msme.get("msme_id"),
            "name": msme.get("business_name"),
            "sector": msme.get("sector"),
            "scale": msme.get("scale"),
            "isNewToCredit": bool(msme.get("is_ntc", False)),
            "isNewToBank": bool(msme.get("is_ntb", False)),
        },
        "financialHealth": {
            "score": result.fhs,
            "band": result.band,
            "bandLabel": result.band_label,
            "pillars": result.pillar_dict(),
            "pdEstimateRuleBased": result.pd_estimate,
            "pdEstimateMl": round(ml_pd, 4) if ml_pd is not None else None,
        },
        "creditRecommendation": {
            "indicativeLimitInr": result.suggested_limit,
            "currency": "INR",
            "basis": "6x assessed free monthly cashflow, health- and PD-adjusted",
        },
        "dataSources": ["GST", "UPI", "AccountAggregator", "EPFO"],
        "consentArtefact": {
            "framework": "DEPA-AA",
            "status": "SIMULATED_FOR_HACKATHON",
        },
        "interoperability": {"uliReady": True, "ocenCompatible": True},
    }
