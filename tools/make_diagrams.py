"""Generate architecture + process-flow diagrams for the IDBI submission deck."""
from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

OUT = os.path.join(os.path.dirname(__file__), "..", "deck-assets")
os.makedirs(OUT, exist_ok=True)

GREEN = "#00A651"
DARK = "#16232E"
INK = "#1A2A38"
GREY = "#EAF0F2"
BLUE = "#2F6FB0"
AMBER = "#E8A100"

plt.rcParams["font.family"] = "DejaVu Sans"


def _box(ax, x, y, w, h, text, fc=GREY, ec=DARK, tc=INK, fs=10, bold=False, lw=1.4):
    ax.add_patch(
        FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0.02,rounding_size=0.06",
            linewidth=lw, edgecolor=ec, facecolor=fc, zorder=2,
        )
    )
    ax.text(
        x + w / 2, y + h / 2, text, ha="center", va="center",
        fontsize=fs, color=tc, fontweight="bold" if bold else "normal",
        zorder=3, wrap=True,
    )


def _arrow(ax, x1, y1, x2, y2, color=GREEN, lw=2.2):
    ax.add_patch(
        FancyArrowPatch(
            (x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=16,
            linewidth=lw, color=color, zorder=1,
        )
    )


# ---------------------------------------------------------------------------
# Architecture diagram
# ---------------------------------------------------------------------------
def architecture():
    fig, ax = plt.subplots(figsize=(11.5, 5.4), dpi=200)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 55)
    ax.axis("off")

    # Column headers
    for cx, label in [(13, "1 · ALTERNATE DATA"), (46, "2 · SCORING & RISK ENGINE"),
                      (81, "3 · DELIVERY & INTEGRATION")]:
        ax.text(cx, 52, label, ha="center", fontsize=11, color=GREEN, fontweight="bold")

    # Data sources (left)
    sources = ["GST returns", "UPI transactions", "Account Aggregator", "EPFO"]
    for i, s in enumerate(sources):
        _box(ax, 2, 36 - i * 8.5, 22, 6.4, s, fc="#FFFFFF", ec=BLUE, tc=INK, fs=10)
    _box(ax, 2, 2.5, 22, 5, "Consent via DEPA-AA", fc="#Eef6ff", ec=BLUE, tc=BLUE,
         fs=9.5, bold=True)

    # Engine (middle)
    _box(ax, 33, 30, 28, 12, "6-Pillar Glass-Box Score\nFHS (0-100) -> Rating Band",
         fc="#E9F8EF", ec=GREEN, tc=DARK, fs=10.5, bold=True)
    _box(ax, 33, 13, 28, 11, "XGBoost Default-Risk Model\n+ SHAP explainability",
         fc="#FFF7E6", ec=AMBER, tc=DARK, fs=10.5, bold=True)
    _box(ax, 33, 2.5, 28, 6, "Feature engineering (20+ signals)",
         fc=GREY, ec=DARK, tc=INK, fs=9.5)

    # Delivery (right)
    outs = [
        ("Streamlit Health\nCard dashboard", "#E9F8EF", GREEN),
        ("OCEN / ULI-ready\nJSON API", "#Eef6ff", BLUE),
        ("Recommendations\n& what-if engine", "#FFF7E6", AMBER),
    ]
    for i, (t, fc, ec) in enumerate(outs):
        _box(ax, 70, 33 - i * 12, 27, 9, t, fc=fc, ec=ec, tc=DARK, fs=10, bold=True)

    # Arrows: sources -> engine
    _arrow(ax, 24, 22, 33, 30)
    _arrow(ax, 24, 22, 33, 18)
    # engine internal
    _arrow(ax, 47, 13, 47, 8.5, color=DARK, lw=1.6)
    # engine -> delivery
    _arrow(ax, 61, 36, 70, 37)
    _arrow(ax, 61, 30, 70, 25)
    _arrow(ax, 61, 18, 70, 15)

    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "architecture.png"), bbox_inches="tight",
                facecolor="white")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Process-flow diagram
# ---------------------------------------------------------------------------
def process_flow():
    fig, ax = plt.subplots(figsize=(11.5, 5.2), dpi=200)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 50)
    ax.axis("off")

    steps_top = [
        ("MSME loan\napplication", BLUE),
        ("AA consent\n(DEPA-AA)", BLUE),
        ("Aggregate GST /\nUPI / AA / EPFO", GREEN),
        ("Feature\nengineering", GREEN),
    ]
    steps_bot = [
        ("6-pillar score +\nML PD", GREEN),
        ("FHS + Band +\nPD + Limit", GREEN),
        ("Decision: Approve /\nReview / Decline", AMBER),
        ("OCEN/ULI payload\nto Bank / LSP", BLUE),
    ]

    y_top, y_bot = 33, 8
    xs = [3, 27, 51, 75]
    w, h = 20, 10

    for (t, ec), x in zip(steps_top, xs):
        _box(ax, x, y_top, w, h, t, fc="#FFFFFF", ec=ec, tc=INK, fs=10, bold=True)
    for (t, ec), x in zip(steps_bot, xs):
        _box(ax, x, y_bot, w, h, t, fc="#FFFFFF", ec=ec, tc=INK, fs=10, bold=True)

    for i in range(3):
        _arrow(ax, xs[i] + w, y_top + h / 2, xs[i + 1], y_top + h / 2)
        _arrow(ax, xs[i] + w, y_bot + h / 2, xs[i + 1], y_bot + h / 2)
    # wrap from top-right down to bottom-left
    _arrow(ax, xs[3] + w / 2, y_top, xs[0] + w / 2, y_bot + h, color=DARK, lw=1.8)

    ax.text(50, 47, "Near real-time · explainable · re-scored on fresh consented data",
            ha="center", fontsize=10.5, color=GREEN, fontweight="bold")

    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "process_flow.png"), bbox_inches="tight",
                facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    architecture()
    process_flow()
    print("Diagrams written to", os.path.abspath(OUT))
