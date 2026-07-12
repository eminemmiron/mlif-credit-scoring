"""Единый стиль графиков (matplotlib, Agg - без дисплея, сохранение в PNG)."""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PALETTE = {
    "neg": "#4C78A8",   # flag=0 (не дефолт)
    "pos": "#E4572E",   # flag=1 (дефолт)
    "accent": "#2E8B8B",
    "grid": "#D9D9D9",
}

plt.rcParams.update({
    "figure.dpi": 120,
    "savefig.dpi": 120,
    "font.size": 11,
    "axes.grid": True,
    "axes.axisbelow": True,
    "grid.color": PALETTE["grid"],
    "grid.linewidth": 0.6,
    "axes.spines.top": False,
    "axes.spines.right": False,
})


def save(fig, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"    saved {path}")
