"""Figures for section 6.2 (system performance)."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

COR_SISTEMA = "#003366"
COR_OTIMO = "#C0392B"
COR_POSITIVO = "#1a7a4a"
COR_NEGATIVO = "#C0392B"


def _cor_eficiencia(e: float) -> str:
    if e >= 70:
        return COR_POSITIVO
    if e >= 50:
        return "#E67E22"
    return COR_NEGATIVO


def plot_desempenho(
    orc: pd.DataFrame,
    titulo: str,
    path: Path,
    label_sistema: str,
    patrimonio_inicial: float,
) -> None:
    """2x2 panel: (A) wealth, (B) price-variation gain, (C) points vs optimum, (D) efficiency."""
    x = orc["rodada_id"]
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(titulo, fontsize=14, fontweight="bold")

    ax = axes[0, 0]
    ax.plot(x, orc["patrimonio_pos_rodada"], marker="o", color=COR_SISTEMA, linewidth=2, label=label_sistema)
    ax.axhline(patrimonio_inicial, color="gray", linestyle="--", alpha=0.7, label=f"Inicial ({patrimonio_inicial:.1f} C$)")
    ax.fill_between(x, patrimonio_inicial, orc["patrimonio_pos_rodada"], alpha=0.15, color=COR_SISTEMA)
    ax.set_title("(A) Evolução Patrimonial (após a rodada)", fontweight="bold")
    ax.set_xlabel("Rodada")
    ax.set_ylabel("Cartoletas")

    ax = axes[0, 1]
    ganho = orc["ganho_variacao_rodada"]
    ax.bar(x, ganho, color=[COR_POSITIVO if g >= 0 else COR_NEGATIVO for g in ganho], alpha=0.85)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_title("(B) Valorização do Time Escalado por Rodada", fontweight="bold")
    ax.set_xlabel("Rodada")
    ax.set_ylabel("Δ Cartoletas")

    ax = axes[1, 0]
    ax.plot(x, orc["pontos_obtidos"], marker="o", color=COR_SISTEMA, linewidth=2, label="Sistema")
    ax.plot(x, orc["pontos_otimos"], marker="s", color=COR_OTIMO, linewidth=2, linestyle="--", label="Ótimo (hindsight)")
    ax.plot(x, orc["pontos_previstos"], marker=".", color="gray", linewidth=1, linestyle=":", label="Previsto pelo sistema")
    ax.set_title("(C) Pontuação por Rodada (capitão ×2)", fontweight="bold")
    ax.set_xlabel("Rodada")
    ax.set_ylabel("Pontos")

    ax = axes[1, 1]
    eff = orc["eficiencia_pct"]
    ax.bar(x, eff, color=[_cor_eficiencia(e) for e in eff], alpha=0.85)
    ax.axhline(eff.mean(), color="black", linestyle="--", linewidth=1.5, label=f"Média: {eff.mean():.1f}%")
    ax.set_ylim(0, max(110.0, float(eff.max()) + 5))
    ax.set_title("(D) Eficiência (Sistema / Ótimo × 100%)", fontweight="bold")
    ax.set_xlabel("Rodada")
    ax.set_ylabel("Eficiência (%)")

    for ax in axes.flat:
        ax.set_xticks(list(x))
        ax.grid(alpha=0.3)
        if ax.get_legend_handles_labels()[0]:
            ax.legend()

    plt.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
