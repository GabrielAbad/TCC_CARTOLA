"""Section 6.2 system-level metrics computed from simulation histories."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd

from models.simulation import SimulationResult


def resumo_cenario(
    res: SimulationResult, cenario: str, patrimonio_inicial: float
) -> dict[str, object]:
    orc = res.historico_orcamento
    sistema = res.historico_escalacoes.query("escalacao == 'sistema'")
    pts_total = float(orc["pontos_obtidos"].sum())
    otimo_total = float(orc["pontos_otimos"].sum())
    patrimonio_final = float(orc["patrimonio_pos_rodada"].iloc[-1])
    return {
        "cenario": cenario,
        "rodadas": f"{int(orc['rodada_id'].min())}-{int(orc['rodada_id'].max())}",
        "n_rodadas": len(orc),
        "pts_sistema_medio": float(orc["pontos_obtidos"].mean()),
        "pts_otimo_medio": float(orc["pontos_otimos"].mean()),
        "pts_previstos_medio": float(orc["pontos_previstos"].mean()),
        "eficiencia_media_pct": float(orc["eficiencia_pct"].mean()),
        "eficiencia_min_pct": float(orc["eficiencia_pct"].min()),
        "eficiencia_max_pct": float(orc["eficiencia_pct"].max()),
        "eficiencia_agregada_pct": pts_total / otimo_total * 100 if otimo_total > 0 else np.nan,
        "bonus_capitao_medio": float(orc["capitao_pontos"].mean()),
        "patrimonio_inicial": patrimonio_inicial,
        "patrimonio_final": patrimonio_final,
        "variacao_patrimonio_pct": (patrimonio_final / patrimonio_inicial - 1) * 100,
        "trocas_medias": float(orc["n_trocas"].iloc[1:].mean()) if len(orc) > 1 else np.nan,
        "mae_escalados": float(sistema["erro_previsao"].abs().mean()),
        "vies_escalados": float(sistema["erro_previsao"].mean()),
    }


def tabela_6_2(
    resultados: Mapping[str, SimulationResult], patrimonio_inicial: Mapping[str, float]
) -> pd.DataFrame:
    return pd.DataFrame(
        [resumo_cenario(res, c, patrimonio_inicial[c]) for c, res in resultados.items()]
    ).round(3)


def erro_por_posicao(resultados: Mapping[str, SimulationResult]) -> pd.DataFrame:
    """Prediction error of the system's selected athletes, per scenario and position.

    ``vies`` = mean(real - previsto): negative means the model overestimated who it picked.
    """
    frames = []
    for cenario, res in resultados.items():
        sistema = res.historico_escalacoes.query("escalacao == 'sistema'")
        agg = (
            sistema.groupby(["posicao_id", "posicao"])
            .agg(
                n=("erro_previsao", "size"),
                mae=("erro_previsao", lambda e: e.abs().mean()),
                vies=("erro_previsao", "mean"),
                pts_previstos_medio=("pontos_previstos", "mean"),
                pts_obtidos_medio=("pontos_obtidos", "mean"),
            )
            .reset_index()
            .assign(cenario=cenario)
        )
        frames.append(agg)
    cols = ["cenario", "posicao_id", "posicao", "n", "mae", "vies", "pts_previstos_medio", "pts_obtidos_medio"]
    return pd.concat(frames, ignore_index=True)[cols].round(3)


def compara_turnos_6_1(tab_primeiro: pd.DataFrame, tab_segundo: pd.DataFrame) -> pd.DataFrame:
    """Side-by-side model metrics (Tabela 6.1) for the two turns, indexed by model."""
    metrics = ["RMSE", "MAE", "R2", "Spearman", "Top3Prec"]
    a = tab_primeiro.set_index("modelo")[metrics].add_suffix("_1turno")
    b = tab_segundo.set_index("modelo")[metrics].add_suffix("_2turno")
    out = a.join(b, how="outer")
    for m in metrics:
        out[f"delta_{m}"] = out[f"{m}_2turno"] - out[f"{m}_1turno"]
    return out.round(3).sort_values("MAE_2turno")
