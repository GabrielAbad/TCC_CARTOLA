from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import pandas as pd
import pytest

from models.budget import ORCAMENTO_INICIAL, EstadoUsuario
from models.candidates import monta_candidatos
from models.evaluation import compara_turnos_6_1, erro_por_posicao, tabela_6_2
from models.report import plot_desempenho
from models.simulation import simula_cenario
from models.test_simulation import make_scenario


@pytest.fixture(scope="module")
def resultados() -> dict[str, object]:
    cand, players = make_scenario([5, 6, 7, 8])
    a = simula_cenario(cand, players, [5, 6], EstadoUsuario(), verbose=False)
    b = simula_cenario(cand, players, [7, 8], a.estado_final, verbose=False)
    return {"A": a, "B": b}


def test_tabela_6_2_one_row_per_scenario(resultados: dict) -> None:
    pat_b = float(resultados["A"].historico_orcamento["patrimonio_pos_rodada"].iloc[-1])
    tab = tabela_6_2(resultados, {"A": ORCAMENTO_INICIAL, "B": pat_b})
    assert tab["cenario"].tolist() == ["A", "B"]
    assert tab["rodadas"].tolist() == ["5-6", "7-8"]
    assert (tab["eficiencia_media_pct"] <= 100 + 1e-6).all()
    assert tab.loc[0, "patrimonio_final"] == pytest.approx(pat_b, abs=1e-3)


def test_erro_por_posicao_counts(resultados: dict) -> None:
    err = erro_por_posicao(resultados)
    a = err[err["cenario"] == "A"].set_index("posicao_id")["n"]
    assert a.to_dict() == {1: 2, 2: 4, 3: 4, 4: 6, 5: 6, 6: 2}
    assert (err["mae"] >= 0).all()


def test_compara_turnos_deltas() -> None:
    t1 = pd.DataFrame({"modelo": ["rf"], "RMSE": [2.0], "MAE": [1.5], "R2": [0.1], "Spearman": [0.4], "Top3Prec": [0.1]})
    t2 = t1.assign(MAE=[1.7])
    out = compara_turnos_6_1(t1, t2)
    assert out.loc["rf", "delta_MAE"] == pytest.approx(0.2)


def test_monta_candidatos_joins_and_validates() -> None:
    pred = pd.DataFrame(
        {"rodada": [5, 5], "atleta_id": [1, 2], "y_true": [3.0, 4.0], "y_hat_rf": [2.5, 3.5]}
    )
    feats = pd.DataFrame(
        {
            "rodada_id": [5, 5, 6],
            "atleta_id": [1, 2, 1],
            "clube_id": [10, 11, 10],
            "preco_num": [5.0, 6.0, 5.5],
            "pts_std_5r": [1.0, 2.0, 1.0],
            "status_id": [7, 6, 7],
            "pontos_num": [3.0, 4.0, 9.0],
        }
    )
    cand = monta_candidatos(pred, feats, "rf")
    assert cand["y_hat"].tolist() == [2.5, 3.5]
    assert cand["preco_num"].tolist() == [5.0, 6.0]
    with pytest.raises(ValueError):
        monta_candidatos(pred, feats, "xgboost")
    with pytest.raises(ValueError):
        monta_candidatos(pred.assign(y_true=[0.0, 4.0]), feats, "rf")


def test_plot_desempenho_writes_png(resultados: dict, tmp_path: Path) -> None:
    out = tmp_path / "fig.png"
    plot_desempenho(resultados["A"].historico_orcamento, "t", out, "RF", ORCAMENTO_INICIAL)
    assert out.exists() and out.stat().st_size > 0
