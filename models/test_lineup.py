from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from models.lineup import (
    MAX_CLUBE,
    POS_TECNICO,
    VAGAS_POS,
    define_capitao,
    escalacao_otima,
    escalacao_sistema,
    otimiza_escalacao,
    pontuacao_com_capitao,
)


def make_pool(n_per_pos: int = 6, n_clubs: int = 4, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    atleta_id = 1
    for pos_id in VAGAS_POS:
        for k in range(n_per_pos):
            rows.append(
                {
                    "atleta_id": atleta_id,
                    "apelido": f"p{atleta_id}",
                    "posicao_id": pos_id,
                    "clube_id": k % n_clubs,
                    "preco_num": float(rng.uniform(2, 15)),
                    "y_hat": float(rng.uniform(0, 10)),
                    "pts_std_5r": float(rng.uniform(0.5, 5)),
                    "pontos_num": float(rng.uniform(-2, 15)),
                }
            )
            atleta_id += 1
    return pd.DataFrame(rows)


def assert_valid_lineup(esc: pd.DataFrame, orcamento: float) -> None:
    assert len(esc) == sum(VAGAS_POS.values()) == 12
    assert esc["posicao_id"].value_counts().to_dict() == VAGAS_POS
    assert esc["preco_num"].sum() <= orcamento + 1e-6
    assert esc["clube_id"].value_counts().max() <= MAX_CLUBE
    assert esc["atleta_id"].is_unique


def test_feasible_lineup_respects_constraints() -> None:
    pool = make_pool()
    esc = otimiza_escalacao(pool, orcamento=200.0)
    assert esc is not None
    assert_valid_lineup(esc, 200.0)


def test_unlimited_budget_picks_best_per_position() -> None:
    pool = make_pool(n_clubs=12)
    esc = otimiza_escalacao(pool, orcamento=1e6)
    assert esc is not None
    expected = sum(
        pool[pool["posicao_id"] == pos]["y_hat"].nlargest(n).sum() for pos, n in VAGAS_POS.items()
    )
    assert esc["y_hat"].sum() == pytest.approx(expected)


def test_tight_budget_buys_cheaper_team() -> None:
    pool = make_pool()
    cheapest = sum(
        pool[pool["posicao_id"] == pos]["preco_num"].nsmallest(n).sum() for pos, n in VAGAS_POS.items()
    )
    orcamento = cheapest + 1.0
    esc = otimiza_escalacao(pool, orcamento=orcamento)
    loose = otimiza_escalacao(pool, orcamento=1e6)
    assert esc is not None and loose is not None
    assert_valid_lineup(esc, orcamento)
    assert esc["y_hat"].sum() <= loose["y_hat"].sum()


def test_budget_below_cheapest_team_is_infeasible() -> None:
    pool = make_pool()
    cheapest = sum(
        pool[pool["posicao_id"] == pos]["preco_num"].nsmallest(n).sum() for pos, n in VAGAS_POS.items()
    )
    assert otimiza_escalacao(pool, orcamento=cheapest - 0.5) is None


def test_missing_position_is_infeasible() -> None:
    pool = make_pool()
    assert otimiza_escalacao(pool[pool["posicao_id"] != 1], orcamento=1e6) is None


def test_club_limit_is_enforced() -> None:
    pool = make_pool(n_per_pos=6, n_clubs=3)
    pool.loc[pool["clube_id"] == 0, "y_hat"] += 100
    esc = otimiza_escalacao(pool, orcamento=1e6)
    assert esc is not None
    assert (esc["clube_id"] == 0).sum() == MAX_CLUBE


def test_missing_columns_raise() -> None:
    with pytest.raises(ValueError):
        otimiza_escalacao(make_pool().drop(columns=["preco_num"]), orcamento=100.0)


def test_captain_is_best_sharpe_and_never_coach() -> None:
    esc = pd.DataFrame(
        {
            "atleta_id": [1, 2, 3],
            "posicao_id": [POS_TECNICO, 4, 5],
            "y_hat": [50.0, 6.0, 8.0],
            "pts_std_5r": [0.1, 1.0, 4.0],
        }
    )
    out = define_capitao(esc)
    assert out["capitao"].sum() == 1
    assert out.loc[out["capitao"], "atleta_id"].item() == 2


def test_captain_nan_risk_uses_median() -> None:
    esc = pd.DataFrame(
        {
            "atleta_id": [1, 2],
            "posicao_id": [4, 5],
            "y_hat": [5.0, 6.0],
            "pts_std_5r": [np.nan, 2.0],
        }
    )
    assert define_capitao(esc)["capitao"].sum() == 1


def test_pontuacao_doubles_captain() -> None:
    esc = pd.DataFrame({"pontos_num": [3.0, 5.0, -1.0], "capitao": [False, True, False]})
    assert pontuacao_com_capitao(esc) == pytest.approx(3 + 5 + -1 + 5)


def test_pontuacao_requires_single_captain() -> None:
    esc = pd.DataFrame({"pontos_num": [3.0, 5.0], "capitao": [True, True]})
    with pytest.raises(ValueError):
        pontuacao_com_capitao(esc)


def test_sistema_lineup_has_one_captain() -> None:
    esc = escalacao_sistema(make_pool(), orcamento=200.0)
    assert esc is not None
    assert esc["capitao"].sum() == 1
    assert POS_TECNICO not in esc.loc[esc["capitao"], "posicao_id"].tolist()


def test_hindsight_beats_system_on_real_points() -> None:
    pool = make_pool(seed=3)
    for orcamento in (90.0, 120.0, 200.0):
        sistema = escalacao_sistema(pool, orcamento)
        otima = escalacao_otima(pool, orcamento)
        assert sistema is not None and otima is not None
        assert_valid_lineup(otima, orcamento)
        assert pontuacao_com_capitao(otima) >= pontuacao_com_capitao(sistema) - 1e-9


def test_hindsight_captain_is_top_real_scorer_among_players() -> None:
    pool = make_pool(seed=5)
    otima = escalacao_otima(pool, orcamento=1e6)
    assert otima is not None
    jogadores = otima[otima["posicao_id"] != POS_TECNICO]
    assert otima.loc[otima["capitao"], "pontos_num"].item() == pytest.approx(
        jogadores["pontos_num"].max()
    )
