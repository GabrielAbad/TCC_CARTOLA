from __future__ import annotations

import pandas as pd
import pytest

from models.budget import (
    ORCAMENTO_INICIAL,
    EstadoUsuario,
    calcula_n_trocas,
    compra_time,
    orcamento_disponivel,
    valoriza_time,
)


@pytest.fixture()
def players() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "rodada_id": [5, 5, 5, 6, 6, 6],
            "atleta_id": [1, 2, 3, 1, 2, 3],
            "apelido": ["a", "b", "c", "a", "b", "c"],
            "preco_num": [10.0, 8.0, 5.0, 11.0, 7.5, 5.0],
            "variacao_num": [1.0, -0.5, 0.0, 1.0, -0.5, 0.0],
        }
    )


def test_first_round_budget_is_initial_cash(players: pd.DataFrame) -> None:
    orcamento, val = orcamento_disponivel(EstadoUsuario(), players)
    assert orcamento == ORCAMENTO_INICIAL
    assert val is None


def test_budget_uses_post_round_prices_of_previous_team(players: pd.DataFrame) -> None:
    estado = EstadoUsuario(saldo=20.0, time_ids=(1, 2), rodada_time=5)
    orcamento, val = orcamento_disponivel(estado, players)
    assert val is not None
    assert val.patrimonio == pytest.approx(18.0)
    assert val.ganho_variacao == pytest.approx(0.5)
    assert orcamento == pytest.approx(38.0)


def test_valoriza_missing_athlete_raises(players: pd.DataFrame) -> None:
    with pytest.raises(ValueError):
        valoriza_time([1, 99], 5, players)


def test_compra_time_updates_cash() -> None:
    estado = compra_time(orcamento=130.0, custo_time=125.5, time_ids=[3, 1], rodada=5)
    assert estado.saldo == pytest.approx(4.5)
    assert estado.time_ids == (3, 1)
    assert estado.rodada_time == 5


def test_compra_time_over_budget_raises() -> None:
    with pytest.raises(ValueError):
        compra_time(orcamento=100.0, custo_time=100.5, time_ids=[1], rodada=5)


def test_budget_conservation_buy_then_value(players: pd.DataFrame) -> None:
    purchase_r5 = {1: 9.0, 2: 8.5}
    custo = sum(purchase_r5.values())
    estado = compra_time(ORCAMENTO_INICIAL, custo, list(purchase_r5), rodada=5)
    orcamento_r6, val = orcamento_disponivel(estado, players)
    assert val is not None
    assert orcamento_r6 == pytest.approx(ORCAMENTO_INICIAL + val.ganho_variacao)


@pytest.mark.parametrize(
    "anterior,novo,esperado",
    [((), (1, 2), 0), ((1, 2, 3), (1, 2, 3), 0), ((1, 2, 3), (1, 4, 5), 2)],
)
def test_calcula_n_trocas(anterior: tuple[int, ...], novo: tuple[int, ...], esperado: int) -> None:
    assert calcula_n_trocas(anterior, novo) == esperado
