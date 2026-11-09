"""Dynamic budget (cartoletas) of a simulated Cartola FC user."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import pandas as pd

ORCAMENTO_INICIAL = 130.0


@dataclass(frozen=True)
class EstadoUsuario:
    """User state between rounds.

    ``saldo`` is the cash left after buying the team in ``rodada_time``;
    ``time_ids`` is that team (empty before the first lineup).
    """

    saldo: float = ORCAMENTO_INICIAL
    time_ids: tuple[int, ...] = field(default_factory=tuple)
    rodada_time: int | None = None


@dataclass(frozen=True)
class Valorizacao:
    patrimonio: float
    ganho_variacao: float
    detalhe: pd.DataFrame


def valoriza_time(
    time_ids: Sequence[int], rodada_jogada: int, players: pd.DataFrame
) -> Valorizacao:
    """Team value after ``rodada_jogada`` using post-round prices from players.csv.

    In players.csv ``preco_num`` of round r is the price after r was played
    (= purchase price for round r + 1), and ``variacao_num`` is the change in r.
    """
    detalhe = players.loc[
        (players["rodada_id"] == rodada_jogada) & players["atleta_id"].isin(time_ids),
        ["atleta_id", "apelido", "preco_num", "variacao_num"],
    ].copy()
    missing = set(time_ids) - set(detalhe["atleta_id"])
    if missing:
        raise ValueError(f"Athletes {sorted(missing)} not found in round {rodada_jogada}")
    return Valorizacao(
        patrimonio=float(detalhe["preco_num"].sum()),
        ganho_variacao=float(detalhe["variacao_num"].sum()),
        detalhe=detalhe.reset_index(drop=True),
    )


def orcamento_disponivel(estado: EstadoUsuario, players: pd.DataFrame) -> tuple[float, Valorizacao | None]:
    """Budget for the next round: cash + market value of the current team."""
    if not estado.time_ids or estado.rodada_time is None:
        return estado.saldo, None
    val = valoriza_time(estado.time_ids, estado.rodada_time, players)
    return estado.saldo + val.patrimonio, val


def compra_time(
    orcamento: float, custo_time: float, time_ids: Sequence[int], rodada: int
) -> EstadoUsuario:
    if custo_time > orcamento + 1e-6:
        raise ValueError(f"Team cost {custo_time:.2f} exceeds budget {orcamento:.2f}")
    return EstadoUsuario(
        saldo=orcamento - custo_time,
        time_ids=tuple(int(i) for i in time_ids),
        rodada_time=rodada,
    )


def calcula_n_trocas(time_anterior: Sequence[int], time_novo: Sequence[int]) -> int:
    """Number of athletes replaced between two lineups (0 for the first one)."""
    if not time_anterior:
        return 0
    return len(set(time_novo) - set(time_anterior))
