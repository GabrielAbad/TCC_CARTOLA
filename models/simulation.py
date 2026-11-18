"""Round-by-round end-to-end simulation: predictions -> ILP lineup -> real points -> budget."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

from models.budget import (
    EstadoUsuario,
    calcula_n_trocas,
    compra_time,
    orcamento_disponivel,
    valoriza_time,
)
from models.eligibility import STATUS_CANDIDATOS
from models.lineup import escalacao_otima, escalacao_sistema, pontuacao_com_capitao

CANDIDATE_COLUMNS = (
    "rodada",
    "atleta_id",
    "apelido",
    "posicao_id",
    "posicao",
    "clube_id",
    "preco_num",
    "pts_std_5r",
    "status_id",
    "pontos_num",
    "y_hat",
)


@dataclass(frozen=True)
class SimulationResult:
    historico_orcamento: pd.DataFrame
    historico_escalacoes: pd.DataFrame
    estado_final: EstadoUsuario


def filtra_candidatos(cand: pd.DataFrame, rodada: int) -> pd.DataFrame:
    missing = [c for c in CANDIDATE_COLUMNS if c not in cand.columns]
    if missing:
        raise ValueError(f"Missing candidate columns: {missing}")
    out = cand[(cand["rodada"] == rodada) & cand["status_id"].isin(STATUS_CANDIDATOS)]
    return out.reset_index(drop=True)


def _lineup_records(
    escalacao: pd.DataFrame, rodada: int, tipo: str, players_round: pd.DataFrame
) -> list[dict[str, object]]:
    variacao = players_round.set_index("atleta_id")["variacao_num"]
    records: list[dict[str, object]] = []
    for _, row in escalacao.iterrows():
        atleta_id = int(row["atleta_id"])
        real = float(row["pontos_num"])
        pred = float(row["y_hat"])
        records.append(
            {
                "rodada_id": rodada,
                "escalacao": tipo,
                "atleta_id": atleta_id,
                "apelido": row["apelido"],
                "posicao_id": int(row["posicao_id"]),
                "posicao": row["posicao"],
                "clube_id": int(row["clube_id"]),
                "preco_entrada": float(row["preco_num"]),
                "variacao_rodada": float(variacao.get(atleta_id, np.nan)),
                "pontos_previstos": pred,
                "pontos_obtidos": real,
                "erro_previsao": real - pred,
                "capitao": bool(row["capitao"]),
            }
        )
    return records


def simula_rodada(
    cand_rodada: pd.DataFrame,
    rodada: int,
    estado: EstadoUsuario,
    players: pd.DataFrame,
) -> tuple[dict[str, object], list[dict[str, object]], EstadoUsuario]:
    orcamento, val_anterior = orcamento_disponivel(estado, players)

    sistema = escalacao_sistema(cand_rodada, orcamento)
    otima = escalacao_otima(cand_rodada, orcamento)
    if sistema is None or otima is None:
        raise RuntimeError(f"Infeasible ILP in round {rodada} with budget {orcamento:.2f}")

    custo = float(sistema["preco_num"].sum())
    novo_estado = compra_time(orcamento, custo, sistema["atleta_id"].tolist(), rodada)
    val_pos = valoriza_time(novo_estado.time_ids, rodada, players)

    pts_sistema = pontuacao_com_capitao(sistema)
    pts_otimo = pontuacao_com_capitao(otima)
    capitao = sistema.loc[sistema["capitao"]].iloc[0]

    resumo: dict[str, object] = {
        "rodada_id": rodada,
        "orcamento_disponivel": orcamento,
        "ganho_variacao_anterior": val_anterior.ganho_variacao if val_anterior else 0.0,
        "custo_time": custo,
        "saldo_banco": novo_estado.saldo,
        "ganho_variacao_rodada": val_pos.ganho_variacao,
        "patrimonio_pos_rodada": novo_estado.saldo + val_pos.patrimonio,
        "pontos_previstos": pontuacao_com_capitao(sistema, points_col="y_hat"),
        "pontos_obtidos": pts_sistema,
        "pontos_obtidos_sem_capitao": float(sistema["pontos_num"].sum()),
        "pontos_otimos": pts_otimo,
        "eficiencia_pct": pts_sistema / pts_otimo * 100 if pts_otimo > 0 else np.nan,
        "custo_time_otimo": float(otima["preco_num"].sum()),
        "capitao": capitao["apelido"],
        "capitao_pontos": float(capitao["pontos_num"]),
        "n_trocas": calcula_n_trocas(estado.time_ids, novo_estado.time_ids),
    }
    players_round = players[players["rodada_id"] == rodada]
    escalacoes = _lineup_records(sistema, rodada, "sistema", players_round)
    escalacoes += _lineup_records(otima, rodada, "otima", players_round)
    return resumo, escalacoes, novo_estado


def simula_cenario(
    cand: pd.DataFrame,
    players: pd.DataFrame,
    rodadas: Sequence[int],
    estado_inicial: EstadoUsuario,
    verbose: bool = True,
) -> SimulationResult:
    estado = estado_inicial
    resumos: list[dict[str, object]] = []
    escalacoes: list[dict[str, object]] = []
    for rodada in rodadas:
        resumo, esc, estado = simula_rodada(filtra_candidatos(cand, rodada), rodada, estado, players)
        resumos.append(resumo)
        escalacoes.extend(esc)
        if verbose:
            print(
                f"R{rodada:02d} | orc={resumo['orcamento_disponivel']:.2f} "
                f"custo={resumo['custo_time']:.2f} | pts={resumo['pontos_obtidos']:.2f} "
                f"otimo={resumo['pontos_otimos']:.2f} ef={resumo['eficiencia_pct']:.1f}% "
                f"| trocas={resumo['n_trocas']} cap={resumo['capitao']}"
            )
    return SimulationResult(
        historico_orcamento=pd.DataFrame(resumos),
        historico_escalacoes=pd.DataFrame(escalacoes),
        estado_final=estado,
    )
