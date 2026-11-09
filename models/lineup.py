"""Cartola FC lineup optimizer (binary ILP via PuLP/CBC) and captain logic."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd
from pulp import (
    PULP_CBC_CMD,
    LpBinary,
    LpMaximize,
    LpProblem,
    LpStatusOptimal,
    LpVariable,
    lpSum,
    value,
)

VAGAS_POS: dict[int, int] = {1: 1, 2: 2, 3: 2, 4: 3, 5: 3, 6: 1}
MAX_CLUBE = 5
POS_TECNICO = 6
SHARPE_EPS = 1e-6
REQUIRED_COLUMNS = ("atleta_id", "posicao_id", "clube_id", "preco_num")


def otimiza_escalacao(
    df_pred: pd.DataFrame,
    orcamento: float,
    score_col: str = "y_hat",
    max_clube: int = MAX_CLUBE,
    vagas_pos: Mapping[int, int] = VAGAS_POS,
    captain_in_objective: bool = False,
) -> pd.DataFrame | None:
    """Solve max sum(score) s.t. budget, slots per position and max players per club.

    With ``captain_in_objective`` the captain's score is counted twice inside the
    objective (used for the hindsight optimum). Returns None when infeasible.
    """
    missing = [c for c in (*REQUIRED_COLUMNS, score_col) if c not in df_pred.columns]
    if missing:
        raise ValueError(f"Missing columns for optimizer: {missing}")

    df = df_pred.reset_index(drop=True)
    idx = list(df.index)
    score = df[score_col].to_numpy(dtype=float)
    price = df["preco_num"].to_numpy(dtype=float)
    pos = df["posicao_id"].to_numpy()
    club = df["clube_id"].to_numpy()

    prob = LpProblem("cartola_escalacao", LpMaximize)
    x = {i: prob.add_variable(f"x_{i}", cat=LpBinary) for i in idx}
    objective = lpSum(score[i] * x[i] for i in idx)

    c: dict[int, LpVariable] = {}
    if captain_in_objective:
        c = {i: prob.add_variable(f"c_{i}", cat=LpBinary) for i in idx if pos[i] != POS_TECNICO}
        objective += lpSum(score[i] * c[i] for i in c)
        prob += lpSum(c.values()) == 1
        for i, ci in c.items():
            prob += ci <= x[i]

    prob += objective
    prob += lpSum(price[i] * x[i] for i in idx) <= orcamento
    for pos_id, n_slots in vagas_pos.items():
        prob += lpSum(x[i] for i in idx if pos[i] == pos_id) == n_slots
    for club_id in np.unique(club):
        prob += lpSum(x[i] for i in idx if club[i] == club_id) <= max_clube

    prob.solve(PULP_CBC_CMD(msg=0))
    if prob.status != LpStatusOptimal:
        return None

    chosen = [i for i in idx if value(x[i]) > 0.5]
    escalacao = df.loc[chosen].copy()
    if captain_in_objective:
        captain_idx = next(i for i, ci in c.items() if value(ci) > 0.5)
        escalacao["capitao"] = escalacao.index == captain_idx
    return escalacao.reset_index(drop=True)


def define_capitao(
    escalacao: pd.DataFrame,
    score_col: str = "y_hat",
    risk_col: str = "pts_std_5r",
) -> pd.DataFrame:
    """Flag the captain as the non-coach player with the highest score / (risk + eps)."""
    out = escalacao.copy()
    score = out[score_col].to_numpy(dtype=float)
    if risk_col in out.columns:
        risk = out[risk_col].fillna(out[risk_col].median()).fillna(0.0).to_numpy(dtype=float)
        sharpe = score / (np.maximum(risk, 0.0) + SHARPE_EPS)
    else:
        sharpe = score.copy()
    sharpe = np.where(out["posicao_id"].to_numpy() == POS_TECNICO, -np.inf, sharpe)
    out["capitao"] = False
    out.loc[out.index[int(np.argmax(sharpe))], "capitao"] = True
    return out


def escalacao_sistema(
    df_pred: pd.DataFrame,
    orcamento: float,
    score_col: str = "y_hat",
    max_clube: int = MAX_CLUBE,
    vagas_pos: Mapping[int, int] = VAGAS_POS,
) -> pd.DataFrame | None:
    """Lineup chosen by the system: ILP on predictions + Sharpe captain."""
    escalacao = otimiza_escalacao(df_pred, orcamento, score_col, max_clube, vagas_pos)
    if escalacao is None:
        return None
    return define_capitao(escalacao, score_col=score_col)


def escalacao_otima(
    df_real: pd.DataFrame,
    orcamento: float,
    points_col: str = "pontos_num",
    max_clube: int = MAX_CLUBE,
    vagas_pos: Mapping[int, int] = VAGAS_POS,
) -> pd.DataFrame | None:
    """Hindsight optimum on real points, with the captain bonus inside the objective."""
    return otimiza_escalacao(
        df_real,
        orcamento,
        score_col=points_col,
        max_clube=max_clube,
        vagas_pos=vagas_pos,
        captain_in_objective=True,
    )


def pontuacao_com_capitao(escalacao: pd.DataFrame, points_col: str = "pontos_num") -> float:
    """Total lineup points with the captain's points doubled."""
    pts = escalacao[points_col].to_numpy(dtype=float)
    captain = escalacao["capitao"].to_numpy(dtype=bool)
    if captain.sum() != 1:
        raise ValueError(f"Lineup must have exactly one captain, got {int(captain.sum())}")
    return float(pts.sum() + pts[captain].sum())
