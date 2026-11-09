"""Build optimizer candidates by joining walk-forward predictions with lineup attributes."""

from __future__ import annotations

import pandas as pd

FEATURE_ATTRS = ["rodada_id", "atleta_id", "clube_id", "preco_num", "pts_std_5r", "status_id", "pontos_num"]


def monta_candidatos(pred: pd.DataFrame, features: pd.DataFrame, modelo: str) -> pd.DataFrame:
    """One row per (rodada, atleta) with ``y_hat`` from ``y_hat_{modelo}``.

    ``preco_num`` comes from df_features, i.e. the purchase price before the round.
    """
    y_col = f"y_hat_{modelo}"
    if y_col not in pred.columns:
        raise ValueError(f"Prediction column {y_col!r} not found")

    cand = pred.merge(
        features[FEATURE_ATTRS],
        left_on=["rodada", "atleta_id"],
        right_on=["rodada_id", "atleta_id"],
        how="left",
        validate="one_to_one",
    )
    if cand["preco_num"].isna().any():
        n_missing = int(cand["preco_num"].isna().sum())
        raise ValueError(f"{n_missing} predictions without a matching df_features row")
    if not (cand["y_true"] - cand["pontos_num"]).abs().lt(1e-9).all():
        raise ValueError("y_true in predictions does not match pontos_num in df_features")

    cand["y_hat"] = cand[y_col]
    return cand.drop(columns=["rodada_id"])
