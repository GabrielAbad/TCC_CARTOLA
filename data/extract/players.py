from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path

import pandas as pd
import requests

MIN_ROUND = 1
MAX_ROUND = 38
CARTOLA_RAW_BASE_URL = (
    "https://raw.githubusercontent.com/henriquepgomide/caRtola/master/data/01_raw"
)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
CACHE_DIR = PROJECT_ROOT / "data" / "database"

SOURCE_TO_OUTPUT = {
    "atletas.rodada_id": "rodada_id",
    "atletas.atleta_id": "atleta_id",
    "atletas.apelido": "apelido",
    "atletas.posicao_id": "posicao_id",
    "atletas.clube_id": "clube_id",
    "atletas.pontos_num": "pontos_num",
    "atletas.preco_num": "preco_num",
    "atletas.media_num": "media_num",
    "atletas.variacao_num": "variacao_num",
    "atletas.status_id": "status_id",
}
SCOUT_COLUMNS = [
    "G",
    "A",
    "FT",
    "FF",
    "FD",
    "FS",
    "FC",
    "CA",
    "CV",
    "DS",
    "DE",
    "SG",
    "GC",
    "GS",
    "I",
    "PC",
    "PS",
    "V",
]
OUTPUT_COLUMNS = [
    "rodada_id",
    "atleta_id",
    "apelido",
    "posicao_id",
    "clube_id",
    "pontos_num",
    "preco_num",
    "media_num",
    "variacao_num",
    "status_id",
    *SCOUT_COLUMNS,
]
ID_COLUMNS = ["rodada_id", "atleta_id", "posicao_id", "clube_id", "status_id"]
FLOAT_COLUMNS = ["pontos_num", "preco_num", "media_num", "variacao_num", *SCOUT_COLUMNS]


def get_players_by_rounds(
    rodada_inicial: int,
    rodada_final: int,
    year: int | None = None,
) -> pd.DataFrame:
    _validate_rounds(rodada_inicial, rodada_final)
    season_year = year if year is not None else date.today().year
    rounds = [
        _normalize_round(_load_round(season_year, round_number))
        for round_number in range(rodada_inicial, rodada_final + 1)
    ]
    return (
        pd.concat(rounds, ignore_index=True)
        .sort_values(by=["rodada_id", "atleta_id"], ignore_index=True)
    )


def _validate_rounds(rodada_inicial: int, rodada_final: int) -> None:
    if not (MIN_ROUND <= rodada_inicial <= rodada_final <= MAX_ROUND):
        raise ValueError(
            "Invalid round range: expected "
            f"{MIN_ROUND} <= rodada_inicial <= rodada_final <= {MAX_ROUND}, "
            f"got rodada_inicial={rodada_inicial}, rodada_final={rodada_final}"
        )


def _round_url(year: int, round_number: int) -> str:
    return f"{CARTOLA_RAW_BASE_URL}/{year}/rodada-{round_number}.csv"


def _cache_path(year: int, round_number: int) -> Path:
    return CACHE_DIR / str(year) / f"rodada-{round_number}.csv"


def _load_round(year: int, round_number: int) -> pd.DataFrame:
    cache_path = _cache_path(year, round_number)
    if cache_path.exists():
        return pd.read_csv(cache_path)

    url = _round_url(year, round_number)
    response = requests.get(url, timeout=30)
    if response.status_code == 404:
        raise FileNotFoundError(
            f"Round data not found: year={year} round={round_number}"
        )
    response.raise_for_status()

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_bytes(response.content)
    return pd.read_csv(cache_path)


def _normalize_round(df: pd.DataFrame) -> pd.DataFrame:
    normalized = df.drop(columns=["Unnamed: 0"], errors="ignore").rename(
        columns=SOURCE_TO_OUTPUT
    )
    for column in SCOUT_COLUMNS:
        if column not in normalized.columns:
            normalized[column] = 0.0

    missing_columns = [
        column for column in OUTPUT_COLUMNS if column not in normalized.columns
    ]
    if missing_columns:
        raise ValueError(f"Missing expected columns: {missing_columns}")

    normalized = normalized.loc[:, OUTPUT_COLUMNS].copy()
    normalized[SCOUT_COLUMNS] = normalized[SCOUT_COLUMNS].fillna(0)
    normalized[ID_COLUMNS] = normalized[ID_COLUMNS].astype("int64")
    normalized[FLOAT_COLUMNS] = normalized[FLOAT_COLUMNS].astype("float64")
    return normalized


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Load Cartola FC player stats for a round range."
    )
    parser.add_argument("rodada_inicial", type=int)
    parser.add_argument("rodada_final", type=int)
    parser.add_argument("--year", type=int, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    players = get_players_by_rounds(
        rodada_inicial=args.rodada_inicial,
        rodada_final=args.rodada_final,
        year=args.year,
    )
    players.to_csv(CACHE_DIR / "players.csv", index=False)
