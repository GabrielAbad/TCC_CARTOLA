from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path

import pandas as pd
import requests

MIN_ROUND = 1
MAX_ROUND = 38
CARTOLA_API_BASE_URL = "https://api.cartolafc.globo.com"
PROJECT_ROOT = Path(__file__).resolve().parents[2]
CACHE_DIR = PROJECT_ROOT / "data" / "database"

OUTPUT_COLUMNS = [
    "rodada_id",
    "clube_casa_id",
    "clube_casa",
    "clube_visitante_id",
    "clube_visitante",
    "placar_oficial_mandante",
    "placar_oficial_visitante",
]
ID_COLUMNS = [
    "rodada_id",
    "clube_casa_id",
    "clube_visitante_id",
    "placar_oficial_mandante",
    "placar_oficial_visitante",
]
SCORE_COLUMNS = ["placar_oficial_mandante", "placar_oficial_visitante"]
MATCH_COLUMNS = [
    "clube_casa_id",
    "clube_visitante_id",
    *SCORE_COLUMNS,
]


def get_games_by_rounds(
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
        .sort_values(by=["rodada_id", "clube_casa_id"], ignore_index=True)
    )


def _validate_rounds(rodada_inicial: int, rodada_final: int) -> None:
    if not (MIN_ROUND <= rodada_inicial <= rodada_final <= MAX_ROUND):
        raise ValueError(
            "Invalid round range: expected "
            f"{MIN_ROUND} <= rodada_inicial <= rodada_final <= {MAX_ROUND}, "
            f"got rodada_inicial={rodada_inicial}, rodada_final={rodada_final}"
        )


def _round_url(round_number: int) -> str:
    return f"{CARTOLA_API_BASE_URL}/partidas/{round_number}"


def _cache_path(year: int, round_number: int) -> Path:
    return CACHE_DIR / str(year) / f"partidas-{round_number}.json"


def _load_round(year: int, round_number: int) -> dict:
    cache_path = _cache_path(year, round_number)
    if cache_path.exists():
        return json.loads(cache_path.read_text(encoding="utf-8"))

    if year != date.today().year:
        raise FileNotFoundError(
            f"Round data not found: year={year} round={round_number}. "
            "The live Cartola API only serves the current season."
        )

    response = requests.get(_round_url(round_number), timeout=30)
    if response.status_code == 404:
        raise FileNotFoundError(
            f"Round data not found: year={year} round={round_number}"
        )
    response.raise_for_status()

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_bytes(response.content)
    return response.json()


def _club_display_name(club: dict) -> str:
    slug = (club.get("slug") or "").strip()
    if slug:
        return slug.replace("-", " ").title()
    return club.get("nome") or club.get("abreviacao") or str(club["id"])


def _club_name_map(payload: dict) -> dict[int, str]:
    clubs = payload.get("clubes")
    if not clubs:
        raise ValueError("Missing expected field: clubes")
    return {int(club["id"]): _club_display_name(club) for club in clubs.values()}


def _normalize_round(payload: dict) -> pd.DataFrame:
    matches = payload.get("partidas")
    if matches is None:
        raise ValueError("Missing expected field: partidas")

    rodada_id = payload.get("rodada")
    if rodada_id is None:
        raise ValueError("Missing expected field: rodada")

    normalized = pd.DataFrame(matches)
    missing_columns = [
        column for column in MATCH_COLUMNS if column not in normalized.columns
    ]
    if missing_columns:
        raise ValueError(f"Missing expected columns: {missing_columns}")

    club_names = _club_name_map(payload)
    normalized = normalized.dropna(subset=SCORE_COLUMNS).copy()
    normalized["rodada_id"] = rodada_id
    normalized["clube_casa"] = normalized["clube_casa_id"].map(club_names)
    normalized["clube_visitante"] = normalized["clube_visitante_id"].map(club_names)

    unknown_ids = set(normalized.loc[normalized["clube_casa"].isna(), "clube_casa_id"])
    unknown_ids.update(normalized.loc[normalized["clube_visitante"].isna(), "clube_visitante_id"])
    if unknown_ids:
        raise ValueError(
            f"Missing club names for ids: {sorted(int(club_id) for club_id in unknown_ids)}"
        )

    normalized = normalized.loc[:, OUTPUT_COLUMNS].copy()
    normalized[ID_COLUMNS] = normalized[ID_COLUMNS].astype("int64")
    return normalized


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Load Cartola FC match results for a round range."
    )
    parser.add_argument("rodada_inicial", type=int)
    parser.add_argument("rodada_final", type=int)
    parser.add_argument("--year", type=int, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    games = get_games_by_rounds(
        rodada_inicial=args.rodada_inicial,
        rodada_final=args.rodada_final,
        year=args.year,
    )
    games.to_csv(CACHE_DIR / "games.csv", index=False)
