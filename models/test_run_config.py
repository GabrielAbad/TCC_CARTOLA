from __future__ import annotations

from pathlib import Path

import pytest

from models.run_config import RunConfig, load_run_config
from models.run_train_eval import parse_args


def test_defaults_match_first_turn() -> None:
    config = load_run_config({})
    assert config.test_rounds == list(range(5, 19))
    assert config.output_tag == ""
    assert config.seed_results_path is None
    assert config.output_name("predicoes_walkforward") == "predicoes_walkforward.csv"


def test_env_overrides() -> None:
    config = load_run_config(
        {
            "TEST_ROUND_START": "19",
            "TEST_ROUND_END": "34",
            "OUTPUT_TAG": "_segundo_turno",
            "SEED_RESULTS_PATH": "models/outputs/resultados_walkforward.csv",
        }
    )
    assert config.test_rounds[0] == 19
    assert config.test_rounds[-1] == 34
    assert len(config.test_rounds) == 16
    assert config.output_name("tabela_6_1") == "tabela_6_1_segundo_turno.csv"
    assert config.output_name("fig_x", ".png") == "fig_x_segundo_turno.png"
    assert config.seed_results_path == Path("models/outputs/resultados_walkforward.csv")


def test_blank_seed_is_none() -> None:
    assert load_run_config({"SEED_RESULTS_PATH": "  "}).seed_results_path is None


def test_env_round_trip() -> None:
    original = RunConfig(19, 34, "_segundo_turno", Path("seed.csv"))
    assert load_run_config(original.to_env()) == original


@pytest.mark.parametrize("start,end", [(0, 18), (19, 18), (5, 39)])
def test_invalid_ranges_raise(start: int, end: int) -> None:
    with pytest.raises(ValueError):
        RunConfig(test_round_start=start, test_round_end=end)


def test_cli_args_build_config() -> None:
    config = parse_args(["--start", "19", "--end", "34", "--tag", "_segundo_turno"])
    assert config.test_rounds == list(range(19, 35))
    assert config.output_tag == "_segundo_turno"
    assert config.seed_results_path is None


def test_cli_defaults() -> None:
    assert parse_args([]) == RunConfig()
