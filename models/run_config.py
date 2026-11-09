"""Walk-forward run configuration shared by train_eval.ipynb and run_train_eval.py."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

ENV_TEST_ROUND_START = "TEST_ROUND_START"
ENV_TEST_ROUND_END = "TEST_ROUND_END"
ENV_OUTPUT_TAG = "OUTPUT_TAG"
ENV_SEED_RESULTS_PATH = "SEED_RESULTS_PATH"

DEFAULT_TEST_ROUND_START = 5
DEFAULT_TEST_ROUND_END = 18
MIN_ROUND = 1
MAX_ROUND = 38


@dataclass(frozen=True)
class RunConfig:
    test_round_start: int = DEFAULT_TEST_ROUND_START
    test_round_end: int = DEFAULT_TEST_ROUND_END
    output_tag: str = ""
    seed_results_path: Path | None = None

    def __post_init__(self) -> None:
        if not (MIN_ROUND <= self.test_round_start <= self.test_round_end <= MAX_ROUND):
            raise ValueError(
                "Invalid test round range: expected "
                f"{MIN_ROUND} <= start <= end <= {MAX_ROUND}, "
                f"got start={self.test_round_start}, end={self.test_round_end}"
            )

    @property
    def test_rounds(self) -> list[int]:
        return list(range(self.test_round_start, self.test_round_end + 1))

    def output_name(self, stem: str, suffix: str = ".csv") -> str:
        return f"{stem}{self.output_tag}{suffix}"

    def to_env(self) -> dict[str, str]:
        return {
            ENV_TEST_ROUND_START: str(self.test_round_start),
            ENV_TEST_ROUND_END: str(self.test_round_end),
            ENV_OUTPUT_TAG: self.output_tag,
            ENV_SEED_RESULTS_PATH: str(self.seed_results_path) if self.seed_results_path else "",
        }


def load_run_config(env: Mapping[str, str]) -> RunConfig:
    seed = env.get(ENV_SEED_RESULTS_PATH, "").strip()
    return RunConfig(
        test_round_start=int(env.get(ENV_TEST_ROUND_START, DEFAULT_TEST_ROUND_START)),
        test_round_end=int(env.get(ENV_TEST_ROUND_END, DEFAULT_TEST_ROUND_END)),
        output_tag=env.get(ENV_OUTPUT_TAG, ""),
        seed_results_path=Path(seed) if seed else None,
    )
