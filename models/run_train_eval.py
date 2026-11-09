"""Execute all code cells from train_eval.ipynb (long-running walk-forward).

Examples:
    python models/run_train_eval.py
    python models/run_train_eval.py --start 19 --end 34 --tag _segundo_turno \
        --seed-results models/outputs/resultados_walkforward.csv
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.notebook_runner import run_notebook  # noqa: E402
from models.run_config import (  # noqa: E402
    DEFAULT_TEST_ROUND_END,
    DEFAULT_TEST_ROUND_START,
    RunConfig,
)


def parse_args(argv: list[str] | None = None) -> RunConfig:
    parser = argparse.ArgumentParser(description="Run the train_eval walk-forward.")
    parser.add_argument("--start", type=int, default=DEFAULT_TEST_ROUND_START)
    parser.add_argument("--end", type=int, default=DEFAULT_TEST_ROUND_END)
    parser.add_argument("--tag", type=str, default="")
    parser.add_argument("--seed-results", type=Path, default=None)
    args = parser.parse_args(argv)
    seed = args.seed_results.resolve() if args.seed_results else None
    return RunConfig(
        test_round_start=args.start,
        test_round_end=args.end,
        output_tag=args.tag,
        seed_results_path=seed,
    )


def main(argv: list[str] | None = None) -> None:
    config = parse_args(argv)
    os.environ.update(config.to_env())
    print(f"Run config: {config}")
    run_notebook(Path(__file__).with_name("train_eval.ipynb"))


if __name__ == "__main__":
    main()
