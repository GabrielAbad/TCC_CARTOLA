"""Execute the code cells of a notebook in a single namespace (no Jupyter needed)."""

from __future__ import annotations

import json
from pathlib import Path


def run_notebook(nb_path: Path) -> dict[str, object]:
    nb = json.loads(nb_path.read_text(encoding="utf-8"))
    globals_dict: dict[str, object] = {"__name__": "__main__"}
    for idx, cell in enumerate(nb["cells"]):
        if cell.get("cell_type") != "code":
            continue
        source = "".join(cell.get("source", []))
        if not source.strip():
            continue
        print(f"\n>>> Running cell {idx}", flush=True)
        exec(compile(source, f"{nb_path.name}:cell{idx}", "exec"), globals_dict)
    print("\nAll notebook code cells finished.")
    return globals_dict
