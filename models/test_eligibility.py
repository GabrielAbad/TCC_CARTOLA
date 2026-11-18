from __future__ import annotations

from models.eligibility import PROBABLE_STATUS_ID, STATUS_CANDIDATOS
from models.simulation import STATUS_CANDIDATOS as SIM_STATUS


def test_status_candidatos_is_probable_only() -> None:
    assert PROBABLE_STATUS_ID == 7
    assert STATUS_CANDIDATOS == frozenset({7})
    assert SIM_STATUS == STATUS_CANDIDATOS
