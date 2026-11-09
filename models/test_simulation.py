from __future__ import annotations

import pandas as pd
import pytest

from models.budget import ORCAMENTO_INICIAL, EstadoUsuario
from models.simulation import filtra_candidatos, simula_cenario
from models.test_lineup import make_pool

POS_NOMES = {1: "Goleiro", 2: "Lateral", 3: "Zagueiro", 4: "Meia", 5: "Atacante", 6: "Técnico"}


def make_scenario(rodadas: list[int]) -> tuple[pd.DataFrame, pd.DataFrame]:
    cands, players = [], []
    for r in rodadas:
        pool = make_pool(seed=r).assign(rodada=r, status_id=7)
        pool["posicao"] = pool["posicao_id"].map(POS_NOMES)
        cands.append(pool)
        players.append(
            pool[["atleta_id", "apelido"]].assign(
                rodada_id=r,
                variacao_num=0.5,
                preco_num=pool["preco_num"] + 0.5,
            )
        )
    return pd.concat(cands, ignore_index=True), pd.concat(players, ignore_index=True)


def test_filtra_candidatos_drops_unavailable_status() -> None:
    cand, _ = make_scenario([5])
    cand.loc[cand.index[:3], "status_id"] = 5
    out = filtra_candidatos(cand, 5)
    assert len(out) == len(cand) - 3
    assert out["status_id"].isin([6, 7]).all()


def test_simula_cenario_outputs_and_budget_flow() -> None:
    cand, players = make_scenario([5, 6, 7])
    res = simula_cenario(cand, players, [5, 6, 7], EstadoUsuario(), verbose=False)
    orc = res.historico_orcamento
    esc = res.historico_escalacoes

    assert orc["rodada_id"].tolist() == [5, 6, 7]
    assert orc.loc[0, "orcamento_disponivel"] == ORCAMENTO_INICIAL
    assert (orc["custo_time"] <= orc["orcamento_disponivel"] + 1e-6).all()
    assert (orc["pontos_obtidos"] <= orc["pontos_otimos"] + 1e-9).all()
    assert orc.loc[0, "n_trocas"] == 0

    # every athlete gains +0.5 per round, so the budget grows by 12 * 0.5
    assert orc.loc[1, "orcamento_disponivel"] == pytest.approx(ORCAMENTO_INICIAL + 6.0)
    assert orc.loc[0, "patrimonio_pos_rodada"] == pytest.approx(ORCAMENTO_INICIAL + 6.0)

    for tipo in ("sistema", "otima"):
        sub = esc[esc["escalacao"] == tipo]
        assert sub.groupby("rodada_id").size().eq(12).all()
        assert sub.groupby("rodada_id")["capitao"].sum().eq(1).all()

    assert res.estado_final.rodada_time == 7
    assert len(res.estado_final.time_ids) == 12


def test_second_scenario_continues_from_final_state() -> None:
    cand, players = make_scenario([5, 6, 7, 8])
    first = simula_cenario(cand, players, [5, 6], EstadoUsuario(), verbose=False)
    second = simula_cenario(cand, players, [7, 8], first.estado_final, verbose=False)
    full = simula_cenario(cand, players, [5, 6, 7, 8], EstadoUsuario(), verbose=False)
    pd.testing.assert_frame_equal(
        second.historico_orcamento.reset_index(drop=True),
        full.historico_orcamento.iloc[2:].reset_index(drop=True),
    )
