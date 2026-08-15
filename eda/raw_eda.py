"""
EDA completa para TCC — Cartola FC
Gera todas as figuras e estatísticas usadas no PDF e no notebook.
"""

from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import seaborn as sns
from scipy import stats
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import warnings
warnings.filterwarnings('ignore')

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "database"
EDA_DIR = PROJECT_ROOT / "data" / "eda"
EDA_DIR.mkdir(parents=True, exist_ok=True)

# ─── Paleta e estilo global ────────────────────────────────────────────────
AZUL      = '#003366'
VERDE     = '#1a7a4a'
LARANJA   = '#E67E22'
CINZA     = '#7F8C8D'
VERMELHO  = '#C0392B'
ROXO      = '#6C3483'
CIANO     = '#148F77'
AMARELO   = '#D4AC0D'

PALETTE_POS = {
    'Goleiro':  '#003366',
    'Zagueiro': '#1a7a4a',
    'Lateral':  '#E67E22',
    'Meia':     '#6C3483',
    'Atacante': '#C0392B',
    'Técnico':  '#148F77',
}

plt.rcParams.update({
    'font.family':       'DejaVu Sans',
    'axes.spines.top':   False,
    'axes.spines.right': False,
    'axes.grid':         True,
    'grid.alpha':        0.25,
    'grid.linestyle':    '--',
    'figure.dpi':        150,
    'axes.titlesize':    13,
    'axes.labelsize':    11,
    'xtick.labelsize':   9,
    'ytick.labelsize':   9,
    'legend.fontsize':   9,
})

# ─── Carrega dados ─────────────────────────────────────────────────────────
players = pd.read_csv(DATA_DIR / "players.csv")
games   = pd.read_csv(DATA_DIR / "games.csv")

POS_MAP = {1:'Goleiro', 2:'Lateral', 3:'Zagueiro', 4:'Meia', 5:'Atacante', 6:'Técnico'}
STATUS_MAP = {7:'Provável', 6:'Dúvida', 5:'Suspenso', 3:'Lesionado', 2:'Não Participará'}
STATUS_PROVAVEL = 7

def _to_bool_series(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series
    if series.dtype == object:
        return series.astype(str).str.strip().str.lower().isin(['true', '1', 'yes'])
    return series.astype(bool)

players['entrou_em_campo'] = _to_bool_series(players['entrou_em_campo'])
players['posicao']      = players['posicao_id'].map(POS_MAP)
players['status_label'] = players['status_id'].map(STATUS_MAP)

mandantes_set = set(zip(games['rodada_id'], games['clube_casa_id']))
players['is_mandante'] = players.apply(
    lambda r: (r['rodada_id'], r['clube_id']) in mandantes_set, axis=1
)
players['mando'] = players['is_mandante'].map({True: 'Mandante', False: 'Visitante'})

# Training population: probable starters who actually entered the field
model_players = players[
    (players['status_id'] == STATUS_PROVAVEL) & players['entrou_em_campo']
].copy()

# ─── Estatísticas sumárias salvas ─────────────────────────────────────────
stats_out = {}
stats_out['funnel'] = {
    'n_total': int(len(players)),
    'n_provavel': int((players['status_id'] == STATUS_PROVAVEL).sum()),
    'n_model': int(len(model_players)),
    'pct_retained': round(len(model_players) / len(players) * 100, 1),
    'n_unique_total': int(players['atleta_id'].nunique()),
    'n_unique_model': int(model_players['atleta_id'].nunique()),
    'n_rounds': int(players['rodada_id'].nunique()),
}

# =============================================================================
# FIG 1 — Distribuição de Pontuações por Posição (violin + strip + stats)
# =============================================================================
def fig_distribuicao_posicao():
    pos_order = ['Goleiro','Lateral','Zagueiro','Meia','Atacante','Técnico']
    colors    = [PALETTE_POS[p] for p in pos_order]
    df = model_players

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.patch.set_facecolor('white')

    # ── Painel A: violino da amostra de treino ──
    ax = axes[0]
    data_all = [df[df['posicao']==p]['pontos_num'].values for p in pos_order]
    parts = ax.violinplot(data_all, positions=range(len(pos_order)),
                          showmedians=True, showextrema=False, widths=0.7)
    for i, (pc, col) in enumerate(zip(parts['bodies'], colors)):
        pc.set_facecolor(col); pc.set_alpha(0.7)
    parts['cmedians'].set_color('black'); parts['cmedians'].set_linewidth(2)
    ax.set_xticks(range(len(pos_order))); ax.set_xticklabels(pos_order, rotation=15)
    ax.set_xlabel('Posição'); ax.set_ylabel('Pontuação (pts)')
    ax.set_title('(A)  Distribuição (violino)\nProvável e em campo', fontweight='bold')
    ax.axhline(0, color='red', linestyle=':', alpha=0.5, linewidth=1)

    # ── Painel B: boxplot da mesma amostra ──
    ax2 = axes[1]
    data_play = [df[df['posicao']==p]['pontos_num'].values for p in pos_order]
    bp = ax2.boxplot(data_play, patch_artist=True, notch=True,
                     medianprops=dict(color='black', linewidth=2),
                     flierprops=dict(marker='o', markersize=2, alpha=0.3))
    for patch, col in zip(bp['boxes'], colors):
        patch.set_facecolor(col); patch.set_alpha(0.7)
    ax2.set_xticks(range(1, len(pos_order)+1)); ax2.set_xticklabels(pos_order, rotation=15)
    ax2.set_xlabel('Posição'); ax2.set_ylabel('Pontuação (pts)')
    ax2.set_title('(B)  Distribuição (boxplot)\nProvável e em campo', fontweight='bold')

    fig.suptitle('Figura 1 — Distribuição de Pontuações por Posição', fontweight='bold', fontsize=14, y=1.01)
    plt.tight_layout()
    plt.savefig(EDA_DIR / 'fig1_dist_posicao.png', dpi=150, bbox_inches='tight')
    plt.close()

    tbl = df.groupby('posicao')['pontos_num'].agg(
        N='count', Media='mean', Mediana='median', Std='std',
        P25=lambda x: x.quantile(0.25), P75=lambda x: x.quantile(0.75),
        Max='max', PctZero=lambda x: (x==0).mean()*100
    ).round(2)
    stats_out['tab_dist_posicao'] = tbl.loc[pos_order]
    print('[OK] Fig 1 — Distribuição por Posição')
    return tbl

# =============================================================================
# FIG 2 — Análise Temporal: média por rodada e séries individuais
# =============================================================================
def fig_temporal():
    pos_order = ['Goleiro','Lateral','Zagueiro','Meia','Atacante']
    df = model_players
    n_rounds = int(df['rodada_id'].max())
    rodada_means = df[df['posicao'].isin(pos_order)].groupby(
        ['rodada_id','posicao'])['pontos_num'].mean().reset_index()

    fig, axes = plt.subplots(2, 1, figsize=(13, 10))
    fig.patch.set_facecolor('white')

    # ── Painel A: evolução temporal por posição ──
    ax = axes[0]
    for pos in pos_order:
        sub = rodada_means[rodada_means['posicao']==pos].sort_values('rodada_id')
        ax.plot(sub['rodada_id'], sub['pontos_num'],
                marker='o', markersize=4, linewidth=2,
                color=PALETTE_POS[pos], label=pos)
    ax.set_xlabel('Rodada'); ax.set_ylabel('Pontuação Média (pts)')
    ax.set_title('(A)  Evolução Temporal da Pontuação Média por Posição', fontweight='bold')
    ax.legend(loc='upper right', framealpha=0.9)
    ax.set_xticks(range(1, n_rounds + 1))

    # ── Painel B: heatmap rodada × posição ──
    ax2 = axes[1]
    pivot = rodada_means.pivot(index='posicao', columns='rodada_id', values='pontos_num')
    pivot = pivot.loc[pos_order]
    sns.heatmap(pivot, ax=ax2, cmap='YlOrRd', annot=True, fmt='.1f',
                linewidths=0.3, linecolor='white',
                cbar_kws={'label': 'Pontuação Média (pts)', 'shrink': 0.8},
                annot_kws={'size': 8})
    ax2.set_xlabel('Rodada'); ax2.set_ylabel('Posição')
    ax2.set_title('(B)  Heatmap de Pontuação Média — Posição × Rodada', fontweight='bold')

    fig.suptitle('Figura 2 — Análise Temporal de Desempenho', fontweight='bold', fontsize=14)
    plt.tight_layout()
    plt.savefig(EDA_DIR / 'fig2_temporal.png', dpi=150, bbox_inches='tight')
    plt.close()

    cv_by_rodada = df.groupby('rodada_id')['pontos_num'].agg(
        Media='mean', Std='std'
    )
    cv_by_rodada['CV'] = (cv_by_rodada['Std'] / cv_by_rodada['Media'] * 100).round(1)
    stats_out['tab_temporal_cv'] = cv_by_rodada.round(2)
    print('[OK] Fig 2 — Temporal')

# =============================================================================
# FIG 3 — ACF/PACF para série de pontuação individual
# =============================================================================
def fig_acf_pacf():
    # Players with enough observed rounds in the training sample (no imputed zeros)
    player_counts = model_players.groupby('atleta_id').agg(
        n_rodadas=('rodada_id','count'), media=('pontos_num','mean'), apelido=('apelido','first'),
        posicao=('posicao','first')
    ).reset_index()
    min_rounds = 10
    player_counts = player_counts[player_counts['n_rodadas'] >= min_rounds]

    exemplos = []
    for pos in ['Atacante', 'Meia', 'Goleiro']:
        sub = player_counts[player_counts['posicao']==pos].sort_values('media', ascending=False)
        if len(sub) > 0:
            exemplos.append(sub.iloc[0])

    if not exemplos:
        print('[SKIP] Fig 3 — ACF/PACF (insufficient series)')
        return

    fig, axes = plt.subplots(len(exemplos), 2, figsize=(13, 4*len(exemplos)))
    fig.patch.set_facecolor('white')

    for i, player_row in enumerate(exemplos):
        pid = player_row['atleta_id']
        apelido = player_row['apelido']
        posicao = player_row['posicao']
        serie = (
            model_players[model_players['atleta_id']==pid]
            .sort_values('rodada_id')['pontos_num']
            .values
        )

        col = PALETTE_POS.get(posicao, AZUL)
        ax_acf  = axes[i, 0] if len(exemplos) > 1 else axes[0]
        ax_pacf = axes[i, 1] if len(exemplos) > 1 else axes[1]

        max_lags_acf = min(10, len(serie)-1)
        plot_acf(serie, ax=ax_acf, lags=max_lags_acf, alpha=0.05,
                 color=col, vlines_kwargs={'colors': col})
        ax_acf.set_title(f'ACF — {apelido} ({posicao})', fontweight='bold')
        ax_acf.set_xlabel('Lag (aparições na amostra)'); ax_acf.set_ylabel('Autocorrelação')

        max_lags = max(1, min(10, len(serie)//2 - 1))
        plot_pacf(serie, ax=ax_pacf, lags=max_lags, alpha=0.05, method='ywm',
                  color=col, vlines_kwargs={'colors': col})
        ax_pacf.set_title(f'PACF — {apelido} ({posicao})', fontweight='bold')
        ax_pacf.set_xlabel('Lag (aparições na amostra)'); ax_pacf.set_ylabel('Autocorrelação Parcial')

    fig.suptitle('Figura 3 — Autocorrelação (ACF) e Autocorrelação Parcial (PACF)\nde Séries de Pontuação Individual', fontweight='bold', fontsize=13)
    plt.tight_layout()
    plt.savefig(EDA_DIR / 'fig3_acf_pacf.png', dpi=150, bbox_inches='tight')
    plt.close()
    print('[OK] Fig 3 — ACF/PACF')

# =============================================================================
# FIG 4 — Análise de Custo-Benefício
# =============================================================================
def fig_custo_beneficio():
    cb = model_players[model_players['preco_num'] > 0].copy()
    cb['cb_ratio'] = cb['pontos_num'] / cb['preco_num']

    pos_order = ['Goleiro','Lateral','Zagueiro','Meia','Atacante']
    cb_pos    = cb[cb['posicao'].isin(pos_order)]

    fig, axes = plt.subplots(1, 3, figsize=(16, 6))
    fig.patch.set_facecolor('white')

    # ── A: Scatter pontos vs preço por posição ──
    ax = axes[0]
    for pos in pos_order:
        sub = cb_pos[cb_pos['posicao']==pos]
        ax.scatter(sub['preco_num'], sub['pontos_num'],
                   alpha=0.18, s=12, color=PALETTE_POS[pos], label=pos)
    ax.set_xlabel('Preço (cartoletas)'); ax.set_ylabel('Pontuação (pts)')
    ax.set_title('(A)  Preço × Pontuação', fontweight='bold')
    ax.legend(markerscale=2, framealpha=0.9)

    # ── B: Distribuição do índice C/B por posição ──
    ax2 = axes[1]
    data_cb = [cb_pos[cb_pos['posicao']==pos]['cb_ratio'].values for pos in pos_order]
    bp = ax2.boxplot(data_cb, patch_artist=True, notch=False,
                     medianprops=dict(color='black', linewidth=2),
                     flierprops=dict(marker='o', markersize=2, alpha=0.3))
    for patch, pos in zip(bp['boxes'], pos_order):
        patch.set_facecolor(PALETTE_POS[pos]); patch.set_alpha(0.75)
    ax2.set_xticklabels(pos_order, rotation=15)
    ax2.set_xlabel('Posição'); ax2.set_ylabel('Pts / Cartoleta')
    ax2.set_title('(B)  Índice Custo-Benefício por Posição', fontweight='bold')

    # ── C: Top 15 jogadores por C/B médio (mínimo 5 rodadas jogadas) ──
    ax3 = axes[2]
    cb_avg = cb_pos.groupby(['atleta_id','apelido','posicao']).agg(
        cb_medio=('cb_ratio','mean'), n=('cb_ratio','count')
    ).reset_index()
    cb_avg = cb_avg[cb_avg['n'] >= 5].sort_values('cb_medio', ascending=False).head(15)
    colors_bar = [PALETTE_POS[p] for p in cb_avg['posicao']]
    bars = ax3.barh(range(len(cb_avg)), cb_avg['cb_medio'], color=colors_bar, alpha=0.8)
    ax3.set_yticks(range(len(cb_avg)))
    ax3.set_yticklabels([f"{r['apelido']} ({r['posicao'][:3]})" for _, r in cb_avg.iterrows()], fontsize=8)
    ax3.set_xlabel('Pts / Cartoleta (média)'); ax3.invert_yaxis()
    ax3.set_title('(C)  Top 15 Custo-Benefício\n(mín. 5 rodadas)', fontweight='bold')
    for bar in bars:
        ax3.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height()/2,
                 f'{bar.get_width():.2f}', va='center', ha='left', fontsize=7)

    fig.suptitle('Figura 4 — Análise de Custo-Benefício', fontweight='bold', fontsize=14)
    plt.tight_layout()
    plt.savefig(EDA_DIR / 'fig4_custo_beneficio.png', dpi=150, bbox_inches='tight')
    plt.close()

    stats_out['tab_cb_posicao'] = cb_pos.groupby('posicao')['cb_ratio'].agg(
        Media='mean', Mediana='median', Std='std', P75=lambda x: x.quantile(0.75)
    ).round(3).loc[pos_order]
    print('[OK] Fig 4 — Custo-Benefício')

# =============================================================================
# FIG 5 — Correlação entre Scouts e Pontuação Final
# =============================================================================
def fig_correlacao_scouts():
    scouts = ['G','A','FT','FF','FD','FS','FC','CA','CV','DS','DE','SG','GC','GS','I']
    scout_names = {
        'G':'Gol', 'A':'Assistência', 'FT':'Finaliz. Trave',
        'FF':'Finaliz. Fora', 'FD':'Finaliz. Def.',
        'FS':'Falta Sofrida', 'FC':'Falta Cometida',
        'CA':'Cartão Amarelo', 'CV':'Cartão Vermelho',
        'DS':'Desarme', 'DE':'Defesa*', 'SG':'Sem Gol*',
        'GC':'Gol Contra', 'GS':'Gol Sofrido', 'I':'Impedimento'
    }

    pos_groups = {
        'Atacante & Meia': model_players[model_players['posicao'].isin(['Atacante','Meia'])],
        'Defensor & Goleiro': model_players[model_players['posicao'].isin(['Goleiro','Lateral','Zagueiro'])],
    }

    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    fig.patch.set_facecolor('white')

    all_corr = {}
    for ax, (group_name, df) in zip(axes, pos_groups.items()):
        corrs_p = {}; corrs_s = {}
        for sc in scouts:
            r_p, p_p = stats.pearsonr(df[sc], df['pontos_num'])
            r_s, p_s = stats.spearmanr(df[sc], df['pontos_num'])
            corrs_p[sc] = r_p; corrs_s[sc] = r_s

        corr_df = pd.DataFrame({'Pearson': corrs_p, 'Spearman': corrs_s}).sort_values('Spearman', key=abs, ascending=False)
        all_corr[group_name] = corr_df

        x = np.arange(len(corr_df))
        width = 0.38
        ax.bar(x - width/2, corr_df['Pearson'],  width, label='Pearson',  color=AZUL,   alpha=0.8)
        ax.bar(x + width/2, corr_df['Spearman'], width, label='Spearman', color=VERDE, alpha=0.8)
        ax.axhline(0, color='black', linewidth=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels([scout_names.get(s, s) for s in corr_df.index], rotation=45, ha='right', fontsize=8)
        ax.set_ylabel('Coeficiente de Correlação')
        ax.set_title(f'({"AB"[list(pos_groups.keys()).index(group_name)]})  {group_name}', fontweight='bold')
        ax.legend(framealpha=0.9)
        ax.set_ylim(-0.6, 0.9)
        # Significance markers
        for i, sc in enumerate(corr_df.index):
            _, pv = stats.spearmanr(df[sc], df['pontos_num'])
            if pv < 0.001: ax.text(i + width/2, corr_df.loc[sc,'Spearman'] + 0.02, '***', ha='center', fontsize=7)
            elif pv < 0.01: ax.text(i + width/2, corr_df.loc[sc,'Spearman'] + 0.02, '**',  ha='center', fontsize=7)

    stats_out['tab_corr_scouts'] = all_corr
    fig.suptitle('Figura 5 — Correlação entre Scouts e Pontuação Final\n(* = exclusivo de Goleiros)', fontweight='bold', fontsize=14)
    plt.tight_layout()
    plt.savefig(EDA_DIR / 'fig5_correlacao_scouts.png', dpi=150, bbox_inches='tight')
    plt.close()
    print('[OK] Fig 5 — Correlação Scouts')

# =============================================================================
# FIG 6 — Análise Mandante vs Visitante
# =============================================================================
def fig_mandante_visitante():
    df = model_players
    pos_order = ['Goleiro','Lateral','Zagueiro','Meia','Atacante']

    fig, axes = plt.subplots(1, 3, figsize=(17, 6))
    fig.patch.set_facecolor('white')

    # ── A: boxplot global ──
    ax = axes[0]
    mando_data = [df[df['mando']=='Mandante']['pontos_num'].values,
                  df[df['mando']=='Visitante']['pontos_num'].values]
    bp = ax.boxplot(mando_data, patch_artist=True, notch=True,
                    medianprops=dict(color='black', linewidth=2),
                    flierprops=dict(marker='o', markersize=2, alpha=0.2))
    bp['boxes'][0].set_facecolor(AZUL);    bp['boxes'][0].set_alpha(0.75)
    bp['boxes'][1].set_facecolor(LARANJA); bp['boxes'][1].set_alpha(0.75)
    ax.set_xticklabels(['Mandante','Visitante'])
    ax.set_ylabel('Pontuação (pts)')
    stat_t, p_val = stats.ttest_ind(
        df[df['mando']=='Mandante']['pontos_num'],
        df[df['mando']=='Visitante']['pontos_num']
    )
    sig = '***' if p_val < 0.001 else ('**' if p_val < 0.01 else ('*' if p_val < 0.05 else 'n.s.'))
    ax.set_title(f'(A)  Global\nt={stat_t:.2f}, p{sig}', fontweight='bold')
    stats_out['mandante_ttest_global'] = {'t': stat_t, 'p': p_val}

    # ── B: diferença média por posição ──
    ax2 = axes[1]
    results = []
    for pos in pos_order:
        grp = df[df['posicao']==pos]
        m_pts = grp[grp['mando']=='Mandante']['pontos_num']
        v_pts = grp[grp['mando']=='Visitante']['pontos_num']
        t, p  = stats.ttest_ind(m_pts, v_pts)
        diff  = m_pts.mean() - v_pts.mean()
        results.append({'posicao': pos, 'diff': diff, 'p': p, 't': t,
                        'mandante_mean': m_pts.mean(), 'visitante_mean': v_pts.mean()})
    res_df = pd.DataFrame(results)
    colors_bar = [VERDE if d > 0 else VERMELHO for d in res_df['diff']]
    bars = ax2.bar(res_df['posicao'], res_df['diff'], color=colors_bar, alpha=0.8)
    ax2.axhline(0, color='black', linewidth=0.8)
    ax2.set_xlabel('Posição'); ax2.set_ylabel('Δ Pontuação (Mandante − Visitante)')
    ax2.set_title('(B)  Diferença por Posição\n(Mandante − Visitante)', fontweight='bold')
    for i, row in res_df.iterrows():
        sig_s = '***' if row['p'] < 0.001 else ('**' if row['p'] < 0.01 else ('*' if row['p'] < 0.05 else ''))
        if sig_s:
            ax2.text(i, row['diff'] + (0.03 if row['diff'] >= 0 else -0.08), sig_s, ha='center', fontsize=10)
    stats_out['tab_mandante_por_posicao'] = res_df.set_index('posicao').round(3)

    # ── C: violin mandante vs visitante por posição ──
    ax3 = axes[2]
    sns.violinplot(data=df[df['posicao'].isin(pos_order)],
                   x='posicao', y='pontos_num', hue='mando',
                   order=pos_order, palette={'Mandante': AZUL, 'Visitante': LARANJA},
                   inner='quartile', alpha=0.7, ax=ax3, split=False)
    ax3.set_xlabel('Posição'); ax3.set_ylabel('Pontuação (pts)')
    ax3.set_title('(C)  Violin por Posição e Mando', fontweight='bold')
    ax3.legend(title='Mando', framealpha=0.9)

    fig.suptitle('Figura 6 — Análise de Mando de Campo', fontweight='bold', fontsize=14)
    plt.tight_layout()
    plt.savefig(EDA_DIR / 'fig6_mandante_visitante.png', dpi=150, bbox_inches='tight')
    plt.close()
    print('[OK] Fig 6 — Mandante vs Visitante')
    return res_df

# =============================================================================
# FIG 7 — Análise de Status e Disponibilidade
# =============================================================================
def fig_status_disponibilidade():
    status_order  = ['Provável','Dúvida','Suspenso','Lesionado','Não Participará']
    status_colors = {
        'Provável':         VERDE,
        'Dúvida':           AMARELO,
        'Suspenso':         VERMELHO,
        'Lesionado':        LARANJA,
        'Não Participará':  CINZA,
    }

    # Participation uses the real entrou_em_campo column (full dataset)
    players['jogou'] = players['entrou_em_campo'].astype(int)

    fig, axes = plt.subplots(2, 2, figsize=(15, 11))
    fig.patch.set_facecolor('white')

    # ── A: Contagem de registros por status ──
    ax = axes[0, 0]
    cnt = players['status_label'].value_counts().reindex(status_order, fill_value=0)
    colors_s = [status_colors[s] for s in status_order]
    bars = ax.barh(status_order, cnt.values, color=colors_s, alpha=0.85)
    ax.set_xlabel('Nº de Registros'); ax.invert_yaxis()
    ax.set_title('(A)  Volume por Status', fontweight='bold')
    for bar, val in zip(bars, cnt.values):
        ax.text(val + 50, bar.get_y() + bar.get_height()/2, f'{val:,}', va='center', fontsize=9)

    # ── B: Pontuação média por status ──
    ax2 = axes[0, 1]
    means = players.groupby('status_label')['pontos_num'].mean().reindex(status_order)
    ax2.bar(status_order, means.values, color=colors_s, alpha=0.85)
    ax2.set_xlabel('Status'); ax2.set_ylabel('Pontuação Média (pts)')
    ax2.set_title('(B)  Pontuação Média por Status', fontweight='bold')
    for i, val in enumerate(means.values):
        ax2.text(i, val + 0.05, f'{val:.2f}', ha='center', fontsize=9, fontweight='bold')
    ax2.set_xticklabels(status_order, rotation=15, ha='right')
    stats_out['tab_status_pontos'] = players.groupby('status_label').agg(
        N=('pontos_num', 'count'),
        Media=('pontos_num', 'mean'),
        Std=('pontos_num', 'std'),
        PctZero=('pontos_num', lambda x: (x == 0).mean() * 100),
        TaxaPartic=('entrou_em_campo', lambda x: x.mean() * 100),
    ).round(2).reindex(status_order)

    # ── C: Taxa de participação (entrou_em_campo) por status ──
    ax3 = axes[1, 0]
    taxa = players.groupby('status_label')['jogou'].mean().reindex(status_order) * 100
    bars3 = ax3.bar(status_order, taxa.values, color=colors_s, alpha=0.85)
    ax3.set_xlabel('Status'); ax3.set_ylabel('Taxa de Participação (%)')
    ax3.set_title('(C)  Taxa de Participação em Campo\n(entrou_em_campo)', fontweight='bold')
    ax3.set_ylim(0, 105)
    for i, val in enumerate(taxa.values):
        ax3.text(i, val + 1, f'{val:.1f}%', ha='center', fontsize=9, fontweight='bold')
    ax3.set_xticklabels(status_order, rotation=15, ha='right')
    stats_out['tab_status_participacao'] = taxa.round(1)

    # ── D: Distribuição de pontuação por status (violin) ──
    ax4 = axes[1, 1]
    data_vio = [players[players['status_label']==s]['pontos_num'].values for s in status_order]
    parts = ax4.violinplot(data_vio, positions=range(len(status_order)),
                           showmedians=True, showextrema=False, widths=0.7)
    for pc, s in zip(parts['bodies'], status_order):
        pc.set_facecolor(status_colors[s]); pc.set_alpha(0.75)
    parts['cmedians'].set_color('black'); parts['cmedians'].set_linewidth(1.5)
    ax4.set_xticks(range(len(status_order)))
    ax4.set_xticklabels(status_order, rotation=15, ha='right')
    ax4.set_xlabel('Status'); ax4.set_ylabel('Pontuação (pts)')
    ax4.set_title('(D)  Distribuição de Pontuação por Status', fontweight='bold')
    ax4.axhline(0, color='red', linestyle=':', alpha=0.5)

    fig.suptitle('Figura 7 — Análise de Status e Disponibilidade', fontweight='bold', fontsize=14)
    plt.tight_layout()
    plt.savefig(EDA_DIR / 'fig7_status_disponibilidade.png', dpi=150, bbox_inches='tight')
    plt.close()
    print('[OK] Fig 7 — Status e Disponibilidade')

# =============================================================================
# FIG 8 — Matriz de correlação dos scouts (heatmap)
# =============================================================================
def fig_heatmap_scouts():
    scouts_sel = ['G','A','FT','FF','FD','FS','FC','CA','CV','DS','DE','SG','GC','GS','pontos_num']
    scout_labels = {
        'G':'Gol','A':'Assist.','FT':'F.Trave','FF':'F.Fora',
        'FD':'F.Def.','FS':'F.Sofrida','FC':'F.Cometida',
        'CA':'Cartão Amarelo','CV':'Cartão Vermelho','DS':'Desarme',
        'DE':'Defesa','SG':'Sem Gol','GC':'Gol Contra','GS':'Gol Sofrido',
        'pontos_num':'Pontuação'
    }

    fig, axes = plt.subplots(1, 2, figsize=(18, 8))
    fig.patch.set_facecolor('white')

    for ax, (label, df) in zip(axes, [
        ('Atacantes & Meias', model_players[model_players['posicao'].isin(['Atacante','Meia'])]),
        ('Defensores & Goleiros', model_players[model_players['posicao'].isin(['Goleiro','Lateral','Zagueiro'])])
    ]):
        corr = df[scouts_sel].corr(method='spearman')
        corr.index   = [scout_labels.get(c, c) for c in corr.index]
        corr.columns = [scout_labels.get(c, c) for c in corr.columns]
        mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
        sns.heatmap(corr, ax=ax, cmap='RdBu_r', center=0, vmin=-1, vmax=1,
                    annot=True, fmt='.2f', annot_kws={'size': 7},
                    linewidths=0.3, linecolor='white',
                    cbar_kws={'label': 'Correlação de Spearman', 'shrink': 0.8})
        ax.set_title(f'{label}\n(Correlação de Spearman)', fontweight='bold')

    fig.suptitle('Figura 8 — Matriz de Correlação entre Scouts e Pontuação', fontweight='bold', fontsize=14)
    plt.tight_layout()
    plt.savefig(EDA_DIR / 'fig8_heatmap_scouts.png', dpi=150, bbox_inches='tight')
    plt.close()
    print('[OK] Fig 8 — Heatmap Scouts')

# ─── Executa tudo ─────────────────────────────────────────────────────────
print("Gerando análises...")
fig_distribuicao_posicao()
fig_temporal()
fig_acf_pacf()
fig_custo_beneficio()
fig_correlacao_scouts()
fig_mandante_visitante()
fig_status_disponibilidade()
fig_heatmap_scouts()

# Salva stats_out para usar no PDF
import pickle
with open(EDA_DIR / 'eda_stats.pkl', 'wb') as f:
    pickle.dump(stats_out, f)
print("\nTodas as figuras geradas com sucesso.")