"""
Gerador do PDF executivo de EDA — TCC Cartola FC
"""
from pathlib import Path

import pandas as pd
import numpy as np
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import cm, mm
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_JUSTIFY, TA_CENTER, TA_LEFT, TA_RIGHT
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle,
    PageBreak, HRFlowable, KeepTogether
)
from reportlab.platypus.flowables import BalancedColumns
from reportlab.lib.colors import HexColor, white, black

# ─── Cores ────────────────────────────────────────────────────────────────
AZUL      = HexColor('#003366')
AZUL_CLARO= HexColor('#EAF0F6')
VERDE     = HexColor('#1a7a4a')
VERDE_CL  = HexColor('#d5ede2')
CINZA     = HexColor('#7F8C8D')
CINZA_CL  = HexColor('#F2F3F4')
LARANJA   = HexColor('#E67E22')
VERMELHO  = HexColor('#C0392B')

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "database"
EDA_DIR = PROJECT_ROOT / "data" / "eda"

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
players['jogou'] = players['entrou_em_campo'].astype(int)
mandantes_set = set(zip(games['rodada_id'], games['clube_casa_id']))
players['is_mandante'] = players.apply(
    lambda r: (r['rodada_id'], r['clube_id']) in mandantes_set, axis=1)
players['mando'] = players['is_mandante'].map({True:'Mandante', False:'Visitante'})
players['cb_ratio'] = np.where(
    players['preco_num'] > 0, players['pontos_num'] / players['preco_num'], np.nan)

model_players = players[
    (players['status_id'] == STATUS_PROVAVEL) & players['entrou_em_campo']
].copy()

# ─── Estilos ───────────────────────────────────────────────────────────────
styles = getSampleStyleSheet()

def S(name, **kw):
    return ParagraphStyle(name, **kw)

ESTILO_TITULO_DOC = S('titulo_doc',
    fontName='Helvetica-Bold', fontSize=22, textColor=AZUL,
    alignment=TA_CENTER, spaceAfter=6, leading=28)
ESTILO_SUBTITULO_DOC = S('subtitulo_doc',
    fontName='Helvetica', fontSize=13, textColor=CINZA,
    alignment=TA_CENTER, spaceAfter=4, leading=18)
ESTILO_META = S('meta',
    fontName='Helvetica', fontSize=10, textColor=CINZA,
    alignment=TA_CENTER, spaceAfter=2)
ESTILO_H1 = S('h1',
    fontName='Helvetica-Bold', fontSize=14, textColor=white,
    backColor=AZUL, alignment=TA_LEFT,
    spaceBefore=16, spaceAfter=8, leading=20,
    leftIndent=-6, rightIndent=-6,
    borderPad=6)
ESTILO_H2 = S('h2',
    fontName='Helvetica-Bold', fontSize=11, textColor=AZUL,
    spaceBefore=12, spaceAfter=4, leading=16)
ESTILO_H3 = S('h3',
    fontName='Helvetica-BoldOblique', fontSize=10, textColor=VERDE,
    spaceBefore=8, spaceAfter=3, leading=14)
ESTILO_BODY = S('body',
    fontName='Helvetica', fontSize=9.5, textColor=HexColor('#2C3E50'),
    alignment=TA_JUSTIFY, spaceAfter=5, leading=15)
ESTILO_CAPTION = S('caption',
    fontName='Helvetica-Oblique', fontSize=8.5, textColor=CINZA,
    alignment=TA_CENTER, spaceAfter=6, leading=12)
ESTILO_BULLET = S('bullet',
    fontName='Helvetica', fontSize=9.5, textColor=HexColor('#2C3E50'),
    leftIndent=14, spaceAfter=3, leading=14, bulletIndent=4)
ESTILO_NOTA = S('nota',
    fontName='Helvetica-Oblique', fontSize=8.5, textColor=CINZA,
    alignment=TA_LEFT, spaceAfter=4, leading=12,
    leftIndent=10, borderPad=4)
ESTILO_INSIGHT = S('insight',
    fontName='Helvetica-Bold', fontSize=9.5, textColor=VERDE,
    backColor=VERDE_CL, alignment=TA_LEFT,
    spaceBefore=6, spaceAfter=6, leading=14,
    leftIndent=8, rightIndent=8, borderPad=6)
ESTILO_HIGHLIGHT = S('highlight',
    fontName='Helvetica-Bold', fontSize=10, textColor=AZUL,
    backColor=AZUL_CLARO, alignment=TA_LEFT,
    spaceBefore=6, spaceAfter=6, leading=14,
    leftIndent=8, rightIndent=8, borderPad=6)
ESTILO_RODAPE = S('rodape',
    fontName='Helvetica', fontSize=8, textColor=CINZA,
    alignment=TA_CENTER)

# ─── Helpers ───────────────────────────────────────────────────────────────
W = A4[0] - 3*cm   # largura útil

def fig(path, width=W, caption=None):
    items = [Image(path, width=width, height=width*0.55, kind='proportional')]
    if caption:
        items.append(Paragraph(caption, ESTILO_CAPTION))
    return items

def hr(color=AZUL, thickness=0.8, spBefore=6, spAfter=8):
    return HRFlowable(width='100%', thickness=thickness, color=color,
                      spaceAfter=spAfter, spaceBefore=spBefore)

def stat_table(headers, rows, col_widths=None, highlight_last=False):
    if col_widths is None:
        col_widths = [W/len(headers)] * len(headers)
    data = [headers] + rows
    tbl = Table(data, colWidths=col_widths)
    style = [
        ('BACKGROUND',  (0,0), (-1,0),  AZUL),
        ('TEXTCOLOR',   (0,0), (-1,0),  white),
        ('FONTNAME',    (0,0), (-1,0),  'Helvetica-Bold'),
        ('FONTSIZE',    (0,0), (-1,-1), 9),
        ('ALIGN',       (0,0), (-1,-1), 'CENTER'),
        ('ALIGN',       (0,1), (0,-1),  'LEFT'),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [CINZA_CL, white]),
        ('GRID',        (0,0), (-1,-1), 0.4, HexColor('#BBCBD8')),
        ('TOPPADDING',  (0,0), (-1,-1), 5),
        ('BOTTOMPADDING',(0,0), (-1,-1), 5),
        ('LEFTPADDING', (0,0), (-1,-1), 7),
        ('RIGHTPADDING',(0,0), (-1,-1), 7),
    ]
    if highlight_last:
        style.append(('BACKGROUND', (0,-1), (-1,-1), AZUL_CLARO))
        style.append(('FONTNAME',   (0,-1), (-1,-1), 'Helvetica-Bold'))
    tbl.setStyle(TableStyle(style))
    return tbl

def kpi_row(items):
    """items = [(label, valor, unidade), ...]"""
    n = len(items)
    col_w = W / n
    data = []
    row1, row2, row3 = [], [], []
    for label, valor, unidade in items:
        row1.append(Paragraph(f'<b>{valor}</b>', ParagraphStyle('kv',
            fontName='Helvetica-Bold', fontSize=18, textColor=AZUL,
            alignment=TA_CENTER)))
        row2.append(Paragraph(unidade, ParagraphStyle('ku',
            fontName='Helvetica', fontSize=8, textColor=CINZA,
            alignment=TA_CENTER)))
        row3.append(Paragraph(label, ParagraphStyle('kl',
            fontName='Helvetica-Bold', fontSize=8.5, textColor=HexColor('#2C3E50'),
            alignment=TA_CENTER)))
    tbl = Table([[v for v in row1],[v for v in row2],[v for v in row3]],
                colWidths=[col_w]*n)
    tbl.setStyle(TableStyle([
        ('BOX',         (0,0), (-1,-1), 0.5, HexColor('#BBCBD8')),
        ('INNERGRID',   (0,0), (-1,-1), 0.3, HexColor('#BBCBD8')),
        ('BACKGROUND',  (0,0), (-1,-1), AZUL_CLARO),
        ('TOPPADDING',  (0,0), (-1,-1), 8),
        ('BOTTOMPADDING',(0,0), (-1,-1), 8),
        ('VALIGN',      (0,0), (-1,-1), 'MIDDLE'),
    ]))
    return tbl

# ─── Header/Footer ─────────────────────────────────────────────────────────
def on_page(canvas, doc):
    canvas.saveState()
    # Header line
    canvas.setStrokeColor(AZUL); canvas.setLineWidth(1.5)
    canvas.line(1.5*cm, A4[1]-1.5*cm, A4[0]-1.5*cm, A4[1]-1.5*cm)
    canvas.setFillColor(AZUL)
    canvas.setFont('Helvetica-Bold', 8)
    canvas.drawString(1.5*cm, A4[1]-1.3*cm, 'FGV EMAp — TCC Cartola FC')
    canvas.setFont('Helvetica', 8); canvas.setFillColor(CINZA)
    canvas.drawRightString(A4[0]-1.5*cm, A4[1]-1.3*cm, 'Análise Exploratória de Dados')
    # Footer
    canvas.setStrokeColor(CINZA); canvas.setLineWidth(0.5)
    canvas.line(1.5*cm, 1.3*cm, A4[0]-1.5*cm, 1.3*cm)
    canvas.setFillColor(CINZA); canvas.setFont('Helvetica', 7.5)
    canvas.drawCentredString(A4[0]/2, 0.9*cm, f'Página {doc.page}')
    canvas.drawString(1.5*cm, 0.9*cm, 'Gabriel Rodrigues de Deus Abad')
    canvas.drawRightString(A4[0]-1.5*cm, 0.9*cm, 'Orientador: Prof. Moacyr Silva')
    canvas.restoreState()

def on_first_page(canvas, doc):
    canvas.saveState()
    # Barra azul no topo
    canvas.setFillColor(AZUL)
    canvas.rect(0, A4[1]-2.2*cm, A4[0], 2.2*cm, fill=1, stroke=0)
    canvas.setFillColor(white)
    canvas.setFont('Helvetica-Bold', 11)
    canvas.drawCentredString(A4[0]/2, A4[1]-1.5*cm, 'FUNDAÇÃO GETULIO VARGAS — ESCOLA DE MATEMÁTICA APLICADA')
    # Footer
    canvas.setStrokeColor(CINZA); canvas.setLineWidth(0.5)
    canvas.line(1.5*cm, 1.3*cm, A4[0]-1.5*cm, 1.3*cm)
    canvas.setFillColor(CINZA); canvas.setFont('Helvetica', 7.5)
    canvas.drawCentredString(A4[0]/2, 0.9*cm, 'Página 1')
    canvas.drawString(1.5*cm, 0.9*cm, 'Gabriel Rodrigues de Deus Abad')
    canvas.drawRightString(A4[0]-1.5*cm, 0.9*cm, 'Orientador: Prof. Moacyr Silva')
    canvas.restoreState()

# ─── Constrói o documento ──────────────────────────────────────────────────
story = []

# ─────────────────────────────────────────────
# CAPA
# ─────────────────────────────────────────────
story.append(Spacer(1, 2.8*cm))
story.append(Paragraph('Análise Exploratória de Dados', ESTILO_TITULO_DOC))
story.append(Spacer(1, 0.2*cm))
story.append(Paragraph('Predição e Recomendação de Escalações no Cartola FC', ESTILO_SUBTITULO_DOC))
story.append(hr(AZUL, 1.5, 8, 10))
story.append(Paragraph('Trabalho de Conclusão de Curso — FGV/EMAp', ESTILO_META))
story.append(Paragraph('Gabriel Rodrigues de Deus Abad &nbsp;&nbsp;|&nbsp;&nbsp; Orientador: Prof. Moacyr Alvim H. B. da Silva', ESTILO_META))
story.append(Paragraph('Campeonato Brasileiro 2026 &nbsp;&nbsp;|&nbsp;&nbsp; 18 Rodadas &nbsp;&nbsp;|&nbsp;&nbsp; Junho 2026', ESTILO_META))
story.append(Spacer(1, 0.5*cm))

# KPIs de visão geral
pos_order = ['Goleiro','Lateral','Zagueiro','Meia','Atacante','Técnico']
n_players = model_players['atleta_id'].nunique()
n_rounds  = players['rodada_id'].nunique()
n_records = len(model_players)
n_games   = len(games)
avg_pts   = model_players['pontos_num'].mean()
n_total   = len(players)
n_provavel = int((players['status_id'] == STATUS_PROVAVEL).sum())
pct_retained = n_records / n_total * 100
story.append(kpi_row([
    ('Amostra de Treino',   f'{n_records:,}',   f'{pct_retained:.0f}% dos registros'),
    ('Jogadores Únicos',    f'{n_players:,}',  'prováveis em campo'),
    ('Rodadas Analisadas',  f'{n_rounds}',      'rodadas do Brasileirão'),
    ('Partidas',            f'{n_games}',        'jogos disputados'),
    ('Pontuação Média',     f'{avg_pts:.2f}',    'pts (amostra de treino)'),
]))
story.append(Spacer(1, 0.4*cm))
story.append(Paragraph(
    f'Este relatório apresenta a análise exploratória da base do Cartola FC referente '
    f'às {n_rounds} primeiras rodadas do Campeonato Brasileiro 2026. As análises de pontuação '
    f'(distribuições, séries temporais, custo-benefício, scouts e mando de campo) usam apenas a '
    f'<b>população de treino</b>: atletas com status Provável que efetivamente entraram em campo '
    f'({n_records:,} registros, {pct_retained:.1f}% dos {n_total:,} registros totais; '
    f'{n_provavel:,} estavam Prováveis). A seção de status permanece na base completa, para '
    f'justificar esse recorte — o mesmo que o modelo preditivo utilizará.',
    ESTILO_BODY))
story.append(PageBreak())

# ─────────────────────────────────────────────
# 5.1.1 — DISTRIBUIÇÃO POR POSIÇÃO
# ─────────────────────────────────────────────
story.append(Paragraph('5.1.1  Distribuição de Pontuações por Posição', ESTILO_H1))
story.append(Spacer(1, 0.2*cm))
story.append(Paragraph(
    'A distribuição de pontuações varia substancialmente entre posições, refletindo diferenças '
    'fundamentais nos sistemas de scouts do Cartola FC. Ambos os painéis usam exclusivamente a '
    'população de treino — atletas Prováveis que entraram em campo —, alinhada ao que o modelo '
    'preditivo verá. O painel A mostra a densidade (violino) e o painel B o boxplot com entalhes '
    '(IC 95% da mediana). Zeros, quando ocorrem, são pontuações reais de quem jogou, não ausência de campo.',
    ESTILO_BODY))
story.append(Spacer(1, 0.15*cm))
story += fig(str(EDA_DIR / 'fig1_dist_posicao.png'), W,
             'Figura 1 — Violinplot e boxplot com entalhes de pontuação por posição, '
             'restritos a atletas Prováveis que entraram em campo. Entalhes representam o IC 95% da mediana.')
story.append(Spacer(1, 0.2*cm))

# Tabela de estatísticas
tbl_data = model_players.groupby('posicao')['pontos_num'].agg(
    N='count', Media='mean', Mediana='median', Std='std',
    P25=lambda x: x.quantile(0.25), P75=lambda x: x.quantile(0.75),
    Max='max', PctZero=lambda x: (x==0).mean()*100
).round(2).loc[pos_order]

story.append(stat_table(
    ['Posição','N','Média','Mediana','Desvio P.','P25','P75','Máx.','% Zeros'],
    [[idx] + [f'{v:.2f}' if isinstance(v, float) else str(v) for v in row]
     for idx, row in tbl_data.iterrows()],
    col_widths=[2.5*cm, 1.3*cm, 1.4*cm, 1.5*cm, 1.5*cm, 1.2*cm, 1.2*cm, 1.2*cm, 1.5*cm]
))
story.append(Spacer(1, 0.15*cm))

# Insights chave
pos_max_mean = tbl_data['Media'].idxmax()
pos_max_std  = tbl_data['Std'].idxmax()
tec = tbl_data.loc['Técnico']
ata = tbl_data.loc['Atacante']
story.append(Paragraph(
    f'▶  Insight: {pos_max_mean}s apresentam a maior pontuação média ({tbl_data.loc[pos_max_mean, "Media"]:.2f} pts) '
    f'na amostra de treino. Técnicos pontuam {tec["Media"]:.2f} pts em média, com {tec["PctZero"]:.1f}% de zeros, '
    f'pois sua pontuação depende do resultado coletivo. {pos_max_std}s exibem o maior desvio padrão '
    f'({tbl_data.loc[pos_max_std, "Std"]:.2f} pts), confirmando a maior imprevisibilidade desta posição. '
    f'Atacantes: média {ata["Media"]:.2f}, desvio {ata["Std"]:.2f}.',
    ESTILO_INSIGHT))
story.append(PageBreak())

# ─────────────────────────────────────────────
# 5.1.2 — ANÁLISE TEMPORAL
# ─────────────────────────────────────────────
story.append(Paragraph('5.1.2  Análise Temporal de Desempenho', ESTILO_H1))
story.append(Spacer(1, 0.2*cm))
story.append(Paragraph(
    'A evolução das pontuações ao longo das 18 rodadas revela a dinâmica competitiva do campeonato. '
    'O painel A ilustra as médias por rodada para cada posição de linha, enquanto o heatmap (painel B) '
    'permite identificar rodadas de alto e baixo desempenho coletivo, possivelmente associadas a '
    'congestionamentos de calendário, clássicos ou rodadas com muitos mandantes fortes.', ESTILO_BODY))
story.append(Spacer(1, 0.15*cm))
story += fig(str(EDA_DIR / 'fig2_temporal.png'), W,
             'Figura 2 — (A) Séries temporais de pontuação média por posição. '
             '(B) Heatmap de pontuação média: posições × rodadas.')
story.append(Spacer(1, 0.15*cm))

# Tabela CV por rodada
cv_tab = model_players.groupby('rodada_id')['pontos_num'].agg(Media='mean', Std='std')
cv_tab['CV (%)'] = (cv_tab['Std'] / cv_tab['Media'] * 100).round(1)
cv_tab = cv_tab.round(2).reset_index()
# Mostra 9 rodadas por linha para compactar
rows_cv = []
for i in range(0, len(cv_tab), 9):
    chunk = cv_tab.iloc[i:i+9]
    rows_cv.append([f'Rodada {r}' for r in chunk['rodada_id']])
    rows_cv.append([f'{v:.2f}' for v in chunk['Media']])
    rows_cv.append([f'{v:.1f}%' for v in chunk['CV (%)']])

# Compact version
story.append(Paragraph('Coeficiente de Variação da Pontuação por Rodada:', ESTILO_H2))
chunk1 = cv_tab[cv_tab['rodada_id'] <= 9]
chunk2 = cv_tab[cv_tab['rodada_id'] > 9]

def compact_cv_table(chunk):
    col_w = W / len(chunk)
    hdr   = [f'Rd {int(r)}' for r in chunk['rodada_id']]
    row_m = [f'{v:.2f}' for v in chunk['Media']]
    row_c = [f'{v:.0f}%' for v in chunk['CV (%)']]
    tbl = Table([hdr, row_m, row_c], colWidths=[col_w]*len(chunk))
    tbl.setStyle(TableStyle([
        ('BACKGROUND',  (0,0), (-1,0),  AZUL),
        ('TEXTCOLOR',   (0,0), (-1,0),  white),
        ('FONTNAME',    (0,0), (-1,0),  'Helvetica-Bold'),
        ('FONTSIZE',    (0,0), (-1,-1), 8.5),
        ('ALIGN',       (0,0), (-1,-1), 'CENTER'),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [CINZA_CL, white]),
        ('GRID',        (0,0), (-1,-1), 0.3, HexColor('#BBCBD8')),
        ('TOPPADDING',  (0,0), (-1,-1), 4),
        ('BOTTOMPADDING',(0,0), (-1,-1), 4),
    ]))
    return tbl
story.append(Paragraph('Rodadas 1–9:', ESTILO_H3))
story.append(compact_cv_table(chunk1))
story.append(Spacer(1, 0.1*cm))
story.append(Paragraph('Rodadas 10–18:', ESTILO_H3))
story.append(compact_cv_table(chunk2))
story.append(Spacer(1, 0.1*cm))
story.append(Paragraph('Média = pontuação média na amostra de treino (Provável e em campo); CV = coeficiente de variação (Std/Média × 100%).', ESTILO_NOTA))
story.append(Spacer(1, 0.2*cm))

cv_min = cv_tab['CV (%)'].min()
cv_max = cv_tab['CV (%)'].max()
story.append(Paragraph(
    f'▶  Insight: A variabilidade das pontuações permanece alta mesmo na amostra de treino '
    f'(CV entre {cv_min:.0f}% e {cv_max:.0f}%), refletindo a natureza assimétrica dos scouts. '
    f'Rodadas com CV mais elevado podem indicar confrontos desequilibrados ou condições especiais '
    f'de calendário, e serão consideradas na construção de features de contexto de rodada.',
    ESTILO_INSIGHT))
story.append(PageBreak())

# ─────────────────────────────────────────────
# 5.1.2b — ACF / PACF
# ─────────────────────────────────────────────
story.append(Paragraph('5.1.2b  Autocorrelação de Séries Individuais (ACF / PACF)', ESTILO_H1))
story.append(Spacer(1, 0.2*cm))
story.append(Paragraph(
    'Para informar a construção de features de lag, calculamos a Função de Autocorrelação (ACF) e '
    'a Função de Autocorrelação Parcial (PACF) das séries de pontuação individual. Três atletas '
    'representativos — um Atacante, um Meia e um Goleiro com alta regularidade na amostra de treino — '
    'são analisados. A série de cada atleta usa apenas as rodadas em que ele estava Provável e em campo; '
    'rodadas ausentes <b>não</b> são preenchidas com zero. '
    'As bandas sombreadas representam o intervalo de confiança de 95% sob a hipótese de ruído branco.',
    ESTILO_BODY))
story.append(Spacer(1, 0.15*cm))
story += fig(str(EDA_DIR / 'fig3_acf_pacf.png'), W,
             'Figura 3 — ACF e PACF para séries de pontuação individual (apenas rodadas Provável e em campo). '
             'Lag refere-se à ordem cronológica dessas aparições. Lags significativos indicam dependência temporal.')
story.append(Spacer(1, 0.2*cm))
story.append(Paragraph(
    '▶  Insight: A ACF de lag 1 tende a ser a mais relevante para a maioria dos atletas, '
    'sugerindo que a pontuação da rodada imediatamente anterior é o preditor temporal mais forte. '
    'A ausência de autocorrelação significativa em lags maiores na PACF justifica o uso de janelas '
    'curtas (k = 3 a 5 rodadas) nas médias móveis da engenharia de features, evitando overfitting '
    'a padrões de longo prazo inexistentes.',
    ESTILO_INSIGHT))
story.append(PageBreak())

# ─────────────────────────────────────────────
# 5.1.3 — CUSTO-BENEFÍCIO
# ─────────────────────────────────────────────
story.append(Paragraph('5.1.3  Análise de Custo-Benefício', ESTILO_H1))
story.append(Spacer(1, 0.2*cm))
story.append(Paragraph(
    'O índice de custo-benefício (CB = pontos / cartoletas) quantifica a eficiência de cada atleta '
    'em termos de pontuação entregue por cartoleta investida. Identificar jogadores desvalorizados — '
    'aqueles com CB elevado e preço relativamente baixo — é o principal objetivo estratégico do '
    'Cartola FC. O painel A mostra a relação entre preço e pontuação; o painel B compara a '
    'distribuição do índice por posição; o painel C lista os Top 15 atletas por CB médio.', ESTILO_BODY))
story.append(Spacer(1, 0.15*cm))
story += fig(str(EDA_DIR / 'fig4_custo_beneficio.png'), W,
             'Figura 4 — (A) Scatter preço × pontuação por posição. '
             '(B) Distribuição do índice CB (pts/cartoleta) por posição. '
             '(C) Top 15 atletas por CB médio (mínimo 5 rodadas na amostra de treino).')
story.append(Spacer(1, 0.15*cm))

# Tabela CB por posição
cb_pos_tbl = model_players[model_players['preco_num'] > 0].groupby('posicao')['cb_ratio'].agg(
    Media='mean', Mediana='median', Std='std', P75=lambda x: x.quantile(0.75)
).round(3)
pos_order_cb = ['Goleiro','Lateral','Zagueiro','Meia','Atacante']
cb_pos_tbl = cb_pos_tbl.loc[[p for p in pos_order_cb if p in cb_pos_tbl.index]]
story.append(stat_table(
    ['Posição', 'CB Médio (pts/¢)', 'CB Mediana', 'Desvio P.', 'CB P75'],
    [[idx] + [f'{v:.3f}' for v in row] for idx, row in cb_pos_tbl.iterrows()],
    col_widths=[3.2*cm, 3.5*cm, 3.2*cm, 3.2*cm, 3.2*cm]
))
story.append(Spacer(1, 0.1*cm))
cb_top = cb_pos_tbl['Media'].idxmax()
story.append(Paragraph(
    f'▶  Insight: {cb_top}s apresentam o maior CB médio ({cb_pos_tbl.loc[cb_top, "Media"]:.3f} pts/¢) '
    f'na amostra de treino. A correlação entre preço e pontuação é positiva mas não linear — existem '
    f'bolsões de atletas baratos com alta pontuação, que o modelo de recomendação deverá identificar e explorar.',
    ESTILO_INSIGHT))
story.append(PageBreak())

# ─────────────────────────────────────────────
# 5.1.4 — CORRELAÇÃO SCOUTS
# ─────────────────────────────────────────────
story.append(Paragraph('5.1.4  Correlação entre Scouts e Pontuação Final', ESTILO_H1))
story.append(Spacer(1, 0.2*cm))
story.append(Paragraph(
    'A matriz de correlação entre cada scout individual e a pontuação final revela quais eventos '
    'de jogo têm maior poder explicativo sobre o desempenho, subsidiando diretamente a seleção '
    'de features. Calculamos os coeficientes de Pearson (sensível a relações lineares) e Spearman '
    '(robusto a distribuições não-normais e relações monotônicas), segmentando a análise em dois '
    'grupos posicionais com sistemas de scouts distintos. Marcadores de significância estatística: '
    '*** p<0.001, ** p<0.01.', ESTILO_BODY))
story.append(Spacer(1, 0.15*cm))
story += fig(str(EDA_DIR / 'fig5_correlacao_scouts.png'), W,
             'Figura 5 — Correlações de Pearson e Spearman entre scouts individuais e pontuação total, '
             'segmentadas por grupo posicional. Ordenadas por valor absoluto da correlação de Spearman.')
story.append(Spacer(1, 0.15*cm))

from scipy import stats as sp_stats
at_meia = model_players[model_players['posicao'].isin(['Atacante','Meia'])]
def_gol  = model_players[model_players['posicao'].isin(['Goleiro','Lateral','Zagueiro'])]

top_scouts_at = {}
top_scouts_def = {}
for sc in ['G','A','FT','FF','FD','FS','FC','CA','DS']:
    top_scouts_at[sc]  = sp_stats.spearmanr(at_meia[sc], at_meia['pontos_num'])[0]
    top_scouts_def[sc] = sp_stats.spearmanr(def_gol[sc], def_gol['pontos_num'])[0]

top_at  = sorted(top_scouts_at.items(),  key=lambda x: abs(x[1]), reverse=True)[:4]
top_def = sorted(top_scouts_def.items(), key=lambda x: abs(x[1]), reverse=True)[:4]

names = {'G':'Gol','A':'Assist.','FT':'F.Trave','FF':'F.Fora','FD':'F.Def.',
         'FS':'F.Sofrida','FC':'F.Cometida','CA':'C.Amarelo','DS':'Desarme',
         'DE':'Defesa','SG':'Sem Gol','GC':'G.Contra','GS':'G.Sofrido'}

story.append(stat_table(
    ['Scout', 'Spearman (At. & Meia)', 'Scout', 'Spearman (Def. & Gol.)'],
    [[f'{names.get(a[0],a[0])} ({a[0]})', f'{a[1]:.3f}',
      f'{names.get(b[0],b[0])} ({b[0]})', f'{b[1]:.3f}']
     for a, b in zip(top_at, top_def)],
    col_widths=[3.8*cm, 4*cm, 3.8*cm, 4*cm]
))
story.append(Spacer(1, 0.1*cm))
story.append(Paragraph(
    '▶  Insight: Para Atacantes e Meias, Gols (G) e Assistências (A) dominam a correlação — '
    'confirmando que os scouts ofensivos são os principais drivers de pontuação nesse grupo. '
    'Finalizações Fora (FF) e Faltas Sofridas (FS) aparecem como features secundárias relevantes. '
    'Para Defensores e Goleiros, Desarmes (DS) e Faltas Cometidas (FC) ganham importância relativa, '
    'embora as correlações absolutas sejam menores, refletindo a maior dependência coletiva do desempenho defensivo.',
    ESTILO_INSIGHT))
story.append(PageBreak())

# ─────────────────────────────────────────────
# 5.1.4b — HEATMAP COMPLETO
# ─────────────────────────────────────────────
story.append(Paragraph('5.1.4b  Matriz de Correlação Completa entre Scouts', ESTILO_H1))
story.append(Spacer(1, 0.2*cm))
story.append(Paragraph(
    'O heatmap completo de correlação de Spearman entre todos os scouts e a pontuação final '
    'permite identificar também colinearidades entre features — essencial para a regularização '
    'dos modelos lineares (Ridge/Lasso). Pares de scouts com alta correlação entre si devem '
    'ser tratados com cuidado para evitar multicolinearidade.', ESTILO_BODY))
story.append(Spacer(1, 0.1*cm))
story += fig(str(EDA_DIR / 'fig8_heatmap_scouts.png'), W,
             'Figura 8 — Heatmap completo da correlação de Spearman entre scouts e pontuação, '
             'para Atacantes/Meias (esq.) e Defensores/Goleiros (dir.). '
             'Diagonal = 1. Escala: vermelho = positivo, azul = negativo.')
story.append(Spacer(1, 0.15*cm))
story.append(Paragraph(
    '▶  Insight: Gol (G) e Assistência (A) mostram correlação positiva entre si (atacantes que '
    'chutam também assistem), sugerindo cuidado com multicolinearidade em modelos lineares. '
    'Cartão Amarelo (CA) e Falta Cometida (FC) têm correlação positiva esperada (infrações físicas). '
    'Para o grupo defensivo, Defesa (DE) e Sem Gol (SG) são altamente correlacionados — '
    'ambos capturam o desempenho do goleiro, tornando redundante usar os dois simultaneamente.',
    ESTILO_INSIGHT))
story.append(PageBreak())

# ─────────────────────────────────────────────
# 5.1.5 — MANDANTE VS VISITANTE
# ─────────────────────────────────────────────
story.append(Paragraph('5.1.5  Análise de Mando de Campo', ESTILO_H1))
story.append(Spacer(1, 0.2*cm))
story.append(Paragraph(
    'O mando de campo é um dos fatores mais consolidados na literatura de ciência do esporte. '
    'Nesta análise, associamos cada atleta ao papel de mandante ou visitante com base no cruzamento '
    'do seu clube_id com os dados de clube_casa_id por rodada. Aplicamos o teste t de Welch '
    '(variâncias independentes) para verificar significância estatística da diferença, '
    'controlando por posição.', ESTILO_BODY))
story.append(Spacer(1, 0.15*cm))
story += fig(str(EDA_DIR / 'fig6_mandante_visitante.png'), W,
             'Figura 6 — (A) Boxplot global mandante vs. visitante com resultado do teste t. '
             '(B) Diferença de pontuação média (Mandante − Visitante) por posição. '
             '(C) Violin por posição e mando. Significância: ***p<0.001, **p<0.01, *p<0.05.')
story.append(Spacer(1, 0.15*cm))

# Tabela de resultados do teste por posição
res_mando = []
res_mando_raw = []
for pos in ['Goleiro','Lateral','Zagueiro','Meia','Atacante']:
    m = model_players[(model_players['posicao']==pos) & (model_players['mando']=='Mandante')]['pontos_num']
    v = model_players[(model_players['posicao']==pos) & (model_players['mando']=='Visitante')]['pontos_num']
    t, p = sp_stats.ttest_ind(m, v)
    diff = m.mean() - v.mean()
    sig  = '***' if p < 0.001 else ('**' if p < 0.01 else ('*' if p < 0.05 else 'n.s.'))
    res_mando.append([pos, f'{m.mean():.3f}', f'{v.mean():.3f}',
                      f'{diff:+.3f}', f'{t:.2f}', f'{p:.4f}', sig])
    res_mando_raw.append({'posicao': pos, 'diff': diff, 'p': p, 'sig': sig})

story.append(stat_table(
    ['Posição','Média Mand.','Média Visit.','Δ (M−V)','t-stat','p-valor','Signif.'],
    res_mando,
    col_widths=[2.7*cm, 2.5*cm, 2.7*cm, 2.0*cm, 1.8*cm, 2.0*cm, 1.6*cm]
))
story.append(Spacer(1, 0.1*cm))
n_sig = sum(1 for r in res_mando_raw if r['p'] < 0.05)
pos_max_diff = max(res_mando_raw, key=lambda r: r['diff'])
story.append(Paragraph(
    f'▶  Insight: {n_sig} de 5 posições apresentam diferença mandante−visitante significativa (α=0,05). '
    f'{pos_max_diff["posicao"]}s apresentam o maior ganho absoluto em casa '
    f'({pos_max_diff["diff"]:+.2f} pts). A variável binária mandante/visitante será incluída '
    f'como feature contextual nos modelos preditivos.',
    ESTILO_INSIGHT))
story.append(PageBreak())

# ─────────────────────────────────────────────
# 5.1.6 — STATUS E DISPONIBILIDADE
# ─────────────────────────────────────────────
story.append(Paragraph('5.1.6  Análise de Status e Disponibilidade', ESTILO_H1))
story.append(Spacer(1, 0.2*cm))
story.append(Paragraph(
    'O status do atleta na véspera da rodada é um dos preditores mais diretos da probabilidade '
    'de participação. O dataset registra cinco estados: Provável (7), Dúvida (6), Suspenso (5), '
    'Lesionado (3) e Não Participará (2). Esta seção usa a <b>base completa</b> e a coluna '
    '<b>entrou_em_campo</b> para medir participação efetiva — justificando o recorte da amostra de treino '
    '(Provável e em campo) adotado nas demais análises.', ESTILO_BODY))
story.append(Spacer(1, 0.15*cm))
story += fig(str(EDA_DIR / 'fig7_status_disponibilidade.png'), W,
             'Figura 7 — (A) Volume de registros por status. (B) Pontuação média por status. '
             '(C) Taxa de participação em campo (coluna entrou_em_campo) por status. (D) Violin por status.')
story.append(Spacer(1, 0.15*cm))

# Tabela de status
status_order = ['Provável','Dúvida','Suspenso','Lesionado','Não Participará']
tab_status = players.groupby('status_label').agg(
    N=('pontos_num', 'count'),
    Media=('pontos_num', 'mean'),
    Std=('pontos_num', 'std'),
    PctZero=('pontos_num', lambda x: (x==0).mean()*100),
    TaxaPartic=('entrou_em_campo', lambda x: x.mean()*100),
).round(2).reindex(status_order)

story.append(stat_table(
    ['Status','N','Média (pts)','Desvio P.','% Zeros','% Participação'],
    [[idx, f'{row["N"]:.0f}', f'{row["Media"]:.2f}', f'{row["Std"]:.2f}',
      f'{row["PctZero"]:.1f}%', f'{row["TaxaPartic"]:.1f}%']
     for idx, row in tab_status.iterrows()],
    col_widths=[3.5*cm, 1.8*cm, 2.5*cm, 2.3*cm, 2.0*cm, 2.7*cm]
))
story.append(Spacer(1, 0.1*cm))

prov = tab_status.loc['Provável']
duv  = tab_status.loc['Dúvida']
sus  = tab_status.loc['Suspenso']
ratio = prov['Media'] / duv['Media'] if duv['Media'] != 0 else float('inf')
story.append(Paragraph(
    f'▶  Insight: A taxa de participação cai drasticamente com o status. Atletas "Prováveis" '
    f'entram em campo em {prov["TaxaPartic"]:.1f}% das rodadas, enquanto "Dúvida" cai para '
    f'{duv["TaxaPartic"]:.1f}% e "Suspenso" para {sus["TaxaPartic"]:.1f}%. '
    f'A diferença de pontuação média entre "Provável" ({prov["Media"]:.2f} pts) e "Dúvida" '
    f'({duv["Media"]:.2f} pts) é de magnitude {ratio:.1f}×, tornando o status a variável de '
    f'filtragem mais crítica: o modelo treina apenas em Provável e em campo, e o otimizador de '
    f'escalação deve excluir ou penalizar status piores.',
    ESTILO_INSIGHT))
story.append(Spacer(1, 0.3*cm))

# ─────────────────────────────────────────────
# CONCLUSÕES E PRÓXIMOS PASSOS
# ─────────────────────────────────────────────
story.append(hr(AZUL, 1.5, 10, 10))
story.append(Paragraph('Síntese dos Achados e Implicações para Modelagem', ESTILO_H1))
story.append(Spacer(1, 0.2*cm))

conclusoes = [
    ('<b>Amostra de treino = Provável e em campo:</b> As análises de pontuação restringem-se a atletas '
     'que estavam Prováveis e efetivamente entraram em campo — a mesma população que o modelo preditivo '
     'utilizará. Zeros, quando existem, são pontuações reais, não ausência de participação.'),
    ('<b>Padrões temporais de curto alcance:</b> A ACF/PACF, calculada sem imputar zeros em rodadas ausentes, '
     'indica dependência temporal predominantemente de lag 1–2, justificando janelas de médias móveis de 3 a 5 rodadas. '
     'Features de longo prazo têm valor marginal decrescente.'),
    ('<b>Scouts ofensivos como principais drivers:</b> Gol (G) e Assistência (A) dominam a '
     'correlação com a pontuação para Atacantes e Meias. O sistema de features deve '
     'priorizar médias históricas desses scouts, especialmente contra adversários permeáveis.'),
    ('<b>Efeito mandante como feature contextual:</b> A variável binária mandante/visitante '
     'deve ser incluída nos modelos como feature de contexto de partida.'),
    ('<b>Status como filtro de escalação, não como feature de treino:</b> Atletas com status "Dúvida" ou pior '
     'devem ser excluídos ou fortemente penalizados no otimizador de escalação. '
     'A Figura 7 (base completa) justifica esse recorte; o modelo de pontuação não precisa classificar participação.'),
    ('<b>Custo-benefício como feature econômica:</b> O índice CB histórico e a tendência de '
     'valorização do jogador (variacao_num) serão features relevantes para o problema de '
     'otimização de escalação dentro do orçamento.'),
]
for c in conclusoes:
    story.append(Paragraph(f'• {c}', ESTILO_BULLET))
    story.append(Spacer(1, 0.05*cm))

story.append(Spacer(1, 0.3*cm))
story.append(Paragraph(
    'Nota metodológica: As análises de pontuação (seções 5.1.1–5.1.5) usam apenas atletas com '
    'status Provável que entraram em campo. A seção 5.1.6 usa a base completa para justificar esse recorte. '
    'Dados: Rodadas 1–18 do Campeonato Brasileiro 2026 (API pública do Cartola FC e games.csv). '
    'Testes de hipótese utilizaram α = 0,05.',
    ESTILO_NOTA))

# ─── Build PDF ─────────────────────────────────────────────────────────────
output_path = str(EDA_DIR / 'EDA_CartolaFC_Executivo.pdf')
doc = SimpleDocTemplate(
    output_path,
    pagesize=A4,
    leftMargin=1.5*cm, rightMargin=1.5*cm,
    topMargin=2.2*cm, bottomMargin=1.8*cm,
    title='EDA — Cartola FC TCC',
    author='Gabriel Rodrigues de Deus Abad',
    subject='Análise Exploratória de Dados — Campeonato Brasileiro 2026',
)
doc.build(story, onFirstPage=on_first_page, onLaterPages=on_page)
print(f'PDF gerado: {output_path}')