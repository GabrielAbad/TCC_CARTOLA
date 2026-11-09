# Documentação da estrutura do repositório

Este documento descreve, de forma breve, a finalidade das pastas e dos arquivos do projeto **TCC Cartola FC**. Arquivos repetitivos de dados, cache, ambiente virtual e bytecode são documentados por padrão de nome, pois possuem a mesma função.

## Visão geral

O projeto coleta dados do Cartola FC, realiza análise exploratória e engenharia de atributos, treina e avalia modelos preditivos em esquema walk-forward e simula a montagem de escalações sob restrições de posição, clube e orçamento.

## Raiz do projeto

- `.git/`: metadados internos do Git, incluindo histórico, referências e configuração local do repositório. Não contém código da aplicação.
- `.gitignore`: define arquivos que o Git deve ignorar, como o ambiente virtual, bytecode Python e bases locais em CSV/JSON.
- `.pytest_cache/`: cache criado pelo pytest para acelerar e registrar execuções de testes.
  - `.gitignore`: impede o versionamento do conteúdo do cache.
  - `CACHEDIR.TAG`: identifica a pasta como cache descartável.
  - `README.md`: explicação padrão sobre o cache do pytest.
  - `v/cache/lastfailed`: registra os testes que falharam na última execução.
  - `v/cache/nodeids`: registra os testes descobertos pelo pytest.
- `pytest-cache-files-*/`: diretórios temporários alternativos do pytest; possuem a mesma finalidade de `.pytest_cache/` e podem ser removidos sem afetar o projeto.
- `venv/`: ambiente virtual Python local, com o interpretador e as dependências instaladas.
  - `Include/`: cabeçalhos usados por pacotes com extensões nativas.
  - `Lib/site-packages/`: código das bibliotecas de terceiros instaladas.
  - `Scripts/`: executáveis do Python, pip, pytest, Jupyter e demais ferramentas.
  - `pyvenv.cfg`: configuração e referência ao Python usado para criar o ambiente.
- `README.md`: apresentação resumida do repositório.
- `requirements.txt`: lista as dependências Python de coleta, análise, modelagem, otimização, notebooks e testes.
- `docs.md`: este guia da estrutura e da responsabilidade de cada parte do repositório.

## `data/` — dados do projeto

- `data/__init__.py`: identifica `data` como pacote Python.

### `data/extract/` — extração e normalização

- `data/extract/__init__.py`: identifica o módulo de extração de dados.
- `data/extract/games.py`: consulta ou lê do cache as partidas por rodada, normaliza clubes e placares e gera `games.csv`.
- `data/extract/players.py`: baixa as estatísticas dos atletas por rodada, padroniza tipos e scouts e gera `players.csv`.
- `data/extract/__pycache__/`: bytecode gerado automaticamente ao importar os módulos de extração.

### `data/database/` — bases locais

- `data/database/.gitkeep`: mantém a pasta no Git mesmo quando não há bases versionadas.
- `data/database/<ano>/`: cache de dados brutos separado por temporada.
  - `rodada-<n>.csv`: estatísticas dos atletas na rodada indicada.
  - `partidas-<n>.json`: resposta bruta da API de partidas na rodada indicada.
- `data/database/players.csv`: base consolidada de atletas, preços, status, pontuações e scouts.
- `data/database/games.csv`: base consolidada de partidas, mandantes, visitantes e placares.
- `data/database/df_features.csv`: base modelável produzida pela engenharia de atributos.
- `data/database/backup_r18/`: cópia de segurança das bases consolidadas até a rodada 18.
  - `players.csv`: backup da base de atletas.
  - `games.csv`: backup da base de partidas.
  - `df_features.csv`: backup da base de atributos.

### `data/eda/` — resultados da análise exploratória

- `data/eda/anotacoes.txt`: anotações textuais de apoio à análise.
- `data/eda/eda_stats.pkl`: estatísticas da EDA serializadas para reutilização.
- `data/eda/EDA_CartolaFC_Executivo.pdf`: relatório executivo consolidado da análise exploratória.
- `data/eda/fig1_dist_posicao.png`: distribuições de pontuação por posição.
- `data/eda/fig2_temporal.png`: evolução temporal das pontuações por rodada.
- `data/eda/fig3_acf_pacf.png`: autocorrelação e autocorrelação parcial das séries de atletas.
- `data/eda/fig4_custo_beneficio.png`: relação entre preço, pontuação e custo-benefício.
- `data/eda/fig5_correlacao_scouts.png`: correlação dos scouts com a pontuação.
- `data/eda/fig6_mandante_visitante.png`: comparação entre atletas mandantes e visitantes.
- `data/eda/fig7_status_disponibilidade.png`: análise de status e participação efetiva.
- `data/eda/fig8_heatmap_scouts.png`: matriz de correlação entre scouts.

#### `data/eda/features/` — diagnósticos dos atributos

- `fig1_nan_cobertura.png`: cobertura dos atributos e proporção de valores ausentes.
- `fig2_spearman_target.png`: correlação de Spearman entre atributos e variável-alvo.
- `fig3_multicolinearidade.png`: diagnóstico de correlação e redundância entre atributos.
- `fig4_h2h_cobertura.png`: cobertura dos atributos de confronto direto.
- `fig5_pagerank_elo.png`: análise dos indicadores PageRank e Elo.
- `fig6_mutual_information.png`: informação mútua dos atributos em relação ao alvo.
- `fig7_features_por_posicao.png`: relevância dos atributos segmentada por posição.
- `fig8_status_residual.png`: relação entre status do atleta e resíduos do modelo.

## `eda/` — exploração e engenharia de atributos

- `eda/raw_eda.py`: gera estatísticas e as oito figuras principais da análise exploratória.
- `eda/pdf_gen.py`: monta o relatório executivo em PDF a partir das bases e figuras da EDA.
- `eda/feature_engineering.ipynb`: notebook que constrói os atributos históricos, contextuais e de desempenho usados pelos modelos.
- `eda/feature_analysis.ipynb`: notebook que avalia cobertura, correlação, multicolinearidade e relevância dos atributos.

## `models/` — modelagem, otimização e simulação

- `models/__init__.py`: identifica `models` como pacote Python.
- `models/budget.py`: representa o estado financeiro do usuário e calcula compra, saldo, patrimônio, valorização e trocas.
- `models/candidates.py`: combina previsões walk-forward com os atributos necessários ao otimizador.
- `models/evaluation.py`: calcula métricas agregadas do sistema, erros por posição e comparação entre turnos.
- `models/lineup.py`: resolve a escalação por programação linear inteira e define o capitão.
- `models/notebook_runner.py`: executa sequencialmente as células de código de um notebook sem depender do servidor Jupyter.
- `models/report.py`: gera gráficos de patrimônio, valorização, pontuação e eficiência do sistema.
- `models/run_config.py`: centraliza intervalo de rodadas, sufixos de saída e configuração por variáveis de ambiente.
- `models/run_train_eval.py`: interface de linha de comando para executar o treinamento e a avaliação walk-forward.
- `models/run_optimize_eval.py`: interface de linha de comando para executar a otimização e a avaliação das escalações.
- `models/simulation.py`: simula rodada a rodada a seleção do time, a pontuação, o orçamento e a valorização.
- `models/train_eval.ipynb`: notebook principal de treinamento, previsão e comparação de modelos.
- `models/optimize_eval.ipynb`: notebook de otimização das escalações e avaliação ponta a ponta do sistema.
- `models/__pycache__/`: bytecode Python gerado automaticamente; é descartável.

### Testes em `models/`

- `models/test_budget.py`: testa regras de orçamento, valorização, compra do time e contagem de trocas.
- `models/test_evaluation.py`: testa as métricas e tabelas de avaliação do sistema.
- `models/test_lineup.py`: testa as restrições da escalação, escolha do capitão e cálculo de pontos.
- `models/test_run_config.py`: testa validação, nomes de saída e leitura da configuração.
- `models/test_simulation.py`: testa filtragem de candidatos e o fluxo de simulação por rodada.

### `models/outputs/` — resultados dos experimentos

Arquivos com o sufixo `_segundo_turno` correspondem à avaliação das rodadas do segundo turno; os equivalentes sem esse sufixo representam a execução-base ou o primeiro turno.

- `feature_importances.csv` e `feature_importances_segundo_turno.csv`: importância dos atributos estimada pelos modelos.
- `predicoes_walkforward.csv` e `predicoes_walkforward_segundo_turno.csv`: previsões por atleta e rodada no esquema walk-forward.
- `resultados_walkforward.csv` e `resultados_walkforward_segundo_turno.csv`: métricas dos modelos em cada rodada avaliada.
- `historico_escalacoes.csv` e `historico_escalacoes_segundo_turno.csv`: atletas selecionados pelo sistema e pelo ótimo retrospectivo.
- `historico_orcamento.csv` e `historico_orcamento_segundo_turno.csv`: evolução de saldo, custo, patrimônio, pontos e eficiência.
- `tabela_6_1.csv` e `tabela_6_1_segundo_turno.csv`: resumo das métricas preditivas dos modelos.
- `tabela_6_1_por_posicao.csv` e `tabela_6_1_por_posicao_segundo_turno.csv`: métricas preditivas segmentadas por posição.
- `tabela_6_1_comparacao_turnos.csv`: comparação lado a lado das métricas dos dois turnos.
- `tabela_6_2.csv`: resumo da avaliação do sistema completo de recomendação.
- `tabela_6_2_erro_posicao.csv`: erros dos atletas escalados, agrupados por posição.
- `fig_comparacao_modelos.png` e `fig_comparacao_modelos_segundo_turno.png`: comparação visual do desempenho dos modelos.
- `fig_feature_importance.png` e `fig_feature_importance_segundo_turno.png`: visualização da importância dos atributos.
- `fig_previsto_vs_real.png` e `fig_previsto_vs_real_segundo_turno.png`: comparação entre pontuações previstas e reais.
- `fig_desempenho_segundo_turno.png`: painel de desempenho do sistema no segundo turno.
- `fig_metricas_por_rodada_segundo_turno.png`: evolução das métricas preditivas ao longo das rodadas.
- `fig_desempenho_sistema.png`: painel geral de patrimônio, valorização, pontos e eficiência.
- `log_walkforward_segundo_turno.txt`: registro textual da execução walk-forward do segundo turno.
- `log_optimize_eval.txt`: registro textual da execução da otimização e avaliação.

## `refs/` — referências acadêmicas e planejamento

- `refs/Projeto_TCC_CartolaFC (2).docx`: documento de referência do projeto de TCC.
- `refs/TCC_CartolaFC_Gabriel.tex`: fonte LaTeX do texto acadêmico do TCC.
- `refs/plano_implementacao_tcc.ipynb`: plano de implementação e organização das etapas do projeto.

## Arquivos gerados e descartáveis

Os diretórios `__pycache__/`, arquivos `*.pyc`, `.pytest_cache/`, `pytest-cache-files-*/` e o conteúdo de `venv/` são produzidos por ferramentas locais. Eles não representam código-fonte nem resultados científicos e podem ser recriados.

Os CSVs, JSONs, PKLs, PNGs, PDFs e logs descritos acima são artefatos de dados ou resultados reproduzíveis pelos scripts e notebooks correspondentes. Alguns são ignorados pelo Git devido ao tamanho ou por dependerem da temporada analisada.
