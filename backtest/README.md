# backtest/ — laboratório de estratégias da Ovelha

Pasta independente do app (`app.py`, `telegram_bot.py` não são usados nem alterados aqui).
Objetivo: testar estratégias de forma **honesta** (sem olhar o futuro, com custos, sem repaint)
antes de levá-las para o site de acompanhamento em tempo real.

O histórico de estudos, resultados e conclusões está em [`ESTUDOS.md`](ESTUDOS.md).
Leia esse arquivo primeiro para continuar o trabalho.

## Como rodar

```bash
pip install pandas numpy requests matplotlib
cd backtest

# 1. Baixar dados (arquivos oficiais da Binance com checksum + API pública no mês corrente)
python -c "import dados_binance as d; d.carregar('BTCUSDT','1h').to_pickle('cache/BTCUSDT_1h.pkl')"

# 2. Conferir integridade
python -c "import pandas as pd, verificar_dados as v; v.verificar(pd.read_pickle('cache/BTCUSDT_1h.pkl'), '1h')"

# 3. Estudos
python rodar_grade.py BTCUSDT 1h                 # grade 3 médias × 4 k
python teste_aleatorio.py BTCUSDT 1h             # sinal × trades aleatórios
python multi_ativos.py 1h BTCUSDT ETHUSDT SOLUSDT ADAUSDT ETHBTC
python experimento_saidas.py 1h BTCUSDT ETHUSDT SOLUSDT ADAUSDT ETHBTC
python gerar_visuais.py BTCUSDT 1h               # PNGs em saida/

# 4. Visualizador de trades
python exportar_visualizador.py 1h BTCUSDT ETHUSDT SOLUSDT ADAUSDT ETHBTC
# serve a pasta com index.html + dados/<ATIVO>_1h.json:
mkdir -p saida/site/dados && cp visualizador/index.html saida/site/ && cp saida/visualizador/*.json saida/site/dados/
python -m http.server -d saida/site 8000         # abrir http://localhost:8000

# 5. Visualizador das operações de FVG (M1, últimos meses) — na sua máquina:
#    precisa dos dados M1 em cache/ (ver passo 1, com '1m' e desde=(2026, 3)), depois:
python exportar_fvg_viz.py 2026-04-01 BTCUSDT ETHUSDT      # gera saida/visualizador_fvg/dados/
cp visualizador_fvg/index.html saida/visualizador_fvg/ && python -m http.server -d saida/visualizador_fvg 8001
```

Indicador do TradingView: `tradingview/ovelha_fvg.pine` (colar no editor Pine). A lógica é conferida contra o
backtest com `python tradingview/conferir_pine.py BTCUSDT 100000`.

Acesso de rede necessário: `data.binance.vision` e `data-api.binance.vision` (sem chave de API).
`api.binance.com` costuma ser bloqueado para servidores nos EUA e não é usado.

`cache/` (dados baixados) e `saida/` (gerados) ficam fora do git; `resultados/` guarda as
saídas dos estudos já feitos.

## Arquitetura

```
dados_binance.py      coleta: ZIP mensal oficial (SHA-256 conferido) + API para o mês corrente.
                      Normaliza timestamps (a Binance usa microssegundos de 2025 em diante)
                      e descarta o candle em formação.
verificar_dados.py    integridade: ordem, duplicados, alinhamento, buracos, OHLC coerente,
                      saltos, comparação entre fontes e agregação H1 -> D1.

estrategia.py         Ovelha V3: média móvel + faixa de ATR com histerese -> coluna 'estado'.
motor.py              execução honesta e métricas. Modos de saída: 'virada', 'stop_movel', 'alvo'.
                      Mede R, MFE/MAE e acerto de 1R/2R/3R antes do stop inicial.

rodar_grade.py        grade de parâmetros (constantes MEDIAS, KS, INICIO, CUSTO_LADO).
teste_aleatorio.py    compara trades do sinal com trades aleatórios de mesma direção e duração.
multi_ativos.py       mesma grade em vários ativos + teste contra o aleatório.
experimento_saidas.py assertividade por tipo de stop e resultado por modo de saída.
gerar_visuais.py      mapa de calor, curva de capital, trades em detalhe (PNG).
exportar_visualizador.py + visualizador/index.html   app de navegação trade a trade
                      (Lightweight Charts 4.2.0; um JSON por ativo, carregado sob demanda).
```

## Regras de honestidade (valem para qualquer estratégia nova)

1. O sinal é decidido no **fechamento** do candle e executado na **abertura do seguinte**.
2. Indicadores só usam dados até o candle atual. Teste obrigatório: calcular os sinais com os
   dados cortados no meio e conferir que o passado não muda (sem repaint).
3. Stops e alvos valem a partir do candle de entrada. Gap além do stop sai na abertura.
   Stop e alvo no mesmo candle contam como **stop**.
4. Custo por lado em toda entrada e saída. Padrão atual: 0,05% (futuros Binance, taker).
5. Pontos de swing (topos e fundos) só podem ser usados **depois de confirmados** — um topo de
   N candles só existe N candles depois. Isso é crítico para estratégias de estrutura/SMC.
6. Toda métrica de acerto vem acompanhada de uma referência aleatória com o mesmo risco,
   direção e janela de tempo.
7. Os números são validados por três caminhos (curva de capital, produto dos trades e cálculo
   independente) antes de qualquer conclusão.
