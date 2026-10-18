# Estudos — diário de pesquisa

Período testado: 01/03/2018 a 18/10/2026 (SOL desde 08/2020). Timeframe: H1. Dados: Binance spot.
Saídas completas de cada estudo estão em [`resultados/`](resultados/).

---

## 0. Ponto de partida: OVELHA V2 (no app)

`calculate_ovelha_v2_signals` em `app.py` treina um Random Forest e prevê **nos mesmos dados**
em que treinou, com um rótulo que usa o retorno 3 candles à frente. O modelo decora o histórico,
então o backtest do app é otimista e não representa a operação real. Foi por isso que este
laboratório começou do zero com regras técnicas simples.

## 1. Dados (aprovado)

- 111 arquivos mensais por ativo, todos com SHA-256 igual ao oficial.
- Arquivo × API (set/2026): idênticos. Diário montado do H1 × diário oficial: preços idênticos,
  exceto 10–11/02/2018 (candles deslocados 28 min após manutenção da Binance).
- ~0,1% de candles faltando, todos em paradas reais da corretora. Não são preenchidos.
- Decisão: começar os testes em 01/03/2018.

## 2. Estratégia Ovelha V3

```
média = SMA(fechamento, N)      ATR = média simples de 14 True Ranges
fechou acima de média + k·ATR  -> COMPRA
fechou abaixo de média − k·ATR -> VENDA
dentro da faixa                -> mantém o estado (histerese)
```

Grade: N ∈ {50, 100, 200} × k ∈ {0,5; 1,0; 1,5; 2,0}. Sempre posicionado (stop-and-reverse).
R = distância da entrada até a faixa oposta no candle do sinal.

## 3. BTC H1, custo 0,1%/lado — `resultados/grade_BTCUSDT_1h.csv`

| | k 0,5 | k 1,0 | k 1,5 | k 2,0 |
|---|---|---|---|---|
| Média 50 | −32% a.a. | −5% | −8% | −5% |
| Média 100 | −17% | −4% | −2% | +8% |
| Média 200 | +12% | +18% | +10% | +13% |

Buy & hold: +27% a.a., drawdown −77%. Estratégia: drawdown de −82% a −98%.
Médias longas são melhores. As vendas perderam em todas as combinações; as compras ganharam em 11/12.
Funcionou bem até 2021 e piorou de 2022 em diante.

## 4. Peso dos custos

Média 100, k 1,0: sem custo +552%; 0,05%/lado +113%; 0,1%/lado −30%.
Ganho bruto médio ≈ 0,17% por trade, custo de ida e volta 0,2% → 0,998^1116 ≈ 0,107.
**O custo não é alto; a vantagem por trade é que é pequena.** Padrão adotado: 0,05%/lado.

## 5. Sinal × aleatório — `resultados/teste_aleatorio_BTCUSDT_1h.txt`

Trades aleatórios com mesma direção e duração. Nas 12 combinações, compras e vendas do sinal
rendem +0,1% a +0,8% por trade a mais que o aleatório. As vendas do BTC perdem porque uma venda
aleatória no BTC perde −0,15% a −0,67% por trade (tendência de alta do ativo).

## 6. Cinco ativos, custo 0,05% — `resultados/multi_ativos_1h.txt`

| Ativo | Buy & hold a.a. | Estratégia (mediana) | Positivas | Vendas lucrativas | Sinal > aleatório |
|---|---|---|---|---|---|
| BTC | +27% | +12% | 11/12 | 0/12 | 12/12 |
| ETH | +13% | +44% | 12/12 | 5/12 | 12/12 |
| SOL | +79% | +21% | 10/12 | 0/12 | 12/12 |
| ADA | −2% | +31% | 9/12 | 5/12 | 12/12 |
| ETH/BTC | −11% | +35% | 12/12 | 12/12 | 12/12 |

Conclusão: o sinal tem vantagem real e consistente, mas modesta. O lado vendido depende do ativo
(funciona no ETH/BTC, perde no BTC e na SOL). Drawdown continua o maior problema (−60% a −87%).

## 7. Saídas e assertividade em R — `resultados/experimento_saidas_1h.txt`

Acerto = chegou a XR antes do stop inicial, dentro da vida do sinal.
Aleatório = mesma direção, mesmo risco, mesma janela.

| Stop | 1R | 2R | 3R | Aleatório 1R/2R/3R |
|---|---|---|---|---|
| Faixa | 48–50% | 30–33% | 21–23% | 38/20/12% |
| 1 ATR | 48–52% | 33–35% | 25–27% | 44–46/29–30/21–22% |
| 2 ATR | 49–51% | 33–34% | 24–26% | 42–44/24–25/16–17% |
| 3 ATR | 48–49% | 29–31% | 20–22% | 37–38/19–20/11–12% |

Retorno anual mediano por saída:

| Saída | BTC | ETH | SOL | ADA | ETH/BTC |
|---|---|---|---|---|---|
| Virada do sinal | +12% | +44% | +21% | +31% | +35% |
| Stop móvel | +8% | +36% | −8% | +22% | +28% |
| Alvo 1R | −19% | −13% | −14% | −9% | −3% |
| Alvo 2R | −16% | −6% | −7% | −2% | +10% |
| Alvo 3R | −13% | −3% | −8% | +5% | +18% |

**Conclusão: é uma estratégia de tendência típica.** Cerca de 30% de trades com lucro e resultado
médio de +0,3R a +0,4R, vindo de poucos trades grandes. Acerta 1R em ~50% (melhor que o aleatório,
mas empate em 1:1). Alvos fixos e stop móvel cortam a cauda e destroem a vantagem.
A melhor saída é a virada do sinal. Para alta assertividade é preciso outro tipo de sinal.

---

> ⚠️ **AVISO (seção 12): os resultados das seções 8 a 11 estão inflados por um viés de seleção.**
> O "1º FVG da pernada" era escolhido pela ordem de formação, o que olha o futuro. Com a regra honesta
> (1º FVG executado no tempo), a vantagem no M1/M5 desaparece. Ver seção 12 antes de usar qualquer número abaixo.

## 8. FVG + estrutura pelos fechamentos no M1 — `resultados/estudo_fvg_1m.csv`

Dados: BTC e ETH M1, 01/10/2023 a 18/10/2026 (1,6 milhão de candles cada, nenhum faltando).
Código: `smc.py` (estrutura, FVG, setups), `estudo_fvg.py` (estudo), `figura_smc.py` (imagens).

Regras (venda; compra é o espelho):
- Pivôs nos fechamentos (n = 2, 3, 5), usados só depois de confirmados. Teste de repaint: ok.
- BOS = fechamento abaixo do último fundo; CHoCH = fechamento acima da origem da pernada
  (maior fechamento entre o fundo rompido e a quebra — "o topo que gerou o último fundo").
- FVG de baixa dentro da pernada, não tocado antes da ativação. Ordem limitada na borda da zona,
  válida 240 candles, cancelada se um fechamento atravessar a zona ou a tendência virar.
- Stop: acima do topo da pernada (estrutura) ou acima dos 3 candles do FVG, com folga de 0,1 ATR.
- Acerto = chegou a 1R antes do stop (horizonte de 1 dia). No candle da execução só o stop conta.
- Referência aleatória: mesma sessão, direção e risco (%). Taxas: 0,02% entrada + 0,05% saída.

Resultado sem filtros: ~300 setups por dia e acerto de 1R **abaixo** do aleatório (−1 a −5 pontos)
em todas as 12 configurações. Filtros por zona premium, médias 13/26/50, sessões e "FVG do impulso
da quebra", sozinhos ou combinados, também ficaram abaixo ou empatados com o aleatório.

**Premium foi pior que discount** (41–44% contra 45–47% do aleatório). No M1, o FVG premium fica
perto da origem da pernada e o stop fica minúsculo (R mediano 0,05–0,11%), então o ruído leva ao stop
e as taxas passam de 1R por trade.

**O único filtro que venceu o aleatório nas 12 configurações: o primeiro FVG executado da pernada**
(+3 a +8 pontos). Combinado com um R mínimo (stop longe o bastante do ruído e das taxas):

| Pivô 5, stop estrutura, 1º FVG | Trades/dia | Acerto 1R | Aleatório | R mediano | Bruto | Líquido |
|---|---|---|---|---|---|---|
| BTC, R ≥ 0,2% | 22,5 | 61% | 50% | 0,32% | +0,23R | +0,01R |
| BTC, R ≥ 0,4% | 7,3 | 64% | 49% | 0,55% | +0,28R | **+0,15R** |
| ETH, R ≥ 0,2% | 32,2 | 60% | 50% | 0,35% | +0,20R | +0,00R |
| ETH, R ≥ 0,4% | 12,8 | 63% | 49% | 0,58% | +0,27R | **+0,15R** |

Robustez do candidato (1º FVG, R ≥ 0,4%):
- Ano a ano 2023–2026: 62–64% em todos os anos, nos dois ativos.
- Compra 63–64%, venda 63%. Todas as sessões entre 61% e 67% (sessões não fazem diferença).
- Pivô 3 dá praticamente o mesmo (63%). Stop no FVG também (64–66%, com menos trades).
- R mínimo de 0,3% a 0,8%: acerto estável (61–64%); o líquido melhora com R maior.
- Execução realista da ordem limitada (preço precisa ultrapassar a entrada em 0,01–0,02%):
  acerto 62–63%, líquido +0,12 a +0,14R.
- Uma operação por vez: BTC 4,7 trades/dia, ETH 7,4/dia, ~63% de acerto, +0,15R líquido por trade.
- Tempo até resolver: mediana 35 min (p25 13 min, p75 91 min).

Leitura: o que funciona no M1 não é o FVG premium com stop curto, e sim **entrar no primeiro
retorno ao FVG logo depois da quebra, com o stop do outro lado da pernada inteira**. O stop largo
fica fora do ruído do M1 e o primeiro pullback após a quebra tende a continuar.

Ainda não considerado: slippage do stop em movimentos rápidos, funding dos perpétuos, e custo
real conforme a conta (o usuário informou que maker e taker podem ser iguais na conta dele).

## 9. Tamanho do stop no FVG (M1) — `resultados/estudo_stop_1m.csv`

Mesmas operações da seção 8 (1º FVG da pernada, pivô 5, R pela estrutura ≥ 0,4%); só o stop muda.

| Stop | BTC acerto 1R | BTC acerto 2R | BTC líquido 1R | BTC líquido 2R | ETH líquido 1R | ETH líquido 2R |
|---|---|---|---|---|---|---|
| Estrutura (anterior) | 64% | 42% | +0,15R | +0,20R | +0,15R | +0,19R |
| 50% da distância da estrutura | 70% | 52% | +0,15R | +0,32R | +0,13R | +0,28R |
| 2 × ATR | 72% | 56% | +0,09R | +0,35R | +0,11R | +0,30R |
| **3 × ATR** | **71%** | **52%** | **+0,19R** | **+0,32R** | **+0,17R** | **+0,29R** |
| 5 × ATR | 66% | 45% | +0,18R | +0,24R | +0,16R | +0,21R |
| Acima do FVG | 70% | 56% | −0,03R | +0,25R | −0,02R | +0,21R |

Aleatório com o mesmo R: ~49% em 1R em todas as linhas. Taxas: 0,02% + 0,05%.

- Stop mais curto aumenta o acerto (o primeiro movimento depois da entrada tende a ir a favor),
  mas aumenta o peso das taxas em R. O equilíbrio fica em ~3 × ATR(14) do M1.
- **Melhor combinação: stop 3 × ATR com alvo 2R** — líquido de +0,24R a +0,38R em todos os anos
  (2023–2026), igual em compra e venda; com execução realista da ordem limitada
  (penetração de 0,02%), +0,25R.
- Não precisa de R mínimo no stop em ATR: operações com R pequeno acertam mais e compensam as taxas
  (líquido +0,35R a +0,41R abaixo de 0,5% de R; acima de 0,5%, +0,20R).
- Correção: o exportador do visualizador reaproveitava setups já avaliados ao calcular a variante
  em ATR (colunas duplicadas); corrigido antes de publicar. O estudo não tinha esse problema.

## 10. Grade de configurações do FVG (M1, BTC + ETH) — `resultados/estudo_configs_1m.csv`, `resultados/ranking_configs.txt`

540 combinações por ativo: pivô (3, 5) × entrada (1º FVG; 1º FVG + média 50 + 2 BOS; todos + média 50 + 2 BOS;
todos + médias alinhadas + 2 BOS; todos os FVGs) × R da estrutura mínimo (0; 0,2%; 0,4%) × stop (estrutura,
2 × ATR, 3 × ATR) × alvo (1R, 2R, 3R) × horário (todas; aberturas em São Paulo: Ásia 22h–01h, Londres 03h–07h,
NY 10h30–13h30). Código: `estudo_configs.py` e `ranking_configs.py`. Ranking só com configurações de R bruto
por operação positivo nos dois ativos e nas duas metades (2023–2024 e 2025–2026): 240 de 540.

Resultados (média BTC/ETH, por ativo):

| Objetivo | Configuração | Acerto | R bruto/op | R líquido/op | R por ano |
|---|---|---|---|---|---|
| Mais R bruto acumulado | pivô 3, 1º FVG, sem R mínimo, 2 × ATR, alvo 3R | 32% | +0,27 | **−0,63** | +9.115 bruto (~94 ops/dia) |
| Mais R líquido, uma por vez | **pivô 5, 1º FVG, R ≥ 0,4%, 2 × ATR, alvo 3R** | 43% | **+0,73** | **+0,42** | +1.172 líquido (~7 ops/dia) |
| Mesmo, alvo 2R | pivô 5, 1º FVG, R ≥ 0,4%, 2 × ATR, alvo 2R | 55% | +0,64 | +0,32 | +1.064 líquido |
| Maior acerto | pivô 5, 1º FVG, R ≥ 0,4%, 2 × ATR, alvo 1R | **71%** | +0,41 | +0,10 | +371 líquido |

- O R bruto acumulado máximo vem de volume: ~94 operações/dia com R de 0,12%. As taxas passam de 0,6R
  por operação e o líquido é muito negativo. Em mercado com custo por contrato (ex.: MNQ, custo ~1% do R)
  essa seria a configuração mais interessante; em cripto não.
- **Entrar em mais FVGs piora**: "todos + média 50 + 2 BOS" e "todos + médias alinhadas + 2 BOS" ficam com
  ~30% de acerto em 2R e R bruto negativo (−0,06 a −0,13R). Os FVGs depois do 1º na mesma pernada têm 42%
  em 1R e −0,17R por operação.
- Exigir média 50 + 2 BOS no 1º FVG também piora (+0,42 a +0,50R contra +0,59 a +0,69R sem o filtro).
- Aberturas de São Paulo: mesmo R por operação que todas as horas (+0,66 contra +0,64), com 40% das operações.
- Correção feita no caminho: a primeira versão da grade aplicava o filtro de R antes de escolher o 1º FVG,
  o que misturava FVGs posteriores da pernada. Corrigido antes do ranking final.

## 11. R da estrutura e outros tempos gráficos — `resultados/estudo_timeframes.csv`, `resultados/ranking_timeframes.txt`

**R da estrutura** = distância da entrada até o extremo da pernada (origem) + folga de 0,1 ATR, em % do preço.
É o tamanho da pernada que gerou o FVG. Não é o stop quando o stop é em ATR; serve de filtro.
Desempenho por faixa (M1, 3 anos, pivô 5, 1º FVG, stop 2 × ATR) — `resultados/r_estrutura_faixas.csv`:

| R da estrutura | BTC acerto 1R | BTC R bruto (alvo 3R) | ETH acerto 1R | ETH R bruto (alvo 3R) |
|---|---|---|---|---|
| < 0,1% | 43% | +0,06 | 45% | +0,03 |
| 0,1–0,2% | 54% | +0,30 | 51% | +0,19 |
| 0,2–0,4% | 64% | +0,57 | 59% | +0,42 |
| 0,4–0,6% | 70% | +0,78 | 67% | +0,63 |
| > 0,6% | 74% | +0,80 | 72% | +0,73 |

Imagens: `resultados/r_estrutura_pequena.png`, `_media.png`, `_grande.png`.

**Tempos gráficos** (M5/M15/H1 montados do M1, mesmo período). Adaptação: pernada mínima medida em ATRs do
próprio tempo gráfico (0, 3, 5, 8, 12) em vez de %; validade da ordem e espera pela quebra ajustadas.
Melhor por tempo gráfico (positiva nos 2 ativos e nas 2 metades; média por ativo):

| TF | Configuração | Acerto | R líquido/op | Ops/ano (uma por vez) | R líquido/ano |
|---|---|---|---|---|---|
| M1 | R estrutura ≥ 0,4% (seção 10), 2 × ATR, 3R | 43% | +0,42 | ~2.680 | +1.172 |
| M5 | pivô 3, pernada ≥ 3 ATR, 2 × ATR, 3R | 39–40% | +0,28 a +0,30 | ~1.140 | +400 |
| M5 | pivô 5, pernada ≥ 5 ATR, 2 × ATR, 3R | 43% | +0,45 | ~490 | +225 |
| M15 | pivô 3/5, pernada ≥ 3 ATR, 2 × ATR, 2R/3R | 37–48% | +0,32 a +0,35 | ~350–410 | +145 |
| H1 | pivô 5, pernada ≥ 3 ATR, 2 × ATR, 3R | 39% | +0,49 a +0,51 | ~95 | +54 |

- Acerto em 1R com stop 2 × ATR e pernada ≥ 3–5 ATR: M5 73%, M15 70%, H1 68% — e positivo líquido.
  No M1 chega a 75%, mas o stop fica em ~0,12% e as taxas tornam o líquido negativo.
- A pernada mínima é o filtro que mais importa em todos os tempos gráficos: R bruto por operação (alvo 2R)
  sobe de ~0,2 (sem filtro) para ~0,5–0,8 com pernada ≥ 5 ATR.
- No M1, o filtro em % (R da estrutura ≥ 0,4%) funciona melhor que o filtro em ATR porque também seleciona
  momentos de volatilidade maior, quando o stop em ATR fica grande o bastante para as taxas não pesarem.
- Média 50: não muda nada (o 1º FVG depois da quebra quase sempre já está do lado certo da média; 98% das
  operações são as mesmas com e sem o filtro). As médias são só visuais hoje.

## 12. Correção: viés de seleção no "1º FVG da pernada" — `resultados/correcao_vies_1fvg.csv`

**O erro.** O backtest escolhia o 1º FVG da pernada com `setups.duplicated(['dir', 'origem'])` sem ordenar, ou seja,
pela **ordem de formação** do FVG, considerando só os que chegaram a executar. Numa venda: o FVG A (mais alto) se
forma antes do B (mais baixo); no pullback o preço toca B primeiro. Se depois o preço subir até A, o backtest ficava
com A e **descartava B** — mas o preço subir até A significa que foi contra B, que provavelmente tomou stop. Na hora
em que B executa ninguém sabe se A vai executar: é olhar o futuro. O efeito foi remover operações perdedoras.

**Como foi descoberto.** Ao conferir o indicador do TradingView (que escolhe o 1º FVG executado no tempo) contra o
backtest: nas operações em comum o resultado era idêntico, mas o indicador tinha operações extras que o backtest
descartava, com 37% de acerto em 1R. A regra "só o FVG mais antigo ainda armado" (sem olhar o futuro) também não
recupera a vantagem (45–47% em 1R).

**Resultado honesto** (1º FVG executado no tempo; pivô 5; stop 2 × ATR; filtro R ≥ 0,4% no M1, pernada ≥ 3 ATR nos outros):

| TF | Ativo | Acerto 1R com viés | Acerto 1R honesto | Aleatório | R bruto (3R) honesto | R líquido (3R) honesto |
|---|---|---|---|---|---|---|
| M1 | BTC | 72% | 50% | 49% | +0,04 | −0,33 |
| M1 | ETH | 70% | 50% | 50% | +0,02 | −0,29 |
| M5 | BTC | 66% | 51% | 48% | +0,03 | −0,31 |
| M5 | ETH | 65% | 51% | 50% | +0,05 | −0,15 |
| M15 | BTC | 65% | 53% | 48% | +0,07 | −0,08 |
| M15 | ETH | 67% | 53% | 49% | +0,07 | −0,02 |
| H1 | BTC | 72% | 56% | 49% | +0,25 | +0,19 |
| H1 | ETH | 64% | 51% | 46% | +0,04 | +0,00 |

- No M1 e no M5 a estratégia honesta é igual ao aleatório e perde com as taxas.
- No M15 e no H1 sobra uma vantagem pequena (+3 a +7 pontos sobre o aleatório), positiva só no BTC H1; precisa de
  confirmação em mais ativos e períodos antes de valer alguma coisa.
- A relação "pernada maior = mais acerto" (seção 11) também era do viés: no BTC M1 honesto vai de 45% (< 0,1%) a 50% (> 0,4%).
- **O que continua válido:** dados (seção 1), Ovelha V3 (seções 2–7, sem essa seleção), a conclusão de que "todos os FVGs"
  não têm vantagem (seção 10) e que FVGs posteriores da pernada são piores.
- Correção no código: `smc.primeiro_da_pernada()` (ordena por execução) passou a ser usado em todos os scripts.
  O visualizador foi regenerado com a seleção honesta. O indicador `tradingview/ovelha_fvg.pine` sempre usou a regra
  honesta; `tradingview/conferir_pine.py` replica a lógica dele em Python e confere: 100% das operações iguais ao backtest corrigido.
- Lição para o método: toda regra de seleção ("o primeiro", "o melhor", "deduplicar") precisa ser checada contra
  uma simulação candle a candle, que só conhece o passado.

## 13. Stop protegido e "FVG validado" (seleção honesta) — `resultados/estudo_stop_protegido.csv`, `resultados/stop_protegido_robustez.csv`

Regra do operador: entrar no FVG mais próximo do preço (1º executado da pernada) e proteger o stop além do próximo
FVG válido da mesma pernada (ou do mais distante, ou da média 50). Tudo conhecido no momento da entrada.
Código: `estudo_stop_protegido.py` (usa `montar_setups(..., candidatos=lista)` para conhecer os FVGs não executados).

- **Onde fica o stop não cria vantagem:** 2 × ATR, FVG próprio, próximo FVG, FVG mais distante e média 50 dão
  praticamente o mesmo acerto, perto do aleatório, em todos os tempos gráficos.
- **O filtro "pernada com 2+ FVGs válidos" cria uma vantagem pequena e consistente** (pivô 5, 1º FVG executado):

| TF | Ativo | Acerto 1R (2+ FVGs) | Aleatório | Vantagem 2023–24 | Vantagem 2025–26 | R bruto (2R) | R líquido (2R) |
|---|---|---|---|---|---|---|---|
| M5 | BTC | 53–54% | 48% | +4,0 a +4,1 | +5,8 a +7,0 | +0,08 a +0,09 | −0,25 a −0,27 |
| M5 | ETH | 53% | 48% | +4,6 a +6,3 | +4,6 a +5,1 | +0,09 a +0,12 | −0,11 a −0,12 |
| M15 | BTC | 54–55% | 49% | +5,8 a +7,3 | +4,2 a +4,9 | +0,10 a +0,12 | −0,04 a −0,06 |
| M15 | ETH | 54–55% | 48–49% | +7,0 a +8,0 | +3,8 a +6,0 | +0,12 a +0,13 | +0,01 a +0,02 |
| H1 | BTC | 55% | 51–53% | +3,0 a +6,0 | +0,6 a +4,7 | +0,15 a +0,18 | +0,09 a +0,10 |
| H1 | ETH | 52–53% | 48–52% | +2,5 a +3,5 | −1,9 a +5,6 | +0,02 | −0,03 |

  Sem o filtro (todas as pernadas), a vantagem é de 0 a +3 pontos. Uma pernada com vários FVGs é um deslocamento
  forte — a versão objetiva de "FVG validado".
- **Obstáculo: taxas de cripto.** A vantagem bruta (+0,08 a +0,13R no M5/M15) é menor que a taxa em R no M5 e
  empata no M15. Num mercado com custo fixo por contrato (MNQ: ~1% do R), o bruto vira praticamente o líquido.

## 14. Candle de duplo rompimento das médias 13/20 — `resultados/estudo_duplo.csv`, `resultados/estudo_duplo.txt`

Regra 100% objetiva (`duplo_rompimento.py`): candle abre abaixo das duas médias (13 e 20) e fecha acima das duas →
compra na abertura seguinte (venda é o espelho). Testado BTC e ETH, M1/M5/M15/H1, ~3 anos, SMA e EMA, 4 stops
(candle − 0,1 ATR; candle − 0,5 ATR; 1 ATR; mínima de 10 candles − 0,1 ATR), com e sem contexto (5 fechamentos
anteriores do outro lado das médias). 128 configurações.

- **Mediana de todas as configurações:** acerto em 1R 49–50% contra 46–47% do aleatório; R bruto em 1R ≈ 0,
  em 2R +0,02 a +0,09. Líquido negativo em todas as medianas (M1: −0,8 a −2,3R por operação).
- O aleatório fica abaixo de 50% por causa das regras conservadoras (no candle de entrada só vale o stop; empate = stop).
  A vantagem real de +3pp vem em boa parte de ser comparada a esse número, e não se converte em R bruto.
- **Melhor caso:** BTC H1, SMA, stop candle − 0,5 ATR, com contexto: 180 operações/ano, acerto 1R 55,4% × 49,5%
  aleatório (+3,1pp em 2023–24, +7,8pp em 2025–26), bruto +0,11R (1R), +0,12R (2R), +0,21R (3R); líquido +0,02R.
  **Não se repete no ETH** (mesma configuração: 50,7% × 53,3% aleatório). Melhor de 128 testes → provável sorte.
- Nenhuma configuração passou de 55,4% em 1:1. **Meta de 65–70% em 1:1 não alcançada.**

Conclusão: o sinal é objetivo e honesto, mas sozinho não tem vantagem estatística relevante em cripto.

## 15. Recuo a favor da tendência com saída na média de 5 — `resultados/estudo_recuo.csv`, `resultados/estudo_recuo.txt`

Regra (`recuo.py`): com o fechamento acima da média de 200, compra quando o RSI(2) < 10 ou após 3 fechamentos
seguidos em queda; entra na abertura seguinte e sai no primeiro fechamento acima da média de 5 (ou em 10 candles).
Stop: nenhum, 2 ATR ou 3 ATR. Venda é o espelho. BTC, ETH, SOL e ADA, H1/H4/D1, desde 2017 (SOL 2020, ADA 2018).
Aleatório: entradas sorteadas em qualquer recuo (mesma tendência, preço do lado errado da média de 5), mesmas saídas.

- **Acerto alto, mas igual ao aleatório:** 65–72% de acerto. O aleatório com a mesma saída também dá 66–72%.
  Em média, o gatilho fica 1,8 ponto **abaixo** do aleatório. O acerto alto vem da forma da saída
  (ganho médio ~+0,22R, perda média ~−0,45R), não do gatilho.
- **R bruto:** +0,02 a +0,04 no H1 e em torno de 0 no H4. No D1, de 0 a +0,10, mas são só 8–10 operações por ano
  por ativo, e o aleatório também fica positivo (alta das criptos no período). Líquido ≈ 0.
- Filtro de tendência ajuda pouco (compra: mediana 68% × 66% sem filtro). Vendas são piores que compras.
- **Conclusão:** acerto de ~70% de verdade, mas sem vantagem: qualquer compra em recuo com essa saída faz o mesmo,
  e o R por operação fica perto de zero.

## Estratégia de FVG: arquivada

Parada por decisão do usuário: a marcação de pernadas e fractais ainda depende de interpretação (o operador consideraria
pernadas que o código não considera). Estado ao arquivar: seção 12/13 — com seleção honesta, acerto ~50–55% em 1R,
vantagem pequena só com o filtro de 2+ FVGs no M5/M15. Retomar quando houver uma definição de pernada fechada com o operador.

## Próximos passos (em aberto)

### A. Estratégia de fair value gap (SMC) — feita no M1 (seção 8); próximos testes abaixo

Reproduzir a operação manual do usuário e medir a assertividade em 1R, 2R e 3R contra o aleatório.
Proposta de regras objetivas (a confirmar com o usuário):

1. **Topos e fundos (swings):** fractal de N candles de cada lado, usado só após confirmado.
2. **Quebra de estrutura (venda):** fechamento abaixo do último fundo confirmado.
3. **Captura de liquidez (opcional):** antes da quebra, o preço passou acima de um topo anterior
   e fechou de volta abaixo.
4. **FVG de baixa:** três candles com `mínima[i−2] > máxima[i]`. A zona é
   `[máxima[i], mínima[i−2]]`, formada no movimento que quebrou a estrutura.
5. **Filtro de média (opcional):** preço abaixo da média.
6. **Entrada:** ordem limitada na zona (início, 50% ou fim), válida por X candles ou até invalidar.
7. **Stop:** acima do topo que originou a queda, com folga (ATR ou fixa).
8. **Alvo:** 1R. Também medir 2R e 3R.
9. **Custos:** entrada limitada (maker ~0,02%), stop a mercado (taker 0,05%).

Compra é o espelho. Timeframes candidatos: M5 e M15 (precisam ser baixados).

Perguntas abertas ao usuário: timeframe operado; ponto de entrada na zona; onde fica o stop
exatamente; se a captura de liquidez é obrigatória; quanto tempo a ordem espera; se vale só o
primeiro FVG após a quebra.

### B. Ovelha V3 em outros timeframes

Rodar `multi_ativos.py` e `experimento_saidas.py` em H4 e D1 (menos trades, menos custo) e em M5.
Precisa baixar esses timeframes com `dados_binance.carregar`.

### C. Filtro da maré

Só vender quando o diário estiver abaixo da média 200 (e só comprar acima), mantendo a saída
na virada. Objetivo: reduzir o drawdown e as perdas do lado vendido no BTC e na SOL.

### D. Continuação do estudo de FVG
- Validar o candidato (1º FVG + R mínimo) em outros ativos (SOL, BNB) e em M5/M15.
- Testar saída parcial (metade em 1R, resto em 2R/3R) com stop 3 × ATR.
- Levar a regra para o indicador do TradingView (Pine) para o usuário acompanhar ao vivo:
  estrutura pelos fechamentos, linha dos 50%, destaque do 1º FVG da pernada, R mínimo e alerta
  com entrada, stop e alvo. Usar `barstate.isconfirmed` para não repintar ao vivo.
- Corrigir no relatório: setups com horizonte incompleto no fim dos dados contam como erro (impacto desprezível).

### E. Depois da seção 13
- Testar o filtro "2+ FVGs" (e medidas de força do deslocamento: tamanho do impulso em ATR, BOS seguidos) em dados
  do MNQ (exportados do TradingView) — onde a taxa não come a vantagem bruta.
- Validar em outros ativos de cripto sem reajustar parâmetros (SOL, BNB, XRP).
- Acompanhar ao vivo com o indicador (opção "só pernadas com 2+ FVGs") e comparar com o backtest.
