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
- Testar alvo 1,5R e 2R para o candidato e a saída parcial.
- Levar a regra para o indicador do TradingView (Pine) para o usuário acompanhar ao vivo:
  estrutura pelos fechamentos, linha dos 50%, destaque do 1º FVG da pernada, R mínimo e alerta
  com entrada, stop e alvo. Usar `barstate.isconfirmed` para não repintar ao vivo.
- Corrigir no relatório: setups com horizonte incompleto no fim dos dados contam como erro (impacto desprezível).
