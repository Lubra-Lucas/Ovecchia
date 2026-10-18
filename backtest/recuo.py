"""
Estratégia de recuo a favor da tendência com saída na média curta (reversão de curto prazo).

Compra (venda é o espelho):
  - tendência: fechamento acima da média simples de 200 (opcional)
  - gatilho no fechamento do candle t:
      'rsi2'    RSI de 2 períodos (Wilder) abaixo de 10
      'quedas3' 3 fechamentos seguidos em queda
  - entrada a mercado na abertura de t+1
  - saída: primeiro fechamento acima da média simples de 5 -> sai na abertura seguinte;
    ou depois de MAX_BARRAS candles na operação (saída por tempo, também na abertura seguinte)
  - stop: 'atr2' / 'atr3' = entrada − 2 ou 3 × ATR(14) do candle do sinal; 'nenhum' = só saída por tempo

R: distância do stop. Na variante 'nenhum', R nominal = 3 × ATR (para comparar com as outras).
Regras de execução: o stop vale desde o candle de entrada; gap além do stop sai na abertura;
no mesmo candle, o stop é conferido antes da saída pelo fechamento. Uma operação por vez.
"""

import numpy as np
import pandas as pd

import estrategia as est

MAX_BARRAS = 10
GATILHOS = ('rsi2', 'quedas3')
STOPS = {'nenhum': 3.0, 'atr2': 2.0, 'atr3': 3.0}      # múltiplo de ATR que define o R


def rsi(close, periodo=2):
    d = pd.Series(close).diff()
    ganho = d.clip(lower=0).ewm(alpha=1 / periodo, adjust=False).mean()
    perda = (-d.clip(upper=0)).ewm(alpha=1 / periodo, adjust=False).mean()
    with np.errstate(divide='ignore', invalid='ignore'):
        return (100 - 100 / (1 + ganho / perda)).to_numpy()


def indicadores(df):
    c = df['close']
    return {'o': df['open'].to_numpy(), 'h': df['high'].to_numpy(), 'l': df['low'].to_numpy(),
            'c': c.to_numpy(), 'm5': c.rolling(5).mean().to_numpy(), 'm200': c.rolling(200).mean().to_numpy(),
            'atr': est.calcular_atr(df).to_numpy(), 'rsi2': rsi(c.to_numpy()),
            'quedas3': (c.diff() < 0).rolling(3).sum().to_numpy() == 3,
            'altas3': (c.diff() > 0).rolling(3).sum().to_numpy() == 3}


def candidatos(ind, d, gatilho, tendencia):
    """Candles em que o sinal acontece (no fechamento). gatilho=None -> só tendência e preço do lado errado da m5."""
    c, m5, m200 = ind['c'], ind['m5'], ind['m200']
    with np.errstate(invalid='ignore'):
        ok = ~np.isnan(m200) & ~np.isnan(ind['atr'])
        if tendencia:
            ok &= (c > m200) if d == 1 else (c < m200)
        if gatilho == 'rsi2':
            ok &= (ind['rsi2'] < 10) if d == 1 else (ind['rsi2'] > 90)
        elif gatilho == 'quedas3':
            ok &= ind['quedas3'] if d == 1 else ind['altas3']
        else:
            ok &= (c < m5) if d == 1 else (c > m5)
    return np.flatnonzero(ok)


def simular(ind, sinais, d, stop):
    """Executa os sinais em sequência (uma operação por vez). Devolve um DataFrame de operações."""
    o, h, l, c, m5, atr = ind['o'], ind['h'], ind['l'], ind['c'], ind['m5'], ind['atr']
    n = len(c)
    mult = STOPS[stop]
    ops = []
    livre = 0
    for t in sinais:
        if t < livre or t + 1 >= n:
            continue
        e = t + 1
        entrada = o[e]
        risco = mult * atr[t]
        nivel = entrada - d * risco if stop != 'nenhum' else None
        saida = x = None
        j = e
        while j < n:
            if nivel is not None:
                if j > e and d * (o[j] - nivel) <= 0:
                    saida, x = o[j], j
                    break
                if (d == 1 and l[j] <= nivel) or (d == -1 and h[j] >= nivel):
                    saida, x = nivel, j
                    break
            if (d == 1 and c[j] > m5[j]) or (d == -1 and c[j] < m5[j]) or j - e + 1 >= MAX_BARRAS:
                if j + 1 < n:
                    saida, x = o[j + 1], j + 1
                break
            j += 1
        if saida is None:
            break
        ops.append((t, e, x, entrada, saida, risco))
        livre = x
    ops = pd.DataFrame(ops, columns=['i_sinal', 'i_exec', 'i_saida', 'entrada', 'saida', 'risco'])
    ops['r'] = d * (ops['saida'] - ops['entrada']) / ops['risco']
    ops['risco_pct'] = ops['risco'] / ops['entrada']
    ops['barras'] = ops['i_saida'] - ops['i_exec']
    return ops
