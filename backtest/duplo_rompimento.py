"""
Estratégia do candle de duplo rompimento das médias.

Sinal de compra no fechamento do candle t: abriu abaixo das duas médias e fechou acima das duas
(médias calculadas com os fechamentos até t, inclusive). Venda é o espelho.
Entrada a mercado na abertura do candle t+1. Stops (compra; venda é o espelho):
  - candle:       mínima do candle de rompimento − 0,1 ATR
  - candle_folga: mínima do candle de rompimento − 0,5 ATR
  - atr1:         entrada − 1 × ATR
  - estrutura:    menor mínima dos últimos 10 candles (até t) − 0,1 ATR
Contexto (opcional): os 5 fechamentos anteriores ao sinal estavam todos do outro lado das duas médias.
"""

import numpy as np
import pandas as pd

import estrategia as est

STOPS = ('candle', 'candle_folga', 'atr1', 'estrutura')


def medias(close, tipo, curta=13, longa=20):
    s = pd.Series(close)
    if tipo == 'EMA':
        return s.ewm(span=curta, adjust=False).mean().to_numpy(), s.ewm(span=longa, adjust=False).mean().to_numpy()
    return s.rolling(curta).mean().to_numpy(), s.rolling(longa).mean().to_numpy()


def sinais(df, tipo='SMA', curta=13, longa=20, contexto=5, janela_estrutura=10):
    o, h, l, c = (df[x].to_numpy() for x in ('open', 'high', 'low', 'close'))
    atr = est.calcular_atr(df).to_numpy()
    m1, m2 = medias(c, tipo, curta, longa)
    baixo, alto = np.minimum(m1, m2), np.maximum(m1, m2)
    with np.errstate(invalid='ignore'):
        compra = (o < baixo) & (c > alto)
        venda = (o > alto) & (c < baixo)
        abaixo = pd.Series(c < baixo).astype(float)
        acima = pd.Series(c > alto).astype(float)
    # contexto: os N fechamentos ANTERIORES ao sinal estavam do outro lado das duas médias
    ctx_compra = abaixo.shift(1).rolling(contexto).sum().to_numpy() == contexto
    ctx_venda = acima.shift(1).rolling(contexto).sum().to_numpy() == contexto
    menor = pd.Series(l).rolling(janela_estrutura).min().to_numpy()
    maior = pd.Series(h).rolling(janela_estrutura).max().to_numpy()

    t = np.flatnonzero((compra | venda) & ~np.isnan(atr))
    t = t[t + 1 < len(c)]
    d = np.where(compra[t], 1, -1)
    e = o[t + 1]
    a = atr[t]
    s = pd.DataFrame({'i_sinal': t, 'i_exec': t + 1, 'dir': d, 'entrada': e, 'atr': a,
                      'contexto': np.where(d == 1, ctx_compra[t], ctx_venda[t]),
                      'corpo_atr': np.abs(c[t] - o[t]) / a})
    s['stop_candle'] = np.where(d == 1, l[t] - 0.1 * a, h[t] + 0.1 * a)
    s['stop_candle_folga'] = np.where(d == 1, l[t] - 0.5 * a, h[t] + 0.5 * a)
    s['stop_atr1'] = e - d * a
    s['stop_estrutura'] = np.where(d == 1, menor[t] - 0.1 * a, maior[t] + 0.1 * a)
    return s
