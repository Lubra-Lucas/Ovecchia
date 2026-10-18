"""
Ovelha V3: média móvel + faixa de ATR com histerese.

  close > média + k·ATR  -> COMPRA (+1)
  close < média - k·ATR  -> VENDA  (-1)
  dentro da faixa        -> mantém o estado anterior

O estado é decidido no fechamento do candle e só usa dados até aquele candle.
"""

import numpy as np
import pandas as pd


def calcular_atr(df, periodo=14):
    fech_ant = df['close'].shift(1)
    tr = pd.concat([
        df['high'] - df['low'],
        (df['high'] - fech_ant).abs(),
        (df['low'] - fech_ant).abs(),
    ], axis=1).max(axis=1)
    return tr.rolling(periodo).mean()


def calcular_sinais(df, periodo_media=100, k=1.0, periodo_atr=14):
    """Devolve uma cópia do df com as colunas media, atr, faixa_sup, faixa_inf e estado (+1, -1 ou 0 antes do 1º sinal)."""
    out = df.copy()
    out['media'] = out['close'].rolling(periodo_media).mean()
    out['atr'] = calcular_atr(out, periodo_atr)
    out['faixa_sup'] = out['media'] + k * out['atr']
    out['faixa_inf'] = out['media'] - k * out['atr']

    gatilho = pd.Series(np.nan, index=out.index)
    gatilho[out['close'] > out['faixa_sup']] = 1
    gatilho[out['close'] < out['faixa_inf']] = -1
    out['estado'] = gatilho.ffill().fillna(0).astype(int)
    return out
