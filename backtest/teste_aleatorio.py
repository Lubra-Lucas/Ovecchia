"""
A estratégia tem vantagem ou só surfa a tendência do ativo?

Compara o retorno bruto médio dos trades da estratégia com o de trades aleatórios
com a MESMA direção e a MESMA duração, sorteados em qualquer ponto do histórico.
Se a estratégia não supera o aleatório, o lucro vem do ativo, não do sinal.

Uso: python teste_aleatorio.py BTCUSDT 1h
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

import estrategia as est
import motor as m
from rodar_grade import INICIO, KS, MEDIAS

SORTEIOS = 500


def comparar_com_aleatorio(df, media, k, rng):
    s = est.calcular_sinais(df, media, k)
    t = m.extrair_trades(s, custo_lado=0)
    t = t[~t['aberto']]
    abertura = df['open'].to_numpy()
    n = len(abertura)

    bruto = t['dir'].to_numpy() * (t['preco_saida'].to_numpy() / t['preco_entrada'].to_numpy() - 1)
    dur = t['duracao'].to_numpy()
    dirs = t['dir'].to_numpy()

    medias_aleat = {1: [], -1: []}
    for _ in range(SORTEIOS):
        ini = rng.integers(0, n - dur - 1)
        r = dirs * (abertura[ini + dur] / abertura[ini] - 1)
        for d in (1, -1):
            medias_aleat[d].append(r[dirs == d].mean())

    out = {}
    for d, nome in ((1, 'compras'), (-1, 'vendas')):
        estrat = bruto[dirs == d].mean()
        aleat = np.array(medias_aleat[d])
        out[nome] = (estrat, aleat.mean(), (aleat >= estrat).mean())
    return out, len(t)


def main(simbolo='BTCUSDT', timeframe='1h'):
    df = pd.read_pickle(Path(__file__).parent / 'cache' / f'{simbolo}_{timeframe}.pkl')
    df = df[df['time'] >= INICIO].reset_index(drop=True)
    rng = np.random.default_rng(42)
    linhas = []
    for media in MEDIAS:
        for k in KS:
            res, n_trades = comparar_com_aleatorio(df, media, k, rng)
            linha = {'media': media, 'k': k, 'trades': n_trades}
            for lado, (estrat, aleat, p) in res.items():
                linha[f'{lado}_estrat'] = estrat
                linha[f'{lado}_aleat'] = aleat
                linha[f'{lado}_vantagem'] = estrat - aleat
                linha[f'{lado}_p'] = p
            linhas.append(linha)
    return pd.DataFrame(linhas)


if __name__ == '__main__':
    res = main(*sys.argv[1:3])
    f = '{:+.2%}'.format
    print(res.to_string(index=False, formatters={c: f for c in res.columns if c.endswith(('estrat', 'aleat', 'vantagem'))}
                        | {c: '{:.2f}'.format for c in res.columns if c.endswith('_p')}))
