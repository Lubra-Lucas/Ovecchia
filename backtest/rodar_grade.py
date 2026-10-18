"""
Roda a grade de parâmetros da Ovelha V3 num ativo/timeframe e salva os resultados.

Uso: python rodar_grade.py BTCUSDT 1h
"""

import sys
from pathlib import Path

import pandas as pd

import estrategia as est
import motor as m

MEDIAS = [50, 100, 200]
KS = [0.5, 1.0, 1.5, 2.0]
INICIO = '2018-03-01'
CUSTO_LADO = 0.0005  # futuros da Binance (taker)
SAIDA = Path(__file__).parent / 'saida'


def buy_and_hold(df):
    eq = pd.Series((df['close'] / df['open'].iloc[0]).to_numpy(), index=df['time'])
    anos = (eq.index[-1] - eq.index[0]).days / 365.25
    return {'retorno_total': eq.iloc[-1] - 1, 'retorno_anual': eq.iloc[-1] ** (1 / anos) - 1,
            'dd_max': m.drawdown_max(eq)}


def retorno_periodo(eq, ini, fim):
    p = eq[(eq.index >= ini) & (eq.index < fim)]
    return p.iloc[-1] / p.iloc[0] - 1


def main(simbolo='BTCUSDT', timeframe='1h'):
    df = pd.read_pickle(Path(__file__).parent / 'cache' / f'{simbolo}_{timeframe}.pkl')
    df = df[df['time'] >= INICIO].reset_index(drop=True)
    SAIDA.mkdir(exist_ok=True)

    linhas = []
    for media in MEDIAS:
        for k in KS:
            s = est.calcular_sinais(df, media, k)
            t = m.extrair_trades(s, CUSTO_LADO)
            eq = m.curva_capital(s, t, CUSTO_LADO)
            r = m.resumo(t, eq)
            r.update({'media': media, 'k': k,
                      'ret_2018_2021': retorno_periodo(eq, '2018-03-01', '2022-01-01'),
                      'ret_2022_2026': retorno_periodo(eq, '2022-01-01', '2100-01-01')})
            for d in ['COMPRA', 'VENDA']:
                td = t[(t['direcao'] == d) & ~t['aberto']]
                r[f'ret_{d.lower()}s'] = (1 + td['retorno']).prod() - 1
                r[f'acerto_{d.lower()}s'] = (td['retorno'] > 0).mean()
            linhas.append(r)

    res = pd.DataFrame(linhas)
    res.to_csv(SAIDA / f'grade_{simbolo}_{timeframe}.csv', index=False)
    bh = buy_and_hold(df)
    return res, bh


if __name__ == '__main__':
    res, bh = main(*sys.argv[1:3])
    pd.set_option('display.width', 250)
    pct = lambda x: f'{x:+.0%}'
    print('Buy & hold:', {k: pct(v) for k, v in bh.items()})
    print(res[['media', 'k', 'retorno_total', 'retorno_anual', 'dd_max', 'trades', 'acerto', 'profit_factor',
               'duracao_media_h', 'ret_2018_2021', 'ret_2022_2026', 'ret_compras', 'ret_vendas']].to_string(
        index=False, formatters={c: pct for c in ['retorno_total', 'retorno_anual', 'dd_max', 'acerto',
                                                  'ret_2018_2021', 'ret_2022_2026', 'ret_compras', 'ret_vendas']}
        | {'profit_factor': '{:.2f}'.format, 'duracao_media_h': '{:.0f}'.format}))
    print()
    print(res[['media', 'k', 'acerto_1r', 'acerto_2r', 'acerto_3r', 'resultado_medio_r', 'risco_mediano', 'mfe_mediano',
               'mae_mediano']].to_string(index=False, formatters={c: '{:.0%}'.format for c in
                                                                  ['acerto_1r', 'acerto_2r', 'acerto_3r']}
                                         | {'resultado_medio_r': '{:+.2f}R'.format}
                                         | {c: '{:.2%}'.format for c in ['risco_mediano', 'mfe_mediano', 'mae_mediano']}))
