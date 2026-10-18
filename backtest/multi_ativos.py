"""
Mesma estratégia e mesma grade em vários ativos: separa o efeito do ativo do efeito do sinal.

Uso: python multi_ativos.py 1h BTCUSDT ETHUSDT SOLUSDT ADAUSDT ETHBTC
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

import estrategia as est
import motor as m
import teste_aleatorio as ta
from rodar_grade import INICIO, KS, MEDIAS, buy_and_hold

CUSTO_FUTUROS = 0.0005


def checar(df, timeframe):
    from dados_binance import TIMEFRAMES_MS
    passo = TIMEFRAMES_MS[timeframe]
    diffs = df['open_time'].diff().iloc[1:]
    esperados = (df['open_time'].iloc[-1] - df['open_time'].iloc[0]) // passo + 1
    ohlc = ((df['high'] < df[['open', 'close', 'low']].max(axis=1)) |
            (df['low'] > df[['open', 'close', 'high']].min(axis=1))).sum()
    return {'faltando_pct': 1 - len(df) / esperados, 'duplicados': int(df['open_time'].duplicated().sum()),
            'fora_ordem': int((diffs <= 0).sum()), 'ohlc_ruim': int(ohlc)}


def analisar(simbolo, timeframe, rng):
    df = pd.read_pickle(Path(__file__).parent / 'cache' / f'{simbolo}_{timeframe}.pkl')
    df = df[df['time'] >= INICIO].reset_index(drop=True)
    chk = checar(df, timeframe)
    bh = buy_and_hold(df)

    linhas = []
    for media in MEDIAS:
        for k in KS:
            s = est.calcular_sinais(df, media, k)
            t = m.extrair_trades(s, CUSTO_FUTUROS)
            eq = m.curva_capital(s, t, CUSTO_FUTUROS)
            fech = t[~t['aberto']]
            bruto = fech['dir'] * (fech['preco_saida'] / fech['preco_entrada'] - 1)
            alea, _ = ta.comparar_com_aleatorio(df, media, k, rng)
            linhas.append({
                'ret_anual': eq.iloc[-1] ** (365.25 / (eq.index[-1] - eq.index[0]).days) - 1,
                'dd': m.drawdown_max(eq),
                'bruto_trade': bruto.mean(),
                'bruto_r': (bruto / fech['risco_pct']).mean(),
                'vant_compra': alea['compras'][0] - alea['compras'][1],
                'vant_venda': alea['vendas'][0] - alea['vendas'][1],
                'ret_vendas': (1 + fech.loc[fech['dir'] == -1, 'retorno']).prod() - 1,
            })
    g = pd.DataFrame(linhas)
    return {
        'ativo': simbolo,
        'inicio': df['time'].iloc[0].strftime('%Y-%m'),
        'dados_ok': chk['duplicados'] == 0 and chk['fora_ordem'] == 0 and chk['ohlc_ruim'] == 0,
        'faltando': chk['faltando_pct'],
        'bh_anual': bh['retorno_anual'],
        'bh_dd': bh['dd_max'],
        'combos_positivas': f"{(g['ret_anual'] > 0).sum()}/12",
        'ret_anual_mediano': g['ret_anual'].median(),
        'dd_mediano': g['dd'].median(),
        'bruto_trade_med': g['bruto_trade'].mean(),
        'bruto_r_med': g['bruto_r'].mean(),
        'vant_compra': g['vant_compra'].mean(),
        'vant_venda': g['vant_venda'].mean(),
        'vendas_positivas': f"{(g['ret_vendas'] > 0).sum()}/12",
        'sinal_vence_aleatorio': f"{((g['vant_compra'] > 0) & (g['vant_venda'] > 0)).sum()}/12",
    }


if __name__ == '__main__':
    timeframe, ativos = sys.argv[1], sys.argv[2:]
    rng = np.random.default_rng(42)
    res = pd.DataFrame([analisar(a, timeframe, rng) for a in ativos])
    pd.set_option('display.width', 250)
    p0 = '{:+.0%}'.format
    p2 = '{:+.2%}'.format
    print(res.to_string(index=False, formatters={
        'faltando': '{:.2%}'.format, 'bh_anual': p0, 'bh_dd': p0, 'ret_anual_mediano': p0, 'dd_mediano': p0,
        'bruto_trade_med': p2, 'bruto_r_med': '{:+.2f}R'.format, 'vant_compra': p2, 'vant_venda': p2}))
