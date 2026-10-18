"""
Estudo do candle de duplo rompimento das médias (duplo_rompimento.py), BTC e ETH, M1/M5/M15/H1, 3 anos.

Métricas em R bruto primeiro: acerto em 1R/2R/3R, R bruto por operação e por ano (alvo 1R), comparação com o
aleatório (50% em 1:1 e entradas sorteadas com mesmo risco e horário) e estabilidade entre 2023–24 e 2025–26.

Uso: python estudo_duplo.py BTCUSDT ETHUSDT
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

import duplo_rompimento as dr
import estudo_fvg as ef
import smc
from estudo_timeframes import TEMPOS, agregar

TAXA = 0.0007
SAIDA = Path(__file__).parent / 'saida'


def main(ativos):
    rng = np.random.default_rng(41)
    linhas = []
    for ativo in ativos:
        d1 = pd.read_pickle(Path(__file__).parent / 'cache' / f'{ativo}_1m.pkl')
        anos_total = (d1['time'].iloc[-1] - d1['time'].iloc[0]).days / 365.25
        for tf, (regra, _, _) in TEMPOS.items():
            df = agregar(d1, regra)
            sessoes = smc.sessao(df['time'])
            ano = df['time'].dt.year.to_numpy()
            for tipo in ('SMA', 'EMA'):
                s = dr.sinais(df, tipo)
                for stop in dr.STOPS:
                    a = smc.avaliar(df, s, f'stop_{stop}')
                    a['sessao'] = sessoes[a['i_exec'].to_numpy()]
                    a['aleat_1r'] = ef.aleatorio_1r(df, a, sessoes, rng)
                    a['ano'] = ano[a['i_exec'].to_numpy()]
                    for contexto in (False, True):
                        x = a[a['contexto']] if contexto else a
                        if len(x) < 30:
                            continue
                        m1 = x['ano'] <= 2024
                        linhas.append({
                            'ativo': ativo[:3], 'tf': tf, 'media': tipo, 'stop': stop, 'contexto': contexto,
                            'ops_ano': len(x) / anos_total, 'r_mediano': x['risco_pct'].median(),
                            'acerto_1r': x['acerto_1r'].mean(), 'aleat_1r': x['aleat_1r'].mean(),
                            'acerto_2r': x['acerto_2r'].mean(), 'acerto_3r': x['acerto_3r'].mean(),
                            'bruto_1r': x['resultado_1r'].mean(), 'bruto_2r': x['resultado_2r'].mean(),
                            'bruto_3r': x['resultado_3r'].mean(),
                            'r_bruto_ano_1r': x['resultado_1r'].sum() / anos_total,
                            'liq_1r': (x['resultado_1r'] - TAXA / x['risco_pct']).mean(),
                            'vant_23_24': x.loc[m1, 'acerto_1r'].mean() - x.loc[m1, 'aleat_1r'].mean(),
                            'vant_25_26': x.loc[~m1, 'acerto_1r'].mean() - x.loc[~m1, 'aleat_1r'].mean(),
                        })
                    print(f'  {ativo} {tf} {tipo} {stop}: {len(a)}', file=sys.stderr)
    res = pd.DataFrame(linhas)
    SAIDA.mkdir(exist_ok=True)
    res.to_csv(SAIDA / 'estudo_duplo.csv', index=False)
    return res


if __name__ == '__main__':
    res = main(sys.argv[1:])
    pd.set_option('display.width', 280)
    pd.set_option('display.max_rows', 500)
    p = '{:.0%}'.format
    r = '{:+.2f}'.format
    print(res.to_string(index=False, formatters={
        'ops_ano': '{:.0f}'.format, 'r_mediano': '{:.2%}'.format, 'acerto_1r': p, 'aleat_1r': p, 'acerto_2r': p,
        'acerto_3r': p, 'bruto_1r': r, 'bruto_2r': r, 'bruto_3r': r, 'r_bruto_ano_1r': '{:+.0f}'.format,
        'liq_1r': r, 'vant_23_24': '{:+.1%}'.format, 'vant_25_26': '{:+.1%}'.format}))
