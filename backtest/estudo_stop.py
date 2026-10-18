"""
Tamanho do stop na estratégia de FVG (M1): mesmas operações, só o stop muda.

Operações: 1º FVG executado da pernada (pivô 5), com R pela estrutura ≥ 0,4% — as mesmas do
visualizador. Variações de stop:
  - estrutura: além do extremo da pernada + 0,1 ATR (o atual)
  - fração da pernada: 25%, 50%, 75% da distância entre a entrada e o stop da estrutura
  - ATR: 2, 3, 5 e 8 × ATR(14) do M1 a partir da entrada (ATR do candle que formou o FVG)
  - FVG: além dos 3 candles do FVG + 0,1 ATR
Cada variação é comparada com entradas aleatórias de mesmo risco, direção e sessão.

Uso: python estudo_stop.py BTCUSDT ETHUSDT
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

import estrategia as est
import estudo_fvg as ef
import smc

R_MINIMO = 0.004
FRACOES = (0.25, 0.5, 0.75)
MULT_ATR = (2, 3, 5, 8)


def main(ativos):
    rng = np.random.default_rng(23)
    linhas = []
    for ativo in ativos:
        df = pd.read_pickle(Path(__file__).parent / 'cache' / f'{ativo}_1m.pkl')
        atr = est.calcular_atr(df).to_numpy()
        e = smc.estrutura(df, 5)
        s = smc.montar_setups(df, e, atr)
        s = smc.primeiro_da_pernada(s).copy()
        s = s[s['dir'] * (s['entrada'] - s['stop_estrutura']) / s['entrada'] >= R_MINIMO].reset_index(drop=True)
        sessoes = smc.sessao(df['time'])
        dias = (df['time'].iloc[-1] - df['time'].iloc[0]).days

        variacoes = {'estrutura (atual)': 'stop_estrutura', 'acima/abaixo do FVG': 'stop_fvg'}
        for f in FRACOES:
            col = f'stop_frac_{int(f * 100)}'
            s[col] = s['entrada'] + f * (s['stop_estrutura'] - s['entrada'])
            variacoes[f'{f:.0%} da distância da estrutura'] = col
        for mlt in MULT_ATR:
            col = f'stop_atr_{mlt}'
            s[col] = s['entrada'] - s['dir'] * mlt * atr[s['i_fvg'].to_numpy()]
            variacoes[f'{mlt} × ATR'] = col

        for nome, col in variacoes.items():
            a = smc.avaliar(df, s, col)
            a['sessao'] = sessoes[a['i_exec'].to_numpy()]
            a['aleat_1r'] = ef.aleatorio_1r(df, a, sessoes, rng)
            r = ef.resumo(a, dias)
            linhas.append({'ativo': ativo[:3], 'stop': nome, 'operacoes': len(a), 'r_mediano': r['r_mediano'],
                           'acerto_1r': r['acerto_1r'], 'aleat_1r': r['aleat_1r'], 'acerto_2r': r['acerto_2r'],
                           'bruto_1r': r['bruto_1r'], 'liq_1r': r['liq_1r'], 'bruto_2r': r['bruto_2r'], 'liq_2r': r['liq_2r'],
                           'taxa_em_r': (ef.TAXA_IDA_VOLTA / a['risco_pct']).median()})
            print(f'  {ativo} {nome}', file=sys.stderr)
    return pd.DataFrame(linhas)


if __name__ == '__main__':
    res = main(sys.argv[1:])
    saida = Path(__file__).parent / 'saida'
    res.to_csv(saida / 'estudo_stop_1m.csv', index=False)
    pd.set_option('display.width', 250)
    pct = '{:.0%}'.format
    print(res.to_string(index=False, formatters={
        'r_mediano': '{:.2%}'.format, 'acerto_1r': pct, 'aleat_1r': pct, 'acerto_2r': pct,
        'bruto_1r': '{:+.2f}R'.format, 'liq_1r': '{:+.2f}R'.format, 'bruto_2r': '{:+.2f}R'.format,
        'liq_2r': '{:+.2f}R'.format, 'taxa_em_r': '{:.2f}R'.format}))
