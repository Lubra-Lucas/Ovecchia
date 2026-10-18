"""
A estratégia de FVG (1º FVG da pernada) em outros tempos gráficos: M1, M5, M15 e H1.

Os candles de M5/M15/H1 são montados a partir do M1 verificado (mesmo período: 10/2023 a 10/2026).
Adaptação: o tamanho da pernada ("R da estrutura") é medido em ATRs do próprio tempo gráfico,
em vez de % do preço, para o filtro valer igual em qualquer tempo gráfico e volatilidade.
Também ajusta a validade da ordem e a espera pela quebra ao tempo gráfico.

Grade por tempo gráfico: pivô (3, 5) × stop (estrutura, 2 × ATR, 3 × ATR) × alvo (1R, 2R, 3R) ×
pernada mínima (0, 3, 5, 8, 12 ATRs) × filtro da média 50 (sem, com).

Uso: python estudo_timeframes.py BTCUSDT ETHUSDT
"""

import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import estrategia as est
import estudo_configs as ec
import smc

# tempo gráfico: (regra de agregação, validade da ordem em candles, espera pela quebra em candles)
TEMPOS = {'1m': (None, 240, 30), '5m': ('5min', 96, 12), '15m': ('15min', 64, 8), '1h': ('1h', 48, 6)}
PIVOS = (3, 5)
STOPS = {'estrutura': None, '2 × ATR': 2, '3 × ATR': 3}
ALVOS = (1, 2, 3)
PERNADA_ATR = (0, 3, 5, 8, 12)
SAIDA = Path(__file__).parent / 'saida'


def agregar(df1m, regra):
    if regra is None:
        return df1m
    g = df1m.set_index('time').resample(regra, label='left', closed='left').agg(
        {'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last'}).dropna().reset_index()
    g['open_time'] = g['time'].astype('int64') // 10**6
    return g


def grupos(ativo):
    df1m = pd.read_pickle(Path(__file__).parent / 'cache' / f'{ativo}_1m.pkl')
    total_anos = (df1m['time'].iloc[-1] - df1m['time'].iloc[0]).days / 365.25
    for tf, (regra, validade, espera) in TEMPOS.items():
        df = agregar(df1m, regra)
        atr = est.calcular_atr(df).to_numpy()
        c = df['close'].to_numpy()
        m50 = df['close'].rolling(50).mean().to_numpy()
        anos = df['time'].dt.year.to_numpy()
        for n in PIVOS:
            e = smc.estrutura(df, n)
            s = smc.montar_setups(df, e, atr, espera_quebra=espera, validade=validade)
            s = smc.primeiro_da_pernada(s).reset_index(drop=True)      # 1º FVG executado da pernada
            d = s['dir'].to_numpy()
            ref = s['i_ativa'].to_numpy() - 1
            s['lado_m50'] = np.where(d == 1, c[ref] > m50[ref], c[ref] < m50[ref])
            s['pernada_atr'] = d * (s['entrada'] - s['stop_estrutura']) / atr[s['i_fvg'].to_numpy()]
            s['r_estrutura'] = d * (s['entrada'] - s['stop_estrutura']) / s['entrada']
            s['ano'] = anos[s['i_exec'].to_numpy()]
            for nome, mult in STOPS.items():
                base = s.copy()
                col = 'stop_estrutura'
                if mult is not None:
                    col = 'stop_atr'
                    base[col] = base['entrada'] - d * mult * atr[base['i_fvg'].to_numpy()]
                a = smc.avaliar(df, base, col)
                print(f'  {ativo} {tf} pivô {n} stop {nome}: {len(a)} operações', file=sys.stderr)
                yield tf, n, nome, a, total_anos


def main(ativos):
    linhas = []
    for ativo in ativos:
        for tf, n, stop, g, anos in grupos(ativo):
            for minimo, media in itertools.product(PERNADA_ATR, (False, True)):
                x = g[g['pernada_atr'] >= minimo]
                if media:
                    x = x[x['lado_m50']]
                for alvo in ALVOS:
                    m = ec.medir(x, alvo, anos)
                    if m:
                        linhas.append({'ativo': ativo, 'tf': tf, 'pivo': n, 'stop': stop, 'alvo': alvo,
                                       'pernada_min_atr': minimo, 'media50': media,
                                       'r_estrutura_mediana': x['r_estrutura'].median(), **m})
    res = pd.DataFrame(linhas)
    SAIDA.mkdir(exist_ok=True)
    res.to_csv(SAIDA / 'estudo_timeframes.csv', index=False)
    return res


if __name__ == '__main__':
    main(sys.argv[1:])
