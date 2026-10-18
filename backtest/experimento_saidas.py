"""
Experimento de saídas e assertividade em R.

Parte A: assertividade (chegou a 1R/2R/3R antes do stop inicial) por tipo de stop,
         comparada com entradas aleatórias com o mesmo risco, direção e janela de tempo.
Parte B: resultado de cada modo de saída (virada, stop móvel, alvo fixo × tamanho de stop).

Uso: python experimento_saidas.py 1h BTCUSDT ETHUSDT SOLUSDT ADAUSDT ETHBTC
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

import estrategia as est
import motor as m
from rodar_grade import INICIO, KS, MEDIAS

CUSTO = 0.0005
STOPS = [None, 1.0, 2.0, 3.0]                  # None = faixa oposta
SORTEIOS = 10


def nome_stop(s):
    return 'faixa' if s is None else f'{s:g} ATR'


def acerto_aleatorio(df, trades, rng):
    """Mesma direção, mesmo risco (%), mesma janela; entrada em um candle sorteado."""
    ab, mx, mn = df['open'].to_numpy(), df['high'].to_numpy(), df['low'].to_numpy()
    n = len(df)
    acertos = {x: [] for x in m.ALVOS_R}
    fechados = trades[~trades['aberto']]
    for t in fechados.itertuples():
        janela = max(t.i_saida - t.i_entrada, 1)
        for ini in rng.integers(0, n - janela - 1, SORTEIOS):
            entrada = ab[ini]
            jmx, jmn = mx[ini:ini + janela], mn[ini:ini + janela]
            stop = entrada * (1 - t.dir * t.risco_pct)
            toque_stop = m._primeiro(jmn <= stop) if t.dir == 1 else m._primeiro(jmx >= stop)
            for x in m.ALVOS_R:
                alvo = entrada * (1 + t.dir * x * t.risco_pct)
                toque = m._primeiro(jmx >= alvo) if t.dir == 1 else m._primeiro(jmn <= alvo)
                acertos[x].append(toque < toque_stop)
    return {x: np.mean(v) for x, v in acertos.items()}


def main(timeframe, ativos):
    rng = np.random.default_rng(7)
    linhas_a, linhas_b = [], []
    for ativo in ativos:
        df = pd.read_pickle(Path(__file__).parent / 'cache' / f'{ativo}_{timeframe}.pkl')
        df = df[df['time'] >= INICIO].reset_index(drop=True)
        sinais = {(mm, k): est.calcular_sinais(df, mm, k) for mm in MEDIAS for k in KS}
        print(f'{ativo}...', file=sys.stderr)

        # ---- Parte A: assertividade por tipo de stop ----
        for stop in STOPS:
            est_r, ale_r = {x: [] for x in m.ALVOS_R}, {x: [] for x in m.ALVOS_R}
            for s in sinais.values():
                t = m.extrair_trades(s, CUSTO, saida='virada', stop_atr=stop)
                r = m.resumo(t, m.curva_capital(s, t, CUSTO))
                a = acerto_aleatorio(df, t, rng)
                for x in m.ALVOS_R:
                    est_r[x].append(r[f'acerto_{x}r'])
                    ale_r[x].append(a[x])
            linha = {'ativo': ativo, 'stop': nome_stop(stop)}
            for x in m.ALVOS_R:
                linha[f'{x}R estrat'] = np.median(est_r[x])
                linha[f'{x}R aleat'] = np.median(ale_r[x])
            linhas_a.append(linha)

        # ---- Parte B: modos de saída ----
        modos = [('virada', {}, 'faixa'), ('stop_movel', {}, 'faixa')]
        modos += [('alvo', {'alvo_r': a, 'stop_atr': s}, nome_stop(s)) for s in STOPS for a in m.ALVOS_R]
        for modo, kw, stop_nome in modos:
            rets, dds, acs, rmed = [], [], [], []
            for s in sinais.values():
                t = m.extrair_trades(s, CUSTO, saida=modo, **kw)
                eq = m.curva_capital(s, t, CUSTO)
                r = m.resumo(t, eq)
                rets.append(r['retorno_anual']); dds.append(r['dd_max'])
                acs.append(r['acerto']); rmed.append(r['resultado_medio_r'])
            rotulo = {'virada': 'Virada do sinal', 'stop_movel': 'Stop móvel'}.get(modo, f"Alvo {kw.get('alvo_r')}R")
            linhas_b.append({'ativo': ativo, 'saida': rotulo, 'stop': stop_nome,
                             'ret_anual_med': np.median(rets), 'dd_med': np.median(dds),
                             'positivas': f'{sum(r > 0 for r in rets)}/12',
                             'acerto_med': np.median(acs), 'resultado_r_med': np.median(rmed)})
    return pd.DataFrame(linhas_a), pd.DataFrame(linhas_b)


if __name__ == '__main__':
    a, b = main(sys.argv[1], sys.argv[2:])
    saida = Path(__file__).parent / 'saida'
    saida.mkdir(exist_ok=True)
    a.to_csv(saida / f'assertividade_{sys.argv[1]}.csv', index=False)
    b.to_csv(saida / f'modos_saida_{sys.argv[1]}.csv', index=False)
    pd.set_option('display.width', 250)
    print('A) Assertividade: % dos sinais que chegam a XR antes do stop inicial (mediana das 12 combinações)')
    print(a.to_string(index=False, formatters={c: '{:.0%}'.format for c in a.columns if 'R ' in c}))
    print('\nB) Resultado por modo de saída (mediana das 12 combinações, custo 0,05%/lado)')
    print(b.to_string(index=False, formatters={'ret_anual_med': '{:+.0%}'.format, 'dd_med': '{:+.0%}'.format,
                                               'acerto_med': '{:.0%}'.format, 'resultado_r_med': '{:+.2f}R'.format}))
