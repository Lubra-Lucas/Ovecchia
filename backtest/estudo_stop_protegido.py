"""
Stop protegido por outros FVGs da mesma pernada (regra do operador), com seleção honesta.

Entrada: 1º FVG executado da pernada (o mais próximo do preço no pullback).
Stop, variações:
  - 2 × ATR                    (referência)
  - FVG próprio                (além dos 3 candles do FVG de entrada + folga)
  - próximo FVG                (além do próximo FVG válido da pernada, do lado do stop; sem ele, FVG próprio)
  - FVG mais distante          (além do FVG válido mais distante da pernada; sem ele, FVG próprio)
  - próximo FVG, só com 2+     (só opera quando existe outro FVG válido para proteger o stop)
  - média 50                   (além da SMA 50, se estiver do lado do stop; senão FVG próprio)
"Válido" = formado antes da execução, ativado sem ter sido tocado, e nenhum fechamento além da borda de fora
até a execução. Tudo o que é usado é conhecido no momento da entrada.

Uso: python estudo_stop_protegido.py BTCUSDT ETHUSDT
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

import estrategia as est
import estudo_fvg as ef
import smc
from estudo_timeframes import TEMPOS, agregar

FOLGA_ATR = 0.25
TAXA = 0.0007
SAIDA = Path(__file__).parent / 'saida'


def stops_protegidos(df, ops, cands, atr, m50):
    c = df['close'].to_numpy()
    pernadas = {}
    for i, d, base, topo, origem, _ in cands:
        pernadas.setdefault((d, origem), []).append((i, base, topo))
    prox, longe, n_outros = [], [], []
    for x in ops.itertuples():
        d, e, f = x.dir, x.entrada, x.i_exec
        folga = FOLGA_ATR * atr[x.i_fvg]
        niveis = []
        for i, base, topo in pernadas.get((d, x.origem), []):
            if i == x.i_fvg or i >= f:
                continue
            if d == -1 and base > e and not (c[i + 1:f] > topo).any():
                niveis.append((base, topo + folga))
            elif d == 1 and topo < e and not (c[i + 1:f] < base).any():
                niveis.append((topo, base - folga))
        n_outros.append(len(niveis))
        if niveis:
            if d == -1:
                prox.append(min(niveis)[1])
                longe.append(max(n[1] for n in niveis))
            else:
                prox.append(max(niveis)[1])
                longe.append(min(n[1] for n in niveis))
        else:
            prox.append(np.nan)
            longe.append(np.nan)
    ops = ops.copy()
    ops['n_outros'] = n_outros
    ops['stop_proprio'] = ops['stop_fvg']
    ops['stop_prox'] = np.where(np.isnan(prox), ops['stop_fvg'], prox)
    ops['stop_longe'] = np.where(np.isnan(longe), ops['stop_fvg'], longe)
    d = ops['dir'].to_numpy()
    ref = ops['i_exec'].to_numpy() - 1
    mm = m50[ref]
    lado = np.where(d == -1, mm > ops['entrada'], mm < ops['entrada'])
    ops['stop_m50'] = np.where(lado, mm - d * FOLGA_ATR * atr[ops['i_fvg'].to_numpy()], ops['stop_fvg'])
    ops['stop_atr2'] = ops['entrada'] - d * 2 * atr[ops['i_fvg'].to_numpy()]
    return ops


def main(ativos):
    rng = np.random.default_rng(31)
    linhas = []
    for ativo in ativos:
        d1 = pd.read_pickle(Path(__file__).parent / 'cache' / f'{ativo}_1m.pkl')
        anos = (d1['time'].iloc[-1] - d1['time'].iloc[0]).days / 365.25
        for tf, (regra, validade, espera) in TEMPOS.items():
            df = agregar(d1, regra)
            atr = est.calcular_atr(df).to_numpy()
            m50 = df['close'].rolling(50).mean().to_numpy()
            e = smc.estrutura(df, 5)
            cands = []
            s = smc.montar_setups(df, e, atr, espera_quebra=espera, validade=validade, candidatos=cands)
            ops = smc.primeiro_da_pernada(s).reset_index(drop=True)
            ops = stops_protegidos(df, ops, cands, atr, m50)
            sessoes = smc.sessao(df['time'])
            variacoes = {'2 × ATR': ('stop_atr2', None), 'FVG próprio': ('stop_proprio', None),
                         'próximo FVG': ('stop_prox', None), 'FVG mais distante': ('stop_longe', None),
                         'próximo FVG, só com 2+ FVGs': ('stop_prox', 'com_outros'), 'média 50': ('stop_m50', None)}
            for nome, (col, filtro) in variacoes.items():
                base = ops[ops['n_outros'] > 0] if filtro else ops
                a = smc.avaliar(df, base.reset_index(drop=True), col)
                a['sessao'] = sessoes[a['i_exec'].to_numpy()]
                a['aleat_1r'] = ef.aleatorio_1r(df, a, sessoes, rng)
                custo = TAXA / a['risco_pct']
                linhas.append({'ativo': ativo[:3], 'tf': tf, 'stop': nome, 'ops_ano': len(a) / anos,
                               'com_2+_fvgs': (base['n_outros'] > 0).mean(), 'r_mediano': a['risco_pct'].median(),
                               'acerto_1r': a['acerto_1r'].mean(), 'aleat_1r': a['aleat_1r'].mean(),
                               'acerto_2r': a['acerto_2r'].mean(), 'acerto_3r': a['acerto_3r'].mean(),
                               'bruto_1r': a['resultado_1r'].mean(), 'bruto_2r': a['resultado_2r'].mean(),
                               'bruto_3r': a['resultado_3r'].mean(), 'liq_2r': (a['resultado_2r'] - custo).mean(),
                               'liq_3r': (a['resultado_3r'] - custo).mean()})
                print(f'  {ativo} {tf} {nome}: {len(a)}', file=sys.stderr)
    res = pd.DataFrame(linhas)
    SAIDA.mkdir(exist_ok=True)
    res.to_csv(SAIDA / 'estudo_stop_protegido.csv', index=False)
    return res


if __name__ == '__main__':
    res = main(sys.argv[1:])
    pd.set_option('display.width', 260)
    p = '{:.0%}'.format
    r = '{:+.2f}'.format
    print(res.to_string(index=False, formatters={'ops_ano': '{:.0f}'.format, 'com_2+_fvgs': p, 'r_mediano': '{:.2%}'.format,
                                                 'acerto_1r': p, 'aleat_1r': p, 'acerto_2r': p, 'acerto_3r': p,
                                                 'bruto_1r': r, 'bruto_2r': r, 'bruto_3r': r, 'liq_2r': r, 'liq_3r': r}))
