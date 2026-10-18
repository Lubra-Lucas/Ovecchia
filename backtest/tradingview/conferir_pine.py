"""
Confere a lógica do indicador ovelha_fvg.pine: replica, candle a candle e na mesma ordem de passos,
o que o Pine Script faz, e compara as operações com as do backtest (smc.py) nos mesmos candles.

Uso: python tradingview/conferir_pine.py BTCUSDT 60000
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import estrategia as est  # noqa: E402
import smc  # noqa: E402

N, MIN_PCT, VALIDADE, ESPERA, MULT, FOLGA, ALVO, HORIZ = 5, 0.4, 240, 30, 2.0, 0.1, 3, 1440


def simular_pine(df, causal=False):
    """causal=True: só executa o FVG mais antigo ainda armado da pernada (regra sem olhar o futuro)."""
    o, h, l, c = (df[x].to_numpy() for x in ('open', 'high', 'low', 'close'))
    atr = est.calcular_atr(df).to_numpy()
    topo, fundo = smc.pivos(c, N)
    trend, orig, ext = 0, None, np.nan
    nc = ic = nb = ib = None
    ultima = -1
    ordens, pend, trades, usadas, feitas = [], [], [], set(), []
    for t in range(len(c)):
        # 1) operações abertas
        for tr in list(trades):
            d, e, st, r = tr['d'], tr['e'], tr['st'], tr['r']
            if (l[t] <= st) if d == 1 else (h[t] >= st):
                px = min(o[t], st) if d == 1 else max(o[t], st)
                if tr['aberta']:
                    tr['res'] = d * (px - e) / r
                trades.remove(tr); feitas.append(tr); continue
            for x in (1, 2, 3):
                if (h[t] >= e + d * x * r) if d == 1 else (l[t] <= e + d * x * r):
                    tr[f'h{x}'] = True
            if tr['aberta'] and tr[f'h{ALVO}']:
                tr['res'], tr['aberta'] = float(ALVO), False
            fim = tr['h3'] or t - tr['f'] >= HORIZ
            if t - tr['f'] >= HORIZ and tr['aberta']:
                tr['res'] = d * (c[t] - e) / r
            if fim:
                trades.remove(tr); feitas.append(tr)
        # 2) execução das ordens
        for od in sorted(ordens, key=lambda x: x['fvg']):
            if od not in ordens:
                continue
            d, e = od['d'], od['e']
            if (h[t] >= e) if d == -1 else (l[t] <= e):
                chave = (od['orig'], d)
                mais_antigo_armado = any(x is not od and x['orig'] == od['orig'] and x['d'] == d and x['fvg'] < od['fvg']
                                         for x in ordens)
                if causal and mais_antigo_armado:
                    ordens.remove(od)
                    continue
                if chave not in usadas:
                    usadas.add(chave)
                    r_estr = d * (e - od['stopE']) / e
                    stop = e - d * MULT * od['atr']
                    r = d * (e - stop)
                    if r_estr * 100 >= MIN_PCT and r > 0:
                        tr = {'d': d, 'e': e, 'st': stop, 'r': r, 'f': t, 'h1': False, 'h2': False, 'h3': False,
                              'aberta': True, 'res': None, 'fvg': od['fvg']}
                        if (l[t] <= stop) if d == 1 else (h[t] >= stop):
                            tr['res'] = -1.0; feitas.append(tr)
                        else:
                            trades.append(tr)
                ordens.remove(od)
        # 3) estrutura
        ev = 0
        k = t - N
        if k > ultima:
            if topo[k] and trend != -1:
                nc, ic = c[k], k
            if fundo[k] and trend != 1:
                nb, ib = c[k], k
        if nb is not None and c[t] < nb:
            ev = -1 if trend == -1 else -2
            j = ib + 1 + int(np.argmax(c[ib + 1:t + 1]))
            orig, ext = j, h[ib + 1:t + 1].max()
            trend, ultima, nc, ic, nb, ib = -1, t, c[j], j, None, None
        elif nc is not None and c[t] > nc:
            ev = 1 if trend == 1 else 2
            j = ic + 1 + int(np.argmin(c[ic + 1:t + 1]))
            orig, ext = j, l[ic + 1:t + 1].min()
            trend, ultima, nb, ib, nc, ic = 1, t, c[j], j, None, None
        # 4) cancelamento
        for od in list(ordens):
            d = od['d']
            atravessou = c[t] > od['topo'] if d == -1 else c[t] < od['base']
            if atravessou or trend != d or t >= od['ini'] + VALIDADE - 1:
                ordens.remove(od)

        def armar(d, e, tp, bs, a, fvg):
            stopE = ext + FOLGA * a if d == -1 else ext - FOLGA * a
            ordens.append({'d': d, 'e': e, 'topo': tp, 'base': bs, 'stopE': stopE, 'atr': a, 'orig': orig,
                           'ini': t + 1, 'fvg': fvg})
        # 5) FVG esperando a quebra
        for p in list(pend):
            d, e, i = p['d'], p['e'], p['i']
            remover = ((h[t] >= e) if d == -1 else (l[t] <= e)) or t > i + ESPERA
            if not remover and ev != 0:
                if np.sign(ev) == d and orig <= i - 2:
                    armar(d, e, p['topo'], p['base'], p['atr'], i)
                remover = True
            if remover:
                pend.remove(p)
        # 6) novos FVGs
        if t >= 2 and not np.isnan(atr[t]):
            for d, tp, bs in ((-1, l[t - 2], h[t]), (1, l[t], h[t - 2])):
                if (d == -1 and l[t - 2] > h[t]) or (d == 1 and h[t - 2] < l[t]):
                    e = bs if d == -1 else tp
                    if trend == d and orig is not None and orig <= t - 2:
                        armar(d, e, tp, bs, atr[t], t)
                    else:
                        pend.append({'d': d, 'e': e, 'topo': tp, 'base': bs, 'i': t, 'atr': atr[t]})
    return pd.DataFrame(feitas)


def backtest(df):
    atr = est.calcular_atr(df).to_numpy()
    e = smc.estrutura(df, N)
    s = smc.montar_setups(df, e, atr, espera_quebra=ESPERA, validade=VALIDADE)
    s = smc.primeiro_da_pernada(s).copy()
    s = s[s['dir'] * (s['entrada'] - s['stop_estrutura']) / s['entrada'] >= MIN_PCT / 100]
    s['stop_atr'] = s['entrada'] - s['dir'] * MULT * atr[s['i_fvg'].to_numpy()]
    return smc.avaliar(df, s.reset_index(drop=True), 'stop_atr', HORIZ)


if __name__ == '__main__':
    ativo, n = sys.argv[1], int(sys.argv[2])
    df = pd.read_pickle(Path(__file__).resolve().parent.parent / 'cache' / f'{ativo}_1m.pkl').iloc[-n:].reset_index(drop=True)
    b = backtest(df)
    b = b[b['i_exec'] < len(df) - HORIZ]
    for causal in (False, True):
        p = simular_pine(df, causal)
        p = p[p['f'] < len(df) - HORIZ]
        nome = 'Pine, 1º executado no tempo' if not causal else 'Pine, FVG mais antigo armado'
        ep, eb = set(zip(p['f'], p['d'], p['e'].round(6))), set(zip(b['i_exec'], b['dir'], b['entrada'].round(6)))
        print(f'{nome:<30}: {len(p):>4} operações | acerto 1R {p.h1.mean():.1%} 2R {p.h2.mean():.1%} 3R {p.h3.mean():.1%} | '
              f'R médio alvo {ALVO}R {p.res.mean():+.3f} | iguais ao backtest {len(ep & eb)}, só aqui {len(ep - eb)}, só no backtest {len(eb - ep)}')
    sys.exit()
    p = p[p['f'] < len(df) - HORIZ]
    b = b[b['i_exec'] < len(df) - HORIZ]
    print(f'Pine (simulado): {len(p)} operações | acerto 1R {p.h1.mean():.1%} 2R {p.h2.mean():.1%} 3R {p.h3.mean():.1%} | '
          f'R médio alvo {ALVO}R {p.res.mean():+.3f}')
    print(f'Backtest:        {len(b)} operações | acerto 1R {b.acerto_1r.mean():.1%} 2R {b.acerto_2r.mean():.1%} '
          f'3R {b.acerto_3r.mean():.1%} | R médio alvo {ALVO}R {b[f"resultado_{ALVO}r"].mean():+.3f}')
    ep, eb = set(zip(p['f'], p['d'], p['e'].round(6))), set(zip(b['i_exec'], b['dir'], b['entrada'].round(6)))
    print(f'Operações iguais (mesmo candle, direção e preço): {len(ep & eb)} | só no Pine: {len(ep - eb)} | só no backtest: {len(eb - ep)}')
