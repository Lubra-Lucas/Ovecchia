"""
Estudo da estratégia de FVG com estrutura pelos fechamentos (M1).

Para cada ativo, tamanho de pivô e tipo de stop, mede a assertividade (1R/2R/3R antes do stop),
comparada com entradas aleatórias de mesmo risco, direção e sessão, e o resultado em R
bruto e com taxas. Os setups são fatiados pelos filtros do operador.

Uso: python estudo_fvg.py BTCUSDT ETHUSDT
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

import estrategia as est
import smc

PIVOS = (2, 3, 5)
STOPS = ('stop_estrutura', 'stop_fvg')
TAXA_IDA_VOLTA = 0.0007        # entrada limitada 0,02% + saída a mercado 0,05%
HORIZONTE = 1440
SAIDA = Path(__file__).parent / 'saida'


def aleatorio_1r(df, linhas, sessoes, rng):
    """Uma entrada sorteada por setup: mesma sessão, direção e risco (%). Mesmas regras de avaliação."""
    o, h, l = df['open'].to_numpy(), df['high'].to_numpy(), df['low'].to_numpy()
    N = len(o)
    pools = {s: np.flatnonzero(sessoes == s) for s in np.unique(sessoes)}
    pools = {s: p[p < N - HORIZONTE - 1] for s, p in pools.items()}
    out = np.zeros(len(linhas), bool)
    for k, (d, r_pct, ses) in enumerate(zip(linhas['dir'], linhas['risco_pct'], linhas['sessao'])):
        r = rng.choice(pools[ses])
        e = o[r]
        stop, alvo = e * (1 - d * r_pct), e * (1 + d * r_pct)
        hh, ll = h[r:r + HORIZONTE], l[r:r + HORIZONTE]
        bs = (hh >= stop) if d == -1 else (ll <= stop)
        ba = (ll[1:] <= alvo) if d == -1 else (hh[1:] >= alvo)
        ts = np.argmax(bs) if bs.any() else np.inf
        ta = 1 + np.argmax(ba) if ba.any() else np.inf
        out[k] = ta < ts
    return out


def uma_por_vez(linhas):
    """Simula uma operação por vez (alvo 1R): ignora setups executados enquanto a anterior está aberta."""
    linhas = linhas.sort_values('i_exec')
    livre, manter = -1, []
    for idx, f, ts, ta in zip(linhas.index, linhas['i_exec'], linhas['t_stop'], linhas['t_1r']):
        if f < livre:
            continue
        manter.append(idx)
        fim = min(ts, ta)
        livre = f + (fim if fim != np.inf else HORIZONTE)
    return linhas.loc[manter]


def resumo(linhas, dias):
    if not len(linhas):
        return {'setups': 0}
    liq = lambda col: (linhas[col] - TAXA_IDA_VOLTA / linhas['risco_pct']).mean()
    return {
        'setups': len(linhas),
        'por_dia': len(linhas) / dias,
        'acerto_1r': linhas['acerto_1r'].mean(),
        'aleat_1r': linhas['aleat_1r'].mean(),
        'acerto_2r': linhas['acerto_2r'].mean(),
        'acerto_3r': linhas['acerto_3r'].mean(),
        'r_mediano': linhas['risco_pct'].median(),
        'bruto_1r': linhas['resultado_1r'].mean(),
        'liq_1r': liq('resultado_1r'),
        'bruto_2r': linhas['resultado_2r'].mean(),
        'liq_2r': liq('resultado_2r'),
    }


def preparar(ativo, rng):
    df = pd.read_pickle(Path(__file__).parent / 'cache' / f'{ativo}_1m.pkl')
    atr = est.calcular_atr(df).to_numpy()
    c = df['close']
    m13, m26, m50 = (c.rolling(p).mean().to_numpy() for p in (13, 26, 50))
    sessoes = smc.sessao(df['time'])
    dias = (df['time'].iloc[-1] - df['time'].iloc[0]).days
    tabelas = []
    for n in PIVOS:
        e = smc.estrutura(df, n)
        setups = smc.montar_setups(df, e, atr)
        for stop in STOPS:
            a = smc.avaliar(df, setups, stop, HORIZONTE)
            ref = a['i_ativa'].to_numpy() - 1
            d = a['dir'].to_numpy()
            a['sessao'] = sessoes[a['i_exec'].to_numpy()]
            a['medias_alinhadas'] = np.where(d == -1, (m13[ref] < m26[ref]) & (m26[ref] < m50[ref]),
                                             (m13[ref] > m26[ref]) & (m26[ref] > m50[ref]))
            a['lado_da_m50'] = np.where(d == -1, c.to_numpy()[ref] < m50[ref], c.to_numpy()[ref] > m50[ref])
            a['premium'] = a['posicao_perna'] > 0.5
            a['primeiro_da_perna'] = a.index.isin(smc.primeiro_da_pernada(a).index)
            a['aleat_1r'] = aleatorio_1r(df, a, sessoes, rng)
            a['ativo'], a['pivo'], a['stop'] = ativo, n, stop
            tabelas.append(a)
            print(f'  {ativo} pivô {n} {stop}: {len(a)} setups', file=sys.stderr)
    return pd.concat(tabelas, ignore_index=True), dias


def main(ativos):
    rng = np.random.default_rng(11)
    SAIDA.mkdir(exist_ok=True)
    todos, dias = [], {}
    for ativo in ativos:
        t, dias[ativo] = preparar(ativo, rng)
        todos.append(t)
    base = pd.concat(todos, ignore_index=True)
    base.to_pickle(SAIDA / 'setups_fvg_1m.pkl')

    filtros = {
        'todos': lambda x: x,
        'premium (>50% da pernada)': lambda x: x[x.premium],
        'discount (<50%)': lambda x: x[~x.premium],
        'FVG do impulso da quebra': lambda x: x[x.via_quebra],
        'médias 13/26/50 alinhadas': lambda x: x[x.medias_alinhadas],
        'sessões (3h Ásia/Londres/NY)': lambda x: x[x.sessao != 'fora'],
        'fora das sessões': lambda x: x[x.sessao == 'fora'],
        'primeiro FVG da pernada': lambda x: x[x.primeiro_da_perna],
        'R ≥ 0,2%': lambda x: x[x.risco_pct >= 0.002],
        'R ≥ 0,4%': lambda x: x[x.risco_pct >= 0.004],
        'COMBINADO: premium + médias + sessões': lambda x: x[x.premium & x.medias_alinhadas & (x.sessao != 'fora')],
        'COMBINADO + R ≥ 0,2%': lambda x: x[x.premium & x.medias_alinhadas & (x.sessao != 'fora') & (x.risco_pct >= 0.002)],
    }
    linhas = []
    for (ativo, n, stop), g in base.groupby(['ativo', 'pivo', 'stop']):
        for nome, f in filtros.items():
            sel = f(g)
            linhas.append({'ativo': ativo, 'pivo': n, 'stop': stop.replace('stop_', ''), 'filtro': nome,
                           **resumo(sel, dias[ativo]),
                           **{f'1x1_{k}': v for k, v in resumo(uma_por_vez(sel), dias[ativo]).items()
                              if k in ('setups', 'acerto_1r', 'liq_1r')}})
    res = pd.DataFrame(linhas)
    res.to_csv(SAIDA / 'estudo_fvg_1m.csv', index=False)

    pd.set_option('display.width', 260)
    pct = '{:.0%}'.format
    fmt = {'por_dia': '{:.1f}'.format, 'acerto_1r': pct, 'aleat_1r': pct, 'acerto_2r': pct, 'acerto_3r': pct,
           'r_mediano': '{:.2%}'.format, 'bruto_1r': '{:+.2f}R'.format, 'liq_1r': '{:+.2f}R'.format,
           'bruto_2r': '{:+.2f}R'.format, 'liq_2r': '{:+.2f}R'.format, '1x1_acerto_1r': pct,
           '1x1_liq_1r': '{:+.2f}R'.format}
    for (ativo, n, stop), g in res.groupby(['ativo', 'pivo', 'stop']):
        print(f'\n=== {ativo} · pivô {n} · stop {stop} ===')
        print(g.drop(columns=['ativo', 'pivo', 'stop']).to_string(index=False, formatters=fmt))


if __name__ == '__main__':
    main(sys.argv[1:])
