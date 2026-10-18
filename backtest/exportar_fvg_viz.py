"""
Exporta os dados do visualizador de operações de FVG (M1).

Três variantes de stop para as mesmas entradas: estrutura (além do extremo da pernada), 2 × ATR(14) e 3 × ATR(14).
Sessões no horário de São Paulo (smc.sessao_sao_paulo).
A estrutura e os setups são calculados no histórico inteiro (para o estado da estrutura ser o
mesmo do estudo); só os últimos meses vão para a página. Candles saem em um arquivo por mês,
carregado sob demanda.

Uso: python exportar_fvg_viz.py 2026-04-01 BTCUSDT ETHUSDT
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import estrategia as est
import smc
from estudo_stop_protegido import stops_protegidos

PIVO = 5
STOP = 'stop_estrutura'
SESSOES = ['fora', 'Ásia', 'Londres', 'Nova York']
SAIDA = Path(__file__).parent / 'saida' / 'visualizador_fvg' / 'dados'


def _t(df, i):
    return int(df['open_time'].iat[i] // 1000)


def exportar(ativo, inicio):
    df = pd.read_pickle(Path(__file__).parent / 'cache' / f'{ativo}_1m.pkl')
    passo = df['open_time'].diff().iloc[1:]
    assert (passo == 60_000).all(), 'há buracos no M1; o formato compacto assume candles contínuos'
    atr = est.calcular_atr(df).to_numpy()
    e = smc.estrutura(df, PIVO)
    cands = []
    setups = smc.montar_setups(df, e, atr, candidatos=cands)
    setups = smc.primeiro_da_pernada(setups).reset_index(drop=True)      # 1º FVG executado (no tempo) da pernada
    # stop protegido além do próximo FVG válido da pernada e quantos outros FVGs válidos a pernada tinha na entrada
    setups = stops_protegidos(df, setups, cands, atr, df['close'].rolling(50).mean().to_numpy())
    sessoes = smc.sessao_sao_paulo(df['time'])

    i0 = int(df['time'].searchsorted(pd.Timestamp(inicio, tz='UTC')))
    destino = SAIDA / ativo
    destino.mkdir(parents=True, exist_ok=True)
    casas = 2

    # ---- candles e eventos de estrutura, um arquivo por mês ----
    sub = df.iloc[i0:]
    meses = []
    for mes, g in sub.groupby(sub['time'].dt.strftime('%Y-%m')):
        a, b = g.index[0], g.index[-1]
        ev_idx = np.flatnonzero(e['evento'][a:b + 1]) + a
        eventos = [[_t(df, t), int(e['evento'][t]), round(float(e['nivel_rompido'][t]), casas),
                    _t(df, int(e['idx_rompido'][t])), _t(df, int(e['origem_idx'][t])),
                    round(float(df['close'].iat[int(e['origem_idx'][t])]), casas)] for t in ev_idx]
        dados = {'t0': _t(df, a), 'o': g['open'].round(casas).tolist(), 'h': g['high'].round(casas).tolist(),
                 'l': g['low'].round(casas).tolist(), 'c': g['close'].round(casas).tolist(), 'eventos': eventos}
        (destino / f'{mes}.json').write_text(json.dumps(dados, separators=(',', ':')))
        meses.append(mes)

    # ---- operações: mesmas entradas, duas variantes de stop ----
    sel = setups[setups['i_exec'] >= i0].copy()          # setups ainda não avaliados: cada variante é avaliada do zero
    for mult in (2, 3):
        sel[f'stop_atr{mult}'] = sel['entrada'] - sel['dir'] * mult * atr[sel['i_fvg'].to_numpy()]
    r_estrutura = (sel['dir'] * (sel['entrada'] - sel['stop_estrutura']) / sel['entrada']).to_numpy()
    h, l = df['high'].to_numpy(), df['low'].to_numpy()
    inf = lambda v: -1 if v == np.inf else int(v)
    variantes = {}
    for nome, col in (('estrutura', 'stop_estrutura'), ('atr2', 'stop_atr2'), ('atr3', 'stop_atr3'), ('prox', 'stop_prox')):
        av = smc.avaliar(df, sel.reset_index(drop=True), col)
        assert len(av) == len(sel)
        ops = []
        for k, s in enumerate(av.itertuples()):
            if s.dir == -1:
                topo, fundo = s.extremo_perna, l[s.origem:s.i_fvg + 1].min()
            else:
                topo, fundo = h[s.origem:s.i_fvg + 1].max(), s.extremo_perna
            ops.append([_t(df, s.i_fvg - 2), _t(df, s.i_exec), int(s.dir), round(s.entrada, casas),
                        round(getattr(s, col), casas), round(s.zona_base, casas), round(s.zona_topo, casas),
                        round(float(s.posicao_perna), 3), round(float(s.risco_pct), 5),
                        int(s.acerto_1r), int(s.acerto_2r), int(s.acerto_3r),
                        inf(s.t_stop), inf(s.t_1r), inf(s.t_2r), inf(s.t_3r),
                        round(float(s.resultado_1r), 3), round(float(s.resultado_2r), 3), round(float(s.resultado_3r), 3),
                        SESSOES.index(sessoes[s.i_exec]), _t(df, s.origem), round(float(topo), casas),
                        round(float(fundo), casas), round(float(r_estrutura[k]), 5), int(s.n_outros)])
        variantes[nome] = ops
    resumo = {'ativo': ativo, 'pivo': PIVO, 'meses': meses, 'casas': casas,
              'inicio': _t(df, i0), 'fim': _t(df, len(df) - 1), 'sessoes': SESSOES,
              'campos': ['t_fvg', 't_exec', 'dir', 'entrada', 'stop', 'zona_base', 'zona_topo', 'posicao',
                         'risco_pct', 'a1', 'a2', 'a3', 't_stop', 't_1r', 't_2r', 't_3r', 'resultado_1r',
                         'resultado_2r', 'resultado_3r', 'sessao', 't_origem', 'topo_perna', 'fundo_perna', 'r_estrutura', 'n_outros'],
              'variantes': variantes}
    (destino / 'operacoes.json').write_text(json.dumps(resumo, separators=(',', ':')))
    tam = sum(f.stat().st_size for f in destino.iterdir()) / 1e6
    print(f'{ativo}: {len(meses)} meses, {len(sel)} operações, {tam:.1f} MB')


if __name__ == '__main__':
    for a in sys.argv[2:]:
        exportar(a, sys.argv[1])
