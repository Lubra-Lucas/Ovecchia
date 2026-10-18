"""
Exporta os dados do visualizador de trades: um JSON por ativo, com candles, as 12 combinações
e os trades de cada modo de saída. Os sinais (entradas) são os mesmos em todos os modos;
só a saída muda.

Uso: python exportar_visualizador.py 1h BTCUSDT ETHUSDT SOLUSDT ADAUSDT ETHBTC
"""

import json
import sys
from pathlib import Path

import pandas as pd

import estrategia as est
import motor as m
from rodar_grade import CUSTO_LADO, INICIO, KS, MEDIAS, SAIDA, buy_and_hold

MODOS = {
    'virada': ('Virada do sinal', {'saida': 'virada'}),
    'stop_movel': ('Stop móvel', {'saida': 'stop_movel'}),
    'alvo1': ('Alvo 1R', {'saida': 'alvo', 'alvo_r': 1}),
    'alvo2': ('Alvo 2R', {'saida': 'alvo', 'alvo_r': 2}),
    'alvo3': ('Alvo 3R', {'saida': 'alvo', 'alvo_r': 3}),
}
MOTIVOS = ['virada', 'em aberto', 'stop móvel', 'stop', 'alvo 1R', 'alvo 2R', 'alvo 3R']
CAMPOS_RESUMO = ['retorno_total', 'retorno_anual', 'dd_max', 'trades', 'acerto', 'profit_factor',
                 'duracao_media_h', 'resultado_medio_r', 'acerto_1r', 'acerto_2r', 'acerto_3r']


def _num(v):
    return None if pd.isna(v) or v in (float('inf'), float('-inf')) else round(float(v), 5)


def exportar(simbolo, timeframe):
    df = pd.read_pickle(Path(__file__).parent / 'cache' / f'{simbolo}_{timeframe}.pkl')
    df = df[df['time'] >= INICIO].reset_index(drop=True)
    casas = 2 if df['close'].median() > 10 else 6
    candles = [[int(r.open_time // 1000), round(r.open, casas), round(r.high, casas), round(r.low, casas),
                round(r.close, casas)] for r in df.itertuples()]

    combos = {}
    for mm in MEDIAS:
        for k in KS:
            s = est.calcular_sinais(df, mm, k)
            base = None
            saidas, resumos = {}, {}
            for chave, (_, kw) in MODOS.items():
                t = m.extrair_trades(s, CUSTO_LADO, **kw)
                r = m.resumo(t, m.curva_capital(s, t, CUSTO_LADO))
                fech = t[~t['aberto']]
                r['ret_compras'] = (1 + fech.loc[fech['dir'] == 1, 'retorno']).prod() - 1
                r['ret_vendas'] = (1 + fech.loc[fech['dir'] == -1, 'retorno']).prod() - 1
                resumos[chave] = {c: _num(r[c]) for c in CAMPOS_RESUMO + ['ret_compras', 'ret_vendas']}
                # s, preço de saída, retorno líquido, mfe_r, mae_r, motivo, aberto
                saidas[chave] = [[int(x.i_saida), round(x.preco_saida, casas), round(x.retorno, 5),
                                  round(x.mfe_r, 2), round(x.mae_r, 2), MOTIVOS.index(x.motivo), int(x.aberto)]
                                 for x in t.itertuples()]
                if base is None:
                    # e, dir, entrada, stop inicial, acertou 1R, 2R, 3R antes do stop
                    base = [[int(x.i_entrada), int(x.dir), round(x.preco_entrada, casas),
                             round(x.stop_inicial, casas), int(x.acerto_1r), int(x.acerto_2r), int(x.acerto_3r)]
                            for x in t.itertuples()]
            combos[f'{mm}_{k}'] = {'media': mm, 'k': k, 'sinais': base, 'saidas': saidas, 'resumo': resumos}

    dados = {'simbolo': simbolo, 'timeframe': timeframe, 'custo_lado': CUSTO_LADO, 'casas': casas,
             'modos': {c: v[0] for c, v in MODOS.items()}, 'motivos': MOTIVOS,
             'buy_hold': {k: float(v) for k, v in buy_and_hold(df).items()},
             'candles': candles, 'combos': combos}
    destino = SAIDA / 'visualizador' / f'{simbolo}_{timeframe}.json'
    destino.parent.mkdir(parents=True, exist_ok=True)
    destino.write_text(json.dumps(dados, separators=(',', ':')))
    return destino


if __name__ == '__main__':
    tf = sys.argv[1]
    for ativo in sys.argv[2:]:
        d = exportar(ativo, tf)
        print(f'{d}  {d.stat().st_size / 1e6:.1f} MB')
