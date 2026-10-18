"""
Estudo do recuo a favor da tendência com saída na média de 5 (recuo.py), H1 / H4 / D1, desde 2017.

Métricas em R bruto primeiro: acerto (operações com R > 0), ganho e perda médios, R bruto médio, fator de lucro,
R bruto por ano, pior sequência de perda (drawdown em R), líquido com taxa e estabilidade 2017–21 × 2022–26.

Referência aleatória: entradas sorteadas entre os candles com a mesma tendência e o preço do lado "errado" da
média de 5 (qualquer recuo, sem o gatilho), mesma direção, mesmas saídas e mesmo stop; 20 sorteios.

Uso: python estudo_recuo.py BTCUSDT ETHUSDT SOLUSDT ADAUSDT
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

import recuo

TAXA = 0.0007
SORTEIOS = 20
TEMPOS = {'1h': None, '4h': '4h', '1d': '1D'}
SAIDA = Path(__file__).parent / 'saida'


def agregar(df, regra):
    if regra is None:
        return df
    g = df.set_index('time').resample(regra, label='left', closed='left').agg(
        {'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last'}).dropna().reset_index()
    return g


def drawdown(r):
    acum = np.cumsum(r)
    return float((np.maximum.accumulate(np.r_[0, acum])[1:] - acum).max()) if len(r) else 0.0


def aleatorio(ind, base, d, stop, alvo_n, rng):
    acertos, medias = [], []
    p = min(1.0, alvo_n / max(len(base), 1))
    for _ in range(SORTEIOS):
        ops = recuo.simular(ind, base[rng.random(len(base)) < p], d, stop)
        if len(ops):
            acertos.append((ops['r'] > 0).mean())
            medias.append(ops['r'].mean())
    return np.mean(acertos), np.mean(medias)


def main(ativos):
    rng = np.random.default_rng(7)
    linhas = []
    for ativo in ativos:
        d1h = pd.read_pickle(Path(__file__).parent / 'cache' / f'{ativo}_1h.pkl')
        for tf, regra in TEMPOS.items():
            df = agregar(d1h, regra)
            anos = (df['time'].iloc[-1] - df['time'].iloc[200]).days / 365.25
            ano = df['time'].dt.year.to_numpy()
            ind = recuo.indicadores(df)
            for d in (1, -1):
                for tendencia in (True, False):
                    base = recuo.candidatos(ind, d, None, tendencia)
                    for gatilho in recuo.GATILHOS:
                        sinais = recuo.candidatos(ind, d, gatilho, tendencia)
                        for stop in recuo.STOPS:
                            ops = recuo.simular(ind, sinais, d, stop)
                            if len(ops) < 20:
                                continue
                            r = ops['r'].to_numpy()
                            g, p = r[r > 0], r[r <= 0]
                            a1 = ano[ops['i_exec'].to_numpy()] <= 2021
                            al_ac, al_r = aleatorio(ind, base, d, stop, len(sinais), rng)
                            linhas.append({
                                'ativo': ativo[:3], 'tf': tf, 'lado': 'compra' if d == 1 else 'venda',
                                'tendencia': tendencia, 'gatilho': gatilho, 'stop': stop,
                                'ops': len(ops), 'ops_ano': len(ops) / anos, 'risco_pct': ops['risco_pct'].median(),
                                'acerto': (r > 0).mean(), 'aleat_acerto': al_ac,
                                'ganho_medio': g.mean() if len(g) else 0.0, 'perda_media': p.mean() if len(p) else 0.0,
                                'r_bruto': r.mean(), 'aleat_r': al_r,
                                'fator_lucro': g.sum() / -p.sum() if p.sum() < 0 else np.inf,
                                'r_ano': r.sum() / anos, 'dd_r': drawdown(r),
                                'r_liquido': (r - TAXA / ops['risco_pct'].to_numpy()).mean(),
                                'barras': ops['barras'].mean(),
                                'acerto_17_21': (r[a1] > 0).mean(), 'acerto_22_26': (r[~a1] > 0).mean(),
                                'r_17_21': r[a1].mean(), 'r_22_26': r[~a1].mean(),
                            })
            print(f'  {ativo} {tf}', file=sys.stderr)
    res = pd.DataFrame(linhas)
    SAIDA.mkdir(exist_ok=True)
    res.to_csv(SAIDA / 'estudo_recuo.csv', index=False)
    return res


if __name__ == '__main__':
    res = main(sys.argv[1:])
    pd.set_option('display.width', 300)
    pd.set_option('display.max_rows', 1000)
    p = '{:.0%}'.format
    r = '{:+.2f}'.format
    print(res.to_string(index=False, formatters={
        'ops_ano': '{:.0f}'.format, 'risco_pct': '{:.1%}'.format, 'acerto': p, 'aleat_acerto': p,
        'ganho_medio': r, 'perda_media': r, 'r_bruto': r, 'aleat_r': r, 'fator_lucro': '{:.2f}'.format,
        'r_ano': '{:+.1f}'.format, 'dd_r': '{:.1f}'.format, 'r_liquido': r, 'barras': '{:.1f}'.format,
        'acerto_17_21': p, 'acerto_22_26': p, 'r_17_21': r, 'r_22_26': r}))
