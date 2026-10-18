"""
Grade de configurações da estratégia de FVG no M1 (BTC e ETH), para achar a mais vencedora
nos dois ativos ao mesmo tempo.

Dimensões:
  pivô           3, 5
  entrada        1º FVG da pernada | 1º FVG + média 50 + 2 BOS | todos os FVGs + média 50 + 2 BOS |
                 todos + médias 13/26/50 alinhadas + 2 BOS | todos os FVGs (referência)
  R estrutura ≥  0; 0,2%; 0,4%
  stop           estrutura | 2 × ATR | 3 × ATR
  alvo           1R, 2R, 3R
  horário        todas as horas | aberturas (São Paulo: Ásia 22h–01h, Londres 03h–07h, NY 10h30–13h30)

"1º FVG" = o primeiro FVG executado da pernada; se ele não passar nos filtros, a pernada não é operada
(os FVGs seguintes da mesma pernada perdem dinheiro: 42% de acerto em 1R, −0,17R por operação).
"Média 50" = fechamento do lado da operação em relação à SMA 50 (acima na compra, abaixo na venda).
"2 BOS" = pelo menos dois BOS seguidos na tendência quando a ordem é ativada.
Métricas: R bruto acumulado por ano (todas as operações e uma por vez), R por operação,
acerto no alvo, R líquido (0,02% + 0,05%) e a estabilidade entre 2023–2024 e 2025–2026.

Uso: python estudo_configs.py BTCUSDT ETHUSDT
"""

import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import estrategia as est
import smc

TAXA = 0.0007
PIVOS = (3, 5)
STOPS = {'estrutura': None, '2 × ATR': 2, '3 × ATR': 3}
ALVOS = (1, 2, 3)
R_MIN = (0.0, 0.002, 0.004)
HORARIOS = ('todas', 'aberturas')
ENTRADAS = ('1º FVG', '1º FVG + média 50 + 2 BOS', 'todos + média 50 + 2 BOS',
            'todos + médias alinhadas + 2 BOS', 'todos os FVGs')
SAIDA = Path(__file__).parent / 'saida'


def aberturas_sao_paulo(tempos):
    """Ásia 22h–01h, Londres 03h–07h, Nova York 10h30–13h30, horário de São Paulo, dias úteis."""
    t = pd.DatetimeIndex(tempos).tz_convert('America/Sao_Paulo')
    m = t.hour * 60 + t.minute
    dia = t.dayofweek                              # 0 = segunda ... 6 = domingo
    asia = ((m >= 22 * 60) & ((dia <= 3) | (dia == 6))) | ((m < 60) & (dia <= 4))
    londres = (m >= 3 * 60) & (m < 7 * 60) & (dia <= 4)
    ny = (m >= 10 * 60 + 30) & (m < 13 * 60 + 30) & (dia <= 4)
    return np.asarray(asia | londres | ny)


def uma_por_vez(execs, duracoes):
    """Índices das operações aceitas operando uma de cada vez (ordem de execução)."""
    ordem = np.argsort(execs, kind='stable')
    livre, aceitas = -1, []
    for k in ordem:
        if execs[k] < livre:
            continue
        aceitas.append(k)
        livre = execs[k] + duracoes[k]
    return np.array(aceitas, dtype=int)


COLUNAS = ['dir', 'origem', 'i_exec', 'risco_pct', 't_stop', 'lado_m50', 'alinhadas', 'bos2', 'abertura',
           'r_estrutura', 'ano'] + [f'{p}_{x}r' for p in ('resultado', 'acerto', 't') for x in ALVOS]


def grupos(ativo):
    """Gera (pivô, stop, setups avaliados, anos) para um ativo, um grupo por vez."""
    df = pd.read_pickle(Path(__file__).parent / 'cache' / f'{ativo}_1m.pkl')
    atr = est.calcular_atr(df).to_numpy()
    c = df['close'].to_numpy()
    m13, m26, m50 = (df['close'].rolling(p).mean().to_numpy() for p in (13, 26, 50))
    abertura = aberturas_sao_paulo(df['time'])
    anos = df['time'].dt.year.to_numpy()
    total_anos = (df['time'].iloc[-1] - df['time'].iloc[0]).days / 365.25
    for n in PIVOS:
        e = smc.estrutura(df, n)
        s = smc.montar_setups(df, e, atr)
        ref = s['i_ativa'].to_numpy() - 1
        d = s['dir'].to_numpy()
        s['lado_m50'] = np.where(d == 1, c[ref] > m50[ref], c[ref] < m50[ref])
        s['alinhadas'] = np.where(d == 1, (m13[ref] > m26[ref]) & (m26[ref] > m50[ref]),
                                  (m13[ref] < m26[ref]) & (m26[ref] < m50[ref]))
        s['bos2'] = e['seq_bos'][s['t_ref'].to_numpy()] >= 2
        s['abertura'] = abertura[s['i_exec'].to_numpy()]
        s['r_estrutura'] = d * (s['entrada'] - s['stop_estrutura']) / s['entrada']
        s['ano'] = anos[s['i_exec'].to_numpy()]
        for nome, mult in STOPS.items():
            base = s.copy()
            col = 'stop_estrutura'
            if mult is not None:
                col = 'stop_atr'
                base[col] = base['entrada'] - d * mult * atr[base['i_fvg'].to_numpy()]
            a = smc.avaliar(df, base, col)[COLUNAS]
            print(f'  {ativo} pivô {n} stop {nome}: {len(a)} setups', file=sys.stderr)
            yield n, nome, a, total_anos


def selecionar(g, entrada):
    """g já vem com o filtro de R e de horário. Nas entradas '1º FVG', g já contém só o 1º FVG da pernada."""
    if entrada == '1º FVG':
        return g
    if entrada == '1º FVG + média 50 + 2 BOS':
        return g[g['lado_m50'] & g['bos2']]
    if entrada == 'todos + média 50 + 2 BOS':
        return g[g['lado_m50'] & g['bos2']]
    if entrada == 'todos + médias alinhadas + 2 BOS':
        return g[g['alinhadas'] & g['bos2']]
    return g


def medir(x, alvo, anos):
    if len(x) < 30:
        return None
    res = x[f'resultado_{alvo}r'].to_numpy()
    liq = res - TAXA / x['risco_pct'].to_numpy()
    t_alvo = x[f't_{alvo}r'].to_numpy()
    t_stop = x['t_stop'].to_numpy()
    dur = np.minimum(t_alvo, t_stop)
    dur = np.where(np.isinf(dur), 1440, dur).astype(int)
    seq = uma_por_vez(x['i_exec'].to_numpy(), dur)
    metade1 = x['ano'].to_numpy() <= 2024
    return {
        'ops_ano': len(x) / anos,
        'acerto': x[f'acerto_{alvo}r'].mean(),
        'r_por_op': res.mean(),
        'liq_por_op': liq.mean(),
        'r_bruto_ano': res.sum() / anos,
        'r_liq_ano': liq.sum() / anos,
        'uma_vez_ops_ano': len(seq) / anos,
        'uma_vez_r_bruto_ano': res[seq].sum() / anos,
        'uma_vez_r_liq_ano': liq[seq].sum() / anos,
        'r_por_op_23_24': res[metade1].mean() if metade1.any() else np.nan,
        'r_por_op_25_26': res[~metade1].mean() if (~metade1).any() else np.nan,
        'r_mediano': np.median(x['risco_pct']),
    }


def main(ativos):
    linhas = []
    for ativo in ativos:
        for n, stop, g, anos in grupos(ativo):
            primeiro = smc.primeiro_da_pernada(g)      # 1º FVG executado (no tempo) da pernada, antes de qualquer filtro
            for entrada, rmin, horario in itertools.product(ENTRADAS, R_MIN, HORARIOS):
                x = primeiro if entrada.startswith('1º FVG') else g
                x = x[x['r_estrutura'] >= rmin]
                if horario == 'aberturas':
                    x = x[x['abertura']]
                x = selecionar(x, entrada)
                for alvo in ALVOS:
                    m = medir(x, alvo, anos)
                    if m:
                        linhas.append({'ativo': ativo, 'pivo': n, 'entrada': entrada, 'r_estr_min': rmin,
                                       'stop': stop, 'alvo': alvo, 'horario': horario, **m})
    res = pd.DataFrame(linhas)
    SAIDA.mkdir(exist_ok=True)
    res.to_csv(SAIDA / 'estudo_configs_1m.csv', index=False)
    return res


if __name__ == '__main__':
    main(sys.argv[1:])
