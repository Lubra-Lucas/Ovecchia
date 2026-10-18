"""
Ranking da grade de configurações (estudo_configs.py): média de BTC e ETH, só configurações
positivas nos dois ativos e nas duas metades do período (2023–2024 e 2025–2026).

Uso: python ranking_configs.py
"""

from pathlib import Path

import pandas as pd

SAIDA = Path(__file__).parent / 'saida'
CHAVE = ['pivo', 'entrada', 'r_estr_min', 'stop', 'alvo', 'horario']
METRICAS = ['ops_ano', 'acerto', 'r_por_op', 'liq_por_op', 'r_bruto_ano', 'r_liq_ano', 'uma_vez_ops_ano',
            'uma_vez_r_bruto_ano', 'uma_vez_r_liq_ano', 'r_por_op_23_24', 'r_por_op_25_26', 'r_mediano']
COLS = CHAVE + ['ops_ano', 'acerto', 'r_por_op', 'liq_por_op', 'r_bruto_ano', 'uma_vez_ops_ano',
                'uma_vez_r_bruto_ano', 'uma_vez_r_liq_ano', 'R%']
FMT = {'ops_ano': '{:.0f}'.format, 'acerto': '{:.0%}'.format, 'r_por_op': '{:+.2f}'.format,
       'liq_por_op': '{:+.2f}'.format, 'r_bruto_ano': '{:+.0f}'.format, 'uma_vez_ops_ano': '{:.0f}'.format,
       'uma_vez_r_bruto_ano': '{:+.0f}'.format, 'uma_vez_r_liq_ano': '{:+.0f}'.format, 'r_estr_min': '{:.1%}'.format}


def montar():
    r = pd.read_csv(SAIDA / 'estudo_configs_1m.csv')
    b = r[r.ativo == 'BTCUSDT'].set_index(CHAVE)
    e = r[r.ativo == 'ETHUSDT'].set_index(CHAVE)
    j = b.join(e, lsuffix='_btc', rsuffix='_eth', how='inner')
    for c in METRICAS:
        j[c] = (j[c + '_btc'] + j[c + '_eth']) / 2
    pos = lambda c: (j[c + '_btc'] > 0) & (j[c + '_eth'] > 0)
    ok = pos('r_por_op') & pos('r_por_op_23_24') & pos('r_por_op_25_26')
    v = j[ok].reset_index()
    v['R%'] = (v['r_mediano'] * 100).round(2)
    return j, v


if __name__ == '__main__':
    j, v = montar()
    pd.set_option('display.width', 300)
    pd.set_option('display.max_colwidth', 34)
    print(f'{len(j)} configurações · {len(v)} positivas (R bruto por operação) nos 2 ativos e nas 2 metades\n')
    for titulo, df, col in (
        ('R BRUTO ACUMULADO POR ANO — todas as operações (média BTC/ETH)', v, 'r_bruto_ano'),
        ('R BRUTO POR ANO — operando uma por vez', v, 'uma_vez_r_bruto_ano'),
        ('R LÍQUIDO POR ANO — operando uma por vez (0,02% + 0,05%)', v, 'uma_vez_r_liq_ano'),
        ('MAIOR ACERTO NO ALVO (≥ 1 operação por dia em cada ativo)', v[(v.ops_ano_btc >= 365) & (v.ops_ano_eth >= 365)], 'acerto'),
    ):
        print(titulo)
        print(df.sort_values(col, ascending=False)[COLS].head(10).to_string(index=False, formatters=FMT))
        print()
