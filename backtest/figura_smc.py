"""
Imagem de conferência da estrutura e dos setups de FVG numa janela de tempo.

Uso: python figura_smc.py BTCUSDT 2026-10-16T13:30 240 3
     (ativo, início UTC, nº de candles M1, tamanho do pivô)
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import estrategia as est
import smc

COR_ALTA, COR_BAIXA, COR_ALVO = '#2563eb', '#dc2626', '#15803d'
TINTA, TINTA2, GRADE = '#1f2937', '#6b7280', '#e5e7eb'
SAIDA = Path(__file__).parent / 'saida'


def desenhar(df, e, setups, ini, tam, titulo, arquivo, n, perna=None):
    fim = ini + tam
    w = df.iloc[ini:fim]
    x = np.arange(ini, fim)
    fig, ax = plt.subplots(figsize=(14, 6.5))
    ax.grid(color=GRADE, lw=0.6)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)

    alta = (w['close'] >= w['open']).to_numpy()
    ax.vlines(x, w['low'], w['high'], color='#9ca3af', lw=0.6, zorder=1)
    base = np.minimum(w['open'], w['close'])
    alt = (w['close'] - w['open']).abs().clip(lower=w['close'].mean() * 2e-6)
    ax.bar(x[alta], alt[alta], bottom=base[alta], width=0.7, color='white', edgecolor='#9ca3af', lw=0.5, zorder=2)
    ax.bar(x[~alta], alt[~alta], bottom=base[~alta], width=0.7, color='#9ca3af', edgecolor='#9ca3af', lw=0.5, zorder=2)
    ax.plot(x, w['close'], color=TINTA, lw=1.1, zorder=3, label='Fechamentos (gráfico de linha)')

    # pivôs (desenhados no candle do pivô; só existem n candles depois)
    for k in range(ini, fim):
        if e['pivo_topo'][k]:
            ax.scatter(k, df['close'].iat[k], s=14, color=TINTA2, zorder=4, marker='v')
        if e['pivo_fundo'][k]:
            ax.scatter(k, df['close'].iat[k], s=14, color=TINTA2, zorder=4, marker='^')

    # quebras de estrutura
    for t in range(ini, fim):
        ev = e['evento'][t]
        if ev == 0:
            continue
        cor = COR_ALTA if ev > 0 else COR_BAIXA
        nome = ('BOS' if abs(ev) == 1 else 'CHoCH')
        k0 = max(e['idx_rompido'][t], ini)
        ax.hlines(e['nivel_rompido'][t], k0, t, color=cor, lw=1.4 if abs(ev) == 2 else 0.9, zorder=5)
        ax.text(t, e['nivel_rompido'][t], f' {nome}', color=cor, fontsize=8 if abs(ev) == 1 else 9,
                fontweight='bold' if abs(ev) == 2 else 'normal', va='bottom' if ev > 0 else 'top', zorder=6)
        o = e['origem_idx'][t]
        if ini <= o < fim:
            ax.scatter(o, df['close'].iat[o], s=55, facecolors='none', edgecolors=cor, lw=1.4, zorder=6)

    # setups escolhidos (um por imagem, com a pernada e os 50%)
    for s in setups.itertuples():
        cor = COR_ALTA if s.dir == 1 else COR_BAIXA
        ax.add_patch(plt.Rectangle((s.i_fvg - 2, s.zona_base), s.i_exec - s.i_fvg + 2, s.zona_topo - s.zona_base,
                                   color=cor, alpha=0.2, lw=0, zorder=1))
        ax.text(s.i_fvg - 2, s.zona_topo if s.dir == -1 else s.zona_base, ' FVG', fontsize=8, color=cor,
                va='bottom' if s.dir == -1 else 'top')
        risco = s.dir * (s.entrada - s.stop_estrutura)
        f_fim = min(s.i_exec + 40, fim - 1)
        niveis = [('Entrada', s.entrada, cor, 1.6), ('Stop (acima da estrutura)' if s.dir == -1 else 'Stop (abaixo da estrutura)', s.stop_estrutura, TINTA, 1.3)]
        niveis += [(f'{x}R', s.entrada + s.dir * x * risco, COR_ALVO, 1.4 - 0.3 * (x - 1)) for x in (1, 2, 3)]
        for nome, preco, c_, lw in niveis:
            ax.hlines(preco, s.i_exec, f_fim, color=c_, ls='--', lw=lw, zorder=6)
            ax.text(f_fim, preco, f' {nome}  {preco:,.2f}', fontsize=8, color=TINTA, va='center', zorder=7)
        # pernada: origem (extremo) até o fim da pernada no momento do FVG, e os 50%
        o = s.origem
        if s.dir == -1:
            topo, fundo = s.extremo_perna, df['low'].iloc[o:s.i_fvg + 1].min()
        else:
            topo, fundo = df['high'].iloc[o:s.i_fvg + 1].max(), s.extremo_perna
        meio = (topo + fundo) / 2
        ax.hlines([topo, fundo], max(o, ini), s.i_exec, color=TINTA2, lw=0.8, ls=':', zorder=4)
        ax.hlines(meio, max(o, ini), s.i_exec, color=TINTA2, lw=1.2, ls='-.', zorder=4)
        ax.text(max(o, ini), meio, '50% da pernada ', fontsize=8, color=TINTA2, ha='right', va='center')
        seta = 'v' if s.dir == -1 else '^'
        ax.scatter([s.i_exec], [s.entrada], marker=seta, s=90, color=cor, zorder=8)
        resultado = 'chegou a ' + ', '.join(f'{x}R' for x in (1, 2, 3) if getattr(s, f'acerto_{x}r')) if s.acerto_1r else 'stop antes de 1R'
        ax.set_title(f'{titulo}\n{"VENDA" if s.dir == -1 else "COMPRA"} em {df["time"].iat[s.i_exec]:%d/%m/%Y %H:%M} UTC · '
                     f'FVG a {s.posicao_perna:.0%} da pernada ({"premium" if s.posicao_perna > 0.5 else "discount"}) · '
                     f'R = {abs(risco) / s.entrada:.2%} · {resultado}', loc='left', fontsize=10, color=TINTA)

    passo = max(tam // 10, 1)
    ax.set_xticks(x[::passo], [d.strftime('%d/%m %H:%M') for d in w['time'].iloc[::passo]], fontsize=8)
    ax.set_xlim(ini - 1, fim + 12)
    if not len(setups):
        ax.set_title(titulo, loc='left', fontsize=10.5, color=TINTA)
    ax.legend(frameon=False, loc='upper left', fontsize=8)
    fig.text(0.01, 0.01, f'Pivô de {n} fechamentos · ▼▲ pivôs · ○ origem da pernada (topo que gerou o fundo) · '
             'linha = nível rompido · retângulo = FVG até a execução · horário UTC',
             fontsize=8, color=TINTA2)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(SAIDA / arquivo, dpi=140)
    plt.close(fig)


def main(ativo, inicio, tam, n):
    df = pd.read_pickle(Path(__file__).parent / 'cache' / f'{ativo}_1m.pkl')
    e = smc.estrutura(df, n)
    atr = est.calcular_atr(df).to_numpy()
    ini = int(df['time'].searchsorted(pd.Timestamp(inicio, tz='UTC')))
    janela = slice(max(ini - 2000, 0), min(ini + tam + 2000, len(df)))
    sub = df.iloc[janela].reset_index(drop=True)
    e_sub = smc.estrutura(sub, n)
    setups = smc.montar_setups(sub, e_sub, atr[janela])
    setups = smc.avaliar(sub, setups, 'stop_estrutura') if len(setups) else setups
    ini_sub = ini - janela.start
    SAIDA.mkdir(exist_ok=True)
    base = f'smc_{ativo}_{pd.Timestamp(inicio):%Y%m%d_%H%M}_n{n}'
    desenhar(sub, e_sub, setups.iloc[0:0], ini_sub, tam, f'{ativo} M1 · estrutura pelos fechamentos', base + '_estrutura.png', n)
    print(SAIDA / (base + '_estrutura.png'))
    na_janela = setups[(setups.i_exec >= ini_sub) & (setups.i_exec < ini_sub + tam)]
    escolhidos = pd.concat([na_janela[na_janela.acerto_1r].head(1), na_janela[~na_janela.acerto_1r].head(1),
                            na_janela[na_janela.posicao_perna > 0.5].head(1)]).drop_duplicates('i_exec')
    for k, s in enumerate(escolhidos.itertuples()):
        a = max(min(s.origem, s.i_fvg - 10) - 10, 0)
        b = min(s.i_exec + 60, len(sub))
        nome = f'{base}_setup{k + 1}.png'
        desenhar(sub, e_sub, escolhidos.iloc[[k]], a, b - a, f'{ativo} M1 · setup de FVG', nome, n)
        print(SAIDA / nome)
    print(f'{len(na_janela)} setups executados na janela de {tam} candles')


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]))
