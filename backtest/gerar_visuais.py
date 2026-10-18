"""
Gera as imagens do relatório e os dados do visualizador de trades.

Uso: python gerar_visuais.py BTCUSDT 1h
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import estrategia as est
import motor as m
from rodar_grade import CUSTO_LADO, INICIO, KS, MEDIAS, SAIDA, buy_and_hold

COR_COMPRA = '#2563eb'
COR_VENDA = '#dc2626'
COR_STOP = '#111827'
COR_ALVO = '#15803d'
COR_TINTA = '#1f2937'
COR_SECUNDARIA = '#6b7280'
COR_GRADE = '#e5e7eb'

plt.rcParams.update({
    'font.size': 10, 'axes.edgecolor': COR_GRADE, 'axes.labelcolor': COR_SECUNDARIA,
    'xtick.color': COR_SECUNDARIA, 'ytick.color': COR_SECUNDARIA, 'axes.grid': True,
    'grid.color': COR_GRADE, 'grid.linewidth': 0.6, 'axes.spines.top': False, 'axes.spines.right': False,
})


def mapa_calor(res, simbolo, timeframe, bh):
    fig, ax = plt.subplots(figsize=(7, 3.6))
    tabela = res.pivot(index='media', columns='k', values='retorno_anual')
    lim = max(abs(tabela.values).max(), 0.01)
    im = ax.imshow(tabela.values, cmap='RdBu', vmin=-lim, vmax=lim, aspect='auto')
    ax.set_xticks(range(len(KS)), [f'k = {k}' for k in KS])
    ax.set_yticks(range(len(MEDIAS)), [f'Média {mm}' for mm in MEDIAS])
    ax.grid(False)
    for i, mm in enumerate(tabela.index):
        for j, k in enumerate(tabela.columns):
            linha = res[(res.media == mm) & (res.k == k)].iloc[0]
            v = tabela.loc[mm, k]
            ax.text(j, i, f'{v:+.0%} a.a.\nDD {linha.dd_max:.0%}', ha='center', va='center', fontsize=9,
                    color='white' if abs(v) > lim * 0.55 else COR_TINTA)
    ax.set_title(f'{simbolo} {timeframe} · retorno anual por combinação (custo {CUSTO_LADO:.1%}/lado)\n'
                 f'Buy & hold no período: {bh["retorno_anual"]:+.0%} a.a., DD {bh["dd_max"]:.0%}',
                 fontsize=10, loc='left', color=COR_TINTA)
    fig.colorbar(im, ax=ax, format=lambda x, _: f'{x:+.0%}', shrink=0.85)
    fig.tight_layout()
    fig.savefig(SAIDA / f'mapa_calor_{simbolo}_{timeframe}.png', dpi=150)
    plt.close(fig)


def curva(df, combos, simbolo, timeframe):
    fig, (ax, ax_dd) = plt.subplots(2, 1, figsize=(10, 6), sharex=True, gridspec_kw={'height_ratios': [3, 1.3]})
    bh = pd.Series((df['close'] / df['open'].iloc[0]).to_numpy(), index=df['time'])
    series = [('Buy & hold', bh, COR_SECUNDARIA)]
    cores = [COR_COMPRA, '#7c3aed']
    for (mm, k), cor in zip(combos, cores):
        s = est.calcular_sinais(df, mm, k)
        t = m.extrair_trades(s, CUSTO_LADO)
        series.append((f'Média {mm}, k {k}', m.curva_capital(s, t, CUSTO_LADO), cor))
    for nome, eq, cor in series:
        ax.plot(eq.index, eq.values, color=cor, lw=1.4, label=nome)
        ax.annotate(f'{eq.iloc[-1] - 1:+.0%}', (eq.index[-1], eq.iloc[-1]), xytext=(4, 0),
                    textcoords='offset points', va='center', fontsize=9, color=COR_TINTA)
        ax_dd.plot(eq.index, (eq / eq.cummax() - 1).values, color=cor, lw=1)
    ax.set_yscale('log')
    ax.yaxis.set_major_formatter(lambda x, _: f'{x - 1:+.0%}')
    ax.set_ylabel('Retorno acumulado (escala log)')
    ax.legend(frameon=False, loc='upper left')
    ax.set_title(f'{simbolo} {timeframe} · capital seguindo o sinal à risca, com custos', loc='left',
                 color=COR_TINTA, fontsize=11)
    ax_dd.yaxis.set_major_formatter(lambda x, _: f'{x:.0%}')
    ax_dd.set_ylabel('Drawdown')
    fig.tight_layout()
    fig.savefig(SAIDA / f'curva_capital_{simbolo}_{timeframe}.png', dpi=150)
    plt.close(fig)


def desenhar_trade(s, t, nome_arquivo, titulo):
    """Candles em volta de um trade, com entrada, stop inicial e alvos 1R/2R/3R tracejados."""
    antes, depois = 40, 15
    ini = max(t.i_entrada - antes, 0)
    fim = min(t.i_saida + depois, len(s) - 1)
    w = s.iloc[ini:fim + 1]
    x = np.arange(len(w))

    fig, ax = plt.subplots(figsize=(11, 5.2))
    alta = w['close'] >= w['open']
    ax.vlines(x, w['low'], w['high'], color=COR_SECUNDARIA, lw=0.7, zorder=2)
    corpo_base = np.minimum(w['open'], w['close'])
    corpo_alt = (w['close'] - w['open']).abs().clip(lower=w['close'].mean() * 1e-5)
    ax.bar(x[alta], corpo_alt[alta], bottom=corpo_base[alta], width=0.7, color='white',
           edgecolor=COR_TINTA, lw=0.7, zorder=3)
    ax.bar(x[~alta], corpo_alt[~alta], bottom=corpo_base[~alta], width=0.7, color=COR_TINTA,
           edgecolor=COR_TINTA, lw=0.7, zorder=3)
    ax.plot(x, w['media'], color=COR_SECUNDARIA, lw=1.2, label='Média')
    ax.plot(x, w['faixa_sup'], color=COR_SECUNDARIA, lw=0.8, ls=':', label='Faixas (média ± k·ATR)')
    ax.plot(x, w['faixa_inf'], color=COR_SECUNDARIA, lw=0.8, ls=':')

    xe, xs = t.i_entrada - ini, t.i_saida - ini
    cor_dir = COR_COMPRA if t.dir == 1 else COR_VENDA
    r_preco = t.preco_entrada * t.risco_pct
    niveis = [('Entrada', t.preco_entrada, cor_dir, '--', 1.6),
              ('Stop inicial', t.stop_inicial, COR_STOP, '--', 1.4)]
    for n_r, alpha in [(1, 1.0), (2, 0.75), (3, 0.5)]:
        niveis.append((f'{n_r}R', t.preco_entrada + t.dir * n_r * r_preco, COR_ALVO, '--', 1.2 * alpha + 0.3))
    for nome, preco, cor, ls, lw in niveis:
        ax.hlines(preco, xe - 0.5, xs + 0.5, color=cor, ls=ls, lw=lw, zorder=4)
        ax.text(xs + 1, preco, f' {nome}  {preco:,.2f}', va='center', fontsize=8.5, color=COR_TINTA, zorder=5)

    seta = '^' if t.dir == 1 else 'v'
    y_seta = w['low'].iloc[xe] * 0.997 if t.dir == 1 else w['high'].iloc[xe] * 1.003
    ax.scatter([xe], [y_seta], marker=seta, s=90, color=cor_dir, zorder=6)
    ax.scatter([xs], [t.preco_saida], marker='X', s=70, color=COR_TINTA, zorder=6,
               label=f'Saída (virada do sinal) {t.preco_saida:,.2f}')
    ax.axvspan(xe - 0.5, xs + 0.5, color=cor_dir, alpha=0.05, zorder=1)

    passo = max(len(w) // 8, 1)
    ax.set_xticks(x[::passo], [d.strftime('%d/%m/%y %Hh') for d in w['time'].iloc[::passo]], fontsize=8)
    ax.set_xlim(-1, len(w) + 12)
    ax.legend(frameon=False, loc='best', fontsize=8.5)
    ax.set_title(titulo, loc='left', fontsize=10.5, color=COR_TINTA)
    fig.tight_layout()
    fig.savefig(SAIDA / nome_arquivo, dpi=150)
    plt.close(fig)


def titulo_trade(t, rotulo):
    status = 'chegou a 1R antes de tocar o stop inicial' if t.acerto_1r else 'não chegou a 1R antes do stop inicial'
    return (f'{rotulo} · {t.direcao} em {t.tempo_entrada:%d/%m/%Y %Hh} UTC · durou {t.duracao} h\n'
            f'Resultado {t.retorno:+.2%} ({t.retorno_r:+.2f}R, R = {t.risco_pct:.2%}) · '
            f'máx. a favor {t.mfe_r:+.1f}R · máx. contra {t.mae_r:+.1f}R · {status}')


def main(simbolo='BTCUSDT', timeframe='1h'):
    df = pd.read_pickle(Path(__file__).parent / 'cache' / f'{simbolo}_{timeframe}.pkl')
    df = df[df['time'] >= INICIO].reset_index(drop=True)
    res = pd.read_csv(SAIDA / f'grade_{simbolo}_{timeframe}.csv')
    bh = buy_and_hold(df)

    mapa_calor(res, simbolo, timeframe, bh)
    curva(df, [(100, 1.0), (200, 1.0)], simbolo, timeframe)

    s = est.calcular_sinais(df, 100, 1.0)
    t = m.extrair_trades(s, CUSTO_LADO)
    fechados = t[~t['aberto']]
    exemplos = {
        'maior_ganho': fechados.loc[fechados['retorno'].idxmax()],
        'perda_tipica': fechados[fechados['retorno'] < 0].sort_values('retorno').iloc[len(fechados[fechados['retorno'] < 0]) // 2],
        'ultimo_fechado': fechados.iloc[-1],
    }
    rotulos = {'maior_ganho': 'Maior ganho', 'perda_tipica': 'Perda típica (mediana das perdas)',
               'ultimo_fechado': 'Último trade fechado'}
    for nome, tr in exemplos.items():
        desenhar_trade(s, tr, f'trade_{nome}_{simbolo}_{timeframe}.png',
                       f'Média 100, k 1.0 · {titulo_trade(tr, rotulos[nome])}')


if __name__ == '__main__':
    main(*sys.argv[1:3])
