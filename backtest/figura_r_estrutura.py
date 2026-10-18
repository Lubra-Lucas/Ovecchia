"""
Explica o "R da estrutura" com operações reais: tamanho da pernada (entrada até o extremo da origem)
em três tamanhos, e o desempenho por faixa nos 3 anos de M1.

Uso: python figura_r_estrutura.py
"""

from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import estrategia as est
import smc

ALTA, BAIXA = '#4f6bed', '#131313'
COMPRA, VENDA, ALVO, TINTA, CINZA = '#2563eb', '#dc2626', '#15803d', '#111827', '#6b7280'
SAIDA = Path(__file__).parent / 'saida'
FUSO = 'America/Sao_Paulo'


def setups(df, mult=2):
    atr = est.calcular_atr(df).to_numpy()
    e = smc.estrutura(df, 5)
    s = smc.montar_setups(df, e, atr)
    s = smc.primeiro_da_pernada(s).reset_index(drop=True)
    s['stop_atr'] = s['entrada'] - s['dir'] * mult * atr[s['i_fvg'].to_numpy()]
    s['r_estrutura'] = s['dir'] * (s['entrada'] - s['stop_estrutura']) / s['entrada']
    s['pernada_atr'] = s['dir'] * (s['entrada'] - s['stop_estrutura']) / atr[s['i_fvg'].to_numpy()]
    return smc.avaliar(df, s, 'stop_atr'), e


def desenhar(df, e, op, arquivo, rotulo):
    d = op.dir
    ini = max(min(op.origem, op.i_fvg - 2) - 25, 0)
    duracao = min(op.t_stop, op.t_3r, 300) if np.isfinite(min(op.t_stop, op.t_3r)) else 300
    fim_op = op.i_exec + max(int(duracao), 25)          # largura mínima para a posição ficar legível
    fim = min(fim_op + 25, len(df) - 1)
    w = df.iloc[ini:fim + 1]
    x = np.arange(ini, fim + 1)
    fig, ax = plt.subplots(figsize=(13, 6.2))
    ax.grid(color='#eef0f3', lw=0.6)
    for s_ in ('top', 'right'):
        ax.spines[s_].set_visible(False)
    sobe = (w['close'] >= w['open']).to_numpy()
    for msk, cor in ((sobe, ALTA), (~sobe, BAIXA)):
        ax.vlines(x[msk], w['low'].to_numpy()[msk], w['high'].to_numpy()[msk], color=cor, lw=0.8, zorder=2)
        base = np.minimum(w['open'], w['close']).to_numpy()[msk]
        alt = np.maximum((w['close'] - w['open']).abs().to_numpy()[msk], w['close'].mean() * 2e-6)
        ax.bar(x[msk], alt, bottom=base, width=0.7, color=cor, zorder=3)
    ax.plot(x, w['close'], color=CINZA, lw=0.9, ls=':', zorder=4, label='Fechamentos')

    cor = COMPRA if d == 1 else VENDA
    risco = d * (op.entrada - op.stop_atr)
    extremo = op.stop_estrutura          # extremo da pernada + folga de 0,1 ATR (o R da estrutura mede até aqui)
    xe = op.i_exec
    xf = fim_op
    # FVG
    ax.add_patch(plt.Rectangle((op.i_fvg - 2, op.zona_base), xe - op.i_fvg + 2, op.zona_topo - op.zona_base,
                               color=cor, alpha=0.18, lw=0, zorder=1))
    ax.text(op.i_fvg - 2, op.zona_topo if d == 1 else op.zona_base, 'FVG', color=cor, fontsize=9,
            va='bottom' if d == 1 else 'top')
    # origem e pernada
    ax.scatter([op.origem], [op.extremo_perna], s=80, facecolors='none', edgecolors=cor, lw=1.6, zorder=6)
    ax.text(op.origem, op.extremo_perna, '  origem da pernada', color=cor, fontsize=9,
            va='top' if d == 1 else 'bottom')
    ax.hlines(op.extremo_perna, op.origem, xe, color=cor, lw=1, ls=':', zorder=4)
    # régua do R da estrutura
    xr = xe - 3
    ax.annotate('', xy=(xr, op.entrada), xytext=(xr, extremo),
                arrowprops=dict(arrowstyle='<->', color='#7c3aed', lw=2), zorder=7)
    ax.text(xr - 2, (op.entrada + extremo) / 2,
            f'R da estrutura\n{op.r_estrutura:.2%}\n({op.pernada_atr:.1f} ATR)', color='#7c3aed',
            fontsize=10, ha='right', va='center', fontweight='bold', zorder=7)
    # posição real (stop 2 × ATR, alvos)
    ax.add_patch(plt.Rectangle((xe, min(op.entrada, op.entrada + 3 * d * risco)), xf - xe, 3 * risco,
                               color=ALVO, alpha=0.12, lw=0, zorder=1))
    ax.add_patch(plt.Rectangle((xe, min(op.entrada, op.stop_atr)), xf - xe, risco,
                               color=VENDA, alpha=0.12, lw=0, zorder=1))
    niveis = [('Entrada', op.entrada, cor, '-'), ('Stop 2 × ATR', op.stop_atr, VENDA, '--')]
    niveis += [(f'{k}R', op.entrada + d * k * risco, ALVO, '--') for k in (1, 2, 3)]
    for nome, p, c_, ls in niveis:
        ax.hlines(p, xe, xf, color=c_, lw=1.4, ls=ls, zorder=6)
        ax.text(xf + 1, p, f' {nome}  {p:,.2f}', fontsize=8.5, color=c_, va='center', zorder=7)
    resultado = ('chegou a 3R' if op.acerto_3r else 'chegou a 2R, depois stop' if op.acerto_2r
                 else 'chegou a 1R, depois stop' if op.acerto_1r else 'stop antes de 1R')
    quando = pd.Timestamp(df['time'].iat[xe]).tz_convert(FUSO)
    ax.set_title(f'{rotulo} · {"COMPRA" if d == 1 else "VENDA"} BTC M1 em {quando:%d/%m/%Y %H:%M} (São Paulo)\n'
                 f'R da estrutura {op.r_estrutura:.2%} ({op.pernada_atr:.1f} ATR) · stop real 2 × ATR = '
                 f'{risco / op.entrada:.2%} · {resultado}', loc='left', fontsize=10.5, color=TINTA)
    passo = max(len(w) // 9, 1)
    ax.set_xticks(x[::passo], [pd.Timestamp(t).tz_convert(FUSO).strftime('%H:%M') for t in w['time'].iloc[::passo]],
                  fontsize=8)
    ax.set_xlim(ini - 1, fim + 22)
    ax.legend(frameon=False, loc='upper left', fontsize=8.5)
    fig.tight_layout()
    fig.savefig(SAIDA / arquivo, dpi=140)
    plt.close(fig)


def main():
    SAIDA.mkdir(exist_ok=True)
    # desempenho por faixa, 3 anos, BTC + ETH
    faixas = [0, 0.001, 0.002, 0.004, 0.006, 1]
    nomes = ['< 0,1%', '0,1–0,2%', '0,2–0,4%', '0,4–0,6%', '> 0,6%']
    tabelas = []
    for ativo in ('BTCUSDT', 'ETHUSDT'):
        df = pd.read_pickle(Path(__file__).parent / 'cache' / f'{ativo}_1m.pkl')
        a, e = setups(df)
        a['faixa'] = pd.cut(a['r_estrutura'], faixas, labels=nomes)
        a['liq_3r'] = a['resultado_3r'] - 0.0007 / a['risco_pct']
        t = a.groupby('faixa', observed=True).agg(
            operacoes=('dir', 'size'), pernada_atr=('pernada_atr', 'median'), acerto_1r=('acerto_1r', 'mean'),
            acerto_3r=('acerto_3r', 'mean'), r_bruto_3r=('resultado_3r', 'mean'), r_liq_3r=('liq_3r', 'mean'))
        t['ativo'] = ativo[:3]
        tabelas.append(t.reset_index())
        if ativo == 'BTCUSDT':
            recentes = a[a['i_exec'] > len(df) - 30 * 1440]
            for rotulo, alvo, arq in (('Pernada pequena', 0.001, 'r_estrutura_pequena.png'),
                                     ('Pernada média', 0.003, 'r_estrutura_media.png'),
                                     ('Pernada grande', 0.007, 'r_estrutura_grande.png')):
                op = recentes.iloc[(recentes['r_estrutura'] - alvo).abs().argsort().iloc[0]]
                desenhar(df, e, op, arq, rotulo)
                print(SAIDA / arq)
    t = pd.concat(tabelas)
    t.to_csv(SAIDA / 'r_estrutura_faixas.csv', index=False)
    print(t.to_string(index=False, formatters={'acerto_1r': '{:.0%}'.format, 'acerto_3r': '{:.0%}'.format,
                                               'r_bruto_3r': '{:+.2f}'.format, 'r_liq_3r': '{:+.2f}'.format,
                                               'pernada_atr': '{:.1f}'.format}))


if __name__ == '__main__':
    main()
