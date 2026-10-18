"""
Motor de backtest da Ovelha V3.

Regras de honestidade:
  - o estado do candle i (fechamento) só é executado na ABERTURA do candle i+1
  - stops e alvos são ordens que só passam a valer a partir do candle de entrada;
    se o candle abrir além do stop, a saída é na abertura (gap), nunca no preço do stop
  - se stop e alvo forem tocados no mesmo candle, conta como stop (postura conservadora)
  - custo por lado aplicado em toda entrada e saída
  - o último trade, se ainda aberto, é marcado no último fechamento e sinalizado como aberto

Modos de saída:
  - 'virada'      sai só quando o sinal vira (sempre posicionado)
  - 'stop_movel'  stop começa na faixa oposta e só anda a favor; tocado, fica fora até a próxima virada
  - 'alvo'        sai no alvo (alvo_r × R) ou no stop inicial; se nenhum for tocado, sai na virada
"""

import numpy as np
import pandas as pd

ALVOS_R = (1, 2, 3)


def _primeiro(mascara):
    """Índice do primeiro True, ou infinito se não houver."""
    return int(np.argmax(mascara)) if mascara.any() else np.inf


def extrair_trades(df, custo_lado=0.001, saida='virada', alvo_r=None, stop_atr=None):
    """
    df precisa ter: time, open, high, low, close, estado, faixa_sup, faixa_inf, atr.
    stop_atr: se informado, o stop inicial fica a stop_atr × ATR da entrada (em vez da faixa oposta).
    """
    estado = df['estado'].to_numpy()
    abertura = df['open'].to_numpy()
    maxima = df['high'].to_numpy()
    minima = df['low'].to_numpy()
    fech = df['close'].to_numpy()
    tempo = df['time'].to_numpy()
    f_sup = df['faixa_sup'].to_numpy()
    f_inf = df['faixa_inf'].to_numpy()
    atr = df['atr'].to_numpy()
    n = len(df)

    # índices dos candles em que o estado mudou (sinal no fechamento)
    viradas = [i for i in range(1, n) if estado[i] != 0 and estado[i] != estado[i - 1]]

    trades = []
    for j, i_sinal in enumerate(viradas):
        e = i_sinal + 1                       # candle de entrada (abertura)
        if e >= n:
            break                             # sinal no último candle: ainda não executado
        d = int(estado[i_sinal])
        entrada = abertura[e]
        faixa_oposta = f_inf if d == 1 else f_sup
        if stop_atr is None:
            stop_inicial = faixa_oposta[i_sinal]
        else:
            stop_inicial = entrada - d * stop_atr * atr[i_sinal]
        risco = d * (entrada - stop_inicial) / entrada        # R como fração do preço

        # fim natural do sinal: abertura do candle seguinte à próxima virada
        if j + 1 < len(viradas) and viradas[j + 1] + 1 < n:
            fim, aberto_natural = viradas[j + 1] + 1, False
            ultimo = fim - 1
        else:
            fim, aberto_natural = n - 1, True
            ultimo = n - 1
        mx, mn, ab = maxima[e:ultimo + 1], minima[e:ultimo + 1], abertura[e:ultimo + 1]

        # ---- assertividade do sinal: chegou a XR antes de tocar o stop inicial? ----
        toque_stop = _primeiro(mn <= stop_inicial) if d == 1 else _primeiro(mx >= stop_inicial)
        acertos = {}
        for x in ALVOS_R:
            preco_alvo = entrada * (1 + d * x * risco)
            toque = _primeiro(mx >= preco_alvo) if d == 1 else _primeiro(mn <= preco_alvo)
            acertos[x] = toque < toque_stop

        # ---- saída conforme o modo ----
        i_saida = fim
        preco_saida = fech[n - 1] if aberto_natural else abertura[fim]
        motivo = 'em aberto' if aberto_natural else 'virada'

        if saida == 'stop_movel':
            # stop vigente no candle e+t = nível conhecido no fechamento anterior (só anda a favor)
            niveis = np.r_[stop_inicial, faixa_oposta[e:ultimo]]
            stop_vig = np.maximum.accumulate(niveis) if d == 1 else np.minimum.accumulate(niveis)
            t = _primeiro(mn <= stop_vig) if d == 1 else _primeiro(mx >= stop_vig)
            if t != np.inf:
                i_saida = e + t
                preco_saida = min(ab[t], stop_vig[t]) if d == 1 else max(ab[t], stop_vig[t])
                motivo = 'stop móvel'
        elif saida == 'alvo':
            preco_alvo = entrada * (1 + d * alvo_r * risco)
            t_alvo = _primeiro(mx >= preco_alvo) if d == 1 else _primeiro(mn <= preco_alvo)
            if toque_stop != np.inf and toque_stop <= t_alvo:
                t = toque_stop
                i_saida = e + t
                preco_saida = min(ab[t], stop_inicial) if d == 1 else max(ab[t], stop_inicial)
                motivo = 'stop'
            elif t_alvo != np.inf:
                t = t_alvo
                i_saida = e + t
                preco_saida = max(ab[t], preco_alvo) if d == 1 else min(ab[t], preco_alvo)
                motivo = f'alvo {alvo_r}R'

        aberto = motivo == 'em aberto'
        saiu_no_candle = motivo not in ('virada', 'em aberto')
        fim_janela = i_saida if saiu_no_candle else (ultimo if aberto else i_saida - 1)
        jmx, jmn = maxima[e:fim_janela + 1], minima[e:fim_janela + 1]
        if d == 1:
            mfe, mae = jmx.max() / entrada - 1, jmn.min() / entrada - 1
        else:
            mfe, mae = 1 - jmn.min() / entrada, 1 - jmx.max() / entrada

        ret_bruto = d * (preco_saida / entrada - 1)
        ret = (1 + ret_bruto) * (1 - custo_lado) ** 2 - 1
        trades.append({
            'direcao': 'COMPRA' if d == 1 else 'VENDA',
            'dir': d,
            'i_sinal': int(i_sinal), 'i_entrada': int(e), 'i_saida': int(i_saida),
            'tempo_entrada': pd.Timestamp(tempo[e]), 'tempo_saida': pd.Timestamp(tempo[i_saida]),
            'preco_entrada': float(entrada), 'preco_saida': float(preco_saida),
            'stop_inicial': float(stop_inicial),
            'risco_pct': float(risco),
            'retorno_bruto': float(ret_bruto),
            'retorno': float(ret),
            'retorno_r': float(ret_bruto / risco) if risco > 0 else np.nan,
            'mfe_pct': float(mfe), 'mae_pct': float(mae),
            'mfe_r': float(mfe / risco) if risco > 0 else np.nan,
            'mae_r': float(mae / risco) if risco > 0 else np.nan,
            'duracao': int(i_saida - e),
            'acerto_1r': acertos[1], 'acerto_2r': acertos[2], 'acerto_3r': acertos[3],
            'motivo': motivo,
            'aberto': aberto,
        })
    return pd.DataFrame(trades)


def curva_capital(df, trades, custo_lado=0.001):
    """Capital marcado no fechamento de cada candle (começa em 1.0). Fora de posição, fica parado."""
    fech = df['close'].to_numpy()
    eq = np.ones(len(df))
    capital = 1.0
    ultimo_i = 0
    for t in trades.itertuples():
        e, s = t.i_entrada, t.i_saida
        eq[ultimo_i:e] = capital
        base = capital * (1 - custo_lado)
        # dentro do trade: marcação a mercado nos fechamentos
        eq[e:s] = base * (1 + t.dir * (fech[e:s] / t.preco_entrada - 1))
        capital = capital * (1 + t.retorno)
        ultimo_i = s
    eq[ultimo_i:] = capital
    return pd.Series(eq, index=df['time'])


def drawdown_max(eq):
    return float((eq / eq.cummax() - 1).min())


def resumo(trades, eq):
    fechados = trades[~trades['aberto']]
    anos = (eq.index[-1] - eq.index[0]).days / 365.25
    ganhos = fechados.loc[fechados['retorno'] > 0, 'retorno'].sum()
    perdas = -fechados.loc[fechados['retorno'] <= 0, 'retorno'].sum()
    sinais = trades.iloc[:-1] if len(trades) and trades['aberto'].iloc[-1] else trades
    return {
        'retorno_total': eq.iloc[-1] - 1,
        'retorno_anual': eq.iloc[-1] ** (1 / anos) - 1,
        'dd_max': drawdown_max(eq),
        'trades': len(fechados),
        'acerto': (fechados['retorno'] > 0).mean(),
        'profit_factor': ganhos / perdas if perdas > 0 else np.inf,
        'duracao_media_h': fechados['duracao'].mean(),
        'resultado_medio_r': fechados['retorno_r'].mean(),
        'acerto_1r': sinais['acerto_1r'].mean(),
        'acerto_2r': sinais['acerto_2r'].mean(),
        'acerto_3r': sinais['acerto_3r'].mean(),
        'risco_mediano': fechados['risco_pct'].median(),
        'mfe_mediano': fechados['mfe_pct'].median(),
        'mae_mediano': fechados['mae_pct'].median(),
    }
