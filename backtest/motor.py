"""
Motor de backtest "sempre posicionado" (stop-and-reverse).

Regras de honestidade:
  - o estado do candle i (fechamento) só é executado na ABERTURA do candle i+1
  - custo por lado aplicado em toda entrada e saída
  - o último trade, se ainda aberto, é marcado no último fechamento e sinalizado como aberto
"""

import numpy as np
import pandas as pd


def extrair_trades(df, custo_lado=0.001):
    """
    df precisa ter: time, open, high, low, close, estado, faixa_sup, faixa_inf.
    Cada trade: entra na abertura do candle seguinte à virada e sai na abertura do candle
    seguinte à próxima virada.
    """
    estado = df['estado'].to_numpy()
    abertura = df['open'].to_numpy()
    maxima = df['high'].to_numpy()
    minima = df['low'].to_numpy()
    fech = df['close'].to_numpy()
    tempo = df['time'].to_numpy()
    f_sup = df['faixa_sup'].to_numpy()
    f_inf = df['faixa_inf'].to_numpy()
    n = len(df)

    # índices dos candles em que o estado mudou (sinal no fechamento)
    viradas = [i for i in range(1, n) if estado[i] != 0 and estado[i] != estado[i - 1]]

    trades = []
    for j, i_sinal in enumerate(viradas):
        e = i_sinal + 1                       # candle de entrada (abertura)
        if e >= n:
            break                             # sinal no último candle: ainda não executado
        direcao = estado[i_sinal]
        entrada = abertura[e]
        stop_inicial = f_inf[i_sinal] if direcao == 1 else f_sup[i_sinal]

        if j + 1 < len(viradas) and viradas[j + 1] + 1 < n:
            i_saida_sinal = viradas[j + 1]
            s = i_saida_sinal + 1             # candle de saída (abertura)
            saida = abertura[s]
            ultimo = s - 1                    # último candle inteiro dentro do trade
            aberto = False
        else:
            s = n - 1
            saida = fech[s]
            ultimo = s
            aberto = True

        ret_bruto = direcao * (saida / entrada - 1)
        ret = (1 + ret_bruto) * (1 - custo_lado) ** 2 - 1

        risco = direcao * (entrada - stop_inicial) / entrada        # R como fração do preço
        mx = maxima[e:ultimo + 1]
        mn = minima[e:ultimo + 1]
        if direcao == 1:
            mfe = mx.max() / entrada - 1
            mae = mn.min() / entrada - 1
            idx_1r = np.argmax(mx >= entrada * (1 + risco)) if (mx >= entrada * (1 + risco)).any() else None
            idx_stop = np.argmax(mn <= stop_inicial) if (mn <= stop_inicial).any() else None
        else:
            mfe = 1 - mn.min() / entrada
            mae = 1 - mx.max() / entrada
            idx_1r = np.argmax(mn <= entrada * (1 - risco)) if (mn <= entrada * (1 - risco)).any() else None
            idx_stop = np.argmax(mx >= stop_inicial) if (mx >= stop_inicial).any() else None

        if idx_1r is None:
            r1_antes_stop = False
        elif idx_stop is None:
            r1_antes_stop = True
        else:
            # empate no mesmo candle conta como stop (postura conservadora)
            r1_antes_stop = idx_1r < idx_stop

        trades.append({
            'direcao': 'COMPRA' if direcao == 1 else 'VENDA',
            'dir': int(direcao),
            'i_sinal': int(i_sinal), 'i_entrada': int(e), 'i_saida': int(s),
            'tempo_entrada': pd.Timestamp(tempo[e]), 'tempo_saida': pd.Timestamp(tempo[s]),
            'preco_entrada': float(entrada), 'preco_saida': float(saida),
            'stop_inicial': float(stop_inicial),
            'risco_pct': float(risco),
            'retorno': float(ret),
            'retorno_r': float(ret_bruto / risco) if risco > 0 else np.nan,
            'mfe_pct': float(mfe), 'mae_pct': float(mae),
            'mfe_r': float(mfe / risco) if risco > 0 else np.nan,
            'mae_r': float(mae / risco) if risco > 0 else np.nan,
            'duracao': int(s - e),
            'r1_antes_stop': bool(r1_antes_stop),
            'tocou_stop_inicial': idx_stop is not None,
            'aberto': aberto,
        })
    return pd.DataFrame(trades)


def curva_capital(df, trades, custo_lado=0.001):
    """Capital marcado no fechamento de cada candle (começa em 1.0)."""
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
    return {
        'retorno_total': eq.iloc[-1] - 1,
        'retorno_anual': eq.iloc[-1] ** (1 / anos) - 1,
        'dd_max': drawdown_max(eq),
        'trades': len(fechados),
        'acerto': (fechados['retorno'] > 0).mean(),
        'profit_factor': ganhos / perdas if perdas > 0 else np.inf,
        'duracao_media_h': fechados['duracao'].mean(),
        'pct_1r': (fechados['mfe_r'] >= 1).mean(),
        'pct_2r': (fechados['mfe_r'] >= 2).mean(),
        'pct_3r': (fechados['mfe_r'] >= 3).mean(),
        'pct_1r_antes_stop': fechados['r1_antes_stop'].mean(),
        'risco_mediano': fechados['risco_pct'].median(),
        'mfe_mediano': fechados['mfe_pct'].median(),
        'mae_mediano': fechados['mae_pct'].median(),
    }
