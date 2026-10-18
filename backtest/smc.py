"""
Estrutura de mercado pelos fechamentos (gráfico de linha) + Fair Value Gaps + setups de entrada.

Estrutura
  - Pivô de topo em i: close[i] maior que os n fechamentos anteriores e >= aos n seguintes.
    Só passa a existir em i+n (quando os n fechamentos seguintes já aconteceram).
  - Em tendência de baixa, o gatilho de continuação é o último fundo (pivô) confirmado depois da
    última quebra; o gatilho de reversão é a ORIGEM da pernada: o maior fechamento entre o fundo
    que foi rompido e o candle da quebra ("o topo que gerou o último fundo").
  - Fechou abaixo do fundo  -> BOS de baixa (se já estava em baixa) ou CHoCH de baixa.
    Fechou acima da origem  -> CHoCH de alta.  Tendência de alta é o espelho.

FVG de baixa (três candles i-2, i-1, i): low[i-2] > high[i]. Zona = [high[i], low[i-2]].
Só é conhecido no fechamento do candle i.

Setup de venda: FVG de baixa dentro da pernada de baixa atual, ainda não tocado.
Entrada por ordem limitada na borda inferior da zona (primeiro toque). Compra é o espelho.
"""

import numpy as np
import pandas as pd

ALVOS_R = (1, 2, 3)


def pivos(close, n):
    s = pd.Series(close)
    ant_max = s.shift(1).rolling(n).max().to_numpy()
    seg_max = s.shift(-n).rolling(n).max().to_numpy()
    ant_min = s.shift(1).rolling(n).min().to_numpy()
    seg_min = s.shift(-n).rolling(n).min().to_numpy()
    with np.errstate(invalid='ignore'):
        topo = (close > ant_max) & (close >= seg_max)
        fundo = (close < ant_min) & (close <= seg_min)
    return topo, fundo


def estrutura(df, n):
    """
    Percorre os candles em ordem, usando só pivôs já confirmados.
    Devolve arrays por candle (estado APÓS o fechamento do candle):
      tendencia (+1/-1/0), evento (+1 BOS alta, +2 CHoCH alta, -1 BOS baixa, -2 CHoCH baixa, 0),
      origem_idx (índice do fechamento de origem da pernada atual),
      extremo (máxima/mínima de pavio da origem: topo da pernada de baixa ou fundo da de alta),
      nivel_choch (fechamento que, se rompido, reverte a tendência).
    """
    c, h, l = df['close'].to_numpy(), df['high'].to_numpy(), df['low'].to_numpy()
    N = len(c)
    eh_topo, eh_fundo = pivos(c, n)

    tendencia = np.zeros(N, np.int8)
    seq_bos = np.zeros(N, np.int16)      # BOS seguidos na tendência atual (CHoCH zera)
    evento = np.zeros(N, np.int8)
    origem_idx = np.full(N, -1, np.int64)
    extremo = np.full(N, np.nan)
    nivel_choch = np.full(N, np.nan)
    nivel_rompido = np.full(N, np.nan)
    idx_rompido = np.full(N, -1, np.int64)

    trend = 0
    seq = 0
    nivel_cima, idx_cima = np.nan, -1      # fechar acima -> quebra de alta
    nivel_baixo, idx_baixo = np.nan, -1    # fechar abaixo -> quebra de baixa
    ultima_quebra = -1
    orig, ext = -1, np.nan

    for t in range(N):
        k = t - n
        if k > ultima_quebra:
            if eh_topo[k] and trend != -1:
                nivel_cima, idx_cima = c[k], k
            if eh_fundo[k] and trend != 1:
                nivel_baixo, idx_baixo = c[k], k

        if nivel_baixo == nivel_baixo and c[t] < nivel_baixo:
            ev = -1 if trend == -1 else -2
            nivel_rompido[t], idx_rompido[t] = nivel_baixo, idx_baixo
            j = idx_baixo + 1 + int(np.argmax(c[idx_baixo + 1:t + 1]))
            orig, ext = j, h[idx_baixo + 1:t + 1].max()
            seq = seq + 1 if ev == -1 else 0
            trend, ultima_quebra = -1, t
            nivel_cima, idx_cima = c[j], j
            nivel_baixo, idx_baixo = np.nan, -1
            evento[t] = ev
        elif nivel_cima == nivel_cima and c[t] > nivel_cima:
            ev = 1 if trend == 1 else 2
            nivel_rompido[t], idx_rompido[t] = nivel_cima, idx_cima
            j = idx_cima + 1 + int(np.argmin(c[idx_cima + 1:t + 1]))
            orig, ext = j, l[idx_cima + 1:t + 1].min()
            seq = seq + 1 if ev == 1 else 0
            trend, ultima_quebra = 1, t
            nivel_baixo, idx_baixo = c[j], j
            nivel_cima, idx_cima = np.nan, -1
            evento[t] = ev

        tendencia[t] = trend
        seq_bos[t] = seq
        origem_idx[t] = orig
        extremo[t] = ext
        nivel_choch[t] = nivel_cima if trend == -1 else (nivel_baixo if trend == 1 else np.nan)

    return {'tendencia': tendencia, 'seq_bos': seq_bos, 'evento': evento, 'origem_idx': origem_idx,
            'extremo': extremo, 'nivel_choch': nivel_choch, 'nivel_rompido': nivel_rompido,
            'idx_rompido': idx_rompido, 'pivo_topo': eh_topo, 'pivo_fundo': eh_fundo}


def fvgs(df):
    """FVGs conhecidos no fechamento do candle i. Devolve (i, direção, base, topo)."""
    h, l = df['high'].to_numpy(), df['low'].to_numpy()
    i = np.arange(2, len(h))
    baixa = l[i - 2] > h[i]
    alta = h[i - 2] < l[i]
    out = [(int(x), -1, h[x], l[x - 2]) for x in i[baixa]]
    out += [(int(x), 1, h[x - 2], l[x]) for x in i[alta]]
    return sorted(out)


def montar_setups(df, est, atr, espera_quebra=30, validade=240, folga_atr=0.1, entrada='borda', penetracao=0.0,
                  candidatos=None):
    """
    Para cada FVG dentro de uma pernada a favor da tendência, cria a ordem limitada e procura a execução.
    - FVG formado com a tendência já a favor: ativa no candle seguinte.
    - FVG formado no impulso que ainda vai quebrar a estrutura: ativa quando a quebra acontece
      (até espera_quebra candles depois), desde que esteja depois da origem da nova pernada.
    - A zona não pode ter sido tocada antes da ativação.
    - penetracao: fração do preço que o mercado precisa ultrapassar a entrada para a ordem executar.
    - A ordem é cancelada se: passar 'validade' candles, um fechamento atravessar a zona,
      ou a tendência mudar antes da execução.
    Dois stops são registrados: acima/abaixo do extremo da pernada (estrutura) e do FVG.
    candidatos: se for uma lista, recebe todos os FVGs válidos (ativados e não tocados antes da ativação),
    executados ou não, como (i_fvg, dir, base, topo, origem, i_ativa) — usado para o stop protegido.
    """
    o, h, l, c = (df[x].to_numpy() for x in ('open', 'high', 'low', 'close'))
    N = len(c)
    tend, orig, ext = est['tendencia'], est['origem_idx'], est['extremo']
    quebras = np.flatnonzero(est['evento'] != 0)

    setups = []
    for i, d, base, topo in fvgs(df):
        if i + 1 >= N:
            continue
        # ---- ativação ----
        if tend[i] == d and orig[i] <= i - 2:
            a = i + 1
            t_ref = i
            via_quebra = False
        else:
            prox = quebras[np.searchsorted(quebras, i + 1):]
            prox = prox[(prox <= i + espera_quebra)]
            if not len(prox):
                continue
            t_q = prox[0]
            if np.sign(est['evento'][t_q]) != d or orig[t_q] > i - 2:
                continue
            a, t_ref = t_q + 1, t_q
            via_quebra = True
        if a >= N:
            continue

        preco = (base if d == -1 else topo) if entrada == 'borda' else (base + topo) / 2
        # zona não pode ter sido tocada entre a formação e a ativação
        if a > i + 1:
            if (d == -1 and h[i + 1:a].max() >= preco) or (d == 1 and l[i + 1:a].min() <= preco):
                continue
        if candidatos is not None:
            candidatos.append((i, d, base, topo, int(orig[t_ref]), a))

        # ---- execução da ordem limitada ----
        fim = min(a + validade, N)
        # penetracao > 0: só executa se o preço ultrapassar a entrada (fila da ordem limitada)
        toque = (h[a:fim] >= preco * (1 + penetracao)) if d == -1 else (l[a:fim] <= preco * (1 - penetracao))
        if not toque.any():
            continue
        f = a + int(np.argmax(toque))
        # cancelamentos antes da execução: fechamento atravessou a zona ou a tendência mudou
        if f > a:
            fech_ant = c[a:f]
            atravessou = (fech_ant > topo).any() if d == -1 else (fech_ant < base).any()
            if atravessou or (tend[a - 1:f] != d).any():
                continue
        elif tend[a - 1] != d:
            continue

        o_idx = orig[t_ref]
        extremo_perna = ext[t_ref]
        folga = folga_atr * atr[i]
        if d == -1:
            fundo_perna = l[o_idx:i + 1].min()
            posicao = (preco - fundo_perna) / (extremo_perna - fundo_perna)
            stop_estrutura = extremo_perna + folga
            stop_fvg = h[i - 2:i + 1].max() + folga
        else:
            topo_perna = h[o_idx:i + 1].max()
            posicao = (topo_perna - preco) / (topo_perna - extremo_perna)
            stop_estrutura = extremo_perna - folga
            stop_fvg = l[i - 2:i + 1].min() - folga
        setups.append({'i_fvg': i, 'i_ativa': a, 'i_exec': f, 'dir': d, 'entrada': preco,
                       'zona_base': base, 'zona_topo': topo, 'origem': int(o_idx),
                       'extremo_perna': extremo_perna, 'posicao_perna': posicao,
                       'stop_estrutura': stop_estrutura, 'stop_fvg': stop_fvg,
                       'via_quebra': via_quebra, 't_ref': int(t_ref), 'tam_fvg_atr': (topo - base) / atr[i] if atr[i] > 0 else np.nan})
    return pd.DataFrame(setups)


def avaliar(df, setups, coluna_stop, horizonte=1440):
    """
    A partir da execução, verifica o que vem primeiro: stop ou alvos de 1R/2R/3R.
    No candle da execução só o stop conta (não se sabe se o alvo veio antes da entrada).
    Stop e alvo no mesmo candle contam como stop. Saída no stop: no stop ou na abertura (gap).
    """
    o, h, l, c = (df[x].to_numpy() for x in ('open', 'high', 'low', 'close'))
    N = len(c)
    linhas = []
    for s in setups.itertuples():
        d, e, f = s.dir, s.entrada, s.i_exec
        stop = getattr(s, coluna_stop)
        risco = d * (e - stop)
        if risco <= 0:
            linhas.append(None)
            continue
        fim = min(f + horizonte, N - 1)
        hh, ll, oo = h[f:fim + 1], l[f:fim + 1], o[f:fim + 1]
        bate_stop = (hh >= stop) if d == -1 else (ll <= stop)
        t_stop = int(np.argmax(bate_stop)) if bate_stop.any() else np.inf
        res = {'risco_pct': risco / e, 't_stop': t_stop}
        if t_stop != np.inf:
            px = stop
            if t_stop > 0:  # abriu além do stop: sai na abertura
                px = max(oo[t_stop], stop) if d == -1 else min(oo[t_stop], stop)
            res['perda_r'] = d * (px - e) / risco
        else:
            res['perda_r'] = np.nan
        for x in ALVOS_R:
            alvo = e + d * x * risco
            bate = (ll[1:] <= alvo) if d == -1 else (hh[1:] >= alvo)
            t_alvo = 1 + int(np.argmax(bate)) if bate.any() else np.inf
            res[f'acerto_{x}r'] = t_alvo < t_stop
            res[f't_{x}r'] = t_alvo
            # resultado em R de um trade com alvo xR
            if t_alvo < t_stop:
                res[f'resultado_{x}r'] = float(x)
            elif t_stop != np.inf:
                res[f'resultado_{x}r'] = res['perda_r']
            else:
                res[f'resultado_{x}r'] = d * (c[fim] - e) / risco
        linhas.append(res)
    ok = [r is not None for r in linhas]
    out = setups[ok].reset_index(drop=True)
    return pd.concat([out, pd.DataFrame([r for r in linhas if r is not None])], axis=1)


def sessao(tempos):
    """Primeiras 3 horas de Ásia (Tóquio 09h), Londres (08h) e Nova York (09h30), no horário local de cada praça."""
    t = pd.DatetimeIndex(tempos)
    rot = np.full(len(t), 'fora', dtype=object)
    for nome, tz, ini in (('Ásia', 'Asia/Tokyo', 9 * 60), ('Londres', 'Europe/London', 8 * 60),
                          ('Nova York', 'America/New_York', 9 * 60 + 30)):
        loc = t.tz_convert(tz)
        minuto = loc.hour * 60 + loc.minute
        dentro = (minuto >= ini) & (minuto < ini + 180) & (loc.dayofweek < 5)
        rot[np.asarray(dentro) & (rot == 'fora')] = nome
    return rot


def sessao_sao_paulo(tempos):
    """
    Aberturas no horário de São Paulo (dias úteis): Ásia 22h–01h, Londres 03h–07h, Nova York 10h30–13h30.
    Devolve 'Ásia', 'Londres', 'Nova York' ou 'fora'.
    """
    t = pd.DatetimeIndex(tempos).tz_convert('America/Sao_Paulo')
    m = np.asarray(t.hour * 60 + t.minute)
    dia = np.asarray(t.dayofweek)                  # 0 = segunda ... 6 = domingo
    rot = np.full(len(t), 'fora', dtype=object)
    rot[((m >= 22 * 60) & ((dia <= 3) | (dia == 6))) | ((m < 60) & (dia <= 4))] = 'Ásia'
    rot[(m >= 3 * 60) & (m < 7 * 60) & (dia <= 4)] = 'Londres'
    rot[(m >= 10 * 60 + 30) & (m < 13 * 60 + 30) & (dia <= 4)] = 'Nova York'
    return rot


def primeiro_da_pernada(setups):
    """
    1º FVG EXECUTADO de cada pernada, pela ordem de execução (regra que dá para seguir ao vivo).
    Atenção: escolher pela ordem de formação (setups.duplicated sem ordenar) olha o futuro — descarta um FVG
    que executou antes só porque um FVG mais antigo da mesma pernada vai executar depois (o que, numa venda,
    significa que o preço subiu contra a primeira operação). Esse viés inflou os estudos das seções 8 a 11.
    """
    x = setups.sort_values(['i_exec', 'i_fvg'], kind='stable')
    return x[~x.duplicated(['dir', 'origem'])].sort_index()
