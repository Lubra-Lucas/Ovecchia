"""
Coleta de candles (klines) do spot da Binance a partir de fontes oficiais e públicas.

Fontes:
  - data.binance.vision      -> arquivos mensais oficiais (ZIP + checksum SHA-256)
  - data-api.binance.vision  -> API pública, usada só para o mês corrente

Nenhuma chave de API é necessária.
"""

import hashlib
import io
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

ARQUIVOS_URL = "https://data.binance.vision/data/spot/monthly/klines"
API_URL = "https://data-api.binance.vision/api/v3/klines"
CACHE_DIR = Path(__file__).parent / "cache"

COLUNAS = [
    "open_time", "open", "high", "low", "close", "volume",
    "close_time", "quote_volume", "trades", "taker_buy_base", "taker_buy_quote", "ignore",
]

# Duração exata de cada timeframe (em milissegundos)
TIMEFRAMES_MS = {
    "1m": 60_000,
    "5m": 300_000,
    "15m": 900_000,
    "1h": 3_600_000,
    "4h": 14_400_000,
    "1d": 86_400_000,
}


def _get(url, params=None, tentativas=4):
    """GET com novas tentativas em caso de falha de rede (2s, 4s, 8s)."""
    for i in range(tentativas):
        try:
            r = requests.get(url, params=params, timeout=60)
            if r.status_code == 404:
                return None
            r.raise_for_status()
            return r
        except requests.RequestException:
            if i == tentativas - 1:
                raise
            time.sleep(2 ** (i + 1))


def _para_ms(serie):
    """A Binance passou a usar microssegundos nos arquivos de 2025 em diante; normaliza tudo para ms."""
    serie = serie.astype("int64")
    return serie.where(serie < 10**14, serie // 1000)


def _ler_csv(conteudo):
    df = pd.read_csv(io.BytesIO(conteudo), header=None, names=COLUNAS)
    # Alguns arquivos trazem cabeçalho; nesse caso a primeira linha não é numérica
    if not str(df.iloc[0, 0]).isdigit():
        df = df.iloc[1:]
    return df


def baixar_mes(simbolo, timeframe, ano, mes):
    """
    Baixa um arquivo mensal oficial, confere o SHA-256 e guarda em cache.
    Retorna (DataFrame, status) onde status é 'ok', 'cache' ou 'inexistente'.
    """
    nome = f"{simbolo}-{timeframe}-{ano}-{mes:02d}.zip"
    destino = CACHE_DIR / simbolo / timeframe / nome
    if destino.exists():
        with zipfile.ZipFile(destino) as z:
            return _ler_csv(z.read(z.namelist()[0])), "cache"

    url = f"{ARQUIVOS_URL}/{simbolo}/{timeframe}/{nome}"
    resp_checksum = _get(url + ".CHECKSUM")
    if resp_checksum is None:
        return None, "inexistente"
    checksum_oficial = resp_checksum.text.split()[0].strip().lower()

    conteudo = _get(url).content
    checksum_calculado = hashlib.sha256(conteudo).hexdigest()
    if checksum_calculado != checksum_oficial:
        raise ValueError(f"Checksum não confere em {nome}: {checksum_calculado} != {checksum_oficial}")

    destino.parent.mkdir(parents=True, exist_ok=True)
    destino.write_bytes(conteudo)
    with zipfile.ZipFile(io.BytesIO(conteudo)) as z:
        return _ler_csv(z.read(z.namelist()[0])), "ok"


def baixar_api(simbolo, timeframe, inicio_ms, fim_ms=None):
    """Baixa candles pela API pública, paginando de 1000 em 1000."""
    linhas = []
    cursor = inicio_ms
    while True:
        params = {"symbol": simbolo, "interval": timeframe, "startTime": cursor, "limit": 1000}
        if fim_ms is not None:
            params["endTime"] = fim_ms
        lote = _get(API_URL, params).json()
        if not lote:
            break
        linhas.extend(lote)
        if len(lote) < 1000:
            break
        cursor = lote[-1][0] + TIMEFRAMES_MS[timeframe]
    return pd.DataFrame(linhas, columns=COLUNAS)


def _meses(inicio, fim):
    ano, mes = inicio
    while (ano, mes) <= fim:
        yield ano, mes
        mes += 1
        if mes > 12:
            ano, mes = ano + 1, 1


def carregar(simbolo, timeframe, desde=(2017, 8), verbose=True):
    """
    Monta o histórico completo: arquivos mensais oficiais até o mês anterior
    e a API pública para o mês corrente. Só devolve candles já fechados.
    """
    agora = datetime.now(timezone.utc)
    mes_anterior = (agora.year, agora.month - 1) if agora.month > 1 else (agora.year - 1, 12)

    partes, relatorio = [], []
    for ano, mes in _meses(desde, mes_anterior):
        df_mes, status = baixar_mes(simbolo, timeframe, ano, mes)
        relatorio.append((f"{ano}-{mes:02d}", status, 0 if df_mes is None else len(df_mes)))
        if df_mes is not None:
            partes.append(df_mes)

    inicio_mes_atual = int(datetime(agora.year, agora.month, 1, tzinfo=timezone.utc).timestamp() * 1000)
    df_api = baixar_api(simbolo, timeframe, inicio_mes_atual)
    relatorio.append((f"{agora.year}-{agora.month:02d} (API)", "ok", len(df_api)))
    partes.append(df_api)

    df = pd.concat(partes, ignore_index=True)
    df["open_time"] = _para_ms(df["open_time"])
    df["close_time"] = _para_ms(df["close_time"])
    for c in ["open", "high", "low", "close", "volume", "quote_volume"]:
        df[c] = df[c].astype(float)
    df["trades"] = df["trades"].astype("int64")

    # Descarta o candle ainda em formação
    agora_ms = int(time.time() * 1000)
    df = df[df["close_time"] < agora_ms]

    df["time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    df = df[["time", "open_time", "open", "high", "low", "close", "volume", "quote_volume", "trades"]]

    if verbose:
        for periodo, status, n in relatorio:
            print(f"  {periodo:<16} {status:<12} {n:>7} candles")
    return df.reset_index(drop=True)
