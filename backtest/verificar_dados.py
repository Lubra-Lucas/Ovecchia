"""
Verificações de integridade de uma série de candles.
Não corrige nada: só aponta problemas para decidirmos o que fazer.
"""

import pandas as pd

from dados_binance import TIMEFRAMES_MS


def verificar(df, timeframe, salto_suspeito=0.15):
    passo = TIMEFRAMES_MS[timeframe]
    print(f"Candles: {len(df)}  |  {df['time'].iloc[0]}  ->  {df['time'].iloc[-1]}")

    # 1. Ordem e duplicados
    print(f"\n[1] Ordem cronológica: {'OK' if df['open_time'].is_monotonic_increasing else 'FORA DE ORDEM'}")
    dup = df['open_time'].duplicated().sum()
    print(f"[2] Duplicados: {dup}")

    # 2. Alinhamento ao timeframe (ex.: H1 sempre começando em hh:00)
    desalinhados = (df['open_time'] % passo != 0).sum() if timeframe != "1d" else (df['open_time'] % 86_400_000 != 0).sum()
    print(f"[3] Candles desalinhados do timeframe: {desalinhados}")

    # 3. Buracos
    diffs = df['open_time'].diff().iloc[1:]
    buracos = df.loc[diffs[diffs != passo].index]
    esperados = (df['open_time'].iloc[-1] - df['open_time'].iloc[0]) // passo + 1
    print(f"[4] Candles esperados: {esperados}  recebidos: {len(df)}  faltando: {esperados - len(df)}")
    if len(buracos):
        print(f"    {len(buracos)} buraco(s):")
        for idx in buracos.index:
            antes = df['time'].iloc[idx - 1]
            depois = df['time'].iloc[idx]
            faltam = int(diffs.loc[idx] // passo) - 1
            print(f"      {antes}  ->  {depois}   ({faltam} candles faltando)")

    # 4. Coerência OHLC
    ohlc_ruim = df[(df['high'] < df[['open', 'close', 'low']].max(axis=1)) |
                   (df['low'] > df[['open', 'close', 'high']].min(axis=1))]
    print(f"\n[5] Candles com OHLC incoerente (máx < abertura/fech. ou mín > abertura/fech.): {len(ohlc_ruim)}")
    nao_positivos = (df[['open', 'high', 'low', 'close']] <= 0).any(axis=1).sum()
    print(f"[6] Preços zero ou negativos: {nao_positivos}")
    vol_zero = (df['volume'] == 0).sum()
    print(f"[7] Candles com volume zero: {vol_zero}")
    if vol_zero:
        print(df.loc[df['volume'] == 0, ['time', 'open', 'high', 'low', 'close', 'trades']].head(10).to_string(index=False))

    # 5. Saltos anormais entre fechamento anterior e abertura seguinte, e candles gigantes
    gap_abertura = (df['open'] / df['close'].shift(1) - 1).abs()
    amplitude = df['high'] / df['low'] - 1
    print(f"\n[8] Saltos fechamento->abertura > {salto_suspeito:.0%}: {(gap_abertura > salto_suspeito).sum()}")
    print("    Maiores saltos fechamento->abertura:")
    for idx in gap_abertura.nlargest(5).index:
        print(f"      {df['time'].iloc[idx]}  {gap_abertura.iloc[idx]:.2%}  (fech. ant. {df['close'].iloc[idx-1]:.2f} -> abert. {df['open'].iloc[idx]:.2f})")
    print("    Maiores amplitudes (máx/mín) num candle:")
    for idx in amplitude.nlargest(5).index:
        r = df.iloc[idx]
        print(f"      {r['time']}  {amplitude.iloc[idx]:.2%}  (mín {r['low']:.2f}  máx {r['high']:.2f})")
    return buracos


def agregar(df, regra):
    """Reamostra candles para um timeframe maior (ex.: '1D', '4h')."""
    return df.set_index('time').resample(regra, label='left', closed='left').agg(
        {'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'}
    ).dropna()


def comparar(a, b, nome_a, nome_b, tol=1e-9):
    """Compara duas séries OHLCV candle a candle, nos horários em comum."""
    a = a.set_index('time') if 'time' in a.columns else a
    b = b.set_index('time') if 'time' in b.columns else b
    comum = a.index.intersection(b.index)
    print(f"\nComparando {nome_a} x {nome_b}: {len(comum)} candles em comum")
    for c in ['open', 'high', 'low', 'close', 'volume']:
        dif_rel = ((a.loc[comum, c] - b.loc[comum, c]).abs() / b.loc[comum, c].abs().clip(lower=1e-12))
        n_dif = (dif_rel > tol).sum()
        print(f"  {c:<7} divergentes: {n_dif:>5}   maior diferença relativa: {dif_rel.max():.2e}")
    return comum
