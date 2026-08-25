#!/usr/bin/env python3
"""
Dhan 1-Hour SELL Signal Scanner + Historical Backtester

Purpose
-------
Scans every currently option-eligible NSE stock (derived from Dhan's
OPTSTK instrument master), downloads 60-minute underlying equity candles,
and finds the bearish setup discussed from the user's NTPC chart.

SELL setup (all conditions by default):
1. RSI(14) >= 50
2. RSI is lower than the previous closed candle
3. Stoch RSI %K crosses below %D
4. %K was >= 80 within the previous 5 candles
5. Close is within 0.5% of the previous 5-candle swing high
6. Signal candle is bearish (Close < Open)

No look-ahead is used for the signal itself. Historical outcome columns
use candles AFTER the signal only.

The script does NOT place orders.
"""

import argparse
import io
import os
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv


# -----------------------------
# User configuration
# -----------------------------
load_dotenv(Path(__file__).resolve().parent / ".env")

DHAN_ACCESS_TOKEN = os.getenv("DHAN_ACCESS_TOKEN")
DHAN_CLIENT_ID = os.getenv("DHAN_CLIENT_ID")

INSTRUMENT_MASTER_URL = (
    "https://images.dhan.co/api-data/api-scrip-master-detailed.csv"
)
HISTORICAL_URL = "https://api.dhan.co/v2/charts/intraday"

TIMEZONE = "Asia/Kolkata"
INTERVAL = "60"
EXCHANGE_SEGMENT = "NSE_EQ"
INSTRUMENT = "EQUITY"

RSI_PERIOD = 14
STOCH_RSI_PERIOD = 14
STOCH_K_PERIOD = 3
STOCH_D_PERIOD = 3

STOCH_OVERBOUGHT = 80.0
RSI_MINIMUM = 50.0
SWING_LOOKBACK = 5
SWING_HIGH_TOLERANCE_PCT = 0.50
RECENT_OVERBOUGHT_BARS = 5

# Backtest diagnostics:
# number of future 1-hour candles to inspect after the signal.
FORWARD_BARS = 6

OUTPUT_DIR = Path("dhan_sell_signal_output")
CACHE_DIR = OUTPUT_DIR / "cache"


def require_credentials():
    if not DHAN_ACCESS_TOKEN or not DHAN_CLIENT_ID:
        raise SystemExit(
            "Missing DHAN_ACCESS_TOKEN or DHAN_CLIENT_ID in .env."
        )


def http_session():
    s = requests.Session()
    s.headers.update(
        {
            "Accept": "application/json",
            "Content-Type": "application/json",
            "access-token": DHAN_ACCESS_TOKEN,
        }
    )
    if DHAN_CLIENT_ID:
        s.headers["client-id"] = DHAN_CLIENT_ID
    return s


def fetch_instrument_master(session):
    print("Downloading Dhan instrument master...")
    r = session.get(INSTRUMENT_MASTER_URL, timeout=60)
    r.raise_for_status()
    return pd.read_csv(io.BytesIO(r.content), low_memory=False)


def get_optionable_stocks(master):
    """
    Build the current stock-option universe from Dhan OPTSTK contracts.

    We intentionally use the option contracts themselves to decide which
    stocks are option eligible, then use the underlying NSE_EQ security ID
    for historical price data.
    """
    required = {
        "EXCH_ID",
        "SEGMENT",
        "INSTRUMENT",
        "UNDERLYING_SECURITY_ID",
        "UNDERLYING_SYMBOL",
    }
    missing = required - set(master.columns)
    if missing:
        raise RuntimeError(
            f"Instrument master is missing columns: {sorted(missing)}"
        )

    m = master.copy()

    opt = m[
        (m["EXCH_ID"].astype(str).str.upper() == "NSE")
        & (m["SEGMENT"].astype(str).str.upper() == "D")
        & (m["INSTRUMENT"].astype(str).str.upper() == "OPTSTK")
    ].copy()

    opt["UNDERLYING_SYMBOL"] = (
        opt["UNDERLYING_SYMBOL"].astype(str).str.strip().str.upper()
    )
    opt["UNDERLYING_SECURITY_ID"] = pd.to_numeric(
        opt["UNDERLYING_SECURITY_ID"], errors="coerce"
    )

    opt = opt.dropna(subset=["UNDERLYING_SECURITY_ID"])
    opt = opt[opt["UNDERLYING_SYMBOL"].ne("")]

    stocks = (
        opt[
            ["UNDERLYING_SYMBOL", "UNDERLYING_SECURITY_ID"]
        ]
        .drop_duplicates()
        .sort_values("UNDERLYING_SYMBOL")
        .reset_index(drop=True)
    )

    stocks["UNDERLYING_SECURITY_ID"] = (
        stocks["UNDERLYING_SECURITY_ID"].astype(int).astype(str)
    )

    return stocks


def date_chunks(start_date, end_date, max_days=80):
    """
    Dhan intraday endpoint permits max 90 days per request.
    Use 80-day chunks to stay safely below that limit.
    """
    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)

    cur = start
    while cur <= end:
        chunk_end = min(cur + pd.Timedelta(days=max_days - 1), end)
        yield cur.date(), chunk_end.date()
        cur = chunk_end + pd.Timedelta(days=1)


def fetch_intraday(session, security_id, from_date, to_date):
    payload = {
        "securityId": str(security_id),
        "exchangeSegment": EXCHANGE_SEGMENT,
        "instrument": INSTRUMENT,
        "interval": INTERVAL,
        "oi": False,
        "fromDate": f"{from_date} 09:00:00",
        # Dhan's toDate is the requested end boundary. Add one day so the
        # requested end date is included.
        "toDate": f"{to_date} 23:59:59",
    }

    for attempt in range(1, 4):
        try:
            r = session.post(HISTORICAL_URL, json=payload, timeout=60)

            if r.status_code == 429:
                wait = 2 * attempt
                print(f"  Rate limited; waiting {wait}s...")
                time.sleep(wait)
                continue

            r.raise_for_status()
            data = r.json()

            if not isinstance(data, dict) or "open" not in data:
                return pd.DataFrame()

            n = len(data.get("timestamp", []))
            if n == 0:
                return pd.DataFrame()

            df = pd.DataFrame(
                {
                    "timestamp": data["timestamp"],
                    "open": data["open"],
                    "high": data["high"],
                    "low": data["low"],
                    "close": data["close"],
                    "volume": data.get("volume", [np.nan] * n),
                }
            )

            df["timestamp"] = pd.to_datetime(
                df["timestamp"], unit="s", utc=True
            ).dt.tz_convert(TIMEZONE)

            for col in ["open", "high", "low", "close", "volume"]:
                df[col] = pd.to_numeric(df[col], errors="coerce")

            df = (
                df.dropna(subset=["timestamp", "open", "high", "low", "close"])
                .sort_values("timestamp")
                .drop_duplicates("timestamp")
                .reset_index(drop=True)
            )
            return df

        except requests.RequestException as exc:
            if attempt == 3:
                print(f"  API error: {exc}")
                return pd.DataFrame()
            time.sleep(attempt)

    return pd.DataFrame()


def fetch_stock_history(session, security_id, start_date, end_date):
    parts = []
    for chunk_start, chunk_end in date_chunks(start_date, end_date):
        part = fetch_intraday(
            session, security_id, chunk_start, chunk_end
        )
        if not part.empty:
            parts.append(part)

        # Keep comfortably below the Dhan data API rate limit.
        time.sleep(0.22)

    if not parts:
        return pd.DataFrame()

    df = (
        pd.concat(parts, ignore_index=True)
        .sort_values("timestamp")
        .drop_duplicates("timestamp")
        .reset_index(drop=True)
    )
    return df


def calculate_rsi(close, period=14):
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(
        alpha=1 / period,
        min_periods=period,
        adjust=False,
    ).mean()
    avg_loss = loss.ewm(
        alpha=1 / period,
        min_periods=period,
        adjust=False,
    ).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def add_indicators(df):
    df = df.copy()

    df["rsi"] = calculate_rsi(df["close"], RSI_PERIOD)

    rsi_low = df["rsi"].rolling(STOCH_RSI_PERIOD).min()
    rsi_high = df["rsi"].rolling(STOCH_RSI_PERIOD).max()
    denom = (rsi_high - rsi_low).replace(0, np.nan)

    raw_stoch = ((df["rsi"] - rsi_low) / denom) * 100

    df["stoch_k"] = raw_stoch.rolling(STOCH_K_PERIOD).mean()
    df["stoch_d"] = df["stoch_k"].rolling(STOCH_D_PERIOD).mean()

    # Previous candles only: avoids using the signal candle's high.
    df["previous_swing_high"] = (
        df["high"].rolling(SWING_LOOKBACK).max().shift(1)
    )

    df["distance_from_swing_high_pct"] = (
        (df["previous_swing_high"] - df["close"])
        / df["previous_swing_high"]
    ) * 100

    df["bearish_candle"] = df["close"] < df["open"]

    df["rsi_weakening"] = df["rsi"] < df["rsi"].shift(1)

    df["stoch_bearish_cross"] = (
        (df["stoch_k"] < df["stoch_d"])
        & (df["stoch_k"].shift(1) >= df["stoch_d"].shift(1))
    )

    df["recent_stoch_overbought"] = (
        df["stoch_k"]
        .rolling(RECENT_OVERBOUGHT_BARS)
        .max()
        .ge(STOCH_OVERBOUGHT)
    )

    df["near_swing_high"] = (
        df["distance_from_swing_high_pct"]
        .between(-0.25, SWING_HIGH_TOLERANCE_PCT)
    )

    # Strict SELL signal.
    df["sell_signal"] = (
        df["rsi"].ge(RSI_MINIMUM)
        & df["rsi_weakening"]
        & df["stoch_bearish_cross"]
        & df["recent_stoch_overbought"]
        & df["near_swing_high"]
        & df["bearish_candle"]
    )

    return df


def signal_rows_for_dates(df, start_date, end_date):
    """
    Signal is evaluated only on completed candles whose local calendar date
    falls inside the requested backtest window.
    """
    if df.empty:
        return pd.DataFrame()

    local_dates = df["timestamp"].dt.date
    mask = (
        (local_dates >= pd.Timestamp(start_date).date())
        & (local_dates <= pd.Timestamp(end_date).date())
        & df["sell_signal"]
    )

    return df.loc[mask].copy()


def add_forward_metrics(df, signal_df):
    """
    Historical diagnostics only. These metrics use candles AFTER the signal,
    so they are never used to create the signal.
    """
    if signal_df.empty:
        return signal_df

    rows = []

    for idx in signal_df.index:
        signal = df.loc[idx]

        future = df.loc[
            df.index > idx
        ].head(FORWARD_BARS)

        if future.empty:
            next_open = np.nan
            min_low = np.nan
            max_high = np.nan
        else:
            next_open = future.iloc[0]["open"]
            min_low = future["low"].min()
            max_high = future["high"].max()

        entry = next_open

        if pd.isna(entry):
            mfe_pct = np.nan
            mae_pct = np.nan
        else:
            # For a bearish underlying signal, favorable movement is down.
            mfe_pct = ((entry - min_low) / entry) * 100
            mae_pct = ((max_high - entry) / entry) * 100

        row = {
            "signal_time": signal["timestamp"],
            "entry_next_1h_open": entry,
            "signal_close": signal["close"],
            "signal_high": signal["high"],
            "rsi": signal["rsi"],
            "stoch_k": signal["stoch_k"],
            "stoch_d": signal["stoch_d"],
            "previous_swing_high": signal["previous_swing_high"],
            "mfe_pct_next_bars": mfe_pct,
            "mae_pct_next_bars": mae_pct,
            "forward_bars_checked": len(future),
        }

        rows.append(row)

    return pd.DataFrame(rows)


def scan_stock(session, symbol, security_id, start_date, end_date):
    # Warm-up is required for RSI/Stoch RSI/swing calculations.
    warmup_start = (
        pd.Timestamp(start_date) - pd.Timedelta(days=30)
    ).date()

    df = fetch_stock_history(
        session,
        security_id,
        warmup_start,
        end_date,
    )

    if df.empty:
        return pd.DataFrame()

    df = add_indicators(df)

    signals = signal_rows_for_dates(
        df,
        start_date,
        end_date,
    )

    if signals.empty:
        return pd.DataFrame()

    out = add_forward_metrics(df, signals)

    if out.empty:
        return out

    out.insert(0, "symbol", symbol)
    out.insert(1, "security_id", security_id)

    return out


def parse_args():
    parser = argparse.ArgumentParser(
        description="Scan all Dhan NSE stock-option underlyings for 1H SELL signals."
    )

    parser.add_argument(
        "--start-date",
        required=True,
        help="Backtest start date, e.g. 2026-07-01",
    )
    parser.add_argument(
        "--end-date",
        required=True,
        help="Backtest end date, e.g. 2026-08-25",
    )
    parser.add_argument(
        "--symbols",
        default="",
        help="Optional comma-separated symbols, e.g. NTPC,RELIANCE,SBIN",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Reserved for future stricter filters.",
    )

    return parser.parse_args()


def main():
    args = parse_args()
    require_credentials()

    start_date = pd.Timestamp(args.start_date).date()
    end_date = pd.Timestamp(args.end_date).date()

    if end_date < start_date:
        raise SystemExit("end-date cannot be earlier than start-date.")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    session = http_session()

    master = fetch_instrument_master(session)
    stocks = get_optionable_stocks(master)

    if args.symbols.strip():
        requested = {
            x.strip().upper()
            for x in args.symbols.split(",")
            if x.strip()
        }
        stocks = stocks[
            stocks["UNDERLYING_SYMBOL"].isin(requested)
        ].copy()

    print(
        f"\nOption-eligible NSE stock underlyings to scan: "
        f"{len(stocks)}"
    )

    if stocks.empty:
        raise SystemExit("No option-eligible stocks found.")

    all_results = []
    errors = []

    for i, row in stocks.iterrows():
        symbol = row["UNDERLYING_SYMBOL"]
        security_id = row["UNDERLYING_SECURITY_ID"]

        print(
            f"[{i + 1}/{len(stocks)}] "
            f"{symbol} ({security_id})"
        )

        try:
            result = scan_stock(
                session,
                symbol,
                security_id,
                start_date,
                end_date,
            )

            if not result.empty:
                print(
                    f"    -> {len(result)} SELL signal(s)"
                )
                all_results.append(result)
            else:
                print("    -> no signal")

        except Exception as exc:
            print(f"    -> ERROR: {exc}")
            errors.append(
                {
                    "symbol": symbol,
                    "security_id": security_id,
                    "error": str(exc),
                }
            )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    if all_results:
        results = pd.concat(
            all_results,
            ignore_index=True,
        ).sort_values(
            ["signal_time", "symbol"]
        )

        result_file = (
            OUTPUT_DIR
            / f"sell_signals_{start_date}_{end_date}_{timestamp}.csv"
        )
        results.to_csv(result_file, index=False)

        print("\n" + "=" * 80)
        print("SELL SIGNALS FOUND")
        print("=" * 80)

        display_columns = [
            "symbol",
            "signal_time",
            "signal_close",
            "rsi",
            "stoch_k",
            "stoch_d",
            "previous_swing_high",
            "entry_next_1h_open",
            "mfe_pct_next_bars",
            "mae_pct_next_bars",
        ]

        print(
            results[display_columns]
            .to_string(index=False)
        )

        print(
            f"\nSaved: {result_file.resolve()}"
        )
        print(
            f"Total SELL signals: {len(results)}"
        )
        print(
            f"Unique stocks: {results['symbol'].nunique()}"
        )

    else:
        print("\nNo SELL signals found for the selected dates.")

    if errors:
        error_file = (
            OUTPUT_DIR
            / f"errors_{start_date}_{end_date}_{timestamp}.csv"
        )
        pd.DataFrame(errors).to_csv(
            error_file,
            index=False,
        )
        print(
            f"Stocks with API/errors: {len(errors)}"
        )
        print(
            f"Error report: {error_file.resolve()}"
        )


if __name__ == "__main__":
    main()