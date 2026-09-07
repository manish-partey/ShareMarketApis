#!/usr/bin/env python3
"""
Dhan NSE stock-F&O daily breakout/breakdown scanner.

Signals are generated from completed daily NSE equity candles:
- BREAKOUT_CONFIRMED: close crosses above the previous lookback day's high.
- BREAKDOWN_CONFIRMED: close crosses below the previous lookback day's low.
- BREAKOUT_POTENTIAL: daily high reaches the previous range high.
- BREAKDOWN_POTENTIAL: daily low reaches the previous range low.

The current candle is never used to calculate its own reference range.
This script only reports signals; it does not place orders.
"""

import argparse
import io
import os
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv


TIMEZONE = "Asia/Kolkata"
IST = ZoneInfo(TIMEZONE)
INSTRUMENT_MASTER_URL = (
    "https://images.dhan.co/api-data/api-scrip-master-detailed.csv"
)
HISTORICAL_URL = "https://api.dhan.co/v2/charts/historical"
EXCHANGE_SEGMENT = "NSE_EQ"
INSTRUMENT = "EQUITY"
REQUEST_DELAY_SECONDS = 0.25
MAX_RETRIES = 4
DEFAULT_LOOKBACK = 20
DEFAULT_HISTORY_DAYS = 120
OUTPUT_DIR = Path(__file__).resolve().parent / "dhan_daily_signal_output"

load_dotenv(Path(__file__).resolve().parent / ".env")
DHAN_ACCESS_TOKEN = os.getenv("DHAN_ACCESS_TOKEN")
DHAN_CLIENT_ID = os.getenv("DHAN_CLIENT_ID")


def require_credentials():
    if not DHAN_ACCESS_TOKEN or not DHAN_CLIENT_ID:
        raise SystemExit(
            "Missing DHAN_ACCESS_TOKEN or DHAN_CLIENT_ID in .env."
        )


def http_session():
    session = requests.Session()
    session.headers.update(
        {
            "Accept": "application/json",
            "Content-Type": "application/json",
            "access-token": DHAN_ACCESS_TOKEN,
            "client-id": DHAN_CLIENT_ID,
        }
    )
    return session


def fetch_instrument_master(session):
    print("Downloading Dhan instrument master...")
    response = session.get(INSTRUMENT_MASTER_URL, timeout=60)
    response.raise_for_status()
    return pd.read_csv(io.BytesIO(response.content), low_memory=False)


def get_fno_universe(master):
    """Return unique NSE equity underlyings with stock F&O contracts."""
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

    fno = master[
        (master["EXCH_ID"].astype(str).str.upper() == "NSE")
        & (master["SEGMENT"].astype(str).str.upper() == "D")
        & master["INSTRUMENT"].astype(str).str.upper().isin(
            ["FUTSTK", "OPTSTK"]
        )
    ].copy()

    fno["UNDERLYING_SYMBOL"] = (
        fno["UNDERLYING_SYMBOL"].astype(str).str.strip().str.upper()
    )
    fno = fno[~fno["UNDERLYING_SYMBOL"].str.contains("NSETEST")]
    fno["UNDERLYING_SECURITY_ID"] = pd.to_numeric(
        fno["UNDERLYING_SECURITY_ID"], errors="coerce"
    )

    universe = (
        fno.dropna(subset=["UNDERLYING_SECURITY_ID"])
        .loc[:, ["UNDERLYING_SYMBOL", "UNDERLYING_SECURITY_ID"]]
        .query("UNDERLYING_SYMBOL != ''")
        .drop_duplicates(subset=["UNDERLYING_SECURITY_ID"])
        .sort_values("UNDERLYING_SYMBOL")
        .reset_index(drop=True)
    )
    universe["UNDERLYING_SECURITY_ID"] = (
        universe["UNDERLYING_SECURITY_ID"].astype(int).astype(str)
    )
    return universe


def fetch_daily_history(session, security_id, start_date, end_date):
    payload = {
        "securityId": str(security_id),
        "exchangeSegment": EXCHANGE_SEGMENT,
        "instrument": INSTRUMENT,
        "oi": False,
        "fromDate": start_date.isoformat(),
        "toDate": end_date.isoformat(),
    }

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = session.post(
                HISTORICAL_URL,
                json=payload,
                timeout=60,
            )
            if response.status_code == 429:
                if attempt == MAX_RETRIES:
                    response.raise_for_status()
                wait_seconds = 2 ** (attempt - 1)
                print(f"  Rate limited; waiting {wait_seconds}s...")
                time.sleep(wait_seconds)
                continue

            response.raise_for_status()
            data = response.json()
            timestamps = data.get("timestamp", [])
            if not timestamps:
                return pd.DataFrame()

            row_count = len(timestamps)
            df = pd.DataFrame(
                {
                    "timestamp": timestamps,
                    "open": data.get("open", [np.nan] * row_count),
                    "high": data.get("high", [np.nan] * row_count),
                    "low": data.get("low", [np.nan] * row_count),
                    "close": data.get("close", [np.nan] * row_count),
                    "volume": data.get("volume", [np.nan] * row_count),
                }
            )
            df["timestamp"] = pd.to_datetime(
                df["timestamp"], unit="s", utc=True
            ).dt.tz_convert(TIMEZONE)

            for column in ["open", "high", "low", "close", "volume"]:
                df[column] = pd.to_numeric(df[column], errors="coerce")

            return (
                df.dropna(
                    subset=["timestamp", "open", "high", "low", "close"]
                )
                .sort_values("timestamp")
                .drop_duplicates("timestamp")
                .reset_index(drop=True)
            )
        except requests.RequestException:
            if attempt == MAX_RETRIES:
                raise
            time.sleep(attempt)

    return pd.DataFrame()


def add_breakout_indicators(df, lookback):
    """Add prior-range levels plus potential and confirmed signals."""
    result = df.copy()
    result["previous_range_high"] = (
        result["high"].rolling(lookback, min_periods=lookback).max().shift(1)
    )
    result["previous_range_low"] = (
        result["low"].rolling(lookback, min_periods=lookback).min().shift(1)
    )

    result["breakout_reached"] = (
        result["high"].ge(result["previous_range_high"])
    )
    result["breakdown_reached"] = (
        result["low"].le(result["previous_range_low"])
    )
    result["breakout_confirmed"] = (
        result["close"].gt(result["previous_range_high"])
        & result["close"].shift(1).le(result["previous_range_high"])
    )
    result["breakdown_confirmed"] = (
        result["close"].lt(result["previous_range_low"])
        & result["close"].shift(1).ge(result["previous_range_low"])
    )
    result["signal"] = np.select(
        [result["breakout_confirmed"], result["breakdown_confirmed"]],
        ["BREAKOUT_CONFIRMED", "BREAKDOWN_CONFIRMED"],
        default="",
    )
    return result


def scan_stock(
    session,
    symbol,
    security_id,
    start_date,
    end_date,
    lookback,
    mode,
    latest_only,
):
    warmup_start = (
        pd.Timestamp(start_date) - pd.Timedelta(days=max(lookback * 3, 60))
    ).date()
    history = fetch_daily_history(
        session,
        security_id,
        warmup_start,
        end_date,
    )
    if history.empty:
        return pd.DataFrame()

    history = add_breakout_indicators(history, lookback)
    local_dates = history["timestamp"].dt.date
    if latest_only:
        signals = history[local_dates <= end_date].tail(1).copy()
    else:
        signals = history[local_dates.between(start_date, end_date)].copy()

    if mode == "potential":
        signals["signal"] = np.select(
            [signals["breakout_reached"], signals["breakdown_reached"]],
            ["BREAKOUT_POTENTIAL", "BREAKDOWN_POTENTIAL"],
            default="",
        )

    signals = signals[signals["signal"].ne("")].copy()
    if signals.empty:
        return pd.DataFrame()

    signals["distance_from_range_pct"] = np.where(
        signals["signal"].str.startswith("BREAKOUT"),
        (signals["close"] - signals["previous_range_high"])
        / signals["previous_range_high"]
        * 100,
        (signals["previous_range_low"] - signals["close"])
        / signals["previous_range_low"]
        * 100,
    )
    signals.insert(0, "symbol", symbol)
    signals.insert(1, "security_id", security_id)
    return signals[
        [
            "symbol",
            "security_id",
            "timestamp",
            "signal",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "previous_range_high",
            "previous_range_low",
            "distance_from_range_pct",
        ]
    ]


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Scan NSE stock-F&O underlyings for daily range breakouts "
            "and breakdowns."
        )
    )
    parser.add_argument(
        "--start-date",
        help="First signal date in YYYY-MM-DD format.",
    )
    parser.add_argument(
        "--end-date",
        help="Last signal date in YYYY-MM-DD format. Defaults to today in IST.",
    )
    parser.add_argument(
        "--lookback",
        type=int,
        default=DEFAULT_LOOKBACK,
        help="Number of previous daily candles for the range (default: 20).",
    )
    parser.add_argument(
        "--symbols",
        default="",
        help="Optional comma-separated symbols, e.g. RELIANCE,SBIN,NTPC.",
    )
    parser.add_argument(
        "--mode",
        choices=["potential", "confirmed"],
        default="potential",
        help=(
            "potential reports a high/low reaching the prior range; "
            "confirmed requires a close beyond it (default: potential)."
        ),
    )
    return parser.parse_args()


def main():
    args = parse_args()
    require_credentials()

    if args.lookback < 2:
        raise SystemExit("--lookback must be at least 2.")

    end_date = (
        pd.Timestamp(args.end_date).date()
        if args.end_date
        else datetime.now(IST).date()
    )
    latest_only = not args.start_date and not args.end_date
    start_date = (
        pd.Timestamp(args.start_date).date()
        if args.start_date
        else (
            pd.Timestamp(end_date)
            - pd.Timedelta(days=DEFAULT_HISTORY_DAYS)
        ).date()
    )
    if end_date < start_date:
        raise SystemExit("--end-date cannot be earlier than --start-date.")

    output_dir = OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    session = http_session()
    stocks = get_fno_universe(fetch_instrument_master(session))

    if args.symbols.strip():
        requested = {
            symbol.strip().upper()
            for symbol in args.symbols.split(",")
            if symbol.strip()
        }
        stocks = stocks[stocks["UNDERLYING_SYMBOL"].isin(requested)]

    if stocks.empty:
        raise SystemExit("No option-eligible stocks found for the selection.")

    print(
        f"Scanning {len(stocks)} NSE stock-F&O underlyings from "
        f"{start_date} through {end_date} using a {args.lookback}-day range..."
    )

    all_signals = []
    errors = []
    for index, row in stocks.iterrows():
        symbol = row["UNDERLYING_SYMBOL"]
        security_id = row["UNDERLYING_SECURITY_ID"]
        print(f"[{index + 1}/{len(stocks)}] {symbol} ({security_id})")
        try:
            signals = scan_stock(
                session,
                symbol,
                security_id,
                start_date,
                end_date,
                args.lookback,
                args.mode,
                latest_only,
            )
            if signals.empty:
                print("    -> no signal")
            else:
                print(f"    -> {len(signals)} signal(s)")
                all_signals.append(signals)
        except (
            requests.RequestException,
            ValueError,
            KeyError,
            TypeError,
            IndexError,
        ) as exc:
            print(f"    -> ERROR: {exc}")
            errors.append(
                {"symbol": symbol, "security_id": security_id, "error": str(exc)}
            )
        time.sleep(REQUEST_DELAY_SECONDS)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if all_signals:
        results = pd.concat(all_signals, ignore_index=True).sort_values(
            ["timestamp", "symbol"]
        )
        result_file = output_dir / (
            f"daily_breakout_breakdown_{start_date}_{end_date}_{timestamp}.csv"
        )
        results.to_csv(result_file, index=False)
        print("\nDAILY BREAKOUT/BREAKDOWN SIGNALS")
        print(results.to_string(index=False))
        print(f"\nSaved: {result_file}")
        print(f"Total signals: {len(results)}")
    else:
        print("\nNo daily breakout/breakdown signals found.")

    if errors:
        error_file = output_dir / f"errors_{start_date}_{end_date}_{timestamp}.csv"
        pd.DataFrame(errors).to_csv(error_file, index=False)
        print(f"Errors for {len(errors)} stocks saved to: {error_file}")


if __name__ == "__main__":
    main()
