#!/usr/bin/env python3
"""
DhanHQ Scanner - First 5-Minute OPEN = HIGH
--------------------------------------------
Purpose:
    Scan NSE stock-F&O underlyings and identify stocks whose
    FIRST 5-minute NSE equity candle (09:15-09:20 IST)
    has Open == High.

This means the first 5-minute candle has NO UPPER WICK.

Signal-only:
    This script DOES NOT place orders.

Method:
    Dhan 1-minute candles are fetched for 09:15-09:19 and
    aggregated locally into the exact 09:15-09:20 candle.

Core condition:
    Open == High
"""

import argparse
import math
import os
import time
from datetime import datetime
from zoneinfo import ZoneInfo

import pandas as pd
import requests
from dotenv import load_dotenv

IST = ZoneInfo("Asia/Kolkata")

INSTRUMENT_URL = "https://images.dhan.co/api-data/api-scrip-master-detailed.csv"
INTRADAY_URL = "https://api.dhan.co/v2/charts/intraday"

INTERVAL = "1"
REQUEST_DELAY_SECONDS = 0.25
MAX_RATE_LIMIT_RETRIES = 5

load_dotenv()

CLIENT_ID = os.getenv("DHAN_CLIENT_ID")
ACCESS_TOKEN = os.getenv("DHAN_ACCESS_TOKEN")

if not CLIENT_ID or not ACCESS_TOKEN:
    raise SystemExit(
        "Missing DHAN_CLIENT_ID or DHAN_ACCESS_TOKEN in .env"
    )

HEADERS = {
    "Accept": "application/json",
    "Content-Type": "application/json",
    "client-id": CLIENT_ID,
    "access-token": ACCESS_TOKEN,
}


def get_fno_universe():
    """Build the current NSE stock-F&O underlying universe."""
    print("Downloading current Dhan instrument master...")
    df = pd.read_csv(INSTRUMENT_URL, low_memory=False)

    required = {
        "EXCH_ID",
        "SEGMENT",
        "INSTRUMENT",
        "UNDERLYING_SECURITY_ID",
        "UNDERLYING_SYMBOL",
    }

    missing = required - set(df.columns)
    if missing:
        raise RuntimeError(
            f"Instrument master columns changed. Missing: {sorted(missing)}\n"
            f"Available columns: {list(df.columns)}"
        )

    fno = df[
        (df["EXCH_ID"].astype(str).str.upper() == "NSE")
        & (df["SEGMENT"].astype(str).str.upper() == "D")
        & (
            df["INSTRUMENT"]
            .astype(str)
            .str.upper()
            .isin(["FUTSTK", "OPTSTK"])
        )
    ].copy()

    fno["UNDERLYING_SECURITY_ID"] = pd.to_numeric(
        fno["UNDERLYING_SECURITY_ID"],
        errors="coerce",
    )

    fno = fno.dropna(subset=["UNDERLYING_SECURITY_ID"])

    universe = (
        fno[
            ["UNDERLYING_SECURITY_ID", "UNDERLYING_SYMBOL"]
        ]
        .drop_duplicates(subset=["UNDERLYING_SECURITY_ID"])
        .sort_values("UNDERLYING_SYMBOL")
        .reset_index(drop=True)
    )

    universe["UNDERLYING_SECURITY_ID"] = (
        universe["UNDERLYING_SECURITY_ID"]
        .astype(int)
        .astype(str)
    )

    print(f"Current NSE stock-F&O universe: {len(universe)} stocks")
    return universe


def fetch_1m_data(security_id, scan_date):
    """
    Fetch 1-minute NSE equity candles around market open.

    We use 09:15, 09:16, 09:17, 09:18 and 09:19
    to construct the exact first 5-minute candle.
    """
    payload = {
        "securityId": str(security_id),
        "exchangeSegment": "NSE_EQ",
        "instrument": "EQUITY",
        "interval": INTERVAL,
        "oi": False,
        "fromDate": f"{scan_date} 09:15:00",
        "toDate": f"{scan_date} 09:21:00",
    }

    response = None

    for retry_number in range(MAX_RATE_LIMIT_RETRIES + 1):
        response = requests.post(
            INTRADAY_URL,
            headers=HEADERS,
            json=payload,
            timeout=15,
        )

        if response.status_code != 429:
            break

        if retry_number == MAX_RATE_LIMIT_RETRIES:
            break

        sleep_for = 2 ** retry_number
        print(
            f"Rate limited for Security ID {security_id}; "
            f"retrying in {sleep_for}s..."
        )
        time.sleep(sleep_for)

    if response is None or response.status_code != 200:
        status = response.status_code if response is not None else "NO_RESPONSE"
        text = response.text[:500] if response is not None else ""
        raise RuntimeError(f"HTTP {status}: {text}")

    data = response.json()

    if not data.get("timestamp"):
        return pd.DataFrame()

    df = pd.DataFrame(
        {
            "timestamp": data.get("timestamp", []),
            "open": data.get("open", []),
            "high": data.get("high", []),
            "low": data.get("low", []),
            "close": data.get("close", []),
            "volume": data.get("volume", []),
        }
    )

    if df.empty:
        return df

    df["datetime"] = pd.to_datetime(
        df["timestamp"],
        unit="s",
        utc=True,
    ).dt.tz_convert(IST)

    for column in ["open", "high", "low", "close", "volume"]:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    df = df.dropna(
        subset=["datetime", "open", "high", "low", "close"]
    )

    return df.sort_values("datetime").reset_index(drop=True)


def build_first_5m_candle(security_id, scan_date):
    """
    Build the exact 09:15-09:20 candle from five 1-minute bars.

    Open   = 09:15 Open
    High   = maximum High of 09:15-09:19
    Low    = minimum Low of 09:15-09:19
    Close  = 09:19 Close
    Volume = sum of 09:15-09:19 volumes
    """
    df = fetch_1m_data(security_id, scan_date)

    if df.empty:
        return None, df

    target_date = datetime.strptime(
        scan_date,
        "%Y-%m-%d",
    ).date()

    first_5 = df[
        (df["datetime"].dt.date == target_date)
        & (df["datetime"].dt.hour == 9)
        & (df["datetime"].dt.minute.between(15, 19))
    ].copy()

    expected_minutes = {15, 16, 17, 18, 19}
    actual_minutes = set(first_5["datetime"].dt.minute.tolist())

    # Do not create a candle unless all five one-minute bars exist.
    if not expected_minutes.issubset(actual_minutes):
        return None, first_5

    first_5 = (
        first_5.sort_values("datetime")
        .drop_duplicates(subset=["datetime"], keep="first")
        .reset_index(drop=True)
    )

    if len(first_5) != 5:
        return None, first_5

    candle = {
        "datetime": first_5.iloc[0]["datetime"],
        "open": float(first_5.iloc[0]["open"]),
        "high": float(first_5["high"].max()),
        "low": float(first_5["low"].min()),
        "close": float(first_5.iloc[-1]["close"]),
        "volume": int(first_5["volume"].fillna(0).sum()),
        "minute_bars": first_5,
    }

    return candle, first_5


def prices_equal(price1, price2):
    """
    Safe decimal comparison.

    This tiny tolerance only prevents floating-point representation
    issues. It does not represent a meaningful price difference.
    """
    return math.isclose(
        float(price1),
        float(price2),
        rel_tol=0.0,
        abs_tol=1e-9,
    )


def scan(scan_date, debug_symbols=None):
    universe = get_fno_universe()

    results = []
    errors = []

    debug_symbols = {
        symbol.strip().upper()
        for symbol in (debug_symbols or [])
        if symbol.strip()
    }

    print(
        f"\nScanning FIRST 5-MINUTE OPEN = HIGH for "
        f"{scan_date} | 09:15-09:20 IST\n"
    )

    for i, row in universe.iterrows():
        symbol = str(row["UNDERLYING_SYMBOL"]).strip()
        security_id = str(row["UNDERLYING_SECURITY_ID"])

        try:
            candle, minute_bars = build_first_5m_candle(
                security_id,
                scan_date,
            )

            if candle is None:
                continue

            o = candle["open"]
            h = candle["high"]
            l = candle["low"]
            c = candle["close"]
            v = candle["volume"]

            # Optional diagnostic output.
            if symbol.upper() in debug_symbols:
                print(f"\n--- DEBUG: {symbol} ({security_id}) ---")

                if not minute_bars.empty:
                    print(
                        minute_bars[
                            [
                                "datetime",
                                "open",
                                "high",
                                "low",
                                "close",
                                "volume",
                            ]
                        ].to_string(index=False)
                    )

                print(
                    f"Aggregated 09:15-09:20: "
                    f"O={o} H={h} L={l} C={c} V={v}"
                )

            # CORE CONDITION:
            # Opening Price == Higher Price
            #
            # For a candlestick, Open == High means the candle has
            # NO UPPER WICK.
            if prices_equal(o, h):
                results.append(
                    {
                        "Symbol": symbol,
                        "Security ID": security_id,
                        "Time": "09:15-09:20",
                        "Open": o,
                        "High": h,
                        "Low": l,
                        "Close": c,
                        "Volume": v,
                        "Bearish Close": "YES" if c < o else "NO",
                    }
                )

        except Exception as exc:
            errors.append(
                {
                    "Symbol": symbol,
                    "Security ID": security_id,
                    "Error": str(exc),
                }
            )

        time.sleep(REQUEST_DELAY_SECONDS)

        if (i + 1) % 25 == 0:
            print(f"Processed {i + 1}/{len(universe)}...")

    result_df = pd.DataFrame(results)

    print("\n" + "=" * 80)
    print(
        f"F&O FIRST 5-MIN OPEN = HIGH | "
        f"{scan_date} | 09:15-09:20 IST"
    )
    print("=" * 80)

    if result_df.empty:
        print("No stocks matched Open == High.")
    else:
        print(result_df.to_string(index=False))

        filename = f"open_high_{scan_date}.csv"
        result_df.to_csv(filename, index=False)

        print(f"\nSaved: {filename}")
        print(f"Total signals: {len(result_df)}")

    if errors:
        error_file = f"scanner_errors_{scan_date}.csv"
        pd.DataFrame(errors).to_csv(error_file, index=False)
        print(
            f"Errors for {len(errors)} symbols saved to: {error_file}"
        )

    print("=" * 80)


def is_market_hours(current_time=None):
    if current_time is None:
        current_time = datetime.now(IST)

    market_open = current_time.replace(
        hour=9,
        minute=15,
        second=0,
        microsecond=0,
    )

    market_close = current_time.replace(
        hour=15,
        minute=30,
        second=0,
        microsecond=0,
    )

    return market_open <= current_time <= market_close


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--date",
        help="Scan date in YYYY-MM-DD. Defaults to today's date in IST.",
    )

    parser.add_argument(
        "--debug",
        nargs="*",
        default=[],
        help=(
            "Print the five raw 1-minute bars and the aggregated "
            "09:15-09:20 candle for selected symbols. Example: "
            "--debug LODHA ONGC PETRONET PNB UNIONBANK"
        ),
    )

    args = parser.parse_args()

    if args.date:
        scan_date = args.date
    else:
        scan_date = datetime.now(IST).strftime("%Y-%m-%d")

    if not args.date and not is_market_hours():
        print(
            "Outside allowed NSE hours (09:15-15:30 IST). Exiting."
        )
        return

    scan(
        scan_date=scan_date,
        debug_symbols=args.debug,
    )


if __name__ == "__main__":
    main()