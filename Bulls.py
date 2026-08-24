#!/usr/bin/env python3
"""
DhanHQ V1 Scanner
-----------------
Scans NSE stock-F&O underlyings and checks the 09:15-09:20 IST
5-minute candle for Open == Low.

Signal-only: this script DOES NOT place orders.
"""

import argparse
import os
import time
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import pandas as pd
import requests
from dotenv import load_dotenv

IST = ZoneInfo("Asia/Kolkata")
INSTRUMENT_URL = "https://images.dhan.co/api-data/api-scrip-master-detailed.csv"
INTRADAY_URL = "https://api.dhan.co/v2/charts/intraday"
REQUEST_DELAY_SECONDS = 1.0
MAX_RATE_LIMIT_RETRIES = 3

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
    """Build current NSE stock-F&O underlying universe from Dhan instrument master."""
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

    # NSE derivatives only.
    fno = df[
        (df["EXCH_ID"].astype(str).str.upper() == "NSE")
        & (df["SEGMENT"].astype(str).str.upper() == "D")
    ].copy()

    # Stock futures/options only; exclude index derivatives.
    fno = fno[
        fno["INSTRUMENT"].astype(str).str.upper().isin(
            ["FUTSTK", "OPTSTK"]
        )
    ].copy()

    fno["UNDERLYING_SECURITY_ID"] = pd.to_numeric(
        fno["UNDERLYING_SECURITY_ID"], errors="coerce"
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
        universe["UNDERLYING_SECURITY_ID"].astype(int).astype(str)
    )

    print(f"Current NSE stock-F&O universe: {len(universe)} stocks")
    return universe


def get_first_5m_candle(security_id, scan_date):
    """Fetch 5-minute data around market open and return 09:15 candle."""
    start = f"{scan_date} 09:15:00"
    end = f"{scan_date} 09:25:00"

    payload = {
        "securityId": str(security_id),
        "exchangeSegment": "NSE_EQ",
        "instrument": "EQUITY",
        "interval": "5",
        "oi": False,
        "fromDate": start,
        "toDate": end,
    }

    for retry_number in range(MAX_RATE_LIMIT_RETRIES + 1):
        r = requests.post(
            INTRADAY_URL,
            headers=HEADERS,
            json=payload,
            timeout=15,
        )

        if r.status_code != 429 or retry_number == MAX_RATE_LIMIT_RETRIES:
            break

        time.sleep(2 ** retry_number)

    if r.status_code != 200:
        raise RuntimeError(f"HTTP {r.status_code}: {r.text[:500]}")

    data = r.json()

    if not data.get("timestamp"):
        return None

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
        return None

    # Dhan v2 uses Unix/Epoch timestamps.
    df["datetime"] = pd.to_datetime(
        df["timestamp"], unit="s", utc=True
    ).dt.tz_convert(IST)

    first = df[
        (df["datetime"].dt.date == datetime.strptime(scan_date, "%Y-%m-%d").date())
        & (df["datetime"].dt.hour == 9)
        & (df["datetime"].dt.minute == 15)
    ]

    if first.empty:
        return None

    return first.iloc[0].to_dict()


def scan(scan_date):
    universe = get_fno_universe()

    results = []
    errors = []

    print(f"\nScanning 09:15–09:20 IST candle for {scan_date}...\n")

    for i, row in universe.iterrows():
        symbol = str(row["UNDERLYING_SYMBOL"])
        security_id = str(row["UNDERLYING_SECURITY_ID"])

        try:
            candle = get_first_5m_candle(security_id, scan_date)

            if candle is None:
                continue

            o = float(candle["open"])
            h = float(candle["high"])
            l = float(candle["low"])
            c = float(candle["close"])
            v = int(candle["volume"])

            # V1: exact Open == Low.
            if o == l:
                results.append(
                    {
                        "Symbol": symbol,
                        "Security ID": security_id,
                        "Time": candle["datetime"].strftime("%H:%M"),
                        "Open": o,
                        "High": h,
                        "Low": l,
                        "Close": c,
                        "Volume": v,
                        "Bullish Close": "YES" if c > o else "NO",
                    }
                )

        except Exception as exc:
            errors.append(
                {"Symbol": symbol, "Security ID": security_id, "Error": str(exc)}
            )

        # Stay below Dhan's data-API request rate limit.
        time.sleep(REQUEST_DELAY_SECONDS)

        if (i + 1) % 25 == 0:
            print(f"Processed {i + 1}/{len(universe)}...")

    result_df = pd.DataFrame(results)

    print("\n" + "=" * 70)
    print(f"F&O 5-MIN OPEN = LOW | {scan_date} | 09:15–09:20 IST")
    print("=" * 70)

    if result_df.empty:
        print("No stocks matched Open == Low.")
    else:
        print(result_df.to_string(index=False))
        filename = f"open_low_{scan_date}.csv"
        result_df.to_csv(filename, index=False)
        print(f"\nSaved: {filename}")
        print(f"Total signals: {len(result_df)}")

    if errors:
        error_file = f"scanner_errors_{scan_date}.csv"
        pd.DataFrame(errors).to_csv(error_file, index=False)
        print(f"Errors for {len(errors)} symbols saved to: {error_file}")

    print("=" * 70)


def is_market_hours(current_time=None):
    """Return whether the scanner is allowed to run during NSE hours."""
    if current_time is None:
        current_time = datetime.now(IST)

    market_open = current_time.replace(hour=9, minute=15, second=0, microsecond=0)
    market_close = current_time.replace(hour=15, minute=30, second=0, microsecond=0)
    return market_open <= current_time <= market_close


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--date",
        help="Scan date in YYYY-MM-DD. Defaults to today's date in IST.",
    )
    args = parser.parse_args()

    if args.date:
        scan_date = args.date
    else:
        scan_date = datetime.now(IST).strftime("%Y-%m-%d")

    if not args.date and not is_market_hours():
        print("Outside allowed NSE hours (09:15-15:30 IST). Exiting.")
        return

    scan(scan_date)


if __name__ == "__main__":
    main()