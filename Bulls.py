#!/usr/bin/env python3
"""
DhanHQ V2 Scanner - First 5-Minute OPEN = LOW
------------------------------------------------
Purpose:
    Scan NSE stock-F&O underlyings and identify stocks whose
    FIRST 5-minute NSE equity candle (09:15:00-09:20:00 IST)
    has Open == Low.

Important:
    We intentionally FETCH 1-MINUTE candles and build the first
    5-minute candle ourselves. This avoids relying on Dhan's
    pre-aggregated 5-minute candle timestamp/alignment.

Signal-only:
    This script DOES NOT place orders.
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

# Dhan data API supports minute data. We use 1-minute data and
# aggregate 09:15, 09:16, 09:17, 09:18 and 09:19 ourselves.
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
    """
    Build current NSE stock-F&O underlying universe from Dhan instrument master.

    We use the underlying equity Security ID because the scanner is checking
    the NSE cash/equity candle, not an individual option contract.
    """
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

    We deliberately request 09:15 through 09:21 so that the five
    complete one-minute bars 09:15, 09:16, 09:17, 09:18 and 09:19
    are available for manual 5-minute aggregation.
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

    if response.status_code != 200:
        raise RuntimeError(
            f"HTTP {response.status_code}: {response.text[:500]}"
        )

    data = response.json()

    timestamps = data.get("timestamp", [])
    if not timestamps:
        return pd.DataFrame()

    df = pd.DataFrame(
        {
            "timestamp": timestamps,
            "open": data.get("open", []),
            "high": data.get("high", []),
            "low": data.get("low", []),
            "close": data.get("close", []),
            "volume": data.get("volume", []),
        }
    )

    if df.empty:
        return df

    # Dhan returns Unix/Epoch timestamps.
    # Convert to IST before selecting the 09:15-09:19 bars.
    df["datetime"] = pd.to_datetime(
        df["timestamp"],
        unit="s",
        utc=True,
    ).dt.tz_convert(IST)

    df["open"] = pd.to_numeric(df["open"], errors="coerce")
    df["high"] = pd.to_numeric(df["high"], errors="coerce")
    df["low"] = pd.to_numeric(df["low"], errors="coerce")
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    df["volume"] = pd.to_numeric(df["volume"], errors="coerce")

    df = df.dropna(
        subset=["datetime", "open", "high", "low", "close"]
    )

    return df.sort_values("datetime").reset_index(drop=True)


def build_first_5m_candle(security_id, scan_date):
    """
    Build the exact 09:15-09:20 candle from five 1-minute bars.

    Required minute bars:
        09:15
        09:16
        09:17
        09:18
        09:19

    Aggregation:
        Open   = first minute Open
        High   = max of five minute Highs
        Low    = min of five minute Lows
        Close  = last minute Close
        Volume = sum of five minute Volumes
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

    # We require all five one-minute bars.
    expected_minutes = {15, 16, 17, 18, 19}
    actual_minutes = set(first_5["datetime"].dt.minute.tolist())

    if not expected_minutes.issubset(actual_minutes):
        return None, first_5

    # In case duplicate timestamps ever appear, keep the first one.
    first_5 = (
        first_5.sort_values("datetime")
        .drop_duplicates(subset=["datetime"], keep="first")
    )

    if len(first_5) != 5:
        return None, first_5

    first_5 = first_5.reset_index(drop=True)

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


def open_equals_low(open_price, low_price):
    """
    Compare prices safely.

    Dhan prices are decimal values. We use a tiny tolerance only to
    avoid binary floating-point comparison issues. This is NOT a
    percentage tolerance and does not turn a meaningful lower wick
    into a signal.
    """
    return math.isclose(
        open_price,
        low_price,
        rel_tol=0.0,
        abs_tol=1e-9,
    )


def scan(scan_date, debug_symbols=None):
    universe = get_fno_universe()

    results = []
    errors = []

    print(
        f"\nScanning exact 09:15–09:20 IST candle for {scan_date} "
        f"using 1-minute aggregation...\n"
    )

    debug_symbols = {
        s.strip().upper()
        for s in (debug_symbols or [])
        if s.strip()
    }

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

            # Optional diagnostic output for selected symbols.
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
            # First 5-minute candle has no lower wick.
            #
            # For the user's exact requirement:
            #     Open == Low
            #
            # We do NOT use the daily Open or daily Low here.
            if open_equals_low(o, l):
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
                        "Bullish Close": "YES" if c > o else "NO",
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
        f"F&O FIRST 5-MIN OPEN = LOW | "
        f"{scan_date} | 09:15–09:20 IST"
    )
    print("=" * 80)

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
        print(
            f"Errors for {len(errors)} symbols saved to: {error_file}"
        )

    print("=" * 80)


def is_market_hours(current_time=None):
    """Return whether the scanner is allowed to run during NSE hours."""
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