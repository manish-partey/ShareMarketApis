#!/usr/bin/env python3
"""
DhanHQ V1 Bearish Scanner
-------------------------
Scans NSE stock-F&O underlyings and checks the 09:15-09:20 IST
5-minute candle for Open == High.

Signal-only: this script DOES NOT place orders.
"""

import argparse
import time
from datetime import datetime

import pandas as pd

from Bulls import get_first_5m_candle, get_fno_universe, is_market_hours, IST


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

            if o == h:
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
                        "Bearish Close": "YES" if c < o else "NO",
                    }
                )

        except Exception as exc:
            errors.append(
                {"Symbol": symbol, "Security ID": security_id, "Error": str(exc)}
            )

        time.sleep(0.22)

        if (i + 1) % 25 == 0:
            print(f"Processed {i + 1}/{len(universe)}...")

    result_df = pd.DataFrame(results)

    print("\n" + "=" * 70)
    print(f"F&O 5-MIN OPEN = HIGH | {scan_date} | 09:15–09:20 IST")
    print("=" * 70)

    if result_df.empty:
        print("No stocks matched Open == High.")
    else:
        print(result_df.to_string(index=False))
        filename = f"open_high_{scan_date}.csv"
        result_df.to_csv(filename, index=False)
        print(f"\nSaved: {filename}")
        print(f"Total signals: {len(result_df)}")

    if errors:
        error_file = f"bearish_scanner_errors_{scan_date}.csv"
        pd.DataFrame(errors).to_csv(error_file, index=False)
        print(f"Errors for {len(errors)} symbols saved to: {error_file}")

    print("=" * 70)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--date",
        help="Scan date in YYYY-MM-DD. Defaults to today's date in IST.",
    )
    args = parser.parse_args()

    if not args.date and not is_market_hours():
        print("Outside allowed NSE hours (09:15-15:30 IST). Exiting.")
        return

    scan_date = args.date or datetime.now(IST).strftime("%Y-%m-%d")
    scan(scan_date)


if __name__ == "__main__":
    main()
