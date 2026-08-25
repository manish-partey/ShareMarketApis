#!/usr/bin/env python3
"""Run the Bulls and Bears scanners for each weekday in the past month."""

import argparse
import subprocess
import sys
from datetime import date, timedelta
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parent


def weekdays_between(start_date, end_date):
    """Yield weekdays from start_date through end_date, inclusive."""
    current_date = start_date
    while current_date <= end_date:
        if current_date.weekday() < 5:
            yield current_date
        current_date += timedelta(days=1)


def run_scanner(script_name, scan_date):
    """Run one scanner for one date and return its exit code."""
    command = [
        sys.executable,
        str(PROJECT_DIR / script_name),
        "--date",
        scan_date.isoformat(),
    ]

    print(f"\nRunning {script_name} for {scan_date.isoformat()}...")
    completed = subprocess.run(command, cwd=PROJECT_DIR)
    return completed.returncode


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Run Bulls.py and Bears.py for every weekday in the previous month."
        )
    )
    parser.add_argument(
        "--end-date",
        type=date.fromisoformat,
        default=date.today(),
        help="Last date to include, in YYYY-MM-DD format. Defaults to today.",
    )
    parser.add_argument(
        "--days",
        type=int,
        default=30,
        help="Number of calendar days to look back. Defaults to 30.",
    )
    args = parser.parse_args()

    if args.days < 1:
        parser.error("--days must be at least 1")

    start_date = args.end_date - timedelta(days=args.days - 1)
    scan_dates = list(weekdays_between(start_date, args.end_date))

    print(
        f"Backtesting {start_date.isoformat()} through "
        f"{args.end_date.isoformat()} ({len(scan_dates)} weekdays)"
    )

    for scan_date in scan_dates:
        for script_name in ("Bulls.py", "Bears.py"):
            return_code = run_scanner(script_name, scan_date)
            if return_code != 0:
                raise SystemExit(
                    f"{script_name} failed for {scan_date.isoformat()} "
                    f"with exit code {return_code}."
                )

    print("\nBacktest completed successfully.")


if __name__ == "__main__":
    main()
