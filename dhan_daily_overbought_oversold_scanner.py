#!/usr/bin/env python3
"""Scan option-eligible NSE stocks for daily overbought/oversold conditions."""

import argparse
import io
import os
import smtplib
import time
from datetime import datetime
from email.message import EmailMessage
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

RSI_PERIOD = 14
STOCH_RSI_PERIOD = 14
STOCH_K_PERIOD = 3
STOCH_D_PERIOD = 3
BOLLINGER_PERIOD = 20
BOLLINGER_STD_DEV = 2.0
RSI_OVERBOUGHT = 70.0
RSI_OVERSOLD = 30.0
STOCH_OVERBOUGHT = 80.0
STOCH_OVERSOLD = 20.0
DEFAULT_HISTORY_DAYS = 180
GMAIL_SMTP_HOST = "smtp.gmail.com"
GMAIL_SMTP_PORT = 465

load_dotenv(Path(__file__).resolve().parent / ".env")
DHAN_ACCESS_TOKEN = os.getenv("DHAN_ACCESS_TOKEN")
DHAN_CLIENT_ID = os.getenv("DHAN_CLIENT_ID")
GMAIL_ADDRESS = os.getenv("GMAIL_ADDRESS")
GMAIL_APP_PASSWORD = os.getenv("GMAIL_APP_PASSWORD")
ALERT_EMAIL_TO = os.getenv("ALERT_EMAIL_TO")


def require_credentials():
    if not DHAN_ACCESS_TOKEN or not DHAN_CLIENT_ID:
        raise SystemExit("Missing DHAN_ACCESS_TOKEN or DHAN_CLIENT_ID in .env.")
    missing_email_settings = [
        name
        for name, value in {
            "GMAIL_ADDRESS": GMAIL_ADDRESS,
            "GMAIL_APP_PASSWORD": GMAIL_APP_PASSWORD,
            "ALERT_EMAIL_TO": ALERT_EMAIL_TO,
        }.items()
        if not value
    ]
    if missing_email_settings:
        raise SystemExit(
            "Missing email settings in .env: "
            + ", ".join(missing_email_settings)
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


def get_optionable_stocks(master):
    """Return NSE equity underlyings that have listed stock options."""
    required = {
        "EXCH_ID",
        "SEGMENT",
        "INSTRUMENT",
        "UNDERLYING_SECURITY_ID",
        "UNDERLYING_SYMBOL",
    }
    missing = required - set(master.columns)
    if missing:
        raise RuntimeError(f"Instrument master is missing columns: {sorted(missing)}")

    options = master[
        (master["EXCH_ID"].astype(str).str.upper() == "NSE")
        & (master["SEGMENT"].astype(str).str.upper() == "D")
        & (master["INSTRUMENT"].astype(str).str.upper() == "OPTSTK")
    ].copy()
    options["UNDERLYING_SYMBOL"] = (
        options["UNDERLYING_SYMBOL"].astype(str).str.strip().str.upper()
    )
    options["UNDERLYING_SECURITY_ID"] = pd.to_numeric(
        options["UNDERLYING_SECURITY_ID"], errors="coerce"
    )

    stocks = (
        options.dropna(subset=["UNDERLYING_SECURITY_ID"])
        .loc[lambda frame: frame["UNDERLYING_SYMBOL"].ne("")]
        [["UNDERLYING_SYMBOL", "UNDERLYING_SECURITY_ID"]]
        .drop_duplicates(subset=["UNDERLYING_SECURITY_ID"])
        .sort_values("UNDERLYING_SYMBOL")
        .reset_index(drop=True)
    )
    stocks["UNDERLYING_SECURITY_ID"] = (
        stocks["UNDERLYING_SECURITY_ID"].astype(int).astype(str)
    )
    return stocks


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
            response = session.post(HISTORICAL_URL, json=payload, timeout=60)
            if response.status_code == 429:
                if attempt == MAX_RETRIES:
                    response.raise_for_status()
                time.sleep(2 ** (attempt - 1))
                continue
            response.raise_for_status()
            data = response.json()
            timestamps = data.get("timestamp", [])
            if not timestamps:
                return pd.DataFrame()

            count = len(timestamps)
            history = pd.DataFrame(
                {
                    "timestamp": timestamps,
                    "open": data.get("open", [np.nan] * count),
                    "high": data.get("high", [np.nan] * count),
                    "low": data.get("low", [np.nan] * count),
                    "close": data.get("close", [np.nan] * count),
                    "volume": data.get("volume", [np.nan] * count),
                }
            )
            history["timestamp"] = pd.to_datetime(
                history["timestamp"], unit="s", utc=True
            ).dt.tz_convert(TIMEZONE)
            for column in ["open", "high", "low", "close", "volume"]:
                history[column] = pd.to_numeric(history[column], errors="coerce")
            return (
                history.dropna(subset=["timestamp", "open", "high", "low", "close"])
                .sort_values("timestamp")
                .drop_duplicates("timestamp")
                .reset_index(drop=True)
            )
        except requests.RequestException:
            if attempt == MAX_RETRIES:
                raise
            time.sleep(attempt)

    return pd.DataFrame()


def calculate_rsi(close, period=RSI_PERIOD):
    delta = close.diff()
    gains = delta.clip(lower=0)
    losses = -delta.clip(upper=0)
    average_gain = gains.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    average_loss = losses.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    rsi = 100 - (100 / (1 + average_gain / average_loss.replace(0, np.nan)))
    return (
        rsi.mask(average_loss.eq(0) & average_gain.gt(0), 100.0)
        .mask(average_loss.eq(0) & average_gain.eq(0), 50.0)
    )


def add_indicators(history):
    result = history.copy()
    result["rsi"] = calculate_rsi(result["close"])
    rsi_low = result["rsi"].rolling(STOCH_RSI_PERIOD).min()
    rsi_high = result["rsi"].rolling(STOCH_RSI_PERIOD).max()
    denominator = (rsi_high - rsi_low).replace(0, np.nan)
    raw_stoch_rsi = ((result["rsi"] - rsi_low) / denominator) * 100
    result["stoch_rsi_k"] = raw_stoch_rsi.rolling(STOCH_K_PERIOD).mean()
    result["stoch_rsi_d"] = result["stoch_rsi_k"].rolling(STOCH_D_PERIOD).mean()
    result["bb_middle"] = result["close"].rolling(BOLLINGER_PERIOD).mean()
    rolling_std = result["close"].rolling(BOLLINGER_PERIOD).std(ddof=0)
    result["bb_upper"] = result["bb_middle"] + BOLLINGER_STD_DEV * rolling_std
    result["bb_lower"] = result["bb_middle"] - BOLLINGER_STD_DEV * rolling_std
    return result


def condition(value, overbought, oversold):
    if pd.isna(value):
        return "Unavailable"
    if value >= overbought:
        return "Overbought"
    if value <= oversold:
        return "Oversold"
    return "Neutral"


def scan_latest(history, symbol, security_id):
    if len(history) < max(RSI_PERIOD + STOCH_RSI_PERIOD + STOCH_K_PERIOD + STOCH_D_PERIOD, BOLLINGER_PERIOD):
        return None
    row = add_indicators(history).iloc[-1]
    rsi_condition = condition(row["rsi"], RSI_OVERBOUGHT, RSI_OVERSOLD)
    stoch_condition = condition(
        row["stoch_rsi_k"], STOCH_OVERBOUGHT, STOCH_OVERSOLD
    )
    if pd.isna(row["close"]):
        return None
    if row["close"] >= row["bb_upper"]:
        bb_condition = "Overbought"
    elif row["close"] <= row["bb_lower"]:
        bb_condition = "Oversold"
    else:
        bb_condition = "Neutral"

    conditions = {
        "RSI": rsi_condition,
        "Stochastic RSI": stoch_condition,
        "Bollinger Bands": bb_condition,
    }
    overbought_conditions = [name for name, value in conditions.items() if value == "Overbought"]
    oversold_conditions = [name for name, value in conditions.items() if value == "Oversold"]
    if len(overbought_conditions) >= 2:
        overall_signal = "SELL / OVERBOUGHT"
        triggered = "; ".join(overbought_conditions)
    elif len(oversold_conditions) >= 2:
        overall_signal = "BUY / OVERSOLD"
        triggered = "; ".join(oversold_conditions)
    else:
        return None

    return {
        "symbol": symbol,
        "security_id": security_id,
        "date": row["timestamp"].date().isoformat(),
        "current_price": row["close"],
        "rsi": row["rsi"],
        "rsi_condition": rsi_condition,
        "stoch_rsi_k": row["stoch_rsi_k"],
        "stoch_rsi_d": row["stoch_rsi_d"],
        "stoch_rsi_condition": stoch_condition,
        "bb_lower": row["bb_lower"],
        "bb_middle": row["bb_middle"],
        "bb_upper": row["bb_upper"],
        "bollinger_condition": bb_condition,
        "overall_signal": overall_signal,
        "triggered_conditions": triggered,
    }


def format_value(value, decimals=2):
    if pd.isna(value):
        return "-"
    return f"{value:.{decimals}f}"


def build_email(results, errors, end_date):
    """Build a readable HTML alert and plain-text fallback."""
    subject = f"Daily options stock signals - {end_date}"
    if results:
        rows = []
        for result in results:
            signal_class = "sell" if result["overall_signal"].startswith("SELL") else "buy"
            rows.append(
                "<tr>"
                f"<td><strong>{result['symbol']}</strong></td>"
                f"<td class='{signal_class}'>{result['overall_signal']}</td>"
                f"<td>{format_value(result['current_price'])}</td>"
                f"<td>{format_value(result['rsi'])} ({result['rsi_condition']})</td>"
                f"<td>{format_value(result['stoch_rsi_k'])} ({result['stoch_rsi_condition']})</td>"
                f"<td>{result['bollinger_condition']}<br>Lower: {format_value(result['bb_lower'])}, "
                f"Upper: {format_value(result['bb_upper'])}</td>"
                f"<td>{result['triggered_conditions']}</td>"
                "</tr>"
            )
        table = (
            "<table><thead><tr>"
            "<th>Symbol</th><th>Signal</th><th>Price</th><th>RSI</th>"
            "<th>Stoch RSI %K</th><th>Bollinger Bands</th><th>Triggered conditions</th>"
            "</tr></thead><tbody>"
            + "".join(rows)
            + "</tbody></table>"
        )
        plain_rows = "\n".join(
            f"{item['symbol']}: {item['overall_signal']} | price {format_value(item['current_price'])} | "
            f"RSI {format_value(item['rsi'])} ({item['rsi_condition']}) | "
            f"Stoch RSI {format_value(item['stoch_rsi_k'])} ({item['stoch_rsi_condition']}) | "
            f"Bollinger {item['bollinger_condition']} | {item['triggered_conditions']}"
            for item in results
        )
    else:
        table = "<p>No multi-indicator overbought/oversold signals found.</p>"
        plain_rows = "No multi-indicator overbought/oversold signals found."

    error_section = ""
    plain_errors = ""
    if errors:
        error_items = "".join(
            f"<li>{item['symbol']} ({item['security_id']}): {item['error']}</li>"
            for item in errors
        )
        error_section = f"<h2>Scan errors ({len(errors)})</h2><ul>{error_items}</ul>"
        plain_errors = "\n\nScan errors:\n" + "\n".join(
            f"{item['symbol']} ({item['security_id']}): {item['error']}"
            for item in errors
        )

    html = f"""<!doctype html>
<html><head><meta charset="utf-8"><style>
body {{ font-family: Arial, sans-serif; color: #202124; }}
table {{ border-collapse: collapse; width: 100%; }}
th, td {{ border: 1px solid #dadce0; padding: 8px; text-align: left; vertical-align: top; }}
th {{ background: #f1f3f4; }}
.buy {{ color: #137333; font-weight: bold; }}
.sell {{ color: #c5221f; font-weight: bold; }}
</style></head><body>
<h1>Daily options stock signals</h1>
<p>Completed daily candle through <strong>{end_date}</strong>. Scanned stocks with listed NSE stock options.</p>
{table}
{error_section}
</body></html>"""
    plain = (
        f"Daily options stock signals for {end_date}\n\n"
        f"{plain_rows}{plain_errors}"
    )
    return subject, plain, html


def send_email(results, errors, end_date):
    subject, plain, html = build_email(results, errors, end_date)
    message = EmailMessage()
    message["From"] = GMAIL_ADDRESS
    message["To"] = ALERT_EMAIL_TO
    message["Subject"] = subject
    message.set_content(plain)
    message.add_alternative(html, subtype="html")

    with smtplib.SMTP_SSL(GMAIL_SMTP_HOST, GMAIL_SMTP_PORT, timeout=60) as smtp:
        smtp.login(GMAIL_ADDRESS, GMAIL_APP_PASSWORD)
        smtp.send_message(message)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Scan NSE stocks with listed options for daily overbought/oversold signals."
    )
    parser.add_argument("--symbols", default="", help="Optional comma-separated symbols.")
    parser.add_argument("--end-date", help="Last completed candle date, YYYY-MM-DD.")
    parser.add_argument(
        "--include-today",
        action="store_true",
        help="Include today's candle when available; it may still be incomplete.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    require_credentials()
    today = datetime.now(IST).date()
    end_date = pd.Timestamp(args.end_date).date() if args.end_date else today
    if not args.end_date and not args.include_today:
        end_date = today - pd.Timedelta(days=1)
    start_date = (pd.Timestamp(end_date) - pd.Timedelta(days=DEFAULT_HISTORY_DAYS)).date()
    if end_date < start_date:
        raise SystemExit("--end-date cannot be earlier than the warmup start date.")

    session = http_session()
    stocks = get_optionable_stocks(fetch_instrument_master(session))
    if args.symbols.strip():
        requested = {item.strip().upper() for item in args.symbols.split(",") if item.strip()}
        stocks = stocks[stocks["UNDERLYING_SYMBOL"].isin(requested)].copy()
    if stocks.empty:
        raise SystemExit("No option-eligible stocks found for the selection.")

    print(f"Scanning {len(stocks)} option-eligible stocks through {end_date}...")
    results = []
    errors = []
    for index, row in stocks.iterrows():
        symbol = row["UNDERLYING_SYMBOL"]
        security_id = row["UNDERLYING_SECURITY_ID"]
        print(f"[{index + 1}/{len(stocks)}] {symbol} ({security_id})")
        try:
            history = fetch_daily_history(session, security_id, start_date, end_date)
            signal = scan_latest(history, symbol, security_id)
            if signal:
                print(f"    -> {signal['overall_signal']}: {signal['triggered_conditions']}")
                results.append(signal)
            else:
                print("    -> no multi-indicator signal")
        except (requests.RequestException, ValueError, KeyError, TypeError, IndexError) as exc:
            print(f"    -> ERROR: {exc}")
            errors.append({"symbol": symbol, "security_id": security_id, "error": str(exc)})
        time.sleep(REQUEST_DELAY_SECONDS)

    if results:
        results.sort(key=lambda item: (item["overall_signal"], item["symbol"]))
        print("\nDAILY OVERBOUGHT/OVERSOLD SIGNALS")
        print(pd.DataFrame(results).to_string(index=False))
    else:
        print("\nNo multi-indicator overbought/oversold signals found.")

    try:
        send_email(results, errors, end_date)
        print(f"Email sent to {ALERT_EMAIL_TO}.")
    except (OSError, smtplib.SMTPException) as exc:
        raise SystemExit(f"Could not send email: {exc}") from exc


if __name__ == "__main__":
    main()