#!/usr/bin/env python3
"""Scan option-eligible NSE stocks for daily overbought/oversold conditions."""

import argparse
import io
import os
import smtplib
import time as time_module
from datetime import datetime, time as dt_time
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
NSE_HOLIDAYS = {
    item.strip()
    for item in os.getenv("NSE_HOLIDAYS", "").split(",")
    if item.strip()
}


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


def fetch_history(session, security_id, start_date, end_date, interval=None):
    payload = {
        "securityId": str(security_id),
        "exchangeSegment": EXCHANGE_SEGMENT,
        "instrument": INSTRUMENT,
        "oi": False,
        "fromDate": start_date.isoformat(),
        "toDate": end_date.isoformat(),
    }
    if interval:
        payload["interval"] = interval

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = session.post(HISTORICAL_URL, json=payload, timeout=60)
            if response.status_code == 429:
                if attempt == MAX_RETRIES:
                    response.raise_for_status()
                time_module.sleep(2 ** (attempt - 1))
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
            time_module.sleep(attempt)

    return pd.DataFrame()


def fetch_daily_history(session, security_id, start_date, end_date):
    return fetch_history(session, security_id, start_date, end_date)


def fetch_hourly_history(session, security_id, start_date, end_date):
    return fetch_history(session, security_id, start_date, end_date, interval="ONE_HOUR")


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
    if value > overbought:
        return "Overbought"
    if value < oversold:
        return "Oversold"
    return "Neutral"


def evaluate_signal(
    symbol,
    security_id,
    timestamp,
    current_price,
    rsi,
    stoch_rsi_k,
    bb_lower,
    bb_middle,
    bb_upper,
):
    triggered_indicators = []
    overbought_count = 0
    oversold_count = 0

    if pd.notna(rsi):
        if rsi > RSI_OVERBOUGHT:
            triggered_indicators.append("RSI")
            overbought_count += 1
        elif rsi < RSI_OVERSOLD:
            triggered_indicators.append("RSI")
            oversold_count += 1

    if pd.notna(stoch_rsi_k):
        if stoch_rsi_k > STOCH_OVERBOUGHT:
            triggered_indicators.append("Stochastic RSI")
            overbought_count += 1
        elif stoch_rsi_k < STOCH_OVERSOLD:
            triggered_indicators.append("Stochastic RSI")
            oversold_count += 1

    if pd.notna(current_price):
        if current_price > bb_upper:
            triggered_indicators.append("Bollinger Bands")
            overbought_count += 1
        elif current_price < bb_lower:
            triggered_indicators.append("Bollinger Bands")
            oversold_count += 1

    if not triggered_indicators:
        return None

    unique_triggered = []
    seen = set()
    for indicator in triggered_indicators:
        if indicator not in seen:
            unique_triggered.append(indicator)
            seen.add(indicator)

    if overbought_count > oversold_count:
        overall_signal = "OVERBOUGHT"
    elif oversold_count > overbought_count:
        overall_signal = "OVERSOLD"
    elif overbought_count > 0:
        overall_signal = "OVERBOUGHT"
    else:
        overall_signal = "OVERSOLD"

    return {
        "symbol": symbol,
        "security_id": security_id,
        "timestamp": timestamp,
        "current_price": current_price,
        "rsi": rsi,
        "stoch_rsi_k": stoch_rsi_k,
        "bb_lower": bb_lower,
        "bb_middle": bb_middle,
        "bb_upper": bb_upper,
        "overall_signal": overall_signal,
        "triggered_indicators": unique_triggered,
    }


def scan_latest(history, symbol, security_id):
    if len(history) < max(RSI_PERIOD + STOCH_RSI_PERIOD + STOCH_K_PERIOD + STOCH_D_PERIOD, BOLLINGER_PERIOD):
        return None
    row = add_indicators(history).iloc[-1]
    signal = evaluate_signal(
        symbol=symbol,
        security_id=security_id,
        timestamp=row["timestamp"],
        current_price=row["close"],
        rsi=row["rsi"],
        stoch_rsi_k=row["stoch_rsi_k"],
        bb_lower=row["bb_lower"],
        bb_middle=row["bb_middle"],
        bb_upper=row["bb_upper"],
    )
    if not signal:
        return None

    if signal["overall_signal"] == "OVERBOUGHT":
        signal["overall_signal"] = "SELL / OVERBOUGHT"
    else:
        signal["overall_signal"] = "BUY / OVERSOLD"

    signal["triggered_conditions"] = "; ".join(signal["triggered_indicators"])
    signal["date"] = signal["timestamp"].date().isoformat()
    signal["rsi_condition"] = condition(signal["rsi"], RSI_OVERBOUGHT, RSI_OVERSOLD)
    signal["stoch_rsi_condition"] = condition(
        signal["stoch_rsi_k"], STOCH_OVERBOUGHT, STOCH_OVERSOLD
    )
    if pd.isna(signal["current_price"]):
        return None
    if signal["current_price"] >= signal["bb_upper"]:
        signal["bollinger_condition"] = "Overbought"
    elif signal["current_price"] <= signal["bb_lower"]:
        signal["bollinger_condition"] = "Oversold"
    else:
        signal["bollinger_condition"] = "Neutral"
    signal["stoch_rsi_d"] = add_indicators(history).iloc[-1]["stoch_rsi_d"]
    return signal


def is_trading_day(current_time=None):
    if current_time is None:
        current_time = datetime.now(IST)
    elif current_time.tzinfo is None:
        current_time = current_time.replace(tzinfo=IST)

    return (
        current_time.weekday() < 5
        and current_time.date().isoformat() not in NSE_HOLIDAYS
    )


def is_within_market_scan_window(current_time=None):
    if current_time is None:
        current_time = datetime.now(IST)
    elif current_time.tzinfo is None:
        current_time = current_time.replace(tzinfo=IST)

    if not is_trading_day(current_time):
        return False

    start_time = datetime.combine(current_time.date(), dt_time(10, 15), tzinfo=IST)
    end_time = datetime.combine(current_time.date(), dt_time(15, 15), tzinfo=IST)
    return start_time <= current_time <= end_time


def get_market_close_reference_time(now=None):
    if now is None:
        now = datetime.now(IST)
    if now.tzinfo is None:
        now = now.replace(tzinfo=IST)
    if now.weekday() >= 5:
        current_day = now.date() - pd.Timedelta(days=(now.weekday() - 4) % 7)
    else:
        current_day = now.date()
    return datetime.combine(current_day, dt_time(15, 15), tzinfo=IST)


def previous_trading_date(current_date):
    candidate = current_date - pd.Timedelta(days=1)
    while candidate.weekday() >= 5 or candidate.isoformat() in NSE_HOLIDAYS:
        candidate -= pd.Timedelta(days=1)
    return candidate


def select_hourly_snapshot(history, target_time=None):
    if history.empty:
        return None
    if target_time is None:
        target_time = dt_time(15, 15)

    history = history.copy()
    history["timestamp_ist"] = history["timestamp"].dt.tz_convert(IST)
    exact_matches = history[history["timestamp_ist"].dt.time == target_time]
    if not exact_matches.empty:
        return exact_matches.sort_values("timestamp_ist").iloc[-1]

    target_datetime = datetime.combine(history["timestamp_ist"].dt.date.iloc[-1], target_time, tzinfo=IST)
    earlier = history[history["timestamp_ist"] <= target_datetime]
    if earlier.empty:
        return None
    return earlier.sort_values("timestamp_ist").iloc[-1]


def select_latest_completed_hourly_snapshot(history, now=None):
    """Return the latest candle whose timestamp is not later than now."""
    if history.empty:
        return None
    if now is None:
        now = datetime.now(IST)
    elif now.tzinfo is None:
        now = now.replace(tzinfo=IST)
    else:
        now = now.astimezone(IST)

    history = history.copy()
    history["timestamp_ist"] = history["timestamp"].dt.tz_convert(IST)
    completed = history[history["timestamp_ist"] <= now]
    if completed.empty:
        return None
    return completed.sort_values("timestamp_ist").iloc[-1]


def signal_from_row(history, symbol, security_id, row):
    if row is None:
        return None
    calculated = add_indicators(history)
    matches = calculated[calculated["timestamp"] == row["timestamp"]]
    if matches.empty:
        return None
    row = matches.iloc[0]

    signal = evaluate_signal(
        symbol=symbol,
        security_id=security_id,
        timestamp=row["timestamp"],
        current_price=row["close"],
        rsi=row["rsi"],
        stoch_rsi_k=row["stoch_rsi_k"],
        bb_lower=row["bb_lower"],
        bb_middle=row["bb_middle"],
        bb_upper=row["bb_upper"],
    )
    if not signal:
        return None

    if signal["overall_signal"] == "OVERBOUGHT":
        signal["overall_signal"] = "SELL / OVERBOUGHT"
    else:
        signal["overall_signal"] = "BUY / OVERSOLD"

    signal["triggered_conditions"] = "; ".join(signal["triggered_indicators"])
    signal["date"] = signal["timestamp"].date().isoformat()
    signal["rsi_condition"] = condition(signal["rsi"], RSI_OVERBOUGHT, RSI_OVERSOLD)
    signal["stoch_rsi_condition"] = condition(
        signal["stoch_rsi_k"], STOCH_OVERBOUGHT, STOCH_OVERSOLD
    )
    if pd.isna(signal["current_price"]):
        return None
    if signal["current_price"] >= signal["bb_upper"]:
        signal["bollinger_condition"] = "Overbought"
    elif signal["current_price"] <= signal["bb_lower"]:
        signal["bollinger_condition"] = "Oversold"
    else:
        signal["bollinger_condition"] = "Neutral"
    signal["stoch_rsi_d"] = row["stoch_rsi_d"]
    return signal


def scan_hourly_market(symbols=""):
    require_credentials()
    now = datetime.now(IST)
    end_date = now.date()
    start_date = (now - pd.Timedelta(days=7)).date()

    session = http_session()
    stocks = get_optionable_stocks(fetch_instrument_master(session))
    if symbols.strip():
        requested = {
            item.strip().upper()
            for item in symbols.split(",")
            if item.strip()
        }
        stocks = stocks[stocks["UNDERLYING_SYMBOL"].isin(requested)].copy()
    if stocks.empty:
        raise SystemExit("No option-eligible stocks found for the selection.")

    results = []
    errors = []
    for _, row in stocks.iterrows():
        symbol = row["UNDERLYING_SYMBOL"]
        security_id = row["UNDERLYING_SECURITY_ID"]
        try:
            history = fetch_hourly_history(session, security_id, start_date, end_date)
            if history.empty:
                continue
            snapshot = select_latest_completed_hourly_snapshot(history, now)
            if snapshot is None:
                continue
            signal = signal_from_row(history, symbol, security_id, snapshot)
            if signal and signal["overall_signal"] in {"SELL / OVERBOUGHT", "BUY / OVERSOLD"}:
                results.append(signal)
        except (requests.RequestException, ValueError, KeyError, TypeError, IndexError) as exc:
            errors.append({"symbol": symbol, "security_id": security_id, "error": str(exc)})
        time_module.sleep(REQUEST_DELAY_SECONDS)

    return {"scan_time": now, "results": results, "errors": errors}


def run_after_hours_summary(symbols=""):
    require_credentials()
    now = datetime.now(IST)
    if is_trading_day(now) and now.time() >= dt_time(15, 15):
        daily_end = now.date()
    else:
        daily_end = previous_trading_date(now.date())

    session = http_session()
    stocks = get_optionable_stocks(fetch_instrument_master(session))
    if symbols.strip():
        requested = {
            item.strip().upper()
            for item in symbols.split(",")
            if item.strip()
        }
        stocks = stocks[stocks["UNDERLYING_SYMBOL"].isin(requested)].copy()
    if stocks.empty:
        raise SystemExit("No option-eligible stocks found for the selection.")

    start_date = (pd.Timestamp(daily_end) - pd.Timedelta(days=DEFAULT_HISTORY_DAYS)).date()

    daily_results = []
    hourly_results = []
    errors = []

    for _, row in stocks.iterrows():
        symbol = row["UNDERLYING_SYMBOL"]
        security_id = row["UNDERLYING_SECURITY_ID"]
        try:
            daily_history = fetch_daily_history(session, security_id, start_date, daily_end)
            daily_signal = scan_latest(daily_history, symbol, security_id)
            if daily_signal:
                daily_results.append(daily_signal)

            hourly_history = fetch_hourly_history(
                session,
                security_id,
                (pd.Timestamp(daily_end) - pd.Timedelta(days=7)).date(),
                daily_end,
            )
            if hourly_history.empty:
                continue
            hourly_snapshot = select_hourly_snapshot(hourly_history, dt_time(15, 15))
            if hourly_snapshot is None:
                continue
            hourly_signal = signal_from_row(hourly_history, symbol, security_id, hourly_snapshot)
            if hourly_signal:
                hourly_results.append(hourly_signal)
        except (requests.RequestException, ValueError, KeyError, TypeError, IndexError) as exc:
            errors.append({"symbol": symbol, "security_id": security_id, "error": str(exc)})
        time_module.sleep(REQUEST_DELAY_SECONDS)

    print("\n=== DAILY TIMEFRAME RESULTS ===")
    if daily_results:
        daily_results.sort(key=lambda item: (item["overall_signal"], item["symbol"]))
        print(pd.DataFrame(daily_results).to_string(index=False))
    else:
        print("No overbought/oversold stocks found on the daily timeframe.")

    print("\n=== HOURLY TIMEFRAME RESULTS (3:15 PM IST) ===")
    if hourly_results:
        hourly_results.sort(key=lambda item: (item["overall_signal"], item["symbol"]))
        print(pd.DataFrame(hourly_results).to_string(index=False))
    else:
        print("No overbought/oversold stocks found at 3:15 PM on the 1-hour timeframe.")

    if errors:
        print("\nErrors:")
        for error in errors:
            print(f"- {error['symbol']} ({error['security_id']}): {error['error']}")

    return {"daily": daily_results, "hourly": hourly_results, "errors": errors}


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
                f"<td>{result.get('timestamp', '-')}</td>"
                "</tr>"
            )
        table = (
            "<table><thead><tr>"
            "<th>Symbol</th><th>Signal</th><th>Price</th><th>RSI</th>"
            "<th>Stoch RSI %K</th><th>Bollinger Bands</th><th>Triggered conditions</th><th>Candle time</th>"
            "</tr></thead><tbody>"
            + "".join(rows)
            + "</tbody></table>"
        )
        plain_rows = "\n".join(
            f"{item['symbol']}: {item['overall_signal']} | price {format_value(item['current_price'])} | "
            f"RSI {format_value(item['rsi'])} ({item['rsi_condition']}) | "
            f"Stoch RSI {format_value(item['stoch_rsi_k'])} ({item['stoch_rsi_condition']}) | "
            f"Bollinger {item['bollinger_condition']} | {item['triggered_conditions']} | "
            f"candle {item.get('timestamp', '-')}"
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


def build_after_hours_email(daily_results, hourly_results, errors, end_date):
    """Build a combined post-market summary for daily and 3:15 PM hourly scans."""

    def format_section(title, items):
        if not items:
            return (
                f"<h2>{title}</h2><p>No overbought/oversold stocks found.</p>",
                f"{title}\nNo overbought/oversold stocks found.\n",
            )

        rows = []
        for item in items:
            signal_class = "sell" if item["overall_signal"].startswith("SELL") else "buy"
            rows.append(
                "<tr>"
                f"<td><strong>{item['symbol']}</strong></td>"
                f"<td class='{signal_class}'>{item['overall_signal']}</td>"
                f"<td>{format_value(item['current_price'])}</td>"
                f"<td>{format_value(item['rsi'])} ({item['rsi_condition']})</td>"
                f"<td>{format_value(item['stoch_rsi_k'])} ({item['stoch_rsi_condition']})</td>"
                f"<td>{item['bollinger_condition']}<br>Lower: {format_value(item['bb_lower'])}, Upper: {format_value(item['bb_upper'])}</td>"
                f"<td>{item['triggered_conditions']}</td>"
                f"<td>{item.get('timestamp', '-')}</td>"
                "</tr>"
            )

        table = (
            "<table><thead><tr>"
            "<th>Symbol</th><th>Signal</th><th>Price</th><th>RSI</th>"
            "<th>Stoch RSI %K</th><th>Bollinger Bands</th><th>Triggered conditions</th><th>Candle time</th>"
            "</tr></thead><tbody>"
            + "".join(rows)
            + "</tbody></table>"
        )
        plain_rows = "\n".join(
            f"{item['symbol']}: {item['overall_signal']} | price {format_value(item['current_price'])} | "
            f"RSI {format_value(item['rsi'])} ({item['rsi_condition']}) | "
            f"Stoch RSI {format_value(item['stoch_rsi_k'])} ({item['stoch_rsi_condition']}) | "
            f"Bollinger {item['bollinger_condition']} | {item['triggered_conditions']}"
            for item in items
        )
        return f"<h2>{title}</h2>{table}", f"{title}\n{plain_rows}\n"

    daily_html, daily_plain = format_section("Daily Timeframe", daily_results)
    hourly_html, hourly_plain = format_section("3:15 PM Hourly Timeframe", hourly_results)

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
<h1>Post-market stock scan summary</h1>
<p>Summary for <strong>{end_date}</strong> using the daily timeframe and the 3:15 PM IST 1-hour candle.</p>
{daily_html}
{hourly_html}
{error_section}
</body></html>"""
    plain = (
        f"Post-market stock scan summary for {end_date}\n\n"
        f"{daily_plain}\n{hourly_plain}{plain_errors}"
    )
    subject = f"Post-market stock scan summary - {end_date}"
    return subject, plain, html


def send_after_hours_summary_email(daily_results, hourly_results, errors, end_date):
    subject, plain, html = build_after_hours_email(daily_results, hourly_results, errors, end_date)
    message = EmailMessage()
    message["From"] = GMAIL_ADDRESS
    message["To"] = ALERT_EMAIL_TO
    message["Subject"] = subject
    message.set_content(plain)
    message.add_alternative(html, subtype="html")

    with smtplib.SMTP_SSL(GMAIL_SMTP_HOST, GMAIL_SMTP_PORT, timeout=60) as smtp:
        smtp.login(GMAIL_ADDRESS, GMAIL_APP_PASSWORD)
        smtp.send_message(message)


def email_after_hours_summary(symbols=""):
    summary = run_after_hours_summary(symbols=symbols)
    send_after_hours_summary_email(
        summary["daily"],
        summary["hourly"],
        summary["errors"],
        datetime.now(IST).date().isoformat(),
    )
    return summary


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


def run_scan(symbols="", requested_end_date=None, include_today=False):
    require_credentials()
    today = datetime.now(IST).date()
    end_date = (
        pd.Timestamp(requested_end_date).date()
        if requested_end_date
        else today
    )
    if not requested_end_date and not include_today:
        end_date = today - pd.Timedelta(days=1)
    start_date = (pd.Timestamp(end_date) - pd.Timedelta(days=DEFAULT_HISTORY_DAYS)).date()
    if end_date < start_date:
        raise SystemExit("--end-date cannot be earlier than the warmup start date.")

    session = http_session()
    stocks = get_optionable_stocks(fetch_instrument_master(session))
    if symbols.strip():
        requested = {
            item.strip().upper()
            for item in symbols.split(",")
            if item.strip()
        }
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
        time_module.sleep(REQUEST_DELAY_SECONDS)

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


def main():
    args = parse_args()
    if args.end_date or args.include_today or args.symbols:
        run_scan(
            symbols=args.symbols,
            requested_end_date=args.end_date,
            include_today=args.include_today,
        )
        return
    run_after_hours_summary(symbols=args.symbols)


if __name__ == "__main__":
    main()