from datetime import datetime

from dhan_daily_overbought_oversold_scanner import (
    build_after_hours_email,
    evaluate_signal,
    is_within_market_scan_window,
)


def test_market_scan_window_allows_weekday_hourly_runs():
    dt = datetime(2026, 9, 8, 10, 15)  # Tuesday
    assert is_within_market_scan_window(dt) is True


def test_market_scan_window_rejects_weekend_and_closed_market_hours():
    saturday = datetime(2026, 9, 12, 10, 15)
    assert is_within_market_scan_window(saturday) is False

    closed_time = datetime(2026, 9, 8, 9, 0)
    assert is_within_market_scan_window(closed_time) is False


def test_evaluate_signal_flags_any_triggered_indicator_for_hourly_scan():
    result = evaluate_signal(
        symbol="RELIANCE",
        security_id="12345",
        timestamp="2026-09-08T10:15:00+05:30",
        current_price=100.0,
        rsi=75.0,
        stoch_rsi_k=85.0,
        bb_lower=80.0,
        bb_middle=90.0,
        bb_upper=110.0,
    )

    assert result["overall_signal"] == "OVERBOUGHT"
    assert result["triggered_indicators"] == ["RSI", "Stochastic RSI", "Bollinger Bands"]


def test_build_after_hours_email_includes_daily_and_hourly_sections():
    daily_results = [{
        "symbol": "TCS",
        "overall_signal": "SELL / OVERBOUGHT",
        "current_price": 3500.0,
        "rsi": 72.5,
        "stoch_rsi_k": 82.0,
        "bb_lower": 3300.0,
        "bb_upper": 3700.0,
        "triggered_conditions": "RSI; Stochastic RSI",
        "rsi_condition": "Overbought",
        "stoch_rsi_condition": "Overbought",
        "bollinger_condition": "Overbought",
    }]
    hourly_results = [{
        "symbol": "RELIANCE",
        "overall_signal": "BUY / OVERSOLD",
        "current_price": 2850.0,
        "rsi": 28.4,
        "stoch_rsi_k": 18.2,
        "bb_lower": 2800.0,
        "bb_upper": 2920.0,
        "triggered_conditions": "RSI; Bollinger Bands",
        "rsi_condition": "Oversold",
        "stoch_rsi_condition": "Oversold",
        "bollinger_condition": "Oversold",
    }]

    subject, plain, html = build_after_hours_email(daily_results, hourly_results, [], "2026-09-08")

    assert "Daily Timeframe" in subject
    assert "Daily Timeframe" in plain
    assert "3:15 PM Hourly Timeframe" in plain
    assert "TCS" in plain
    assert "RELIANCE" in plain
    assert "Daily Timeframe" in html
    assert "3:15 PM Hourly Timeframe" in html
