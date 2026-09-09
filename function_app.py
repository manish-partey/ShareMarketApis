import json
import logging

import azure.functions as func

from dhan_daily_overbought_oversold_scanner import (
    email_after_hours_summary,
    is_trading_day,
    scan_hourly_market,
    send_email,
)


app = func.FunctionApp()


@app.timer_trigger(
    schedule="0 45 4-9 * * 1-5",
    arg_name="timer",
    run_on_startup=False,
    use_monitor=True,
)
def hourly_option_overbought_oversold(timer: func.TimerRequest) -> None:
    """Scan the latest completed hourly candle and email the results."""
    if timer.past_due:
        logging.warning("Hourly option scanner timer is past due")

    logging.info("Starting hourly option scanner")
    if not is_trading_day():
        logging.info("Skipping hourly option scan on a weekend or NSE holiday")
        return

    result = scan_hourly_market()
    send_email(
        result["results"],
        result["errors"],
        result["scan_time"].date().isoformat(),
    )
    logging.info(
        "Hourly option scan and email completed with %s signal(s) and %s error(s)",
        len(result["results"]),
        len(result["errors"]),
    )


@app.route(route="manual-post-market-scan", methods=["GET", "POST"], auth_level=func.AuthLevel.FUNCTION)
def manual_post_market_scan(req: func.HttpRequest) -> func.HttpResponse:
    """Send a post-market summary with daily and 3:15 PM hourly results."""
    try:
        summary = email_after_hours_summary()
        return func.HttpResponse(
            json.dumps(summary, default=str),
            status_code=200,
            mimetype="application/json",
        )
    except SystemExit as exc:
        logging.exception("Post-market manual scan failed")
        return func.HttpResponse(str(exc), status_code=500)
    except Exception as exc:
        logging.exception("Post-market manual scan failed")
        return func.HttpResponse(str(exc), status_code=500)
