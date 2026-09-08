import logging
from datetime import datetime

import azure.functions as func

from dhan_daily_overbought_oversold_scanner import (
    IST,
    email_after_hours_summary,
    is_within_market_scan_window,
    scan_hourly_market,
    send_after_hours_summary_email,
)


app = func.FunctionApp()


@app.timer_trigger(
    schedule="0 45 4-9 * * 1-5",
    arg_name="timer",
    run_on_startup=False,
    use_monitor=True,
)
def hourly_option_overbought_oversold(timer: func.TimerRequest) -> None:
    """Run the hourly option-eligible stock scan during Indian market hours."""
    if timer.past_due:
        logging.warning("Hourly option scanner timer is past due")

    current_time = datetime.now(IST)
    if not is_within_market_scan_window(current_time):
        logging.info("Skipping hourly option scan outside the Indian market window")
        return

    logging.info("Starting hourly option scanner for Indian market")
    result = scan_hourly_market()
    logging.info(
        "Hourly option scan completed with %s signal(s) and %s error(s)",
        len(result["results"]),
        len(result["errors"]),
    )


@app.route(route="manual-post-market-scan", methods=["GET", "POST"], auth_level=func.AuthLevel.FUNCTION)
def manual_post_market_scan(req: func.HttpRequest) -> func.HttpResponse:
    """Send a post-market summary with daily and 3:15 PM hourly results."""
    try:
        summary = email_after_hours_summary()
        return func.HttpResponse(
            "Post-market scan email sent successfully.",
            status_code=200,
        )
    except SystemExit as exc:
        logging.exception("Post-market manual scan failed")
        return func.HttpResponse(str(exc), status_code=500)
    except Exception as exc:
        logging.exception("Post-market manual scan failed")
        return func.HttpResponse(str(exc), status_code=500)
