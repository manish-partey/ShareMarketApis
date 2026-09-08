import logging

import azure.functions as func

from dhan_daily_overbought_oversold_scanner import run_scan


app = func.FunctionApp()


@app.timer_trigger(
    schedule="0 30 4 * * *",
    arg_name="timer",
    run_on_startup=False,
    use_monitor=True,
)
def daily_overbought_oversold(timer: func.TimerRequest) -> None:
    """Run the completed-daily-candle scan at 10:00 AM IST (04:30 UTC)."""
    if timer.past_due:
        logging.warning("Daily overbought/oversold timer is past due")

    logging.info("Starting daily overbought/oversold scan")
    run_scan()
    logging.info("Daily overbought/oversold scan completed")
