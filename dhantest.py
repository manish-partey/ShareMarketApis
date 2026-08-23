import os

from dotenv import load_dotenv
from dhanhq import DhanContext, dhanhq

load_dotenv()

client_id = os.getenv("DHAN_CLIENT_ID")
access_token = os.getenv("DHAN_ACCESS_TOKEN")

if not client_id or not access_token:
    raise SystemExit("Missing DHAN_CLIENT_ID or DHAN_ACCESS_TOKEN in .env")

context = DhanContext(client_id, access_token)
dhan = dhanhq(context)

fund_limits = dhan.get_fund_limits()
print(fund_limits)