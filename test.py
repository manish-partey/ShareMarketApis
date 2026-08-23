import os
import requests
from dotenv import load_dotenv

load_dotenv()

client_id = os.getenv("DHAN_CLIENT_ID")
access_token = os.getenv("DHAN_ACCESS_TOKEN")

url = "https://api.dhan.co/v2/profile"

headers = {
    "access-token": access_token
}

response = requests.get(url, headers=headers)

print("HTTP Status:", response.status_code)
print(response.text)