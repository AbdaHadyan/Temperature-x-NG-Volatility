"""Download European Union gas storage data from the AGSI API."""

import requests
import pandas as pd
import os
from dotenv import load_dotenv
from config import RAW_DATA


# API Key for AGSI API, loaded from .env file
load_dotenv()
API_KEY = os.getenv("AGSI_API_KEY")
if not API_KEY:
    raise ValueError("AGSI_API_KEY not set. Add API key to .env file")


# URL for AGSI API
URL = "https://agsi.gie.eu/api/data/EU"

# Date range for weather forecast data
START_DATE = "2022-01-01"
END_DATE = "2025-12-31"

# EU Country Code
COUNTRY = "EU"

def main():
    """Download all API pages and save the storage dataset."""
    headers = {
        "x-key": API_KEY,
        "Accept": "application/json"
    }
    params = {
        "country": COUNTRY,
        "from": START_DATE,
        "to": END_DATE
    }

    response = requests.get(URL, headers=headers, params=params)
    response.raise_for_status()

    json_data = response.json()

    last_page = json_data["last_page"]
    all_data = json_data["data"]

    print(f"Found {last_page} pages")

    # Remaining pages
    for page in range(2, last_page + 1):
        params["page"] = page

        response = requests.get(URL, headers=headers, params=params)
        response.raise_for_status()

        page_data = response.json()["data"]
        all_data.extend(page_data)

        print(f"Downloaded page {page}/{last_page}")

    df = pd.DataFrame(all_data)

    df = df[["gasDayStart","gasInStorage", "workingGasVolume", "full", "trend"]]

    df["gasDayStart"] = pd.to_datetime(df["gasDayStart"])
    for col in ["gasInStorage", "workingGasVolume", "full", "trend"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.sort_values("gasDayStart")

    df.to_csv(RAW_DATA / "gas_storage_data.csv", index=False)
    print(f"Rows downloaded: {len(df)}")
    print(f"Gas storage data saved: {len(df)} rows")


    

if __name__ == "__main__":
    main()

