"""Download TTF futures data and calculate volatility features."""

import pandas as pd
import numpy as np
import yfinance as yf

from config import RAW_DATA


# TTF=F is Yahoo Finance's ticker for Dutch TTF natural gas futures.
TICKER = "TTF=F"

# Date range for TTF futures data
START_DATE = "2022-01-01"
END_DATE = "2025-12-31"

def main():
    """Download daily prices and save returns and volatility features."""
    data = yf.download(
        tickers=TICKER,
        start=START_DATE,
        end=END_DATE,
        interval="1d"
    )


    ttf_data = data.reset_index()[["Date", "Close"]]
    ttf_data.columns = ["date", "price"]

    ttf_data["month"] = pd.to_datetime(ttf_data["date"]).dt.month
    month_dummies = pd.get_dummies(
        ttf_data["month"], prefix="month", drop_first=True
    ).astype(int)

    ttf_data = pd.concat([ttf_data, month_dummies], axis=1)
    ttf_data = ttf_data.drop(columns=["month"])

    # Calculate daily returns and realised volatility.
    ttf_data["daily_return"] = np.log(
        ttf_data["price"] / ttf_data["price"].shift(1)
    )
    ttf_data["daily_rv"] = ttf_data["daily_return"] ** 2

    eps = 1e-12

    ttf_data["log_daily_rv"] = np.log(ttf_data["daily_rv"] + eps)
    ttf_data["log_vol_lag1"] = np.log(ttf_data["daily_rv"].shift(1) + eps)
    ttf_data["log_vol_lag5"] = np.log(
        ttf_data["daily_rv"].shift(1).rolling(5).mean() + eps
    )
    ttf_data["log_vol_lag22"] = np.log(
        ttf_data["daily_rv"].shift(1).rolling(22).mean() + eps
    )

    ttf_data["future_log_vol_5d"] = np.log(ttf_data["daily_rv"].rolling(5).mean().shift(-5) + eps)

    ttf_data.to_csv(RAW_DATA / "ttf_data.csv", index=False)



if __name__ == "__main__":
    main()


