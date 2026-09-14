"""Download weather observations and forecasts, then calculate forecast errors."""

import requests
import pandas as pd

from config import RAW_DATA

# Coordinates for cities analysed
CITIES = {
    "Paris": (48.8566, 2.3522),
    "Amsterdam": (52.3676, 4.9041),
    "Brussels": (50.8503, 4.3517),
    "Frankfurt": (50.1109, 8.6821)
}

# Date range for weather forecast data
START_DATE = "2022-01-01"
END_DATE = "2025-12-31"


def realised_weather(city, lat, lon):
    """Download daily realised temperatures for one city."""
    url = "https://archive-api.open-meteo.com/v1/archive"

    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": START_DATE,
        "end_date": END_DATE,
        "daily": "temperature_2m_mean",
        "models": "gfs_seamless",
        "timezone": "GMT"
    }

    r = requests.get(url, params=params)
    r.raise_for_status()
    data = r.json()

    df = pd.DataFrame({
        "date": data["daily"]["time"],
        "temp_realised": data["daily"]["temperature_2m_mean"]
    })

    df["city"] = city
    return df

def forecast_weather(city, lat, lon):
    """Download hourly historical forecasts for one city."""
    url = "https://historical-forecast-api.open-meteo.com/v1/forecast"

    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": START_DATE,
        "end_date": END_DATE,
        "hourly": "temperature_2m_previous_day3",
        "models": "gfs_seamless",
        "timezone": "GMT"
    }

    r = requests.get(url, params=params)
    r.raise_for_status()
    data = r.json()

    df = pd.DataFrame({
        "time": data["hourly"]["time"],
        "temp_forecast": data["hourly"]["temperature_2m_previous_day3"]
    })

    df["city"] = city
    return df

def main():
    """Download weather data and save the aggregated forecast errors."""
    realised_dfs = []
    forecast_dfs = []

    for city, (lat, lon) in CITIES.items():
        print(f"Downloading weather data for {city} (lat: {lat}, lon: {lon})")
        realised_dfs.append(realised_weather(city, lat, lon))
        forecast_dfs.append(forecast_weather(city, lat, lon))

    realised = pd.concat(realised_dfs, ignore_index=True)
    forecast = pd.concat(forecast_dfs, ignore_index=True)

    realised["date"] = pd.to_datetime(realised["date"])
    forecast["time"] = pd.to_datetime(forecast["time"])

    forecast["forecast_date"] = forecast["time"].dt.floor("D")

    forecast["issue_date"] = (
        forecast["forecast_date"].dt.floor("D") - pd.Timedelta(days=3)
    )

    forecast_daily_city = (
        forecast
        .groupby(["issue_date", "forecast_date", "city"], as_index=False)
        ["temp_forecast"]
        .mean()
    )

    forecast_daily = (
        forecast_daily_city
        .groupby(["issue_date", "forecast_date"], as_index=False)
        ["temp_forecast"]
        .mean()
    )

    realised_daily = (
        realised
        .groupby("date", as_index=False)
        ["temp_realised"]
        .mean()
    )

    # Heating degree days use 18 degrees Celsius as the base temperature.
    realised_daily["HDD"] = (
        realised_daily["temp_realised"]
        .clip(upper=18)
        .rsub(18)
    )

    merged = forecast_daily.merge(
        realised_daily,
        left_on="forecast_date",
        right_on="date",
        how="inner"
    )
    merged["forecast_error"] = (
        merged["temp_realised"] - merged["temp_forecast"]
    )

    merged["abs_forecast_error"] = (
        merged["forecast_error"].abs()
    )
    
    output = (
        merged[
            [
                "forecast_date",
                "temp_realised",
                "issue_date",
                "temp_forecast",
                "HDD",
                "forecast_error",
                "abs_forecast_error"
            ]
        ]
        .rename(
            columns={
                "forecast_date": "date",
                "temp_realised": "realised_temp",
                "temp_forecast": "forecast_temp"
            }
        )
        .sort_values("date")
    )


    output.to_csv(RAW_DATA / "weather_forecast_errors.csv", index=False)
    print(f"Weather data saved: {len(output):,} rows")

if __name__ == "__main__":
    main()

