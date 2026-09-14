
"""Combine the input datasets and save the analysis-ready versions."""

import pandas as pd

from config import PROCESSED_DATA, RAW_DATA


def main():
    """Merge the raw datasets, record missing values, and save outputs."""
    gas = pd.read_csv(RAW_DATA / "gas_storage_data.csv")
    ttf = pd.read_csv(RAW_DATA / "ttf_data.csv")
    weather = pd.read_csv(RAW_DATA / "weather_forecast_errors.csv")
    gpr = pd.read_excel(RAW_DATA / "data_gpr_daily_recent.xls")

    # Normalise dates before performing the as-of merges.
    gas["gasDayStart"] = pd.to_datetime(gas["gasDayStart"]).dt.normalize()
    ttf["date"] = pd.to_datetime(ttf["date"]).dt.normalize()
    weather["date"] = pd.to_datetime(weather["date"]).dt.normalize()
    weather["issue_date"] = pd.to_datetime(weather["issue_date"]).dt.normalize()
    gpr = gpr.rename(columns={"DAY": "issue_date"})
    gpr["issue_date"] = pd.to_datetime(gpr["issue_date"], format="%Y%m%d").dt.normalize()
    gpr = gpr[["issue_date", "GPRD"]]


    month_cols = [f"month_{i}" for i in range(2, 13)]

    ttf_dates = set(ttf["date"])
    weather = weather[weather["date"].isin(ttf_dates)]
    gas_sorted = (
        gas
        .rename(columns={"gasDayStart": "issue_date"})
        .sort_values("issue_date")
    )

    df = pd.merge_asof(
        weather.sort_values("issue_date"),
        gas_sorted,
        on="issue_date",
        direction="backward"
    )
    ttf_sorted = ttf.sort_values("date").reset_index(drop=True)

    ttf_issue_columns = [
        "date",
        "price",
        "log_vol_lag1",
        "log_vol_lag5",
        "log_vol_lag22",
        "future_log_vol_5d",
        "daily_rv",
        "log_daily_rv",
        *month_cols,
    ]
    df = pd.merge_asof(
        df.sort_values("issue_date"),
        ttf_sorted[ttf_issue_columns].rename(
            columns={
                "date": "issue_date",
                "price": "issue_date_price",
            }
        ),
        on="issue_date",
        direction="backward"
    )

    df = pd.merge_asof(
        df.sort_values("date"),
        ttf_sorted[["date", "price"]].rename(columns={"price": "actual_date_price"}),
        on="date",
        direction="backward"
    )

    df = df.sort_values("date").reset_index(drop=True)
    trading_days = set(ttf["date"])
    df["actual_date_price"] = df["actual_date_price"].where(
        df["date"].isin(trading_days)
    )

    df = pd.merge_asof(
        df.sort_values("issue_date"),
        gpr,
        on="issue_date",
        direction="backward"
    )

    df = df.sort_values(["issue_date", "date"]).reset_index(drop=True)

    # Standardise continuous variables for the condition-number check.
    std_df = df.copy()
    continuous_vars = [
        "forecast_error",
        "log_vol_lag1",
        "log_vol_lag5",
        "log_vol_lag22",
        "full",
        "HDD",
        "GPRD",
    ]

    for var in continuous_vars:
        std_df[var] = (df[var] - df[var].mean()) / df[var].std()

    # Save a concise report before dropping incomplete rows.
    missing_df = df[df.isnull().any(axis=1)].copy()
    print(f"Rows with missing values: {len(missing_df)}")
    missing_df["missing_cols"] = (
        missing_df.isna()
        .apply(lambda row: row.index[row].tolist(), axis=1)
    )

    missing_df[["issue_date", "date", "missing_cols"]].to_csv(
        PROCESSED_DATA / "missing_data.csv", index=False
    )

    df = df.dropna()
    std_df = std_df.dropna()

    df.to_csv(PROCESSED_DATA / "weather_ttf_storage_aggregated.csv", index=False)
    std_df.to_csv(PROCESSED_DATA / "standardised_data.csv", index=False)

    print(f"Standardised data saved: {len(std_df):,} rows")


if __name__ == "__main__":
    main()
