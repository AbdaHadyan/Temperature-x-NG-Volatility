"""Create exploratory plots and summary statistics for the analysis dataset."""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import skew, kurtosis, jarque_bera
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller
import statsmodels.api as sm

from config import OUTPUT_GRAPHS, PROCESSED_DATA


def main():
    """Print descriptive statistics and save exploratory analysis plots."""
    df = pd.read_csv(PROCESSED_DATA / "weather_ttf_storage_aggregated.csv")

    df["date"] = pd.to_datetime(df["date"])
    df["forecast_date"] = pd.to_datetime(df["issue_date"])

    summary_columns = [
        "future_log_vol_5d",
        "forecast_error",
        "log_vol_lag1",
        "log_vol_lag5",
        "log_vol_lag22",
        "full",
        "HDD",
        "GPRD",
        "daily_rv",
        "log_daily_rv",
    ]
    for col in summary_columns:
        print(f"\nSummary statistics for {col}:")
        print(df[col].describe())
        print(f"Skewness: {skew(df[col])}")
        print(f"Kurtosis: {kurtosis(df[col], fisher=True)}")
        jb_stat, jb_pvalue = jarque_bera(df[col])
        print(f"Jarque-Bera test statistic: {jb_stat}, p-value: {jb_pvalue}")
        adf = adfuller(df[col])
        print(f"ADF test statistic: {adf[0]}, p-value: {adf[1]}")
        if col == "future_log_vol_5d":
            adf = adfuller(df[col], regression="ct")
            print(f"ADF test statistic (with trend): {adf[0]}, p-value: {adf[1]}")

    sns.set_theme(style="whitegrid", context="paper")

    plots = [
        ("future_log_vol_5d", "$RV_{t+1:t+5}$", "Future Log Volatility (5d)", "blue"),
        ("forecast_error", "$TFE_{t}$", "Forecast Error", "orange")
    ]

    fig, axes = plt.subplots(
        nrows=len(plots), ncols=1, figsize=(10, 6), sharex=True
    )

    for ax, (column, label, title, colour) in zip(axes, plots):
        sns.lineplot(
            data=df,
            x="date",
            y=column,
            ax=ax,
            color=colour
        )

        ax.set_title(title)
        ax.set_xlabel("")
        ax.set_ylabel(label)

    axes[-1].set_xlabel("Date")


    fig.savefig(OUTPUT_GRAPHS / "1_time_series_plots.png",dpi=300)
    plt.show()

    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 6))
    sns.histplot(df["daily_rv"], bins=30, stat="density", kde=True, ax=axes[0])
    axes[0].set_title("Distribution of Daily Realised Volatility")
    axes[0].set_xlabel("$RV_{t}$")
    axes[0].set_ylabel("Frequency")

    sns.histplot(df["log_daily_rv"], bins=30, stat="density", kde=True, ax=axes[1])
    axes[1].set_title("Distribution of Log Daily Realised Volatility")
    axes[1].set_xlabel("$ln(RV_{t})$")
    axes[1].set_ylabel("Frequency")

   
    plt.savefig(OUTPUT_GRAPHS / "2_volatility_distributions.png", dpi=300)
    plt.show()

    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 6))
    plot_acf(df["future_log_vol_5d"], lags=30, ax=axes[0])
    axes[0].set_title("Autocorrelation of Forward Volatility")
    axes[0].set_xlabel("Lag")
    axes[0].set_ylabel("Autocorrelation")

    plot_pacf(df["future_log_vol_5d"], lags=30, ax=axes[1])
    axes[1].set_title("Partial Autocorrelation of Forward Volatility")
    axes[1].set_xlabel("Lag")
    axes[1].set_ylabel("Partial Autocorrelation")

    plt.savefig(OUTPUT_GRAPHS / "3_autocorrelation_functions.png", dpi=300)
    plt.show()

    fig, ax = plt.subplots(figsize=(10, 6))

    hb = ax.hexbin(
        df["future_log_vol_5d"],
        df["forecast_error"],
        gridsize=35,
        cmap="Blues",
        mincnt=1
    )

    fig.colorbar(hb, ax=ax, label="Number of observations")

    ax.set_title("Forecast Error vs Forward Log Volatility")
    ax.set_xlabel(r"$RV_{t+1:t+5}$")
    ax.set_ylabel(r"$TFE_t$")
    ax.grid(True, alpha=0.2)


    plt.savefig(OUTPUT_GRAPHS / "4_forecast_error_vs_volatility.png",dpi=300)
    plt.show()

    fig, ax = plt.subplots(figsize=(10, 6))
    correlation_columns = [
        "future_log_vol_5d",
        "forecast_error",
        "log_vol_lag1",
        "log_vol_lag5",
        "log_vol_lag22",
        "full",
        "HDD",
        "GPRD",
    ]
    regression_columns = [
        "forecast_error",
        "log_vol_lag1",
        "log_vol_lag5",
        "log_vol_lag22",
        "full",
        "HDD",
        "GPRD",
        *[f"month_{month}" for month in range(2, 13)],
    ]
    corr_matrix = df[correlation_columns].corr().round(3)
    condition_no = np.linalg.cond(sm.add_constant(df[regression_columns]))
    print(condition_no)
    sns.heatmap(corr_matrix, fmt=".3f", annot=True, cmap="coolwarm", center=0)
    labels = [
        "$RV_{t+1:+5}$",
        "$TFE_{t}$",
        "$RV_{d}$",
        "$RV_{w}$",
        "$RV_{m}$",
        "$FULL_{t}$",
        "$HDD_{t}$",
        "$GPRD_{t}$",
    ]
    ax.set_title("Correlation Matrix")
    ax.set_xticklabels(labels, rotation=45)
    ax.set_yticklabels(labels, rotation=0)


    plt.savefig(OUTPUT_GRAPHS / "5_correlation_matrix.png", dpi=300)
    plt.show()




if __name__ == "__main__":
    main()


