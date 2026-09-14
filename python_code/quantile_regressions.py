import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.formula.api as smf
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.stats.stattools import durbin_watson

from config import PROCESSED_DATA, OUTPUT_GRAPHS



FORMULA = (
    "future_log_vol_5d ~ forecast_error + log_vol_lag1 + log_vol_lag5 + log_vol_lag22 + GPRD + HDD + full + "
    "month_2 + month_3 + month_4 + month_5 + month_6 + month_7 + month_8 + month_9 + month_10 + month_11 + month_12"
)

NULL_FORMULA = (
    "future_log_vol_5d ~  log_vol_lag1 + log_vol_lag5 + log_vol_lag22 + GPRD + HDD + full + "
    "month_2 + month_3 + month_4 + month_5 + month_6 + month_7 + month_8 + month_9 + month_10 + month_11 + month_12"
)

QUANTILES = [0.10, 0.25, 0.50, 0.75, 0.90]

ROBUST_COV_TYPE = "robust"
MAX_ITER = 5000
CI_ALPHA = 0.10
INTERVAL_ALPHA = 0.20







# model fitting

def fit_quantile_models(df, formula):
    """Fit the configured quantile regressions for a dataset."""
    return {
        q: smf.quantreg(formula, df).fit(
            q=q,
            vcov=ROBUST_COV_TYPE,
            max_iter=MAX_ITER,
        )
        for q in QUANTILES
    }


def extract_forecast_error_summary(models):
    """Extract forecast-error coefficients and confidence intervals."""

    rows = []

    for q, model in models.items():
        ci = model.conf_int(alpha=CI_ALPHA).loc["forecast_error"]

        rows.append(
            {
                "quantile": q,
                "coefficient": model.params["forecast_error"],
                "ci_low": ci[0],
                "ci_high": ci[1],
                "p_value": model.pvalues["forecast_error"],
            }
        )

    return pd.DataFrame(rows)


# diagnostics

def run_residual_diagnostics(models):
    """Run residual autocorrelation diagnostics for the 90th quantile."""
    residuals = models[0.90].resid

    ljung_box = acorr_ljungbox(
        residuals,
        lags=[5, 10, 20],
        return_df=True,
    )

    durbin_watson_stat = durbin_watson(residuals)

    print("90th Quantile Residual Diagnostics")
    print(ljung_box[["lb_stat", "lb_pvalue"]].to_string())
    print(f"Durbin-Watson statistic: {durbin_watson_stat:.4f}")

    return {
        "ljung_box": ljung_box,
        "durbin_watson": durbin_watson_stat,
    }


# forecast evaluation

def generate_predictions(models, df):
    """Generate predictions for each quantile."""
    predictions = pd.DataFrame(index=df.index)

    for q, model in models.items():
        column = f"q_{int(q * 100)}"
        predictions[column] = model.predict(df)

    predictions["realised"] = df["future_log_vol_5d"]

    return predictions



# prediction interval evaluation


def evaluate_interval(predictions):
    """Calculate coverage of the 80% prediction interval."""
    lower = predictions["q_10"]
    upper = predictions["q_90"]
    realised = predictions["realised"]

    within_interval = (realised >= lower) & (realised <= upper)

    coverage = within_interval.mean()


    return {
        "coverage": coverage
    }


def print_interval_comparison(full_predictions, null_predictions):
    """Print and return interval coverage for both model specifications."""
    full = evaluate_interval(full_predictions)
    null = evaluate_interval(null_predictions)



    print("\n80% Prediction Interval Performance")
    print(f"Full model coverage: {full['coverage']:.3f}")
    print(f"Null model coverage: {null['coverage']:.3f}")

    return {
        "full": full,
        "null": null
    }


# quantile crossing


def check_quantile_crossing(predictions):
    """Count prediction crossings between adjacent quantiles."""
    pairs = [
        ("q_10", "q_25"),
        ("q_25", "q_50"),
        ("q_50", "q_75"),
        ("q_75", "q_90"),
    ]

    crossing_rates = {}


    print("\nQuantile Crossing")

    for lower, upper in pairs:
        crossing = predictions[lower] > predictions[upper]
        crossing_rates[f"{lower}_vs_{upper}"] = crossing.mean()
        print(f"\nCrossing between {lower} and {upper}:")
        print(f"Number of crossings: {crossing.sum()}")
        print(f"Crossing rate: {crossing.mean():.4%}")

    return crossing_rates




def run_model(df, sample_name):

    print(sample_name)


    # Fit full and null models
    full_models = fit_quantile_models(df, FORMULA)
    null_models = fit_quantile_models(df, NULL_FORMULA)

    # Coefficient results
    summary = extract_forecast_error_summary(full_models)

    # Diagnostics
    diagnostics = run_residual_diagnostics(full_models)

    # Predictions
    full_predictions = generate_predictions(full_models, df)
    null_predictions = generate_predictions(null_models, df)

    # Quantile crossing
    crossings = check_quantile_crossing(full_predictions)



    # Prediction intervals
    interval_results = print_interval_comparison(
        full_predictions,
        null_predictions,
    )

    return {
        "models": full_models,
        "null_models": null_models,
        "summary_df": summary,
        "diagnostics": diagnostics,
        "full_predictions": full_predictions,
        "null_predictions": null_predictions,
        "crossings": crossings,
        "interval_results": interval_results,
    }


def compare_samples(full_sample, post_crisis_sample):
    """Compare forecast-error coefficients across samples."""
    return (
        full_sample["summary_df"]
        .rename(
            columns={
                "coefficient": "coef_full",
                "p_value": "p_full",
                "ci_low": "ci_low_full",
                "ci_high": "ci_high_full",
            }
        )
        .merge(
            post_crisis_sample["summary_df"].rename(
                columns={
                    "coefficient": "coef_post_crisis",
                    "p_value": "p_post_crisis",
                    "ci_low": "ci_low_post_crisis",
                    "ci_high": "ci_high_post_crisis",
                }
            ),
            on="quantile",
        )
    )


def plot_coefficient_comparison(full_sample, post_crisis_sample):
    """Plot forecast-error coefficients and confidence intervals."""
    full = full_sample["summary_df"]
    post_crisis = post_crisis_sample["summary_df"]
    sns.set_theme(style="whitegrid", context="paper")
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(
        full["quantile"],
        full["coefficient"],
        marker="o",
        color="steelblue",
        linewidth=2,
        label="Full Sample",
    )

    ax.fill_between(
        full["quantile"],
        full["ci_low"],
        full["ci_high"],
        color="steelblue",
        alpha=0.15,
    )

    ax.plot(
        post_crisis["quantile"],
        post_crisis["coefficient"],
        marker="s",
        color="orange",
        linewidth=2,
        label="Post-Crisis",
    )

    ax.fill_between(
        post_crisis["quantile"],
        post_crisis["ci_low"],
        post_crisis["ci_high"],
        color="orange",
        alpha=0.15,
    )

    ax.axhline(
        0,
        color="black",
        linestyle=":",
        linewidth=1,
    )

    ax.set(
        xlabel="Quantile (τ)",
        ylabel="Coefficient on Temperature Forecast Error",
        title="Quantile Regression Coefficients: "
              "Full Sample vs Post-Crisis",
        xticks=QUANTILES,
    )

    ax.legend()

    fig.tight_layout()
    fig.savefig(OUTPUT_GRAPHS / "7_quantile_regression_comparison.png",dpi=300)

    plt.show()



def analyse_upper_mid_tail(df):
    """Analyse observations in the upper quartile of future volatility."""
    threshold = df["future_log_vol_5d"].quantile(0.75)
    upper_mid = df[df["future_log_vol_5d"] >= threshold]

    print("\nUpper-Mid Tail Analysis")

    print("Observations by year:")
    print(
        upper_mid["date"]
        .dt.year
        .value_counts()
        .sort_index()
    )

    print("\nForecast error statistics:")
    print(upper_mid["forecast_error"].describe())

def load_data():
    """Load and prepare the analysis dataset."""
    df = pd.read_csv(
        PROCESSED_DATA / "weather_ttf_storage_aggregated.csv"
    )

    df["date"] = pd.to_datetime(df["date"])

    return df


def main():
    df = load_data()

    # Run analyses
    full_sample = run_model(
        df,
        "Full Sample",
    )

    post_crisis_sample = run_model(
        df[df["date"] >= "2023-01-01"].copy(),
        "Post-Crisis Sample",
    )

    # Compare coefficients
    comparison = compare_samples(
        full_sample,
        post_crisis_sample,
    )

    print("\nCoefficient Comparison")
    print(comparison.round(4).to_string(index=False))

    # Plot results
    plot_coefficient_comparison(
        full_sample,
        post_crisis_sample,
    )

    # Tail analysis
    analyse_upper_mid_tail(df)
    


if __name__ == "__main__":
    main()
