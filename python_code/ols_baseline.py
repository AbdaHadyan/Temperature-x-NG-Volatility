"""Fit the baseline OLS model and run residual diagnostics."""

import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from statsmodels.stats.diagnostic import het_breuschpagan, linear_reset
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.stattools import durbin_watson
import statsmodels.api as sm
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.diagnostic import acorr_ljungbox
import seaborn as sns

from config import PROCESSED_DATA, OUTPUT_GRAPHS


def main():
    """Fit the baseline and full models, then print diagnostic results."""
    df = pd.read_csv(PROCESSED_DATA / "weather_ttf_storage_aggregated.csv")

    control_columns = [
        "log_vol_lag1",
        "log_vol_lag5",
        "log_vol_lag22",
        "full",
        "HDD",
        "GPRD",
        *[f"month_{month}" for month in range(2, 13)],
    ]
    X_null = df[control_columns]
    y = df["future_log_vol_5d"]
    X_null = sm.add_constant(X_null)
    null_model = sm.OLS(y, X_null).fit(
        cov_type="HAC", cov_kwds={"maxlags": 5}
    )

    print(null_model.summary())

    X = df[["forecast_error", *control_columns]]
    X = sm.add_constant(X)
    model = sm.OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": 5})

    print(model.summary())

    
    # ols diagnostics
    
    jb_value, jb_pvalue = stats.jarque_bera(model.resid)
    skew = stats.skew(model.resid)
    kurtosis = stats.kurtosis(model.resid, fisher=True)
    print(f"Jarque-Bera test for normality of residuals:")
    print(f"JB Statistic: {jb_value}")
    print(f"P-Value: {jb_pvalue}")
    print(f"Skewness: {skew}")
    print(f"Kurtosis: {kurtosis}")
    
    # residual autocorrelation check
    sns.set_theme(style="whitegrid", context="paper")
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 6))

    plot_acf(model.resid, lags=30, ax=axes[0])
    axes[0].set_title("Autocorrelation of Residuals")
    axes[0].set_xlabel("Lag")
    axes[0].set_ylabel("Autocorrelation")
    plot_pacf(model.resid, lags=30, ax=axes[1])
    axes[1].set_title("Partial Autocorrelation of Residuals")
    axes[1].set_xlabel("Lag")
    axes[1].set_ylabel("Partial Autocorrelation")
    plt.savefig(OUTPUT_GRAPHS / "6_residuals_autocorrelation_functions.png", dpi=300)
    plt.show()

    result = acorr_ljungbox(model.resid, return_df=True)
    print("\nLjung-Box Test for Residual Autocorrelation")
    print(result)

    dw_statistic = durbin_watson(model.resid)
    print(f"Durbin-Watson test for autocorrelation of residuals:")
    print(f"DW Statistic: {dw_statistic}")

    bp_value, bp_pvalue, f_value, f_pvalue = het_breuschpagan(model.resid, model.model.exog)
    print(f"Breusch-Pagan test for heteroskedasticity:")
    print(f"BP Statistic: {bp_value}")
    print(f"P-Value: {bp_pvalue}")

    reset_test= linear_reset(model, power=3, use_f=True)
    print(f"Linear RESET test for model specification:")
    print(f"RESET Statistic: {reset_test.statistic}")
    print(f"P-Value: {reset_test.pvalue}")

    vif_data = pd.DataFrame({
        "variable": model.model.exog_names,
        "VIF": [variance_inflation_factor(model.model.exog, i) for i in range(model.model.exog.shape[1])]})

    print(vif_data.sort_values("VIF", ascending=False))





if __name__ == "__main__" :
    main()
