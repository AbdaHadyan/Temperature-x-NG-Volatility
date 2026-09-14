import data_aggregation
import exploratory_analysis
import gas_storage
import ols_baseline
import quantile_regressions
import ttf
import weather


def main():
    """Run the complete download, processing, and analysis pipeline."""
    print("Downloading weather forecast data")
    weather.main()
    print("Downloading gas storage data")
    gas_storage.main()
    print("Downloading TTF price data")
    ttf.main()
    print("Running data aggregation")
    data_aggregation.main()
    print("Running exploratory analysis")
    exploratory_analysis.main()
    print("Running OLS baseline regression")
    ols_baseline.main()
    print("Running quantile regression")
    quantile_regressions.main()


if __name__ == "__main__":
    main()
