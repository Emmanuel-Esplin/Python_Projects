# Predict the gold price (GLD ETF) from the S&P 500, oil (USO) and the EUR/USD rate.
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from xgboost import XGBRegressor

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
sns.set_style("darkgrid", {"grid.color": ".6", "grid.linestyle": ":"})

_HERE = Path(__file__).parent
_DATA_PATH = _HERE / "data" / "gold_price_data.csv"
_OUTPUT_DIR = _HERE / "outputs" / "gold_price_prediction"

TARGET = "GLD"
TREND_WINDOW = 20


def load_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["Date"], date_format="%m/%d/%Y")
    df = df.sort_values("Date").set_index("Date")
    logging.info("Loaded data with shape %s", df.shape)
    logging.info("Missing values per column:\n%s", df.isna().sum().to_string())
    return df


def save_figure(fig: plt.Figure, filename: str) -> None:
    _OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out = _OUTPUT_DIR / filename
    fig.savefig(out, bbox_inches="tight", dpi=150)
    plt.close(fig)
    logging.info("Saved plot: %s", out)


def plot_correlation(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.heatmap(df.corr(), annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax)
    ax.set_title("Correlation Matrix Heatmap")
    save_figure(fig, "correlation_matrix.png")


def plot_price_and_trend(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(df.index, df[TARGET], alpha=0.5, label="Daily close")
    # The first TREND_WINDOW - 1 values of the rolling mean are NaN and are not drawn
    ax.plot(df.index, df[TARGET].rolling(window=TREND_WINDOW).mean(), label=f"{TREND_WINDOW}-day rolling mean")
    ax.set_title("Gold Price (GLD) Over Time")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price (USD)")
    ax.legend()
    save_figure(fig, "gold_price_trend.png")


def plot_distributions(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(12, 7))
    fig.suptitle("Distribution of data across columns", fontsize=16)
    for ax, col in zip(axes.flat, df.columns):
        sns.histplot(df[col], kde=True, ax=ax)
    for ax in axes.flat[len(df.columns):]:
        ax.set_visible(False)
    fig.tight_layout()
    save_figure(fig, "distributions.png")


def plot_boxplots(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(12, 6))
    for ax, col in zip(axes.flat, df.columns):
        sns.boxplot(x=df[col], color="lightblue", ax=ax)
    for ax in axes.flat[len(df.columns):]:
        ax.set_visible(False)
    fig.tight_layout()
    save_figure(fig, "boxplots.png")


def clip_outliers(train: pd.DataFrame, test: pd.DataFrame, lower: float = 0.05, upper: float = 0.95) -> tuple[pd.DataFrame, pd.DataFrame]:
    # Winsorize each column using limits learned from the training set only (avoids test-set leakage).
    lo, hi = train.quantile(lower), train.quantile(upper)
    return train.clip(lo, hi, axis=1), test.clip(lo, hi, axis=1)


def evaluate(name: str, model, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    pred = model.predict(X_test)
    result = {
        "model": name,
        "r2": r2_score(y_test, pred),
        "rmse": float(np.sqrt(mean_squared_error(y_test, pred))),
    }
    logging.info("%s -> R2: %.3f, RMSE: %.3f", name, result["r2"], result["rmse"])
    return result


def train_models(X_train: pd.DataFrame, y_train: pd.Series) -> dict:
    # Folds respect time order so validation data always comes after the data it is trained on.
    cv = TimeSeriesSplit(n_splits=5)

    lasso = GridSearchCV(
        make_pipeline(StandardScaler(), PolynomialFeatures(), Lasso(max_iter=50_000)),
        {"polynomialfeatures__degree": [1, 2, 3], "lasso__alpha": [0.01, 0.1, 1.0]},
        cv=cv,
        scoring="r2",
        n_jobs=-1,
    )
    forest = GridSearchCV(
        RandomForestRegressor(random_state=42),
        {"n_estimators": [100, 300], "max_depth": [None, 10]},
        cv=cv,
        scoring="r2",
        n_jobs=-1,
    )
    xgb = GridSearchCV(
        XGBRegressor(random_state=42),
        {"n_estimators": [200, 500], "max_depth": [3, 6], "learning_rate": [0.05, 0.1]},
        cv=cv,
        scoring="r2",
        n_jobs=-1,
    )

    models = {"Polynomial Lasso": lasso, "Random Forest": forest, "XGBoost": xgb}
    for name, search in models.items():
        search.fit(X_train, y_train)
        logging.info("%s best params: %s", name, search.best_params_)
    return {name: search.best_estimator_ for name, search in models.items()}


def main() -> None:
    df = load_data(_DATA_PATH)

    plot_correlation(df)
    plot_price_and_trend(df)
    plot_distributions(df)
    logging.info("Skewness:\n%s", df.skew().to_string())

    # SLV (silver ETF) moves almost in lockstep with gold (corr ~0.87), so it is excluded
    # to keep the model from simply reading the answer off a near-duplicate series.
    features = df.drop(columns=[TARGET, "SLV"])
    # Square root reduces the right skew in USO
    features["USO"] = np.sqrt(features["USO"])
    plot_boxplots(features.join(df[TARGET]))

    # Chronological split: a random split on daily prices leaks future information into training.
    X_train, X_test, y_train, y_test = train_test_split(features, df[TARGET], test_size=0.2, shuffle=False)
    X_train, X_test = clip_outliers(X_train, X_test)

    models = train_models(X_train, y_train)
    results = [evaluate(name, model, X_test, y_test) for name, model in models.items()]

    # Naive baseline: predict the last price seen in training for every test day.
    baseline = np.full(len(y_test), y_train.iloc[-1])
    results.append({
        "model": "Baseline (last train value)",
        "r2": r2_score(y_test, baseline),
        "rmse": float(np.sqrt(mean_squared_error(y_test, baseline))),
    })
    print(pd.DataFrame(results).to_string(index=False))


if __name__ == "__main__":
    main()
