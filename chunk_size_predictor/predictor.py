"""
Train stratified LinearRegression models on the **master CSV only**, then
predict for all eligible rows and perform end-of-run plotting.

This script does **not** scan raw benchmark folders. It expects a master CSV
already produced by `make_master_csv.py`.

Usage examples:
  python stratified_models.py \
      --master_csv csp_data_a100_llama3_8b \
      --pred_csv metrics_predictions.csv \
      --min_samples_per_bin 100
"""

import os
import argparse
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import plotly.express as px
import plotly.graph_objects as go

FEATURES = [
    "batch_num_tokens",
    "batch_num_prefill_tokens",
    "batch_num_decode_tokens",
    # "batch_size",
    "batch_decode_context",
    "batch_prefill_context",
]

PLOT_METRICS = True

TOKEN_RANGES: List[Tuple[float, float, str]] = [
    (0, 512, "tokens_0_512"),
    (513, float("inf"), "tokens_513_plus"),
]


def _resolve_path_relative_to_script(path_like: str) -> str:
    """
    If `path_like` is absolute, return as-is.
    Otherwise, treat it as relative to the directory containing this script.
    """
    if os.path.isabs(path_like):
        return path_like
    script_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(script_dir, path_like)


def train_stratified_models(df: pd.DataFrame, min_samples_per_bin: int = 100):
    df_filtered = df[df["batch_execution_time"] <= 1.0].copy()
    print(
        f"Filtered data shape: {df_filtered.shape} "
        f"(removed {len(df) - len(df_filtered)} rows with execution time > 1s)"
    )

    models = {}
    for low, high, name in TOKEN_RANGES:
        if high == float("inf"):
            rng_df = df_filtered[df_filtered["batch_num_tokens"] > low]
            print(f"Training model for batch_num_tokens > {low}")
        else:
            rng_df = df_filtered[
                (df_filtered["batch_num_tokens"] > low)
                & (df_filtered["batch_num_tokens"] <= high)
            ]
            print(f"Training model for {low} < batch_num_tokens <= {high}")

        print(f"Samples in bin '{name}': {len(rng_df)}")
        if len(rng_df) < min_samples_per_bin:
            print(f"Warning: Too few samples in range {low}-{high}, skipping bin '{name}'")
            continue

        X = rng_df[FEATURES]
        y = rng_df["batch_execution_time"]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        model = LinearRegression()
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        rmse = float(np.sqrt(mse))
        r2 = float(r2_score(y_test, y_pred))

        coef_df = (
            pd.DataFrame({"Feature": FEATURES, "Coefficient": model.coef_})
            .assign(Abs_Coefficient=lambda d: d["Coefficient"].abs())
            .sort_values("Abs_Coefficient", ascending=False)
        )

        print(f"Model for {name}: MSE={mse:.6f}, RMSE={rmse:.6f}, R²={r2:.4f}")
        print(coef_df)
        print(f"Intercept: {model.intercept_:.6f}")

        models[name] = {
            "model": model,
            "low": low,
            "high": high,
            "metrics": {"mse": float(mse), "rmse": rmse, "r2": r2},
            "coefficients": coef_df,
            "intercept": float(model.intercept_),
        }

    if not models:
        raise RuntimeError("No stratified models were trained — bins had insufficient samples.")

    return models


def predict_with_models(df: pd.DataFrame, models: Dict[str, dict]) -> pd.DataFrame:
    df = df.copy()
    mask_valid = df["batch_execution_time"] <= 1.0

    df.loc[mask_valid, "model_used"] = None
    df.loc[mask_valid, "predicted_time"] = np.nan

    for name, info in models.items():
        low, high = info["low"], info["high"]
        if high == float("inf"):
            mask_bin = mask_valid & (df["batch_num_tokens"] > low)
        else:
            mask_bin = mask_valid & (df["batch_num_tokens"] > low) & (df["batch_num_tokens"] <= high)

        rows = int(mask_bin.sum())
        if rows > 0:
            X = df.loc[mask_bin, FEATURES]
            preds = info["model"].predict(X)
            df.loc[mask_bin, "predicted_time"] = preds
            df.loc[mask_bin, "model_used"] = name

    # Fallback for unmatched valid rows
    missing = mask_valid & df["predicted_time"].isna()
    if missing.any():
        first_model_name = next(iter(models.keys()))
        print(f"Fallback: applying '{first_model_name}' to {int(missing.sum())} unmatched rows")
        X = df.loc[missing, FEATURES]
        preds = models[first_model_name]["model"].predict(X)
        df.loc[missing, "predicted_time"] = preds
        df.loc[missing, "model_used"] = f"{first_model_name}_fallback"

    return df


def end_of_run_plots_and_metrics(df_pred: pd.DataFrame, fig_dir: str = "figures") -> None:
    os.makedirs(fig_dir, exist_ok=True)

    usable = df_pred.dropna(subset=["predicted_time"]).copy()
    if usable.empty:
        print("No rows with predictions to plot.")
        return

    actual = usable["batch_execution_time"].values
    predicted = usable["predicted_time"].values

    mse = mean_squared_error(actual, predicted)
    rmse = float(np.sqrt(mse))
    r2 = float(r2_score(actual, predicted))

    print("=== Overall Model Performance ===")
    print(f"MSE: {mse:.6f}, RMSE: {rmse:.6f}, R²: {r2:.4f}")

    max_val = float(max(usable["batch_execution_time"].max(), usable["predicted_time"].max()))

    if PLOT_METRICS:
        fig_scatter = px.scatter(
            usable,
            x="batch_execution_time",
            y="predicted_time",
            color="model_used",
            labels={"batch_execution_time": "Actual Time", "predicted_time": "Predicted Time"},
            title=f"Actual vs Predicted (All Models Combined) - R²: {r2:.4f}",
        )
        fig_scatter.add_trace(
            go.Scatter(
                x=[0, max_val],
                y=[0, max_val],
                mode="lines",
                name="Perfect Prediction",
                line=dict(dash="dash"),
            )
        )

        usable["error"] = usable["batch_execution_time"] - usable["predicted_time"]
        fig_error = px.histogram(
            usable,
            x="error",
            color="model_used",
            nbins=50,
            title="Error Distribution by Model",
        )

        scatter_path = os.path.join(fig_dir, "overall_scatter.html")
        error_path = os.path.join(fig_dir, "error_histogram.html")
        fig_scatter.write_html(scatter_path)
        fig_error.write_html(error_path)

        print(f"Figures saved to:- {scatter_path}- {error_path}")

        # Only show figures at the very end
        fig_scatter.show()
        fig_error.show()


def main():
    parser = argparse.ArgumentParser(
        description="Train & evaluate stratified models using a prebuilt master CSV."
    )
    # Default to 'csp_data_a100_llama3_8b' (no extension), resolved relative to the script file.
    parser.add_argument(
        "--master_csv",
        type=str,
        default="csp_data_a100_llama3_8b",
        help="Master CSV filename (no extension by default). Relative paths are resolved next to this script.",
    )
    parser.add_argument(
        "--pred_csv",
        type=str,
        default="metrics_predictions.csv",
        help="Path to write predictions CSV (resolved from current working directory).",
    )
    parser.add_argument(
        "--min_samples_per_bin",
        type=int,
        default=100,
        help="Minimum rows required per bin to train a model",
    )

    args = parser.parse_args()

    # Always load master CSV from the script's directory (unless an absolute path was provided).
    master_csv_path = _resolve_path_relative_to_script(args.master_csv)

    print(f"Loading master CSV from: {master_csv_path}")
    df_all = pd.read_csv(master_csv_path)

    print("Training stratified models...")
    models = train_stratified_models(df_all, min_samples_per_bin=args.min_samples_per_bin)

    print("Predicting across the full dataset...")
    df_pred = predict_with_models(df_all, models)

    df_pred.to_csv(args.pred_csv, index=False)
    print(f"Predictions saved to: {args.pred_csv}")

    end_of_run_plots_and_metrics(df_pred, fig_dir="figures")


if __name__ == "__main__":
    main()
