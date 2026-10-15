"""Self-contained GluonTS benchmark: DeepAR, PatchTST, TFT.

Reads raw Coffee Shop Sales data, does minimal preprocessing, and runs
three deep learning models on the same 5-window rolling backtest as the
production pipeline.  All models run on CPU.

Outputs CSVs to reports/model_quality/deep_learning_benchmark/.
"""

from __future__ import annotations

import json
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

DATA_DIR = Path(__file__).parent
SALES_FILE = DATA_DIR / "Coffee Shop Sales.xlsx"
WEATHER_FILE = DATA_DIR / "nyc_weather_2023_h1.csv"
OUTPUT_DIR = DATA_DIR / "reports" / "model_quality" / "deep_learning_benchmark"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PREDICTION_LOOKAHEAD = 7
BACKTEST_WINDOW_COUNT = 5
ENTITY_KEYS = ["store_id", "product_id"]


# ── Data loading (minimal, self-contained) ─────────────────────────────
def load_data() -> pd.DataFrame:
    """Aggregate raw data to store × product × day with full panel."""
    sales = pd.read_excel(SALES_FILE)
    sales["date"] = pd.to_datetime(sales["transaction_date"])

    df = sales.groupby(["date", "store_id", "product_id"]).agg(
        qty=("transaction_qty", "sum"),
    ).reset_index()

    # Build full panel to fill zeros
    dates = pd.date_range(df["date"].min(), df["date"].max(), freq="D")
    stores = sorted(df["store_id"].unique())
    products = sorted(df["product_id"].unique())

    full_idx = pd.MultiIndex.from_product(
        [dates, stores, products], names=["date", "store_id", "product_id"]
    )
    df_full = pd.DataFrame(index=full_idx).reset_index()
    df_full["date"] = pd.to_datetime(df_full["date"])
    df = df_full.merge(df, on=["date", "store_id", "product_id"], how="left")
    df["qty"] = df["qty"].fillna(0).astype(int)
    df["item_id"] = df["store_id"].astype(str) + "_" + df["product_id"].astype(str)
    return df


def select_backtest_origins(dates) -> List[pd.Timestamp]:
    """Same logic as production: latest fixed non-overlapping origins."""
    normalized = pd.DatetimeIndex(pd.to_datetime(dates)).sort_values().unique()
    required = BACKTEST_WINDOW_COUNT * PREDICTION_LOOKAHEAD
    if len(normalized) < required:
        raise ValueError(f"Need at least {required} dates, got {len(normalized)}")
    return [
        pd.Timestamp(normalized[-required + i * PREDICTION_LOOKAHEAD])
        for i in range(BACKTEST_WINDOW_COUNT)
    ]


# ── GluonTS imports ────────────────────────────────────────────────────
try:
    from gluonts.torch import (
        DeepAREstimator,
        PatchTSTEstimator,
    )
    from gluonts.torch.model.tft import TemporalFusionTransformerEstimator
    from gluonts.dataset.pandas import PandasDataset
    HAS_GLUONTS = True
except ImportError as exc:
    HAS_GLUONTS = False
    print(f"ERROR: GluonTS not available: {exc}", file=sys.stderr)
    sys.exit(1)

TRAINER_KWARGS = {
    "max_epochs": 40,
    "accelerator": "cpu",
}

FREQ = "D"
CTX = max(PREDICTION_LOOKAHEAD * 4, 28)
HORIZON = PREDICTION_LOOKAHEAD


def build_estimator(method: str):
    """Return a GluonTS estimator for the given method."""
    if method == "deepar":
        return DeepAREstimator(
            freq=FREQ, prediction_length=HORIZON,
            lr=1e-3, patience=8, batch_size=32,
            num_batches_per_epoch=30,
            trainer_kwargs=TRAINER_KWARGS,
        )
    elif method == "patchtst":
        return PatchTSTEstimator(
            prediction_length=HORIZON,
            context_length=CTX,
            patch_len=7, stride=3, d_model=32,
            batch_size=32, num_batches_per_epoch=30,
            trainer_kwargs=TRAINER_KWARGS,
        )
    elif method == "tft":
        return TemporalFusionTransformerEstimator(
            freq=FREQ, prediction_length=HORIZON,
            context_length=CTX,
            hidden_dim=32, num_heads=2, dropout_rate=0.1,
            lr=1e-3, patience=8, batch_size=32,
            num_batches_per_epoch=30,
            trainer_kwargs=TRAINER_KWARGS,
        )
    raise ValueError(f"Unknown method: {method}")


def run_model(method: str, df: pd.DataFrame, origins: list) -> pd.DataFrame:
    """Train & predict for all origins, return merged predictions."""
    rows = []
    for idx, origin in enumerate(origins):
        print(f"   [{method}] origin {origin.date()} ({idx+1}/{len(origins)})")
        cutoff = origin - pd.Timedelta(days=2 * PREDICTION_LOOKAHEAD)
        train_df = df[df["date"] < cutoff].copy()

        # Build GluonTS-compatible dict dataset
        from gluonts.dataset.common import ListDataset
        from gluonts.dataset.field_names import FieldName
        data_entries = []
        for item_id, grp in train_df.groupby("item_id"):
            grp = grp.sort_values("date")
            data_entries.append({
                FieldName.ITEM_ID: item_id,
                FieldName.START: pd.Period(grp["date"].min(), freq="D"),
                FieldName.TARGET: grp["qty"].to_numpy(dtype=np.float32),
            })
        dataset = ListDataset(data_entries, freq=FREQ)

        try:
            predictor = build_estimator(method).train(dataset)

            for fc in predictor.predict(dataset):
                item = fc.item_id
                median = np.asarray(fc.quantile(0.5), dtype=float)
                for h in range(HORIZON):
                    test_date = origin + pd.Timedelta(days=h)
                    rows.append({
                        "item_id": item,
                        "date": test_date,
                        "forecast_origin": origin,
                        "method": method,
                        "horizon_day": h + 1,
                        "prediction": float(max(0, median[h])),
                    })
        except Exception as e:
            print(f"   [{method}] ERROR origin {origin.date()}: {e}")
            continue

    if not rows:
        return pd.DataFrame()

    result = pd.DataFrame(rows)
    # Parse store_id / product_id from item_id
    parts = result["item_id"].str.split("_", n=1, expand=True)
    result["store_id"] = parts[0].astype(int)
    result["product_id"] = parts[1].astype(int)

    # Join ground truth
    gt = df[["date", "store_id", "product_id", "qty"]].copy()
    result = result.merge(gt, on=["date", "store_id", "product_id"], how="left")
    return result


# ── Metrics (minimal) ──────────────────────────────────────────────────
def calc_metrics(preds: pd.DataFrame) -> dict:
    y = preds["qty"].to_numpy(dtype=float)
    y_hat = preds["prediction"].to_numpy(dtype=float)
    if len(y) == 0:
        return {}
    mae = float(np.mean(np.abs(y - y_hat)))
    denom = np.sum(y) or 1.0
    wape = float(np.sum(np.abs(y - y_hat)) / denom)
    ss_res = np.sum((y - y_hat) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    # Direction accuracy
    imports = preds.sort_values(["forecast_origin", "store_id", "product_id", "date"])
    groups = imports.groupby(["forecast_origin", "store_id", "product_id"], sort=False)
    dy = groups["qty"].diff()
    dp = groups["prediction"].diff()
    valid = dy.notna() & dp.notna()
    if valid.sum() > 0:
        dir_hit = float(np.mean(np.sign(dy[valid]) == np.sign(dp[valid])))
    else:
        dir_hit = float("nan")

    return {
        "count": len(preds),
        "mae": round(mae, 6),
        "wape": round(float(wape), 6),
        "r2": round(float(r2), 6),
        "direction_hit_rate": round(dir_hit, 6),
    }


# ── Main ───────────────────────────────────────────────────────────────
def main():
    print("=" * 64)
    print("GluonTS Benchmark: DeepAR, PatchTST, TFT")
    print("=" * 64)

    print("\n1. Loading data...")
    df = load_data()
    origins = select_backtest_origins(df["date"].unique())
    print(f"   {df['date'].nunique()} days, {df['item_id'].nunique()} series, "
          f"{len(origins)} backtest origins")

    results = {}
    for method in ["deepar", "patchtst", "tft"]:
        print(f"\n2. Running {method}...")
        preds = run_model(method, df, origins)
        if len(preds) == 0:
            print(f"   {method}: NO predictions")
            continue
        results[method] = preds
        print(f"   {method}: {len(preds)} predictions")

    # Combine all predictions
    print("\n3. Saving predictions...")
    if not results:
        print("   No predictions generated by any model.")
        return
    all_preds = pd.concat(results.values(), ignore_index=True)
    preds_path = OUTPUT_DIR / "dl_benchmark_predictions.csv"
    all_preds.to_csv(preds_path, index=False)
    print(f"   {len(all_preds)} rows → {preds_path}")

    # Metrics
    print("\n4. Metrics:")
    quality = []
    for method, preds in results.items():
        m = calc_metrics(preds)
        m["method"] = method
        quality.append(m)
        print(f"   {method:10s}  MAE={m['mae']:.4f}  WAPE={m['wape']:.4f}  "
              f"R²={m['r2']:.4f}  Dir={m['direction_hit_rate']:.4f}")

    quality_df = pd.DataFrame(quality)
    quality_path = OUTPUT_DIR / "dl_benchmark_quality.csv"
    quality_df.to_csv(quality_path, index=False)
    print(f"\n   quality → {quality_path}")

    # Metadata
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    meta = {
        "run_id": run_id,
        "version": "dl-benchmark-v1",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "backtest_windows": len(origins),
        "prediction_horizon": PREDICTION_LOOKAHEAD,
        "models": list(results.keys()),
        "cpu_only": True,
    }
    meta_path = OUTPUT_DIR / "dl_benchmark_run.json"
    meta_path.write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")
    print(f"   metadata → {meta_path}")
    return all_preds, quality_df, meta


if __name__ == "__main__":
    main()