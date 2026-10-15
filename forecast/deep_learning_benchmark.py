"""Deep learning benchmark: DeepAR, N-BEATS, TFT vs production LightGBM.

Each model is evaluated on the same 5-window rolling backtest with identical
train/test splits as the production pipeline.  All models run on CPU and
respect the strict temporal order: training data ends before the forecast
origin, and predictions cover T+1 through T+7.

Results are saved as CSV in reports/model_quality/deep_learning_benchmark/
with metrics comparable to the existing rolling_backtest_quality.csv.
"""

from __future__ import annotations

import json
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ── LightGBM baseline imports ──────────────────────────────────────────
from backtest_pipeline import (
    BACKTEST_WINDOW_COUNT,
    ENTITY_KEYS,
    QUANTILE_LEVELS,
    REPORT_DIR,
    calculate_metrics,
    select_backtest_origins,
)
from first_round_model_experiments import (
    _apply_store_calibration,
    _fit_lightgbm,
    _prediction_rows,
    build_direct_feature_frames,
    get_direct_feature_cols,
    split_direct_horizon_frame,
)
from fourth_round_sparse_features import (
    _attach_sparsity_features,
    _compute_sku_sparsity,
)
from predict_pipeline import (
    PREDICTION_LOOKAHEAD,
    STORE_NAME_MAP,
    load_and_preprocess,
)

# ── GluonTS imports ────────────────────────────────────────────────────
try:
    from gluonts.torch import DeepAREstimator, NBEATSEstimator
    from gluonts.torch.model.tft import TemporalFusionTransformerEstimator
    from gluonts.dataset.pandas import PandasDataset
    from gluonts.dataset.split import split
    HAS_GLUONTS = True
except ImportError:
    HAS_GLUONTS = False
    print("⚠️  GluonTS not available; install with `pip install gluonts`")


# ── Configuration ──────────────────────────────────────────────────────
OUTPUT_DIR = REPORT_DIR / "deep_learning_benchmark"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
EXPERIMENT_VERSION = "dl-benchmark-v1"
TWEEDIE_PARAMS = {"objective": "tweedie", "tweedie_variance_power": 1.3}

# GluonTS common config — conservative CPU settings
GLUONTS_TRAINER_KWARGS = {
    "max_epochs": 40,          # enough for 240 short series
    "learning_rate": 1e-3,
    "batch_size": 32,
    "patience": 8,
    "num_batches_per_epoch": 30,
}

DEEP_MODELS = [
    {
        "method": "deepar",
        "enabled": True,
        "description": "DeepAR (probabilistic RNN)",
    },
    {
        "method": "nbeats",
        "enabled": True,
        "description": "N-BEATS (basis expansion)",
    },
    {
        "method": "tft",
        "enabled": True,
        "description": "Temporal Fusion Transformer",
    },
]


# ── Data preparation for GluonTS ───────────────────────────────────────
def _build_gluonts_dataset(df: pd.DataFrame, origin: pd.Timestamp) -> pd.DataFrame:
    """Extract training series up to the internal validation cutoff.

    Mirrors the production pipeline: training data must end before
    (origin - 2 * horizon), so the model never sees any future labels.
    """
    cutoff = origin - pd.Timedelta(days=2 * PREDICTION_LOOKAHEAD)
    train = df[df["date"] < cutoff].copy()

    # Pivot to wide format: one row per (store_id, product_id)
    # Each row is a time series
    series_list = []
    for (store, prod), grp in train.groupby(["store_id", "product_id"]):
        grp = grp.sort_values("date")
        series_list.append({
            "item_id": f"{store}_{prod}",
            "store_id": store,
            "product_id": prod,
            "start": grp["date"].min(),
            "target": grp["qty"].tolist(),
        })
    return pd.DataFrame(series_list)


def _predict_gluonts(
    estimator,
    dataset: pd.DataFrame,
    prediction_length: int = PREDICTION_LOOKAHEAD,
) -> pd.DataFrame:
    """Train and forecast for every time series in the dataset."""
    gluon_ds = PandasDataset.from_long_dataset(
        dataset.rename(columns={"start": "timestamp"}),
        item_id="item_id",
        target="target",
    )
    predictor = estimator.train(gluon_ds)
    forecasts_it = predictor.predict(gluon_ds)

    rows = []
    for fc in forecasts_it:
        item = fc.item_id
        store_id, product_id = item.split("_")
        median = np.asarray(fc.quantile(0.5), dtype=float)
        for h in range(prediction_length):
            rows.append({
                "item_id": item,
                "store_id": int(store_id),
                "product_id": int(product_id),
                "horizon_day": h + 1,
                "prediction_raw": float(median[h]),
            })
    return pd.DataFrame(rows)


# ── GluonTS model builders ─────────────────────────────────────────────
def _build_deepar(freq: str = "D", prediction_length: int = PREDICTION_LOOKAHEAD):
    return DeepAREstimator(
        freq=freq,
        prediction_length=prediction_length,
        trainer_kwargs=GLUONTS_TRAINER_KWARGS,
    )


def _build_nbeats(freq: str = "D", prediction_length: int = PREDICTION_LOOKAHEAD):
    return NBEATSEstimator(
        freq=freq,
        prediction_length=prediction_length,
        trainer_kwargs=GLUONTS_TRAINER_KWARGS,
        context_length=max(PREDICTION_LOOKAHEAD * 4, 28),
        widths=[128, 128, 128],
        num_blocks=[2, 2, 2],
        num_stacks=3,
    )


def _build_tft(freq: str = "D", prediction_length: int = PREDICTION_LOOKAHEAD):
    return TemporalFusionTransformerEstimator(
        freq=freq,
        prediction_length=prediction_length,
        trainer_kwargs=GLUONTS_TRAINER_KWARGS,
        context_length=max(PREDICTION_LOOKAHEAD * 4, 28),
        hidden_dim=32,
        num_heads=2,
        dropout_rate=0.1,
    )


DL_BUILDERS = {
    "deepar": _build_deepar,
    "nbeats": _build_nbeats,
    "tft": _build_tft,
}


# ── LightGBM baseline (same as production, per origin) ─────────────────
def _train_lgb_prediction_for_origin(
    df, frame_dict, feature_cols, origin
) -> pd.DataFrame:
    """Run production LightGBM pipeline for one forecast origin."""
    rows = []
    for horizon_day in range(1, PREDICTION_LOOKAHEAD + 1):
        frame = frame_dict[horizon_day]
        fit, val, cal, test = split_direct_horizon_frame(
            frame, feature_cols, origin, horizon_day
        )
        model = _fit_lightgbm(fit, val, feature_cols, TWEEDIE_PARAMS)
        test_pred = model.predict(test[feature_cols])
        rows.append(
            _prediction_rows(
                test, test_pred, "lightgbm_tweedie", origin, horizon_day,
            )
        )

        # calibration
        cal_pred = model.predict(cal[feature_cols])
        cal_rows_df = _prediction_rows(
            cal, cal_pred, "lightgbm_tweedie", origin, horizon_day,
        )
        cal_rows_df, _ = _apply_store_calibration(cal_rows_df, rows[-1])
        # We only need the calibration info for store bias; apply to test
        cal_for_factor = cal_rows_df.rename(
            columns={"prediction": "pred_lightgbm"}
        )
        from backtest_pipeline import calculate_store_bias_factors
        factors = calculate_store_bias_factors(cal_for_factor)
        factor_map = factors.set_index("store_id")["store_bias_factor"]
        rows[-1]["store_bias_factor"] = (
            rows[-1]["store_id"].map(factor_map).fillna(1.0)
        )
        rows[-1]["prediction_model"] = rows[-1]["prediction"]
        rows[-1]["prediction"] = (
            rows[-1]["prediction_model"] * rows[-1]["store_bias_factor"]
        )

    return pd.concat(rows, ignore_index=True)


# ── Main benchmark ─────────────────────────────────────────────────────
def run_deep_learning_benchmark():
    print("=" * 64)
    print("Deep Learning Benchmark: DeepAR, N-BEATS, TFT vs LightGBM")
    print("=" * 64)

    # 1.  Load and prepare data
    print("\n1. Loading data...")
    df = load_and_preprocess()
    dates = df["date"].unique()
    origins = select_backtest_origins(dates)
    first_origin = min(origins)
    cutoff = first_origin - pd.Timedelta(days=1)

    # 2.  Pre-compute sparsity features (pre-first-origin)
    print("2. Computing sparsity labels...")
    sparsity = _compute_sku_sparsity(df, cutoff)

    # 3.  Build direct feature frames (shared by LightGBM)
    print("3. Building direct feature frames...")
    frame_dict = build_direct_feature_frames(df)

    # 4.  Run LightGBM baseline (per origin)
    print("\n4. Running LightGBM baseline...")
    lgb_predictions = []
    for origin in origins:
        print(f"   LightGBM origin {origin.date()}...")
        # Attach sparsity to each frame
        for h in range(1, PREDICTION_LOOKAHEAD + 1):
            frame_dict[h] = _attach_sparsity_features(frame_dict[h], sparsity)
        feature_cols = get_direct_feature_cols(frame_dict[1])
        preds = _train_lgb_prediction_for_origin(
            df, frame_dict, feature_cols, origin,
        )
        lgb_predictions.append(preds)
    lgb_all = pd.concat(lgb_predictions, ignore_index=True)
    lgb_all["method"] = "lightgbm_tweedie"
    print(f"   LightGBM: {len(lgb_all):,} prediction rows")

    # 5.  Run deep learning models (per origin)
    all_predictions = [lgb_all]
    if not HAS_GLUONTS:
        print("\n⚠️  GluonTS not installed; skipping deep learning models.")
    else:
        print("\n5. Running deep learning models (CPU)...")
        for model_cfg in DEEP_MODELS:
            if not model_cfg["enabled"]:
                continue
            method = model_cfg["method"]
            print(f"\n   === {model_cfg['description']} ===")
            dl_rows = []
            for idx, origin in enumerate(origins):
                print(f"   [{method}] origin {origin.date()} ({idx+1}/{len(origins)})...")
                try:
                    ts_df = _build_gluonts_dataset(df, origin)
                    builder = DL_BUILDERS[method]
                    estimator = builder()
                    raw_preds = _predict_gluonts(estimator, ts_df)
                    # Map back to full test dates
                    for h in range(1, PREDICTION_LOOKAHEAD + 1):
                        test_date = origin + pd.Timedelta(days=h - 1)
                        hp = raw_preds[raw_preds["horizon_day"] == h].copy()
                        hp["date"] = pd.Timestamp(test_date)
                        hp["forecast_origin"] = origin
                        hp["method"] = method
                        hp["qty"] = 0.0  # placeholder; filled from ground truth
                        hp["prediction"] = np.clip(hp["prediction_raw"], 0, None)
                        hp["store_name"] = hp["store_id"].map(STORE_NAME_MAP)
                        hp["category"] = ""
                        hp["product_type"] = ""
                        hp["prediction_raw"] = hp["prediction_raw"].astype(float)
                        hp["prediction_model"] = hp["prediction"].astype(float)
                        hp["store_bias_factor"] = 1.0
                        dl_rows.append(hp)
                except Exception as e:
                    print(f"   [{method}] ERROR for origin {origin.date()}: {e}")
                    import traceback
                    traceback.print_exc()
                    continue

            if dl_rows:
                dl_all = pd.concat(dl_rows, ignore_index=True)
                dl_all["method"] = method
                # Join ground truth qty
                gt = df[["date", "store_id", "product_id", "qty", "category", "product_type"]].copy()
                dl_all = dl_all.drop(columns=["qty", "category", "product_type"]).merge(
                    gt, on=["date", "store_id", "product_id"], how="left"
                )
                dl_all["qty"] = dl_all["qty"].fillna(0)
                all_predictions.append(dl_all)
                print(f"   [{method}] {len(dl_all):,} prediction rows")
            else:
                print(f"   [{method}] NO predictions generated")

    # 6.  Combine and save predictions
    print("\n6. Saving predictions...")
    predictions = pd.concat(all_predictions, ignore_index=True)
    predictions_path = OUTPUT_DIR / "dl_benchmark_predictions.csv.gz"
    predictions.to_csv(predictions_path, index=False, compression="gzip")
    print(f"   {len(predictions):,} rows → {predictions_path}")

    # 7.  Calculate metrics per method
    print("\n7. Calculating metrics...")
    quality_rows = []
    segment_rows = []

    for method, grp in predictions.groupby("method"):
        method_preds = grp.copy()
        if "pred_lightgbm" not in method_preds.columns:
            method_preds["pred_lightgbm"] = method_preds["prediction"]

        # Overall metrics
        metrics = calculate_metrics(method_preds, prediction_col="prediction")
        quality_rows.append({
            "method": method,
            "level": "overall",
            "count": len(method_preds),
            **{k: v for k, v in metrics.items() if k not in ("level", "count")},
        })

        # Window-level
        for origin, win_grp in method_preds.groupby("forecast_origin"):
            wm = calculate_metrics(win_grp, prediction_col="prediction")
            quality_rows.append({
                "method": method,
                "level": "window",
                "count": len(win_grp),
                "window_origin": str(origin.date()),
                **{k: v for k, v in wm.items() if k not in ("level", "count")},
            })

        # Sparse / dense breakdown
        sparsity_df = sparsity.copy()
        seg = method_preds.merge(
            sparsity_df, on=["store_id", "product_id"], how="left"
        )
        seg["segment"] = seg["is_sparse"].map({True: "sparse", False: "dense"})
        for seg_label, seg_grp in seg.groupby("segment"):
            sm = calculate_metrics(seg_grp, prediction_col="prediction")
            segment_rows.append({
                "method": method,
                "segment": seg_label,
                "count": len(seg_grp),
                **{k: v for k, v in sm.items() if k not in ("level", "count")},
            })

    quality = pd.DataFrame(quality_rows)
    segment = pd.DataFrame(segment_rows)

    quality_path = OUTPUT_DIR / "dl_benchmark_quality.csv"
    segment_path = OUTPUT_DIR / "dl_benchmark_segments.csv"
    quality.to_csv(quality_path, index=False)
    segment.to_csv(segment_path, index=False)
    print(f"   quality → {quality_path}")
    print(f"   segments → {segment_path}")

    # 8.  Summary table
    print("\n" + "=" * 64)
    print("BENCHMARK SUMMARY")
    print("=" * 64)
    overall = quality[quality["level"] == "overall"].set_index("method")
    cols = ["mae", "wape", "r2", "direction_hit_rate", "volatility_ratio"]
    available = [c for c in cols if c in overall.columns]
    print(overall[available].to_string(float_format=lambda x: f"{x:.4f}"))

    # 9.  Metadata
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    metadata = {
        "run_id": run_id,
        "version": EXPERIMENT_VERSION,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "backtest_windows": len(origins),
        "origins": [str(o.date()) for o in origins],
        "prediction_horizon": PREDICTION_LOOKAHEAD,
        "models": [m["method"] for m in DEEP_MODELS] + ["lightgbm_tweedie"],
        "cpu_only": True,
    }
    meta_path = OUTPUT_DIR / "dl_benchmark_run.json"
    meta_path.write_text(json.dumps(metadata, indent=2, default=str), encoding="utf-8")
    print(f"\n   metadata → {meta_path}")

    return predictions, quality, segment, metadata


if __name__ == "__main__":
    run_deep_learning_benchmark()