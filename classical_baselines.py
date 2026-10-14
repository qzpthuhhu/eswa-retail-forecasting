#!/usr/bin/env python3
"""Croston, SBA, and TSB point forecasts for the manuscript's rolling-origin protocol.

Pass ``--series data/daily_sku_sales.csv`` (columns ``sku``, ``date``, ``y``;
181 daily rows per SKU). Without ``--series`` the script runs an internal
self-check and exits 2. It does not impute MAE, WAPE, R^2, or direction hit.

Protocol, matching Section 4 of the manuscript:

* Panel length 181 days. Five origins. The first evaluation index is 146
  (0-based), aligned to ``data/rolling_backtest_predictions.csv``
  (calendar 2023-05-27, 2023-06-03, 2023-06-10, 2023-06-17, 2023-06-24).
  Origins then step by 7 days: 146, 153, 160, 167, 174.
  Training rows are ``[origin - 120, origin)``. Evaluation rows are
  ``[origin, origin + 7)``.
* The older script default 120, 127, 134, 141, 148 is not this file.
  Those origins produce different errors and are not the manuscript table.
* Each SKU is fit alone. The one-step rate is repeated for horizons 1..7.
* Croston and SBA use alpha = 0.1. TSB uses alpha = 0.1 and beta = 0.1.
* If a training window is all zeros, the rate is 0.
* Metrics pool every SKU, origin, and horizon with equal weight per case.
  WAPE and R^2 are null when the denominator is zero. They are never filled
  with a placeholder number.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from dataclasses import dataclass


N_DAYS = 181
TRAIN = 120
HORIZON = 7
N_WINDOWS = 5
STEP = 7
# 0-based index of 2023-05-27 when 2023-01-01 is index 0.
# Matches forecast_origin in data/rolling_backtest_predictions.csv.
FIRST_ORIGIN = 146
ALPHA = 0.1
BETA = 0.1


def rolling_origins(
    n_days: int = N_DAYS,
    train: int = TRAIN,
    horizon: int = HORIZON,
    n_windows: int = N_WINDOWS,
    step: int = STEP,
    first_origin: int = FIRST_ORIGIN,
) -> list[int]:
    origins = [first_origin + i * step for i in range(n_windows)]
    if origins[0] < train:
        raise ValueError(
            f"first origin {origins[0]} is shorter than the {train}-day training window"
        )
    last = origins[-1] + horizon
    if last > n_days:
        raise ValueError(f"origin window ends at {last}, past series length {n_days}")
    return origins


def _first_positive(y: list[float]) -> int | None:
    for i, value in enumerate(y):
        if value > 0:
            return i
    return None


def croston_rate(y: list[float], alpha: float = ALPHA, sba: bool = False) -> float:
    """Croston ratio z/p, or the SBA factor (1 - alpha/2) * z/p.

    Initialization follows the usual 1-based indexing of the first positive
    demand: z is that demand, p is its 1-based index. Later positive demands
    update z and p with the gap in index units (1 for consecutive days).
    """
    if alpha <= 0 or alpha >= 1:
        raise ValueError("alpha must lie in (0, 1)")
    first = _first_positive(y)
    if first is None:
        return 0.0
    z = float(y[first])
    p = float(first + 1)
    previous = first
    for t in range(first + 1, len(y)):
        if y[t] <= 0:
            continue
        gap = float(t - previous)
        z = z + alpha * (float(y[t]) - z)
        p = p + alpha * (gap - p)
        previous = t
    if p <= 0:
        raise ValueError("inter-demand interval collapsed to a non-positive value")
    rate = z / p
    if sba:
        rate = (1.0 - alpha / 2.0) * rate
    return rate


def tsb_rate(y: list[float], alpha: float = ALPHA, beta: float = BETA) -> float:
    """Teunter--Syntetos--Babai rate pi * z.

    At the first positive demand, z is that demand and pi is 1 over its
    1-based index. Every later day updates pi. z updates only on positive demand.
    """
    if alpha <= 0 or alpha >= 1 or beta <= 0 or beta >= 1:
        raise ValueError("alpha and beta must lie in (0, 1)")
    first = _first_positive(y)
    if first is None:
        return 0.0
    z = float(y[first])
    pi = 1.0 / float(first + 1)
    for t in range(first + 1, len(y)):
        occurred = 1.0 if y[t] > 0 else 0.0
        pi = pi + beta * (occurred - pi)
        if y[t] > 0:
            z = z + alpha * (float(y[t]) - z)
    return pi * z


def rate_for(method: str, y: list[float], alpha: float = ALPHA, beta: float = BETA) -> float:
    key = method.lower()
    if key == "croston":
        return croston_rate(y, alpha=alpha, sba=False)
    if key == "sba":
        return croston_rate(y, alpha=alpha, sba=True)
    if key == "tsb":
        return tsb_rate(y, alpha=alpha, beta=beta)
    raise ValueError(f"unknown method {method}")


@dataclass
class MetricResult:
    method: str
    n: int
    mae: float | None
    wape: float | None
    r2: float | None
    direction_hit: float | None

    def as_dict(self) -> dict:
        return {
            "method": self.method,
            "n": self.n,
            "mae": self.mae,
            "wape": self.wape,
            "r2": self.r2,
            "direction_hit": self.direction_hit,
            "status": "estimated",
        }


def _metrics(actual: list[float], predicted: list[float], origin_level: list[float]) -> tuple:
    n = len(actual)
    if n == 0 or n != len(predicted) or n != len(origin_level):
        raise ValueError("metric inputs must be non-empty and aligned")
    abs_err = [abs(a - p) for a, p in zip(actual, predicted)]
    mae = sum(abs_err) / n
    denom = sum(abs(a) for a in actual)
    wape = (sum(abs_err) / denom) if denom > 0 else None
    mean_y = sum(actual) / n
    ss_tot = sum((a - mean_y) ** 2 for a in actual)
    ss_res = sum((a - p) ** 2 for a, p in zip(actual, predicted))
    r2 = (1.0 - ss_res / ss_tot) if ss_tot > 0 else None
    hits = 0
    for a, p, base in zip(actual, predicted, origin_level):
        if (p - base > 0) == (a - base > 0) and (p - base < 0) == (a - base < 0):
            hits += 1
    # Equality of signs, with sign(0) == 0. The comparison above is true when
    # both differences are positive, both negative, or both zero.
    return mae, wape, r2, hits / n


def evaluate_panel(
    series: dict[str, list[float]],
    method: str,
    alpha: float = ALPHA,
    beta: float = BETA,
) -> MetricResult:
    origins = rolling_origins()
    actual: list[float] = []
    predicted: list[float] = []
    origin_level: list[float] = []
    for sku in sorted(series):
        y = series[sku]
        if len(y) != N_DAYS:
            raise ValueError(f"{sku} has {len(y)} days; expected {N_DAYS}")
        if any(value < 0 for value in y):
            raise ValueError(f"{sku} has a negative demand")
        for origin in origins:
            train = y[origin - TRAIN : origin]
            rate = rate_for(method, train, alpha=alpha, beta=beta)
            base = y[origin - 1]
            for h in range(HORIZON):
                actual.append(y[origin + h])
                predicted.append(rate)
                origin_level.append(base)
    mae, wape, r2, hit = _metrics(actual, predicted, origin_level)
    return MetricResult(method=method, n=len(actual), mae=mae, wape=wape, r2=r2, direction_hit=hit)


def load_series(path: str) -> dict[str, list[float]]:
    grouped: dict[str, list[tuple[str, float]]] = defaultdict(list)
    with open(path, newline="") as handle:
        reader = csv.DictReader(handle)
        fields = set(reader.fieldnames or [])
        required = {"sku", "date", "y"}
        if not required.issubset(fields):
            raise ValueError(f"{path} must have columns sku, date, y; found {sorted(fields)}")
        for row in reader:
            grouped[row["sku"]].append((row["date"], float(row["y"])))
    if not grouped:
        raise ValueError(f"{path} has no rows")
    series: dict[str, list[float]] = {}
    for sku, rows in grouped.items():
        rows.sort(key=lambda item: item[0])
        series[sku] = [value for _, value in rows]
    return series


def self_check() -> None:
    origins = rolling_origins()
    if origins != [146, 153, 160, 167, 174]:
        raise AssertionError(origins)

    constant = [4.0] * 30
    if croston_rate(constant) != 4.0:
        raise AssertionError(croston_rate(constant))
    if abs(croston_rate(constant, sba=True) - 3.8) > 1e-12:
        raise AssertionError(croston_rate(constant, sba=True))
    if tsb_rate(constant) != 4.0:
        raise AssertionError(tsb_rate(constant))

    if croston_rate([0.0, 0.0, 0.0]) != 0.0 or tsb_rate([0.0, 0.0]) != 0.0:
        raise AssertionError("all-zero series must forecast 0")

    sparse = [0.0, 0.0, 10.0, 0.0, 0.0, 0.0, 20.0]
    # z, p after the second demand: z = 11, p = 3.1
    expected_croston = 11.0 / 3.1
    expected_sba = (1.0 - 0.05) * expected_croston
    if abs(croston_rate(sparse) - expected_croston) > 1e-12:
        raise AssertionError(croston_rate(sparse))
    if abs(croston_rate(sparse, sba=True) - expected_sba) > 1e-12:
        raise AssertionError(croston_rate(sparse, sba=True))

    # TSB hand roll from the first positive demand at index 2.
    z = 10.0
    pi = 1.0 / 3.0
    for occurred, demand in ((0, 0.0), (0, 0.0), (0, 0.0), (1, 20.0)):
        pi = pi + 0.1 * (occurred - pi)
        if demand > 0:
            z = z + 0.1 * (demand - z)
    if abs(tsb_rate(sparse) - (pi * z)) > 1e-12:
        raise AssertionError((tsb_rate(sparse), pi * z))

    zeros = {f"s{i}": [0.0] * N_DAYS for i in range(3)}
    result = evaluate_panel(zeros, "croston")
    if result.mae != 0.0 or result.wape is not None or result.r2 is not None:
        raise AssertionError(result)
    if result.n != 3 * N_WINDOWS * HORIZON:
        raise AssertionError(result.n)
    # Flat zero forecasts on a zero panel: every sign pair is (0, 0), a hit.
    if result.direction_hit != 1.0:
        raise AssertionError(result.direction_hit)

    try:
        rolling_origins(n_days=130)
    except ValueError:
        pass
    else:
        raise AssertionError("short panels must be rejected")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--series", help="CSV with columns sku, date, y")
    parser.add_argument("--self-check", action="store_true", help="Run checks and exit 0")
    parser.add_argument("--alpha", type=float, default=ALPHA)
    parser.add_argument("--beta", type=float, default=BETA)
    args = parser.parse_args(argv)
    self_check()
    if args.self_check or not args.series:
        print("self-check ok", file=sys.stderr)
        if args.self_check and not args.series:
            return 0
        if not args.series:
            print(
                "TODO: Croston, SBA, and TSB were not estimated. "
                "Pass --series data/daily_sku_sales.csv (columns sku, date, y; "
                "181 daily rows per SKU). No metric file was written.",
                file=sys.stderr,
            )
            return 2
    series = load_series(args.series)
    payload = {
        "protocol": {
            "n_days": N_DAYS,
            "train": TRAIN,
            "horizon": HORIZON,
            "origins_0based": rolling_origins(),
            "alpha": args.alpha,
            "beta": args.beta,
            "note": (
                "Pooled over SKUs, origins, and horizons with equal weight per case. "
                "Origins 146, 153, 160, 167, 174 match forecast_origin in "
                "data/rolling_backtest_predictions.csv. Origins 120, 127, 134, 141, 148 "
                "are not this evaluation."
            ),
        },
        "results": [
            evaluate_panel(series, method, alpha=args.alpha, beta=args.beta).as_dict()
            for method in ("croston", "sba", "tsb")
        ],
    }
    json.dump(payload, sys.stdout, indent=2)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
