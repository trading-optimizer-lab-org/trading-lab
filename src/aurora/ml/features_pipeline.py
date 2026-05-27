"""Reusable feature engineering pipeline for DL/ML strategies.

Used by ml/lstm.py, ml/transformer.py and ml/rl_agent.py to produce a
standardized feature matrix from a price (and optionally OHLCV) DataFrame.

Anti-lookahead by construction: every feature at bar ``t`` only uses data
through ``t``. Standardization uses rolling z-score (past N bars only) by
default to avoid fit-time leakage when the same pipeline is applied to
out-of-sample data.

Public API:
- FeaturePipelineConfig: dataclass of feature toggles and parameters.
- FeaturePipeline:       fit/transform/fit_transform/feature_names.

Standard feature blocks:
  1. Rolling stats per window: mean return, std, skew, kurtosis, min, max.
  2. Return lags per lag.
  3. Technicals: RSI(14), MACD(12,26,9) line+signal+hist, BBands(20,2)
     upper/middle/lower/width.
  4. Volatility: EWMA volatility, realized vol from high-low if H/L given.
  5. Microstructure: Corwin-Schultz spread, signed volume (requires H/L
     and volume columns).
  6. Standardization: rolling z-score / expanding z-score / none.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class FeaturePipelineConfig:
    """Configuration for FeaturePipeline.

    Attributes:
        rolling_windows:     windows for rolling stats (mean/std/skew/kurt/min/max).
        return_lags:         lags for pct_change features.
        include_technicals:  include RSI, MACD, BBands.
        include_microstructure: include Corwin-Schultz spread, signed volume.
            Requires 'high', 'low', 'volume' columns; gracefully skipped (with
            warning) when those columns are missing.
        include_volatility:  include EWMA vol and high-low realized vol.
        standardize:         whether to apply standardization at all.
        standardize_method:  'rolling_zscore' | 'expanding_zscore' | 'none'.
        standardize_window:  window N for rolling z-score (only past N bars).
        price_col:           name of close price column (default 'close').
    """

    rolling_windows: tuple = (5, 10, 20, 60)
    return_lags: tuple = (1, 2, 3, 5, 10)
    include_technicals: bool = True
    include_microstructure: bool = False
    include_volatility: bool = True
    standardize: bool = True
    standardize_method: str = "rolling_zscore"
    standardize_window: int = 252
    price_col: str = "close"


# ---------------------------------------------------------------------------
# Internal feature computations (anti-lookahead)
# ---------------------------------------------------------------------------

def _rolling_stats(returns: pd.Series, window: int) -> dict[str, pd.Series]:
    """mean / std / skew / kurt / min / max over a rolling window of returns.

    Min/max of *price* often dominate scale; we compute them on returns to keep
    the feature matrix homogeneous in units (return-space).
    """
    r = returns.rolling(window=window, min_periods=window)
    return {
        f"roll_mean_{window}": r.mean(),
        f"roll_std_{window}": r.std(ddof=1),
        f"roll_skew_{window}": r.skew(),
        f"roll_kurt_{window}": r.kurt(),
        f"roll_min_{window}": r.min(),
        f"roll_max_{window}": r.max(),
    }


def _return_lag(returns: pd.Series, lag: int) -> pd.Series:
    """Return lagged by ``lag`` bars (no lookahead since shift is positive)."""
    return returns.shift(lag)


def _rsi(prices: pd.Series, period: int = 14) -> pd.Series:
    """Wilder RSI. Mirrors core.engine_jit.compute_rsi_jit semantics."""
    p = prices.astype(float).to_numpy()
    n = int(period)
    out = np.full(len(p), np.nan)
    if len(p) < n + 1:
        return pd.Series(out, index=prices.index)
    diff = np.diff(p)
    gain = np.where(diff > 0, diff, 0.0)
    loss = np.where(diff < 0, -diff, 0.0)
    ag = gain[:n].mean()
    al = loss[:n].mean()
    for i in range(n, len(p)):
        if i > n:
            ag = (ag * (n - 1) + gain[i - 1]) / n
            al = (al * (n - 1) + loss[i - 1]) / n
        out[i] = 100.0 if al == 0 else 100.0 - 100.0 / (1.0 + ag / al)
    return pd.Series(out, index=prices.index)


def _ema(s: pd.Series, span: int) -> pd.Series:
    """Pandas EWMA with adjust=False (matches typical MACD implementations)."""
    return s.ewm(span=span, adjust=False, min_periods=span).mean()


def _macd(
    prices: pd.Series,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> dict[str, pd.Series]:
    ema_fast = _ema(prices, fast)
    ema_slow = _ema(prices, slow)
    line = ema_fast - ema_slow
    sig = _ema(line, signal)
    hist = line - sig
    return {"macd_line": line, "macd_signal": sig, "macd_hist": hist}


def _bbands(
    prices: pd.Series,
    window: int = 20,
    n_std: float = 2.0,
) -> dict[str, pd.Series]:
    mid = prices.rolling(window=window, min_periods=window).mean()
    std = prices.rolling(window=window, min_periods=window).std(ddof=1)
    upper = mid + n_std * std
    lower = mid - n_std * std
    # Guard against ``mid == 0`` which would explode bb_width to inf and
    # propagate non-finite values through downstream standardization. We map
    # zero rolling means to NaN bb_width at those bars.
    safe_mid = mid.replace(0.0, np.nan)
    width = (upper - lower) / safe_mid
    return {
        "bb_upper": upper,
        "bb_middle": mid,
        "bb_lower": lower,
        "bb_width": width,
    }


def _ewma_vol(returns: pd.Series, span: int = 20) -> pd.Series:
    """GARCH-like EWMA volatility on returns."""
    return returns.ewm(span=span, adjust=False, min_periods=span).std()


def _hl_realized_vol(
    high: pd.Series,
    low: pd.Series,
    window: int = 20,
) -> pd.Series:
    """Parkinson-style realized vol from high/low.

    Per-bar variance = ln(H/L)^2 / (4 ln 2). Rolling mean -> annualized later
    by the caller if desired. We return the un-annualized rolling mean to keep
    units consistent with std-based features.

    Non-positive bars (high <= 0 or low <= 0) cannot produce a valid log ratio.
    We require all-positive high/low up front rather than letting numpy emit a
    RuntimeWarning and return NaN/-inf, which would silently leak through the
    standardization step.
    """
    h = high.astype(float)
    l = low.astype(float)
    if not bool((h > 0).all()) or not bool((l > 0).all()):
        raise ValueError(
            "_hl_realized_vol requires strictly positive high and low; "
            "non-positive values would make np.log(H/L) ill-defined."
        )
    rng = np.log(h / l) ** 2 / (4.0 * np.log(2.0))
    return rng.rolling(window=window, min_periods=window).mean().pow(0.5)


# ---------------------------------------------------------------------------
# Standardization
# ---------------------------------------------------------------------------

def _rolling_zscore(df: pd.DataFrame, window: int) -> pd.DataFrame:
    """Rolling z-score using only past (and current) N bars per column.

    z_t = (x_t - mean(x[t-N+1..t])) / std(x[t-N+1..t]).
    """
    mean = df.rolling(window=window, min_periods=window).mean()
    std = df.rolling(window=window, min_periods=window).std(ddof=1)
    out = (df - mean) / std.replace(0.0, np.nan)
    return out


def _expanding_zscore(df: pd.DataFrame, min_periods: int = 30) -> pd.DataFrame:
    """Expanding z-score: uses all past observations including current bar."""
    mean = df.expanding(min_periods=min_periods).mean()
    std = df.expanding(min_periods=min_periods).std(ddof=1)
    return (df - mean) / std.replace(0.0, np.nan)


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

class FeaturePipeline:
    """Reusable feature pipeline with fit / transform / fit_transform.

    The pipeline is mostly stateless (every feature is computed bar-by-bar from
    the data given to ``transform``). ``fit`` is provided for sklearn-style
    interoperability and to record the feature column order seen at fit time.
    """

    def __init__(self, config: FeaturePipelineConfig):
        self.config = config
        self._fitted: bool = False
        self._feature_names: list[str] = []

    # ---- public API ----------------------------------------------------

    def fit(self, prices: pd.DataFrame) -> "FeaturePipeline":
        """Build the feature column list from a sample of input data.

        Does not learn any statistics from ``prices`` (rolling z-score uses
        only past N bars within transform), but does freeze the feature column
        order so transform() on new data produces identical columns.
        """
        feats = self._compute_raw(prices)
        self._feature_names = list(feats.columns)
        self._fitted = True
        return self

    def transform(self, prices: pd.DataFrame) -> pd.DataFrame:
        """Compute features for ``prices``. Anti-lookahead by construction."""
        feats = self._compute_raw(prices)
        if self._fitted:
            # Reorder / restrict to the columns seen at fit time. Add any
            # missing columns as NaN so downstream shape stays predictable.
            for col in self._feature_names:
                if col not in feats.columns:
                    feats[col] = np.nan
            feats = feats[self._feature_names]
        if self.config.standardize and self.config.standardize_method != "none":
            feats = self._standardize(feats)
        return feats

    def fit_transform(self, prices: pd.DataFrame) -> pd.DataFrame:
        """fit() then transform() on the same data."""
        return self.fit(prices).transform(prices)

    def feature_names(self) -> list[str]:
        """Names of the feature columns (post-fit)."""
        return list(self._feature_names)

    # ---- internals -----------------------------------------------------

    def _close_series(self, prices: pd.DataFrame) -> pd.Series:
        col = self.config.price_col
        if col not in prices.columns:
            raise ValueError(
                f"FeaturePipeline: price_col '{col}' not in input columns "
                f"{list(prices.columns)}"
            )
        return prices[col].astype(float)

    def _compute_raw(self, prices: pd.DataFrame) -> pd.DataFrame:
        """Compute the raw (pre-standardization) feature matrix."""
        if not isinstance(prices, pd.DataFrame):
            raise TypeError("FeaturePipeline expects a pd.DataFrame input")

        cfg = self.config
        close = self._close_series(prices)
        rets = close.pct_change()

        out: dict[str, pd.Series] = {}

        # Rolling stats on returns
        for w in cfg.rolling_windows:
            out.update(_rolling_stats(rets, int(w)))

        # Return lags
        for lag in cfg.return_lags:
            out[f"ret_lag_{lag}"] = _return_lag(rets, int(lag))

        # Technicals
        if cfg.include_technicals:
            out["rsi_14"] = _rsi(close, period=14)
            out.update(_macd(close, fast=12, slow=26, signal=9))
            out.update(_bbands(close, window=20, n_std=2.0))

        # Volatility
        if cfg.include_volatility:
            out["ewma_vol_20"] = _ewma_vol(rets, span=20)
            if "high" in prices.columns and "low" in prices.columns:
                out["hl_realized_vol_20"] = _hl_realized_vol(
                    prices["high"].astype(float),
                    prices["low"].astype(float),
                    window=20,
                )

        # Microstructure
        if cfg.include_microstructure:
            self._add_microstructure(prices, close, out)

        df = pd.DataFrame(out, index=prices.index)
        return df

    def _add_microstructure(
        self,
        prices: pd.DataFrame,
        close: pd.Series,
        out: dict[str, pd.Series],
    ) -> None:
        """Append microstructure features in-place to ``out`` if columns OK."""
        # Lazy import so missing optional deps in microstructure don't break
        # the rest of the pipeline.
        try:
            from aurora.ml.microstructure import (
                corwin_schultz_spread,
                signed_volume,
            )
        except ImportError as e:  # pragma: no cover
            warnings.warn(
                f"FeaturePipeline: microstructure module not importable ({e}); "
                "skipping microstructure features.",
                stacklevel=3,
            )
            return

        missing = [c for c in ("high", "low", "volume") if c not in prices.columns]
        if missing:
            warnings.warn(
                "FeaturePipeline: include_microstructure=True but input is "
                f"missing column(s) {missing}; skipping microstructure features.",
                stacklevel=3,
            )
            return

        out["cs_spread"] = corwin_schultz_spread(
            prices["high"].astype(float),
            prices["low"].astype(float),
            window=2,
        )
        out["signed_volume"] = signed_volume(
            close,
            prices["volume"].astype(float),
            method="tick_rule",
        )

    def _standardize(self, df: pd.DataFrame) -> pd.DataFrame:
        method = self.config.standardize_method
        if method == "rolling_zscore":
            return _rolling_zscore(df, int(self.config.standardize_window))
        if method == "expanding_zscore":
            return _expanding_zscore(df, min_periods=30)
        if method == "none":
            return df
        raise ValueError(
            f"FeaturePipeline: unknown standardize_method '{method}'. "
            "Expected 'rolling_zscore', 'expanding_zscore' or 'none'."
        )


__all__ = [
    "FeaturePipelineConfig",
    "FeaturePipeline",
]
