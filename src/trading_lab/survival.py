from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from math import sqrt
from typing import Any

import numpy as np
import pandas as pd

from trading_lab.backtest import run_backtest

TRAIN_END = "2013-10-18"
VALIDATION_START = "2013-10-21"
VALIDATION_END = "2019-12-31"


@dataclass(frozen=True)
class SurvivalCriteria:
    min_train_calmar: float = 1.25
    min_validation_calmar: float = 1.25
    max_train_calmar: float = 8.0
    max_train_validation_ratio: float = 2.5
    min_train_cagr: float = 0.06
    min_validation_cagr: float = 0.06
    max_train_mdd: float = 0.30
    max_validation_mdd: float = 0.30
    min_trades_per_year: float = 12.0
    max_trades_per_year: float = 90.0
    min_long_fraction: float = 0.25
    max_long_fraction: float = 0.75
    max_validation_negative_years: int = 0
    max_features_per_candidate: int = 5

    def rejection_reason(self, row: dict[str, Any]) -> str | None:
        train_calmar = float(row["train_calmar"])
        validation_calmar = float(row["validation_calmar"])
        if train_calmar < self.min_train_calmar:
            return "train_calmar"
        if validation_calmar < self.min_validation_calmar:
            return "validation_calmar"
        if train_calmar > self.max_train_calmar:
            return "train_calmar_too_high"
        if validation_calmar == 0 or train_calmar / validation_calmar > self.max_train_validation_ratio:
            return "train_validation_gap"
        if float(row["train_cagr"]) < self.min_train_cagr:
            return "train_cagr"
        if float(row["validation_cagr"]) < self.min_validation_cagr:
            return "validation_cagr"
        if abs(float(row["train_mdd"])) > self.max_train_mdd:
            return "train_mdd"
        if abs(float(row["validation_mdd"])) > self.max_validation_mdd:
            return "validation_mdd"
        trades_per_year = float(row["trades_per_year"])
        if trades_per_year < self.min_trades_per_year:
            return "too_few_trades"
        if trades_per_year > self.max_trades_per_year:
            return "too_many_trades"
        long_fraction = float(row["long_fraction"])
        if long_fraction < self.min_long_fraction:
            return "too_little_long"
        if long_fraction > self.max_long_fraction:
            return "too_much_long"
        if int(row["validation_negative_years"]) > self.max_validation_negative_years:
            return "validation_negative_years"
        if int(row["feature_count"]) > self.max_features_per_candidate:
            return "too_many_features"
        return None

    def pass_count(self, row: dict[str, Any]) -> int:
        return sum(1 for passed in self.checks(row).values() if passed)

    def checks(self, row: dict[str, Any]) -> dict[str, bool]:
        train_calmar = float(row["train_calmar"])
        validation_calmar = float(row["validation_calmar"])
        ratio_ok = validation_calmar != 0 and train_calmar / validation_calmar <= self.max_train_validation_ratio
        return {
            "train_calmar_min": train_calmar >= self.min_train_calmar,
            "validation_calmar_min": validation_calmar >= self.min_validation_calmar,
            "train_calmar_max": train_calmar <= self.max_train_calmar,
            "train_validation_ratio": ratio_ok,
            "train_cagr": float(row["train_cagr"]) >= self.min_train_cagr,
            "validation_cagr": float(row["validation_cagr"]) >= self.min_validation_cagr,
            "train_mdd": abs(float(row["train_mdd"])) <= self.max_train_mdd,
            "validation_mdd": abs(float(row["validation_mdd"])) <= self.max_validation_mdd,
            "trades_per_year_min": float(row["trades_per_year"]) >= self.min_trades_per_year,
            "trades_per_year_max": float(row["trades_per_year"]) <= self.max_trades_per_year,
            "long_fraction_min": float(row["long_fraction"]) >= self.min_long_fraction,
            "long_fraction_max": float(row["long_fraction"]) <= self.max_long_fraction,
            "validation_negative_years": int(row["validation_negative_years"]) <= self.max_validation_negative_years,
            "feature_count": int(row["feature_count"]) <= self.max_features_per_candidate,
        }


def split_train_validation(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    train = data.loc[data.index <= pd.Timestamp(TRAIN_END)]
    validation = data.loc[
        (data.index >= pd.Timestamp(VALIDATION_START))
        & (data.index <= pd.Timestamp(VALIDATION_END))
    ]
    return train.copy(), validation.copy()


def build_survival_grid(
    parameter_space: dict[str, list[Any]],
    *,
    stage: int,
    total_stages: int,
) -> list[dict[str, Any]]:
    base = _build_rule_candidates(parameter_space)
    grid = [candidate for candidate in base if _valid_candidate(candidate)]
    return [params for index, params in enumerate(grid) if index % total_stages == stage]


def _build_rule_candidates(parameter_space: dict[str, list[Any]]) -> list[dict[str, Any]]:
    if "rule" not in parameter_space:
        names = list(parameter_space)
        values = [parameter_space[name] for name in names]
        return [
            {"rule": "ma_crossover", **dict(zip(names, combination, strict=True))}
            for combination in product(*values)
        ]

    candidates: list[dict[str, Any]] = []
    for rule in parameter_space["rule"]:
        names = _rule_parameter_names(str(rule), parameter_space)
        values = [parameter_space[name] for name in names]
        for combination in product(*values):
            candidates.append({"rule": rule, **dict(zip(names, combination, strict=True))})
    return candidates


def _rule_parameter_names(rule: str, parameter_space: dict[str, list[Any]]) -> list[str]:
    rule_names = {
        "ma_crossover": ["fast_window", "slow_window"],
        "momentum_threshold": ["momentum_window", "threshold"],
        "mean_reversion": ["reversion_window", "entry_zscore", "exit_zscore"],
        "rsi_reversion": ["rsi_window", "rsi_buy", "rsi_sell"],
        "breakout": ["breakout_window", "exit_window"],
        "volatility_momentum": ["momentum_window", "volatility_window", "volatility_quantile"],
        "linear_score": ["fast_return_window", "slow_return_window", "risk_window", "score_threshold"],
    }[rule]
    return [name for name in rule_names if name in parameter_space]


def evaluate_survival_candidate(
    data: pd.DataFrame,
    params: dict[str, Any],
    *,
    initial_cash: float,
    commission_bps: float,
    slippage_bps: float,
    criteria: SurvivalCriteria | None = None,
) -> dict[str, Any]:
    criteria = criteria or SurvivalCriteria()
    train, validation = split_train_validation(data)
    train_result = _run_candidate(train, params, initial_cash, commission_bps, slippage_bps)
    validation_result = _run_candidate(validation, params, initial_cash, commission_bps, slippage_bps)
    train_metrics = _survival_metrics(train_result.equity_curve, train_result.metrics, train)
    validation_metrics = _survival_metrics(validation_result.equity_curve, validation_result.metrics, validation)
    row = {
        "candidate_id": _candidate_id(params),
        **params,
        "feature_count": _feature_count(params),
        "train_calmar": train_metrics["calmar"],
        "validation_calmar": validation_metrics["calmar"],
        "train_cagr": train_metrics["cagr"],
        "validation_cagr": validation_metrics["cagr"],
        "train_mdd": train_metrics["mdd"],
        "validation_mdd": validation_metrics["mdd"],
        "trades_per_year": train_metrics["trades_per_year"],
        "long_fraction": train_metrics["long_fraction"],
        "validation_negative_years": _negative_year_count(validation_result.equity_curve),
        "locked_opened": False,
    }
    checks = criteria.checks(row)
    row["robust_passes"] = criteria.pass_count(row)
    row["robust_total"] = len(checks)
    row["rejection_reason"] = criteria.rejection_reason(row)
    row["accepted"] = row["rejection_reason"] is None
    row["survival_score"] = survival_score(row)
    return row


def survival_score(row: dict[str, Any]) -> float:
    validation_calmar = float(row.get("validation_calmar", 0.0) or 0.0)
    train_calmar = float(row.get("train_calmar", 0.0) or 0.0)
    train_cagr = float(row.get("train_cagr", 0.0) or 0.0)
    validation_cagr = float(row.get("validation_cagr", 0.0) or 0.0)
    train_mdd = abs(float(row.get("train_mdd", 0.0) or 0.0))
    validation_mdd = abs(float(row.get("validation_mdd", 0.0) or 0.0))
    trades_per_year = float(row.get("trades_per_year", 0.0) or 0.0)
    long_fraction = float(row.get("long_fraction", 0.0) or 0.0)
    robust_passes = int(row.get("robust_passes", 0) or 0)
    gap = max(0.0, train_calmar - 2.0 * max(validation_calmar, 0.0))
    complexity = max(0, int(row.get("feature_count", 1)) - 2) * 0.15
    train_too_high = max(0.0, train_calmar - 8.0)
    trade_penalty = max(0.0, 12.0 - trades_per_year) * 0.08 + max(0.0, trades_per_year - 90.0) * 0.02
    long_penalty = max(0.0, 0.25 - long_fraction) * 4.0 + max(0.0, long_fraction - 0.75) * 4.0
    drawdown_penalty = max(0.0, train_mdd - 0.30) * 3.0 + max(0.0, validation_mdd - 0.30) * 3.0
    negative_year_penalty = int(row.get("validation_negative_years", 0) or 0) * 0.8
    return float(
        robust_passes * 10.0
        + validation_calmar
        + 0.25 * train_calmar
        + validation_cagr
        + 0.5 * train_cagr
        - 0.35 * gap
        - complexity
        - train_too_high
        - trade_penalty
        - long_penalty
        - drawdown_penalty
        - negative_year_penalty
    )


def _run_candidate(
    data: pd.DataFrame,
    params: dict[str, Any],
    initial_cash: float,
    commission_bps: float,
    slippage_bps: float,
):
    rule = params.get("rule")
    if rule == "ma_crossover":
        return run_backtest(
            data,
            params=params,
            initial_cash=initial_cash,
            commission_bps=commission_bps,
            slippage_bps=slippage_bps,
        )
    signals = _signals_for_rule(data, params)
    return _run_signals_backtest(
        data,
        signals=signals,
        initial_cash=initial_cash,
        commission_bps=commission_bps,
        slippage_bps=slippage_bps,
    )


def _signals_for_rule(data: pd.DataFrame, params: dict[str, Any]) -> pd.Series:
    rule = params.get("rule")
    if rule == "momentum_threshold":
        return _momentum_signals(
            data,
            window=int(params["momentum_window"]),
            threshold=float(params["threshold"]),
        )
    if rule == "mean_reversion":
        return _mean_reversion_signals(
            data,
            window=int(params["reversion_window"]),
            entry_zscore=float(params["entry_zscore"]),
            exit_zscore=float(params["exit_zscore"]),
        )
    if rule == "rsi_reversion":
        return _rsi_reversion_signals(
            data,
            window=int(params["rsi_window"]),
            buy_level=float(params["rsi_buy"]),
            sell_level=float(params["rsi_sell"]),
        )
    if rule == "breakout":
        return _breakout_signals(
            data,
            breakout_window=int(params["breakout_window"]),
            exit_window=int(params["exit_window"]),
        )
    if rule == "volatility_momentum":
        return _volatility_momentum_signals(
            data,
            momentum_window=int(params["momentum_window"]),
            volatility_window=int(params["volatility_window"]),
            volatility_quantile=float(params["volatility_quantile"]),
        )
    if rule == "linear_score":
        return _linear_score_signals(
            data,
            fast_return_window=int(params["fast_return_window"]),
            slow_return_window=int(params["slow_return_window"]),
            risk_window=int(params["risk_window"]),
            score_threshold=float(params["score_threshold"]),
        )
    raise ValueError(f"unsupported survival rule: {rule}")


def _momentum_signals(data: pd.DataFrame, *, window: int, threshold: float) -> pd.Series:
    momentum = data["close"].pct_change(window)
    signal = (momentum > threshold).astype(int)
    signal[momentum.isna()] = 0
    return signal


def _mean_reversion_signals(
    data: pd.DataFrame,
    *,
    window: int,
    entry_zscore: float,
    exit_zscore: float,
) -> pd.Series:
    close = data["close"].astype(float)
    mean = close.rolling(window).mean()
    std = close.rolling(window).std(ddof=0)
    zscore = (close - mean) / std.replace(0, np.nan)
    signal = pd.Series(0, index=data.index, dtype=int)
    in_market = False
    for timestamp, value in zscore.items():
        if np.isnan(value):
            signal.loc[timestamp] = int(in_market)
            continue
        if value <= -entry_zscore:
            in_market = True
        elif value >= exit_zscore:
            in_market = False
        signal.loc[timestamp] = int(in_market)
    return signal


def _rsi_reversion_signals(
    data: pd.DataFrame,
    *,
    window: int,
    buy_level: float,
    sell_level: float,
) -> pd.Series:
    close = data["close"].astype(float)
    diff = close.diff()
    gain = diff.clip(lower=0).rolling(window).mean()
    loss = (-diff.clip(upper=0)).rolling(window).mean()
    rs = gain / loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    signal = pd.Series(0, index=data.index, dtype=int)
    in_market = False
    for timestamp, value in rsi.items():
        if np.isnan(value):
            signal.loc[timestamp] = int(in_market)
            continue
        if value <= buy_level:
            in_market = True
        elif value >= sell_level:
            in_market = False
        signal.loc[timestamp] = int(in_market)
    return signal


def _breakout_signals(
    data: pd.DataFrame,
    *,
    breakout_window: int,
    exit_window: int,
) -> pd.Series:
    close = data["close"].astype(float)
    prior_high = close.rolling(breakout_window).max().shift(1)
    prior_low = close.rolling(exit_window).min().shift(1)
    signal = pd.Series(0, index=data.index, dtype=int)
    in_market = False
    for timestamp, price in close.items():
        high = prior_high.loc[timestamp]
        low = prior_low.loc[timestamp]
        if pd.notna(high) and price > high:
            in_market = True
        elif pd.notna(low) and price < low:
            in_market = False
        signal.loc[timestamp] = int(in_market)
    return signal


def _volatility_momentum_signals(
    data: pd.DataFrame,
    *,
    momentum_window: int,
    volatility_window: int,
    volatility_quantile: float,
) -> pd.Series:
    close = data["close"].astype(float)
    returns = close.pct_change()
    momentum = close.pct_change(momentum_window)
    volatility = returns.rolling(volatility_window).std(ddof=0)
    volatility_limit = volatility.rolling(252, min_periods=volatility_window).quantile(volatility_quantile)
    signal = ((momentum > 0) & (volatility < volatility_limit)).astype(int)
    signal[momentum.isna() | volatility.isna() | volatility_limit.isna()] = 0
    return signal


def _linear_score_signals(
    data: pd.DataFrame,
    *,
    fast_return_window: int,
    slow_return_window: int,
    risk_window: int,
    score_threshold: float,
) -> pd.Series:
    close = data["close"].astype(float)
    fast_return = close.pct_change(fast_return_window)
    slow_return = close.pct_change(slow_return_window)
    volatility = close.pct_change().rolling(risk_window).std(ddof=0)
    drawdown = close / close.rolling(risk_window).max() - 1.0
    score = (
        _rolling_zscore(fast_return, 252)
        + _rolling_zscore(slow_return, 252)
        - _rolling_zscore(volatility, 252)
        + _rolling_zscore(drawdown, 252)
    )
    signal = (score > score_threshold).astype(int)
    signal[score.isna()] = 0
    return signal


def _rolling_zscore(series: pd.Series, window: int) -> pd.Series:
    mean = series.rolling(window, min_periods=max(20, window // 4)).mean()
    std = series.rolling(window, min_periods=max(20, window // 4)).std(ddof=0)
    return (series - mean) / std.replace(0, np.nan)


def _run_signals_backtest(
    data: pd.DataFrame,
    *,
    signals: pd.Series,
    initial_cash: float,
    commission_bps: float,
    slippage_bps: float,
):
    from trading_lab.backtest import BacktestResult, calculate_metrics

    desired_position = signals.shift(1).fillna(0).astype(int)
    returns = data["close"].pct_change().fillna(0.0)
    position = desired_position.reindex(data.index).fillna(0).astype(float)
    cost_rate = (commission_bps + slippage_bps) / 10_000
    turnover = position.diff().abs().fillna(position.abs())
    strategy_returns = position * returns - turnover * cost_rate
    equity = initial_cash * (1.0 + strategy_returns).cumprod()
    equity_curve = pd.DataFrame({"timestamp": data.index, "equity": equity.to_numpy()})
    trades = _trades_from_position(data, position, initial_cash)
    metrics = calculate_metrics(equity_curve, trades, initial_cash)
    return BacktestResult(metrics=metrics, trades=trades, equity_curve=equity_curve)


def _survival_metrics(
    equity_curve: pd.DataFrame,
    metrics: dict[str, float],
    data: pd.DataFrame,
) -> dict[str, float]:
    years = max(len(data) / 252.0, 1 / 252.0)
    final_equity = float(metrics["final_equity"])
    initial_cash = float(metrics["initial_cash"])
    cagr = (final_equity / initial_cash) ** (1.0 / years) - 1.0
    mdd = float(metrics["max_drawdown_pct"]) / 100.0
    calmar = cagr / abs(mdd) if mdd < 0 else 0.0
    trades_per_year = float(metrics["trade_count"]) / years
    long_fraction = _long_fraction(equity_curve)
    return {
        "calmar": float(calmar),
        "cagr": float(cagr),
        "mdd": float(mdd),
        "trades_per_year": float(trades_per_year),
        "long_fraction": float(long_fraction),
    }


def _long_fraction(equity_curve: pd.DataFrame) -> float:
    equity = equity_curve["equity"].astype(float)
    return float((equity.pct_change().fillna(0) != 0).mean())


def _negative_year_count(equity_curve: pd.DataFrame) -> int:
    frame = equity_curve.copy()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"])
    frame["year"] = frame["timestamp"].dt.year
    negatives = 0
    for _, year_frame in frame.groupby("year"):
        first = float(year_frame.iloc[0]["equity"])
        last = float(year_frame.iloc[-1]["equity"])
        if last < first:
            negatives += 1
    return int(negatives)


def _trades_from_position(
    data: pd.DataFrame,
    position: pd.Series,
    initial_cash: float,
) -> pd.DataFrame:
    rows = []
    in_trade = False
    entry_time = None
    entry_price = 0.0
    current_equity = initial_cash
    for timestamp, target in position.items():
        close = float(data.loc[timestamp, "close"])
        if target == 1 and not in_trade:
            in_trade = True
            entry_time = timestamp
            entry_price = close
        elif target == 0 and in_trade:
            return_pct = close / entry_price - 1
            current_equity *= 1 + return_pct
            rows.append(
                {
                    "entry_time": entry_time,
                    "exit_time": timestamp,
                    "entry_price": entry_price,
                    "exit_price": close,
                    "pnl": current_equity - initial_cash,
                    "return_pct": return_pct * 100,
                    "exit_equity": current_equity,
                }
            )
            in_trade = False
    return pd.DataFrame(
        rows,
        columns=["entry_time", "exit_time", "entry_price", "exit_price", "pnl", "return_pct", "exit_equity"],
    )


def _candidate_id(params: dict[str, Any]) -> str:
    clean = "_".join(f"{key}-{value}" for key, value in sorted(params.items()))
    return clean.replace(".", "p").replace("-", "_")


def _feature_count(params: dict[str, Any]) -> int:
    rule_feature_count = {
        "ma_crossover": 2,
        "momentum_threshold": 1,
        "mean_reversion": 2,
        "rsi_reversion": 1,
        "breakout": 2,
        "volatility_momentum": 2,
        "linear_score": 4,
    }
    return rule_feature_count.get(str(params.get("rule")), 1)


def _valid_candidate(params: dict[str, Any]) -> bool:
    if params.get("rule") == "ma_crossover":
        return int(params["fast_window"]) < int(params["slow_window"])
    if params.get("rule") == "mean_reversion":
        return float(params["entry_zscore"]) > float(params["exit_zscore"])
    if params.get("rule") == "rsi_reversion":
        return float(params["rsi_buy"]) < float(params["rsi_sell"])
    if params.get("rule") == "breakout":
        return int(params["exit_window"]) <= int(params["breakout_window"])
    if params.get("rule") == "linear_score":
        return int(params["fast_return_window"]) < int(params["slow_return_window"])
    return True
