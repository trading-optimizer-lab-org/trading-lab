"""Performance metrics. All take np.array of returns (any frequency) + ppy=periods/year.

Key metrics:
- CAGR, MDD, Calmar
- Sharpe, Sortino
- MAR (CAGR / MDD)
- Deflated Sharpe Ratio (Bailey-Lopez de Prado) — corrects for selection bias
- Skew, kurtosis
- Win rate, profit factor
"""
from __future__ import annotations
from dataclasses import dataclass, asdict
import math
import numpy as np
from scipy import stats


@dataclass
class Metrics:
    cagr: float
    mdd: float
    calmar: float
    sharpe: float
    sortino: float
    mar: float
    skew: float
    kurtosis: float
    win_rate: float
    profit_factor: float
    n_periods: int
    final_nav: float
    # Disambiguated period counts. ``n_periods_raw`` is the raw input length
    # (NaN bars included); it drives the CAGR annualization basis so warm-up
    # bars count as calendar time. ``n_periods_finite`` is the count after
    # NaN bars are dropped; it drives Sharpe / Sortino / skew / kurtosis.
    # ``n_periods`` is kept as a synonym for ``n_periods_finite`` for
    # backwards compatibility with older callers.
    n_periods_raw: int = 0
    n_periods_finite: int = 0

    def to_dict(self) -> dict:
        return asdict(self)


def compute_metrics(returns, ppy: int = 252) -> Metrics:
    """Compute full metric suite from returns array.

    Args:
        returns: np.array of period returns (e.g. daily, monthly). The CAGR /
            annualization basis is the RAW input length divided by ``ppy``,
            BEFORE NaN bars are dropped. This means warm-up bars (NaN signals)
            still count towards calendar time. Callers that want CAGR computed
            only over the post-warm-up window must drop those NaN bars
            themselves before calling this function.
        ppy: periods per year

    Returns:
        Metrics dataclass.

        ``n_periods`` historically meant "number of finite (non-NaN) bars used
        for moment statistics", but the same field name was overloaded in
        callers that wanted the raw input length. The two are now exposed
        separately as ``n_periods_raw`` (full input length, drives CAGR
        annualization) and ``n_periods_finite`` (post-NaN-drop count, drives
        Sharpe / Sortino / skew / kurtosis). ``n_periods`` is preserved as a
        synonym for ``n_periods_finite``.

    Notes:
        Calmar handles a zero (or near-zero) drawdown explicitly:
            - cagr > 0 and |mdd| < 1e-9 -> +inf
            - cagr < 0 and |mdd| < 1e-9 -> -inf
            - cagr == 0 and |mdd| < 1e-9 -> 0.0
        This avoids the inf/inf NaN trap and gives downstream consumers a
        well-defined sign for "no drawdown observed".
    """
    raw = np.asarray(returns, dtype=float)
    original_len = len(raw)
    r = raw[~np.isnan(raw)]
    if len(r) < 2:
        # Use NaN sentinels for empty/single-bar series. The previous final_nav=1.0
        # implied "no PnL" when the series was actually undefined.
        return Metrics(
            float("nan"), float("nan"), float("nan"), float("nan"),
            float("nan"), float("nan"), float("nan"), float("nan"),
            float("nan"), float("nan"), len(r), float("nan"),
            n_periods_raw=original_len, n_periods_finite=len(r),
        )

    nav = np.cumprod(1.0 + r)
    final = float(nav[-1])
    # Annualization basis uses the RAW input length (warm-up NaN bars count as
    # calendar time). See docstring.
    years = original_len / ppy if ppy > 0 else 0.0
    if years > 0 and final > 0:
        cagr = (final ** (1.0 / years)) - 1.0
    elif final <= 0:
        # Full bankruptcy / negative NAV: capital wiped out. Report -1.0
        # (i.e. "lost 100%/yr") instead of 0.0 so downstream consumers don't
        # mistake a wipeout for a flat strategy.
        cagr = -1.0
    else:
        cagr = 0.0

    cummax = np.maximum.accumulate(nav)
    dd = (nav - cummax) / cummax
    mdd = float(dd.min())

    if abs(mdd) < 1e-9:
        if cagr > 0:
            calmar = float("inf")
        elif cagr < 0:
            calmar = float("-inf")
        else:
            calmar = 0.0
    else:
        calmar = cagr / abs(mdd)
    mar = calmar  # same definition

    std = r.std()
    mean = r.mean()
    sharpe = (mean / std) * math.sqrt(ppy) if std > 1e-12 else 0.0

    downside = r[r < 0]
    dstd = downside.std() if len(downside) > 1 else std
    sortino = (mean / dstd) * math.sqrt(ppy) if dstd > 1e-12 else 0.0

    if std > 1e-12:
        sk = float(stats.skew(r)) if len(r) > 2 else 0.0
        kt = float(stats.kurtosis(r, fisher=False)) if len(r) > 3 else 0.0
    else:
        sk = 0.0
        kt = 0.0

    wins = r[r > 0]; losses = r[r < 0]
    win_rate = len(wins) / len(r) if len(r) > 0 else 0.0
    profit_factor = (
        wins.sum() / abs(losses.sum()) if len(losses) > 0 and losses.sum() != 0 else 0.0
    )

    return Metrics(
        cagr=round(cagr * 100, 4),
        mdd=round(mdd * 100, 4),
        calmar=round(calmar, 4),
        sharpe=round(sharpe, 4),
        sortino=round(sortino, 4),
        mar=round(mar, 4),
        skew=round(sk, 4),
        kurtosis=round(kt, 4),
        win_rate=round(win_rate, 4),
        profit_factor=round(profit_factor, 4),
        n_periods=len(r),
        final_nav=round(final, 6),
        n_periods_raw=original_len,
        n_periods_finite=len(r),
    )


def deflated_sharpe(observed_sharpe: float, n_trials: int, n_periods: int,
                    skew: float = 0.0, kurtosis: float = 0.0) -> float:
    """Deflated Sharpe Ratio (Bailey & Lopez de Prado 2014).

    Corrects raw Sharpe for selection bias when N strategies tested.
    Returns probability that observed Sharpe > 0 given selection from N trials.

    Args:
        observed_sharpe: max Sharpe found across N trials (annualized)
        n_trials: number of strategies tested
        n_periods: backtest length (in periods)
        skew, kurtosis: of selected strategy returns

    Returns:
        DSR in [0, 1]. > 0.95 = passes 95% confidence.
    """
    if n_trials < 2 or n_periods < 10:
        return 0.0

    # Expected max Sharpe under H0 (no skill)
    e_max = ((1 - np.euler_gamma) * stats.norm.ppf(1 - 1.0 / n_trials)
             + np.euler_gamma * stats.norm.ppf(1 - 1.0 / (n_trials * np.e)))

    # Variance of estimated Sharpe given skew/kurtosis (Mertens)
    var_sr = (1 - skew * observed_sharpe + ((kurtosis - 1) / 4.0) * observed_sharpe**2) / (n_periods - 1)
    if var_sr <= 0:
        return 0.0
    sigma_sr = math.sqrt(var_sr)

    z = (observed_sharpe - e_max) / sigma_sr
    dsr = float(stats.norm.cdf(z))
    return round(dsr, 4)


def probabilistic_sharpe(observed_sharpe: float, benchmark_sharpe: float,
                        n_periods: int, skew: float = 0.0, kurtosis: float = 0.0) -> float:
    """Probabilistic Sharpe Ratio (Bailey-Lopez de Prado).

    Probability observed Sharpe truly exceeds benchmark.
    """
    var_sr = (1 - skew * observed_sharpe + ((kurtosis - 1) / 4.0) * observed_sharpe**2) / (n_periods - 1)
    if var_sr <= 0: return 0.0
    z = (observed_sharpe - benchmark_sharpe) / math.sqrt(var_sr)
    return round(float(stats.norm.cdf(z)), 4)
