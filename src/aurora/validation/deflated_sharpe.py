"""Deflated Sharpe Ratio gate (Bailey & Lopez de Prado 2014).

Use after selection of best strategy from N trials.

Unit convention
---------------
``deflated_sharpe`` and ``probabilistic_sharpe`` in ``core/metrics.py`` apply the
Mertens variance ``(1 - skew*SR + ((kurt-1)/4)*SR^2) / (n-1)``. That formula
assumes ``observed_sharpe`` is in **per-period** units matched to ``n_periods``
(e.g. daily Sharpe with ``n_periods`` = number of daily bars). Passing an
**annualized** Sharpe with a per-period ``n_periods`` inflates the variance term
and silently inflates DSR/PSR.

To make the conversion explicit at the call site we expose:

* ``deflated_sharpe_check`` -- legacy entry point. It now accepts an optional
  ``ppy`` argument and converts ``observed_sharpe`` to per-period units before
  delegating, so callers passing an annualized Sharpe get the correct DSR.
* ``deflated_sharpe_annualized`` -- a thin convenience wrapper that takes an
  annualized Sharpe + ``ppy`` and returns a ``DSRReport`` based on the
  converted per-period Sharpe.
"""
from __future__ import annotations
import math
import warnings
from dataclasses import dataclass
from aurora.core.metrics import deflated_sharpe, probabilistic_sharpe


@dataclass
class DSRReport:
    observed_sharpe: float
    n_trials: int
    n_periods: int
    skew: float
    kurtosis: float
    dsr: float
    psr_vs_zero: float
    passed: bool | None

    def passes(self, min_dsr: float = 0.95) -> bool:
        return self.dsr >= min_dsr


def _to_per_period_sharpe(observed_sharpe: float, ppy: int | None) -> float:
    """Convert an annualized Sharpe to per-period units when ``ppy`` is given.

    When ``ppy`` is None or <= 1 the input is treated as already per-period.
    """
    if ppy is None or ppy <= 1:
        return float(observed_sharpe)
    return float(observed_sharpe) / math.sqrt(float(ppy))


def deflated_sharpe_check(observed_sharpe: float, n_trials: int, n_periods: int,
                          skew: float = 0.0, kurtosis: float = 0.0,
                          min_dsr: float = 0.95,
                          ppy: int | None = None,
                          min_psr: float | None = None) -> DSRReport:
    """Run the DSR gate and emit a ``DSRReport``.

    Args:
        observed_sharpe: Sharpe of the selected strategy. If ``ppy`` is provided
            it is treated as **annualized** and converted to per-period internally
            before applying the Mertens variance. If ``ppy`` is None it is
            treated as already in per-period units (legacy behaviour).
        n_trials: number of strategies evaluated during selection.
        n_periods: bars in the OOS sample (must be in the same time unit as the
            per-period Sharpe; e.g. daily bars when ``ppy`` is daily).
        skew, kurtosis: of the strategy returns.
        min_dsr: gate threshold for ``passed``.
        ppy: periods per year. If given, ``observed_sharpe`` is divided by
            ``sqrt(ppy)`` before the Mertens variance is applied.
        min_psr: optional independent threshold for ``psr_vs_zero``. When set
            and ``n_trials > 1``, the ``passed`` flag also requires
            ``psr_vs_zero >= min_psr`` (so callers can keep DSR and PSR gates
            decoupled).
    """
    sr_pp = _to_per_period_sharpe(observed_sharpe, ppy)

    if n_trials == 1:
        warnings.warn(
            "DSR with n_trials=1 reduces to standard Sharpe; "
            "multiplicity correction inactive. Gate is reported as None to "
            "avoid the knife-edge of comparing PSR-vs-zero to a high min_dsr "
            "(e.g. 0.95). Pass min_psr explicitly to enable a PSR-only gate.",
            UserWarning,
            stacklevel=2,
        )
        # With no multiplicity we still compute PSR-vs-zero for visibility,
        # but ``passed`` is set to None unless the caller has provided an
        # explicit ``min_psr`` threshold. This avoids the previous behaviour
        # where strategies were silently failed because PSR (typically ~0.5
        # for borderline-positive Sharpes) was compared to a 0.95 default.
        psr = probabilistic_sharpe(sr_pp, 0.0, n_periods, skew, kurtosis)
        if min_psr is None:
            passed = None
        else:
            passed = psr >= float(min_psr)
        return DSRReport(
            observed_sharpe=observed_sharpe,
            n_trials=n_trials,
            n_periods=n_periods,
            skew=skew,
            kurtosis=kurtosis,
            dsr=psr,
            psr_vs_zero=psr,
            passed=passed,
        )

    dsr = deflated_sharpe(sr_pp, n_trials, n_periods, skew, kurtosis)
    psr = probabilistic_sharpe(sr_pp, 0.0, n_periods, skew, kurtosis)
    passed = dsr >= min_dsr
    if min_psr is not None:
        passed = passed and (psr >= float(min_psr))
    return DSRReport(
        observed_sharpe=observed_sharpe,
        n_trials=n_trials,
        n_periods=n_periods,
        skew=skew,
        kurtosis=kurtosis,
        dsr=dsr,
        psr_vs_zero=psr,
        passed=passed,
    )


def deflated_sharpe_annualized(observed_sharpe_ann: float, n_trials: int,
                               n_periods: int, ppy: int,
                               skew: float = 0.0, kurtosis: float = 0.0,
                               min_dsr: float = 0.95,
                               min_psr: float | None = None) -> DSRReport:
    """Convenience wrapper: input is the annualized Sharpe and bar count.

    Equivalent to ``deflated_sharpe_check(observed_sharpe_ann, ..., ppy=ppy)``.
    Provided so call sites that already work in annualized units do not have to
    remember to pass ``ppy`` explicitly.
    """
    return deflated_sharpe_check(
        observed_sharpe=observed_sharpe_ann,
        n_trials=n_trials,
        n_periods=n_periods,
        skew=skew,
        kurtosis=kurtosis,
        min_dsr=min_dsr,
        ppy=ppy,
        min_psr=min_psr,
    )
