from __future__ import annotations

import json
import sys


def main() -> int:
    checks: dict[str, str] = {}
    try:
        import optuna

        checks["optuna"] = optuna.__version__
        optuna.samplers.TPESampler(seed=1)
        optuna.pruners.HyperbandPruner(min_resource=1, max_resource=3)
    except Exception as exc:
        raise RuntimeError(f"Optuna TPE/Hyperband smoke failed: {exc}") from exc

    try:
        import ConfigSpace
        from ConfigSpace import ConfigurationSpace, UniformFloatHyperparameter

        checks["ConfigSpace"] = getattr(ConfigSpace, "__version__", "unknown")
        cs = ConfigurationSpace()
        cs.add_hyperparameter(UniformFloatHyperparameter("x", lower=0.0, upper=1.0))
        cs.sample_configuration()
    except Exception as exc:
        raise RuntimeError(f"ConfigSpace smoke failed: {exc}") from exc

    try:
        import dehb
        from dehb import DEHB

        checks["dehb"] = getattr(dehb, "__version__", "unknown")
        if DEHB is None:
            raise RuntimeError("DEHB class missing")
    except Exception as exc:
        raise RuntimeError(f"DEHB smoke failed: {exc}") from exc

    try:
        import hpbandster
        from hpbandster.optimizers import BOHB

        checks["hpbandster"] = getattr(hpbandster, "__version__", "unknown")
        if BOHB is None:
            raise RuntimeError("BOHB class missing")
    except Exception as exc:
        raise RuntimeError(f"HpBandSter BOHB smoke failed: {exc}") from exc

    try:
        import smac
        from smac import MultiFidelityFacade, Scenario

        checks["smac"] = getattr(smac, "__version__", "unknown")
        if MultiFidelityFacade is None or Scenario is None:
            raise RuntimeError("SMAC facade/scenario missing")
    except Exception as exc:
        raise RuntimeError(f"SMAC smoke failed: {exc}") from exc

    print(json.dumps({"real_hpo_dependency_smoke": "ok", "versions": checks}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
