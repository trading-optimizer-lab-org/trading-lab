# Trading Lab

Proyecto base para backtesting y optimizacion por fases usando Python,
GitHub Actions y GitHub Codespaces.

## Uso rapido local

```bash
python -m pip install -e ".[dev]"
python scripts/run_backtest.py --config configs/base.yaml
python scripts/run_optimization_stage.py --config configs/optimization.yaml --stage 0 --total-stages 16
python scripts/merge_leaderboards.py --input-glob "outputs/optimization/stage_*.csv"
```

## En GitHub

- `ci.yml`: ejecuta tests en push y pull request.
- `backtest-manual.yml`: lanza un backtest manual desde Actions.
- `optimization-staged.yml`: divide la optimizacion en 16 jobs y une el leaderboard final.
- `public-data-optimization.yml`: descarga SPY diario publico y ejecuta la optimizacion en 16 jobs.
- `survival-search.yml`: descarga SPY diario publico y ejecuta una busqueda survival en 64 jobs.

## Runs en GitHub sin usar tu PC

1. Entra en `Actions`.
2. Lanza `Public Data Optimization` con `Run workflow`.
3. Espera a que termine y descarga el artifact `public-optimization-leaderboard`.
4. Lanza `Survival Search` con `Run workflow`.
5. Espera a que termine y descarga el artifact `survival-leaderboard`.

El archivo clave de cada run es `summary.json`:

- `best`: mejor candidata encontrada.
- `rows` / `candidates_evaluated`: candidatas evaluadas.
- `accepted`: candidatas que pasan filtros survival.
- `rejection_counts`: motivos de rechazo agrupados.
- `robust_passes` / `robust_total`: cuantos filtros pasa la mejor candidata.

Los datos de mercado se descargan dentro de GitHub Actions en cada run. No se suben datos privados al repo. La descarga intenta Stooq primero y usa Yahoo Finance como respaldo publico si Stooq pide apikey.
Cuando usa Yahoo, los precios de SPY se ajustan por dividendos para que el historico largo no quede castigado artificialmente.

## Formato de datos

El CSV debe contener:

```text
timestamp,open,high,low,close,volume
```

No subas datos sensibles, claves API ni estrategias privadas a un repo publico.
El mercado ya muerde bastante sin darle cubiertos.
