# Run Pendiente: Weekly SPY Sharpe 10 Methods 9h Waves

Estado: `pendiente_no_lanzar`

Regla principal: este run no debe ejecutarse hasta que el usuario diga explicitamente:

```text
lanza el run pendiente de 10 métodos 9h
```

## Objetivo

Comparar 10 metodos de forma justa por olas, maximizando `train_sharpe`.

Validacion queda solo como `report_only`.

Locked queda cerrado: `locked_opened=false`.

## Metodos

| Metodo |
|---|
| `beam` |
| `genetic` |
| `sobol_random_asha_real` |
| `optuna_tpe_hyperband` |
| `dehb_real` |
| `bohb_real` |
| `smac_mf_real` |
| `bandit` |
| `aurora_ml` |
| `github_ml` |

## Diseno

| Parametro | Valor |
|---|---:|
| Olas | 5 |
| Jobs por ola | 180 |
| Jobs por metodo por ola | 18 |
| Jobs totales | 900 |
| Jobs por metodo | 90 |
| Tiempo efectivo por job | 85 min |
| Maximo paralelo por ola | 180 |
| Limite total previsto | 9 horas |

Cada ola tiene la matriz balanceada por `stage` primero y `method` despues.

## Archivos Activos Preparados

Config:

```text
configs/weekly_spy_sharpe_10methods_9h_waves.yaml
```

Workflows manuales:

```text
.github/workflows/weekly-spy-sharpe-10methods-9h-waves.yml
.github/workflows/weekly-spy-sharpe-10methods-9h-wave.yml
.github/workflows/weekly-spy-sharpe-10methods-9h-merge-now.yml
.github/workflows/weekly-spy-sharpe-10methods-9h-stop.yml
```

Scripts:

```text
scripts/run_weekly_spy_sharpe_10methods_9h_stage.py
scripts/merge_weekly_spy_sharpe_10methods_9h.py
scripts/merge_weekly_spy_sharpe_10methods_9h_state.py
```

## Lanzamiento Manual

No lanzar automaticamente.

Cuando el usuario lo ordene, ejecutar el workflow:

```text
Weekly SPY Sharpe 10 Methods 9h Waves
```

Inputs previstos:

```text
waves=5
minutes_per_method_stage=85
max_parallel=180
```

## Merge Bajo Demanda

Si el run esta vivo, cancelado o terminado, se puede mergear con:

```text
Weekly SPY Sharpe 10 Methods 9h Merge Now
```

Input:

```text
source_run_id=<run_id_del_search>
```

## Stop

Para parar el search sin mergear:

```text
Weekly SPY Sharpe 10 Methods 9h Stop
```

Input:

```text
source_run_id=<run_id_del_search>
```

Despues de parar, lanzar `Merge Now` si se quieren resultados parciales.
