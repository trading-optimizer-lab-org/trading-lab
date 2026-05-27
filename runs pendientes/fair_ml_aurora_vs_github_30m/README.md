# Fair ML Aurora vs GitHub 30m

Run pendiente para comparar Aurora ML contra el motor `machine_learning` de `trading-lab`
en GitHub Actions, con comparación normalizada:

- `normalized`: SPY diario normalizado, locked cerrado y misma regla de validez.
- Se ejecuta dos veces con orden invertido: Aurora primero y GitHub primero.

Workflow:

```text
Fair ML Aurora vs GitHub 30m
```

Artifact final:

```text
fair-ml-aurora-vs-github-30m-results
```

Regla de candidata válida:

- `train_calmar > 1`
- `validation_calmar > 1`
- `validation_calmar >= 0.80 * train_calmar`
- `train_cagr >= 4%`
- `validation_cagr >= 3%`
- `locked_opened=false`

Ganador:

1. mayor `valid_per_min` en `normalized`;
2. desempate por `best_validation_calmar`;
3. desempate por `best_train_calmar`;
4. desempate por `evaluated_per_min`.
