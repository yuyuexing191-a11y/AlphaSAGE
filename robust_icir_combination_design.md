# Robust ICIR Factor Combination Design

## 1. Motivation

The current structure-aware action mask can improve the quality of generated single factors, but it does not directly optimize the final ensemble objective. Current adaptive combination mainly filters factors by rolling `RIC` / `RICIR`, then fits an unregularized least-squares model. This creates two problems:

- High single-factor IC does not guarantee marginal contribution to the combined factor.
- Highly correlated selected factors can make regression weights unstable and reduce out-of-sample IC / ICIR.

The goal of this change is to improve final combination `Test IC` and `Test ICIR` without changing the GFlowNet generation path.

## 2. Recommended Approach

Add a new robust combination mode to `run_adaptive_combination.py`:

```text
--selection_mode robust_icir
```

The existing behavior remains available as:

```text
--selection_mode threshold
```

This makes the change easy to ablate:

```text
original combiner vs robust_icir combiner
```

## 3. Robust ICIR Selection

For each rolling window, compute per-factor statistics:

```text
ic_mean, ic_std, ric_mean, ric_std
icir = ic_mean / ic_std
ricir = ric_mean / ric_std
```

Then rank candidate factors by a stable score:

```text
score = ic_weight * abs(icir) + ric_weight * abs(ricir)
```

Default weights:

```text
ic_weight = 0.7
ric_weight = 0.3
```

This prioritizes Pearson IC stability while still retaining rank-correlation information.

## 4. Redundancy Control

After ranking factors by stable score, select factors greedily with a rolling-window correlation cap:

```text
abs(corr(candidate, selected_factor)) <= corr_threshold
```

This prevents the combination from repeatedly selecting near-duplicate factors with similar historical ICIR. If the filter is too strict and no factor is selected, the top-scoring factor is kept as a fallback.

## 5. Ridge Weighting

Replace the unregularized least-squares solve in robust mode with Ridge regression:

```text
coef = (X.T @ X + alpha * I)^-1 @ X.T @ y
```

The intercept is not penalized. This uses the existing `--ridge_alpha` argument instead of leaving it unused. Ridge should reduce weight instability when selected factors remain partially collinear.

## 6. CLI Controls

New or activated arguments:

```text
--selection_mode threshold|robust_icir
--ic_weight 0.7
--ric_weight 0.3
--corr_threshold 0.95
--ridge_alpha 1e-4
```

Existing threshold mode keeps old behavior and still uses the previous selection logic.

## 7. Testing Strategy

Unit tests should cover:

- Robust selection prefers high stable ICIR / RICIR factors.
- Robust selection skips highly correlated lower-ranked factors.
- Robust selection falls back to the top factor when all candidates are filtered.
- Ridge regression shrinks unstable collinear weights and does not penalize the intercept.

The implementation should be testable without loading QLib data or running a full adaptive-combination experiment.

## 8. Expected Experiment

Run the same expression pools with both modes:

```bash
python run_adaptive_combination.py --expressions_file <pool.json> --selection_mode threshold
python run_adaptive_combination.py --expressions_file <pool.json> --selection_mode robust_icir --ridge_alpha 1e-4 --corr_threshold 0.9
```

Primary comparison metrics:

- Validation IC / ICIR
- Test IC / ICIR
- Test RIC / RICIR

Secondary metrics:

- RET
- RET_SR
- RET_MDD

