# Taylor Expansion Network Combiner Design

## 1. Motivation

The current adaptive factor combination step uses a linear regression over selected alpha factors. This keeps the final signal interpretable, but it cannot express nonlinear interactions such as `factor_a * factor_b`. If the useful signal only appears when two factors jointly activate, a pure linear combiner can underfit even when the underlying factors are individually useful.

This design adds a Taylor Expansion Network (TEN) combiner for the final signal-construction stage. It keeps the model white-box by expanding selected factors into explicit first-order and second-order terms, then fitting a regularized linear model over those terms.

## 2. Model Form

For a stock `i` at date `t`, selected factors are:

```text
f_1, f_2, ..., f_N
```

The TEN score is:

```text
phi_i,t = w_0
        + sum_j w_j * f_j,i
        + sum_j sum_{k=j}^N w_j,k * (f_j,i * f_k,i)
```

This is equivalent to a second-order multivariate Taylor expansion over the selected factors. It captures:

- First-order standalone factor contribution: `w_j`
- Second-order self curvature: `w_j,j * f_j^2`
- Pairwise interaction contribution: `w_j,k * f_j * f_k`

## 3. Implementation Choice

The first implementation should not use a generic black-box neural network training loop. Instead, it should use:

```text
explicit Taylor feature expansion + standardized Ridge regression
```

This is mathematically equivalent to a single TEN layer, but is easier to train inside the rolling adaptive combiner and more stable for small rolling windows.

## 4. Feature Construction

Given `X` with shape `[samples, N]`, construct:

```text
linear terms:      X[:, j]
interaction terms: X[:, j] * X[:, k], for 0 <= j <= k < N
intercept:         1
```

The same interaction pairs must be reused for the prediction row `to_pred`.

To prevent second-order terms from dominating by scale, fit standardization statistics on the training window after expansion:

```text
Z_train = (Z_train_raw - mean) / std
Z_pred  = (Z_pred_raw  - mean) / std
```

The intercept column is appended after standardization and is not standardized.

## 5. Regularization

TEN can overfit because interaction features grow as `N * (N + 1) / 2`. Use grouped Ridge regularization:

```text
linear terms:      ten_linear_alpha
interaction terms: ten_interaction_alpha
intercept:         0
```

The interaction penalty should default higher than the linear penalty.

Default values:

```text
ten_linear_alpha = 1e-4
ten_interaction_alpha = 1e-1
```

## 6. CLI Design

Add a new combiner switch independent from factor selection:

```text
--combiner linear|ridge|ten
```

Behavior:

- `linear`: original least-squares combiner.
- `ridge`: first-order Ridge combiner.
- `ten`: second-order Taylor Expansion Network combiner.

The existing `--selection_mode` controls which factors are selected; `--combiner` controls how selected factors are combined. `--ten_blend` anchors TEN to the linear prediction, where `0.0` equals the linear combiner and `1.0` uses the full second-order TEN prediction.

## 7. Interpretability

The fitted TEN coefficients are directly interpretable:

- `linear_weights[j]` corresponds to factor `j` standalone contribution.
- `interaction_weights[(j, k)]` corresponds to the explicit interaction `factor_j * factor_k`.

A future reporting step can aggregate rolling TEN coefficients and print top absolute interactions, such as:

```text
factor_3 x factor_7: +0.018
factor_1 x factor_1: -0.012
```

## 8. Test Coverage

Unit tests should verify:

- Taylor expansion produces correct linear and pairwise features.
- TEN can fit a target that depends on `x1 * x2` while a linear model cannot represent that interaction exactly.
- TEN returns interpretable coefficient metadata for linear and interaction terms.
- CLI help exposes the new `--combiner` and TEN regularization arguments.
