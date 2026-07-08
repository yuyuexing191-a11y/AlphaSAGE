# Combination-Aware GFlowNet for Alpha Factor Discovery

## 1. 背景问题

当前 Alpha 因子搜索主要优化单个因子的 IC：

```text
reward = single_factor_ic + novelty_reward
```

这个目标有一个问题：**单因子训练 IC 高，不代表它加入因子池后能提升最终组合因子的表现**。

你现在的实验也说明了这一点：

- 单个因子的 `best IC` 有时能到 `0.04` 左右。
- 但经过 `run_adaptive_combination.py` 组合后，最终 `Test IC / RET / Sharpe` 不一定明显提升。
- 说明当前搜索目标和最终评价目标并不完全一致。

因此，真正要解决的问题不是单纯生成更复杂或单因子 IC 更高的因子，而是：

> 如何让 GFlowNet 更快找到一组互补、低冗余、组合后泛化更好的 Alpha 因子。

## 2. 原始方案：结构感知 Action Mask

结构感知 action mask 的作用是：在因子生成过程中，提前限制明显危险或无效的表达式扩展路径。

例如，如果当前表达式中已经出现较多危险算子：

- `Div`
- `Inv`
- `Pow`
- `Log`
- 多层 `Greater / Less`

则暂时禁止继续添加更多高风险算子，减少数值不稳定、接近常数、过拟合严重的复杂表达式。

这个方法的优点是：

- 实现简单。
- 不改变原始 reward。
- 可以减少一部分无效搜索。
- 可以用 `--enable_structure_mask` 控制开关，对旧实验影响较小。

但它的问题也很明显：

- 它只改变生成路径，不直接优化组合因子表现。
- 它可能提高单因子训练 IC。
- 但组合后的 `Test IC / ICIR / RIC / RET / Sharpe` 不一定稳定提升。

所以，结构感知 action mask 更适合作为辅助模块，而不是论文主创新。

## 3. 更适合作为主创新的方向

更合理的主线是：

> 从“单因子 IC 驱动搜索”改为“组合贡献驱动搜索”。

也就是说，候选因子进入 pool 时，不只看它自己的 IC，而是看：

> 这个因子加入当前因子池后，能不能提升最终组合因子的效果。

这个方向可以命名为：

```text
Combination-Aware GFlowNet
```

或者：

```text
Marginal Contribution Guided GFlowNet
```

## 4. 核心 Reward 设计

原始 reward 可以理解为：

```text
reward = single_ic + ssl_weight * ssl_reward + novelty_weight * novelty_reward
```

改进后建议变成：

```text
reward = single_ic
       + lambda_1 * marginal_ensemble_gain
       + lambda_2 * residual_ic
       + lambda_3 * novelty_reward
       - lambda_4 * redundancy_penalty
```

### 4.1 Single IC

衡量候选因子自己的预测能力：

```text
single_ic = abs(IC(candidate_factor, target))
```

它仍然重要，因为完全没有预测能力的因子不应该进入因子池。

但它不应该是唯一目标。

### 4.2 Marginal Ensemble Gain

这是最重要的新增部分。

它衡量新因子加入当前 pool 后，组合因子的验证集表现有没有提升：

```text
marginal_ensemble_gain =
    IC(ensemble(pool + candidate), target)
  - IC(ensemble(pool), target)
```

如果加入候选因子后组合 IC 提高，则给正奖励。

如果加入后组合 IC 不变甚至下降，则不给奖励或轻微惩罚。

这个设计直接对准最终目标：组合因子的表现。

### 4.3 Residual IC

Residual IC 用来判断候选因子是否提供了已有因子解释不了的新信息。

思路是：

1. 用当前 pool 中已有因子解释候选因子。
2. 得到残差。
3. 看残差和 target 是否仍然有 IC。

可以理解为：

```text
candidate = explained_by_existing_factors + residual
residual_ic = IC(residual, target)
```

如果 residual IC 高，说明这个因子不是简单重复已有因子，而是带来了新信息。

### 4.4 Novelty Reward

Novelty Reward 继续保留。

它的作用是防止候选因子和已有因子太像。

但它和 residual IC 不完全一样：

- Novelty 更偏“相似度去重”。
- Residual IC 更偏“是否有新增预测信息”。

### 4.5 Redundancy Penalty

如果候选因子和已有因子高度相关，则给惩罚：

```text
redundancy_penalty = max_corr(candidate, existing_factors)
```

这可以减少多重共线性，提升组合稳定性。

## 5. 结构感知 Action Mask 的定位

结构感知 action mask 不建议作为主创新。

它更适合作为辅助模块：

```text
Combination-Aware Reward 是主创新
Structure-Aware Action Mask 是搜索加速和稳定化模块
```

论文里可以这样表述：

> We further introduce a conservative structure-aware action mask to prune numerically unstable expression prefixes, reducing invalid exploration without changing the original search objective when disabled.

也就是说：

- 主创新解决“搜索目标和最终组合目标不一致”。
- mask 解决“搜索空间中存在明显无效复杂表达式”的问题。

## 6. 为什么这个方向比单独 Mask 更站得住

单独 mask 的问题是：

```text
它只能告诉模型不要走某些危险路径，
但不能告诉模型什么因子组合起来更好。
```

而组合感知 reward 能直接回答：

```text
这个新因子加入 pool 后，最终组合有没有变强？
```

这和最终实验指标一致，因此更适合作为论文主线。

如果最终论文指标是：

- Test IC
- Test ICIR
- Test RIC
- Test RICIR
- RET
- Sharpe

那么方法也应该围绕这些组合指标优化，而不是只优化单因子训练 IC。

## 7. 实验设计

建议实验分为四组。

### 7.1 Baseline

原始 GFlowNet，不加结构 mask，不加组合感知 reward。

```text
baseline = original GFlowNet
```

### 7.2 Baseline + Structure Mask

只加入结构感知 action mask。

```text
baseline + structure_mask
```

目的：验证 mask 是否能提升搜索稳定性和单因子质量。

### 7.3 Baseline + Combination-Aware Reward

加入组合感知 reward，但不加 mask。

```text
baseline + marginal_ensemble_gain + residual_ic
```

目的：验证主创新是否能提升最终组合指标。

### 7.4 Full Model

组合感知 reward 和结构感知 mask 都加入。

```text
baseline + combination_aware_reward + structure_mask
```

目的：验证两个模块是否互补。

## 8. 需要报告的指标

不要只报告单因子 best IC。

应该同时报告：

```text
Best Single Train IC
Best Single Test IC
Adaptive Combination Test IC
Adaptive Combination Test ICIR
Adaptive Combination Test RIC
Adaptive Combination Test RICIR
RET
Sharpe
Max Drawdown
```

其中最重要的是：

```text
Adaptive Combination Test IC
Adaptive Combination Test ICIR
Adaptive Combination Test RIC
Adaptive Combination Test RICIR
```

因为这些才是最终组合因子的真实表现。

## 9. 预期贡献

本文方法的贡献可以总结为三点：

1. 提出组合感知的 GFlowNet 因子搜索框架，使搜索目标从单因子 IC 转向组合因子的边际贡献。
2. 引入 residual IC 机制，鼓励模型发现已有因子无法解释的新信息。
3. 使用结构感知 action mask 辅助剪枝，减少数值不稳定和明显无效复杂表达式的搜索。

## 10. 简单总结

单独加 structure-aware action mask 不足以作为强论文创新。

更合理的方案是：

> 用 Combination-Aware Reward 作为主创新，用 Structure-Aware Action Mask 作为辅助搜索稳定模块。

最终目标不是找到一个训练 IC 很高的因子，而是找到一组：

- 有预测能力。
- 彼此互补。
- 低冗余。
- 组合后测试集表现更强。
- 泛化更稳定。

的 Alpha 因子。
