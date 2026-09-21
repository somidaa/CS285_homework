# Berkeley CS285 Homework 1 — Imitation Learning (Action Chunking)

Push-T 模仿学习：用 MSE 和 Flow-Matching 两种动作分块策略完成 T 型物体推送任务。
本仓库为本人从零实现的作业代码，包含完整可运行训练流程。

## 环境

- Python 3.11（uv 管理）
- PyTorch 2.x

```powershell
uv sync
```

## 运行

离线模式（不需要 wandb 账号）：

```powershell
$env:WANDB_MODE="offline"

# MSE 策略
uv run python src/hw1_imitation/train.py --policy-type mse

# Flow-Matching 策略
uv run python src/hw1_imitation/train.py --policy-type flow
```

## 实现内容

- `src/hw1_imitation/model.py`
  - `MSEPolicy`：MLP 直接输出扁平化动作块，MSE 损失拟合专家动作。
  - `FlowMatchingPolicy`：条件速度场，线性插值噪声→专家动作，欧拉积分去噪生成动作块。
- `src/hw1_imitation/train.py`：主训练循环（Adam、周期性评估、训练日志、最终评分导出）。

## 结果

| 策略 | 最终 eval/mean_reward | 要求 |
|---|---|---|
| MSE | **0.62** | ≥ 0.5 |
| Flow-Matching | **0.83** | ≥ 0.7 |

对比图见 `report/mse_vs_flow.png`。

## 目录

```
src/hw1_imitation/   # 模型与训练代码
scripts/             # 结果绘图脚本
report/              # 对比图与报告
```
