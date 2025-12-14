# MuZero-lite for AI3603-Billiards

## 目标
实现一个可训练的 `NewAgent`，并在 `evaluate.py` 中与 `BasicAgent` 对战。训练阶段使用 MuZero 方法的核心结构：
- 学习一个模型（动力学 + 奖励）
- 学习 policy/value
- 用 MCTS 在模型上做规划来产生更好的自博弈数据

同时满足课程工程要求：
- 训练相关代码放在 `train/`
- 推理/评测阶段只依赖 `agent.py`（加载 `eval/` checkpoint）
- 训练阶段隐藏环境与物理引擎的终端输出

## 观测编码 (Observation)
环境给出的观测是 `(balls, my_targets, table)`，其中 `balls` 是 16 个球的物理状态。

为了让模型学习，我们把观测编码成固定维度向量：
- 按固定顺序 `cue, 1..15` 拼接每个球的 `(x, y, pocketed)`
- 额外拼接一个 `is_solid`（由 `my_targets` 是否包含 `'1'` 推断）

总维度：$16 \times 3 + 1 = 49$。

## 动作离散化 (Action Space)
原始动作为连续 5 维：`V0, phi, theta, a, b`。

为了能做 MCTS/分类策略网络，采用离散化：
- `V0` 取 6 个值
- `phi` 取 24 个方向（15° 间隔）
- `theta` 取 3 个值
- `a=b=0`（简化，后续可扩充）

总动作数：$6\times 24\times 3 = 432$。

## 模型结构（sklearn 版本）
由于当前环境未预装 PyTorch，本实现用 `scikit-learn` 的 MLP 近似 MuZero 的三部分：

1) **Policy 网络** `π(a|s)`
- `MLPClassifier(obs -> action_id)`

2) **Value 网络** `v(s)`
- `MLPRegressor(obs -> [-1,1])`
- 监督信号来自整局终局胜负（当前玩家视角：胜 +1 / 负 -1 / 平 0）

3) **Dynamics 网络** `g(s,a) -> (s', r, cont)`
- 输入：`obs` + `action_features`
- 输出：
  - 预测的下一步球状态特征（不含 `is_solid`）
  - 归一化 reward
  - `cont`（是否继续同一玩家出杆的概率）

`action_features` 使用连续嵌入（归一化速度、角度 sin/cos 等），避免 432 维 one-hot。

## MCTS 规划
在自博弈采样时，使用 MCTS 在**学习到的动力学模型**上搜索：
- 根节点用 policy 给 prior，并加入 Dirichlet 噪声鼓励探索
- 仅扩展 prior 最大的 `top_k_expand` 个动作，控制算力
- 选择策略用 PUCT：$Q + U$
- 回传时使用 `cont` 决定是否需要 value 符号翻转（换人则翻转）

这样在不直接调用环境做深层 rollout 的情况下，仍保留 MuZero 的“模型内规划”核心。

## 奖励设计
环境 `take_shot` 返回 `step_info`，其中包含进球、犯规等字段。

奖励函数尽量对齐 `agent.py` 的 `analyze_shot_for_reward` 逻辑：
- 进己方球：+50/球
- 进对方球：-20/球
- 白球进袋：-100
- 误进黑8：-150；合法进黑8：+100
- 首球犯规 / 无碰库犯规 / 未击中任何球：-30
- 合法无进球：+10

最后把 reward 除以 150 并裁剪到 [-1,1] 便于网络训练。

## 训练流程
训练脚本：[train/train_muzero.py](train/train_muzero.py)
- 自博弈 `n_games` 局
- 将 (obs, action, reward, cont, next_state, outcome) 写入 replay
- 每 `fit_every` 局在 replay 上抽样小批量，迭代更新三套 MLP
- 训练结束导出 checkpoint：`eval/muzero_sklearn.joblib`

## 静默训练输出
`poolenv.py` 在每杆击球会大量 `print`，训练时会淹没终端。

因此训练时将 `env.take_shot` 包在 [train/silent.py](train/silent.py) 的 `suppress_output()` 中，把 stdout/stderr 重定向到 `os.devnull`，保证训练日志只保留必要进度。

## 推理 (NewAgent)
`NewAgent` 在评测时加载 `eval/muzero_sklearn.joblib`：
- 用 policy 网络产生高概率候选动作
- （默认）直接选取概率最大动作
- （可选）做少量 MCTS（可通过环境变量开关）

该策略相比完全随机有可解释的“训练产物”，并且保持推理速度。

## 文件结构对齐
- `train/`: 训练代码与说明
- `eval/`: checkpoint 与评测说明
- `agent.py`: `NewAgent` 推理逻辑（满足“评测阶段只修改 agent.py/utils.py”约束）
