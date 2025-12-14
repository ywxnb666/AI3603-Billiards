# Train (MuZero-lite, sklearn)

本目录提供一个**可复现训练流程**：用 MuZero 的结构（预测网络 + 学到的动力学模型 + MCTS）在台球对战环境上进行自博弈训练，并将 checkpoint 输出到 `eval/` 供 `NewAgent` 加载。

## 依赖
- Python 3.10+（你当前 Conda 环境即可）
- 需要 `scikit-learn`、`joblib`（本机 Anaconda 通常已包含）
- 需要项目依赖：`pooltool`、`bayesian-optimization` 等（已能运行 `evaluate.py` 则说明 OK）

## 训练命令
在项目根目录运行：

```bash
python -m train.train_muzero --n_games 30 --silent
```

参数：
- `--n_games`: 自博弈局数（越大训练越久）
- `--silent`: 训练阶段**屏蔽** `poolenv.py` 与物理仿真的大量终端输出

## 输出
训练完成后默认保存到：
- `eval/muzero_sklearn.joblib`

该文件包含：
- 离散动作表
- sklearn 的 policy/value/dynamics 三个模型

## 说明
这是一个“MuZero-lite”实现：
- 动作空间离散化（角度/速度等）
- 用 sklearn MLP 近似表示/动力学/预测网络
- 在 MCTS 中使用动力学模型进行一步展开与 value 回传

如果需要更强效果：可提高 `--n_games`，并在 [train/train_muzero.py](train/train_muzero.py) 中调大 `mcts_sims`/`top_k_expand` 与网络规模。
