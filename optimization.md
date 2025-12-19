# 台球 MuZero 调优记录

## 主要改动
- **动作空间**：更细力度网格、10° 水平角、3 档抬杆，并加入侧旋/上旋偏移（a,b ∈ {-0.25,0,0.25}）。
- **搜索**：MCTS 模拟 160 次，根节点扩展 top_k=48，Dirichlet 噪声减小（α=0.15，frac=0.15）；推理候选 top_k=64，评估最多 48 个。
- **训练循环**：默认 3000 局自博弈，每局都训练，batch 384，梯度步 16；折扣 γ=0.97 做 TD 价值目标。
- **模型**：改用 Torch MLP（2×512，LayerNorm + dropout）用于 policy/value/dynamics；梯度裁剪 5.0。
- **数据**：增加左右镜像增强（obs、action、next_obs 同步镜像）。
- **检查点**：使用 Torch `.pt` 存储于 `eval/muzero_sklearn.pt`，包含 state_dict 和动作列表，NewAgent 已指向此路径。

## 重新训练
```bash
python train/train_muzero.py --n_games 3000 --out eval/muzero_sklearn.pt
```
- 有 GPU 则用 CUDA，否则 CPU。
- TensorBoard 日志：默认写入 `runs/muzero_pytorch/<时间戳>/`（可用 `--logdir` 覆盖）。

## 评估方式
- `NewAgent` 会加载 `eval/muzero_sklearn.pt`，用 Torch policy 生成候选并配合物理评分。
- 训练完成后直接运行现有 `evaluate.py` 与 `BasicAgent` 对打比较。

## 备注
- 价值目标：`reward + γ * (cont * V(next) + (1-cont) * (-V(next)))`。
- 动力学头预测：下一步球特征（48）、奖励（tanh）、是否继续击球（sigmoid）。
- 旧的 joblib 模型不再被 NewAgent 使用，如需旧版请重新训练生成新的 `.pt`。
