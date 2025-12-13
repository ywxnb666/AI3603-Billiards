import numpy as np


BALL_ORDER = ["cue"] + [str(i) for i in range(1, 16)]
ACTION_KEYS = ["V0", "phi", "theta", "a", "b"]

ACTION_BOUNDS = {
    "V0": (0.5, 8.0),
    "phi": (0.0, 360.0),
    "theta": (0.0, 90.0),
    "a": (-0.5, 0.5),
    "b": (-0.5, 0.5),
}


def obs_dim() -> int:
    """返回观测向量维度。

    当前设计：每个球 4 维特征 (x_norm, y_norm, pocketed, is_my_target) × 16 个球。
    """

    return len(BALL_ORDER) * 4


def action_dim() -> int:
    """返回动作向量维度（连续 5 维）。"""

    return len(ACTION_KEYS)


def encode_observation(balls, my_targets, table) -> np.ndarray:
    """将环境返回的 (balls, my_targets, table) 编码为固定长度向量。

    - 位置按桌面尺寸归一化，便于网络学习
    - pocketed: 是否进袋
    - is_my_target: 是否为我方目标球
    """

    features = []
    table_l = float(getattr(table, "l", 1.0)) or 1.0
    table_w = float(getattr(table, "w", 1.0)) or 1.0

    for bid in BALL_ORDER:
        ball = balls[bid]
        pos = ball.state.rvw[0]
        x = float(pos[0]) / table_l
        y = float(pos[1]) / table_w
        pocketed = 1.0 if int(ball.state.s) == 4 else 0.0
        is_my = 1.0 if bid in my_targets else 0.0
        features.extend([x, y, pocketed, is_my])

    return np.asarray(features, dtype=np.float32)


def squash_raw_action(raw_action: np.ndarray) -> np.ndarray:
    """将未约束的动作通过 tanh 压缩到 [-1, 1] 区间。"""

    return np.tanh(raw_action)


def scaled_action_from_unit(unit_action: np.ndarray) -> dict:
    """将 [-1, 1]^5 的向量映射到环境动作空间并返回 dict。"""

    unit_action = np.asarray(unit_action, dtype=np.float32)
    unit_action = np.clip(unit_action, -1.0, 1.0)

    action = {}
    for i, key in enumerate(ACTION_KEYS):
        low, high = ACTION_BOUNDS[key]
        u = float(unit_action[i])
        val = low + (u + 1.0) * 0.5 * (high - low)
        action[key] = float(val)
    return action


def unit_action_from_env(action: dict) -> np.ndarray:
    """可选：将环境动作反推到 [-1,1] 空间（目前训练中不强制使用）。"""

    vals = []
    for key in ACTION_KEYS:
        low, high = ACTION_BOUNDS[key]
        v = float(action[key])
        u = (v - low) * 2.0 / (high - low) - 1.0
        vals.append(u)
    return np.asarray(vals, dtype=np.float32)
