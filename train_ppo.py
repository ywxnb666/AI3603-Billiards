"""PPO 训练脚本

在 PoolEnv 上训练 NewAgent 使用的策略网络。

注意：
- 训练时会对 poolenv 中的大量打印进行重定向，避免刷屏；
- 训练得到的 checkpoint 默认保存在 eval/newagent_ppo.pth，
  供 agent.NewAgent 在 evaluate.py 中加载使用。
"""

import sys
import os
# 强制控制台使用 UTF-8，避免在 Windows 上打印 unicode 出错（如 "⚪"）
if sys.platform.startswith("win"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        os.environ.setdefault("PYTHONIOENCODING", "utf-8")

from contextlib import redirect_stdout
import io
from io import StringIO

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import time
import traceback
import warnings

from poolenv import PoolEnv
from agent import Agent, BasicAgent
from obs_utils import encode_observation, squash_raw_action, scaled_action_from_unit
from ppo_model import ActorCritic, save_checkpoint




class RolloutBuffer:
    def __init__(self):
        self.states = []
        self.actions_raw = []
        self.rewards = []
        self.dones = []
        self.logprobs = []
        self.values = []

    def add(self, state, action_raw, reward, done, logprob, value):
        self.states.append(state)
        self.actions_raw.append(action_raw)
        self.rewards.append(reward)
        self.dones.append(done)
        self.logprobs.append(logprob)
        self.values.append(value)

    def clear(self):
        self.__init__()


def compute_returns_and_advantages(rewards, dones, values, gamma=0.99, lam=0.95):
    rewards = np.asarray(rewards, dtype=np.float32)
    dones = np.asarray(dones, dtype=np.float32)
    values = np.asarray(values, dtype=np.float32)

    advantages = np.zeros_like(rewards, dtype=np.float32)
    gae = 0.0
    next_value = 0.0
    for t in reversed(range(len(rewards))):
        mask = 1.0 - dones[t]
        delta = rewards[t] + gamma * next_value * mask - values[t]
        gae = delta + gamma * lam * mask * gae
        advantages[t] = gae
        next_value = values[t]
    returns = advantages + values
    return returns, advantages


def step_env_quiet(env: PoolEnv, action: dict, max_retries: int = 3):
    """只在这一击期间静音环境的打印，并对 pooltool 的 simulate 错误做重试/降级处理。"""
    with redirect_stdout(StringIO()):
        last_exc = None
        for attempt in range(max_retries):
            try:
                info = env.take_shot(action)
                return info
            except Exception as e:
                last_exc = e
                print(f"[train_ppo] simulate failed attempt {attempt+1}/{max_retries}: {e}", flush=True)
                time.sleep(0.2)
        # 所有重试失败：打印 traceback 并重置 env，返回空 dict（reward_from_step 可处理）
        print("[train_ppo] simulate failed after retries, resetting environment.", flush=True)
        traceback.print_exception(type(last_exc), last_exc, last_exc.__traceback__)
        try:
            env.reset()
        except Exception:
            pass
        return {}


def reward_from_step(step_info: dict, done: bool, info: dict | None, player: str) -> float:
    """基于环境返回的信息构造奖励函数。

    设计目标：
    - 强烈惩罚一切犯规，尤其是导致**立即判负**的犯规；
    - 适度鼓励进自己球、轻微惩罚帮对手进球；
    - 对单纯空杆给小的时间惩罚，鼓励积极寻找机会；
    - 在局终盘给出明显的胜负奖励/惩罚。
    """

    r = 0.0

    # 基础信息
    me = step_info.get("ME_INTO_POCKET", [])
    enemy = step_info.get("ENEMY_INTO_POCKET", [])
    white_in = step_info.get("WHITE_BALL_INTO_POCKET", False)
    black_in = step_info.get("BLACK_BALL_INTO_POCKET", False)
    foul_first_hit = step_info.get("FOUL_FIRST_HIT", False)
    foul_no_hit = step_info.get("NO_HIT", False)
    foul_no_pocket_no_rail = step_info.get("NO_POCKET_NO_RAIL", False)

    winner = info.get("winner") if (done and info is not None) else None
    lose = winner not in (None, "SAME", player)
    win = winner == player

    # 是否属于“黑8导致的立即判负”（包括白球+黑8同时落袋、提前打黑8等）
    immediate_loss_by_black = bool(black_in and lose)

    # 1) 进球相关奖励（鼓励进自己球，轻惩帮对手进球）
    if me:
        # 每打进一个自己的目标球奖励 2.0（略大一些）
        r += 2.0 * len(me)
    if enemy:
        # 帮对手进球：中等惩罚
        r -= 1.0 * len(enemy)

    # 2) 各类犯规惩罚
    # 普通犯规：明显加大惩罚力度
    if white_in:
        # 白球落袋：重罚
        r -= 8.0
    if foul_first_hit:
        # 首次碰撞错误目标球/黑八
        r -= 6.0
    if foul_no_hit:
        # 白球完全未碰到任何球
        r -= 6.0
    if foul_no_pocket_no_rail:
        # 无进球且母球和目标球均未碰库
        r -= 4.0

    # 3) 普通空杆的小惩罚（非犯规且没有进球）
    if not me and not enemy and not black_in:
        # 只要这一杆既没进球、也没打黑8，就给一点时间惩罚
        r -= 0.1

    # 4) 黑八相关 + 终局胜负奖励
    if done and info is not None:
        if immediate_loss_by_black:
            # 黑8相关的**立即判负**，给最高级别惩罚
            r -= 40.0
        else:
            # 非黑8导致的普通终局（比如被对手清台、或达到最大杆数等）
            if win:
                # 赢一局：较大正奖励
                r += 20.0
            elif lose:
                # 输一局：较大负奖励
                r -= 20.0

        # 若是合法清黑8获胜，再给一层额外奖励
        if black_in and win:
            r += 15.0

    return float(r)


def train_ppo(
    total_episodes: int = 500,
    batch_size: int = 4096,
    ppo_epochs: int = 5,
    clip_eps: float = 0.2,
    lr: float = 2e-4,
    gamma: float = 0.995,
    lam: float = 0.97,
    entropy_coef: float = 0.005,
    device: str = "cuda",
):
    device = torch.device(device)

    env = PoolEnv()
    # 训练时的固定对手：使用 agent.BasicAgent
    opponent = BasicAgent()

    # 从中后期开始使用“自我对弈”作为对手：
    # 前 selfplay_start_ep 局对手为随机；之后对手使用当前策略网络决策。
    selfplay_start_ep = int(total_episodes * 0.4)

    # 更稳健的网络结构：分离 actor/critic + LayerNorm + 正交初始化（见 ppo_model.ActorCritic）
    # 这里适度加深网络容量，通常会带来更好的拟合能力和更稳定的 early training。
    policy = ActorCritic(hidden_sizes=(256, 256, 256), activation="tanh", layer_norm=True, shared_trunk=False).to(device)
    optimizer = optim.Adam(policy.parameters(), lr=lr)

    buffer = RolloutBuffer()

    global_step = 0
    for ep in range(total_episodes):
        use_selfplay_opponent = ep >= selfplay_start_ep
        mode = "SELF-PLAY" if use_selfplay_opponent else "RANDOM-OPP"
        print(f"开始 Episode {ep+1}/{total_episodes}  (opponent={mode})", flush=True)
        target_ball = "solid" if ep % 2 == 0 else "stripe"
        env.reset(target_ball=target_ball)

        while True:
            player = env.get_curr_player()
            obs_tuple = env.get_observation(player)

            if player == "A":
                balls, my_targets, table = obs_tuple
                state_np = encode_observation(balls, my_targets, table)

                state_tensor = torch.tensor(state_np, dtype=torch.float32, device=device).unsqueeze(0)
                dist, value = policy.get_dist_and_value(state_tensor)
                raw_action = dist.rsample()
                logprob = dist.log_prob(raw_action).sum(dim=-1)

                raw_action_np = raw_action.squeeze(0).detach().cpu().numpy()
                # sanitize: 替换 nan/inf，并裁剪到合理范围，避免 quartic 求根时出现数值异常
                raw_action_np = np.nan_to_num(raw_action_np, nan=0.0, posinf=1e3, neginf=-1e3)
                raw_action_np = np.clip(raw_action_np, -10.0, 10.0)

                unit_action = squash_raw_action(raw_action_np)
                env_action = scaled_action_from_unit(unit_action)

                step_info = step_env_quiet(env, env_action)
                done, info = env.get_done()

                reward = reward_from_step(step_info, done, info if done else None, player="A")

                buffer.add(
                    state_np,
                    raw_action_np,
                    reward,
                    float(done),
                    logprob.item(),
                    value.item(),
                )
                global_step += 1

            else:
                # 对手回合：
                #   - 训练前期：使用 BasicAgent 作为固定对手；
                #   - 训练中后期：使用当前策略网络作为对手（自我对弈）。
                balls, my_targets, table = obs_tuple

                if not use_selfplay_opponent:
                    # BasicAgent 对手：直接基于物理仿真搜索动作（训练时静音其内部打印）
                    with redirect_stdout(StringIO()):
                        env_action = opponent.decision(balls, my_targets, table)
                else:
                    # 自我对弈对手：用当前 policy 根据对手视角的观测决策
                    with torch.no_grad():
                        opp_state_np = encode_observation(balls, my_targets, table)
                        opp_state_tensor = torch.tensor(
                            opp_state_np, dtype=torch.float32, device=device
                        ).unsqueeze(0)
                        opp_dist, _ = policy.get_dist_and_value(opp_state_tensor)
                        # 对手使用较“稳”的决策：取分布均值，而非采样
                        opp_raw_action = opp_dist.mean
                        opp_raw_action_np = (
                            opp_raw_action.squeeze(0).detach().cpu().numpy()
                        )
                        # 数值保护：避免 NaN / Inf 以及过大幅度
                        opp_raw_action_np = np.nan_to_num(
                            opp_raw_action_np,
                            nan=0.0,
                            posinf=1e3,
                            neginf=-1e3,
                        )
                        opp_raw_action_np = np.clip(opp_raw_action_np, -10.0, 10.0)

                        opp_unit_action = squash_raw_action(opp_raw_action_np)
                        env_action = scaled_action_from_unit(opp_unit_action)

                step_env_quiet(env, env_action)

            done, info = env.get_done()
            if done:
                break

        # 一局结束后，如缓冲足够大则进行一次 PPO 更新
        if len(buffer.states) >= batch_size:
            states = torch.tensor(np.asarray(buffer.states), dtype=torch.float32, device=device)
            actions_raw = torch.tensor(np.asarray(buffer.actions_raw), dtype=torch.float32, device=device)
            old_logprobs = torch.tensor(np.asarray(buffer.logprobs), dtype=torch.float32, device=device)
            values_np = np.asarray(buffer.values, dtype=np.float32)
            rewards_np = np.asarray(buffer.rewards, dtype=np.float32)
            dones_np = np.asarray(buffer.dones, dtype=np.float32)

            returns_np, adv_np = compute_returns_and_advantages(rewards_np, dones_np, values_np, gamma, lam)
            returns = torch.tensor(returns_np, dtype=torch.float32, device=device)
            advantages = torch.tensor(adv_np, dtype=torch.float32, device=device)
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            dataset_size = states.shape[0]
            for _ in range(ppo_epochs):
                idx = np.random.permutation(dataset_size)
                for start in range(0, dataset_size, 256):
                    end = start + 256
                    batch_idx = idx[start:end]
                    b_states = states[batch_idx]
                    b_actions_raw = actions_raw[batch_idx]
                    b_old_logprobs = old_logprobs[batch_idx]
                    b_returns = returns[batch_idx]
                    b_adv = advantages[batch_idx]

                    dist, value = policy.get_dist_and_value(b_states)
                    new_logprobs = dist.log_prob(b_actions_raw).sum(dim=-1)
                    entropy = dist.entropy().sum(dim=-1).mean()

                    ratio = (new_logprobs - b_old_logprobs).exp()
                    surr1 = ratio * b_adv
                    surr2 = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * b_adv
                    policy_loss = -torch.min(surr1, surr2).mean()
                    value_loss = nn.functional.mse_loss(value, b_returns)

                    loss = policy_loss + 0.5 * value_loss - entropy_coef * entropy

                    optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(policy.parameters(), max_norm=0.5)
                    optimizer.step()

            buffer.clear()

        if (ep + 1) % 10 == 0:
            print(f"[train_ppo] Episode {ep+1}/{total_episodes}, global_step={global_step}")

    # 训练结束后保存模型
    save_dir = os.path.join(os.path.dirname(__file__), "eval")
    os.makedirs(save_dir, exist_ok=True)
    ckpt_path = os.path.abspath(os.path.join(save_dir, "newagent_ppo.pth"))
    save_checkpoint(policy.cpu(), ckpt_path)
    print(f"[train_ppo] 训练完成，模型已保存到: {ckpt_path}")


if __name__ == "__main__":
    # 默认在 CPU 上跑一版示例训练；如需 GPU，可将 device 改为 "cuda"（前提是 PyTorch 安装正确）。
    train_ppo()
