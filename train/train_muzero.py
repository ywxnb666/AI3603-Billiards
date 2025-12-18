from __future__ import annotations

import argparse
import math
import os
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:  # pragma: no cover - fall back if torch is unavailable
    from tensorboardX import SummaryWriter  # type: ignore

from poolenv import PoolEnv
from train.silent import suppress_output
from train.muzero_sklearn import (
    DEFAULT_ACTION_SPACE,
    encode_observation,
    decode_next_obs,
    compute_reward_norm,
)


@dataclass
class TrainConfig:
    n_games: int = 3000
    mcts_sims: int = 160
    c_puct: float = 1.25
    dirichlet_alpha: float = 0.15
    dirichlet_frac: float = 0.15
    top_k_expand: int = 48
    replay_max: int = 50000
    fit_every: int = 1
    fit_loops: int = 16
    batch_size: int = 384
    hidden: Tuple[int, int] = (512, 512)
    gamma: float = 0.97
    log_dir: str = os.path.join("runs", "muzero_sklearn")
    seed: int = 0
    silent_env: bool = True


class Replay:
    def __init__(self, max_size: int):
        self.max_size = max_size
        self.obs: List[np.ndarray] = []
        self.act_id: List[int] = []
        self.act_feat: List[np.ndarray] = []
        self.reward: List[float] = []
        self.cont: List[float] = []
        self.next_ball_feat: List[np.ndarray] = []
        self.value: List[float] = []

    def add(
        self,
        obs: np.ndarray,
        act_id: int,
        act_feat: np.ndarray,
        reward: float,
        cont: float,
        next_ball_feat: np.ndarray,
        value: float,
    ):
        self.obs.append(obs)
        self.act_id.append(act_id)
        self.act_feat.append(act_feat)
        self.reward.append(reward)
        self.cont.append(cont)
        self.next_ball_feat.append(next_ball_feat)
        self.value.append(value)

        if len(self.obs) > self.max_size:
            for arr in (
                self.obs,
                self.act_id,
                self.act_feat,
                self.reward,
                self.cont,
                self.next_ball_feat,
                self.value,
            ):
                del arr[0]

    def sample(self, batch: int, rng: np.random.Generator):
        n = len(self.obs)
        idx = rng.integers(0, n, size=(min(batch, n),))
        obs = np.stack([self.obs[i] for i in idx], axis=0)
        act_id = np.array([self.act_id[i] for i in idx], dtype=np.int64)
        act_feat = np.stack([self.act_feat[i] for i in idx], axis=0)
        reward = np.array([self.reward[i] for i in idx], dtype=np.float32)
        cont = np.array([self.cont[i] for i in idx], dtype=np.float32)
        next_ball_feat = np.stack([self.next_ball_feat[i] for i in idx], axis=0)
        value = np.array([self.value[i] for i in idx], dtype=np.float32)
        return obs, act_id, act_feat, reward, cont, next_ball_feat, value


def mirror_state_features(obs: np.ndarray) -> np.ndarray:
    """Mirror x-axis for all balls, keeping pocket flags and is_solid flag."""
    mirrored = obs.copy()
    for i in range(16):
        mirrored[i * 3] = -mirrored[i * 3]
    return mirrored


def mirror_action_feat(act_feat: np.ndarray) -> np.ndarray:
    """Mirror action embedding: flip cos(phi) and side-spin offset a."""
    mirrored = act_feat.copy()
    # act_feat: [v0, sin(phi), cos(phi), theta, a_norm, b_norm]
    mirrored[2] = -mirrored[2]  # cos flips when reflecting x
    mirrored[4] = 1.0 - mirrored[4]  # a_norm corresponds to [-0.5,0.5]
    return mirrored


def _make_mlp(input_dim: int, hidden: Tuple[int, int], output_dim: int) -> nn.Sequential:
    layers: List[nn.Module] = []
    last = input_dim
    for h in hidden:
        layers.extend([
            nn.Linear(last, h),
            nn.LayerNorm(h),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.1),
        ])
        last = h
    layers.append(nn.Linear(last, output_dim))
    return nn.Sequential(*layers)


class ModelBundle:
    def __init__(self, obs_dim: int, n_actions: int, hidden: Tuple[int, int], seed: int, device: str = "cpu"):
        self.obs_dim = obs_dim
        self.n_actions = n_actions
        self.device = torch.device(device)

        torch.manual_seed(seed)

        self.policy_net = _make_mlp(obs_dim, hidden, n_actions).to(self.device)
        self.value_net = _make_mlp(obs_dim, hidden, 1).to(self.device)
        # dynamics: (obs + act_feat) -> (next_ball_feat[48] + reward[1] + cont_logit[1])
        self.dynamics_net = _make_mlp(obs_dim + 6, hidden, 50).to(self.device)

        self.optimizer = Adam(
            list(self.policy_net.parameters())
            + list(self.value_net.parameters())
            + list(self.dynamics_net.parameters()),
            lr=3e-4,
            weight_decay=1e-5,
        )

    @torch.no_grad()
    def policy_probs(self, obs: np.ndarray) -> np.ndarray:
        x = torch.from_numpy(obs.astype(np.float32)).unsqueeze(0).to(self.device)
        logits = self.policy_net(x)
        p = torch.softmax(logits, dim=-1)[0].cpu().numpy().astype(np.float32)
        return p

    @torch.no_grad()
    def value_pred(self, obs: np.ndarray) -> float:
        x = torch.from_numpy(obs.astype(np.float32)).unsqueeze(0).to(self.device)
        v = self.value_net(x)[0, 0].clamp(-1.0, 1.0)
        return float(v.cpu().item())

    @torch.no_grad()
    def step_model(self, obs: np.ndarray, act_feat: np.ndarray) -> Tuple[np.ndarray, float, float]:
        x = torch.from_numpy(np.concatenate([obs, act_feat], axis=0).astype(np.float32)).unsqueeze(0).to(self.device)
        y = self.dynamics_net(x)[0]
        next_ball_feat = y[:48].cpu().numpy().astype(np.float32)
        reward = float(torch.tanh(y[48]).cpu().item())
        cont = float(torch.sigmoid(y[49]).cpu().item())
        next_obs = decode_next_obs(obs, next_ball_feat, cont)
        return next_obs, reward, cont


class Node:
    def __init__(self, obs: np.ndarray):
        self.obs = obs
        self.visit = 0
        self.value_sum = 0.0
        self.children = {}  # act_id -> (prior, reward, cont, child_node)

    @property
    def value(self) -> float:
        return 0.0 if self.visit == 0 else self.value_sum / self.visit


def mcts_select_action(
    root_obs: np.ndarray,
    model: ModelBundle,
    action_feats: np.ndarray,
    sims: int,
    c_puct: float,
    rng: np.random.Generator,
    dirichlet_alpha: float,
    dirichlet_frac: float,
    top_k_expand: int,
) -> Tuple[int, np.ndarray]:
    root = Node(root_obs)

    priors = model.policy_probs(root_obs)

    # Dirichlet noise at root
    noise = rng.dirichlet([dirichlet_alpha] * len(priors)).astype(np.float32)
    priors = (1.0 - dirichlet_frac) * priors + dirichlet_frac * noise

    # Expand root with top-k priors
    k = min(int(top_k_expand), len(priors))
    if k <= 0:
        return int(rng.integers(0, model.n_actions)), np.zeros((model.n_actions,), dtype=np.float32)
    top_ids = np.argpartition(-priors, k - 1)[:k]
    for act_id in top_ids:
        act_feat = action_feats[act_id]
        next_obs, reward, cont = model.step_model(root_obs, act_feat)
        child = Node(next_obs)
        root.children[int(act_id)] = (float(priors[act_id]), reward, cont, child)

    for _ in range(sims):
        node = root
        path = []  # (node, act_id)
        player_sign = 1.0

        while node.children:
            best_score = -1e9
            best_act = None
            best_child = None
            best_edge = None
            sqrt_parent = math.sqrt(node.visit + 1e-8)

            for act_id, (prior, reward, cont, child) in node.children.items():
                u = c_puct * prior * sqrt_parent / (1.0 + child.visit)
                q = child.value
                score = q + u
                if score > best_score:
                    best_score = score
                    best_act = act_id
                    best_child = child
                    best_edge = (reward, cont)

            path.append((node, best_act, best_edge))
            node = best_child
            # flip sign if turn switches
            _, cont = best_edge
            if cont < 0.5:
                player_sign *= -1.0

        leaf_v = model.value_pred(node.obs)

        # Backprop
        v = leaf_v
        for parent, act_id, (reward, cont) in reversed(path):
            # value from parent's perspective
            v = reward + (v if cont >= 0.5 else -v)
            child = parent.children[act_id][3]
            child.visit += 1
            child.value_sum += v
            parent.visit += 1

        root.visit += 1

    # Choose action by max visit
    visit_counts = np.zeros((model.n_actions,), dtype=np.int32)
    for act_id, (_, _, _, child) in root.children.items():
        visit_counts[act_id] = child.visit

    best_act = int(np.argmax(visit_counts))
    return best_act, visit_counts.astype(np.float32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_games", type=int, default=3000)
    parser.add_argument("--out", type=str, default=os.path.join("eval", "muzero_sklearn.pt"))
    parser.add_argument("--silent", action="store_true")
    parser.add_argument("--logdir", type=str, default=None, help="TensorBoard log directory")
    parser.add_argument("--fit-loops", type=int, default=None, help="Gradient steps per update block")
    parser.add_argument("--batch-size", type=int, default=None, help="Replay batch size")
    args = parser.parse_args()

    cfg = TrainConfig(
        n_games=args.n_games,
        silent_env=bool(args.silent),
        log_dir=args.logdir or TrainConfig.log_dir,
        fit_loops=args.fit_loops or TrainConfig.fit_loops,
        batch_size=args.batch_size or TrainConfig.batch_size,
    )

    writer = SummaryWriter(log_dir=cfg.log_dir)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    rng = np.random.default_rng(cfg.seed)
    env = PoolEnv()

    actions = DEFAULT_ACTION_SPACE.all_actions()
    action_feats = np.stack([DEFAULT_ACTION_SPACE.action_features(a) for a in actions], axis=0)

    obs_dim = 16 * 3 + 1
    model = ModelBundle(obs_dim=obs_dim, n_actions=len(actions), hidden=cfg.hidden, seed=cfg.seed, device=device)

    replay = Replay(max_size=cfg.replay_max)
    fit_step = 0

    for game_i in range(cfg.n_games):
        target_ball = "solid" if (game_i % 2 == 0) else "stripe"
        env.reset(target_ball=target_ball)

        traj = []  # store indices into replay to set value later
        game_return = 0.0

        while True:
            player = env.get_curr_player()
            balls, my_targets, _table = env.get_observation(player)
            obs = encode_observation(balls, my_targets)

            with suppress_output(cfg.silent_env):
                act_id, _visits = mcts_select_action(
                    root_obs=obs,
                    model=model,
                    action_feats=action_feats,
                    sims=cfg.mcts_sims,
                    c_puct=cfg.c_puct,
                    rng=rng,
                    dirichlet_alpha=cfg.dirichlet_alpha,
                    dirichlet_frac=cfg.dirichlet_frac,
                    top_k_expand=cfg.top_k_expand,
                )
                balls_before = {bid: balls[bid] for bid in balls}
                step_info = env.take_shot(actions[act_id])

            done, info = env.get_done()

            # continue_turn: compare player after shot
            next_player = env.get_curr_player() if not done else player
            cont = 1.0 if next_player == player else 0.0

            reward = compute_reward_norm(step_info, my_targets, balls_before)
            game_return += reward

            next_balls = step_info["BALLS"]
            next_player_obs = env.get_observation(next_player)[1] if not done else my_targets
            next_obs = encode_observation(next_balls, next_player_obs)

            replay.add(
                obs=obs,
                act_id=act_id,
                act_feat=action_feats[act_id],
                reward=reward,
                cont=cont,
                next_ball_feat=next_obs[:-1].copy(),
                value=0.0,
            )
            idx_main = len(replay.obs) - 1
            # 简单镜像增强：左右对称局面/动作
            mirrored_obs = mirror_state_features(obs)
            mirrored_next_obs = mirror_state_features(next_obs)
            mirrored_act_feat = mirror_action_feat(action_feats[act_id])
            replay.add(
                obs=mirrored_obs,
                act_id=act_id,
                act_feat=mirrored_act_feat,
                reward=reward,
                cont=cont,
                next_ball_feat=mirrored_next_obs[:-1].copy(),
                value=0.0,
            )
            traj.append(idx_main)

            if done:
                winner = info.get("winner")
                # assign outcome for each step from that step's player perspective
                for idx in traj:
                    step_is_solid = float(replay.obs[idx][-1])
                    # map winner to is_solid via target_ball
                    if winner == "SAME" or winner is None:
                        v = 0.0
                    else:
                        winner_is_solid = 1.0 if ((target_ball == "solid" and winner == "A") or (target_ball == "stripe" and winner == "B")) else 0.0
                        v = 1.0 if step_is_solid == winner_is_solid else -1.0
                    replay.value[idx] = float(v)
                break

        if (game_i + 1) % cfg.fit_every == 0 and len(replay.obs) >= cfg.batch_size:
            # Fit models on a random minibatch a few times
            for _ in range(cfg.fit_loops):
                obs_b, act_id_b, act_feat_b, reward_b, cont_b, next_ball_feat_b, value_b = replay.sample(cfg.batch_size, rng)

                obs_t = torch.from_numpy(obs_b).to(device)
                act_id_t = torch.from_numpy(act_id_b).long().to(device)
                act_feat_t = torch.from_numpy(act_feat_b).to(device)
                reward_t = torch.from_numpy(reward_b).to(device)
                cont_t = torch.from_numpy(cont_b).to(device)
                next_ball_feat_t = torch.from_numpy(next_ball_feat_b).to(device)

                logits = model.policy_net(obs_t)
                policy_ce = F.cross_entropy(logits, act_id_t)
                with torch.no_grad():
                    policy_acc = float((torch.argmax(logits, dim=1) == act_id_t).float().mean().cpu().item())

                value_pred = model.value_net(obs_t).squeeze(1).clamp(-1.0, 1.0)

                # Reconstruct next obs to bootstrap value
                cur_is_solid = obs_t[:, -1]
                next_is_solid = torch.where(cont_t >= 0.5, cur_is_solid, 1.0 - cur_is_solid)
                next_obs_t = torch.cat([next_ball_feat_t, next_is_solid.unsqueeze(1)], dim=1)
                with torch.no_grad():
                    next_value = model.value_net(next_obs_t).squeeze(1).clamp(-1.0, 1.0)
                value_target = reward_t + cfg.gamma * (cont_t * next_value + (1.0 - cont_t) * (-next_value))
                value_mse = F.mse_loss(value_pred, value_target)

                dyn_out = model.dynamics_net(torch.cat([obs_t, act_feat_t], dim=1))
                next_ball_pred = dyn_out[:, :48]
                reward_pred = torch.tanh(dyn_out[:, 48])
                cont_logit_pred = dyn_out[:, 49]
                cont_pred = torch.sigmoid(cont_logit_pred)

                dyn_ball_mse = F.mse_loss(next_ball_pred, next_ball_feat_t)
                reward_mse = F.mse_loss(reward_pred, reward_t)
                cont_bce = F.binary_cross_entropy_with_logits(cont_logit_pred, cont_t)

                loss = policy_ce + value_mse + dyn_ball_mse + reward_mse + cont_bce

                model.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.optimizer.param_groups[0]["params"], max_norm=5.0)
                model.optimizer.step()

                # TensorBoard logging
                writer.add_scalar("loss/total", float(loss.item()), fit_step)
                writer.add_scalar("loss/policy_ce", float(policy_ce.item()), fit_step)
                writer.add_scalar("metrics/policy_acc", policy_acc, fit_step)
                writer.add_scalar("loss/value_mse", float(value_mse.item()), fit_step)
                writer.add_scalar("loss/dyn_next_ball_mse", float(dyn_ball_mse.item()), fit_step)
                writer.add_scalar("loss/dyn_reward_mse", float(reward_mse.item()), fit_step)
                writer.add_scalar("loss/dyn_cont_bce", float(cont_bce.item()), fit_step)
                writer.add_scalar("replay/size", len(replay.obs), fit_step)

                fit_step += 1

        if (game_i + 1) % 5 == 0:
            print(f"[train] games={game_i+1}/{cfg.n_games} replay={len(replay.obs)}")

        writer.add_scalar("game/return", game_return, game_i)
        writer.add_scalar("game/len", len(traj), game_i)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    torch.save(
        {
            "action_space": DEFAULT_ACTION_SPACE,
            "actions": actions,
            "obs_dim": obs_dim,
            "hidden": cfg.hidden,
            "policy_state": model.policy_net.state_dict(),
            "value_state": model.value_net.state_dict(),
            "dynamics_state": model.dynamics_net.state_dict(),
        },
        args.out,
    )
    writer.flush()
    writer.close()
    print(f"[train] saved: {args.out}")


if __name__ == "__main__":
    main()
