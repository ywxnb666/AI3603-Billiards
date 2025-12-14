from __future__ import annotations

import argparse
import math
import os
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
from sklearn.neural_network import MLPClassifier, MLPRegressor
import joblib

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
    n_games: int = 1000
    mcts_sims: int = 80
    c_puct: float = 1.25
    dirichlet_alpha: float = 0.3
    dirichlet_frac: float = 0.25
    top_k_expand: int = 24
    replay_max: int = 50000
    fit_every: int = 2
    hidden: Tuple[int, int] = (512, 512)
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


class ModelBundle:
    def __init__(self, obs_dim: int, n_actions: int, hidden: Tuple[int, int], seed: int):
        self.obs_dim = obs_dim
        self.n_actions = n_actions

        self.policy = MLPClassifier(
            hidden_layer_sizes=hidden,
            activation="relu",
            solver="adam",
            alpha=1e-4,
            batch_size=256,
            learning_rate_init=3e-4,
            max_iter=1,
            random_state=seed,
        )
        self.value = MLPRegressor(
            hidden_layer_sizes=hidden,
            activation="relu",
            solver="adam",
            alpha=1e-4,
            batch_size=256,
            learning_rate_init=3e-4,
            max_iter=1,
            random_state=seed,
        )
        # dynamics: (obs + act_feat) -> (next_ball_feat[48] + reward[1] + cont[1])
        self.dynamics = MLPRegressor(
            hidden_layer_sizes=hidden,
            activation="relu",
            solver="adam",
            alpha=1e-4,
            batch_size=256,
            learning_rate_init=3e-4,
            max_iter=1,
            random_state=seed,
        )

        self._initialized = False

    def ensure_initialized(self, rng: np.random.Generator):
        if self._initialized:
            return
        # One dummy fit to initialize sklearn internal structures
        X = rng.normal(size=(8, self.obs_dim)).astype(np.float32)
        y_cls = rng.integers(0, self.n_actions, size=(8,))
        self.policy.partial_fit(X, y_cls, classes=np.arange(self.n_actions, dtype=np.int64))

        y_val = rng.uniform(-0.05, 0.05, size=(8,)).astype(np.float32)
        self.value.partial_fit(X, y_val)

        Xd = rng.normal(size=(8, self.obs_dim + 6)).astype(np.float32)
        yd = rng.normal(size=(8, 50)).astype(np.float32)
        self.dynamics.partial_fit(Xd, yd)

        self._initialized = True

    def policy_probs(self, obs: np.ndarray) -> np.ndarray:
        p = self.policy.predict_proba(obs.reshape(1, -1))[0]
        return p.astype(np.float32)

    def value_pred(self, obs: np.ndarray) -> float:
        v = float(self.value.predict(obs.reshape(1, -1))[0])
        return float(np.clip(v, -1.0, 1.0))

    def step_model(self, obs: np.ndarray, act_feat: np.ndarray) -> Tuple[np.ndarray, float, float]:
        y = self.dynamics.predict(np.concatenate([obs, act_feat], axis=0).reshape(1, -1))[0]
        next_ball_feat = y[:48].astype(np.float32)
        reward = float(np.clip(y[48], -1.0, 1.0))
        cont = float(1.0 / (1.0 + np.exp(-float(y[49]))))
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
    parser.add_argument("--n_games", type=int, default=1000)
    parser.add_argument("--out", type=str, default=os.path.join("eval", "muzero_sklearn.joblib"))
    parser.add_argument("--silent", action="store_true")
    args = parser.parse_args()

    cfg = TrainConfig(n_games=args.n_games, silent_env=bool(args.silent))

    rng = np.random.default_rng(cfg.seed)
    env = PoolEnv()

    actions = DEFAULT_ACTION_SPACE.all_actions()
    action_feats = np.stack([DEFAULT_ACTION_SPACE.action_features(a) for a in actions], axis=0)

    obs_dim = len(actions[0])  # wrong, just placeholder
    obs_dim = 16 * 3 + 1
    model = ModelBundle(obs_dim=obs_dim, n_actions=len(actions), hidden=cfg.hidden, seed=cfg.seed)
    model.ensure_initialized(rng)

    replay = Replay(max_size=cfg.replay_max)

    for game_i in range(cfg.n_games):
        target_ball = "solid" if (game_i % 2 == 0) else "stripe"
        env.reset(target_ball=target_ball)

        traj = []  # store indices into replay to set value later

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
            traj.append(len(replay.obs) - 1)

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

        if (game_i + 1) % cfg.fit_every == 0 and len(replay.obs) >= 256:
            # Fit models on a random minibatch a few times
            for _ in range(100):
                obs_b, act_id_b, act_feat_b, reward_b, cont_b, next_ball_feat_b, value_b = replay.sample(512, rng)

                # policy
                model.policy.partial_fit(obs_b, act_id_b)

                # value
                model.value.partial_fit(obs_b, value_b)

                # dynamics
                Xd = np.concatenate([obs_b, act_feat_b], axis=1)
                yd = np.concatenate([
                    next_ball_feat_b,
                    reward_b.reshape(-1, 1),
                    cont_b.reshape(-1, 1),
                ], axis=1)
                model.dynamics.partial_fit(Xd, yd)

        if (game_i + 1) % 5 == 0:
            print(f"[train] games={game_i+1}/{cfg.n_games} replay={len(replay.obs)}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    joblib.dump(
        {
            "action_space": DEFAULT_ACTION_SPACE,
            "actions": actions,
            "policy": model.policy,
            "value": model.value,
            "dynamics": model.dynamics,
        },
        args.out,
    )
    print(f"[train] saved: {args.out}")


if __name__ == "__main__":
    main()
