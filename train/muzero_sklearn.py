from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np


BALL_ORDER: List[str] = ["cue"] + [str(i) for i in range(1, 16)]


@dataclass(frozen=True)
class ActionSpace:
    v0: Tuple[float, ...]
    phi: Tuple[float, ...]
    theta: Tuple[float, ...]
    a: Tuple[float, ...]
    b: Tuple[float, ...]

    def all_actions(self) -> List[Dict[str, float]]:
        actions: List[Dict[str, float]] = []
        for v0 in self.v0:
            for phi in self.phi:
                for theta in self.theta:
                    for a in self.a:
                        for b in self.b:
                            actions.append({
                                "V0": float(v0),
                                "phi": float(phi),
                                "theta": float(theta),
                                "a": float(a),
                                "b": float(b),
                            })
        return actions

    def action_features(self, action: Dict[str, float]) -> np.ndarray:
        # Small continuous embedding for dynamics model
        v0 = (action["V0"] - 0.5) / (8.0 - 0.5)
        phi_rad = math.radians(action["phi"])
        theta = action["theta"] / 90.0
        a = (action["a"] + 0.5) / 1.0
        b = (action["b"] + 0.5) / 1.0
        return np.array([v0, math.sin(phi_rad), math.cos(phi_rad), theta, a, b], dtype=np.float32)


DEFAULT_ACTION_SPACE = ActionSpace(
    v0=(1.0, 1.8, 2.6, 3.6, 4.8, 6.2),
    phi=tuple(float(i) for i in range(0, 360, 15)),
    theta=(0.0, 10.0, 25.0),
    a=(0.0,),
    b=(0.0,),
)


def encode_observation(balls: Dict[str, object], my_targets: List[str]) -> np.ndarray:
    """Encode pooltool balls + my_targets into a fixed-size vector.

    Vector layout:
      - for each ball in BALL_ORDER: (x, y, pocketed)
      - last dim: is_solid (1 if targets include '1', else 0)
    """
    feat = np.zeros((len(BALL_ORDER) * 3 + 1,), dtype=np.float32)
    for i, bid in enumerate(BALL_ORDER):
        b = balls[bid]
        x = float(b.state.rvw[0][0])
        y = float(b.state.rvw[0][1])
        pocketed = 1.0 if int(b.state.s) == 4 else 0.0
        feat[i * 3 + 0] = x
        feat[i * 3 + 1] = y
        feat[i * 3 + 2] = pocketed

    is_solid = 1.0 if ("1" in set(my_targets)) else 0.0
    feat[-1] = is_solid
    return feat


def decode_next_obs(current_obs: np.ndarray, next_ball_feat: np.ndarray, continue_turn: float) -> np.ndarray:
    """Reconstruct next obs vector from predicted next ball features and a continue-turn scalar."""
    next_obs = np.zeros_like(current_obs)
    next_obs[:-1] = next_ball_feat
    cur_is_solid = float(current_obs[-1])
    if continue_turn >= 0.5:
        next_obs[-1] = cur_is_solid
    else:
        next_obs[-1] = 1.0 - cur_is_solid
    return next_obs


def compute_reward_norm(step_info: Dict, my_targets: List[str], balls_before: Dict[str, object]) -> float:
    """Reward shaped similarly to analyze_shot_for_reward, normalized to [-1, 1]."""
    me = step_info.get("ME_INTO_POCKET", []) or []
    enemy = step_info.get("ENEMY_INTO_POCKET", []) or []

    white = bool(step_info.get("WHITE_BALL_INTO_POCKET", False))
    black = bool(step_info.get("BLACK_BALL_INTO_POCKET", False))

    foul_first = bool(step_info.get("FOUL_FIRST_HIT", False))
    foul_no_rail = bool(step_info.get("NO_POCKET_NO_RAIL", False))
    no_hit = bool(step_info.get("NO_HIT", False))

    score = 0.0

    if white and black:
        score -= 150.0
    elif white:
        score -= 100.0
    elif black:
        remaining_own_before = [bid for bid in my_targets if balls_before[bid].state.s != 4]
        score += 100.0 if len(remaining_own_before) == 0 else -150.0

    if foul_first:
        score -= 30.0
    if foul_no_rail or no_hit:
        score -= 30.0

    score += 50.0 * float(len(me))
    score -= 20.0 * float(len(enemy))

    if (
        score == 0.0
        and (not white)
        and (not black)
        and (not foul_first)
        and (not foul_no_rail)
        and (not no_hit)
    ):
        score = 10.0

    score = float(np.clip(score / 150.0, -1.0, 1.0))
    return score
