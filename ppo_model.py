import torch
import torch.nn as nn
import torch.nn.functional as F

from obs_utils import obs_dim, action_dim


def _orthogonal_init(module: nn.Module, gain: float = 1.0):
    if isinstance(module, nn.Linear):
        nn.init.orthogonal_(module.weight, gain=gain)
        if module.bias is not None:
            nn.init.constant_(module.bias, 0.0)


def _make_mlp(
    in_dim: int,
    hidden_sizes: tuple[int, ...],
    activation: str = "tanh",
    layer_norm: bool = True,
) -> nn.Sequential:
    if activation.lower() == "tanh":
        act_layer: nn.Module = nn.Tanh()
        gain = nn.init.calculate_gain("tanh")
    elif activation.lower() in ("relu", "relu6"):
        act_layer = nn.ReLU()
        gain = nn.init.calculate_gain("relu")
    elif activation.lower() in ("silu", "swish"):
        act_layer = nn.SiLU()
        gain = 1.0
    else:
        raise ValueError(f"Unsupported activation: {activation}")

    layers: list[nn.Module] = []
    prev = in_dim
    for h in hidden_sizes:
        lin = nn.Linear(prev, h)
        _orthogonal_init(lin, gain=gain)
        layers.append(lin)
        if layer_norm:
            layers.append(nn.LayerNorm(h))
        layers.append(act_layer)
        prev = h
    return nn.Sequential(*layers)


class ActorCritic(nn.Module):
    """更稳健的 MLP Actor-Critic 网络（仍兼容原 get_dist_and_value 接口）。"""

    def __init__(
        self,
        hidden_size: int = 256,
        *,
        hidden_sizes: tuple[int, ...] | None = None,
        activation: str = "tanh",
        layer_norm: bool = True,
        shared_trunk: bool = False,
        log_std_init: float = -0.5,
        log_std_min: float = -5.0,
        log_std_max: float = 2.0,
    ):
        super().__init__()
        self.obs_dim = obs_dim()
        self.act_dim = action_dim()

        if hidden_sizes is None:
            hidden_sizes = (hidden_size, hidden_size)

        self.hidden_sizes = tuple(int(x) for x in hidden_sizes)
        self.activation = activation
        self.layer_norm = bool(layer_norm)
        self.shared_trunk = bool(shared_trunk)
        self.log_std_min = float(log_std_min)
        self.log_std_max = float(log_std_max)

        if self.shared_trunk:
            self.trunk = _make_mlp(self.obs_dim, self.hidden_sizes, activation=self.activation, layer_norm=self.layer_norm)
            last_dim = self.hidden_sizes[-1] if len(self.hidden_sizes) > 0 else self.obs_dim
            self.actor_mu = nn.Linear(last_dim, self.act_dim)
            self.critic_v = nn.Linear(last_dim, 1)
        else:
            self.actor_mlp = _make_mlp(self.obs_dim, self.hidden_sizes, activation=self.activation, layer_norm=self.layer_norm)
            self.critic_mlp = _make_mlp(self.obs_dim, self.hidden_sizes, activation=self.activation, layer_norm=self.layer_norm)
            last_dim = self.hidden_sizes[-1] if len(self.hidden_sizes) > 0 else self.obs_dim
            self.actor_mu = nn.Linear(last_dim, self.act_dim)
            self.critic_v = nn.Linear(last_dim, 1)

        # 输出层使用更小/更合适的初始化增益，减少一开始策略过激
        _orthogonal_init(self.actor_mu, gain=0.01)
        _orthogonal_init(self.critic_v, gain=1.0)

        self.log_std = nn.Parameter(torch.full((self.act_dim,), float(log_std_init)))

    def _forward_features(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.shared_trunk:
            feat = self.trunk(x)
            return feat, feat
        return self.actor_mlp(x), self.critic_mlp(x)

    def forward(self, x: torch.Tensor):
        actor_feat, critic_feat = self._forward_features(x)
        mu = self.actor_mu(actor_feat)
        value = self.critic_v(critic_feat).squeeze(-1)
        return mu, value

    def get_dist_and_value(self, obs: torch.Tensor):
        mu, value = self.forward(obs)
        log_std = torch.clamp(self.log_std, self.log_std_min, self.log_std_max)
        std = log_std.exp().expand_as(mu)
        dist = torch.distributions.Normal(mu, std)
        return dist, value


def save_checkpoint(model: ActorCritic, path: str):
    state = {
        "state_dict": model.state_dict(),
        "obs_dim": model.obs_dim,
        "act_dim": model.act_dim,
        "arch": "actor_critic_mlp_v2",
        "hidden_sizes": getattr(model, "hidden_sizes", None),
        "activation": getattr(model, "activation", None),
        "layer_norm": getattr(model, "layer_norm", None),
        "shared_trunk": getattr(model, "shared_trunk", None),
    }
    torch.save(state, path)


def _is_legacy_state_dict(state_dict: dict) -> bool:
    return "fc1.weight" in state_dict and "fc2.weight" in state_dict and "mu_head.weight" in state_dict


class _LegacyActorCritic(nn.Module):
    """旧版两层 ReLU 网络（仅用于加载旧 checkpoint）。"""

    def __init__(self, hidden_size: int):
        super().__init__()
        self.obs_dim = obs_dim()
        self.act_dim = action_dim()

        self.fc1 = nn.Linear(self.obs_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.mu_head = nn.Linear(hidden_size, self.act_dim)
        self.log_std = nn.Parameter(torch.zeros(self.act_dim))
        self.v_head = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        mu = self.mu_head(x)
        value = self.v_head(x).squeeze(-1)
        return mu, value

    def get_dist_and_value(self, obs: torch.Tensor):
        mu, value = self.forward(obs)
        std = self.log_std.exp().expand_as(mu)
        dist = torch.distributions.Normal(mu, std)
        return dist, value


def load_checkpoint(path: str, device: torch.device | None = None) -> ActorCritic:
    if device is None:
        device = torch.device("cpu")
    ckpt = torch.load(path, map_location=device)

    state_dict = ckpt["state_dict"]
    if _is_legacy_state_dict(state_dict):
        hidden_size = int(state_dict["fc1.weight"].shape[0])
        model_any: nn.Module = _LegacyActorCritic(hidden_size=hidden_size)
    else:
        hidden_sizes = ckpt.get("hidden_sizes")
        if isinstance(hidden_sizes, (list, tuple)) and len(hidden_sizes) > 0:
            hidden_sizes_tuple = tuple(int(x) for x in hidden_sizes)
        else:
            hidden_sizes_tuple = (256, 256)
        model_any = ActorCritic(
            hidden_sizes=hidden_sizes_tuple,
            activation=str(ckpt.get("activation") or "tanh"),
            layer_norm=bool(True if ckpt.get("layer_norm") is None else ckpt.get("layer_norm")),
            shared_trunk=bool(False if ckpt.get("shared_trunk") is None else ckpt.get("shared_trunk")),
        )

    if ckpt.get("obs_dim", None) is not None and ckpt["obs_dim"] != getattr(model_any, "obs_dim"):
        raise ValueError(f"obs_dim mismatch: ckpt {ckpt['obs_dim']} vs current {getattr(model_any, 'obs_dim')}")
    if ckpt.get("act_dim", None) is not None and ckpt["act_dim"] != getattr(model_any, "act_dim"):
        raise ValueError(f"act_dim mismatch: ckpt {ckpt['act_dim']} vs current {getattr(model_any, 'act_dim')}")

    model_any.load_state_dict(state_dict)
    model_any.to(device)
    model_any.eval()
    return model_any  # type: ignore[return-value]
