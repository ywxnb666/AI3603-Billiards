import torch
import torch.nn as nn
import torch.nn.functional as F

from obs_utils import obs_dim, action_dim


class ActorCritic(nn.Module):
    """简单 MLP 结构的 Actor-Critic 网络。"""

    def __init__(self, hidden_size: int = 256):
        super().__init__()
        self.obs_dim = obs_dim()
        self.act_dim = action_dim()

        self.fc1 = nn.Linear(self.obs_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)

        self.mu_head = nn.Linear(hidden_size, self.act_dim)
        self.log_std = nn.Parameter(torch.zeros(self.act_dim))

        self.v_head = nn.Linear(hidden_size, 1)

    def forward(self, x):
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


def save_checkpoint(model: ActorCritic, path: str):
    state = {
        "state_dict": model.state_dict(),
        "obs_dim": model.obs_dim,
        "act_dim": model.act_dim,
    }
    torch.save(state, path)


def load_checkpoint(path: str, device: torch.device | None = None) -> ActorCritic:
    if device is None:
        device = torch.device("cpu")
    ckpt = torch.load(path, map_location=device)
    model = ActorCritic()
    if ckpt.get("obs_dim", None) is not None and ckpt["obs_dim"] != model.obs_dim:
        raise ValueError(f"obs_dim mismatch: ckpt {ckpt['obs_dim']} vs current {model.obs_dim}")
    if ckpt.get("act_dim", None) is not None and ckpt["act_dim"] != model.act_dim:
        raise ValueError(f"act_dim mismatch: ckpt {ckpt['act_dim']} vs current {model.act_dim}")
    model.load_state_dict(ckpt["state_dict"])
    model.to(device)
    model.eval()
    return model
