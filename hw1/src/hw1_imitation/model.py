"""Model definitions for Push-T imitation policies."""

from __future__ import annotations

import abc
from typing import Literal, TypeAlias

import torch
from torch import nn


class BasePolicy(nn.Module, metaclass=abc.ABCMeta):
    """Base class for action chunking policies."""

    def __init__(self, state_dim: int, action_dim: int, chunk_size: int) -> None:
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.chunk_size = chunk_size

    @abc.abstractmethod
    def compute_loss(
        self, state: torch.Tensor, action_chunk: torch.Tensor
    ) -> torch.Tensor:
        """Compute training loss for a batch."""

    @abc.abstractmethod
    def sample_actions(
        self,
        state: torch.Tensor,
        *,
        num_steps: int = 10,  # only applicable for flow policy
    ) -> torch.Tensor:
        """Generate a chunk of actions with shape (batch, chunk_size, action_dim)."""


class MSEPolicy(BasePolicy):
    """Predicts action chunks with an MSE loss."""

    ### TODO: IMPLEMENT MSEPolicy HERE ###
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        chunk_size: int,
        hidden_dims: tuple[int, ...] = (256, 256, 256),
    ):
        super().__init__(state_dim, action_dim, chunk_size)

        # 根据 hidden_dims 元组逐层搭建 MLP
        layers: list[nn.Module] = []
        in_dim = state_dim
        for h in hidden_dims:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.ReLU())
            in_dim = h
        # 输出层：扁平化的完整动作块，维度 = chunk_size * action_dim
        out_dim = action_dim * chunk_size
        layers.append(nn.Linear(in_dim, out_dim))
        self.mlp = nn.Sequential(*layers)

    def compute_loss(
        self,
        state: torch.Tensor,
        action_chunk: torch.Tensor,
    ) -> torch.Tensor:
        # state: (B, state_dim)
        # action_chunk: (B, chunk_size, action_dim) —— 数据集直接给三维
        pred_chunk = self.mlp(state)  # (B, chunk_size * action_dim)
        # 把目标动作块拍平成和预测一致的二维再算 MSE
        target = action_chunk.reshape(pred_chunk.shape[0], -1)
        loss = torch.mean((pred_chunk - target) ** 2)
        return loss

    def sample_actions(
        self,
        state: torch.Tensor,
        *,
        num_steps: int = 10,  # MSE 策略不使用此参数，仅为与基类签名一致
    ) -> torch.Tensor:
        # 返回 (B, chunk_size, action_dim)，评估代码会用 action_chunk[chunk_index] 索引
        B = state.shape[0]
        pred_chunk = self.mlp(state)  # (B, chunk_size * action_dim)
        return pred_chunk.reshape(B, self.chunk_size, self.action_dim)


class FlowMatchingPolicy(BasePolicy):
    """Predicts action chunks with a flow matching loss."""

    ### TODO: IMPLEMENT FlowMatchingPolicy HERE ###
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        chunk_size: int,
        hidden_dims: tuple[int, ...] = (256, 256, 256),
    ) -> None:
        super().__init__(state_dim, action_dim, chunk_size)

        out_dim = action_dim * chunk_size
        in_dim = state_dim + out_dim + 1
        layers:list[nn.Module] = []
        d = in_dim
        for h in hidden_dims:
            layers.append(nn.Linear(d,h))
            layers.append(nn.ReLU())
            d = h
        layers.append(nn.Linear(d, out_dim))
        self.mlp = nn.Sequential(*layers)

    def compute_loss(
        self,
        state: torch.Tensor,
        action_chunk: torch.Tensor,
    ) -> torch.Tensor:

        B = state.shape[0]
        device = state.device
        out_dim = self.action_dim * self.chunk_size

        A = action_chunk.reshape(B, out_dim)
        A0 = torch.randn_like(A)
        tau = torch.rand(B, 1, device=device)
        A_tau = tau * A + (1.0 - tau) * A0

        v_pred = self.mlp(torch.cat([state, A_tau, tau], dim=-1))
        v_target = A - A0
        loss = torch.mean((v_pred - v_target) ** 2)
        return loss


    def sample_actions(
        self,
        state: torch.Tensor,
        *,
        num_steps: int = 10,
    ) -> torch.Tensor:

        B = state.shape[0]
        device = state.device
        out_dim = self.action_dim * self.chunk_size

        A = torch.randn(B, out_dim, device=device)
        dt = 1.0 / num_steps
        for i in range(num_steps):
            tau = i * dt
            tau_batch = torch.full((B,1), tau, device=device)
            v = self.mlp(torch.cat([state, A, tau_batch], dim=-1))
            A = A + dt * v

        return A.reshape(B, self.chunk_size, self.action_dim)


PolicyType: TypeAlias = Literal["mse", "flow"]


def build_policy(
    policy_type: PolicyType,
    *,
    state_dim: int,
    action_dim: int,
    chunk_size: int,
    hidden_dims: tuple[int, ...] = (128, 128),
) -> BasePolicy:
    if policy_type == "mse":
        return MSEPolicy(
            state_dim=state_dim,
            action_dim=action_dim,
            chunk_size=chunk_size,
            hidden_dims=hidden_dims,
        )
    if policy_type == "flow":
        return FlowMatchingPolicy(
            state_dim=state_dim,
            action_dim=action_dim,
            chunk_size=chunk_size,
            hidden_dims=hidden_dims,
        )
    raise ValueError(f"Unknown policy type: {policy_type}")
