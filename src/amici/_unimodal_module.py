"""Unimodal distance-bias AMICI module."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from scvi.module.base import auto_move_data

from ._module import AMICIModule


def _inverse_softplus(value: float) -> float:
    """Return a raw parameter value whose softplus is approximately ``value``."""
    if value <= 0:
        raise ValueError("Softplus target must be positive.")
    return math.log(math.expm1(value))


class AMICIUnimodalAttentionModule(AMICIModule):
    """AMICI module with a learned unimodal Huber distance bias.

    The distance bias is added to attention logits and is maximal at a learned
    preferred radius. This avoids exponentials while allowing attention to favor
    intermediate-range neighborhoods rather than strictly decaying with distance.
    """

    def __init__(
        self,
        *args,
        unimodal_preferred_radius_init: float = 10.0,
        unimodal_alpha_init: float = 0.01,
        unimodal_huber_delta: float = 5.0,
        unimodal_radius_offset: float = 0.0,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.unimodal_preferred_radius_init = unimodal_preferred_radius_init
        self.unimodal_alpha_init = unimodal_alpha_init
        self.unimodal_huber_delta = unimodal_huber_delta
        self.unimodal_radius_offset = unimodal_radius_offset

        radius_init = unimodal_preferred_radius_init / self.distance_kernel_unit_scale
        self.preferred_radius_raw = nn.Parameter(
            torch.full((self.n_labels, self.n_heads), _inverse_softplus(radius_init))
        )
        self.radius_alpha_raw = nn.Parameter(
            torch.full((self.n_labels, self.n_heads), _inverse_softplus(unimodal_alpha_init))
        )

    def _compute_unimodal_distance_bias(self, labels: torch.Tensor, nn_dist: torch.Tensor) -> dict[str, torch.Tensor]:
        """Compute a per-neighbor Huber radial bias for attention logits."""
        label_idx = labels.squeeze(-1)
        preferred_radius = F.softplus(self.preferred_radius_raw[label_idx]) + self.unimodal_radius_offset
        radius_alpha = F.softplus(self.radius_alpha_raw[label_idx])
        scaled_dist = nn_dist / self.distance_kernel_unit_scale
        distance_error = scaled_dist.unsqueeze(-1) - preferred_radius.unsqueeze(1)
        distance_penalty = F.huber_loss(
            distance_error,
            torch.zeros_like(distance_error),
            reduction="none",
            delta=self.unimodal_huber_delta,
        )
        distance_bias = -radius_alpha.unsqueeze(1) * distance_penalty
        return {
            "distance_bias": distance_bias,
            "preferred_radius": preferred_radius,
            "radius_alpha": radius_alpha,
            "distance_penalty": distance_penalty,
        }

    @auto_move_data
    def generative(
        self,
        labels,
        label_embed,
        nn_embed,
        nn_dist,
        return_attention_patterns: bool = False,
        return_attention_scores: bool = False,
        return_v: bool = False,
        attention_mask=None,
    ):
        return_attention_patterns = self.attention_penalty_coef > 0.0 or return_attention_patterns
        return_v = self.value_l1_penalty_coef > 0.0 or return_v

        query_embed = self.query_embed(label_embed)
        query_embed = rearrange(query_embed, "b (h d) -> b 1 h d", h=self.n_heads)

        unimodal_out = self._compute_unimodal_distance_bias(labels, nn_dist)

        kv_embed = self.kv_embed(nn_embed)
        kv_embed = rearrange(kv_embed, "b n (h d) -> b n h d", h=self.n_heads)
        if self.training and self.neighbor_dropout > 0.0:
            dropout_mask = (
                torch.rand((kv_embed.shape[0], kv_embed.shape[1]), device=kv_embed.device) > self.neighbor_dropout
            ).int()
            attention_mask = dropout_mask if attention_mask is None else attention_mask * dropout_mask

        attn_outs = self.attention_layer(
            query_embed,
            kv_embed,
            kv_embed,
            attention_mask=attention_mask,
            pos_attn_score=unimodal_out["distance_bias"],
            return_base_attn_scores=return_attention_scores,
            return_attn_patterns=return_attention_patterns,
            return_v=return_v,
        )
        residual_embed = rearrange(attn_outs["x"], "b 1 d -> b d")

        attention_scores = None
        if return_attention_scores:
            attention_scores = attn_outs["base_attn_scores"][:, :, 0, :]

        attention_patterns = None
        if return_attention_patterns:
            attention_patterns = attn_outs["attn_patterns"][:, :, 0, :]

        attention_v = None
        if return_v:
            attention_v = attn_outs["v"]

        residual = self.hook_final_residual(self.linear_head(residual_embed).float())
        batch_ct_means = self.ct_profiles[labels.squeeze(-1)].squeeze()
        prediction = (batch_ct_means + residual).float()

        return {
            "residual_embed": residual_embed,
            "residual": residual,
            "prediction": prediction,
            "attention_scores": attention_scores,
            "attention_patterns": attention_patterns,
            "attention_v": attention_v,
            "pos_coefs": unimodal_out["radius_alpha"],
            "preferred_radius": unimodal_out["preferred_radius"] * self.distance_kernel_unit_scale,
            "radius_alpha": unimodal_out["radius_alpha"],
            "distance_penalty": unimodal_out["distance_penalty"],
        }


UnimodalAMICIModule = AMICIUnimodalAttentionModule
