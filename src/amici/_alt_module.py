import math
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, reduce, repeat
from scvi import REGISTRY_KEYS
from scvi.module.base import BaseModuleClass, LossOutput, auto_move_data
from torch import Tensor
from transformer_lens.hook_points import HookedRootModule, HookPoint

from ._components import AttentionBlock, ResNetMLP
from ._constants import NN_REGISTRY_KEYS


class SpatialPositionalEncoding(nn.Module):
    """Sinusoidal distance embedding used by the older unconstrained AMICI model."""

    def __init__(
        self,
        ndim: int,
        kernel: Literal["identity", "rbf"] = "identity",
        **kernel_kwargs,
    ):
        super().__init__()
        self.ndim = ndim
        if kernel == "identity":
            self.kernel = lambda dist: dist
        elif kernel == "rbf":
            sigma_sq = kernel_kwargs.get("sigma_sq", 1.0)
            self.kernel = lambda dist: 100 * (1 - torch.exp(-(dist**2) / (2 * sigma_sq)))
        else:
            raise ValueError(f"Unrecognized positional encoding kernel {kernel}.")

    @torch.no_grad()
    def compute_pe(self, nn_dist: Tensor) -> Tensor:
        position = rearrange(self.kernel(nn_dist), "batch neighbor -> batch neighbor 1")
        div_term = torch.exp(torch.arange(0, self.ndim, 2, device=nn_dist.device) * (-math.log(10000.0) / self.ndim))
        pe = torch.zeros(nn_dist.size(0), nn_dist.size(1), self.ndim, device=nn_dist.device)
        pe[:, :, 0::2] = torch.sin(position * rearrange(div_term, "embed -> 1 1 embed"))
        pe[:, :, 1::2] = torch.cos(position * rearrange(div_term, "embed -> 1 1 embed"))
        return pe

    def forward(self, nn_dist: Tensor) -> Tensor:
        return self.compute_pe(nn_dist)


class AMICIUnconstrainedAttentionModule(HookedRootModule, BaseModuleClass):
    """Older AMICI module with distance positional embeddings in unconstrained attention keys.

    This intentionally keeps the old positional-embedding architecture separate from the
    current constrained distance-kernel module. It does not include the abandoned histology
    branch from the old fork.
    """

    def __init__(
        self,
        n_genes: int,
        n_labels: int,
        empirical_ct_means: torch.Tensor,
        n_label_embed: int = 32,
        n_pe_dim: int = 256,
        n_pe_label_embed: int = 256,
        n_pe_label_hidden: int = 512,
        n_query_embed_hidden: int = 512,
        n_query_dim: int = 64,
        n_nn_embed: int = 256,
        n_nn_embed_hidden: int = 1024,
        n_head_size: int = 16,
        n_heads: int = 4,
        neighbor_dropout: float = 0.1,
        attention_dummy_score: float = 3.0,
        attention_penalty_coef: float = 0.0,
        value_l1_penalty_coef: float = 0.0,
        value_embed_l1_penalty_coef: float | None = None,
        residual_l2_penalty_coef: float = 0.0,
        ct_means_l2_penalty_coef: float = 0.0,
        add_res_connection: bool = False,
        use_empirical_ct_means: bool = True,
        positional_encoding_kernel: Literal["identity", "rbf"] = "identity",
    ):
        super().__init__()
        self.n_genes = n_genes
        self.n_labels = n_labels
        self.n_label_embed = n_label_embed
        self.n_pe_dim = n_pe_dim
        self.n_pe_label_embed = n_pe_label_embed
        self.n_pe_label_hidden = n_pe_label_hidden
        self.n_query_embed_hidden = n_query_embed_hidden
        self.n_query_dim = n_query_dim
        self.n_nn_embed = n_nn_embed
        self.n_nn_embed_hidden = n_nn_embed_hidden
        self.n_head_size = n_head_size
        self.n_heads = n_heads
        self.attention_dummy_score = attention_dummy_score
        self.neighbor_dropout = neighbor_dropout
        self.attention_penalty_coef = attention_penalty_coef
        self.value_l1_penalty_coef = (
            value_l1_penalty_coef if value_embed_l1_penalty_coef is None else value_embed_l1_penalty_coef
        )
        self.residual_l2_penalty_coef = residual_l2_penalty_coef
        self.ct_means_l2_penalty_coef = ct_means_l2_penalty_coef
        self.add_res_connection = add_res_connection
        self.empirical_ct_means = empirical_ct_means
        self.use_empirical_ct_means = use_empirical_ct_means

        if self.use_empirical_ct_means:
            self.register_buffer("ct_profiles", self.empirical_ct_means)
        else:
            self.ct_profiles = nn.Parameter(self.empirical_ct_means.clone().detach())

        self.ct_embed = nn.Embedding(self.n_labels, self.n_label_embed)
        self.query_embed = ResNetMLP(
            n_input=self.n_label_embed,
            n_output=self.n_query_dim,
            n_layers=2,
            n_hidden=self.n_query_embed_hidden,
            dropout=0.0,
        )
        self.nn_embed = ResNetMLP(
            n_input=self.n_genes,
            n_output=self.n_nn_embed,
            n_layers=2,
            n_hidden=self.n_nn_embed_hidden,
            dropout=0.0,
        )
        self.spatial_pe = SpatialPositionalEncoding(ndim=self.n_pe_dim, kernel=positional_encoding_kernel)
        self.pe_label_embed = ResNetMLP(
            n_input=self.n_pe_dim + self.n_label_embed,
            n_output=self.n_pe_label_embed,
            n_layers=2,
            n_hidden=self.n_pe_label_hidden,
            dropout=0.0,
        )

        self.key_nn_embed = nn.Linear(
            self.n_nn_embed + self.n_pe_label_embed,
            self.n_nn_embed + self.n_pe_label_embed,
            bias=False,
        )
        self.value_nn_embed = nn.Linear(
            self.n_nn_embed,
            self.n_nn_embed + self.n_pe_label_embed,
            bias=False,
        )
        self.attention_layer = AttentionBlock(
            self.n_query_dim,
            self.n_nn_embed + self.n_pe_label_embed,
            self.n_head_size,
            self.n_heads,
            dummy_attn_score=self.attention_dummy_score,
            add_res_connection=self.add_res_connection,
        )
        self.linear_head = nn.Linear(self.n_heads * self.n_head_size, self.n_genes, bias=False)

        self.hook_label_embed = HookPoint()
        self.hook_nn_embed = HookPoint()
        self.hook_final_residual = HookPoint()
        self.hook_pe_embed = HookPoint()

        self.setup()

    def _get_inference_input(self, tensors):
        labels = tensors[REGISTRY_KEYS.LABELS_KEY]
        nn_X = tensors[NN_REGISTRY_KEYS.NN_X_KEY]
        return {"labels": labels, "nn_X": nn_X}

    def inference(self, labels, nn_X):
        label_embed = self.hook_label_embed(rearrange(self.ct_embed(labels), "batch 1 embed -> batch embed"))
        nn_embed = self.hook_nn_embed(self.nn_embed(nn_X))
        return {"label_embed": label_embed, "nn_embed": nn_embed}

    def _get_generative_input(self, tensors, inference_outputs):
        labels = tensors[REGISTRY_KEYS.LABELS_KEY]
        label_embed = inference_outputs["label_embed"]
        nn_embed = inference_outputs["nn_embed"]
        nn_dist = tensors[NN_REGISTRY_KEYS.NN_DIST_KEY]
        return {
            "labels": labels,
            "label_embed": label_embed,
            "nn_embed": nn_embed,
            "nn_dist": nn_dist,
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
        return_value_embeds: bool = False,
        attention_mask=None,
    ):
        return_attention_patterns = self.attention_penalty_coef > 0.0 or return_attention_patterns
        return_v = self.value_l1_penalty_coef > 0.0 or return_v or return_value_embeds

        query_embed = self.query_embed(label_embed)
        query_embed = repeat(query_embed, "batch embed -> batch 1 head embed", head=self.n_heads)

        spatial_pe = self.spatial_pe(nn_dist)
        spatial_pe_reshaped = rearrange(spatial_pe, "batch neighbor embed -> (batch neighbor) embed")
        label_embed_reshaped = repeat(
            label_embed, "batch embed -> (batch neighbor) embed", neighbor=spatial_pe.shape[1]
        )
        pe_label = torch.cat((label_embed_reshaped, spatial_pe_reshaped), dim=-1)
        pe_label_embed = self.hook_pe_embed(self.pe_label_embed(pe_label))
        pe_label_embed = rearrange(
            pe_label_embed,
            "(batch neighbor) embed -> batch neighbor embed",
            batch=spatial_pe.shape[0],
            neighbor=spatial_pe.shape[1],
        )

        key_embed = self.key_nn_embed(torch.cat((nn_embed, pe_label_embed), dim=-1))
        value_embed = self.value_nn_embed(nn_embed)
        key_embed = repeat(key_embed, "batch neighbor embed -> batch neighbor head embed", head=self.n_heads)
        value_embed = repeat(value_embed, "batch neighbor embed -> batch neighbor head embed", head=self.n_heads)

        if self.training and self.neighbor_dropout > 0.0:
            dropout_mask = (
                torch.rand((key_embed.shape[0], key_embed.shape[1]), device=key_embed.device) > self.neighbor_dropout
            ).int()
            attention_mask = dropout_mask if attention_mask is None else attention_mask * dropout_mask

        attn_outs = self.attention_layer(
            query_embed,
            key_embed,
            value_embed,
            attention_mask=attention_mask,
            return_base_attn_scores=return_attention_scores,
            return_attn_patterns=return_attention_patterns,
            return_v=return_v,
        )
        residual_embed = rearrange(attn_outs["x"], "batch 1 embed -> batch embed")

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
            "pos_coefs": None,
        }

    def loss(self, tensors, inference_outputs, generative_outputs, kl_weight=1.0):
        """Loss computation."""
        true_X = tensors[REGISTRY_KEYS.X_KEY]
        prediction = generative_outputs["prediction"]
        reconstruction_loss = F.gaussian_nll_loss(
            prediction,
            true_X,
            var=torch.ones_like(prediction),
            reduction="none",
        ).sum(-1)

        attention_penalty = torch.zeros(true_X.shape[0], device=true_X.device)
        if self.attention_penalty_coef > 0.0:
            attention_patterns = generative_outputs["attention_patterns"]
            eps = torch.finfo(attention_patterns.dtype).eps
            attention_entropy_terms = (
                -1 * attention_patterns * torch.log(torch.clamp(attention_patterns, min=eps, max=1 - eps))
            )
            attention_penalty = reduce(
                reduce(attention_entropy_terms, "batch head_index key_pos -> batch head_index", "sum"),
                "batch head_index -> batch",
                "mean",
            )

        value_l1_penalty = torch.zeros(true_X.shape[0], device=true_X.device)
        if self.value_l1_penalty_coef > 0.0:
            attention_v = generative_outputs["attention_v"]
            value_l1_penalty = reduce(
                reduce(torch.abs(attention_v), "batch key_pos head_index head_size -> batch key_pos", "sum"),
                "batch key_pos -> batch",
                "mean",
            )

        residual_l2_penalty = torch.zeros(true_X.shape[0], device=true_X.device)
        if self.residual_l2_penalty_coef > 0.0:
            residual_l2_penalty = reduce(generative_outputs["residual"] ** 2, "batch gene -> batch", "mean")

        ct_means_l2_penalty = torch.zeros(self.n_labels, device=true_X.device)
        if not self.use_empirical_ct_means and self.ct_means_l2_penalty_coef > 0.0:
            ct_means_shift = self.ct_profiles - self.empirical_ct_means.to(self.ct_profiles.device)
            ct_means_l2_penalty = reduce(ct_means_shift**2, "label gene -> label", "sum")

        loss = torch.mean(
            reconstruction_loss
            + self.attention_penalty_coef * attention_penalty
            + self.value_l1_penalty_coef * value_l1_penalty
            + self.residual_l2_penalty_coef * residual_l2_penalty
        )
        loss += torch.mean(self.ct_means_l2_penalty_coef * ct_means_l2_penalty)

        return LossOutput(
            loss=loss,
            reconstruction_loss=reconstruction_loss,
            kl_local={
                "attention_penalty": self.attention_penalty_coef * attention_penalty,
                "value_l1_penalty": self.value_l1_penalty_coef * value_l1_penalty,
                "residual_l2_penalty": self.residual_l2_penalty_coef * residual_l2_penalty,
                "ct_means_l2_penalty": torch.sum(self.ct_means_l2_penalty_coef * ct_means_l2_penalty),
            },
            extra_metrics={"attention_penalty_coef": torch.tensor(self.attention_penalty_coef)},
        )


AltAMICIModule = AMICIUnconstrainedAttentionModule
