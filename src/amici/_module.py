import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, reduce, repeat
from scvi import REGISTRY_KEYS
from scvi.module.base import BaseModuleClass, LossOutput, auto_move_data
from transformer_lens.hook_points import HookedRootModule, HookPoint

from ._components import AttentionBlock, ResNetMLP
from ._constants import NN_REGISTRY_KEYS


class AMICIModule(HookedRootModule, BaseModuleClass):
    def __init__(
        self,
        n_genes: int,
        n_labels: int,
        empirical_ct_means: torch.Tensor,
        n_label_embed: int = 32,
        n_kv_dim: int = 256,
        n_query_embed_hidden: int = 512,
        n_query_dim: int = 64,
        n_nn_embed: int = 256,
        n_nn_embed_hidden: int = 1024,
        n_pos_coef_mlp_hidden: int = 512,
        n_head_size: int = 16,
        n_heads: int = 4,
        neighbor_dropout: float = 0.1,
        attention_dummy_score: float = 3.0,
        attention_penalty_coef: float = 0.0,
        value_l1_penalty_coef: float = 0.0,
        pos_coef_offset: float = -2.0,
        distance_kernel_unit_scale: float = 1.0,
    ):
        super().__init__()
        self.n_genes = n_genes
        self.n_labels = n_labels
        self.n_label_embed = n_label_embed
        self.n_query_embed_hidden = n_query_embed_hidden
        self.n_query_dim = n_query_dim
        self.n_kv_dim = n_kv_dim
        self.n_nn_embed = n_nn_embed
        self.n_nn_embed_hidden = n_nn_embed_hidden
        self.n_pos_coef_mlp_hidden = n_pos_coef_mlp_hidden
        self.attention_dummy_score = attention_dummy_score
        self.neighbor_dropout = neighbor_dropout
        self.attention_penalty_coef = attention_penalty_coef
        self.value_l1_penalty_coef = value_l1_penalty_coef
        self.distance_kernel_unit_scale = distance_kernel_unit_scale
        self.pos_coef_offset = pos_coef_offset
        self.n_head_size = n_head_size
        self.n_heads = n_heads
        self.empirical_ct_means = empirical_ct_means

        self.register_buffer("ct_profiles", self.empirical_ct_means)

        self.ct_embed = nn.Embedding(self.n_labels, self.n_label_embed)

        self.query_embed = ResNetMLP(
            n_input=self.n_label_embed,
            n_output=self.n_heads * self.n_query_dim,
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

        self.pos_coef_mlp = ResNetMLP(
            n_input=self.n_nn_embed + self.n_label_embed,
            n_output=self.n_heads,
            n_layers=2,
            n_hidden=self.n_pos_coef_mlp_hidden,
            dropout=0.0,
            use_final_layer_norm=False,
        )

        self.kv_embed = ResNetMLP(
            n_input=self.n_nn_embed,
            n_output=self.n_heads * self.n_kv_dim,
            n_layers=2,
            n_hidden=self.n_kv_dim,
            dropout=0.0,
        )

        self.attention_layer = AttentionBlock(
            self.n_query_dim,
            self.n_kv_dim,
            self.n_head_size,
            self.n_heads,
            dummy_attn_score=self.attention_dummy_score,
            add_res_connection=True,
        )

        self.linear_head = nn.Linear(
            self.n_heads * self.n_head_size,
            self.n_genes,
            bias=False,
        )

        self.hook_label_embed = HookPoint()  # [batch, n_label_embed, 1]
        self.hook_nn_embed = HookPoint()  # [batch, n_nn_embed, n_neighbors]
        self.hook_final_residual = HookPoint()  # [batch, n_genes]
        self.hook_pe_embed = HookPoint()

        self.setup()  # setup hook points

    def _get_inference_input(self, tensors):
        labels = tensors[REGISTRY_KEYS.LABELS_KEY]

        nn_X = tensors[NN_REGISTRY_KEYS.NN_X_KEY]

        return {
            "labels": labels,
            "nn_X": nn_X,
        }

    def inference(self, labels, nn_X):
        # convert target labels into cell type embeddings by a lookup table w arb dimension
        label_embed = self.hook_label_embed(rearrange(self.ct_embed(labels), "b 1 d -> b d"))  # batch x n_label_embed

        # embed neighbor expressions
        nn_embed = self.hook_nn_embed(self.nn_embed(nn_X))  # batch x n_neighbors x n_nn_embed

        return {
            "label_embed": label_embed,
            "nn_embed": nn_embed,
        }

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
    ):
        return_attention_patterns = self.attention_penalty_coef > 0.0 or return_attention_patterns
        return_v = self.value_l1_penalty_coef > 0.0 or return_v

        query_embed = self.query_embed(label_embed)
        query_embed = rearrange(query_embed, "b (h d) -> b 1 h d", h=self.n_heads)

        label_embed_repeated = repeat(label_embed, "b d -> b n d", n=nn_embed.shape[1])
        pos_coefs = F.softplus(
            self.pos_coef_mlp(torch.cat([nn_embed, label_embed_repeated], dim=-1)) + self.pos_coef_offset
        )  # batch x n_neighbors x n_heads
        pos_attn_score = -pos_coefs * (nn_dist.unsqueeze(-1) / self.distance_kernel_unit_scale)

        kv_embed = self.kv_embed(nn_embed)
        kv_embed = rearrange(kv_embed, "b n (h d) -> b n h d", h=self.n_heads)
        attention_mask = None
        if self.training and self.neighbor_dropout > 0.0:
            attention_mask = (
                torch.rand((kv_embed.shape[0], kv_embed.shape[1]), device=kv_embed.device) > self.neighbor_dropout
            ).int()

        attn_outs = self.attention_layer(
            query_embed,
            kv_embed,
            kv_embed,
            attention_mask=attention_mask,
            pos_attn_score=pos_attn_score,
            return_base_attn_scores=return_attention_scores,
            return_attn_patterns=return_attention_patterns,
            return_v=return_v,
        )
        residual_embed = attn_outs["x"]
        residual_embed = rearrange(residual_embed, "b 1 d -> b d")  # batch x n_genes

        attention_scores = None
        if return_attention_scores:
            attention_scores = attn_outs["base_attn_scores"][:, :, 0, :]

        attention_patterns = None
        if return_attention_patterns:
            attention_patterns = attn_outs["attn_patterns"][:, :, 0, :]

        attention_v = None
        if return_v:
            attention_v = attn_outs["v"]

        # linear layer output into prediction of gene expression residual
        residual = self.hook_final_residual(self.linear_head(residual_embed).float())  # batch x n_genes

        # have another matrix w/ cell type specific gene expression mean vectors
        # can be learned or start by just making them the empirical means of that cell type in the data
        # normalize x at all stages to help nn and to make loss simply the mse
        batch_ct_means = self.ct_profiles[labels.squeeze(-1)].squeeze()
        prediction = (batch_ct_means + residual).float()

        gen_outs = {
            "residual_embed": residual_embed,
            "residual": residual,
            "prediction": prediction,
            "attention_scores": attention_scores,
            "attention_patterns": attention_patterns,
            "attention_v": attention_v,
            "pos_coefs": pos_coefs,
        }
        return gen_outs

    # HACK: kl_weight argument exists to support VAEMixin get_reconstruction_error
    def loss(self, tensors, inference_outputs, generative_outputs, kl_weight=1.0):
        """Loss computation."""
        true_X = tensors[REGISTRY_KEYS.X_KEY]
        prediction = generative_outputs["prediction"]

        reconstruction_loss = F.gaussian_nll_loss(
            prediction, true_X, var=torch.ones_like(prediction), reduction="none"
        ).sum(-1)

        attention_penalty = torch.zeros(true_X.shape[0], device=true_X.device)
        if self.attention_penalty_coef > 0.0:
            attention_patterns = generative_outputs["attention_patterns"]  # batch x head_index x key_pos
            eps = torch.finfo(attention_patterns.dtype).eps
            attention_entropy_terms = (
                -1 * attention_patterns * torch.log(torch.clamp(attention_patterns, min=eps, max=1 - eps))
            )
            attention_penalty = reduce(
                reduce(
                    attention_entropy_terms,
                    "batch head_index key_pos -> batch head_index",
                    "sum",
                ),
                "batch head_index -> batch",
                "mean",
            )

        value_l1_penalty = torch.zeros(true_X.shape[0], device=true_X.device)
        if self.value_l1_penalty_coef > 0.0:
            attention_v = generative_outputs["attention_v"]
            value_l1_penalty = reduce(
                reduce(
                    torch.abs(attention_v),
                    "batch key_pos head_index head_size -> batch key_pos",
                    "sum",
                ),
                "batch key_pos -> batch",
                "mean",
            )

        loss = torch.mean(
            reconstruction_loss
            + self.attention_penalty_coef * attention_penalty
            + self.value_l1_penalty_coef * value_l1_penalty
        )

        return LossOutput(
            loss=loss,
            reconstruction_loss=reconstruction_loss,
            kl_local={
                "attention_penalty": self.attention_penalty_coef * attention_penalty,
                "value_l1_penalty": self.value_l1_penalty_coef * value_l1_penalty,
            },
            extra_metrics={"attention_penalty_coef": torch.tensor(self.attention_penalty_coef)},
        )
