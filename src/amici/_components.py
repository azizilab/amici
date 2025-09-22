import math
from typing import Literal, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import einsum, rearrange, repeat
from torch import Tensor
from transformer_lens.hook_points import HookPoint


class ResNetMLP(nn.Module):
    def __init__(
        self,
        n_input: int,
        n_output: Optional[int] = None,
        n_layers: int = 2,
        n_hidden: int = 128,
        dropout: float = 0.0,
        nonlinearity: Literal["relu", "tanh", "gelu"] = "relu",
        use_final_layer_norm: bool = True,
    ):
        super().__init__()
        if n_output is None:
            n_output = n_input
        self.n_input = n_input
        self.n_output = n_output
        self.n_hidden = n_hidden
        self.use_final_layer_norm = use_final_layer_norm

        if nonlinearity == "relu":
            self.nonlinearity = nn.ReLU()
        elif nonlinearity == "tanh":
            self.nonlinearity = nn.Tanh()
        elif nonlinearity == "gelu":
            self.nonlinearity = nn.GELU()
        else:
            raise ValueError(f"Unrecognized nonlinearity {nonlinearity}.")

        self.nn = nn.Sequential(
            *(
                [
                    nn.Linear(self.n_input, self.n_hidden),
                    nn.LayerNorm(self.n_hidden),
                    self.nonlinearity,
                ]
                * (n_layers - 1)
                + [
                    nn.Linear(self.n_hidden, self.n_input),
                    (
                        nn.Identity()
                        if not self.use_final_layer_norm and n_output == n_input
                        else nn.LayerNorm(self.n_input)
                    ),
                ]
            )
        )
        if n_output != n_input:
            self.project_nn = nn.Sequential(
                nn.Linear(self.n_input, self.n_output),
                (nn.LayerNorm(self.n_output) if self.use_final_layer_norm else nn.Identity()),
            )
        self.dropout = nn.Dropout(dropout)

        self.hook_pre = HookPoint()  # [batch, n_input]
        self.hook_pre_dropout = HookPoint()  # [batch, n_input]
        self.hook_post_dropout = HookPoint()  # [batch, n_input]

    def forward(self, x: Tensor) -> Tensor:
        h = self.hook_pre(x + self.nn(x))
        h = self.nonlinearity(h)
        if self.n_output != self.n_input:
            h = self.project_nn(h)
        h = self.hook_pre_dropout(h)
        h = self.hook_post_dropout(self.dropout(h))
        return h


class AttentionBlock(nn.Module):
    def __init__(
        self,
        query_dim: int,
        kv_dim: int,
        head_size: int,
        num_heads: int,
        dummy_attn_score: float = 0.0,
        dropout: float = 0.1,
        add_res_connection: bool = False,
    ):
        super().__init__()
        self.kv_dim = kv_dim
        self.query_dim = query_dim
        self.head_size = head_size
        self.num_heads = num_heads
        self.dummy_attn_score = dummy_attn_score
        self.dropout = dropout

        self.add_res_connection = add_res_connection

        self.embed_dim = self.num_heads * self.head_size
        self.W_Q = nn.Parameter(
            torch.empty(self.head_size, self.num_heads, self.query_dim),
        )
        self.b_Q = nn.Parameter(
            torch.zeros(self.num_heads, self.head_size),
        )
        self.W_K = nn.Parameter(
            torch.empty(self.head_size, self.num_heads, self.kv_dim),
        )
        self.b_K = nn.Parameter(
            torch.zeros(self.num_heads, self.head_size),
        )
        self.W_V = nn.Parameter(
            torch.empty(self.head_size, self.num_heads, self.kv_dim),
        )
        self.b_V = nn.Parameter(
            torch.zeros(self.num_heads, self.head_size),
        )
        self.W_O = nn.Parameter(
            torch.empty(self.embed_dim, self.num_heads, self.head_size),
        )
        self.reset_parameters()

        self.norm_o = nn.LayerNorm(self.embed_dim)

        self.dummy_attn_score = dummy_attn_score
        self.dropout = dropout

        self.register_buffer(
            "IGNORE",
            torch.tensor(-float("inf")),
        )

        self.hook_q_input = HookPoint()  # [batch, query_pos, head_index, head_size]
        self.hook_k_input = HookPoint()  # [batch, key_pos, head_index, head_size]
        self.hook_v_input = HookPoint()  # [batch, key_pos, head_index, head_size]
        self.hook_q = HookPoint()  # [batch, query_pos, head_index, head_size]
        self.hook_k = HookPoint()  # [batch, key_pos, head_index, head_size]
        self.hook_v = HookPoint()  # [batch, key_pos, head_index, head_size]
        self.hook_attn_scores = HookPoint()  # [batch, head_index, query_pos, key_pos]
        self.hook_pattern = HookPoint()  # [batch, head_index, query_pos, key_pos]
        self.hook_z = HookPoint()  # [batch, query_pos, head_index, head_size]
        self.hook_attn_out = HookPoint()  # [batch, query_pos, embed_dim]
        self.hook_output = HookPoint()  # [batch, query_pos, embed_dim]

    def reset_parameters(self):
        # Use Pytorch default linear initialization
        def init_weights(weight, bias, fan_in_multiplier=1):
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(weight)
            bound = 1 / math.sqrt(fan_in * fan_in_multiplier) if fan_in > 0 else 0
            nn.init.uniform_(weight, -bound, bound)
            if bias is not None:
                nn.init.uniform_(bias, -bound, bound)

        init_weights(
            self.W_Q,
            self.b_Q,
        )
        init_weights(self.W_K, self.b_K)
        init_weights(self.W_V, self.b_V)
        init_weights(self.W_O, None)

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        attention_mask: Optional[Tensor] = None,
        pos_attn_score: Optional[Tensor] = None,
        return_base_attn_scores: bool = False,
        return_attn_patterns: bool = False,
        return_v: bool = False,
    ) -> Union[Tensor, tuple[Tensor, Tensor]]:
        query = self.hook_q_input(query)
        key = self.hook_k_input(key)
        value = self.hook_v_input(value)

        q, k, v = self.compute_qkv_matrices(query, key, value)

        base_attn_scores = self.compute_base_attention_scores(q, k)
        attn_scores = base_attn_scores
        if pos_attn_score is not None:
            pos_attn_score = self._append_pos_attn_score_col(pos_attn_score)
            attn_scores = base_attn_scores + repeat(pos_attn_score, "b n h -> b h 1 n")
        attn_scores = self._apply_mask(attn_scores, attention_mask)
        attn_scores = self.hook_attn_scores(attn_scores)

        attn_patterns = F.softmax(attn_scores, dim=-1)
        attn_patterns = torch.where(torch.isnan(attn_patterns), torch.zeros_like(attn_patterns), attn_patterns)
        attn_patterns = self.hook_pattern(attn_patterns)

        z = self.compute_z_scores(v, attn_patterns)
        attn_out = self.compute_attn_out(z)

        out = attn_out
        if self.add_res_connection:
            out = out + rearrange(
                q,
                "batch query_pos head_index head_size -> batch query_pos (head_index head_size)",
            )

        out = self.hook_output(self.norm_o(out))

        ret_dict = {"x": out}
        if return_base_attn_scores:
            ret_dict["base_attn_scores"] = base_attn_scores
            ret_dict["attn_scores"] = attn_scores
        if return_attn_patterns:
            ret_dict["attn_patterns"] = attn_patterns
        if return_v:
            ret_dict["v"] = v
        return ret_dict

    def compute_qkv_matrices(self, query: Tensor, key: Tensor, value: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        q = self.hook_q(
            einsum(
                query,
                self.W_Q,
                "batch pos head_index query_dim, head_size head_index query_dim -> batch pos head_index head_size",
            )
            + self.b_Q
        )

        k = self.hook_k(
            einsum(
                key,
                self.W_K,
                "batch pos head_index kv_dim, head_size head_index kv_dim -> batch pos head_index head_size",
            )
            + self.b_K
        )

        v = self.hook_v(
            einsum(
                value,
                self.W_V,
                "batch pos head_index kv_dim, head_size head_index kv_dim -> batch pos head_index head_size",
            )
            + self.b_V
        )
        return q, k, v

    def compute_base_attention_scores(self, q: Tensor, k: Tensor) -> Tensor:
        attn_scores = einsum(
            q,
            k,
            "batch query_pos head_index head_size, "
            "batch key_pos head_index head_size "
            "-> batch head_index query_pos key_pos",
        ) / torch.sqrt(torch.tensor(self.head_size, dtype=torch.float32))
        attn_scores = self._append_dummy_attn_score_col(attn_scores)
        return attn_scores

    def compute_z_scores(self, v: Tensor, pattern: Tensor) -> Tensor:
        v = self._append_dummy_v_row(v)
        z = einsum(
            v,
            pattern,
            "batch key_pos head_index head_size, "
            "batch head_index query_pos key_pos -> batch query_pos head_index head_size",
        )
        return self.hook_z(z)

    def compute_attn_out(self, z: Tensor) -> Tensor:
        return self.hook_attn_out(
            einsum(
                z,
                self.W_O,
                "batch query_pos head_index head_size, " "embed_dim head_index head_size -> batch query_pos embed_dim",
            )
        )

    def _append_dummy_v_row(self, v: Tensor) -> Tensor:
        dummy_v_row = torch.zeros((v.size(0), 1, v.size(2), v.size(3))).to(v.device)
        return torch.cat([v, dummy_v_row], dim=1)

    def _append_dummy_attn_score_col(self, attn_scores: Tensor) -> Tensor:
        dummy_attn_score_col = torch.full(
            (attn_scores.size(0), attn_scores.size(1), attn_scores.size(2), 1),
            self.dummy_attn_score,
        ).to(attn_scores.device)
        return torch.cat([attn_scores, dummy_attn_score_col], dim=3)

    def _append_pos_attn_score_col(self, pos_attn_score: Tensor) -> Tensor:
        dummy_pos_attn_score_col = torch.zeros(
            (pos_attn_score.shape[0], 1, pos_attn_score.shape[2]),
            device=pos_attn_score.device,
        )
        return torch.cat([pos_attn_score, dummy_pos_attn_score_col], dim=1)

    def _apply_mask(
        self,
        attn_scores: Tensor,
        attention_mask: Optional[Tensor] = None,  # [batch, key_pos]
    ) -> Tensor:
        if attention_mask is None:
            return attn_scores
        attention_mask = torch.concat(
            [attention_mask, torch.ones_like(attention_mask[:, :1])], dim=1
        )  # handle dummy dimension
        attention_mask = repeat(
            attention_mask,
            "batch key_pos -> batch head_index query_pos key_pos",
            head_index=attn_scores.shape[1],
            query_pos=attn_scores.shape[2],
        )
        attention_mask = attention_mask.bool()
        return torch.where(attention_mask, attn_scores, self.IGNORE)
