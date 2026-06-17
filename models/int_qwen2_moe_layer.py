import torch
import torch.nn.functional as F
import torch.nn as nn
from transformers.activations import ACT2FN
from transformers.cache_utils import Cache, DynamicCache, StaticCache
from typing import List, Optional, Tuple, Union
from quantize.int_linear import QuantLinear
from quantize.int_matmul import QuantMatMul
from quantize.du_norm import DuOlmoeRMSNorm
from collections import OrderedDict

import logging
import math
import sys

from transformers.models.qwen2_moe.modeling_qwen2_moe import (
    Qwen2MoeConfig,
    Qwen2MoeRotaryEmbedding,
    apply_rotary_pos_emb,
    repeat_kv,
)

logger = logging.getLogger(__name__)



def _clone_linear(module: nn.Linear) -> nn.Linear:
    cloned = nn.Linear(
        module.in_features,
        module.out_features,
        bias=module.bias is not None,
        device=module.weight.device,
        dtype=module.weight.dtype,
    )
    with torch.no_grad():
        cloned.weight.copy_(module.weight)
        if module.bias is not None:
            cloned.bias.copy_(module.bias)
    return cloned


def _topk_outlier_input_channels(
    module: nn.Linear,
    topk: int,
    args=None,
    module_name=None,
) -> torch.Tensor:
    topk = min(int(topk), module.in_features)
    if topk <= 0:
        return torch.empty(0, dtype=torch.long)

    if module_name is None:
        raise ValueError("module_name is required for module-level moe_outlier_scores")

    scores_by_module = getattr(args, "moe_outlier_scores", None) or {}
    if module_name not in scores_by_module:
        raise KeyError(f"missing moe_outlier_scores for {module_name}")

    scores = scores_by_module[module_name].detach().float().cpu().flatten()
    if scores.numel() != module.in_features:
        raise ValueError(
            f"moe_outlier_scores for {module_name} have {scores.numel()} channels, "
            f"expected {module.in_features}"
        )

    scores = torch.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0).clamp(min=0)
    if scores.numel() == 0 or scores.max().item() <= 0:
        raise ValueError(f"invalid all-zero moe_outlier_scores for {module_name}")

    return torch.topk(scores, k=topk, largest=True, sorted=True).indices.to(torch.long)


def _split_linear_outlier_columns(module: nn.Linear, topk: int, args=None, module_name=None):
    outlier_idx = _topk_outlier_input_channels(
        module,
        topk,
        args=args,
        module_name=module_name,
    )
    if outlier_idx.numel() == 0:
        return module, None, outlier_idx

    main_module = _clone_linear(module)
    outlier_module = nn.Linear(
        outlier_idx.numel(),
        module.out_features,
        bias=False,
        device=module.weight.device,
        dtype=module.weight.dtype,
    )
    outlier_idx_device = outlier_idx.to(module.weight.device)
    with torch.no_grad():
        outlier_module.weight.copy_(module.weight.index_select(1, outlier_idx_device))
        main_module.weight.index_fill_(1, outlier_idx_device, 0)
    return main_module, outlier_module, outlier_idx


def _resolve_outlier_quant_mode(args):
    mode = getattr(args, "moe_outlier_quant", "same")
    if mode == "same":
        return mode, int(getattr(args, "wbits", 16)), int(getattr(args, "abits", 16))
    if mode == "fp16":
        return mode, 16, 16
    if mode == "w16a16":
        return mode, 16, 16
    if mode == "w8a8":
        return mode, 8, 8
    if mode == "w4a4":
        return mode, 4, 4
    raise ValueError(f"unsupported moe_outlier_quant mode: {mode}")


def _get_moe_quant_plan_entry(args, module_name: str):
    plan = getattr(args, "moe_quant_plan_dict", None)
    if not plan:
        return None
    if module_name in plan:
        return plan[module_name]

    # Be tolerant to plans saved with or without a leading model prefix.
    for key, value in plan.items():
        if module_name.endswith("." + key) or key.endswith("." + module_name):
            return value
    return None


def _normalize_group_size(value):
    if value is None:
        return None
    value = int(value)
    if value <= 0:
        return None
    return value


def _resolve_moe_proj_quant_params(args, module_name: str, weight_quant_params: dict, act_quant_params: dict):
    weight_params = dict(weight_quant_params)
    act_params = dict(act_quant_params)
    entry = _get_moe_quant_plan_entry(args, module_name)
    if entry is not None:
        weight_params["n_bits"] = int(entry["wbits"])
        act_params["n_bits"] = int(entry["abits"])
        if "weight_group_size" in entry:
            weight_params["group_size"] = _normalize_group_size(entry.get("weight_group_size"))
        elif "w_group_size" in entry:
            weight_params["group_size"] = _normalize_group_size(entry.get("w_group_size"))
        if "act_group_size" in entry:
            act_params["act_group_size"] = _normalize_group_size(entry.get("act_group_size"))
        elif "a_group_size" in entry:
            act_params["act_group_size"] = _normalize_group_size(entry.get("a_group_size"))
    return weight_params, act_params, entry


class MoeOutlierExpertBank(nn.Module):
    def __init__(self, gate_modules, up_modules, gate_indices, up_indices, args=None):
        super().__init__()
        self.num_experts = len(gate_modules)
        self.outlier_channels = gate_indices[0].numel() if gate_indices else 0
        self.outlier_output_features = gate_modules[0].out_features if gate_modules else 0
        self.quant_mode, self.weight_bits, self.act_bits = _resolve_outlier_quant_mode(args)
        self.symmetric = getattr(args, "symmetric", False)
        self.use_weight_quant = False
        self.use_act_quant = False
        self.weight_quantized = False

        gate_weight = torch.cat([module.weight.detach().clone() for module in gate_modules], dim=0)
        up_weight = torch.cat([module.weight.detach().clone() for module in up_modules], dim=0)
        self.register_buffer("gate_weight", gate_weight, persistent=True)
        self.register_buffer("up_weight", up_weight, persistent=True)
        self.register_buffer("gate_indices", torch.stack(gate_indices, dim=0).long(), persistent=True)
        self.register_buffer("up_indices", torch.stack(up_indices, dim=0).long(), persistent=True)

    def _fake_quant(self, x, n_bits, dim, symmetric=False):
        if self.quant_mode == "fp16" or n_bits >= 16:
            return x
        dtype = x.dtype
        x_float = x.float()
        if symmetric:
            qmin = -(2 ** (n_bits - 1))
            qmax = 2 ** (n_bits - 1) - 1
            scale = x_float.abs().amax(dim=dim, keepdim=True).clamp(min=1e-5) / max(qmax, 1)
            x_int = torch.round(x_float / scale).clamp(qmin, qmax)
            return (x_int * scale).to(dtype)
        qmin = 0
        qmax = 2 ** n_bits - 1
        xmin = x_float.amin(dim=dim, keepdim=True)
        xmax = x_float.amax(dim=dim, keepdim=True)
        scale = (xmax - xmin).clamp(min=1e-5) / max(qmax - qmin, 1)
        zero = torch.round(qmin - xmin / scale).clamp(qmin, qmax)
        x_int = (torch.round(x_float / scale) + zero).clamp(qmin, qmax)
        return ((x_int - zero) * scale).to(dtype)

    def _quantize_weight(self, weight):
        return self._fake_quant(weight, self.weight_bits, dim=1, symmetric=self.symmetric)

    def _quantize_input(self, x):
        return self._fake_quant(x, self.act_bits, dim=-1, symmetric=False)

    def set_quant_state(self, weight_quant=False, act_quant=False):
        self.use_weight_quant = weight_quant
        self.use_act_quant = act_quant

    @torch.no_grad()
    def quantize_weight_inplace(self):
        if self.quant_mode == "fp16" or self.weight_bits >= 16 or self.weight_quantized:
            return
        self.gate_weight.copy_(self._quantize_weight(self.gate_weight))
        self.up_weight.copy_(self._quantize_weight(self.up_weight))
        self.weight_quantized = True

    def _slice_weight(self, tensor, expert_idx):
        start = int(expert_idx) * self.outlier_output_features
        end = start + self.outlier_output_features
        return tensor[start:end]

    def forward(self, expert_idx, x):
        if self.outlier_channels == 0:
            return None, None
        gate_weight = self._slice_weight(self.gate_weight, expert_idx)
        up_weight = self._slice_weight(self.up_weight, expert_idx)
        gate_idx = self.gate_indices[expert_idx].to(x.device)
        up_idx = self.up_indices[expert_idx].to(x.device)
        gate_x = x.index_select(-1, gate_idx)
        up_x = x.index_select(-1, up_idx)

        if self.use_weight_quant and not self.weight_quantized:
            gate_weight = self._quantize_weight(gate_weight)
            up_weight = self._quantize_weight(up_weight)
        if self.use_act_quant:
            gate_x = self._quantize_input(gate_x)
            up_x = self._quantize_input(up_x)

        gate_outlier = F.linear(gate_x, gate_weight, None)
        up_outlier = F.linear(up_x, up_weight, None)
        return gate_outlier, up_outlier


class QuantQwen2MoeMLP(nn.Module):
    def __init__(
        self,
        org_module: nn.Module,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
        args=None,
        module_name="mlp",
        gate_proj=None,
        up_proj=None,
    ):
        super().__init__()
        disable_gate_up_rotate = getattr(args, "disable_moe_gate_up_duquant_rotation", False)
        gate_up_rotate = not disable_gate_up_rotate
        gate_proj = gate_proj if gate_proj is not None else org_module.gate_proj
        up_proj = up_proj if up_proj is not None else org_module.up_proj

        gate_weight_quant_params, gate_act_quant_params, gate_plan_entry = _resolve_moe_proj_quant_params(
            args, f"{module_name}.gate_proj", args.gate_weight_quant_params, args.gate_act_quant_params
        )
        down_weight_quant_params, down_act_quant_params, down_plan_entry = _resolve_moe_proj_quant_params(
            args, f"{module_name}.down_proj", args.down_weight_quant_params, args.down_act_quant_params
        )
        up_weight_quant_params, up_act_quant_params, up_plan_entry = _resolve_moe_proj_quant_params(
            args, f"{module_name}.up_proj", args.up_weight_quant_params, args.up_act_quant_params
        )
        self.quant_plan_entries = {
            "gate_proj": gate_plan_entry,
            "up_proj": up_plan_entry,
            "down_proj": down_plan_entry,
        }

        self.gate_proj = QuantLinear(
            gate_proj,
            gate_weight_quant_params,
            gate_act_quant_params,
            f"{module_name}.gate_proj",
            rotate=gate_up_rotate,
        )
        self.down_proj = QuantLinear(
            org_module.down_proj,
            down_weight_quant_params,
            down_act_quant_params,
            f"{module_name}.down_proj",
        )
        self.up_proj = QuantLinear(
            up_proj,
            up_weight_quant_params,
            up_act_quant_params,
            f"{module_name}.up_proj",
            rotate=gate_up_rotate,
        )
        self.act_fn = ACT2FN[hidden_act]
        self.init_duquant_params = torch.tensor(0) if gate_weight_quant_params["quant_method"] == "duquant" else torch.tensor(1)

    def _add_outlier(self, base, outlier):
        if outlier is None:
            return base
        return base + outlier.to(dtype=base.dtype)

    def forward(self, x, gate_outlier=None, up_outlier=None):
        gate = self.gate_proj(x)
        if not self.init_duquant_params:
            self.init_duquant_params = torch.tensor(1)
            self.up_proj.copy_quantizers_duquant_params(self.gate_proj)
        up = self.up_proj(x)
        gate = self._add_outlier(gate, gate_outlier)
        up = self._add_outlier(up, up_outlier)
        return self.down_proj(self.act_fn(gate) * up)


class QuantQwen2MoeAttention(nn.Module):
    def __init__(
        self,
        org_module: nn.Module,
        config: Qwen2MoeConfig,
        layer_idx: Optional[int] = None,
        args=None,
    ):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        if layer_idx is None:
            logger.warning(
                f"Instantiating {self.__class__.__name__} without passing layer_idx may break KV-cache decoding."
            )

        self.attention_dropout = config.attention_dropout
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        self.is_causal = True

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got hidden_size={self.hidden_size}, num_heads={self.num_heads})"
            )

        self.q_proj = QuantLinear(org_module.q_proj, args.q_weight_quant_params, args.q_act_quant_params, "q_proj")
        self.k_proj = QuantLinear(org_module.k_proj, args.k_weight_quant_params, args.k_act_quant_params, "k_proj")
        self.v_proj = QuantLinear(org_module.v_proj, args.v_weight_quant_params, args.v_act_quant_params, "v_proj")
        self.o_proj = QuantLinear(org_module.o_proj, args.o_weight_quant_params, args.o_act_quant_params, "o_proj")
        self.qkt_matmul = QuantMatMul(
            args.q_quant_params, args.k_quant_params, "qkt_matmul", matmul_func=torch.matmul, rotate=None
        )
        self.pv_matmul = QuantMatMul(
            args.p_quant_params, args.v_quant_params, "pv_matmul", matmul_func=torch.matmul, rotate=None
        )
        self.rotary_emb = Qwen2MoeRotaryEmbedding(
            self.head_dim,
            max_position_embeddings=self.max_position_embeddings,
            base=self.rope_theta,
        )

        self.use_weight_quant = False
        self.use_act_quant = False
        self.init_duquant_params = torch.tensor(0) if args.gate_weight_quant_params["quant_method"] == "duquant" else torch.tensor(1)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        if not self.init_duquant_params:
            self.k_proj.copy_quantizers_duquant_params(self.q_proj)
        key_states = self.k_proj(hidden_states)
        if not self.init_duquant_params:
            self.v_proj.copy_quantizers_duquant_params(self.q_proj)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)
        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

        query_states = self.qkt_matmul.quant_x1(query_states)
        key_states = self.qkt_matmul.quant_x2(key_states)

        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        attn_weights = self.qkt_matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)
        if attention_mask is not None:
            causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
            attn_weights = attn_weights + causal_mask

        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)
        attn_weights = self.pv_matmul.quant_x1(attn_weights)
        value_states = self.pv_matmul.quant_x2(value_states)
        attn_output = self.pv_matmul(attn_weights, value_states)

        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"attn_output should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, got {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2).contiguous().reshape(bsz, q_len, self.hidden_size)
        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        self.init_duquant_params = torch.tensor(1)
        return attn_output, attn_weights, past_key_value

    def set_quant_state(self, weight_quant: bool = False, act_quant: bool = False):
        self.use_weight_quant = weight_quant
        self.use_act_quant = act_quant
        for m in self.modules():
            if isinstance(m, (QuantLinear, QuantMatMul)):
                m.set_quant_state(weight_quant, act_quant)


class QuantQwen2MoeSparseMoeBlock(nn.Module):
    def __init__(
        self,
        org_module: nn.Module,
        config,
        args=None,
        layer_idx=None,
    ):
        super().__init__()
        self.num_experts = config.num_experts
        self.top_k = config.num_experts_per_tok
        self.norm_topk_prob = config.norm_topk_prob
        self.gate = QuantLinear(org_module.gate, args.router_weight_quant_params, args.router_act_quant_params, "router")
        shared_expert_gate_weight_quant_params = dict(args.router_weight_quant_params)
        shared_expert_gate_weight_quant_params["dynamic_method"] = "per_channel_tensor"
        shared_expert_gate_weight_quant_params["router_top_k"] = None
        self.shared_expert_gate = QuantLinear(
            org_module.shared_expert_gate,
            shared_expert_gate_weight_quant_params,
            args.router_act_quant_params,
            "shared_expert_gate",
        )
        outlier_topk = min(int(getattr(args, "moe_outlier_topk", 0)), config.hidden_size)

        experts = []
        gate_outlier_modules = []
        up_outlier_modules = []
        gate_outlier_indices = []
        up_outlier_indices = []
        for expert_idx in range(self.num_experts):
            org_expert = org_module.experts[expert_idx]
            expert_module_name = (
                f"model.layers.{layer_idx}.mlp.experts.{expert_idx}"
                if layer_idx is not None
                else f"experts.{expert_idx}"
            )
            gate_module_name = f"{expert_module_name}.gate_proj"
            up_module_name = f"{expert_module_name}.up_proj"
            gate_proj = org_expert.gate_proj
            up_proj = org_expert.up_proj
            if outlier_topk > 0:
                gate_proj, gate_outlier_proj, gate_outlier_idx = _split_linear_outlier_columns(
                    org_expert.gate_proj,
                    outlier_topk,
                    args=args,
                    module_name=gate_module_name,
                )
                up_proj, up_outlier_proj, up_outlier_idx = _split_linear_outlier_columns(
                    org_expert.up_proj,
                    outlier_topk,
                    args=args,
                    module_name=up_module_name,
                )
                gate_outlier_modules.append(gate_outlier_proj)
                up_outlier_modules.append(up_outlier_proj)
                gate_outlier_indices.append(gate_outlier_idx)
                up_outlier_indices.append(up_outlier_idx)
            experts.append(
                QuantQwen2MoeMLP(
                    org_module=org_expert,
                    hidden_size=config.hidden_size,
                    intermediate_size=config.moe_intermediate_size,
                    hidden_act=config.hidden_act,
                    args=args,
                    module_name=expert_module_name,
                    gate_proj=gate_proj,
                    up_proj=up_proj,
                )
            )

        self.experts = nn.ModuleList(experts)
        self.outlier_expert = (
            MoeOutlierExpertBank(gate_outlier_modules, up_outlier_modules, gate_outlier_indices, up_outlier_indices, args=args)
            if outlier_topk > 0
            else None
        )

        shared_module_name = (
            f"model.layers.{layer_idx}.mlp.shared_expert" if layer_idx is not None else "shared_expert"
        )
        shared_gate_module_name = f"{shared_module_name}.gate_proj"
        shared_up_module_name = f"{shared_module_name}.up_proj"
        shared_gate_proj = org_module.shared_expert.gate_proj
        shared_up_proj = org_module.shared_expert.up_proj
        shared_gate_outlier_modules = []
        shared_up_outlier_modules = []
        shared_gate_outlier_indices = []
        shared_up_outlier_indices = []
        if outlier_topk > 0:
            shared_gate_proj, shared_gate_outlier_proj, shared_gate_outlier_idx = _split_linear_outlier_columns(
                org_module.shared_expert.gate_proj,
                outlier_topk,
                args=args,
                module_name=shared_gate_module_name,
            )
            shared_up_proj, shared_up_outlier_proj, shared_up_outlier_idx = _split_linear_outlier_columns(
                org_module.shared_expert.up_proj,
                outlier_topk,
                args=args,
                module_name=shared_up_module_name,
            )
            shared_gate_outlier_modules.append(shared_gate_outlier_proj)
            shared_up_outlier_modules.append(shared_up_outlier_proj)
            shared_gate_outlier_indices.append(shared_gate_outlier_idx)
            shared_up_outlier_indices.append(shared_up_outlier_idx)

        self.shared_expert = QuantQwen2MoeMLP(
            org_module=org_module.shared_expert,
            hidden_size=config.hidden_size,
            intermediate_size=config.shared_expert_intermediate_size,
            hidden_act=config.hidden_act,
            args=args,
            module_name=shared_module_name,
            gate_proj=shared_gate_proj,
            up_proj=shared_up_proj,
        )
        self.shared_outlier_expert = (
            MoeOutlierExpertBank(
                shared_gate_outlier_modules,
                shared_up_outlier_modules,
                shared_gate_outlier_indices,
                shared_up_outlier_indices,
                args=args,
            )
            if outlier_topk > 0
            else None
        )

    def forward_expert(self, expert_idx, current_state):
        if self.outlier_expert is None:
            return self.experts[expert_idx](current_state)
        gate_outlier, up_outlier = self.outlier_expert(expert_idx, current_state)
        return self.experts[expert_idx](current_state, gate_outlier=gate_outlier, up_outlier=up_outlier)

    def forward_shared_expert(self, hidden_states):
        if self.shared_outlier_expert is None:
            return self.shared_expert(hidden_states)
        gate_outlier, up_outlier = self.shared_outlier_expert(0, hidden_states)
        return self.shared_expert(hidden_states, gate_outlier=gate_outlier, up_outlier=up_outlier)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)
        router_logits = self.gate(hidden_states)

        routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
        routing_weights, selected_experts = torch.topk(routing_weights, self.top_k, dim=-1)
        if self.norm_topk_prob:
            routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
        routing_weights = routing_weights.to(hidden_states.dtype)

        final_hidden_states = torch.zeros(
            (batch_size * sequence_length, hidden_dim), dtype=hidden_states.dtype, device=hidden_states.device
        )

        expert_mask = torch.nn.functional.one_hot(selected_experts, num_classes=self.num_experts).permute(2, 1, 0)

        for expert_idx in range(self.num_experts):
            idx, top_x = torch.where(expert_mask[expert_idx])
            if len(top_x) == 0:
                continue
            current_state = hidden_states[None, top_x].reshape(-1, hidden_dim)
            current_hidden_states = self.forward_expert(expert_idx, current_state) * routing_weights[top_x, idx, None]
            final_hidden_states.index_add_(0, top_x, current_hidden_states.to(hidden_states.dtype))

        shared_expert_output = self.forward_shared_expert(hidden_states)
        shared_expert_output = F.sigmoid(self.shared_expert_gate(hidden_states)) * shared_expert_output
        final_hidden_states = final_hidden_states + shared_expert_output

        final_hidden_states = final_hidden_states.reshape(batch_size, sequence_length, hidden_dim)
        return final_hidden_states, router_logits


class QuantQwen2MoeDecoderLayer(nn.Module):
    def __init__(
        self,
        config: Qwen2MoeConfig,
        layer_idx: int,
        ori_layer,
        args,
    ):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = QuantQwen2MoeAttention(
            org_module=ori_layer.self_attn,
            config=config,
            layer_idx=layer_idx,
            args=args,
        )
        self.mlp = QuantQwen2MoeSparseMoeBlock(
            org_module=ori_layer.mlp,
            config=config,
            args=args,
            layer_idx=layer_idx,
        )
        self.input_layernorm = DuOlmoeRMSNorm(
            ori_layer.input_layernorm,
            eps=ori_layer.input_layernorm.variance_epsilon,
        )
        self.post_attention_layernorm = DuOlmoeRMSNorm(
            ori_layer.post_attention_layernorm,
            eps=ori_layer.post_attention_layernorm.variance_epsilon,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: Optional[bool] = False,
        output_router_logits: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states, router_logits = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)
        if output_attentions:
            outputs += (self_attn_weights,)
        if use_cache:
            outputs += (present_key_value,)
        if output_router_logits:
            outputs += (router_logits,)
        return outputs

    def set_quant_state(self, weight_quant: bool = False, act_quant: bool = False):
        # setting weight quantization here does not affect actual forward pass
        self.use_weight_quant = weight_quant
        self.use_act_quant = act_quant
        names = []
        for name, m in self.named_modules():
            if isinstance(m, (QuantLinear, QuantMatMul)):
                names.append(name)
                m.set_quant_state(weight_quant, act_quant)
            elif hasattr(m, "quantize_weight_inplace") and hasattr(m, "set_quant_state"):
                m.set_quant_state(weight_quant, act_quant)
      

    def clear_temp_variable(self):
       for name, module in self.named_modules():
            if isinstance(module, QuantLinear):
                del module.temp_weight
                del module.temp_bias

    def register_duquant_params(self):        
        for name, module in self.named_modules():
            if isinstance(module, QuantQwen2MoeMLP) or isinstance(module, QuantQwen2MoeAttention):
                delattr(module, 'init_duquant_params')
                module.register_buffer('init_duquant_params', torch.tensor(1))
            if isinstance(module, QuantLinear):
                module.weight_quantizer.register_duquant_params(scale_zp = True)
                module.act_quantizer.register_duquant_params(scale_zp = False)
    
    def load_duquant_params(self, state_dict, device):
        for k, v in state_dict.items():
            if k.find('R') > -1 or k.find('permutation_list') > -1 or k.find('init_duquant_params') > -1 or k.find('scales') > -1 or k.find('zeros') > -1:
                if "shared_expert" not in k:
                    k = k.replace("experts.", "experts[").replace(".gate_outlier_proj", "].gate_outlier_proj").replace(".up_outlier_proj", "].up_outlier_proj").replace(".gate_proj", "].gate_proj").replace(".up_proj", "].up_proj").replace(".down_proj", "].down_proj")
                exec(f'self.{k} = v.to(device)')
    
    def load_smooth_params(self, state_dict, device):
        for k, v in state_dict.items():
            if k.find('smooth') > -1:
                # exec(f'self.{k} = v')
                self.register_parameter(k, torch.nn.Parameter(v.to(device), requires_grad=False))
    
    def load_post_params(self, state_dict, device):
        for k, v in state_dict.items():
            if k.find('post') > -1:
                # exec(f'self.{k} = v')
                rg = False if k.find('down') > -1 else True
                self.register_parameter(k, torch.nn.Parameter(v.to(device), requires_grad=rg))
