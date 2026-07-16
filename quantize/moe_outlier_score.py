import torch
import torch.nn as nn
import torch.nn.functional as F

from models.olmoe.modeling_olmoe import OlmoeDecoderLayer
from models.pangu_moe.modeling_pangu_moe import PanguProMoEDecoderLayer

try:
    from transformers.models.qwen2_moe.modeling_qwen2_moe import Qwen2MoeDecoderLayer
except Exception:
    class Qwen2MoeDecoderLayer(nn.Module):
        pass


def _normalize_group_size(value):
    if value is None:
        return None
    value = int(value)
    if value <= 0:
        return None
    return value


def _normalize_weight_channel_group_size(value):
    if value is None:
        return None
    value = int(value)
    if value == -1:
        return -1
    if value <= 1:
        return None
    return value


def _reshape_weight_to_channel_groups(weight, weight_channel_group_size, group_size):
    weight_channel_group_size = _normalize_weight_channel_group_size(weight_channel_group_size)
    if weight_channel_group_size is None or weight.ndim != 2:
        return None, None

    out_channels, in_channels = weight.shape
    output_group_size = out_channels if weight_channel_group_size == -1 else min(weight_channel_group_size, out_channels)
    if output_group_size <= 1:
        return None, None

    out_pad = (output_group_size - out_channels % output_group_size) % output_group_size
    input_group_size = _normalize_group_size(group_size)
    if input_group_size is not None:
        in_pad = (input_group_size - in_channels % input_group_size) % input_group_size
    else:
        in_pad = 0

    x = weight
    if out_pad > 0 or in_pad > 0:
        x = F.pad(x, (0, in_pad, 0, out_pad))

    padded_out, padded_in = x.shape
    out_groups = padded_out // output_group_size
    if input_group_size is not None:
        input_groups = padded_in // input_group_size
        x = x.reshape(out_groups, output_group_size, input_groups, input_group_size)
        reduce_dims = (1, 3)
    else:
        x = x.reshape(out_groups, output_group_size, padded_in)
        reduce_dims = (1, 2)

    meta = {
        "original_shape": tuple(weight.shape),
        "padded_shape": (padded_out, padded_in),
        "out_pad": out_pad,
        "in_pad": in_pad,
        "reduce_dims": reduce_dims,
    }
    return x, meta


def _restore_weight_channel_groups(x, meta):
    padded_out, padded_in = meta["padded_shape"]
    x = x.reshape(padded_out, padded_in)
    out_channels, in_channels = meta["original_shape"]
    if meta["out_pad"] > 0:
        x = x.narrow(0, 0, out_channels)
    if meta["in_pad"] > 0:
        x = x.narrow(1, 0, in_channels)
    return x


@torch.no_grad()
def fake_quant_weight_for_outlier_score(weight, n_bits, symmetric, group_size=None, weight_channel_group_size=None):
    if int(n_bits) >= 16:
        return weight.detach().float()

    x = weight.detach().float()
    original_shape = tuple(x.shape)
    group_size = _normalize_group_size(group_size)
    channel_grouped_x, channel_group_meta = _reshape_weight_to_channel_groups(
        x,
        weight_channel_group_size,
        group_size,
    )
    padded_shape = None
    pad = 0
    if channel_grouped_x is not None:
        x = channel_grouped_x
        reduce_dims = channel_group_meta["reduce_dims"]
    elif group_size is not None:
        last_dim = x.shape[-1]
        deficiency = last_dim % group_size
        pad = 0 if deficiency == 0 else group_size - deficiency
        if pad > 0:
            x = F.pad(x, (0, pad))
        padded_shape = tuple(x.shape)
        x = x.reshape(-1, group_size)
        reduce_dims = -1
    else:
        x = x.reshape(-1, x.shape[-1])
        reduce_dims = -1

    if symmetric:
        qmin = -(2 ** (int(n_bits) - 1))
        qmax = 2 ** (int(n_bits) - 1) - 1
        scale = x.abs().amax(dim=reduce_dims, keepdim=True).clamp(min=1e-5) / max(qmax, 1)
        x_int = torch.round(x / scale).clamp(qmin, qmax)
        x_dequant = x_int * scale
    else:
        qmin = 0
        qmax = 2 ** int(n_bits) - 1
        xmin = x.amin(dim=reduce_dims, keepdim=True)
        xmax = x.amax(dim=reduce_dims, keepdim=True)
        scale = (xmax - xmin).clamp(min=1e-5) / max(qmax - qmin, 1)
        zero = torch.round(qmin - xmin / scale).clamp(qmin, qmax)
        x_int = (torch.round(x / scale) + zero).clamp(qmin, qmax)
        x_dequant = (x_int - zero) * scale

    if channel_group_meta is not None:
        x_dequant = _restore_weight_channel_groups(x_dequant, channel_group_meta)
    elif padded_shape is not None:
        x_dequant = x_dequant.reshape(padded_shape)
        if pad > 0:
            x_dequant = x_dequant.narrow(-1, 0, original_shape[-1])
    else:
        x_dequant = x_dequant.reshape(original_shape)
    return x_dequant


@torch.no_grad()
def weight_quant_error_by_input_channel(module, args):
    n_bits = int(getattr(args, "wbits", 16))
    symmetric = bool(getattr(args, "symmetric", False))
    group_size = getattr(args, "group_size", None)
    weight_channel_group_size = getattr(args, "weight_channel_group_size", None)
    weight = module.weight.detach().float().cpu()
    quant_weight = fake_quant_weight_for_outlier_score(
        weight,
        n_bits,
        symmetric,
        group_size,
        weight_channel_group_size,
    )
    return (weight - quant_weight).pow(2).sum(dim=0)


def _weight_max_by_input_channel(module):
    return module.weight.detach().abs().amax(dim=0).float().cpu()


def _clean_scores(scores):
    return torch.nan_to_num(scores.detach().float().cpu().flatten(), nan=0.0, posinf=0.0, neginf=0.0).clamp(min=0)


def _warn(logger, message):
    if logger is not None:
        logger.warning(message)


def _sanitize_or_fallback(scores, module, module_name, logger):
    scores = _clean_scores(scores)
    if scores.numel() != module.in_features:
        raise ValueError(
            f"moe_outlier_scores for {module_name} have {scores.numel()} channels, "
            f"expected {module.in_features}"
        )
    if scores.numel() == 0 or scores.max().item() <= 0:
        _warn(
            logger,
            f"invalid all-zero moe_outlier_scores for {module_name}; "
            "falling back to weight_max",
        )
        scores = _clean_scores(_weight_max_by_input_channel(module))
    if scores.numel() == 0 or scores.max().item() <= 0:
        raise ValueError(f"invalid all-zero fallback moe_outlier_scores for {module_name}")
    return scores


def _module_weight_score(module, module_name, score_method, args, logger):
    if score_method == "weight_max":
        scores = _weight_max_by_input_channel(module)
    elif score_method == "weight_error":
        scores = weight_quant_error_by_input_channel(module, args)
    else:
        raise ValueError(f"unsupported weight score method: {score_method}")
    return _sanitize_or_fallback(scores, module, module_name, logger)


def _smooth_scale_for_layer(layer_name, moe_fc1_smooth_scales):
    gate_name = f"{layer_name}.mlp.gate"
    if moe_fc1_smooth_scales is None or gate_name not in moe_fc1_smooth_scales:
        raise KeyError(f"missing MoE fc1 smooth scale for {gate_name}")
    return moe_fc1_smooth_scales[gate_name]


def _add_expert_scores(scores, module_name, gate_proj, up_proj, score_method, args, logger, layer_score=None):
    gate_name = f"{module_name}.gate_proj"
    up_name = f"{module_name}.up_proj"
    if score_method == "smooth_scale":
        scores[gate_name] = _sanitize_or_fallback(layer_score, gate_proj, gate_name, logger)
        scores[up_name] = _sanitize_or_fallback(layer_score, up_proj, up_name, logger)
    else:
        scores[gate_name] = _module_weight_score(gate_proj, gate_name, score_method, args, logger)
        scores[up_name] = _module_weight_score(up_proj, up_name, score_method, args, logger)


@torch.no_grad()
def prepare_moe_outlier_scores(
    model,
    *,
    score_method,
    moe_fc1_smooth_scales=None,
    args=None,
    model_name=None,
    logger=None,
):
    if int(getattr(args, "moe_outlier_topk", 0)) <= 0:
        return {}
    if score_method not in ("smooth_scale", "weight_max", "weight_error"):
        raise ValueError(f"unsupported moe_outlier_score: {score_method}")

    scores = {}
    seen_moe_layer = False
    for layer_name, module in model.named_modules():
        if isinstance(module, OlmoeDecoderLayer):
            seen_moe_layer = True
            layer_score = _smooth_scale_for_layer(layer_name, moe_fc1_smooth_scales) if score_method == "smooth_scale" else None
            for expert_idx, expert in enumerate(module.mlp.experts):
                _add_expert_scores(
                    scores,
                    f"{layer_name}.mlp.experts.{expert_idx}",
                    expert.gate_proj,
                    expert.up_proj,
                    score_method,
                    args,
                    logger,
                    layer_score=layer_score,
                )

        elif isinstance(module, Qwen2MoeDecoderLayer):
            seen_moe_layer = True
            layer_score = _smooth_scale_for_layer(layer_name, moe_fc1_smooth_scales) if score_method == "smooth_scale" else None
            for expert_idx, expert in enumerate(module.mlp.experts):
                _add_expert_scores(
                    scores,
                    f"{layer_name}.mlp.experts.{expert_idx}",
                    expert.gate_proj,
                    expert.up_proj,
                    score_method,
                    args,
                    logger,
                    layer_score=layer_score,
                )
            _add_expert_scores(
                scores,
                f"{layer_name}.mlp.shared_expert",
                module.mlp.shared_expert.gate_proj,
                module.mlp.shared_expert.up_proj,
                score_method,
                args,
                logger,
                layer_score=layer_score,
            )

        elif isinstance(module, PanguProMoEDecoderLayer):
            raise NotImplementedError(
                "prepare_moe_outlier_scores does not yet know the Pangu-MoE "
                "module naming contract. Add explicit Pangu gate/up names before "
                "using MoE outlier extraction on this model."
            )

    if not seen_moe_layer and model_name is not None:
        _warn(logger, f"no supported MoE decoder layers found for model_name={model_name}")
    if logger is not None:
        logger.info(
            f"prepared module-level moe_outlier_scores: {len(scores)} entries; "
            f"score_method={score_method}"
        )
    return scores
