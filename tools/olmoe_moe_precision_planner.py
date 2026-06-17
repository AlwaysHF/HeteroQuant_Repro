#!/usr/bin/env python3
import argparse
import json
import math
import os
import re
import sys
import time
from array import array
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from datautils import get_loaders
from models.LMClass import LMClass
from quantize.smooth import build_moe_fc1_smooth_scales, smooth_lm
from utils import (
    get_act_per_channel_scales,
    get_act_samples,
    get_act_scales,
    get_moe_act_means,
    get_moe_act_p99s,
    get_router_logits,
    get_weight_scores,
)


class PrintLogger:
    def info(self, message):
        print(message)

    def warning(self, message):
        print(f"WARNING: {message}")

    def warn(self, message):
        self.warning(message)



@dataclass(frozen=True)
class CandidateQuant:
    name: str
    abits: int
    wbits: int
    act_group_size: int = None
    weight_group_size: int = None
    sum_bits_cost: float = None
    cost_units: int = None

    @property
    def effective_avg_bits(self) -> float:
        return float(self.sum_bits_cost) / 2.0


@dataclass
class ModuleErrorRecord:
    module_name: str
    layer_index: int
    expert_index: int
    proj_name: str
    quantization: str
    abits: int
    wbits: int
    act_group_size: int
    weight_group_size: int
    sum_bits_cost: float
    effective_avg_bits: float
    cost_units: int
    sum_sq_diff: float
    sum_abs_diff: float
    error_norm: float
    mse: float
    mae: float
    num_calls: int
    num_tokens: int
    num_elements: int


class ErrorAccumulator:
    def __init__(self, module_name, layer_index, expert_index, proj_name, candidate):
        self.module_name = module_name
        self.layer_index = int(layer_index)
        self.expert_index = int(expert_index)
        self.proj_name = proj_name
        self.candidate = candidate
        self.sum_sq_diff = 0.0
        self.sum_abs_diff = 0.0
        self.num_calls = 0
        self.num_tokens = 0
        self.num_elements = 0

    def update(self, ref_out, quant_out):
        ref = ref_out.detach().float()
        q = quant_out.detach().float()
        diff = q - ref
        self.sum_sq_diff += float((diff * diff).sum().item())
        self.sum_abs_diff += float(diff.abs().sum().item())
        self.num_calls += 1
        self.num_elements += int(diff.numel())
        if diff.ndim >= 2:
            self.num_tokens += int(diff.reshape(-1, diff.shape[-1]).shape[0])
        else:
            self.num_tokens += 1

    def finalize(self):
        denom = max(self.num_elements, 1)
        return ModuleErrorRecord(
            module_name=self.module_name,
            layer_index=self.layer_index,
            expert_index=self.expert_index,
            proj_name=self.proj_name,
            quantization=self.candidate.name,
            abits=self.candidate.abits,
            wbits=self.candidate.wbits,
            act_group_size=self.candidate.act_group_size,
            weight_group_size=self.candidate.weight_group_size,
            sum_bits_cost=float(self.candidate.sum_bits_cost),
            effective_avg_bits=float(self.candidate.effective_avg_bits),
            cost_units=int(self.candidate.cost_units),
            sum_sq_diff=self.sum_sq_diff,
            sum_abs_diff=self.sum_abs_diff,
            error_norm=math.sqrt(self.sum_sq_diff),
            mse=self.sum_sq_diff / denom,
            mae=self.sum_abs_diff / denom,
            num_calls=self.num_calls,
            num_tokens=self.num_tokens,
            num_elements=self.num_elements,
        )


def parse_int_list(text):
    if text is None or str(text).strip() == "":
        return None
    return [int(x) for x in str(text).split(",") if x.strip()]


def parse_str_list(text):
    if text is None or str(text).strip() == "":
        return None
    return [x.strip() for x in str(text).split(",") if x.strip()]


def normalize_group_size(value):
    if value is None:
        return None
    value = int(value)
    if value <= 0:
        return None
    return value


def group_suffix(group_size):
    return -1 if group_size is None else int(group_size)


def parse_cost_map(text):
    result = {}
    if text is None or str(text).strip() == "":
        return result
    for item in str(text).split(","):
        item = item.strip()
        if not item:
            continue
        if "=" in item:
            key, value = item.split("=", 1)
        elif ":" in item:
            key, value = item.split(":", 1)
        else:
            raise ValueError(f"bad cost map item {item!r}; expected candidate=value")
        result[key.strip().lower()] = float(value)
    return result


def normalize_cost_map_keys(cost_map, cost_scale=1000):
    normalized = {}
    for key, value in (cost_map or {}).items():
        try:
            norm_key = parse_candidate_quant(key, cost_scale=cost_scale).name
        except Exception:
            norm_key = str(key).strip().lower()
        normalized[norm_key] = float(value)
    return normalized


def make_candidate(name, abits, wbits, act_group_size=None, weight_group_size=None, sum_bits_cost=None, cost_scale=1000):
    abits = int(abits)
    wbits = int(wbits)
    act_group_size = normalize_group_size(act_group_size)
    weight_group_size = normalize_group_size(weight_group_size)
    if abits not in (4, 8, 16):
        raise ValueError(f"activation bits must be 4, 8, or 16, got {abits} in {name}")
    if not (2 <= wbits <= 16):
        raise ValueError(f"weight bits must be in [2, 16], got {wbits} in {name}")
    if abits < wbits:
        raise ValueError(f"kernel constraint violated by {name}: activation bits must be >= weight bits")
    if sum_bits_cost is None:
        sum_bits_cost = float(abits + wbits)
    sum_bits_cost = float(sum_bits_cost)
    cost_units = int(round(sum_bits_cost * int(cost_scale)))
    if cost_units <= 0:
        raise ValueError(f"non-positive cost for {name}: {sum_bits_cost}")
    return CandidateQuant(
        name=name,
        abits=abits,
        wbits=wbits,
        act_group_size=act_group_size,
        weight_group_size=weight_group_size,
        sum_bits_cost=sum_bits_cost,
        cost_units=cost_units,
    )


def parse_candidate_quant(text, cost_scale=1000):
    raw = text.strip().lower()
    name = raw.replace("_", "")

    legacy = re.fullmatch(r"a(\d+)w(\d+)", name)
    if legacy is not None:
        abits = int(legacy.group(1))
        wbits = int(legacy.group(2))
        return make_candidate(
            name=f"a{abits}w{wbits}",
            abits=abits,
            wbits=wbits,
            act_group_size=None,
            weight_group_size=None,
            cost_scale=cost_scale,
        )

    patterns = [
        re.fullmatch(r"w(\d+)g(-?\d+)-a(\d+)g(-?\d+)", name),
        re.fullmatch(r"a(\d+)g(-?\d+)-w(\d+)g(-?\d+)", name),
    ]
    if patterns[0] is not None:
        m = patterns[0]
        wbits, wg, abits, ag = int(m.group(1)), int(m.group(2)), int(m.group(3)), int(m.group(4))
    elif patterns[1] is not None:
        m = patterns[1]
        abits, ag, wbits, wg = int(m.group(1)), int(m.group(2)), int(m.group(3)), int(m.group(4))
    else:
        raise ValueError(
            f"bad candidate quant name: {text}; expected e.g. a4w3 or w4g128-a4g128"
        )

    ag_norm = normalize_group_size(ag)
    wg_norm = normalize_group_size(wg)
    norm_name = f"w{wbits}g{group_suffix(wg_norm)}-a{abits}g{group_suffix(ag_norm)}"
    return make_candidate(
        name=norm_name,
        abits=abits,
        wbits=wbits,
        act_group_size=ag_norm,
        weight_group_size=wg_norm,
        cost_scale=cost_scale,
    )


def apply_candidate_cost_overrides(candidates, avg_costs=None, sum_costs=None, cost_scale=1000):
    avg_costs = avg_costs or {}
    sum_costs = sum_costs or {}
    updated = []
    for cand in candidates:
        sum_cost = cand.sum_bits_cost
        if cand.name in avg_costs:
            sum_cost = 2.0 * float(avg_costs[cand.name])
        if cand.name in sum_costs:
            sum_cost = float(sum_costs[cand.name])
        updated.append(
            make_candidate(
                name=cand.name,
                abits=cand.abits,
                wbits=cand.wbits,
                act_group_size=cand.act_group_size,
                weight_group_size=cand.weight_group_size,
                sum_bits_cost=sum_cost,
                cost_scale=cost_scale,
            )
        )
    return updated


def parse_candidate_quants(text, cost_scale=1000, candidate_avg_costs=None, candidate_sum_costs=None):
    candidates = [parse_candidate_quant(x, cost_scale=cost_scale) for x in text.split(",") if x.strip()]
    candidates = apply_candidate_cost_overrides(
        candidates,
        avg_costs=normalize_cost_map_keys(parse_cost_map(candidate_avg_costs), cost_scale=cost_scale),
        sum_costs=normalize_cost_map_keys(parse_cost_map(candidate_sum_costs), cost_scale=cost_scale),
        cost_scale=cost_scale,
    )
    names = [c.name for c in candidates]
    if len(names) != len(set(names)):
        raise ValueError(f"duplicate candidates: {names}")
    return candidates


def parse_module_name(name):
    m = re.search(r"model\.layers\.(\d+)\.mlp\.experts\.(\d+)\.(gate_proj|up_proj|down_proj)$", name)
    if m is None:
        raise ValueError(f"cannot parse OLMoE expert projection name: {name}")
    return int(m.group(1)), int(m.group(2)), m.group(3)


def collect_target_modules(model, selected_layers=None, selected_experts=None, selected_projs=None):
    if selected_projs is None:
        selected_projs = ["gate_proj", "up_proj", "down_proj"]
    selected = []
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        if ".mlp.experts." not in name:
            continue
        if not any(name.endswith("." + proj) for proj in selected_projs):
            continue
        try:
            layer_idx, expert_idx, proj_name = parse_module_name(name)
        except ValueError:
            continue
        if selected_layers is not None and layer_idx not in selected_layers:
            continue
        if selected_experts is not None and expert_idx not in selected_experts:
            continue
        selected.append((name, module, layer_idx, expert_idx, proj_name))
    selected.sort(key=lambda item: (item[2], item[3], item[4]))
    return selected


def collect_target_experts(model, selected_layers=None, selected_experts=None):
    selected = []
    for name, module in model.named_modules():
        m = re.fullmatch(r"model\.layers\.(\d+)\.mlp\.experts\.(\d+)", name)
        if m is None:
            continue
        if not all(hasattr(module, attr) for attr in ("gate_proj", "up_proj", "down_proj", "act_fn")):
            continue
        layer_idx = int(m.group(1))
        expert_idx = int(m.group(2))
        if selected_layers is not None and layer_idx not in selected_layers:
            continue
        if selected_experts is not None and expert_idx not in selected_experts:
            continue
        selected.append((name, module, layer_idx, expert_idx))
    selected.sort(key=lambda item: (item[2], item[3]))
    return selected


def _reshape_last_dim_to_groups(x, group_size):
    group_size = normalize_group_size(group_size)
    if group_size is None:
        return x, None, 0
    original_shape = tuple(x.shape)
    last_dim = original_shape[-1]
    deficiency = last_dim % group_size
    pad = 0 if deficiency == 0 else group_size - deficiency
    if pad > 0:
        x = F.pad(x, (0, pad))
    return x.reshape(*x.shape[:-1], -1, group_size), original_shape, pad


def _restore_grouped_last_dim(x, original_shape, pad):
    if original_shape is None:
        return x
    padded_shape = list(original_shape)
    if pad > 0:
        padded_shape[-1] += pad
    x = x.reshape(padded_shape)
    if pad > 0:
        x = x.narrow(-1, 0, original_shape[-1])
    return x


def quant_minmax(x, bits, reduce_dim=-1, symmetric=False, group_size=None):
    if bits >= 16 or x.numel() == 0:
        return x
    dtype = x.dtype
    xf = x.detach().float()
    if normalize_group_size(group_size) is not None:
        if reduce_dim not in (-1, xf.ndim - 1, 1):
            raise ValueError(f"group quant only supports last/input dim, got reduce_dim={reduce_dim}")
        xf, original_shape, pad = _reshape_last_dim_to_groups(xf, group_size)
        reduce_dim = -1
    else:
        original_shape, pad = None, 0
    if symmetric:
        qmin = -(2 ** (bits - 1))
        qmax = 2 ** (bits - 1) - 1
        xmax = xf.abs().amax(dim=reduce_dim, keepdim=True).clamp(min=1e-5)
        scale = xmax / max(qmax, 1)
        q = torch.round(xf / scale).clamp(qmin, qmax)
        out = q * scale
        return _restore_grouped_last_dim(out, original_shape, pad).to(dtype)
    qmin = 0
    qmax = 2 ** bits - 1
    xmin = xf.amin(dim=reduce_dim, keepdim=True)
    xmax = xf.amax(dim=reduce_dim, keepdim=True)
    scale = (xmax - xmin).clamp(min=1e-5) / max(qmax - qmin, 1)
    zero = torch.round(qmin - xmin / scale).clamp(qmin, qmax)
    q = (torch.round(xf / scale) + zero).clamp(qmin, qmax)
    out = (q - zero) * scale
    return _restore_grouped_last_dim(out, original_shape, pad).to(dtype)


def topk_outlier_input_channels(weight, topk):
    topk = min(int(topk), int(weight.shape[1]))
    if topk <= 0:
        return torch.empty(0, dtype=torch.long, device=weight.device)
    scores = weight.detach().abs().amax(dim=0).float()
    return torch.topk(scores, k=topk, largest=True, sorted=True).indices.to(torch.long)


def resolve_outlier_bits(mode, candidate):
    mode = str(mode).lower()
    if mode == "same":
        return candidate.abits, candidate.wbits
    if mode in ("fp16", "w16a16"):
        return 16, 16
    if mode == "w8a8":
        return 8, 8
    if mode == "w4a4":
        return 4, 4
    raise ValueError(f"unsupported moe_outlier_quant: {mode}")


def load_calibration(args):
    cache_path = Path(args.cache_dir) / f"planner_dataloader_olmoe_{args.calib_dataset}_{args.nsamples}_{args.seq_length}_{args.seed}.cache"
    if args.reuse_dataloader_cache and cache_path.exists():
        dataloader = torch.load(cache_path, map_location="cpu")
    else:
        dataloader, _ = get_loaders(
            args.calib_dataset,
            nsamples=args.nsamples,
            seed=args.seed,
            model=args.model,
            seqlen=args.seq_length,
        )
        if args.reuse_dataloader_cache:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(dataloader, cache_path)
    return dataloader[: args.nsamples]


def cleanup_memory():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    import gc

    gc.collect()


def build_lm(args):
    lm_args = argparse.Namespace(
        model=args.model,
        model_name="olmoe",
        batch_size=1,
        attn_implementation=args.attn_implementation,
    )
    lm = LMClass(lm_args)
    lm.seqlen = args.seq_length
    lm.model.eval()
    for param in lm.model.parameters():
        param.requires_grad = False
    return lm


def apply_optional_smooth(lm, args, dataloader):
    if not args.smooth:
        return
    print(
        f"Applying smooth: fc1_scale_merge={args.fc1_scale_merge}, "
        f"act_mean_beta={args.act_mean_beta}"
    )
    model = lm.model
    if args.fc1_scale_merge not in ("act_mean", "act_p99"):
        raise ValueError("--fc1_scale_merge must be act_mean or act_p99")
    moe_stat_fn = get_moe_act_means if args.fc1_scale_merge == "act_mean" else get_moe_act_p99s
    moe_act_stats = moe_stat_fn(model, dataloader, args.nsamples)
    moe_fc1_smooth_scales = build_moe_fc1_smooth_scales(
        moe_act_stats,
        act_mean_beta=args.act_mean_beta,
        model=model,
    )
    act_scales = get_act_scales(model, dataloader, args.nsamples)
    act_per_channel_scales = get_act_per_channel_scales(model, dataloader, args.nsamples)
    smooth_lm(
        model,
        act_scales,
        act_per_channel_scales,
        {},
        {},
        {},
        fc1_scale_merge=args.fc1_scale_merge,
        alpha=args.alpha,
        otsu_ratio=args.otsu_ratio,
        otsu_smooth_rate=args.otsu_smooth_rate,
        moe_fc1_smooth_scales=moe_fc1_smooth_scales,
        logger=PrintLogger(),
    )
    cleanup_memory()


def collect_routing_frequency(model, dataloader, device, selected_layers=None):
    stats = {}
    handles = []
    layers = model.model.layers
    for layer_idx, layer in enumerate(layers):
        if selected_layers is not None and layer_idx not in selected_layers:
            continue
        moe = getattr(layer, "mlp", None)
        if moe is None or not hasattr(moe, "gate") or not hasattr(moe, "experts"):
            continue
        num_experts = len(moe.experts)
        top_k = int(getattr(moe, "top_k", getattr(model.config, "num_experts_per_tok", 2)))
        stats[layer_idx] = {
            "layer_index": layer_idx,
            "num_experts": num_experts,
            "top_k": top_k,
            "counts": torch.zeros(num_experts, dtype=torch.long),
            "num_tokens_seen": 0,
            "num_assignments": 0,
        }

        def make_hook(idx, n_exp, k):
            def hook(mod, inputs, output):
                logits = output[0] if isinstance(output, tuple) else output
                if not torch.is_tensor(logits):
                    return
                logits = logits.reshape(-1, n_exp)
                top_idx = torch.topk(logits, k=k, dim=-1).indices
                stats[idx]["counts"] += torch.bincount(top_idx.reshape(-1).cpu(), minlength=n_exp)
                stats[idx]["num_tokens_seen"] += int(logits.shape[0])
                stats[idx]["num_assignments"] += int(top_idx.numel())

            return hook

        handles.append(moe.gate.register_forward_hook(make_hook(layer_idx, num_experts, top_k)))

    try:
        for batch in tqdm(dataloader, desc="routing frequency"):
            input_ids = batch[0].to(device)
            with torch.no_grad():
                model(input_ids=input_ids, use_cache=False)
            del input_ids
    finally:
        for handle in handles:
            handle.remove()
    results = []
    for layer_idx in sorted(stats):
        item = stats[layer_idx]
        total = int(item["num_assignments"])
        counts = item["counts"].tolist()
        freqs = [float(c / total) if total else 0.0 for c in counts]
        results.append(
            {
                "layer_index": layer_idx,
                "num_experts": int(item["num_experts"]),
                "top_k": int(item["top_k"]),
                "num_tokens_seen": int(item["num_tokens_seen"]),
                "num_assignments": total,
                "expert_counts": counts,
                "expert_frequencies": freqs,
                "frequency_sum": float(sum(freqs)),
            }
        )
    cleanup_memory()
    return results


def run_projection_error_analysis(model, dataloader, selected_modules, candidates, args, device):
    all_records = []
    model.config.use_cache = False

    for candidate in candidates:
        accs = {
            name: ErrorAccumulator(name, layer_idx, expert_idx, proj_name, candidate)
            for name, _, layer_idx, expert_idx, proj_name in selected_modules
        }
        handles = []
        weight_cache = {}
        outlier_cache = {}

        def get_cached_weight(mod):
            if candidate.wbits >= 16:
                return mod.weight
            cache_key = ("weight", id(mod), candidate.name)
            if args.weight_cache == "none":
                return quant_minmax(mod.weight, candidate.wbits, reduce_dim=1, symmetric=args.symmetric, group_size=candidate.weight_group_size)
            if cache_key not in weight_cache:
                wq_tmp = quant_minmax(mod.weight, candidate.wbits, reduce_dim=1, symmetric=args.symmetric, group_size=candidate.weight_group_size).detach()
                if args.weight_cache == "cpu":
                    wq_tmp = wq_tmp.cpu()
                weight_cache[cache_key] = wq_tmp
            return weight_cache[cache_key]

        def get_outlier_parts(mod):
            cache_key = ("outlier", id(mod), int(args.moe_outlier_topk))
            if args.weight_cache != "none" and cache_key in outlier_cache:
                return outlier_cache[cache_key]
            idx = topk_outlier_input_channels(mod.weight, args.moe_outlier_topk)
            main_weight = mod.weight.detach().clone()
            main_weight.index_fill_(1, idx.to(main_weight.device), 0)
            outlier_weight = mod.weight.detach().index_select(1, idx.to(mod.weight.device)).clone()
            if args.weight_cache == "cpu":
                parts = (idx.cpu(), main_weight.cpu(), outlier_weight.cpu())
                outlier_cache[cache_key] = parts
                return parts
            if args.weight_cache == "device":
                parts = (idx, main_weight, outlier_weight)
                outlier_cache[cache_key] = parts
                return parts
            return idx, main_weight, outlier_weight

        def make_hook(name, module, proj_name):
            def hook(mod, inputs, output):
                x = inputs[0]
                xq = quant_minmax(x, candidate.abits, reduce_dim=-1, symmetric=False, group_size=candidate.act_group_size)

                if args.moe_outlier_topk > 0 and proj_name in ("gate_proj", "up_proj"):
                    idx, main_weight, outlier_weight = get_outlier_parts(mod)
                    idx = idx.to(device=x.device)
                    main_weight = main_weight.to(device=x.device, dtype=x.dtype, non_blocking=True)
                    outlier_weight = outlier_weight.to(device=x.device, dtype=x.dtype, non_blocking=True)
                    main_wq = quant_minmax(main_weight, candidate.wbits, reduce_dim=1, symmetric=args.symmetric, group_size=candidate.weight_group_size)
                    qout = F.linear(xq, main_wq, mod.bias)

                    outlier_abits, outlier_wbits = resolve_outlier_bits(args.moe_outlier_quant, candidate)
                    outlier_x = x.index_select(-1, idx)
                    if outlier_abits < 16:
                        outlier_x = quant_minmax(outlier_x, outlier_abits, reduce_dim=-1, symmetric=False)
                    if outlier_wbits < 16:
                        outlier_weight = quant_minmax(outlier_weight, outlier_wbits, reduce_dim=1, symmetric=args.symmetric)
                    qout = qout + F.linear(outlier_x, outlier_weight, None)
                else:
                    wq = get_cached_weight(mod)
                    wq = wq.to(device=x.device, dtype=x.dtype, non_blocking=True)
                    qout = F.linear(xq, wq, mod.bias)

                accs[name].update(output, qout)
                return output

            return hook

        for name, module, _, _, proj_name in selected_modules:
            handles.append(module.register_forward_hook(make_hook(name, module, proj_name)))

        try:
            for batch in tqdm(dataloader, desc=f"projection MSE {candidate.name}"):
                input_ids = batch[0].to(device)
                with torch.no_grad():
                    model(input_ids=input_ids, use_cache=False)
                del input_ids
        finally:
            for handle in handles:
                handle.remove()

        all_records.extend(asdict(acc.finalize()) for _, acc in sorted(accs.items()))
        del weight_cache, outlier_cache, accs
        cleanup_memory()

    return all_records


def run_expert_output_error_analysis(model, dataloader, selected_experts, selected_projs, candidates, args, device):
    all_records = []
    model.config.use_cache = False
    selected_projs = selected_projs or ["gate_proj", "up_proj", "down_proj"]
    selected_projs = [proj for proj in selected_projs if proj in ("gate_proj", "up_proj", "down_proj")]
    if not selected_projs:
        raise ValueError("no valid expert projection selected for expert-output analysis")

    for candidate in candidates:
        accs = {}
        for expert_name, _, layer_idx, expert_idx in selected_experts:
            for proj_name in selected_projs:
                module_name = f"{expert_name}.{proj_name}"
                accs[module_name] = ErrorAccumulator(module_name, layer_idx, expert_idx, proj_name, candidate)

        handles = []
        weight_cache = {}
        outlier_cache = {}

        def get_cached_weight(mod):
            if candidate.wbits >= 16:
                return mod.weight
            cache_key = ("weight", id(mod), candidate.name)
            if args.weight_cache == "none":
                return quant_minmax(mod.weight, candidate.wbits, reduce_dim=1, symmetric=args.symmetric, group_size=candidate.weight_group_size)
            if cache_key not in weight_cache:
                wq_tmp = quant_minmax(mod.weight, candidate.wbits, reduce_dim=1, symmetric=args.symmetric, group_size=candidate.weight_group_size).detach()
                if args.weight_cache == "cpu":
                    wq_tmp = wq_tmp.cpu()
                weight_cache[cache_key] = wq_tmp
            return weight_cache[cache_key]

        def get_outlier_parts(mod):
            cache_key = ("outlier", id(mod), int(args.moe_outlier_topk))
            if args.weight_cache != "none" and cache_key in outlier_cache:
                return outlier_cache[cache_key]
            idx = topk_outlier_input_channels(mod.weight, args.moe_outlier_topk)
            main_weight = mod.weight.detach().clone()
            main_weight.index_fill_(1, idx.to(main_weight.device), 0)
            outlier_weight = mod.weight.detach().index_select(1, idx.to(mod.weight.device)).clone()
            if args.weight_cache == "cpu":
                parts = (idx.cpu(), main_weight.cpu(), outlier_weight.cpu())
                outlier_cache[cache_key] = parts
                return parts
            if args.weight_cache == "device":
                parts = (idx, main_weight, outlier_weight)
                outlier_cache[cache_key] = parts
                return parts
            return idx, main_weight, outlier_weight

        def quantized_linear(mod, x, proj_name):
            xq = quant_minmax(x, candidate.abits, reduce_dim=-1, symmetric=False, group_size=candidate.act_group_size)
            if args.moe_outlier_topk > 0 and proj_name in ("gate_proj", "up_proj"):
                idx, main_weight, outlier_weight = get_outlier_parts(mod)
                idx = idx.to(device=x.device)
                main_weight = main_weight.to(device=x.device, dtype=x.dtype, non_blocking=True)
                outlier_weight = outlier_weight.to(device=x.device, dtype=x.dtype, non_blocking=True)
                main_wq = quant_minmax(main_weight, candidate.wbits, reduce_dim=1, symmetric=args.symmetric, group_size=candidate.weight_group_size)
                out = F.linear(xq, main_wq, mod.bias)

                outlier_abits, outlier_wbits = resolve_outlier_bits(args.moe_outlier_quant, candidate)
                outlier_x = x.index_select(-1, idx)
                if outlier_abits < 16:
                    outlier_x = quant_minmax(outlier_x, outlier_abits, reduce_dim=-1, symmetric=False)
                if outlier_wbits < 16:
                    outlier_weight = quant_minmax(outlier_weight, outlier_wbits, reduce_dim=1, symmetric=args.symmetric)
                return out + F.linear(outlier_x, outlier_weight, None)

            wq = get_cached_weight(mod)
            wq = wq.to(device=x.device, dtype=x.dtype, non_blocking=True)
            return F.linear(xq, wq, mod.bias)

        def make_hook(expert_name, expert_module):
            def hook(mod, inputs, output):
                x = inputs[0]
                gate_ref = mod.gate_proj(x)
                up_ref = mod.up_proj(x)
                hidden_ref = mod.act_fn(gate_ref) * up_ref
                ref_out = output[0] if isinstance(output, tuple) else output

                if "gate_proj" in selected_projs:
                    gate_q = quantized_linear(mod.gate_proj, x, "gate_proj")
                    q_out = mod.down_proj(mod.act_fn(gate_q) * up_ref)
                    accs[f"{expert_name}.gate_proj"].update(ref_out, q_out)

                if "up_proj" in selected_projs:
                    up_q = quantized_linear(mod.up_proj, x, "up_proj")
                    q_out = mod.down_proj(mod.act_fn(gate_ref) * up_q)
                    accs[f"{expert_name}.up_proj"].update(ref_out, q_out)

                if "down_proj" in selected_projs:
                    q_out = quantized_linear(mod.down_proj, hidden_ref, "down_proj")
                    accs[f"{expert_name}.down_proj"].update(ref_out, q_out)

                return output

            return hook

        for expert_name, expert_module, _, _ in selected_experts:
            handles.append(expert_module.register_forward_hook(make_hook(expert_name, expert_module)))

        try:
            for batch in tqdm(dataloader, desc=f"expert output MSE {candidate.name}"):
                input_ids = batch[0].to(device)
                with torch.no_grad():
                    model(input_ids=input_ids, use_cache=False)
                del input_ids
        finally:
            for handle in handles:
                handle.remove()

        all_records.extend(asdict(acc.finalize()) for _, acc in sorted(accs.items()))
        del weight_cache, outlier_cache, accs
        cleanup_memory()

    return all_records


def objective_value(record, metric):
    if metric == "sse":
        return float(record["sum_sq_diff"])
    if metric == "mse":
        return float(record["mse"])
    if metric == "error_norm":
        return float(record["error_norm"])
    if metric == "mae":
        return float(record["mae"])
    raise ValueError(f"unknown objective metric: {metric}")


def normalize_records_for_candidates(records, candidates):
    candidate_map = {cand.name: cand for cand in candidates}
    normalized = []
    skipped = Counter()
    for rec in records:
        qname = str(rec.get("quantization", "")).lower()
        if qname not in candidate_map:
            skipped[qname] += 1
            continue
        cand = candidate_map[qname]
        item = dict(rec)
        item["quantization"] = cand.name
        item["abits"] = int(cand.abits)
        item["wbits"] = int(cand.wbits)
        item["act_group_size"] = cand.act_group_size
        item["weight_group_size"] = cand.weight_group_size
        item["sum_bits_cost"] = float(cand.sum_bits_cost)
        item["effective_avg_bits"] = float(cand.effective_avg_bits)
        item["cost_units"] = int(cand.cost_units)
        normalized.append(item)
    if not normalized:
        raise ValueError("no candidate error records matched --candidates")
    if skipped:
        print(f"Skipped cached records for candidates not requested: {dict(skipped)}")
    return normalized


def pareto_options(options):
    by_cost = {}
    for option in options:
        cost = int(option["cost_units"])
        if cost not in by_cost or option["objective"] < by_cost[cost]["objective"]:
            by_cost[cost] = option
    ordered = [by_cost[cost] for cost in sorted(by_cost)]
    kept = []
    best_error = float("inf")
    for option in ordered:
        if option["objective"] <= best_error:
            kept.append(option)
            best_error = option["objective"]
    return kept


def optimize_exact_budget(module_table, candidate_names, target_avg_sum_bits, cost_scale):
    module_names = sorted(module_table)
    n = len(module_names)
    if n == 0:
        raise ValueError("no modules to optimize")

    raw_options_by_module = []
    all_costs = []
    target_total_cost_raw = int(round(float(target_avg_sum_bits) * int(cost_scale) * n))
    all_costs.append(target_total_cost_raw)
    for module_name in module_names:
        options = pareto_options(module_table[module_name]["options"])
        if not options:
            raise ValueError(f"no valid options for {module_name}")
        raw_options_by_module.append(options)
        all_costs.extend(int(opt["cost_units"]) for opt in options)

    gcd_value = 0
    for cost in all_costs:
        gcd_value = math.gcd(gcd_value, abs(int(cost)))
    gcd_value = max(gcd_value, 1)

    target_total_cost = target_total_cost_raw // gcd_value
    options_by_module = []
    min_total_cost = 0
    for options in raw_options_by_module:
        min_cost = min(int(opt["cost_units"]) // gcd_value for opt in options)
        min_total_cost += min_cost
        normalized = []
        for idx, opt in enumerate(options):
            item = dict(opt)
            item["option_index"] = idx
            item["normalized_cost_units"] = int(item["cost_units"]) // gcd_value
            item["extra_cost"] = int(item["normalized_cost_units"]) - min_cost
            normalized.append(item)
        options_by_module.append(normalized)

    extra_budget = target_total_cost - min_total_cost
    if extra_budget < 0:
        raise ValueError(
            f"target_avg_sum_bits={target_avg_sum_bits} is below minimum feasible average "
            f"{(min_total_cost * gcd_value) / (n * int(cost_scale)):.6f}"
        )

    inf = float("inf")
    dp = [inf] * (extra_budget + 1)
    dp[0] = 0.0
    parents = []

    for options in tqdm(options_by_module, desc="exact budget DP"):
        new_dp = [inf] * (extra_budget + 1)
        choice = array("h", [-1]) * (extra_budget + 1)
        for used_extra, prev_val in enumerate(dp):
            if prev_val == inf:
                continue
            for opt_idx, opt in enumerate(options):
                next_extra = used_extra + int(opt["extra_cost"])
                if next_extra > extra_budget:
                    continue
                value = prev_val + float(opt["objective"])
                if value < new_dp[next_extra]:
                    new_dp[next_extra] = value
                    choice[next_extra] = opt_idx
        dp = new_dp
        parents.append(choice)

    feasible = [(val, b) for b, val in enumerate(dp) if val < inf]
    if not feasible:
        raise RuntimeError("no feasible quantization plan")
    if args_use_full_budget_global:
        exact = [(val, b) for b, val in feasible if b == extra_budget]
        if exact:
            best_value, best_extra = exact[0]
        else:
            best_value, best_extra = min(feasible, key=lambda x: (abs(extra_budget - x[1]), x[0]))
    else:
        best_value, best_extra = min(feasible, key=lambda x: x[0])

    chosen = {}
    cur = best_extra
    for i in range(n - 1, -1, -1):
        opt_idx = int(parents[i][cur])
        if opt_idx < 0:
            raise RuntimeError("failed to reconstruct quantization plan")
        module_name = module_names[i]
        opt = options_by_module[i][opt_idx]
        chosen[module_name] = opt
        cur -= int(opt["extra_cost"])

    return chosen, best_value


args_use_full_budget_global = True


def build_plan(records, args):
    global args_use_full_budget_global
    args_use_full_budget_global = bool(args.use_full_budget)
    module_table = {}
    candidate_names = []
    for rec in records:
        qname = rec["quantization"]
        if qname not in candidate_names:
            candidate_names.append(qname)
        module_name = rec["module_name"]
        if module_name not in module_table:
            module_table[module_name] = {
                "module_name": module_name,
                "layer_index": int(rec["layer_index"]),
                "expert_index": int(rec["expert_index"]),
                "proj_name": rec["proj_name"],
                "options": [],
            }
        option = dict(rec)
        option["objective"] = objective_value(rec, args.objective_metric)
        module_table[module_name]["options"].append(option)

    missing = []
    expected = set(candidate_names)
    for module_name, info in module_table.items():
        found = {opt["quantization"] for opt in info["options"]}
        if found != expected:
            missing.append((module_name, sorted(expected - found)))
    if missing:
        raise ValueError(f"some modules miss candidate results, first examples: {missing[:5]}")

    chosen, objective = optimize_exact_budget(module_table, candidate_names, args.target_avg_sum_bits, args.cost_scale)
    plan = []
    per_module = {}
    total_cost = 0.0
    total_objective = 0.0
    for module_name in sorted(chosen):
        opt = chosen[module_name]
        entry = {
            "module_name": module_name,
            "layer_index": int(opt["layer_index"]),
            "expert_index": int(opt["expert_index"]),
            "proj_name": opt["proj_name"],
            "quantization": opt["quantization"],
            "abits": int(opt["abits"]),
            "wbits": int(opt["wbits"]),
            "act_group_size": opt.get("act_group_size"),
            "weight_group_size": opt.get("weight_group_size"),
            "sum_bits_cost": float(opt["sum_bits_cost"]),
            "effective_avg_bits": float(opt.get("effective_avg_bits", float(opt["sum_bits_cost"]) / 2.0)),
            "cost_units": int(opt["cost_units"]),
            "objective": float(opt["objective"]),
            "mse": float(opt["mse"]),
            "sum_sq_diff": float(opt["sum_sq_diff"]),
            "num_tokens": int(opt["num_tokens"]),
            "num_elements": int(opt["num_elements"]),
        }
        plan.append(entry)
        per_module[module_name] = entry
        total_cost += float(entry["sum_bits_cost"])
        total_objective += entry["objective"]

    summary = summarize_plan(plan, args, total_objective, objective)
    return {"plan": plan, "per_module": per_module, "summary": summary}, module_table


def summarize_plan(plan, args, total_objective, dp_objective):
    quant_counts = Counter(item["quantization"] for item in plan)
    by_layer = defaultdict(Counter)
    by_proj = defaultdict(Counter)
    for item in plan:
        by_layer[int(item["layer_index"])][item["quantization"]] += 1
        by_proj[item["proj_name"]][item["quantization"]] += 1
    avg_sum_bits = sum(float(item["sum_bits_cost"]) for item in plan) / max(len(plan), 1)
    return {
        "format": "duquant_olmoe_moe_quant_plan_v1",
        "num_modules": len(plan),
        "candidate_quants": args.candidates,
        "target_avg_sum_bits": float(args.target_avg_sum_bits),
        "actual_avg_sum_bits": float(avg_sum_bits),
        "target_equivalent_avg_aw_bits": float(args.target_avg_sum_bits) / 2.0,
        "actual_equivalent_avg_aw_bits": float(avg_sum_bits) / 2.0,
        "objective_metric": args.objective_metric,
        "total_objective": float(total_objective),
        "dp_objective": float(dp_objective),
        "quantization_counts": dict(quant_counts),
        "by_layer": {str(k): dict(v) for k, v in sorted(by_layer.items())},
        "by_proj_name": {k: dict(v) for k, v in sorted(by_proj.items())},
    }


def save_json(obj, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def plot_plan_distributions(plan, candidate_names, output_dir):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir = Path(output_dir)
    layers = sorted({int(item["layer_index"]) for item in plan})
    by_layer = defaultdict(Counter)
    by_proj = defaultdict(Counter)
    for item in plan:
        by_layer[int(item["layer_index"])][item["quantization"]] += 1
        by_proj[item["proj_name"]][item["quantization"]] += 1

    plt.figure(figsize=(max(10, len(layers) * 0.55), 5.5))
    bottoms = [0] * len(layers)
    x = list(range(len(layers)))
    for qname in candidate_names:
        vals = [by_layer[layer][qname] for layer in layers]
        plt.bar(x, vals, bottom=bottoms, label=qname)
        bottoms = [b + v for b, v in zip(bottoms, vals)]
    plt.xticks(x, [str(layer) for layer in layers])
    plt.xlabel("Layer")
    plt.ylabel("Module count")
    plt.title("MoE quant assignment by layer")
    plt.legend(ncol=min(5, len(candidate_names)))
    plt.tight_layout()
    layer_path = output_dir / "assignment_by_layer.png"
    plt.savefig(layer_path, dpi=220)
    plt.close()

    proj_order = ["gate_proj", "up_proj", "down_proj"]
    plt.figure(figsize=(9, 5.5))
    bottoms = [0] * len(proj_order)
    x = list(range(len(proj_order)))
    for qname in candidate_names:
        vals = [by_proj[proj][qname] for proj in proj_order]
        plt.bar(x, vals, bottom=bottoms, label=qname)
        bottoms = [b + v for b, v in zip(bottoms, vals)]
    plt.xticks(x, proj_order)
    plt.xlabel("Projection")
    plt.ylabel("Module count")
    plt.title("MoE quant assignment by projection")
    plt.legend(ncol=min(5, len(candidate_names)))
    plt.tight_layout()
    proj_path = output_dir / "assignment_by_projection.png"
    plt.savefig(proj_path, dpi=220)
    plt.close()
    return str(layer_path), str(proj_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=str(ROOT_DIR / "local_models" / "olmoe_compat"))
    parser.add_argument("--cache_dir", default=str(ROOT_DIR / "cache"))
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--calib_dataset", default="wikitext2", choices=["wikitext2", "ptb", "c4", "mix", "pile"])
    parser.add_argument("--nsamples", type=int, default=16)
    parser.add_argument("--seq_length", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=2)
    parser.add_argument("--attn_implementation", default="eager", choices=["eager", "sdpa", "flash_attention_2"])
    parser.add_argument("--candidates", default="a4w3,a4w4,a8w4,a8w8")
    parser.add_argument("--candidate_avg_costs", default=None, help="comma map candidate=avg_bits_per_number; internally doubled to A+W sum cost")
    parser.add_argument("--candidate_sum_costs", default=None, help="comma map candidate=sum_cost for A+W; overrides default abits+wbits")
    parser.add_argument("--cost_scale", type=int, default=1000, help="integer scale for fractional costs in exact DP")
    parser.add_argument("--target_avg_sum_bits", type=float, default=8.0)
    parser.add_argument("--objective_metric", default="sse", choices=["sse", "mse", "error_norm", "mae"])
    parser.add_argument("--analysis_target", default="expert", choices=["expert", "projection"])
    parser.add_argument("--layers", default=None, help="comma separated layer ids; default all")
    parser.add_argument("--experts", default=None, help="comma separated expert ids; default all")
    parser.add_argument("--projs", default="gate_proj,up_proj,down_proj")
    parser.add_argument("--symmetric", action="store_true")
    parser.add_argument("--smooth", action="store_true")
    parser.add_argument("--fc1_scale_merge", default="act_mean")
    parser.add_argument("--act_mean_beta", type=float, default=2.0)
    parser.add_argument("--alpha", type=float, default=0.6)
    parser.add_argument("--otsu_ratio", type=float, default=0.65)
    parser.add_argument("--otsu_smooth_rate", type=float, default=0.7)
    parser.add_argument("--moe_outlier_topk", type=int, default=0)
    parser.add_argument("--moe_outlier_quant", default="w16a16", choices=["same", "fp16", "w16a16", "w8a8", "w4a4"])
    parser.add_argument("--weight_cache", default="none", choices=["none", "cpu", "device"])
    parser.add_argument("--reuse_dataloader_cache", action="store_true")
    parser.add_argument("--reuse_errors", action="store_true")
    parser.add_argument("--errors_only", action="store_true", help="only compute/save candidate errors, do not build a quantization plan")
    parser.add_argument("--errors_json", default=None)
    parser.add_argument("--skip_frequency", action="store_true")
    parser.add_argument("--allow_under_budget", action="store_true", help="minimize MSE with cost <= target instead of forcing the full target budget")
    args = parser.parse_args()
    args.use_full_budget = not args.allow_under_budget

    started = time.perf_counter()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    candidates = parse_candidate_quants(
        args.candidates,
        cost_scale=args.cost_scale,
        candidate_avg_costs=args.candidate_avg_costs,
        candidate_sum_costs=args.candidate_sum_costs,
    )
    candidate_names = [cand.name for cand in candidates]
    errors_path = Path(args.errors_json) if args.errors_json else output_dir / f"candidate_{args.analysis_target}_errors.json"

    if args.reuse_errors and errors_path.exists():
        with open(errors_path, "r", encoding="utf-8") as f:
            records = json.load(f)
        print(f"Loaded existing errors from {errors_path}")
    else:
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(args.seed)
        lm = build_lm(args)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        lm.model.to(device)
        dataloader = load_calibration(args)
        apply_optional_smooth(lm, args, dataloader)
        selected_layers = parse_int_list(args.layers)
        selected_experts_arg = parse_int_list(args.experts)
        selected_projs = parse_str_list(args.projs)

        if not args.skip_frequency:
            fre = collect_routing_frequency(lm.model, dataloader, device, selected_layers=selected_layers)
            save_json(fre, output_dir / "routing_frequency.json")
            print(f"Saved routing frequency to {output_dir / 'routing_frequency.json'}")

        if args.analysis_target == "expert":
            selected_experts = collect_target_experts(
                lm.model,
                selected_layers=selected_layers,
                selected_experts=selected_experts_arg,
            )
            if not selected_experts:
                raise RuntimeError("no OLMoE expert modules selected")
            print(
                f"Selected {len(selected_experts)} experts x "
                f"{len(selected_projs or ['gate_proj', 'up_proj', 'down_proj'])} projections "
                f"for expert-output MSE"
            )
            records = run_expert_output_error_analysis(
                lm.model, dataloader, selected_experts, selected_projs, candidates, args, device
            )
        else:
            selected_modules = collect_target_modules(
                lm.model,
                selected_layers=selected_layers,
                selected_experts=selected_experts_arg,
                selected_projs=selected_projs,
            )
            if not selected_modules:
                raise RuntimeError("no OLMoE expert projection modules selected")
            print(f"Selected {len(selected_modules)} expert projection modules")
            records = run_projection_error_analysis(lm.model, dataloader, selected_modules, candidates, args, device)
        records = normalize_records_for_candidates(records, candidates)
        save_json(records, errors_path)
        print(f"Saved candidate errors to {errors_path}")
        del lm
        cleanup_memory()

    records = normalize_records_for_candidates(records, candidates)
    if args.errors_only:
        metadata = {
            "format": "duquant_olmoe_candidate_errors_v2",
            "errors_json": str(errors_path),
            "analysis_target": args.analysis_target,
            "candidates": [asdict(cand) for cand in candidates],
            "model": args.model,
            "calib_dataset": args.calib_dataset,
            "nsamples": args.nsamples,
            "seq_length": args.seq_length,
            "seed": args.seed,
            "smooth": bool(args.smooth),
            "fc1_scale_merge": args.fc1_scale_merge,
            "act_mean_beta": args.act_mean_beta,
            "moe_outlier_topk": args.moe_outlier_topk,
            "moe_outlier_quant": args.moe_outlier_quant,
            "num_records": len(records),
            "elapsed_seconds": time.perf_counter() - started,
        }
        metadata_path = output_dir / f"candidate_{args.analysis_target}_errors_metadata.json"
        save_json(metadata, metadata_path)
        print(f"Saved candidate error metadata to {metadata_path}")
        print("errors_only set; skip plan optimization")
        return

    plan_obj, _ = build_plan(records, args)
    plan_obj["summary"].update(
        {
            "model": args.model,
            "calib_dataset": args.calib_dataset,
            "nsamples": args.nsamples,
            "seq_length": args.seq_length,
            "seed": args.seed,
            "smooth": bool(args.smooth),
            "fc1_scale_merge": args.fc1_scale_merge,
            "act_mean_beta": args.act_mean_beta,
            "moe_outlier_topk": args.moe_outlier_topk,
            "analysis_target": args.analysis_target,
            "moe_outlier_quant": args.moe_outlier_quant,
            "elapsed_seconds": time.perf_counter() - started,
        }
    )
    plan_path = output_dir / "moe_quant_plan.json"
    summary_path = output_dir / "moe_quant_plan_summary.json"
    save_json(plan_obj, plan_path)
    save_json(plan_obj["summary"], summary_path)
    layer_png, proj_png = plot_plan_distributions(plan_obj["plan"], candidate_names, output_dir)

    print(f"Saved plan: {plan_path}")
    print(f"Saved summary: {summary_path}")
    print(f"Saved layer distribution: {layer_png}")
    print(f"Saved projection distribution: {proj_png}")
    print(f"Actual avg sum bits: {plan_obj['summary']['actual_avg_sum_bits']:.6f}")
    print(f"Equivalent avg A/W bits: {plan_obj['summary']['actual_equivalent_avg_aw_bits']:.6f}")


if __name__ == "__main__":
    main()
