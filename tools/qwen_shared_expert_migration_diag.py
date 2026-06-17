#!/usr/bin/env python3
import argparse
import csv
import glob
import json
import logging
import os
import sys
import time
from collections import defaultdict
from types import SimpleNamespace

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from datautils import get_loaders
from models.LMClass import LMClass
from quantize.smooth import build_moe_fc1_smooth_scales, smooth_lm
from quantize.moe_outlier_score import prepare_moe_outlier_scores
from utils import convert_device, get_smooth_activation_stats


def setup_logger(output_dir):
    os.makedirs(output_dir, exist_ok=True)
    logger = logging.getLogger("qwen_shared_diag")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("[%(asctime)s] %(levelname)s %(message)s")
    stream = logging.StreamHandler(sys.stdout)
    stream.setFormatter(fmt)
    logger.addHandler(stream)
    file_handler = logging.FileHandler(os.path.join(output_dir, "run.log"))
    file_handler.setFormatter(fmt)
    logger.addHandler(file_handler)
    return logger


def normalize_group_size(value):
    if value is None:
        return None
    value = int(value)
    return value if value > 0 else None


def fake_quant_weight(weight, n_bits, group_size=None, symmetric=False):
    if n_bits >= 16:
        return weight.detach().float().cpu()
    x = weight.detach().float().cpu()
    original_shape = tuple(x.shape)
    group_size = normalize_group_size(group_size)
    pad = 0
    padded_shape = None
    if group_size is not None:
        last_dim = x.shape[-1]
        deficiency = last_dim % group_size
        pad = 0 if deficiency == 0 else group_size - deficiency
        if pad > 0:
            x = F.pad(x, (0, pad))
        padded_shape = tuple(x.shape)
        x = x.reshape(-1, group_size)
    else:
        x = x.reshape(-1, x.shape[-1])

    if symmetric:
        qmin = -(2 ** (n_bits - 1))
        qmax = 2 ** (n_bits - 1) - 1
        scale = x.abs().amax(dim=-1, keepdim=True).clamp(min=1e-5) / max(qmax, 1)
        x_int = torch.round(x / scale).clamp(qmin, qmax)
        x_dequant = x_int * scale
    else:
        qmin = 0
        qmax = 2 ** n_bits - 1
        xmin = x.amin(dim=-1, keepdim=True)
        xmax = x.amax(dim=-1, keepdim=True)
        scale = (xmax - xmin).clamp(min=1e-5) / max(qmax - qmin, 1)
        zero = torch.round(qmin - xmin / scale).clamp(qmin, qmax)
        x_int = (torch.round(x / scale) + zero).clamp(qmin, qmax)
        x_dequant = (x_int - zero) * scale

    if padded_shape is not None:
        x_dequant = x_dequant.reshape(padded_shape)
        if pad > 0:
            x_dequant = x_dequant.narrow(-1, 0, original_shape[-1])
    else:
        x_dequant = x_dequant.reshape(original_shape)
    return x_dequant


def tensor_p99(x):
    x = x.detach().abs().float().flatten().cpu()
    if x.numel() == 0:
        return 0.0
    return float(torch.quantile(x, 0.99).item())


def gini(values):
    values = values.detach().float().flatten().cpu().abs()
    if values.numel() == 0:
        return 0.0
    values = torch.sort(values)[0]
    total = values.sum()
    if total.item() <= 0:
        return 0.0
    n = values.numel()
    index = torch.arange(1, n + 1, dtype=values.dtype)
    return float(((2 * index - n - 1) * values).sum().item() / (n * total.item()))


def weight_distribution_stats(weight, topk=64):
    w = weight.detach().float().cpu()
    abs_w = w.abs().flatten()
    col_l2 = w.pow(2).sum(dim=0).sqrt()
    col_linf = w.abs().amax(dim=0)
    k = min(int(topk), col_l2.numel())
    energy = col_l2.pow(2)
    topk_energy = energy.topk(k).values.sum() if k > 0 else torch.tensor(0.0)
    total_energy = energy.sum().clamp(min=1e-12)
    return {
        "mean_abs": float(abs_w.mean().item()),
        "std": float(w.std().item()),
        "max_abs": float(abs_w.max().item()),
        "p99_abs": float(torch.quantile(abs_w, 0.99).item()),
        "p999_abs": float(torch.quantile(abs_w, 0.999).item()),
        "col_l2_median": float(col_l2.median().item()),
        "col_l2_p99": float(torch.quantile(col_l2, 0.99).item()),
        "col_l2_max": float(col_l2.max().item()),
        "col_linf_p99": float(torch.quantile(col_linf, 0.99).item()),
        "topk_energy_share": float((topk_energy / total_energy).item()),
        "col_l2_gini": gini(col_l2),
    }


def load_moe_plan(path):
    if not path:
        return {}
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    entries = payload.get("plan", payload if isinstance(payload, list) else [])
    return {entry["module_name"]: entry for entry in entries}


def module_wbits(module_name, role, args, plan):
    if role in ("router", "shared_expert_gate"):
        return int(args.router_wbits)
    entry = plan.get(module_name)
    if entry is not None and "wbits" in entry:
        return int(entry["wbits"])
    return int(args.wbits)


def select_routed_expert_indices(num_experts, sample):
    sample = int(sample)
    if sample <= 0 or sample >= num_experts:
        return list(range(num_experts))
    if sample == 1:
        return [0]
    step = (num_experts - 1) / float(sample - 1)
    indices = sorted({int(round(i * step)) for i in range(sample)})
    return indices


def iter_qwen_moe_linears(model, routed_expert_sample=4):
    for layer_idx, layer in enumerate(model.model.layers):
        mlp = layer.mlp
        prefix = f"model.layers.{layer_idx}.mlp"
        yield {
            "layer": layer_idx,
            "role": "router",
            "expert": -2,
            "proj": "gate",
            "name": f"{prefix}.gate",
            "module": mlp.gate,
            "sampled": True,
        }
        yield {
            "layer": layer_idx,
            "role": "shared_expert_gate",
            "expert": -1,
            "proj": "shared_expert_gate",
            "name": f"{prefix}.shared_expert_gate",
            "module": mlp.shared_expert_gate,
            "sampled": True,
        }
        expert_indices = select_routed_expert_indices(len(mlp.experts), routed_expert_sample)
        for expert_idx in expert_indices:
            expert = mlp.experts[expert_idx]
            for proj in ("gate_proj", "up_proj", "down_proj"):
                yield {
                    "layer": layer_idx,
                    "role": f"routed_{proj.replace('_proj', '')}",
                    "expert": expert_idx,
                    "proj": proj,
                    "name": f"{prefix}.experts.{expert_idx}.{proj}",
                    "module": getattr(expert, proj),
                    "sampled": routed_expert_sample > 0 and routed_expert_sample < len(mlp.experts),
                }
        for proj in ("gate_proj", "up_proj", "down_proj"):
            yield {
                "layer": layer_idx,
                "role": f"shared_{proj.replace('_proj', '')}",
                "expert": -1,
                "proj": proj,
                "name": f"{prefix}.shared_expert.{proj}",
                "module": getattr(mlp.shared_expert, proj),
                "sampled": True,
            }


def collect_weight_rows(model, stage, args, plan):
    rows = []
    items = list(iter_qwen_moe_linears(model, args.routed_expert_sample))
    for item in tqdm(items, desc=f"weight stats {stage}"):
        weight = item["module"].weight
        stats = weight_distribution_stats(weight, topk=args.moe_outlier_topk)
        bits = module_wbits(item["name"], item["role"], args, plan)
        qweight = fake_quant_weight(weight, bits, group_size=args.group_size, symmetric=args.symmetric)
        w = weight.detach().float().cpu()
        diff = qweight - w
        rel_mse = diff.pow(2).sum() / w.pow(2).sum().clamp(min=1e-12)
        q_cos = F.cosine_similarity(w.flatten(), qweight.flatten(), dim=0)
        row = {
            "stage": stage,
            "layer": item["layer"],
            "role": item["role"],
            "expert": item["expert"],
            "proj": item["proj"],
            "module_name": item["name"],
            "wbits": bits,
            "shape": "x".join(str(v) for v in tuple(weight.shape)),
            "sampled": item.get("sampled", False),
            "routed_expert_sample": args.routed_expert_sample,
            "quant_rel_mse": float(rel_mse.item()),
            "quant_cosine": float(q_cos.item()),
        }
        row.update(stats)
        rows.append(row)
    return rows


def write_csv(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not rows:
        with open(path, "w", newline="", encoding="utf-8") as f:
            f.write("")
        return
    keys = []
    seen = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                keys.append(key)
                seen.add(key)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def summarize_by_role(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["stage"], row["role"])].append(row)
    summary = []
    metrics = [
        "max_abs",
        "p99_abs",
        "p999_abs",
        "col_l2_p99",
        "topk_energy_share",
        "col_l2_gini",
        "quant_rel_mse",
        "quant_cosine",
    ]
    for (stage, role), items in sorted(grouped.items()):
        out = {"stage": stage, "role": role, "count": len(items)}
        for metric in metrics:
            vals = torch.tensor([float(item[metric]) for item in items], dtype=torch.float32)
            out[f"{metric}_mean"] = float(vals.mean().item())
            out[f"{metric}_median"] = float(vals.median().item())
            out[f"{metric}_p95"] = float(torch.quantile(vals, 0.95).item())
        summary.append(out)
    return summary


def compare_before_after(before_rows, after_rows):
    before = {row["module_name"]: row for row in before_rows}
    rows = []
    for row in after_rows:
        base = before.get(row["module_name"])
        if base is None:
            continue
        out = {
            "layer": row["layer"],
            "role": row["role"],
            "expert": row["expert"],
            "proj": row["proj"],
            "module_name": row["module_name"],
        }
        for metric in ("max_abs", "p99_abs", "p999_abs", "col_l2_p99", "topk_energy_share", "col_l2_gini"):
            denom = max(float(base[metric]), 1e-12)
            out[f"{metric}_before"] = float(base[metric])
            out[f"{metric}_after"] = float(row[metric])
            out[f"{metric}_ratio"] = float(row[metric]) / denom
        rows.append(out)
    return rows


def topk_indices(values, k):
    k = min(int(k), values.numel())
    if k <= 0:
        return torch.empty(0, dtype=torch.long)
    return torch.topk(values.float().cpu(), k=k, largest=True, sorted=False).indices.long()


def set_overlap(a, b):
    a_set = set(int(x) for x in a.tolist())
    b_set = set(int(x) for x in b.tolist())
    if not a_set and not b_set:
        return 1.0, 1.0
    inter = len(a_set & b_set)
    union = len(a_set | b_set)
    recall = inter / max(len(b_set), 1)
    jaccard = inter / max(union, 1)
    return jaccard, recall


def weight_error_scores(weight, bits, args):
    q = fake_quant_weight(weight, bits, group_size=args.group_size, symmetric=args.symmetric)
    w = weight.detach().float().cpu()
    return (w - q).pow(2).sum(dim=0)


def collect_ose_coverage_rows(model, args, plan):
    rows = []
    for item in tqdm(list(iter_qwen_moe_linears(model, args.routed_expert_sample)), desc="OSE coverage"):
        if item["proj"] not in ("gate_proj", "up_proj"):
            continue
        if item["role"] not in ("routed_gate", "routed_up", "shared_gate", "shared_up"):
            continue
        layer = item["layer"]
        weight = item["module"].weight.detach().float().cpu()
        bits = module_wbits(item["name"], item["role"], args, plan)
        current_score = args.moe_outlier_scores.get(item["name"])
        if current_score is None:
            current_idx = topk_indices(weight.abs().amax(dim=0), args.moe_outlier_topk)
            current_source = "module_weight_max"
        else:
            current_idx = topk_indices(current_score, args.moe_outlier_topk)
            current_source = f"module_{args.moe_outlier_score}"

        weight_max_idx = topk_indices(weight.abs().amax(dim=0), args.moe_outlier_topk)
        weight_error_idx = topk_indices(weight_error_scores(weight, bits, args), args.moe_outlier_topk)
        for target_name, target_idx in (
            ("module_weight_max", weight_max_idx),
            ("module_weight_error", weight_error_idx),
        ):
            jaccard, recall = set_overlap(current_idx, target_idx)
            rows.append({
                "layer": layer,
                "role": item["role"],
                "expert": item["expert"],
                "proj": item["proj"],
                "module_name": item["name"],
                "wbits": bits,
                "topk": args.moe_outlier_topk,
                "current_source": current_source,
                "target_source": target_name,
                "jaccard": jaccard,
                "recall": recall,
            })
    return rows


def collect_gate_inputs_and_router_counts(model, dataloader, num_samples, max_tokens, top_k, logger):
    model.eval()
    device = next(model.parameters()).device
    inputs = defaultdict(list)
    token_counts = defaultdict(int)
    router_counts = {}

    def append_input(name, tensor):
        hidden_dim = tensor.shape[-1]
        flat = tensor.detach().reshape(-1, hidden_dim)
        remaining = max_tokens - token_counts[name]
        if remaining <= 0:
            return
        take = min(remaining, flat.shape[0])
        inputs[name].append(flat[:take].float().cpu())
        token_counts[name] += take

    def hook(name, module, x, y):
        tensor = x[0] if isinstance(x, tuple) else x
        append_input(name, tensor)
        if name.endswith(".mlp.gate"):
            logits = y.detach().float()
            selected = torch.topk(logits.reshape(-1, logits.shape[-1]), k=top_k, dim=-1).indices.cpu()
            counts = torch.bincount(selected.flatten(), minlength=logits.shape[-1]).float()
            if name in router_counts:
                router_counts[name] += counts
            else:
                router_counts[name] = counts

    handles = []
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        if name.endswith(".mlp.gate") or name.endswith(".mlp.shared_expert_gate"):
            handles.append(module.register_forward_hook(lambda m, x, y, n=name: hook(n, m, x, y)))

    with torch.no_grad():
        for i in tqdm(range(min(num_samples, len(dataloader))), desc="collect gate inputs"):
            model(dataloader[i][0].to(device))

    for handle in handles:
        handle.remove()

    logger.info("collected gate inputs for %d modules", len(inputs))
    inputs = {name: torch.cat(chunks, dim=0) for name, chunks in inputs.items() if chunks}
    return inputs, router_counts


def collect_gate_error_rows(model, inputs, router_counts, args):
    rows = []
    top_k = int(model.config.num_experts_per_tok)
    for layer_idx, layer in enumerate(model.model.layers):
        mlp = layer.mlp
        for role, module, name in (
            ("router", mlp.gate, f"model.layers.{layer_idx}.mlp.gate"),
            ("shared_expert_gate", mlp.shared_expert_gate, f"model.layers.{layer_idx}.mlp.shared_expert_gate"),
        ):
            if name not in inputs:
                continue
            x = inputs[name]
            w = module.weight.detach().float().cpu()
            qw = fake_quant_weight(w, args.router_wbits, group_size=args.group_size, symmetric=args.symmetric)
            y = F.linear(x, w, module.bias.detach().float().cpu() if module.bias is not None else None)
            yq = F.linear(x, qw, module.bias.detach().float().cpu() if module.bias is not None else None)
            diff = yq - y
            rel_mse = diff.pow(2).mean() / y.pow(2).mean().clamp(min=1e-12)
            row = {
                "layer": layer_idx,
                "role": role,
                "module_name": name,
                "tokens": int(x.shape[0]),
                "linear_rel_mse": float(rel_mse.item()),
                "linear_abs_err_p99": tensor_p99(diff),
            }
            if role == "router":
                top_orig = torch.topk(y, k=top_k, dim=-1).indices
                top_quant = torch.topk(yq, k=top_k, dim=-1).indices
                overlap = []
                for a, b in zip(top_orig, top_quant):
                    overlap.append(len(set(a.tolist()) & set(b.tolist())) / top_k)
                row["router_topk_overlap"] = float(torch.tensor(overlap).mean().item())
                counts = router_counts.get(name)
                if counts is not None:
                    row["router_top1_count"] = float(counts.max().item())
                    row["router_usage_gini"] = gini(counts)
            else:
                sig = torch.sigmoid(y)
                sigq = torch.sigmoid(yq)
                sigdiff = sigq - sig
                row["sigmoid_rel_mse"] = float((sigdiff.pow(2).mean() / sig.pow(2).mean().clamp(min=1e-12)).item())
                row["sigmoid_abs_err_p99"] = tensor_p99(sigdiff)
            rows.append(row)
    return rows


def build_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=os.path.join(ROOT, "local_models", "Qwen1.5-MoE-A2.7B"))
    parser.add_argument("--cache_dir", default=os.path.join(ROOT, "cache"))
    parser.add_argument("--output_dir", default=os.path.join(ROOT, "experiments", "qwen_shared_expert_migration_diag"))
    parser.add_argument("--moe_quant_plan", default=os.path.join(ROOT, "plans", "qwen_routed_range", "moe_quant_plan_routed_range.json"))
    parser.add_argument("--calib_dataset", default="wikitext2", choices=["wikitext2", "ptb", "c4", "mix", "pile"])
    parser.add_argument("--nsamples", type=int, default=128)
    parser.add_argument("--functional_nsamples", type=int, default=8)
    parser.add_argument("--seq_length", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--wbits", type=int, default=4)
    parser.add_argument("--abits", type=int, default=8)
    parser.add_argument("--router_wbits", type=int, default=8)
    parser.add_argument("--router_abits", type=int, default=8)
    parser.add_argument("--group_size", type=int, default=-1)
    parser.add_argument("--act_group_size", type=int, default=-1)
    parser.add_argument("--moe_outlier_topk", type=int, default=64)
    parser.add_argument("--fc1_scale_merge", default="act_p99")
    parser.add_argument("--act_mean_beta", type=float, default=1.0)
    parser.add_argument("--alpha", type=float, default=0.6)
    parser.add_argument("--otsu_ratio", type=float, default=0.65)
    parser.add_argument("--otsu_smooth_rate", type=float, default=0.7)
    parser.add_argument("--moe_down_smooth_mode", default="otsu", choices=["duquant", "otsu"])
    parser.add_argument("--max_gate_tokens", type=int, default=1024)
    parser.add_argument("--routed_expert_sample", type=int, default=4, help="number of routed experts sampled per layer for weight/OSE diagnostics; <=0 means all experts")
    parser.add_argument("--symmetric", action="store_true")
    args = parser.parse_args()

    args.model_name = "qwen2_moe"
    args.attn_implementation = "eager"
    args.quant_method = "duquant"
    args.smooth = True
    args.moe_outlier_score = "smooth_scale"
    args.moe_outlier_scores = {}
    args.moe_outlier_quant = "w8a8"
    args.w_dynamic_method = "per_channel_tensor"
    args.a_dynamic_method = "per_token"
    args.router_w_dynamic_method = "per_channel_kl_top0"
    args.lac = None
    args.swc = None
    args.block_size = 128
    args.max_rotation_step = 256
    args.permutation_times = 1
    args.expert_token_num_ratio = 2.0
    return args


def main():
    args = build_args()
    logger = setup_logger(args.output_dir)
    start = time.perf_counter()
    logger.info("output_dir: %s", args.output_dir)
    logger.info("model: %s", args.model)
    logger.info("nsamples=%d functional_nsamples=%d seq_length=%d", args.nsamples, args.functional_nsamples, args.seq_length)

    os.makedirs(args.cache_dir, exist_ok=True)
    lm = LMClass(args)
    lm.seqlen = args.seq_length
    args.expert_ratio = lm.model.config.num_experts_per_tok / lm.model.config.num_experts
    args.expert_token_num = int(args.expert_ratio * args.seq_length * args.expert_token_num_ratio)
    plan = load_moe_plan(args.moe_quant_plan)
    logger.info("loaded plan entries: %d", len(plan))

    cache_dataloader = os.path.join(
        args.cache_dir,
        f"dataloader_{args.model_name}_{args.calib_dataset}_{args.nsamples}_{args.seq_length}.cache",
    )
    if not os.path.exists(cache_dataloader):
        pattern = os.path.join(
            args.cache_dir,
            f"dataloader_{args.model_name}_{args.calib_dataset}_*_{args.seq_length}.cache",
        )
        candidates = sorted(glob.glob(pattern))
        if candidates:
            cache_dataloader = candidates[-1]
            logger.info("exact dataloader cache missing; fallback to %s", cache_dataloader)

    if os.path.exists(cache_dataloader):
        dataloader = torch.load(cache_dataloader)
        if len(dataloader) > args.nsamples:
            dataloader = dataloader[: args.nsamples]
        logger.info("loaded dataloader cache: %s", cache_dataloader)
    else:
        dataloader, _ = get_loaders(
            args.calib_dataset,
            nsamples=args.nsamples,
            seed=args.seed,
            model=args.model,
            seqlen=lm.seqlen,
        )
        torch.save(dataloader, cache_dataloader)
        logger.info("saved dataloader cache: %s", cache_dataloader)

    before_rows = collect_weight_rows(lm.model, "before_smooth", args, plan)
    write_csv(os.path.join(args.output_dir, "weight_stats_before.csv"), before_rows)

    logger.info("moving model to GPU and computing smooth stats")
    convert_device(lm, args.model_name)
    stats = get_smooth_activation_stats(
        lm.model,
        dataloader,
        args.nsamples,
        collect_moe_act_p99=True,
        collect_act_scales=True,
        collect_act_per_channel_scales=True,
    )
    moe_act_p99s = stats["moe_act_p99s"]
    moe_fc1_smooth_scales = build_moe_fc1_smooth_scales(
        moe_act_p99s,
        act_mean_beta=args.act_mean_beta,
        model=lm.model,
    )
    logger.info("built moe_fc1_smooth_scales entries: %d", len(moe_fc1_smooth_scales))

    logger.info("applying smooth_lm")
    smooth_lm(
        lm.model,
        stats["act_scales"],
        stats["act_per_channel_scales"],
        {},
        {},
        {},
        fc1_scale_merge=args.fc1_scale_merge,
        alpha=args.alpha,
        otsu_ratio=args.otsu_ratio,
        otsu_smooth_rate=args.otsu_smooth_rate,
        moe_down_smooth_mode=args.moe_down_smooth_mode,
        moe_fc1_smooth_scales=moe_fc1_smooth_scales,
        logger=logger,
    )
    args.moe_outlier_scores = prepare_moe_outlier_scores(
        lm.model,
        score_method=args.moe_outlier_score,
        moe_fc1_smooth_scales=moe_fc1_smooth_scales,
        args=args,
        model_name=args.model_name,
        logger=logger,
    )

    gate_inputs, router_counts = collect_gate_inputs_and_router_counts(
        lm.model,
        dataloader,
        args.functional_nsamples,
        args.max_gate_tokens,
        int(lm.model.config.num_experts_per_tok),
        logger,
    )
    gate_error_rows = collect_gate_error_rows(lm.model, gate_inputs, router_counts, args)
    write_csv(os.path.join(args.output_dir, "gate_functional_error_after_smooth.csv"), gate_error_rows)

    after_rows = collect_weight_rows(lm.model, "after_smooth", args, plan)
    write_csv(os.path.join(args.output_dir, "weight_stats_after.csv"), after_rows)
    write_csv(os.path.join(args.output_dir, "weight_stats_summary.csv"), summarize_by_role(before_rows + after_rows))
    write_csv(os.path.join(args.output_dir, "migration_stat_ratios.csv"), compare_before_after(before_rows, after_rows))

    coverage_rows = collect_ose_coverage_rows(lm.model, args, plan)
    write_csv(os.path.join(args.output_dir, "ose_coverage_stats.csv"), coverage_rows)

    summary = {
        "output_dir": args.output_dir,
        "model": args.model,
        "nsamples": args.nsamples,
        "functional_nsamples": args.functional_nsamples,
        "seq_length": args.seq_length,
        "routed_expert_sample": args.routed_expert_sample,
        "num_weight_rows": len(after_rows),
        "num_coverage_rows": len(coverage_rows),
        "num_gate_error_rows": len(gate_error_rows),
        "elapsed_seconds": time.perf_counter() - start,
        "files": [
            "weight_stats_before.csv",
            "weight_stats_after.csv",
            "weight_stats_summary.csv",
            "migration_stat_ratios.csv",
            "ose_coverage_stats.csv",
            "gate_functional_error_after_smooth.csv",
        ],
    }
    with open(os.path.join(args.output_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    logger.info("done: %s", json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
