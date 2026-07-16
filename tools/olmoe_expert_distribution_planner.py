#!/usr/bin/env python3
import argparse
import csv
import hashlib
import json
import math
import os
import re
import sys
import time
from collections import Counter, defaultdict
from dataclasses import asdict
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
from quantize.moe_outlier_score import prepare_moe_outlier_scores
from utils import (
    get_act_per_channel_scales,
    get_act_samples,
    get_act_scales,
    get_moe_act_means,
    get_moe_act_p99s,
    get_router_logits,
    get_weight_scores,
)

try:
    from olmoe_moe_precision_planner import parse_candidate_quant, quant_minmax
except ImportError:
    from tools.olmoe_moe_precision_planner import parse_candidate_quant, quant_minmax


class PrintLogger:
    def __init__(self, verbose=False):
        self.verbose = bool(verbose)

    def _should_print(self, message):
        if self.verbose:
            return True
        text = str(message)
        noisy_prefixes = ("[smooth_lm]", "scale=")
        noisy_exact = {"smooth qkv", "calcu_outlier_mask", "get_scale", "build_moe_fc1_smooth_scales"}
        return not (text.startswith(noisy_prefixes) or text in noisy_exact)

    def info(self, message):
        if self._should_print(message):
            print(message)

    def warning(self, message):
        print(f"WARNING: {message}")

    def warn(self, message):
        self.warning(message)


def safe_cache_tag(value):
    return str(value).replace("/", "_").replace("\\", "_").replace(":", "_").replace(" ", "_")


def smooth_stats_cache_path(args, stat_name):
    key = "|".join(
        [
            "smooth_stats_v2",
            str(args.model),
            str(args.model_name),
            str(args.calib_dataset),
            str(args.nsamples),
            str(args.seq_length),
            str(args.seed),
            str(args.attn_implementation),
        ]
    )
    digest = hashlib.md5(key.encode("utf-8")).hexdigest()[:16]
    model_tag = safe_cache_tag(Path(args.model).name)
    filename = (
        f"smooth_stats_v2_{stat_name}_{model_tag}_{args.model_name}_"
        f"{args.calib_dataset}_{args.nsamples}_{args.seq_length}_{args.seed}_{digest}.pt"
    )
    return os.path.join(args.cache_dir, filename)


def load_or_compute_smooth_stat(args, logger, stat_name, compute_fn):
    cache_path = smooth_stats_cache_path(args, stat_name)
    if not args.disable_act_stats_cache and not args.refresh_act_stats_cache and os.path.exists(cache_path):
        try:
            logger.info(f"load {stat_name} from {cache_path}")
            return torch.load(cache_path, map_location="cpu")
        except Exception as exc:
            logger.warning(f"failed to load {stat_name} cache {cache_path}: {exc}; recomputing")

    value = compute_fn()
    if not args.disable_act_stats_cache:
        Path(args.cache_dir).mkdir(parents=True, exist_ok=True)
        tmp_path = f"{cache_path}.tmp.{os.getpid()}"
        torch.save(value, tmp_path)
        os.replace(tmp_path, cache_path)
        logger.info(f"save {stat_name} to {cache_path}")
    return value


def save_json(obj, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def parse_str_list(text):
    if text is None or str(text).strip() == "":
        return []
    return [item.strip() for item in str(text).split(",") if item.strip()]


def normalize_group_size(value):
    if value is None:
        return None
    value = int(value)
    if value <= 0:
        return None
    return value


def cleanup_memory():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    import gc

    gc.collect()


class DistributionStats:
    def __init__(self, max_samples=16384, sample_per_call=2048):
        self.max_samples = int(max_samples)
        self.sample_per_call = int(sample_per_call)
        self.count = 0
        self.sum_abs = 0.0
        self.sum_sq = 0.0
        self.sum_fourth = 0.0
        self.max_abs = 0.0
        self._samples = []
        self._sample_count = 0

    def update(self, x):
        if x is None or not torch.is_tensor(x) or x.numel() == 0:
            return
        xf = x.detach().float().reshape(-1)
        if xf.numel() == 0:
            return
        xf = torch.nan_to_num(xf, nan=0.0, posinf=0.0, neginf=0.0)
        abs_x = xf.abs()
        self.count += int(abs_x.numel())
        self.sum_abs += float(abs_x.sum().item())
        sq = xf * xf
        self.sum_sq += float(sq.sum().item())
        self.sum_fourth += float((sq * sq).sum().item())
        self.max_abs = max(self.max_abs, float(abs_x.max().item()))

        if self.max_samples <= 0:
            return
        take = min(int(abs_x.numel()), max(1, self.sample_per_call))
        if take < int(abs_x.numel()):
            idx = torch.randint(0, int(abs_x.numel()), (take,), device=abs_x.device)
            sample = abs_x.index_select(0, idx).cpu()
        else:
            sample = abs_x.cpu()
        self._samples.append(sample)
        self._sample_count += int(sample.numel())
        if self._sample_count > 2 * self.max_samples:
            self._compress_samples()

    def _compress_samples(self):
        if not self._samples:
            return torch.empty(0)
        samples = torch.cat(self._samples)
        if samples.numel() > self.max_samples:
            idx = torch.randperm(samples.numel())[: self.max_samples]
            samples = samples.index_select(0, idx)
        self._samples = [samples]
        self._sample_count = int(samples.numel())
        return samples

    def finalize(self, prefix):
        eps = 1e-12
        samples = self._compress_samples()
        count = max(int(self.count), 1)
        mean_abs = self.sum_abs / count
        rms = math.sqrt(max(self.sum_sq / count, 0.0))
        if rms > eps:
            kurtosis = (self.sum_fourth / count) / max(rms ** 4, eps)
        else:
            kurtosis = 0.0

        if samples.numel() > 0:
            samples = samples.float()
            p50 = float(torch.quantile(samples, 0.50).item())
            p90 = float(torch.quantile(samples, 0.90).item())
            p99 = float(torch.quantile(samples, 0.99).item())
            p999 = float(torch.quantile(samples, 0.999).item()) if samples.numel() >= 1000 else p99
        else:
            p50 = p90 = p99 = p999 = 0.0

        gaussian_max = rms * math.sqrt(max(2.0 * math.log(max(count, 2)), eps))
        return {
            f"{prefix}_count": int(self.count),
            f"{prefix}_mean_abs": float(mean_abs),
            f"{prefix}_rms": float(rms),
            f"{prefix}_max_abs": float(self.max_abs),
            f"{prefix}_p50_abs": float(p50),
            f"{prefix}_p90_abs": float(p90),
            f"{prefix}_p99_abs": float(p99),
            f"{prefix}_p999_abs": float(p999),
            f"{prefix}_tail_p99_p50": float(p99 / max(p50, eps)),
            f"{prefix}_tail_p999_p50": float(p999 / max(p50, eps)),
            f"{prefix}_range_max_p99": float(self.max_abs / max(p99, eps)),
            f"{prefix}_range_max_gaussian": float(self.max_abs / max(gaussian_max, eps)),
            f"{prefix}_kurtosis_raw": float(kurtosis),
        }


def build_lm(args):
    lm_args = argparse.Namespace(
        model=args.model,
        model_name=args.model_name,
        batch_size=1,
        attn_implementation=args.attn_implementation,
    )
    lm = LMClass(lm_args)
    lm.seqlen = args.seq_length
    lm.model.eval()
    for param in lm.model.parameters():
        param.requires_grad = False
    return lm


def load_calibration(args):
    cache_path = Path(args.cache_dir) / f"planner_dataloader_{args.model_name}_{args.calib_dataset}_{args.nsamples}_{args.seq_length}_{args.seed}.cache"
    if args.reuse_dataloader_cache and cache_path.exists():
        return torch.load(cache_path, map_location="cpu")[: args.nsamples]
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


def apply_optional_smooth(lm, args, dataloader):
    if not args.smooth:
        return
    logger = PrintLogger(verbose=args.verbose_smooth_log)
    logger.info(
        f"Applying smooth before distribution planning: "
        f"fc1_scale_merge={args.fc1_scale_merge}, act_mean_beta={args.act_mean_beta}"
    )
    model = lm.model
    if args.fc1_scale_merge not in ("act_mean", "act_p99"):
        raise ValueError("--fc1_scale_merge must be act_mean or act_p99")
    moe_stat_name = "moe_act_means" if args.fc1_scale_merge == "act_mean" else "moe_act_p99s"
    moe_stat_fn = get_moe_act_means if args.fc1_scale_merge == "act_mean" else get_moe_act_p99s
    moe_act_stats = load_or_compute_smooth_stat(
        args,
        logger,
        moe_stat_name,
        lambda: moe_stat_fn(model, dataloader, args.nsamples),
    )
    moe_fc1_smooth_scales = build_moe_fc1_smooth_scales(
        moe_act_stats,
        act_mean_beta=args.act_mean_beta,
        model=model,
    )
    logger.info(f"built moe_fc1_smooth_scales entries: {len(moe_fc1_smooth_scales)}")
    act_scales = load_or_compute_smooth_stat(
        args,
        logger,
        "act_scales",
        lambda: get_act_scales(model, dataloader, args.nsamples),
    )
    act_per_channel_scales = load_or_compute_smooth_stat(
        args,
        logger,
        "act_per_channel_scales",
        lambda: get_act_per_channel_scales(model, dataloader, args.nsamples),
    )
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
        logger=logger,
    )
    args.moe_outlier_scores = prepare_moe_outlier_scores(
        model,
        score_method=args.moe_outlier_score,
        moe_fc1_smooth_scales=moe_fc1_smooth_scales,
        args=args,
        model_name=args.model_name,
        logger=logger,
    )
    cleanup_memory()


def parse_expert_name(name):
    m = re.fullmatch(r"model\.layers\.(\d+)\.mlp\.experts\.(\d+)", name)
    if m is None:
        return None
    return int(m.group(1)), int(m.group(2))


def _topk_outlier_input_channels_for_weight(weight, args, module_name):
    topk = min(int(getattr(args, "moe_outlier_topk", 0)), int(weight.shape[1]))
    if topk <= 0:
        return torch.empty(0, dtype=torch.long)

    scores_by_module = getattr(args, "moe_outlier_scores", None) or {}
    if module_name in scores_by_module:
        scores = scores_by_module[module_name].detach().float().cpu().flatten()
    elif getattr(args, "moe_outlier_score", "weight_max") == "weight_max":
        scores = weight.detach().abs().amax(dim=0).float().cpu()
    else:
        raise KeyError(f"missing moe_outlier_scores for {module_name}")

    if scores.numel() != int(weight.shape[1]):
        raise ValueError(
            f"moe_outlier_scores for {module_name} have {scores.numel()} channels, "
            f"expected {int(weight.shape[1])}"
        )
    scores = torch.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0).clamp(min=0)
    if scores.numel() == 0 or scores.max().item() <= 0:
        raise ValueError(f"invalid all-zero moe_outlier_scores for {module_name}")
    return torch.topk(scores, k=topk, largest=True, sorted=True).indices.to(torch.long)


def _weight_for_post_outlier_residual_stats(weight, args, layer_idx, expert_idx, proj_name):
    if proj_name not in ("gate_proj", "up_proj"):
        return weight, 0
    module_name = f"model.layers.{layer_idx}.mlp.experts.{expert_idx}.{proj_name}"
    outlier_idx = _topk_outlier_input_channels_for_weight(weight, args, module_name)
    if outlier_idx.numel() == 0:
        return weight, 0
    residual_weight = weight.detach().clone()
    residual_weight.index_fill_(1, outlier_idx.to(residual_weight.device), 0)
    return residual_weight, int(outlier_idx.numel())


def _relative_weight_quant_error(weight, candidate, weight_channel_group_size=None):
    if weight is None or not torch.is_tensor(weight) or weight.numel() == 0:
        return 0.0
    if int(candidate.wbits) >= 16:
        return 0.0
    wf = weight.detach().float()
    qwf = quant_minmax(
        weight,
        int(candidate.wbits),
        reduce_dim=1,
        symmetric=False,
        group_size=candidate.weight_group_size,
        output_channel_group_size=weight_channel_group_size,
    ).detach().float()
    diff_sq = (wf - qwf).pow(2).sum()
    base_sq = wf.pow(2).sum().clamp(min=1e-12)
    return float((diff_sq / base_sq).item())


def quant_name(candidate):
    return candidate.name


def collect_weight_rows(model, args):
    rows = {}
    low_qerr_candidate = getattr(args, "_qerr_low_candidate", parse_candidate_quant(args.low_quant))
    high_qerr_candidate = getattr(args, "_qerr_high_candidate", parse_candidate_quant(args.high_quant))
    weight_score_projs = set(parse_str_list(getattr(args, "weight_score_projs", "gate_proj,up_proj,down_proj")))
    valid_projs = {"gate_proj", "up_proj", "down_proj"}
    if not weight_score_projs:
        raise ValueError("--weight_score_projs must select at least one projection")
    invalid_projs = sorted(weight_score_projs - valid_projs)
    if invalid_projs:
        raise ValueError(f"invalid --weight_score_projs entries: {invalid_projs}")
    for name, module in model.named_modules():
        parsed = parse_expert_name(name)
        if parsed is None:
            continue
        layer_idx, expert_idx = parsed
        item = {
            "layer_index": int(layer_idx),
            "expert_index": int(expert_idx),
        }
        tail_vals = []
        range_vals = []
        kurt_vals = []
        qerr_vals = []
        qerr_gain_vals = []
        for proj_name in ("gate_proj", "up_proj", "down_proj"):
            proj = getattr(module, proj_name)
            stats = DistributionStats(
                max_samples=args.weight_max_samples,
                sample_per_call=args.weight_sample_per_call,
            )
            weight_for_stats, removed_channels = _weight_for_post_outlier_residual_stats(
                proj.weight,
                args,
                layer_idx,
                expert_idx,
                proj_name,
            )
            stats.update(weight_for_stats)
            summary = stats.finalize(f"weight_{proj_name}")
            item.update(summary)
            item[f"weight_{proj_name}_outlier_removed_channels"] = int(removed_channels)
            low_qerr = _relative_weight_quant_error(
                weight_for_stats,
                low_qerr_candidate,
                weight_channel_group_size=args.weight_channel_group_size,
            )
            high_qerr = _relative_weight_quant_error(
                weight_for_stats,
                high_qerr_candidate,
                weight_channel_group_size=args.weight_channel_group_size,
            )
            item[f"weight_{proj_name}_residual_qerr"] = float(low_qerr)
            item[f"weight_{proj_name}_residual_qerr_gain"] = float(max(low_qerr - high_qerr, 0.0))
            if proj_name in weight_score_projs:
                tail_vals.append(summary[f"weight_{proj_name}_tail_p99_p50"])
                range_vals.append(summary[f"weight_{proj_name}_range_max_p99"])
                kurt_vals.append(summary[f"weight_{proj_name}_kurtosis_raw"])
                qerr_vals.append(low_qerr)
                qerr_gain_vals.append(max(low_qerr - high_qerr, 0.0))
        item["weight_tail"] = float(sum(tail_vals) / max(len(tail_vals), 1))
        item["weight_range"] = float(sum(range_vals) / max(len(range_vals), 1))
        item["weight_kurtosis"] = float(sum(kurt_vals) / max(len(kurt_vals), 1))
        item["weight_residual_qerr"] = float(sum(qerr_vals) / max(len(qerr_vals), 1))
        item["weight_residual_qerr_gain"] = float(sum(qerr_gain_vals) / max(len(qerr_gain_vals), 1))
        rows[(layer_idx, expert_idx)] = item
    return rows


def collect_activation_rows(model, dataloader, args, device):
    input_stats = {}
    down_stats = {}
    route_counts = {}
    route_mass = {}
    num_tokens_seen = defaultdict(int)
    num_assignments = defaultdict(int)
    handles = []

    layers = model.model.layers
    for layer_idx, layer in enumerate(layers):
        moe = getattr(layer, "mlp", None)
        if moe is None or not hasattr(moe, "experts") or not hasattr(moe, "gate"):
            continue
        num_experts = len(moe.experts)
        route_counts[layer_idx] = torch.zeros(num_experts, dtype=torch.long)
        route_mass[layer_idx] = torch.zeros(num_experts, dtype=torch.float64)
        top_k = int(getattr(moe, "top_k", getattr(model.config, "num_experts_per_tok", 2)))
        norm_topk_prob = bool(getattr(moe, "norm_topk_prob", False))

        for expert_idx, expert in enumerate(moe.experts):
            input_stats[(layer_idx, expert_idx)] = DistributionStats(
                max_samples=args.activation_max_samples,
                sample_per_call=args.activation_sample_per_call,
            )
            down_stats[(layer_idx, expert_idx)] = DistributionStats(
                max_samples=args.activation_max_samples,
                sample_per_call=args.activation_sample_per_call,
            )

            def make_expert_pre_hook(l_idx, e_idx):
                def hook(mod, inputs):
                    x = inputs[0] if inputs else None
                    input_stats[(l_idx, e_idx)].update(x)

                return hook

            def make_down_pre_hook(l_idx, e_idx):
                def hook(mod, inputs):
                    x = inputs[0] if inputs else None
                    down_stats[(l_idx, e_idx)].update(x)

                return hook

            handles.append(expert.register_forward_pre_hook(make_expert_pre_hook(layer_idx, expert_idx)))
            handles.append(expert.down_proj.register_forward_pre_hook(make_down_pre_hook(layer_idx, expert_idx)))

        def make_gate_hook(l_idx, n_exp, k, normalize_topk):
            def hook(mod, inputs, output):
                logits = output[0] if isinstance(output, tuple) else output
                if not torch.is_tensor(logits) or logits.numel() == 0:
                    return
                logits = logits.reshape(-1, n_exp)
                routing_weights = F.softmax(logits.float(), dim=1)
                top_weights, top_idx = torch.topk(routing_weights, k=k, dim=-1)
                if normalize_topk:
                    top_weights = top_weights / top_weights.sum(dim=-1, keepdim=True).clamp(min=1e-12)
                route_counts[l_idx] += torch.bincount(top_idx.reshape(-1).cpu(), minlength=n_exp)
                mass = torch.zeros(n_exp, dtype=torch.float64)
                mass.scatter_add_(0, top_idx.reshape(-1).cpu(), top_weights.reshape(-1).double().cpu())
                route_mass[l_idx] += mass
                num_tokens_seen[l_idx] += int(logits.shape[0])
                num_assignments[l_idx] += int(top_idx.numel())

            return hook

        handles.append(moe.gate.register_forward_hook(make_gate_hook(layer_idx, num_experts, top_k, norm_topk_prob)))

    try:
        model.config.use_cache = False
        for batch in tqdm(dataloader, desc="collect expert distribution stats"):
            input_ids = batch[0].to(device)
            with torch.no_grad():
                model(input_ids=input_ids, use_cache=False)
            del input_ids
    finally:
        for handle in handles:
            handle.remove()

    rows = {}
    for key in sorted(input_stats):
        layer_idx, expert_idx = key
        item = {
            "layer_index": int(layer_idx),
            "expert_index": int(expert_idx),
            "route_count": int(route_counts[layer_idx][expert_idx].item()),
            "route_mass": float(route_mass[layer_idx][expert_idx].item()),
            "num_tokens_seen": int(num_tokens_seen[layer_idx]),
            "num_assignments": int(num_assignments[layer_idx]),
        }
        total_count = max(int(num_assignments[layer_idx]), 1)
        total_mass = max(float(route_mass[layer_idx].sum().item()), 1e-12)
        item["route_frequency"] = float(item["route_count"] / total_count)
        item["route_mass_fraction"] = float(item["route_mass"] / total_mass)
        item.update(input_stats[key].finalize("input"))
        item.update(down_stats[key].finalize("down"))
        item["input_tail"] = item["input_tail_p99_p50"]
        item["input_range"] = item["input_range_max_p99"]
        item["input_kurtosis"] = item["input_kurtosis_raw"]
        item["down_tail"] = item["down_tail_p99_p50"]
        item["down_range"] = item["down_range_max_p99"]
        item["down_kurtosis"] = item["down_kurtosis_raw"]
        rows[key] = item
    return rows


def rank_values(rows, key):
    indexed = [(idx, float(row.get(key, 0.0))) for idx, row in enumerate(rows)]
    indexed.sort(key=lambda item: item[1])
    ranks = [0.0] * len(rows)
    denom = max(len(rows) - 1, 1)
    for rank, (idx, _) in enumerate(indexed):
        ranks[idx] = float(rank / denom)
    return ranks


def score_component(value, score_transform):
    value = max(float(value), 0.0)
    if score_transform == "log":
        return math.log1p(value)
    if score_transform == "raw":
        return value
    raise ValueError(f"unknown score_transform: {score_transform}")


def add_scores(
    rows,
    route_alpha,
    score_transform="log",
    input_score_coef=1.0,
    down_score_coef=1.0,
    weight_score_coef=0.5,
):
    input_score_coef = float(input_score_coef)
    down_score_coef = float(down_score_coef)
    weight_score_coef = float(weight_score_coef)
    for row in rows:
        row["tail_score"] = (
            input_score_coef * score_component(row.get("input_tail", 0.0), score_transform)
            + down_score_coef * score_component(row.get("down_tail", 0.0), score_transform)
            + weight_score_coef * score_component(row.get("weight_tail", 0.0), score_transform)
        )
        row["range_score"] = (
            input_score_coef * score_component(row.get("input_range", 0.0), score_transform)
            + down_score_coef * score_component(row.get("down_range", 0.0), score_transform)
            + weight_score_coef * score_component(row.get("weight_range", 0.0), score_transform)
        )
        row["kurtosis_score"] = (
            input_score_coef * score_component(row.get("input_kurtosis", 0.0), score_transform)
            + down_score_coef * score_component(row.get("down_kurtosis", 0.0), score_transform)
            + weight_score_coef * score_component(row.get("weight_kurtosis", 0.0), score_transform)
        )
        row["input_score"] = row["tail_score"] + row["range_score"]
        row["down_score"] = (
            score_component(row.get("down_tail", 0.0), score_transform)
            + score_component(row.get("down_range", 0.0), score_transform)
            + score_component(row.get("down_kurtosis", 0.0), score_transform)
        )
        row["weight_score"] = (
            score_component(row.get("weight_tail", 0.0), score_transform)
            + score_component(row.get("weight_range", 0.0), score_transform)
            + score_component(row.get("weight_kurtosis", 0.0), score_transform)
        )
        random_key = f"{int(row['layer_index'])}:{int(row['expert_index'])}".encode("utf-8")
        row["random_score"] = int(hashlib.sha256(random_key).hexdigest()[:16], 16) / float(16 ** 16)

    for base_key in ("tail_score", "range_score", "kurtosis_score", "input_score", "down_score", "weight_score"):
        ranks = rank_values(rows, base_key)
        for row, rank in zip(rows, ranks):
            row[f"{base_key}_rank"] = rank

    by_layer = defaultdict(list)
    for row in rows:
        by_layer[int(row["layer_index"])].append(row)
    for layer_rows in by_layer.values():
        mean_route = sum(float(row["route_mass_fraction"]) for row in layer_rows) / max(len(layer_rows), 1)
        for row in layer_rows:
            route_norm = float(row["route_mass_fraction"]) / max(mean_route, 1e-12)
            row["route_weight"] = float(max(route_norm, 0.0) ** float(route_alpha))

    for row in rows:
        input_kurt = max(row.get("input_kurtosis", 0.0), 0.0)
        down_kurt = max(row.get("down_kurtosis", 0.0), 0.0)
        weight_kurt = max(row.get("weight_kurtosis", 0.0), 0.0)

        row["input_kurtosis_log_score"] = math.log1p(input_kurt)
        row["down_kurtosis_log_score"] = math.log1p(down_kurt)
        row["weight_kurtosis_log_score"] = math.log1p(weight_kurt)
        row["kurtosis_equal_log_score"] = (
            row["input_kurtosis_log_score"]
            + row["down_kurtosis_log_score"]
            + row["weight_kurtosis_log_score"]
        )
        row["input_kurtosis_raw_score"] = float(input_kurt)
        row["down_kurtosis_raw_score"] = float(down_kurt)
        row["weight_kurtosis_raw_score"] = float(weight_kurt)
        row["kurtosis_equal_raw_score"] = float(input_kurt + down_kurt + weight_kurt)

        row["combo_score"] = (
            row["tail_score_rank"] + row["range_score_rank"] + row["kurtosis_score_rank"]
        ) / 3.0
        row["routed_tail_score"] = row["tail_score"] * row["route_weight"]
        row["routed_range_score"] = row["range_score"] * row["route_weight"]
        row["routed_kurtosis_score"] = row["kurtosis_score"] * row["route_weight"]
        row["routed_input_kurtosis_log_score"] = row["input_kurtosis_log_score"] * row["route_weight"]
        row["routed_down_kurtosis_log_score"] = row["down_kurtosis_log_score"] * row["route_weight"]
        row["routed_weight_kurtosis_log_score"] = row["weight_kurtosis_log_score"] * row["route_weight"]
        row["routed_kurtosis_equal_log_score"] = row["kurtosis_equal_log_score"] * row["route_weight"]
        row["routed_input_kurtosis_raw_score"] = row["input_kurtosis_raw_score"] * row["route_weight"]
        row["routed_down_kurtosis_raw_score"] = row["down_kurtosis_raw_score"] * row["route_weight"]
        row["routed_weight_kurtosis_raw_score"] = row["weight_kurtosis_raw_score"] * row["route_weight"]
        row["routed_kurtosis_equal_raw_score"] = row["kurtosis_equal_raw_score"] * row["route_weight"]
        row["routed_combo_score"] = row["combo_score"] * row["route_weight"]
        row["residual_qerr_score"] = float(max(row.get("weight_residual_qerr", 0.0), 0.0))
        row["residual_qerr_gain_score"] = float(max(row.get("weight_residual_qerr_gain", 0.0), 0.0))
        row["routed_residual_qerr_score"] = row["residual_qerr_score"] * row["route_weight"]
        row["routed_residual_qerr_gain_score"] = row["residual_qerr_gain_score"] * row["route_weight"]
        row["route_score"] = float(row["route_mass_fraction"])


def metric_to_score_key(metric):
    mapping = {
        "tail": "tail_score",
        "range": "range_score",
        "kurtosis": "kurtosis_score",
        "combo": "combo_score",
        "random": "random_score",
        "routed_tail": "routed_tail_score",
        "routed_range": "routed_range_score",
        "routed_kurtosis": "routed_kurtosis_score",
        "routed_input_kurtosis_log": "routed_input_kurtosis_log_score",
        "routed_down_kurtosis_log": "routed_down_kurtosis_log_score",
        "routed_weight_kurtosis_log": "routed_weight_kurtosis_log_score",
        "routed_kurtosis_equal_log": "routed_kurtosis_equal_log_score",
        "routed_input_kurtosis_raw": "routed_input_kurtosis_raw_score",
        "routed_down_kurtosis_raw": "routed_down_kurtosis_raw_score",
        "routed_weight_kurtosis_raw": "routed_weight_kurtosis_raw_score",
        "routed_kurtosis_equal_raw": "routed_kurtosis_equal_raw_score",
        "routed_combo": "routed_combo_score",
        "residual_qerr": "residual_qerr_score",
        "residual_qerr_gain": "residual_qerr_gain_score",
        "routed_residual_qerr": "routed_residual_qerr_score",
        "routed_residual_qerr_gain": "routed_residual_qerr_gain_score",
        "input": "input_score",
        "down": "down_score",
        "weight": "weight_score",
        "route": "route_score",
    }
    if metric not in mapping:
        raise ValueError(f"unknown metric {metric!r}; choices: {sorted(mapping)}")
    return mapping[metric]


def candidate_to_entry(candidate, module_name, layer_idx, expert_idx, proj_name, score, metric, route_mass_fraction):
    return {
        "module_name": module_name,
        "layer_index": int(layer_idx),
        "expert_index": int(expert_idx),
        "proj_name": proj_name,
        "quantization": quant_name(candidate),
        "abits": int(candidate.abits),
        "wbits": int(candidate.wbits),
        "act_group_size": candidate.act_group_size,
        "weight_group_size": candidate.weight_group_size,
        "sum_bits_cost": float(candidate.sum_bits_cost),
        "effective_avg_bits": float(candidate.effective_avg_bits),
        "cost_units": int(candidate.cost_units),
        "selection_metric": metric,
        "selection_score": float(score),
        "route_mass_fraction": float(route_mass_fraction),
    }


def choose_high_experts(rows, score_key, high_fraction, selection_scope):
    selected = set()
    groups = defaultdict(list)
    if selection_scope == "global":
        groups[0] = rows
    elif selection_scope == "per_layer":
        for row in rows:
            groups[int(row["layer_index"])].append(row)
    else:
        raise ValueError(f"unknown selection_scope: {selection_scope}")

    for _, group_rows in sorted(groups.items()):
        group_rows = sorted(group_rows, key=lambda row: float(row.get(score_key, 0.0)), reverse=True)
        n_high = int(round(len(group_rows) * float(high_fraction)))
        n_high = max(0, min(len(group_rows), n_high))
        for row in group_rows[:n_high]:
            selected.add((int(row["layer_index"]), int(row["expert_index"])))
    return selected


def parse_fraction_list(text):
    if text is None or str(text).strip() == "":
        return None
    values = [float(item.strip()) for item in str(text).split(",") if item.strip()]
    if not values:
        return None
    if any(value < 0 for value in values):
        raise ValueError(f"negative tier fraction in {text!r}")
    total = sum(values)
    if total <= 0:
        raise ValueError(f"tier fractions must sum to a positive value: {text!r}")
    return [value / total for value in values]


def tier_counts(n_items, fractions):
    raw = [float(frac) * int(n_items) for frac in fractions]
    counts = [int(math.floor(value)) for value in raw]
    while sum(counts) < n_items:
        order = sorted(range(len(raw)), key=lambda idx: (raw[idx] - counts[idx], idx), reverse=True)
        counts[order[0]] += 1
    while sum(counts) > n_items:
        order = sorted(range(len(raw)), key=lambda idx: (counts[idx] - raw[idx], counts[idx], idx), reverse=True)
        for idx in order:
            if counts[idx] > 0:
                counts[idx] -= 1
                break
    return counts


def projection_weight_components(row, proj_name):
    prefix = f"weight_{proj_name}_"
    return {
        "tail": float(row.get(prefix + "tail_p99_p50", 0.0)),
        "range": float(row.get(prefix + "range_max_p99", 0.0)),
        "kurtosis": float(row.get(prefix + "kurtosis_raw", 0.0)),
    }


def projection_score(
    row,
    metric,
    proj_name,
    score_transform="log",
    input_score_coef=1.0,
    down_score_coef=1.0,
    weight_score_coef=0.5,
):
    routed = False
    base_metric = metric
    if base_metric.startswith("routed_"):
        routed = True
        base_metric = base_metric[len("routed_"):]

    route_weight = float(row.get("route_weight", 1.0))
    if base_metric == "route":
        return float(row.get("route_score", row.get("route_mass_fraction", 0.0)))

    if proj_name == "down_proj":
        act_prefix = "down"
    else:
        act_prefix = "input"
    weight = projection_weight_components(row, proj_name)

    tail_score = (
        float(input_score_coef if act_prefix == "input" else down_score_coef)
        * score_component(row.get(f"{act_prefix}_tail", 0.0), score_transform)
        + float(weight_score_coef) * score_component(weight["tail"], score_transform)
    )
    range_score = (
        float(input_score_coef if act_prefix == "input" else down_score_coef)
        * score_component(row.get(f"{act_prefix}_range", 0.0), score_transform)
        + float(weight_score_coef) * score_component(weight["range"], score_transform)
    )
    kurtosis_score = (
        float(input_score_coef if act_prefix == "input" else down_score_coef)
        * score_component(row.get(f"{act_prefix}_kurtosis", 0.0), score_transform)
        + float(weight_score_coef) * score_component(weight["kurtosis"], score_transform)
    )

    if base_metric == "tail":
        score = tail_score
    elif base_metric == "range":
        score = range_score
    elif base_metric == "kurtosis":
        score = kurtosis_score
    elif base_metric == "combo":
        score = (tail_score + range_score + kurtosis_score) / 3.0
    elif base_metric == "input":
        score = float(row.get("input_score", 0.0))
    elif base_metric == "down":
        score = float(row.get("down_score", 0.0)) if proj_name == "down_proj" else 0.0
    elif base_metric == "weight":
        score = (
            score_component(weight["tail"], score_transform)
            + score_component(weight["range"], score_transform)
            + score_component(weight["kurtosis"], score_transform)
        )
    elif base_metric == "residual_qerr":
        score = float(row.get(f"weight_{proj_name}_residual_qerr", row.get("weight_residual_qerr", 0.0)))
    elif base_metric == "residual_qerr_gain":
        score = float(row.get(f"weight_{proj_name}_residual_qerr_gain", row.get("weight_residual_qerr_gain", 0.0)))
    else:
        score = float(row.get(metric_to_score_key(metric), 0.0))

    if routed:
        score *= route_weight
    return float(score)


def make_plan_records(rows, args, metric, score_key):
    projs = parse_str_list(args.plan_projs) or ["gate_proj", "up_proj", "down_proj"]
    records = []
    for row in sorted(rows, key=lambda item: (int(item["layer_index"]), int(item["expert_index"]))):
        layer_idx = int(row["layer_index"])
        expert_idx = int(row["expert_index"])
        if args.plan_granularity == "expert":
            records.append(
                {
                    "record_index": len(records),
                    "layer_index": layer_idx,
                    "expert_index": expert_idx,
                    "proj_name": None,
                    "proj_names": projs,
                    "row": row,
                    "score": float(row.get(score_key, 0.0)),
                }
            )
        elif args.plan_granularity == "projection":
            for proj_name in projs:
                records.append(
                    {
                        "record_index": len(records),
                        "layer_index": layer_idx,
                        "expert_index": expert_idx,
                        "proj_name": proj_name,
                        "proj_names": [proj_name],
                        "row": row,
                        "score": projection_score(
                            row,
                            metric,
                            proj_name,
                            score_transform=args.score_transform,
                            input_score_coef=args.input_score_coef,
                            down_score_coef=args.down_score_coef,
                            weight_score_coef=args.weight_score_coef,
                        ),
                    }
                )
        else:
            raise ValueError(f"unknown plan_granularity: {args.plan_granularity}")
    return records


def group_records(records, selection_scope):
    groups = defaultdict(list)
    if selection_scope == "global":
        groups[0] = records
    elif selection_scope == "per_layer":
        for record in records:
            groups[int(record["layer_index"])].append(record)
    else:
        raise ValueError(f"unknown selection_scope: {selection_scope}")
    return groups


def resolve_tier_candidates(args, low_candidate, high_candidate):
    if args.tier_quants:
        candidates = [parse_candidate_quant(item) for item in parse_str_list(args.tier_quants)]
        if len(candidates) < 2:
            raise ValueError("--tier_quants must contain at least two candidates")
    else:
        candidates = [low_candidate, high_candidate]
    costs = [float(candidate.sum_bits_cost) for candidate in candidates]
    if costs != sorted(costs):
        raise ValueError("tier candidates must be ordered from low cost to high cost")
    return candidates


def resolve_tier_fractions(args, candidates):
    fractions = parse_fraction_list(args.tier_fractions)
    if fractions is not None:
        if len(fractions) != len(candidates):
            raise ValueError("--tier_fractions length must match --tier_quants")
        return fractions

    if len(candidates) != 2:
        raise ValueError("--tier_fractions is required when using more than two candidates")

    low_cost = float(candidates[0].sum_bits_cost)
    high_cost = float(candidates[1].sum_bits_cost)
    if args.high_fraction is None:
        if abs(high_cost - low_cost) < 1e-12:
            high_fraction = 0.0
        else:
            high_fraction = (float(args.target_avg_sum_bits) - low_cost) / (high_cost - low_cost)
    else:
        high_fraction = float(args.high_fraction)
    high_fraction = max(0.0, min(1.0, high_fraction))
    return [1.0 - high_fraction, high_fraction]


def assign_tier_candidates(records, candidates, fractions, selection_scope):
    assignments = {}
    groups = group_records(records, selection_scope)
    for _, group in sorted(groups.items()):
        group = sorted(group, key=lambda record: float(record.get("score", 0.0)), reverse=True)
        counts = tier_counts(len(group), fractions)
        pos = 0
        for cand_idx in range(len(candidates) - 1, -1, -1):
            count = counts[cand_idx]
            for record in group[pos:pos + count]:
                assignments[int(record["record_index"])] = candidates[cand_idx]
            pos += count
    return assignments


def build_metric_plan(rows, args, metric, low_candidate, high_candidate):
    score_key = metric_to_score_key(metric)
    candidates = resolve_tier_candidates(args, low_candidate, high_candidate)
    fractions = resolve_tier_fractions(args, candidates)
    records = make_plan_records(rows, args, metric, score_key)
    assignments = assign_tier_candidates(records, candidates, fractions, args.selection_scope)
    low_name = candidates[0].name

    plan = []
    per_module = {}
    high_experts = set()
    expert_quant_counts = Counter()
    record_score_values = []
    selected_scores = []

    for record in sorted(records, key=lambda item: (int(item["layer_index"]), int(item["expert_index"]), str(item.get("proj_name") or ""))):
        layer_idx = int(record["layer_index"])
        expert_idx = int(record["expert_index"])
        candidate = assignments[int(record["record_index"])]
        row = record["row"]
        score = float(record.get("score", 0.0))
        record_score_values.append(score)
        if candidate.name != low_name:
            high_experts.add((layer_idx, expert_idx))
            selected_scores.append(score)
        expert_quant_counts[candidate.name] += 1
        for proj_name in record["proj_names"]:
            module_name = f"model.layers.{layer_idx}.mlp.experts.{expert_idx}.{proj_name}"
            entry = candidate_to_entry(
                candidate,
                module_name,
                layer_idx,
                expert_idx,
                proj_name,
                score,
                metric,
                row.get("route_mass_fraction", 0.0),
            )
            plan.append(entry)
            per_module[module_name] = entry

    quant_counts = Counter(item["quantization"] for item in plan)
    by_layer = defaultdict(Counter)
    by_proj = defaultdict(Counter)
    for item in plan:
        by_layer[int(item["layer_index"])][item["quantization"]] += 1
        by_proj[item["proj_name"]][item["quantization"]] += 1
    avg_sum_bits = sum(float(item["sum_bits_cost"]) for item in plan) / max(len(plan), 1)
    all_scores = record_score_values
    summary = {
        "format": f"duquant_{args.model_name}_expert_distribution_plan_v2",
        "metric": metric,
        "score_key": score_key,
        "selection_scope": args.selection_scope,
        "plan_granularity": args.plan_granularity,
        "target_avg_sum_bits": float(args.target_avg_sum_bits),
        "actual_avg_sum_bits": float(avg_sum_bits),
        "target_equivalent_avg_aw_bits": float(args.target_avg_sum_bits) / 2.0,
        "actual_equivalent_avg_aw_bits": float(avg_sum_bits) / 2.0,
        "weight_channel_group_size": args.weight_channel_group_size,
        "score_transform": args.score_transform,
        "input_score_coef": float(args.input_score_coef),
        "down_score_coef": float(args.down_score_coef),
        "weight_score_coef": float(args.weight_score_coef),
        "weight_score_projs": parse_str_list(args.weight_score_projs),
        "num_experts": len(rows),
        "num_records": len(records),
        "num_modules": len(plan),
        "tier_fractions": [float(value) for value in fractions],
        "tier_quants": [asdict(candidate) for candidate in candidates],
        "high_fraction_target": float(sum(fractions[1:])),
        "high_expert_count": len(high_experts),
        "high_expert_fraction": float(len(high_experts) / max(len(rows), 1)),
        "low_quant": asdict(candidates[0]),
        "high_quant": asdict(candidates[-1]),
        "expert_quantization_counts": dict(expert_quant_counts),
        "quantization_counts": dict(quant_counts),
        "by_layer": {str(k): dict(v) for k, v in sorted(by_layer.items())},
        "by_proj_name": {k: dict(v) for k, v in sorted(by_proj.items())},
        "score_min": float(min(all_scores)) if all_scores else 0.0,
        "score_max": float(max(all_scores)) if all_scores else 0.0,
        "score_mean": float(sum(all_scores) / max(len(all_scores), 1)),
        "selected_score_mean": float(sum(selected_scores) / max(len(selected_scores), 1)) if selected_scores else 0.0,
    }
    return {"plan": plan, "per_module": per_module, "summary": summary}, high_experts


def save_rows_csv(rows, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fieldnames = sorted({key for row in rows for key in row.keys()})
    preferred = [
        "layer_index",
        "expert_index",
        "route_count",
        "route_frequency",
        "route_mass",
        "route_mass_fraction",
        "route_weight",
        "tail_score",
        "range_score",
        "kurtosis_score",
        "combo_score",
        "routed_combo_score",
        "random_score",
        "input_tail",
        "input_range",
        "input_kurtosis",
        "down_tail",
        "down_range",
        "down_kurtosis",
        "weight_tail",
        "weight_range",
        "weight_kurtosis",
        "weight_residual_qerr",
        "weight_residual_qerr_gain",
        "residual_qerr_score",
        "routed_residual_qerr_score",
    ]
    ordered = [name for name in preferred if name in fieldnames] + [name for name in fieldnames if name not in preferred]
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=ordered)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def plot_heatmaps(rows, high_experts_by_metric, args, output_dir):
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"WARNING: matplotlib unavailable, skip plots: {exc}")
        return []

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    max_layer = max(int(row["layer_index"]) for row in rows)
    max_expert = max(int(row["expert_index"]) for row in rows)
    paths = []
    for metric, high_experts in high_experts_by_metric.items():
        score_key = metric_to_score_key(metric)
        score_mat = torch.full((max_layer + 1, max_expert + 1), float("nan"))
        assign_mat = torch.zeros((max_layer + 1, max_expert + 1))
        for row in rows:
            layer_idx = int(row["layer_index"])
            expert_idx = int(row["expert_index"])
            score_mat[layer_idx, expert_idx] = float(row.get(score_key, 0.0))
            assign_mat[layer_idx, expert_idx] = 1.0 if (layer_idx, expert_idx) in high_experts else 0.0

        plt.figure(figsize=(max(10, (max_expert + 1) * 0.22), max(5, (max_layer + 1) * 0.32)))
        plt.imshow(score_mat.numpy(), aspect="auto", interpolation="nearest")
        plt.colorbar(label=score_key)
        plt.xlabel("Expert")
        plt.ylabel("Layer")
        plt.title(f"Expert score heatmap: {metric}")
        plt.tight_layout()
        score_path = output_dir / f"expert_score_{metric}.png"
        plt.savefig(score_path, dpi=220)
        plt.close()
        paths.append(str(score_path))

        plt.figure(figsize=(max(10, (max_expert + 1) * 0.22), max(5, (max_layer + 1) * 0.32)))
        plt.imshow(assign_mat.numpy(), aspect="auto", interpolation="nearest", vmin=0, vmax=1)
        plt.colorbar(label="W8A8 expert")
        plt.xlabel("Expert")
        plt.ylabel("Layer")
        plt.title(f"W8A8 assignment: {metric}")
        plt.tight_layout()
        assign_path = output_dir / f"expert_assignment_{metric}.png"
        plt.savefig(assign_path, dpi=220)
        plt.close()
        paths.append(str(assign_path))
    return paths


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=str(ROOT_DIR / "local_models" / "olmoe_compat"))
    parser.add_argument("--model_name", default="olmoe", help="model structure name, e.g. olmoe or qwen2_moe")
    parser.add_argument("--cache_dir", default=str(ROOT_DIR / "cache"))
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--calib_dataset", default="wikitext2", choices=["wikitext2", "ptb", "c4", "mix", "pile"])
    parser.add_argument("--nsamples", type=int, default=32)
    parser.add_argument("--seq_length", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=2)
    parser.add_argument("--attn_implementation", default="eager", choices=["eager", "sdpa", "flash_attention_2"])
    parser.add_argument("--metrics", default="tail,range,kurtosis,routed_combo")
    parser.add_argument("--selection_scope", default="per_layer", choices=["per_layer", "global"])
    parser.add_argument("--plan_granularity", default="expert", choices=["expert", "projection"])
    parser.add_argument("--route_alpha", type=float, default=0.5)
    parser.add_argument("--score_transform", default="log", choices=["log", "raw"])
    parser.add_argument("--input_score_coef", type=float, default=1.0)
    parser.add_argument("--down_score_coef", type=float, default=1.0)
    parser.add_argument("--weight_score_coef", type=float, default=0.5)
    parser.add_argument("--weight_score_projs", default="gate_proj,up_proj,down_proj")
    parser.add_argument("--low_quant", default="w4g-1-a4g-1")
    parser.add_argument("--high_quant", default="w8g-1-a8g-1")
    parser.add_argument("--tier_quants", default=None, help="comma-separated candidates ordered from low cost to high cost")
    parser.add_argument("--tier_fractions", default=None, help="comma-separated fractions matching --tier_quants; low-to-high order")
    parser.add_argument("--target_avg_sum_bits", type=float, default=12.0)
    parser.add_argument("--high_fraction", type=float, default=None)
    parser.add_argument("--plan_projs", default="gate_proj,up_proj,down_proj")
    parser.add_argument(
        "--weight_channel_group_size",
        type=int,
        default=None,
        help="output-channel group size for residual qerr simulation; -1 means one tensor scale per matrix",
    )
    parser.add_argument("--moe_outlier_topk", type=int, default=0)
    parser.add_argument("--moe_outlier_score", default="weight_max", choices=["smooth_scale", "weight_max", "weight_error"])
    parser.add_argument("--smooth", action="store_true")
    parser.add_argument("--fc1_scale_merge", default="act_mean")
    parser.add_argument("--act_mean_beta", type=float, default=2.0)
    parser.add_argument("--alpha", type=float, default=0.6)
    parser.add_argument("--otsu_ratio", type=float, default=0.65)
    parser.add_argument("--otsu_smooth_rate", type=float, default=0.7)
    parser.add_argument("--activation_max_samples", type=int, default=16384)
    parser.add_argument("--activation_sample_per_call", type=int, default=2048)
    parser.add_argument("--weight_max_samples", type=int, default=16384)
    parser.add_argument("--weight_sample_per_call", type=int, default=16384)
    parser.add_argument("--reuse_dataloader_cache", action="store_true")
    parser.add_argument("--disable_act_stats_cache", action="store_true")
    parser.add_argument("--refresh_act_stats_cache", action="store_true")
    parser.add_argument("--verbose_smooth_log", action="store_true")
    args = parser.parse_args()

    if args.moe_outlier_score == "smooth_scale" and not args.smooth:
        parser.error("--moe_outlier_score smooth_scale requires --smooth")
    args.moe_outlier_scores = {}

    if "qwen" in args.model_name.lower():
        args.model_name = "qwen2_moe"
    elif "olmoe" in args.model_name.lower():
        args.model_name = "olmoe"
    else:
        args.model_name = args.model_name.split("-")[0]
    started = time.perf_counter()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    low_candidate = parse_candidate_quant(args.low_quant)
    high_candidate = parse_candidate_quant(args.high_quant)
    if high_candidate.sum_bits_cost < low_candidate.sum_bits_cost:
        raise ValueError("--high_quant must not be cheaper than --low_quant")
    qerr_tier_candidates = [parse_candidate_quant(item) for item in parse_str_list(args.tier_quants)] if args.tier_quants else [low_candidate, high_candidate]
    args._qerr_low_candidate = qerr_tier_candidates[0]
    args._qerr_high_candidate = qerr_tier_candidates[-1]

    lm = build_lm(args)
    lm.model.to(device)
    dataloader = load_calibration(args)
    apply_optional_smooth(lm, args, dataloader)
    if not args.smooth and args.moe_outlier_topk > 0 and args.moe_outlier_score in ("weight_max", "weight_error"):
        args.moe_outlier_scores = prepare_moe_outlier_scores(
            lm.model,
            score_method=args.moe_outlier_score,
            args=args,
            model_name=args.model_name,
            logger=PrintLogger(verbose=args.verbose_smooth_log),
        )

    weight_rows = collect_weight_rows(lm.model, args)
    activation_rows = collect_activation_rows(lm.model, dataloader, args, device)
    rows = []
    for key in sorted(weight_rows):
        row = dict(weight_rows[key])
        row.update(activation_rows.get(key, {}))
        rows.append(row)
    add_scores(
        rows,
        args.route_alpha,
        score_transform=args.score_transform,
        input_score_coef=args.input_score_coef,
        down_score_coef=args.down_score_coef,
        weight_score_coef=args.weight_score_coef,
    )

    scores_csv = output_dir / "expert_distribution_scores.csv"
    scores_json = output_dir / "expert_distribution_scores.json"
    save_rows_csv(rows, scores_csv)
    save_json(rows, scores_json)
    print(f"Saved expert distribution scores: {scores_csv}")

    metrics = parse_str_list(args.metrics)
    high_experts_by_metric = {}
    plan_paths = {}
    for metric in metrics:
        plan_obj, high_experts = build_metric_plan(rows, args, metric, low_candidate, high_candidate)
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
                "route_alpha": args.route_alpha,
                "score_transform": args.score_transform,
                "input_score_coef": float(args.input_score_coef),
                "down_score_coef": float(args.down_score_coef),
                "weight_score_coef": float(args.weight_score_coef),
                "weight_score_projs": parse_str_list(args.weight_score_projs),
                "moe_outlier_topk": args.moe_outlier_topk,
                "moe_outlier_score": args.moe_outlier_score,
                "weight_stats_mode": "post_outlier_residual" if args.moe_outlier_topk > 0 else "full_weight",
                "elapsed_seconds": time.perf_counter() - started,
            }
        )
        plan_path = output_dir / f"moe_quant_plan_{metric}.json"
        summary_path = output_dir / f"moe_quant_plan_{metric}_summary.json"
        save_json(plan_obj, plan_path)
        save_json(plan_obj["summary"], summary_path)
        high_experts_by_metric[metric] = high_experts
        plan_paths[metric] = str(plan_path)
        print(
            f"Saved {metric} plan: {plan_path} "
            f"(actual_avg_sum_bits={plan_obj['summary']['actual_avg_sum_bits']:.6f})"
        )

    plot_paths = plot_heatmaps(rows, high_experts_by_metric, args, output_dir)
    metadata = {
        "format": f"duquant_{args.model_name}_expert_distribution_planner_v1",
        "model": args.model,
        "output_dir": str(output_dir),
        "metrics": metrics,
        "plan_paths": plan_paths,
        "scores_csv": str(scores_csv),
        "scores_json": str(scores_json),
        "plot_paths": plot_paths,
        "low_quant": asdict(low_candidate),
        "high_quant": asdict(high_candidate),
        "target_avg_sum_bits": args.target_avg_sum_bits,
        "selection_scope": args.selection_scope,
        "moe_outlier_topk": args.moe_outlier_topk,
        "moe_outlier_score": args.moe_outlier_score,
        "weight_stats_mode": "post_outlier_residual" if args.moe_outlier_topk > 0 else "full_weight",
        "elapsed_seconds": time.perf_counter() - started,
    }
    save_json(metadata, output_dir / "planner_metadata.json")
    print(f"Saved planner metadata: {output_dir / 'planner_metadata.json'}")


if __name__ == "__main__":
    main()
