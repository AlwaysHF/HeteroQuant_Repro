import os
import sys
import random
import hashlib
import json
import re
import gc
import numpy as np
import torch


def _configure_torch_threads_from_env():
    num_threads = os.environ.get("DUQUANT_TORCH_NUM_THREADS")
    interop_threads = os.environ.get("DUQUANT_TORCH_NUM_INTEROP_THREADS")
    if num_threads:
        torch.set_num_threads(int(num_threads))
    if interop_threads:
        torch.set_num_interop_threads(int(interop_threads))


_configure_torch_threads_from_env()

from models.LMClass import LMClass
import time
from datautils import get_loaders
from pprint import pprint
import torch.nn as nn
from quantize.duquant import duquant
from tqdm import tqdm
import utils
from pathlib import Path
from categories import subcategories, categories
from utils import *

from quantize.smooth import smooth_lm
import logging
torch.backends.cudnn.benchmark = True


def _safe_cache_tag(value):
    return str(value).replace("/", "_").replace("\\", "_").replace(":", "_").replace(" ", "_")


def _smooth_stats_cache_path(args, stat_name):
    key = "|".join(
        [
            "smooth_stats_v2",
            str(args.model),
            str(args.net),
            str(args.model_family),
            str(args.calib_dataset),
            str(args.nsamples),
            str(args.seq_length),
            str(args.seed),
            str(args.attn_implementation),
        ]
    )
    digest = hashlib.md5(key.encode("utf-8")).hexdigest()[:16]
    model_tag = _safe_cache_tag(Path(args.model).name)
    filename = (
        f"smooth_stats_v2_{stat_name}_{model_tag}_{args.model_family}_"
        f"{args.calib_dataset}_{args.nsamples}_{args.seq_length}_{args.seed}_{digest}.pt"
    )
    return os.path.join(args.cache_dir, filename)


def _sync_cuda_for_timing():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _load_smooth_stat_cache(args, logger, stat_name):
    cache_path = _smooth_stats_cache_path(args, stat_name)
    if args.disable_act_stats_cache or args.refresh_act_stats_cache or not os.path.exists(cache_path):
        return False, None
    try:
        logger.info(f"load {stat_name} from {cache_path}")
        return True, torch.load(cache_path, map_location="cpu")
    except Exception as exc:
        logger.warning(f"failed to load {stat_name} cache {cache_path}: {exc}; recomputing")
        return False, None


def _save_smooth_stat_cache(args, logger, stat_name, value):
    if args.disable_act_stats_cache:
        return
    cache_path = _smooth_stats_cache_path(args, stat_name)
    Path(args.cache_dir).mkdir(parents=True, exist_ok=True)
    tmp_path = f"{cache_path}.tmp.{os.getpid()}"
    torch.save(value, tmp_path)
    os.replace(tmp_path, cache_path)
    logger.info(f"save {stat_name} to {cache_path}")


def _load_or_compute_smooth_stat(args, logger, stat_name, compute_fn):
    found, value = _load_smooth_stat_cache(args, logger, stat_name)
    if found:
        return value
    value = compute_fn()
    _save_smooth_stat_cache(args, logger, stat_name, value)
    return value


def _needs_moe_outlier_act_energy(args):
    return (
        int(getattr(args, "moe_outlier_topk", 0)) > 0
        and getattr(args, "moe_outlier_score", "weight_max") == "act_weight_error"
    )


def _layer_gate_name_from_moe_key(key):
    if ".mlp.experts." in key:
        return key.split(".mlp.experts.", 1)[0] + ".mlp.gate"
    return key if key.endswith(".mlp.gate") else None


def _adjust_moe_outlier_act_energy_for_smooth(args, act_energy, moe_act_means):
    if not act_energy:
        return {}
    if getattr(args, "fc1_scale_merge", None) != "act_mean" or moe_act_means is None:
        return act_energy

    adjusted = {}
    beta = float(getattr(args, "act_mean_beta", 2.0))
    for key, energy in act_energy.items():
        gate_name = _layer_gate_name_from_moe_key(key)
        if gate_name not in moe_act_means:
            adjusted[key] = energy
            continue
        act_mean = moe_act_means[gate_name].float().clamp(min=1e-5)
        scale = act_mean / act_mean.mean().clamp(min=1e-5) * beta
        scale = scale.float().cpu().clamp(min=1e-5)
        adjusted[key] = energy.float().cpu() / scale.pow(2)
    return adjusted


def _normalize_group_size(value):
    if value is None:
        return None
    value = int(value)
    if value <= 0:
        return None
    return value


def _parse_quantization_name(quant_name):
    lowered = str(quant_name).lower().replace("_", "")
    legacy = re.fullmatch(r"a(\d+)w(\d+)", lowered)
    if legacy is not None:
        return {
            "quantization": lowered,
            "abits": int(legacy.group(1)),
            "wbits": int(legacy.group(2)),
            "act_group_size": None,
            "weight_group_size": None,
        }
    w_first = re.fullmatch(r"w(\d+)g(-?\d+)-a(\d+)g(-?\d+)", lowered)
    a_first = re.fullmatch(r"a(\d+)g(-?\d+)-w(\d+)g(-?\d+)", lowered)
    if w_first is not None:
        wbits, wg, abits, ag = int(w_first.group(1)), int(w_first.group(2)), int(w_first.group(3)), int(w_first.group(4))
    elif a_first is not None:
        abits, ag, wbits, wg = int(a_first.group(1)), int(a_first.group(2)), int(a_first.group(3)), int(a_first.group(4))
    else:
        return None
    wg = _normalize_group_size(wg)
    ag = _normalize_group_size(ag)
    normalized = f"w{wbits}g{wg if wg is not None else -1}-a{abits}g{ag if ag is not None else -1}"
    return {
        "quantization": normalized,
        "abits": abits,
        "wbits": wbits,
        "act_group_size": ag,
        "weight_group_size": wg,
    }


def _load_moe_quant_plan(plan_path):
    with open(plan_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, dict) and "per_module" in data:
        items = data["per_module"].items()
    elif isinstance(data, dict) and "plan" in data and isinstance(data["plan"], list):
        items = ((item["module_name"], item) for item in data["plan"])
    elif isinstance(data, list):
        items = ((item["module_name"], item) for item in data)
    elif isinstance(data, dict):
        items = data.items()
    else:
        raise ValueError(f"unsupported moe quant plan format: {plan_path}")

    plan = {}
    for module_name, item in items:
        if isinstance(item, str):
            parsed = _parse_quantization_name(item)
            if parsed is None:
                raise ValueError(f"cannot parse quantization name {item!r} for {module_name}")
            entry = parsed
        else:
            entry = dict(item)
            quant_name = entry.get("quantization") or entry.get("quant")
            parsed = _parse_quantization_name(quant_name) if quant_name else None
            if parsed is not None:
                entry.setdefault("quantization", parsed["quantization"])
                entry.setdefault("abits", parsed["abits"])
                entry.setdefault("wbits", parsed["wbits"])
                entry.setdefault("act_group_size", parsed["act_group_size"])
                entry.setdefault("weight_group_size", parsed["weight_group_size"])
            if "abits" not in entry and "a_bits" in entry:
                entry["abits"] = entry["a_bits"]
            if "wbits" not in entry and "w_bits" in entry:
                entry["wbits"] = entry["w_bits"]
            if "act_group_size" not in entry and "a_group_size" in entry:
                entry["act_group_size"] = entry["a_group_size"]
            if "act_group_size" not in entry and "activation_group_size" in entry:
                entry["act_group_size"] = entry["activation_group_size"]
            if "weight_group_size" not in entry and "w_group_size" in entry:
                entry["weight_group_size"] = entry["w_group_size"]

        if "abits" not in entry or "wbits" not in entry:
            raise ValueError(f"missing abits/wbits in moe quant plan entry for {module_name}")
        entry["abits"] = int(entry["abits"])
        entry["wbits"] = int(entry["wbits"])
        entry["act_group_size"] = _normalize_group_size(entry.get("act_group_size"))
        entry["weight_group_size"] = _normalize_group_size(entry.get("weight_group_size"))
        if entry["abits"] not in (4, 8, 16):
            raise ValueError(f"unsupported activation bits {entry['abits']} for {module_name}")
        if entry["wbits"] > entry["abits"]:
            raise ValueError(f"invalid plan entry {module_name}: abits must be >= wbits")
        plan[module_name] = entry
    return plan


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, help="model name of model path")
    parser.add_argument("--cache_dir", default="./cache", type=str, help="cache dir of dataset, leading to faster debug")
    parser.add_argument("--output_dir", default="./log/", type=str, help="direction of logging file")
    parser.add_argument("--calib_dataset",type=str,default="wikitext2",
        choices=["wikitext2", "ptb", "c4", "mix","pile"],
        help="Where to extract calibration data from.",
    )
    parser.add_argument('--test_dataset', type=str, default='wikitext2', help='dataset for testing')
    parser.add_argument("--nsamples", type=int, default=128, help="Number of calibration data samples.")
    parser.add_argument("--batch_size", type=int, default=1, help="batch size.")
    parser.add_argument("--seed", type=int, default=2, help="Seed for sampling the calibration data.")
    parser.add_argument("--tasks", default="")
    parser.add_argument("--eval_ppl", action="store_true")
    parser.add_argument("--num_fewshot", type=int, default=0)
    parser.add_argument("--wbits", type=int, default=4)
    parser.add_argument("--abits", type=int, default=4)
    parser.add_argument("--router_wbits", type=int, default=8)
    parser.add_argument("--router_abits", type=int, default=8)
    parser.add_argument("--group_size", type=int, default=None)
    parser.add_argument("--alpha", type=float, default=0.6)
    parser.add_argument("--otsu_ratio", type=float, default=0.65)
    parser.add_argument("--otsu_smooth_rate", type=float, default=0.7)
    parser.add_argument("--act_group_size", type=int, default=None)
    parser.add_argument("--smooth",default=False, action="store_true")
    parser.add_argument("--disable_act_stats_cache", default=False, action="store_true", help="disable smooth activation/statistics cache")
    parser.add_argument("--refresh_act_stats_cache", default=False, action="store_true", help="recompute and overwrite smooth activation/statistics cache")
    parser.add_argument("--symmetric",default=False, action="store_true", help="symmetric quantization")
    parser.add_argument("--a_dynamic_method", type=str, default="per_token")
    parser.add_argument("--w_dynamic_method", type=str, default="per_channel")
    parser.add_argument("--net", type=str, default=None)

    parser.add_argument("--seq_length", type=int, default=4096)
    parser.add_argument("--fc1_scale_merge", type=str, default="max", help="expert-aware smoothing aggregation strategy")
    parser.add_argument("--moe_down_smooth_mode", type=str, default="duquant", choices=["duquant", "otsu"], help="MoE down_proj smoothing mode; duquant matches the original DuQuant fc-fc scale")
    parser.add_argument("--act_mean_beta", type=float, default=2.0, help="beta for act_mean MoE fc1 smoothing scale")
    parser.add_argument("--router_w_dynamic_method", type=str, default="per_channel", help="the dynamic_method of the gate layer's weight")
    parser.add_argument("--expert_token_num_ratio", type=float, default=2.0, help="the ratio of expect expert_token_num over average expert_token_num")
    parser.add_argument("--fast_moe_down_calibration", default=False, action="store_true", help="skip routed expert-token calibration and calibrate MoE experts with unrouted mlp.gate tokens")
    parser.add_argument("--fast_moe_calib_tokens", type=int, default=512, help="number of unrouted mlp.gate tokens used by --fast_moe_down_calibration")

    # DuQuant
    parser.add_argument("--max_rotation_step", type=int, default=256, help="max steps for rotation transformation")
    parser.add_argument("--permutation_times", type=int, default=1, help="times of permutation transformation")
    parser.add_argument("--lac", type=float, default=None, help="activation clipping ratio")
    parser.add_argument("--swc", type=float, default=None, help="static weight clipping ratio")
    parser.add_argument("--block_size", type=int, default=128, help="block size for rotation matrices")
    parser.add_argument(
        "--disable_moe_gate_up_duquant_rotation",
        default=False,
        action="store_true",
        help="disable DuQuant rotation for MoE expert gate_proj and up_proj layers",
    )
    parser.add_argument(
        "--moe_outlier_topk",
        type=int,
        default=0,
        help="split top-k input channels from each MoE expert gate_proj and up_proj into separate column outlier branches",
    )
    parser.add_argument(
        "--moe_outlier_quant",
        type=str,
        default="same",
        choices=["same", "fp16", "w16a16", "w8a8", "w4a4"],
        help="quantization mode for the layer-level MoE gate/up outlier expert bank",
    )
    parser.add_argument(
        "--moe_outlier_score",
        type=str,
        default="weight_max",
        choices=["weight_max", "weight_error", "act_weight_error"],
        help="score used to select MoE gate/up outlier input columns",
    )
    parser.add_argument(
        "--moe_outlier_shared_layer_score",
        type=str,
        default="none",
        choices=["none", "smooth_scale"],
        help="force MoE gate/up outlier columns in the same layer to use one shared top-k score",
    )
    parser.add_argument(
        "--moe_quant_plan",
        type=str,
        default=None,
        help="JSON plan assigning per-module MoE expert gate/up/down abits and wbits",
    )

    args = parser.parse_args()
    if args.moe_outlier_topk < 0:
        raise ValueError("--moe_outlier_topk must be non-negative")
    if args.moe_outlier_shared_layer_score == "smooth_scale" and not args.smooth:
        raise ValueError("--moe_outlier_shared_layer_score smooth_scale requires --smooth")
    args.moe_outlier_act_energy = {}
    args.moe_outlier_shared_layer_scores = {}
    if args.fast_moe_calib_tokens <= 0:
        raise ValueError("--fast_moe_calib_tokens must be positive")
    args.moe_quant_plan_dict = {}
    if args.moe_quant_plan:
        args.moe_quant_plan_dict = _load_moe_quant_plan(args.moe_quant_plan)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    args.attn_implementation = "eager"
    args.quant_method = "duquant"
    args.expert_ratio = 8/64.0   # default for olmoe; updated after model load for other MoE models
    args.expert_token_num = int(args.expert_ratio * args.seq_length * args.expert_token_num_ratio)

    # init logger
    args.output_dir = os.path.join(args.output_dir, f"{args.model.split('/')[-1]}_w{args.wbits}a{args.abits}_Rw{args.router_wbits}a{args.router_abits}")   
    print(f"args.output_dir: {args.output_dir}")
    
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    if args.cache_dir:
        Path(args.cache_dir).mkdir(parents=True, exist_ok=True)
    output_dir = Path(args.output_dir)
    logger = utils.create_logger(output_dir)
    logger.info(f"args.output_dir: {args.output_dir}")
    if args.moe_quant_plan_dict:
        logger.info(
            f"loaded moe quant plan from {args.moe_quant_plan}: "
            f"{len(args.moe_quant_plan_dict)} expert projection entries"
        )
    
    # load model
    if args.net is None:
        args.net = args.model.split('/')[-1]
    lm = LMClass(args)
    if "qwen" in args.net.lower():
        args.model_family = "qwen2_moe"
    elif "olmoe" in args.net.lower():
        args.model_family = "olmoe"
    elif "pangumoe" in args.net.lower():
        args.model_family = "pangumoe"
    else:
        args.model_family = args.net.split('-')[0]
    if hasattr(lm.model.config, "num_experts") and hasattr(lm.model.config, "num_experts_per_tok"):
        args.expert_ratio = lm.model.config.num_experts_per_tok / lm.model.config.num_experts
        args.expert_token_num = int(args.expert_ratio * args.seq_length * args.expert_token_num_ratio)
    lm.seqlen = args.seq_length
    print(f"lm.device: {args.output_dir}")
    lm.model.eval()
    for param in lm.model.parameters():
        param.requires_grad = False
    args_for_log = argparse.Namespace(**vars(args))
    if args.moe_quant_plan_dict:
        args_for_log.moe_quant_plan_dict = f"<{len(args.moe_quant_plan_dict)} entries>"
    logger.info(args_for_log)

    args.weight_quant_params = {
        "n_bits": args.wbits,
        "per_channel_axes": [0],
        "symmetric": args.symmetric,
        "dynamic_method": args.w_dynamic_method,
        "group_size": args.group_size,
        "swc":args.swc,
        "quant_method": args.quant_method,
        "block_size": args.block_size,
        "max_rotation_step": args.max_rotation_step,
        "permutation_times": args.permutation_times,
    }
    args.act_quant_params = {
        "n_bits":  args.abits,
        "per_channel_axes": [],
        "symmetric": False,
        "lac":args.lac,
        "act_group_size": args.act_group_size,
        "dynamic_method": args.a_dynamic_method,
        "quant_method": args.quant_method,
        "block_size": args.block_size,
        "max_rotation_step": args.max_rotation_step,
        "permutation_times": args.permutation_times,
    }
    
    args.router_weight_quant_params = {
        "n_bits": args.router_wbits,
        "per_channel_axes": [0],
        "symmetric": args.symmetric,
        "dynamic_method": args.router_w_dynamic_method,
        "router_top_k": getattr(lm.model.config, "num_experts_per_tok", None),
        "group_size": args.group_size,
        "swc":args.swc,
        "quant_method": args.quant_method,
        "block_size": args.block_size,
        "max_rotation_step": args.max_rotation_step,
        "permutation_times": args.permutation_times,
    }
    args.router_act_quant_params = {
        "n_bits":  args.router_abits,
        "per_channel_axes": [],
        "symmetric": False,
        "lac":args.lac,
        "act_group_size": args.act_group_size,
        "dynamic_method": args.a_dynamic_method,
        "quant_method": args.quant_method,
        "block_size": args.block_size,
        "max_rotation_step": args.max_rotation_step,
        "permutation_times": args.permutation_times,
    }
    args.q_quant_params = {
        "n_bits": args.abits,
        "per_channel_axes": [],
        "symmetric": False,
        "dynamic_method": args.a_dynamic_method,
        "quant_method": args.quant_method,
        "block_size": args.block_size,
        "max_rotation_step": args.max_rotation_step,
    }
    args.k_quant_params = {
        "n_bits": args.abits,
        "per_channel_axes": [],
        "symmetric": False,
        "dynamic_method": args.a_dynamic_method,
        "quant_method": args.quant_method,
        "block_size": args.block_size,
    }
    args.v_quant_params = {
        "n_bits": args.abits,
        "per_channel_axes": [],
        "symmetric": False,
        "dynamic_method": args.a_dynamic_method,
    }
    args.p_quant_params = {
        "n_bits": 16,
        "metric": "fix0to1",
    }
    dataloader = None
    moe_act_means = None
    act_samples = None
    weight_scores = None
    router_logits = None
    moe_outlier_act_energy = None
    act_scales = None
    act_per_channel_scales = None

    # quantization
    if args.wbits < 16 or args.abits <16:
        logger.info("=== start quantization ===")
        dataloader_start = time.perf_counter()
        # load calibration dataset
        cache_dataloader = f'{args.cache_dir}/dataloader_{args.model_family}_{args.calib_dataset}_{args.nsamples}_{args.seq_length}.cache'
        if os.path.exists(cache_dataloader):
            dataloader = torch.load(cache_dataloader)
            logger.info(f"load calibration from {cache_dataloader}")
        else:
            dataloader, _ = get_loaders(
                args.calib_dataset,
                nsamples=args.nsamples,
                seed=args.seed,
                model=args.model,
                seqlen=lm.seqlen,
            )
            torch.save(dataloader, cache_dataloader)
        dataloader_seconds = time.perf_counter() - dataloader_start
        logger.info(f"[timing] calibration_dataloader_seconds: {dataloader_seconds:.6f}")

        _sync_cuda_for_timing()
        calibration_start = time.perf_counter()
        smooth_seconds = 0.0
        if args.smooth:
            _sync_cuda_for_timing()
            smooth_start = time.perf_counter()
            convert_device(lm, args)
            # selected_experts = get_router_selected_experts(lm.model, dataloader, lm.model.config.num_experts_per_tok, args.nsamples, args.net)
            if args.fc1_scale_merge in ("act_mean", "act_p99"):
                moe_stat_name = "moe_act_means" if args.fc1_scale_merge == "act_mean" else "moe_act_p99s"
                act_samples = {}
                weight_scores = {}
                router_logits = {}

                smooth_stat_values = {}
                missing_smooth_stats = []
                for stat_name in (moe_stat_name, "act_scales", "act_per_channel_scales"):
                    found, value = _load_smooth_stat_cache(args, logger, stat_name)
                    if found:
                        smooth_stat_values[stat_name] = value
                    else:
                        missing_smooth_stats.append(stat_name)

                if missing_smooth_stats:
                    logger.info(
                        "compute smooth activation stats in one forward for missing caches: "
                        + ",".join(missing_smooth_stats)
                    )
                    combined_stats = get_smooth_activation_stats(
                        lm.model,
                        dataloader,
                        args.nsamples,
                        collect_moe_act_mean="moe_act_means" in missing_smooth_stats,
                        collect_moe_act_p99="moe_act_p99s" in missing_smooth_stats,
                        collect_act_scales="act_scales" in missing_smooth_stats,
                        collect_act_per_channel_scales="act_per_channel_scales" in missing_smooth_stats,
                    )
                    for stat_name in missing_smooth_stats:
                        smooth_stat_values[stat_name] = combined_stats[stat_name]
                        _save_smooth_stat_cache(args, logger, stat_name, combined_stats[stat_name])

                moe_act_means = smooth_stat_values[moe_stat_name]
                act_scales = smooth_stat_values["act_scales"]
                act_per_channel_scales = smooth_stat_values["act_per_channel_scales"]
                if args.moe_outlier_shared_layer_score == "smooth_scale":
                    args.moe_outlier_shared_layer_scores = {}
                    for stat_key, stat_value in moe_act_means.items():
                        if ".mlp.gate" not in stat_key:
                            continue
                        try:
                            layer_idx = int(stat_key.split(".layers.")[1].split(".")[0])
                        except (IndexError, ValueError):
                            continue
                        args.moe_outlier_shared_layer_scores[layer_idx] = stat_value.detach().float().cpu().flatten()
                    logger.info(
                        "prepared smooth_scale shared layer outlier scores for "
                        f"{len(args.moe_outlier_shared_layer_scores)} MoE layers"
                    )
            else:
                if args.moe_outlier_shared_layer_score == "smooth_scale":
                    raise ValueError("--moe_outlier_shared_layer_score smooth_scale requires --fc1_scale_merge act_mean or act_p99")
                moe_act_means = None
                act_samples = _load_or_compute_smooth_stat(
                    args,
                    logger,
                    "act_samples",
                    lambda: get_act_samples(lm.model, dataloader, args.nsamples),
                )
                weight_scores = _load_or_compute_smooth_stat(
                    args,
                    logger,
                    "weight_scores",
                    lambda: get_weight_scores(lm.model),
                )
                router_logits = _load_or_compute_smooth_stat(
                    args,
                    logger,
                    "router_logits",
                    lambda: get_router_logits(lm.model, dataloader, args.nsamples),
                )
                act_scales = _load_or_compute_smooth_stat(
                    args,
                    logger,
                    "act_scales",
                    lambda: get_act_scales(lm.model, dataloader, args.nsamples),
                )
                act_per_channel_scales = _load_or_compute_smooth_stat(
                    args,
                    logger,
                    "act_per_channel_scales",
                    lambda: get_act_per_channel_scales(lm.model, dataloader, args.nsamples),
                )
            moe_outlier_act_energy = None
            if _needs_moe_outlier_act_energy(args):
                moe_outlier_act_energy = _load_or_compute_smooth_stat(
                    args,
                    logger,
                    "moe_expert_act_square_means",
                    lambda: get_moe_expert_act_square_means(lm.model, dataloader, args.nsamples),
                )
            smooth_lm(lm.model, act_scales, act_per_channel_scales, act_samples, weight_scores, router_logits, 
                    fc1_scale_merge=args.fc1_scale_merge, alpha=args.alpha, otsu_ratio=args.otsu_ratio, otsu_smooth_rate=args.otsu_smooth_rate,
                    moe_down_smooth_mode=args.moe_down_smooth_mode,
                    moe_act_means=moe_act_means, act_mean_beta=args.act_mean_beta, logger=logger)
            if moe_outlier_act_energy is not None:
                args.moe_outlier_act_energy = _adjust_moe_outlier_act_energy_for_smooth(
                    args, moe_outlier_act_energy, moe_act_means
                )
                logger.info(
                    f"prepared moe_outlier_act_energy entries: {len(args.moe_outlier_act_energy)} "
                    f"for moe_outlier_score={args.moe_outlier_score}"
                )
            lm.model.cpu()

            _sync_cuda_for_timing()
            smooth_seconds = time.perf_counter() - smooth_start
            logger.info(f"[timing] smooth_seconds: {smooth_seconds:.6f}")

        if not args.smooth and _needs_moe_outlier_act_energy(args):
            convert_device(lm, args)
            args.moe_outlier_act_energy = _load_or_compute_smooth_stat(
                args,
                logger,
                "moe_expert_act_square_means",
                lambda: get_moe_expert_act_square_means(lm.model, dataloader, args.nsamples),
            )
            logger.info(
                f"prepared moe_outlier_act_energy entries: {len(args.moe_outlier_act_energy)} "
                f"for moe_outlier_score={args.moe_outlier_score}"
            )
            lm.model.cpu()

        _sync_cuda_for_timing()
        duquant_start = time.perf_counter()
        duquant(
            lm,
            args,
            dataloader,
            logger=logger,
        )
        _sync_cuda_for_timing()
        duquant_seconds = time.perf_counter() - duquant_start
        calibration_seconds = time.perf_counter() - calibration_start
        logger.info(f"[timing] duquant_seconds: {duquant_seconds:.6f}")
        logger.info(f"[timing] calibration_seconds: {calibration_seconds:.6f}")

        # Keep evaluation memory stable for large MoE models: smooth/stat caches
        # are only needed before duquant builds the final quantized model.
        dataloader = None
        moe_act_means = None
        act_samples = None
        weight_scores = None
        router_logits = None
        moe_outlier_act_energy = None
        act_scales = None
        act_per_channel_scales = None
        gc.collect()
        torch.cuda.empty_cache()
    logger.info(f"args.output_dir: {args.output_dir}")
    logger.info(f"args.tasks: {args.tasks}")
    evaluate(lm, args,logger)


if __name__ == "__main__":
    print(sys.argv)
    main()
