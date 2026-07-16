# =============================================================================
# Runtime setup and imports
# =============================================================================

import os
import sys
import random
import hashlib
import json
import re
import gc
import numpy as np
import torch


# 功能: 从环境变量读取并设置 PyTorch CPU 线程数。
# 输入: 无显式参数；读取 DUQUANT_TORCH_NUM_THREADS 和 DUQUANT_TORCH_NUM_INTEROP_THREADS。
# 输出: 无返回值；副作用是修改 torch 的线程配置。
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

from quantize.smooth import build_moe_fc1_smooth_scales, smooth_lm
from quantize.moe_outlier_score import prepare_moe_outlier_scores
import logging
torch.backends.cudnn.benchmark = True


# =============================================================================
# Smooth-stat cache helpers
# =============================================================================


# 功能: 将模型名/路径片段转换成适合放进缓存文件名的安全字符串。
# 输入: value，任意可转字符串的模型名、路径或标识。
# 输出: str，替换了路径分隔符、冒号和空格后的文件名片段。
def _safe_cache_tag(value):
    return str(value).replace("/", "_").replace("\\", "_").replace(":", "_").replace(" ", "_")


# 功能: 根据模型、数据集和统计量名称生成唯一的 Smooth 统计缓存路径。
# 输入: cache_dir 和 model/model_name/数据集/样本数/序列长度/seed/attention/stat_name 等缓存 key 字段。
# 输出: str，指向某个 Smooth 统计量 .pt 缓存文件的完整路径。
def _smooth_stats_cache_path(
    *,
    cache_dir,
    model,
    model_name,
    calib_dataset,
    nsamples,
    seq_length,
    seed,
    attn_implementation,
    stat_name,
):
    key = "|".join(
        [
            "smooth_stats_v2",
            str(model),
            str(model_name),
            str(calib_dataset),
            str(nsamples),
            str(seq_length),
            str(seed),
            str(attn_implementation),
        ]
    )
    digest = hashlib.md5(key.encode("utf-8")).hexdigest()[:16]
    model_tag = _safe_cache_tag(Path(model).name)
    filename = (
        f"smooth_stats_v2_{stat_name}_{model_tag}_{model_name}_"
        f"{calib_dataset}_{nsamples}_{seq_length}_{seed}_{digest}.pt"
    )
    return os.path.join(cache_dir, filename)


# 功能: 在计时前后同步 CUDA，避免异步执行导致耗时统计不准。
# 输入: 无显式参数；内部检查 torch.cuda.is_available()。
# 输出: 无返回值；有 CUDA 时阻塞等待当前 CUDA 任务完成。
def _sync_cuda_for_timing():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


# 功能: 尝试读取 Smooth 统计缓存，处理禁用/刷新/缺失/损坏 cache 的情况。
# 输入: 缓存 key 字段、disable_act_stats_cache、refresh_act_stats_cache、logger、stat_name。
# 输出: (found, value)；found 为 bool，value 为缓存对象或 None。
def _load_smooth_stat_cache(
    *,
    cache_dir,
    model,
    model_name,
    calib_dataset,
    nsamples,
    seq_length,
    seed,
    attn_implementation,
    disable_act_stats_cache,
    refresh_act_stats_cache,
    logger,
    stat_name,
):
    cache_path = _smooth_stats_cache_path(
        cache_dir=cache_dir,
        model=model,
        model_name=model_name,
        calib_dataset=calib_dataset,
        nsamples=nsamples,
        seq_length=seq_length,
        seed=seed,
        attn_implementation=attn_implementation,
        stat_name=stat_name,
    )
    if disable_act_stats_cache or refresh_act_stats_cache or not os.path.exists(cache_path):
        return False, None
    try:
        logger.info(f"load {stat_name} from {cache_path}")
        return True, torch.load(cache_path, map_location="cpu")
    except Exception as exc:
        logger.warning(f"failed to load {stat_name} cache {cache_path}: {exc}; recomputing")
        return False, None


# 功能: 将 Smooth 统计量安全写入缓存文件，先写临时文件再原子替换。
# 输入: 缓存 key 字段、disable_act_stats_cache、logger、stat_name、value。
# 输出: 无返回值；副作用是按需创建目录并保存 .pt cache。
def _save_smooth_stat_cache(
    *,
    cache_dir,
    model,
    model_name,
    calib_dataset,
    nsamples,
    seq_length,
    seed,
    attn_implementation,
    disable_act_stats_cache,
    logger,
    stat_name,
    value,
):
    if disable_act_stats_cache:
        return
    cache_path = _smooth_stats_cache_path(
        cache_dir=cache_dir,
        model=model,
        model_name=model_name,
        calib_dataset=calib_dataset,
        nsamples=nsamples,
        seq_length=seq_length,
        seed=seed,
        attn_implementation=attn_implementation,
        stat_name=stat_name,
    )
    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    tmp_path = f"{cache_path}.tmp.{os.getpid()}"
    torch.save(value, tmp_path)
    os.replace(tmp_path, cache_path)
    logger.info(f"save {stat_name} to {cache_path}")



# =============================================================================
# MoE quantization-plan parser
# =============================================================================


# 功能: 统一规范 group size，非正数按不分组处理。
# 输入: value，可能是 None、字符串或数字形式的 group size。
# 输出: int 或 None；正数转 int，None/非正数返回 None。
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


# 功能: 解析量化配置字符串，提取 abits/wbits 和可选 group size。
# 输入: quant_name，如 a4w4、w4g128-a8g128、a8g128-w4g128。
# 输出: dict 或 None；成功时返回标准化量化配置，失败时返回 None。
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


# 功能: 读取并校验 MoE per-module 异构量化计划 JSON。
# 输入: plan_path，JSON 文件路径，支持 per_module、plan list、list 或直接 dict 格式。
# 输出: dict；键为 module_name，值为包含 abits/wbits/group size 的规范化配置。
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
            if "weight_channel_group_size" not in entry and "w_channel_group_size" in entry:
                entry["weight_channel_group_size"] = entry["w_channel_group_size"]
            if "weight_channel_group_size" not in entry and "output_channel_group_size" in entry:
                entry["weight_channel_group_size"] = entry["output_channel_group_size"]

        if "abits" not in entry or "wbits" not in entry:
            raise ValueError(f"missing abits/wbits in moe quant plan entry for {module_name}")
        entry["abits"] = int(entry["abits"])
        entry["wbits"] = int(entry["wbits"])
        entry["act_group_size"] = _normalize_group_size(entry.get("act_group_size"))
        entry["weight_group_size"] = _normalize_group_size(entry.get("weight_group_size"))
        entry["weight_channel_group_size"] = _normalize_weight_channel_group_size(
            entry.get("weight_channel_group_size")
        )
        if entry["abits"] not in (4, 8, 16):
            raise ValueError(f"unsupported activation bits {entry['abits']} for {module_name}")
        if entry["wbits"] > entry["abits"]:
            raise ValueError(f"invalid plan entry {module_name}: abits must be >= wbits")
        plan[module_name] = entry
    return plan


# =============================================================================
# Pipeline stages: smooth_lm, duquant, evaluate
# =============================================================================


def run_smooth_lm_stage(
    *,
    lm,  # LMClass 包装对象，内部持有待平滑的模型 lm.model
    dataloader,  # 校准数据，用于前向收集 activation 统计量
    logger,  # 日志对象，用于记录统计缓存和耗时

    # cache identity
    cache_dir,  # Smooth 统计量缓存目录
    model_path,  # 模型路径/名称，参与缓存 key 构造
    model_name,  # 规范化模型结构名，参与缓存 key 构造
    calib_dataset,  # 校准集名称，参与缓存 key 构造
    nsamples,  # 校准样本数量，控制统计前向次数
    seq_length,  # 校准序列长度，参与缓存 key 构造
    seed,  # 校准数据随机种子，参与缓存 key 构造
    attn_implementation,  # attention 实现类型，参与缓存 key 构造
    disable_act_stats_cache,  # 是否禁用 Smooth activation 统计缓存
    refresh_act_stats_cache,  # 是否强制重新计算并覆盖 Smooth 统计缓存

    # smooth / migration config
    fc1_scale_merge,  # MoE fc1/gate/up 平滑尺度构造策略；当前仅支持 act_mean/act_p99
    alpha,  # SmoothQuant 中 activation scale 与 weight scale 的平衡系数
    otsu_ratio,  # Otsu outlier 判断使用的分位比例
    otsu_smooth_rate,  # Otsu outlier 通道的平滑强度
    moe_down_smooth_mode,  # MoE down_proj 平滑模式，duquant 或 otsu
    act_mean_beta,  # act_mean 策略下对均值归一化 scale 的放大系数

    # OSE score preparation config
    moe_outlier_score,  # smooth 后、DuQuant 前统一生成 OSE 通道选择分数的方法
    args,  # argparse.Namespace，提供 topk、量化位宽/group size 等 score 配置
):
    _sync_cuda_for_timing()
    smooth_start = time.perf_counter()
    convert_device(lm, model_name)  # 设置分片

    if fc1_scale_merge not in ("act_mean", "act_p99"):
        raise ValueError(
            "--fc1_scale_merge must be either 'act_mean' or 'act_p99' "
            "in the current MoE-aware smoothing pipeline."
        )

    moe_stat_name = "moe_act_means" if fc1_scale_merge == "act_mean" else "moe_act_p99s"
    act_samples = {}
    weight_scores = {}
    router_logits = {}

    smooth_stat_values = {}
    missing_smooth_stats = []
    for stat_name in (moe_stat_name, "act_scales", "act_per_channel_scales"):
        found, value = _load_smooth_stat_cache(
            cache_dir=cache_dir,
            model=model_path,
            model_name=model_name,
            calib_dataset=calib_dataset,
            nsamples=nsamples,
            seq_length=seq_length,
            seed=seed,
            attn_implementation=attn_implementation,
            disable_act_stats_cache=disable_act_stats_cache,
            refresh_act_stats_cache=refresh_act_stats_cache,
            logger=logger,
            stat_name=stat_name,
        )
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
            nsamples,
            collect_moe_act_mean="moe_act_means" in missing_smooth_stats,
            collect_moe_act_p99="moe_act_p99s" in missing_smooth_stats,
            collect_act_scales="act_scales" in missing_smooth_stats,
            collect_act_per_channel_scales="act_per_channel_scales" in missing_smooth_stats,
        )
        for stat_name in missing_smooth_stats:
            smooth_stat_values[stat_name] = combined_stats[stat_name]
            _save_smooth_stat_cache(
                cache_dir=cache_dir,
                model=model_path,
                model_name=model_name,
                calib_dataset=calib_dataset,
                nsamples=nsamples,
                seq_length=seq_length,
                seed=seed,
                attn_implementation=attn_implementation,
                disable_act_stats_cache=disable_act_stats_cache,
                logger=logger,
                stat_name=stat_name,
                value=combined_stats[stat_name],
            )

    moe_act_stats = smooth_stat_values[moe_stat_name]
    act_scales = smooth_stat_values["act_scales"]
    act_per_channel_scales = smooth_stat_values["act_per_channel_scales"]

    moe_fc1_smooth_scales = build_moe_fc1_smooth_scales(
        moe_act_stats,
        act_mean_beta=act_mean_beta,
        model=lm.model,
    )
    logger.info(
        f"built moe_fc1_smooth_scales entries: {len(moe_fc1_smooth_scales)} "
        f"from {moe_stat_name}"
    )

    smooth_lm(
        lm.model,
        act_scales,
        act_per_channel_scales,
        act_samples,
        weight_scores,
        router_logits,
        fc1_scale_merge=fc1_scale_merge,
        alpha=alpha,
        otsu_ratio=otsu_ratio,
        otsu_smooth_rate=otsu_smooth_rate,
        moe_down_smooth_mode=moe_down_smooth_mode,
        moe_fc1_smooth_scales=moe_fc1_smooth_scales,
        logger=logger,
    )

    moe_outlier_scores = prepare_moe_outlier_scores(
        lm.model,
        score_method=moe_outlier_score,
        moe_fc1_smooth_scales=moe_fc1_smooth_scales,
        args=args,
        model_name=model_name,
        logger=logger,
    )

    lm.model.cpu()
    _sync_cuda_for_timing()
    smooth_seconds = time.perf_counter() - smooth_start
    logger.info(f"[timing] smooth_seconds: {smooth_seconds:.6f}")
    return {
        "smooth_seconds": smooth_seconds,
        "moe_fc1_smooth_scales": moe_fc1_smooth_scales,
        "moe_outlier_scores": moe_outlier_scores,
    }


def run_duquant_stage(*, lm, duquant_args, dataloader, logger):
    _sync_cuda_for_timing()
    duquant_start = time.perf_counter()
    duquant(
        lm,
        duquant_args,
        dataloader,
        logger=logger,
    )
    _sync_cuda_for_timing()
    duquant_seconds = time.perf_counter() - duquant_start
    logger.info(f"[timing] duquant_seconds: {duquant_seconds:.6f}")
    return duquant_seconds


def main():
    import argparse

    # =============================================================================
    # CLI arguments
    # =============================================================================

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
    parser.add_argument(
        "--weight_channel_group_size",
        type=int,
        default=None,
        help=(
            "number of output channels sharing one weight scale; "
            "None/1 keeps per-output-channel scales, -1 uses one tensor scale per Linear"
        ),
    )
    parser.add_argument(
        "--router_weight_channel_group_size",
        type=int,
        default=None,
        help=(
            "output-channel group size for router weights; "
            "defaults to --weight_channel_group_size when unset"
        ),
    )
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
    parser.add_argument("--model_name", type=str, default=None, help="normalized model structure name, e.g. olmoe or qwen2_moe")

    parser.add_argument("--seq_length", type=int, default=4096)
    parser.add_argument(
        "--fc1_scale_merge",
        type=str,
        default="act_mean",
        choices=["act_mean", "act_p99"],
        help="MoE gate/up migration scale construction method",
    )
    parser.add_argument("--moe_down_smooth_mode", type=str, default="duquant", choices=["duquant", "otsu"], help="MoE down_proj smoothing mode; duquant matches the original DuQuant fc-fc scale")
    parser.add_argument("--act_mean_beta", type=float, default=2.0, help="beta for act_mean MoE fc1 smoothing scale")
    parser.add_argument("--router_w_dynamic_method", type=str, default="per_channel", help="the dynamic_method of the gate layer's weight")
    parser.add_argument("--scale_search_steps", type=int, default=100, help="number of clipping-ratio candidates for weight scale search; 0 disables multi-trial search")
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
        default="smooth_scale",
        choices=["smooth_scale", "weight_max", "weight_error"],
        help="score used to select MoE gate/up outlier input columns",
    )
    parser.add_argument(
        "--moe_quant_plan",
        type=str,
        default=None,
        help="JSON plan assigning per-module MoE expert gate/up/down abits and wbits",
    )

    args = parser.parse_args()

    # =============================================================================
    # Argument validation, reproducibility, and fixed runtime defaults
    # =============================================================================

    if args.moe_outlier_topk < 0:
        raise ValueError("--moe_outlier_topk must be non-negative")
    if args.weight_channel_group_size is not None and args.weight_channel_group_size == 0:
        raise ValueError("--weight_channel_group_size must be positive, -1, or unset")
    if args.router_weight_channel_group_size is not None and args.router_weight_channel_group_size == 0:
        raise ValueError("--router_weight_channel_group_size must be positive, -1, or unset")
    if args.router_weight_channel_group_size is None:
        args.router_weight_channel_group_size = args.weight_channel_group_size
    if args.moe_outlier_score == "smooth_scale" and not args.smooth:
        raise ValueError("--moe_outlier_score smooth_scale requires --smooth")
    args.moe_outlier_scores = {}
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

    # =============================================================================
    # Output directory and logger
    # =============================================================================

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
    
    # =============================================================================
    # Model loading and model-name setup
    # =============================================================================

    if args.model_name is None:
        args.model_name = args.model.split('/')[-1]
    lm = LMClass(args)
    if "qwen" in args.model_name.lower():
        args.model_name = "qwen2_moe"
    elif "olmoe" in args.model_name.lower():
        args.model_name = "olmoe"
    elif "pangumoe" in args.model_name.lower():
        args.model_name = "pangumoe"
    else:
        args.model_name = args.model_name.split('-')[0]
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

    # =============================================================================
    # Quantizer parameter assembly
    # =============================================================================

    args.weight_quant_params = {
        "n_bits": args.wbits,
        "per_channel_axes": [0],
        "symmetric": args.symmetric,
        "dynamic_method": args.w_dynamic_method,
        "group_size": args.group_size,
        "weight_channel_group_size": args.weight_channel_group_size,
        "swc":args.swc,
        "quant_method": args.quant_method,
        "block_size": args.block_size,
        "max_rotation_step": args.max_rotation_step,
        "permutation_times": args.permutation_times,
        "scale_search_steps": args.scale_search_steps,
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
        "weight_channel_group_size": args.router_weight_channel_group_size,
        "swc":args.swc,
        "quant_method": args.quant_method,
        "block_size": args.block_size,
        "max_rotation_step": args.max_rotation_step,
        "permutation_times": args.permutation_times,
        "scale_search_steps": args.scale_search_steps,
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

    # =============================================================================
    # Calibration dataset setup
    # =============================================================================

    if args.wbits < 16 or args.abits <16:
        logger.info("=== start quantization ===")
        dataloader_start = time.perf_counter()
        # load calibration dataset
        cache_dataloader = f'{args.cache_dir}/dataloader_{args.model_name}_{args.calib_dataset}_{args.nsamples}_{args.seq_length}.cache'
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

        # =============================================================================
        # Stage 1: Smooth_LM transform and smooth-dependent statistics
        # =============================================================================

        if args.smooth:
            smooth_stage = run_smooth_lm_stage(
                lm=lm,
                dataloader=dataloader,
                logger=logger,
                cache_dir=args.cache_dir,
                model_path=args.model,
                model_name=args.model_name,
                calib_dataset=args.calib_dataset,
                nsamples=args.nsamples,
                seq_length=args.seq_length,
                seed=args.seed,
                attn_implementation=args.attn_implementation,
                disable_act_stats_cache=args.disable_act_stats_cache,
                refresh_act_stats_cache=args.refresh_act_stats_cache,
                fc1_scale_merge=args.fc1_scale_merge,
                alpha=args.alpha,
                otsu_ratio=args.otsu_ratio,
                otsu_smooth_rate=args.otsu_smooth_rate,
                moe_down_smooth_mode=args.moe_down_smooth_mode,
                act_mean_beta=args.act_mean_beta,
                moe_outlier_score=args.moe_outlier_score,
                args=args,
            )
            args.moe_outlier_scores = smooth_stage["moe_outlier_scores"]
        elif args.moe_outlier_topk > 0 and args.moe_outlier_score in ("weight_max", "weight_error"):
            logger.info(
                "prepare module-level moe_outlier_scores before DuQuant "
                f"without smooth; score_method={args.moe_outlier_score}"
            )
            args.moe_outlier_scores = prepare_moe_outlier_scores(
                lm.model,
                score_method=args.moe_outlier_score,
                args=args,
                model_name=args.model_name,
                logger=logger,
            )

        if args.moe_outlier_topk > 0 and not args.moe_outlier_scores:
            raise ValueError(
                "--moe_outlier_topk > 0 requires non-empty args.moe_outlier_scores "
                "before run_duquant_stage"
            )

        # =============================================================================
        # Stage 2: DuQuant layer-wise calibration and quantization
        # =============================================================================

        run_duquant_stage(
            lm=lm,
            duquant_args=args,
            dataloader=dataloader,
            logger=logger,
        )
        calibration_seconds = time.perf_counter() - calibration_start
        logger.info(f"[timing] calibration_seconds: {calibration_seconds:.6f}")

        dataloader = None
        gc.collect()
        torch.cuda.empty_cache()

    # =============================================================================
    # Stage 3: Evaluation
    # =============================================================================

    logger.info(f"args.output_dir: {args.output_dir}")
    logger.info(f"args.tasks: {args.tasks}")
    evaluate(lm, args, logger)


if __name__ == "__main__":
    print(sys.argv)
    main()
