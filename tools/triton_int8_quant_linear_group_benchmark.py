#!/usr/bin/env python3
"""Single-layer INT8 quantized Linear/MoE benchmark with output-channel groups.

Project convention:
  X is [M, K], W is [N, K], and Y = X @ W.T.

This benchmark measures the runtime path after activation/weight quantization:
  X_q int8, W_q int8 -> INT32 GEMM -> FP16 output with activation and weight
  scales applied in the epilogue.

Weight scale sharing follows the project's output-channel grouping:
  W[n, k] ~= W_q[n, k] * s_w[floor(n / group_size)]

The GEMM body is identical across group sizes.  Only legal epilogue scale
indexing/loading changes.  The `moe_expert` scope launches gate, up, and down
projections for a routed expert batch to mimic one MoE MLP layer shape.
"""

import argparse
import csv
import math
import os
import statistics
from typing import Dict, Iterable, List, Sequence, Tuple

import torch
import triton
import triton.language as tl


LINEAR_SHAPES: Dict[str, Tuple[int, int]] = {
    # name: (K, N)
    "olmoe_gate": (2048, 1024),
    "olmoe_up": (2048, 1024),
    "olmoe_down": (1024, 2048),
    "qwen_gate": (2048, 1408),
    "qwen_up": (2048, 1408),
    "qwen_down": (1408, 2048),
}

MOE_PRESETS: Dict[str, Tuple[int, int, int]] = {
    # name: (hidden K, intermediate I, output N)
    "olmoe": (2048, 1024, 2048),
    "qwen": (2048, 1408, 2048),
}


@triton.jit
def _int8_gemm_output_scale_kernel(
    x_ptr,
    w_ptr,
    act_scale_ptr,
    weight_scale_ptr,
    y_ptr,
    M: tl.constexpr,
    K: tl.constexpr,
    N: tl.constexpr,
    CHANNELS_PER_SCALE: tl.constexpr,
    ACT_PER_TOKEN: tl.constexpr,
    WEIGHT_PER_TENSOR: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    acc = tl.zeros((BLOCK_M, BLOCK_N), tl.int32)

    for k0 in range(0, K, BLOCK_K):
        k = k0 + offs_k
        x = tl.load(
            x_ptr + offs_m[:, None] * K + k[None, :],
            mask=(offs_m[:, None] < M) & (k[None, :] < K),
            other=0,
        )
        w = tl.load(
            w_ptr + offs_n[None, :] * K + k[:, None],
            mask=(offs_n[None, :] < N) & (k[:, None] < K),
            other=0,
        )
        acc += tl.dot(x, w, out_dtype=tl.int32)

    if ACT_PER_TOKEN:
        sx = tl.load(act_scale_ptr + offs_m, mask=offs_m < M, other=1.0).to(tl.float32)
    else:
        sx = tl.load(act_scale_ptr + 0).to(tl.float32)

    if WEIGHT_PER_TENSOR:
        sw = tl.load(weight_scale_ptr + 0).to(tl.float32)
    else:
        scale_ids = offs_n // CHANNELS_PER_SCALE
        sw = tl.load(weight_scale_ptr + scale_ids, mask=offs_n < N, other=1.0).to(tl.float32)

    out = acc.to(tl.float32) * sx[:, None] * sw[None, :]
    tl.store(
        y_ptr + offs_m[:, None] * N + offs_n[None, :],
        out.to(tl.float16),
        mask=(offs_m[:, None] < M) & (offs_n[None, :] < N),
    )


@triton.jit
def _grouped_int8_gemm_output_scale_kernel(
    x_ptr,
    w_ptr,
    act_scale_ptr,
    weight_scale_ptr,
    y_ptr,
    E: tl.constexpr,
    T: tl.constexpr,
    K: tl.constexpr,
    N: tl.constexpr,
    NUM_WEIGHT_SCALES: tl.constexpr,
    CHANNELS_PER_SCALE: tl.constexpr,
    ACT_PER_TOKEN: tl.constexpr,
    WEIGHT_PER_TENSOR: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_t = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_e = tl.program_id(2)
    offs_t = pid_t * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    acc = tl.zeros((BLOCK_M, BLOCK_N), tl.int32)

    x_base = pid_e * T * K
    w_base = pid_e * N * K
    for k0 in range(0, K, BLOCK_K):
        k = k0 + offs_k
        x = tl.load(
            x_ptr + x_base + offs_t[:, None] * K + k[None, :],
            mask=(offs_t[:, None] < T) & (k[None, :] < K),
            other=0,
        )
        w = tl.load(
            w_ptr + w_base + offs_n[None, :] * K + k[:, None],
            mask=(offs_n[None, :] < N) & (k[:, None] < K),
            other=0,
        )
        acc += tl.dot(x, w, out_dtype=tl.int32)

    if ACT_PER_TOKEN:
        sx = tl.load(
            act_scale_ptr + pid_e * T + offs_t,
            mask=offs_t < T,
            other=1.0,
        ).to(tl.float32)
    else:
        sx = tl.load(act_scale_ptr + 0).to(tl.float32)

    scale_base = pid_e * NUM_WEIGHT_SCALES
    if WEIGHT_PER_TENSOR:
        sw = tl.load(weight_scale_ptr + scale_base).to(tl.float32)
    else:
        scale_ids = offs_n // CHANNELS_PER_SCALE
        sw = tl.load(
            weight_scale_ptr + scale_base + scale_ids,
            mask=offs_n < N,
            other=1.0,
        ).to(tl.float32)

    y_base = pid_e * T * N
    out = acc.to(tl.float32) * sx[:, None] * sw[None, :]
    tl.store(
        y_ptr + y_base + offs_t[:, None] * N + offs_n[None, :],
        out.to(tl.float16),
        mask=(offs_t[:, None] < T) & (offs_n[None, :] < N),
    )


@triton.jit
def _epilogue_output_scale_kernel(
    acc_ptr,
    act_scale_ptr,
    weight_scale_ptr,
    y_ptr,
    M: tl.constexpr,
    N: tl.constexpr,
    CHANNELS_PER_SCALE: tl.constexpr,
    ACT_PER_TOKEN: tl.constexpr,
    WEIGHT_PER_TENSOR: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    acc = tl.load(
        acc_ptr + offs_m[:, None] * N + offs_n[None, :],
        mask=(offs_m[:, None] < M) & (offs_n[None, :] < N),
        other=0,
    ).to(tl.float32)

    if ACT_PER_TOKEN:
        sx = tl.load(act_scale_ptr + offs_m, mask=offs_m < M, other=1.0).to(tl.float32)
    else:
        sx = tl.load(act_scale_ptr + 0).to(tl.float32)

    if WEIGHT_PER_TENSOR:
        sw = tl.load(weight_scale_ptr + 0).to(tl.float32)
    else:
        scale_ids = offs_n // CHANNELS_PER_SCALE
        sw = tl.load(weight_scale_ptr + scale_ids, mask=offs_n < N, other=1.0).to(tl.float32)

    tl.store(
        y_ptr + offs_m[:, None] * N + offs_n[None, :],
        (acc * sx[:, None] * sw[None, :]).to(tl.float16),
        mask=(offs_m[:, None] < M) & (offs_n[None, :] < N),
    )


def parse_int_list(value: str) -> List[int]:
    out = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not out:
        raise ValueError(f"empty integer list: {value!r}")
    return out


def parse_str_list(value: str) -> List[str]:
    out = [item.strip() for item in value.split(",") if item.strip()]
    if not out:
        raise ValueError(f"empty list: {value!r}")
    return out


def parse_linear_shapes(value: str) -> List[Tuple[str, int, int]]:
    shapes: List[Tuple[str, int, int]] = []
    for item in parse_str_list(value):
        if item in LINEAR_SHAPES:
            k, n = LINEAR_SHAPES[item]
            shapes.append((item, k, n))
            continue
        parts = item.split(":")
        if len(parts) != 3:
            valid = ",".join(sorted(LINEAR_SHAPES))
            raise ValueError(f"invalid shape {item!r}; use preset ({valid}) or name:K:N")
        name, k_str, n_str = parts
        shapes.append((name, int(k_str), int(n_str)))
    return shapes


def channels_and_scales(group_size: int, n: int) -> Tuple[int, int, bool]:
    if group_size <= 0:
        channels = n
    else:
        channels = min(group_size, n)
    num_scales = triton.cdiv(n, channels)
    return channels, num_scales, num_scales == 1


def make_scales(num: int, device: str = "cuda") -> torch.Tensor:
    return (torch.rand((num,), device=device, dtype=torch.float32) * 0.01 + 0.001).contiguous()


def scale_per_channel(scales: torch.Tensor, n: int, channels: int, per_tensor: bool) -> torch.Tensor:
    if per_tensor:
        return scales.reshape(-1)[0].expand(n)
    idx = torch.arange(n, device=scales.device) // channels
    return scales.reshape(-1)[idx]


def act_scale_for_ref(scales: torch.Tensor, m: int, act_mode: str) -> torch.Tensor:
    if act_mode == "per_token":
        return scales[:m]
    return scales.reshape(-1)[0].expand(m)


def reference_single(
    x: torch.Tensor,
    w: torch.Tensor,
    act_scale: torch.Tensor,
    weight_scale: torch.Tensor,
    channels: int,
    weight_per_tensor: bool,
    act_mode: str,
) -> torch.Tensor:
    acc = torch.matmul(x.to(torch.float32), w.to(torch.float32).t())
    sx = act_scale_for_ref(act_scale, x.shape[0], act_mode)
    sw = scale_per_channel(weight_scale, w.shape[0], channels, weight_per_tensor)
    return (acc * sx[:, None] * sw[None, :]).to(torch.float16)


def reference_epilogue(
    acc: torch.Tensor,
    act_scale: torch.Tensor,
    weight_scale: torch.Tensor,
    channels: int,
    weight_per_tensor: bool,
    act_mode: str,
) -> torch.Tensor:
    sx = act_scale_for_ref(act_scale, acc.shape[0], act_mode)
    sw = scale_per_channel(weight_scale, acc.shape[1], channels, weight_per_tensor)
    return (acc.to(torch.float32) * sx[:, None] * sw[None, :]).to(torch.float16)


def error_stats(actual: torch.Tensor, expected: torch.Tensor) -> Tuple[float, float]:
    diff = (actual.to(torch.float32) - expected.to(torch.float32)).abs()
    rel = diff / expected.to(torch.float32).abs().clamp_min(1e-6)
    return float(diff.max().item()), float(rel.max().item())


def measure_ms(fn, warmup: int, repeat: int, rounds: int) -> Tuple[float, float, float, float]:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    values = []
    for _ in range(rounds):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(repeat):
            fn()
        end.record()
        torch.cuda.synchronize()
        values.append(start.elapsed_time(end) / repeat)
    values = sorted(values)
    q20 = values[max(0, min(len(values) - 1, int(round(0.2 * (len(values) - 1)))))]
    q80 = values[max(0, min(len(values) - 1, int(round(0.8 * (len(values) - 1)))))]
    stdev = statistics.stdev(values) if len(values) > 1 else 0.0
    return statistics.median(values), q20, q80, stdev


def write_csv(path: str, rows: Iterable[dict], fieldnames: Sequence[str]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def run_single_case(args, shape_name: str, m: int, k: int, n: int, group_size: int, act_mode: str, scope: str) -> dict:
    channels, num_scales, weight_per_tensor = channels_and_scales(group_size, n)
    act_per_token = act_mode == "per_token"
    act_scale_count = m if act_per_token else 1
    act_scale = make_scales(act_scale_count)
    weight_scale = make_scales(num_scales)
    y = torch.empty((m, n), device="cuda", dtype=torch.float16)

    if scope == "single_gemm":
        x = torch.randint(-128, 127, (m, k), device="cuda", dtype=torch.int8)
        w = torch.randint(-128, 127, (n, k), device="cuda", dtype=torch.int8)
        grid = (triton.cdiv(m, args.block_m), triton.cdiv(n, args.block_n))

        def fn():
            _int8_gemm_output_scale_kernel[grid](
                x,
                w,
                act_scale,
                weight_scale,
                y,
                M=m,
                K=k,
                N=n,
                CHANNELS_PER_SCALE=channels,
                ACT_PER_TOKEN=act_per_token,
                WEIGHT_PER_TENSOR=weight_per_tensor,
                BLOCK_M=args.block_m,
                BLOCK_N=args.block_n,
                BLOCK_K=args.block_k,
                num_warps=args.num_warps,
                num_stages=4,
            )

        kernel_name = "triton_i8i8_i32_gemm_output_scale"
        fn()
        torch.cuda.synchronize()
        max_abs, max_rel = error_stats(
            y,
            reference_single(x, w, act_scale, weight_scale, channels, weight_per_tensor, act_mode),
        )
    elif scope == "epilogue_only":
        acc = torch.randint(-65536, 65536, (m, n), device="cuda", dtype=torch.int32)
        grid = (triton.cdiv(m, args.block_m), triton.cdiv(n, args.block_n))

        def fn():
            _epilogue_output_scale_kernel[grid](
                acc,
                act_scale,
                weight_scale,
                y,
                M=m,
                N=n,
                CHANNELS_PER_SCALE=channels,
                ACT_PER_TOKEN=act_per_token,
                WEIGHT_PER_TENSOR=weight_per_tensor,
                BLOCK_M=args.block_m,
                BLOCK_N=args.block_n,
                num_warps=args.num_warps,
                num_stages=4,
            )

        kernel_name = "triton_i32_epilogue_output_scale"
        fn()
        torch.cuda.synchronize()
        max_abs, max_rel = error_stats(
            y,
            reference_epilogue(acc, act_scale, weight_scale, channels, weight_per_tensor, act_mode),
        )
    else:
        raise ValueError(f"unknown single scope: {scope}")

    median, q20, q80, stdev = measure_ms(fn, args.warmup, args.repeat, args.rounds)
    return make_row(
        args,
        scope,
        shape_name,
        m,
        k,
        n,
        0,
        0,
        group_size,
        channels,
        num_scales,
        act_mode,
        act_scale_count,
        median,
        q20,
        q80,
        stdev,
        max_abs,
        max_rel,
        kernel_name,
    )


def run_moe_case(args, preset: str, tokens: int, experts: int, group_size: int, act_mode: str, scope: str) -> dict:
    hidden, intermediate, out_features = MOE_PRESETS[preset]
    channels_gate, gate_scales, gate_per_tensor = channels_and_scales(group_size, intermediate)
    channels_down, down_scales, down_per_tensor = channels_and_scales(group_size, out_features)
    act_per_token = act_mode == "per_token"
    act_scale_count = experts * tokens if act_per_token else 1

    act_scale_hidden = make_scales(act_scale_count)
    act_scale_inter = make_scales(act_scale_count)
    gate_scale = make_scales(experts * gate_scales).view(experts, gate_scales)
    up_scale = make_scales(experts * gate_scales).view(experts, gate_scales)
    down_scale = make_scales(experts * down_scales).view(experts, down_scales)

    grid_gate = (triton.cdiv(tokens, args.block_m), triton.cdiv(intermediate, args.block_n), experts)
    grid_down = (triton.cdiv(tokens, args.block_m), triton.cdiv(out_features, args.block_n), experts)

    if scope == "moe_expert":
        x_hidden = torch.randint(-128, 127, (experts, tokens, hidden), device="cuda", dtype=torch.int8)
        x_inter = torch.randint(-128, 127, (experts, tokens, intermediate), device="cuda", dtype=torch.int8)
        w_gate = torch.randint(-128, 127, (experts, intermediate, hidden), device="cuda", dtype=torch.int8)
        w_up = torch.randint(-128, 127, (experts, intermediate, hidden), device="cuda", dtype=torch.int8)
        w_down = torch.randint(-128, 127, (experts, out_features, intermediate), device="cuda", dtype=torch.int8)
        y_gate = torch.empty((experts, tokens, intermediate), device="cuda", dtype=torch.float16)
        y_up = torch.empty((experts, tokens, intermediate), device="cuda", dtype=torch.float16)
        y_down = torch.empty((experts, tokens, out_features), device="cuda", dtype=torch.float16)

        def fn():
            _grouped_int8_gemm_output_scale_kernel[grid_gate](
                x_hidden,
                w_gate,
                act_scale_hidden,
                gate_scale,
                y_gate,
                E=experts,
                T=tokens,
                K=hidden,
                N=intermediate,
                NUM_WEIGHT_SCALES=gate_scales,
                CHANNELS_PER_SCALE=channels_gate,
                ACT_PER_TOKEN=act_per_token,
                WEIGHT_PER_TENSOR=gate_per_tensor,
                BLOCK_M=args.block_m,
                BLOCK_N=args.block_n,
                BLOCK_K=args.block_k,
                num_warps=args.num_warps,
                num_stages=4,
            )
            _grouped_int8_gemm_output_scale_kernel[grid_gate](
                x_hidden,
                w_up,
                act_scale_hidden,
                up_scale,
                y_up,
                E=experts,
                T=tokens,
                K=hidden,
                N=intermediate,
                NUM_WEIGHT_SCALES=gate_scales,
                CHANNELS_PER_SCALE=channels_gate,
                ACT_PER_TOKEN=act_per_token,
                WEIGHT_PER_TENSOR=gate_per_tensor,
                BLOCK_M=args.block_m,
                BLOCK_N=args.block_n,
                BLOCK_K=args.block_k,
                num_warps=args.num_warps,
                num_stages=4,
            )
            _grouped_int8_gemm_output_scale_kernel[grid_down](
                x_inter,
                w_down,
                act_scale_inter,
                down_scale,
                y_down,
                E=experts,
                T=tokens,
                K=intermediate,
                N=out_features,
                NUM_WEIGHT_SCALES=down_scales,
                CHANNELS_PER_SCALE=channels_down,
                ACT_PER_TOKEN=act_per_token,
                WEIGHT_PER_TENSOR=down_per_tensor,
                BLOCK_M=args.block_m,
                BLOCK_N=args.block_n,
                BLOCK_K=args.block_k,
                num_warps=args.num_warps,
                num_stages=4,
            )

        kernel_name = "3x_triton_grouped_i8i8_i32_gemm_output_scale"
        max_abs = 0.0
        max_rel = 0.0
    elif scope == "moe_epilogue_only":
        acc_gate = torch.randint(-65536, 65536, (tokens, intermediate), device="cuda", dtype=torch.int32)
        acc_up = torch.randint(-65536, 65536, (tokens, intermediate), device="cuda", dtype=torch.int32)
        acc_down = torch.randint(-65536, 65536, (tokens, out_features), device="cuda", dtype=torch.int32)
        y_gate = torch.empty((tokens, intermediate), device="cuda", dtype=torch.float16)
        y_up = torch.empty((tokens, intermediate), device="cuda", dtype=torch.float16)
        y_down = torch.empty((tokens, out_features), device="cuda", dtype=torch.float16)
        # Epilogue-only uses one representative expert's qparams but repeats the
        # three projection shapes; this isolates scale indexing/apply cost.
        gate_scale_1 = gate_scale[0]
        up_scale_1 = up_scale[0]
        down_scale_1 = down_scale[0]
        grid_gate_single = (triton.cdiv(tokens, args.block_m), triton.cdiv(intermediate, args.block_n))
        grid_down_single = (triton.cdiv(tokens, args.block_m), triton.cdiv(out_features, args.block_n))

        def fn():
            _epilogue_output_scale_kernel[grid_gate_single](
                acc_gate,
                act_scale_hidden,
                gate_scale_1,
                y_gate,
                M=tokens,
                N=intermediate,
                CHANNELS_PER_SCALE=channels_gate,
                ACT_PER_TOKEN=act_per_token,
                WEIGHT_PER_TENSOR=gate_per_tensor,
                BLOCK_M=args.block_m,
                BLOCK_N=args.block_n,
                num_warps=args.num_warps,
                num_stages=4,
            )
            _epilogue_output_scale_kernel[grid_gate_single](
                acc_up,
                act_scale_hidden,
                up_scale_1,
                y_up,
                M=tokens,
                N=intermediate,
                CHANNELS_PER_SCALE=channels_gate,
                ACT_PER_TOKEN=act_per_token,
                WEIGHT_PER_TENSOR=gate_per_tensor,
                BLOCK_M=args.block_m,
                BLOCK_N=args.block_n,
                num_warps=args.num_warps,
                num_stages=4,
            )
            _epilogue_output_scale_kernel[grid_down_single](
                acc_down,
                act_scale_inter,
                down_scale_1,
                y_down,
                M=tokens,
                N=out_features,
                CHANNELS_PER_SCALE=channels_down,
                ACT_PER_TOKEN=act_per_token,
                WEIGHT_PER_TENSOR=down_per_tensor,
                BLOCK_M=args.block_m,
                BLOCK_N=args.block_n,
                num_warps=args.num_warps,
                num_stages=4,
            )

        kernel_name = "3x_triton_i32_epilogue_output_scale"
        max_abs = 0.0
        max_rel = 0.0
    else:
        raise ValueError(f"unknown MoE scope: {scope}")

    fn()
    torch.cuda.synchronize()
    median, q20, q80, stdev = measure_ms(fn, args.warmup, args.repeat, args.rounds)
    return make_row(
        args,
        scope,
        preset,
        experts * tokens,
        hidden,
        out_features,
        experts,
        tokens,
        group_size,
        channels_gate,
        gate_scales + gate_scales + down_scales,
        act_mode,
        act_scale_count,
        median,
        q20,
        q80,
        stdev,
        max_abs,
        max_rel,
        kernel_name,
    )


def make_row(
    args,
    scope: str,
    shape_name: str,
    m: int,
    k: int,
    n: int,
    experts: int,
    tokens: int,
    group_size: int,
    channels: int,
    num_weight_scales: int,
    act_mode: str,
    act_scale_count: int,
    latency: float,
    q20: float,
    q80: float,
    stdev: float,
    max_abs: float,
    max_rel: float,
    kernel_name: str,
) -> dict:
    return {
        "scope": scope,
        "shape_name": shape_name,
        "M": m,
        "K": k,
        "N": n,
        "num_active_experts": experts,
        "tokens_per_expert": tokens,
        "group_size": group_size,
        "channels_per_scale": channels,
        "num_weight_scales": num_weight_scales,
        "weight_scale_reduction_vs_group1": "",
        "act_scale_mode": act_mode,
        "num_act_scales": act_scale_count,
        "latency_ms": f"{latency:.6f}",
        "latency_q20_ms": f"{q20:.6f}",
        "latency_q80_ms": f"{q80:.6f}",
        "latency_std_ms": f"{stdev:.6f}",
        "speedup_over_group1": "",
        "max_abs_error": f"{max_abs:.6e}",
        "max_rel_error": f"{max_rel:.6e}",
        "kernel_name": kernel_name,
        "block_m": args.block_m,
        "block_n": args.block_n,
        "block_k": args.block_k,
        "num_warps": args.num_warps,
        "dtype_x": "int8",
        "dtype_w": "int8",
        "dtype_acc": "int32",
        "dtype_y": "float16",
    }


def add_speedups(rows: List[dict]) -> None:
    baselines: Dict[Tuple, Tuple[float, int]] = {}
    for row in rows:
        key = (
            row["scope"],
            row["shape_name"],
            row["M"],
            row["K"],
            row["N"],
            row["num_active_experts"],
            row["tokens_per_expert"],
            row["act_scale_mode"],
        )
        if int(row["group_size"]) == 1:
            baselines[key] = (float(row["latency_ms"]), max(int(row["num_weight_scales"]), 1))
    for row in rows:
        key = (
            row["scope"],
            row["shape_name"],
            row["M"],
            row["K"],
            row["N"],
            row["num_active_experts"],
            row["tokens_per_expert"],
            row["act_scale_mode"],
        )
        baseline = baselines.get(key)
        if baseline:
            base_latency, base_scales = baseline
            row["speedup_over_group1"] = f"{base_latency / float(row['latency_ms']):.6f}"
            row["weight_scale_reduction_vs_group1"] = f"{base_scales / max(int(row['num_weight_scales']), 1):.6f}"


FIELDS = [
    "scope",
    "shape_name",
    "M",
    "K",
    "N",
    "num_active_experts",
    "tokens_per_expert",
    "group_size",
    "channels_per_scale",
    "num_weight_scales",
    "weight_scale_reduction_vs_group1",
    "act_scale_mode",
    "num_act_scales",
    "latency_ms",
    "latency_q20_ms",
    "latency_q80_ms",
    "latency_std_ms",
    "speedup_over_group1",
    "max_abs_error",
    "max_rel_error",
    "kernel_name",
    "block_m",
    "block_n",
    "block_k",
    "num_warps",
    "dtype_x",
    "dtype_w",
    "dtype_acc",
    "dtype_y",
]


def write_report(output_dir: str) -> None:
    with open(os.path.join(output_dir, "report.md"), "w") as f:
        f.write("# INT8 Quantized Linear Group-Size Benchmark\n\n")
        f.write("This benchmark uses real INT8 dot-product kernels (`tl.dot` with INT32 accumulation). ")
        f.write("It measures the runtime after activation and weight tensors have already been quantized.\n\n")
        f.write("## Scopes\n\n")
        f.write("- `single_gemm`: one quantized Linear projection, including INT8 GEMM and output scaling.\n")
        f.write("- `epilogue_only`: only INT32-to-FP16 scale application; this is an upper bound on group-size runtime benefit.\n")
        f.write("- `moe_expert`: gate, up, and down grouped expert projections for a routed MoE MLP shape.\n")
        f.write("- `moe_epilogue_only`: epilogue-only version of the three MoE projections.\n\n")
        f.write("Group size changes only output-channel weight scale sharing. ")
        f.write("If full GEMM speedup is small while epilogue-only speedup is larger, the reason is that GEMM math and weight traffic dominate the full layer.\n")


def run(args: argparse.Namespace) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    torch.manual_seed(args.seed)
    torch.cuda.set_device(args.device)
    os.makedirs(args.output_dir, exist_ok=True)

    scopes = set(parse_str_list(args.scopes))
    linear_shapes = parse_linear_shapes(args.linear_shapes)
    m_values = parse_int_list(args.m_values)
    group_sizes = parse_int_list(args.group_sizes)
    act_modes = parse_str_list(args.act_scale_modes)
    moe_presets = parse_str_list(args.moe_presets)
    active_experts = parse_int_list(args.num_active_experts)
    tokens_per_expert = parse_int_list(args.tokens_per_expert)

    print(f"[int8-group-bench] output_dir={args.output_dir}")
    print(f"[int8-group-bench] device={torch.cuda.get_device_name(args.device)}")
    print(f"[int8-group-bench] scopes={sorted(scopes)}")
    print(f"[int8-group-bench] group_sizes={group_sizes}")
    print(f"[int8-group-bench] act_scale_modes={act_modes}")
    print(f"[int8-group-bench] warmup={args.warmup} repeat={args.repeat} rounds={args.rounds}")

    rows: List[dict] = []
    for act_mode in act_modes:
        if act_mode not in {"per_token", "per_tensor_static"}:
            raise ValueError("--act_scale_modes supports per_token,per_tensor_static")
        for shape_name, k, n in linear_shapes:
            for m in m_values:
                for group_size in group_sizes:
                    if "single_gemm" in scopes:
                        rows.append(run_single_case(args, shape_name, m, k, n, group_size, act_mode, "single_gemm"))
                        print(f"[int8-group-bench] single_gemm {shape_name} M={m} group={group_size} act={act_mode}")
                    if "epilogue_only" in scopes:
                        rows.append(run_single_case(args, shape_name, m, k, n, group_size, act_mode, "epilogue_only"))
                        print(f"[int8-group-bench] epilogue_only {shape_name} M={m} group={group_size} act={act_mode}")
        for preset in moe_presets:
            if preset not in MOE_PRESETS:
                raise ValueError(f"unknown MoE preset {preset!r}; valid={sorted(MOE_PRESETS)}")
            for experts in active_experts:
                for tokens in tokens_per_expert:
                    for group_size in group_sizes:
                        if "moe_expert" in scopes:
                            rows.append(run_moe_case(args, preset, tokens, experts, group_size, act_mode, "moe_expert"))
                            print(f"[int8-group-bench] moe_expert {preset} E={experts} T={tokens} group={group_size} act={act_mode}")
                        if "moe_epilogue_only" in scopes:
                            rows.append(run_moe_case(args, preset, tokens, experts, group_size, act_mode, "moe_epilogue_only"))
                            print(f"[int8-group-bench] moe_epilogue_only {preset} E={experts} T={tokens} group={group_size} act={act_mode}")

    add_speedups(rows)
    detail_path = os.path.join(args.output_dir, "int8_quant_linear_group_detail.csv")
    write_csv(detail_path, rows, FIELDS)
    write_report(args.output_dir)
    print(f"[int8-group-bench] detail: {detail_path}")
    print(f"[int8-group-bench] report: {os.path.join(args.output_dir, 'report.md')}")


def main() -> None:
    parser = argparse.ArgumentParser(description="INT8 quantized Linear/MoE group-size latency benchmark.")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--scopes", default="single_gemm,epilogue_only,moe_expert,moe_epilogue_only")
    parser.add_argument("--linear_shapes", default="olmoe_gate,olmoe_down")
    parser.add_argument("--m_values", default="1,2,4,8,16,32")
    parser.add_argument("--moe_presets", default="olmoe")
    parser.add_argument("--num_active_experts", default="2,4,8")
    parser.add_argument("--tokens_per_expert", default="1,2,4,8")
    parser.add_argument("--group_sizes", default="1,2,4,8,16,32,64,128,256,512,1024,2048")
    parser.add_argument("--act_scale_modes", default="per_token,per_tensor_static")
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--repeat", type=int, default=500)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--block_m", type=int, default=16)
    parser.add_argument("--block_n", type=int, default=64)
    parser.add_argument("--block_k", type=int, default=64)
    parser.add_argument("--num_warps", type=int, default=4)
    parser.add_argument("--seed", type=int, default=2)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
