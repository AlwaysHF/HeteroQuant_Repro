#!/usr/bin/env python3
"""Fair output-channel scale-sharing benchmark for quantized Linear/MoE.

Project convention:
  X is [M, K], W is [N, K], and Y = X @ W.T.

The benchmark intentionally follows the project's output-channel scale sharing:
  W[n, k] ~= QW[n, k] * scale[floor(n / C)]

Therefore all granularities use the same integer GEMM body.  The only legal
specialization is in the output-scale epilogue: grouped/per-channel uses a
compile-time channels-per-scale index, while per-tensor loads one scalar scale.
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


PRESET_SHAPES: Dict[str, Tuple[int, int]] = {
    # name: (K, N)
    "olmoe_gate_up": (2048, 1024),
    "olmoe_down": (1024, 2048),
    "qwen_gate_up": (2048, 1408),
    "qwen_down": (1408, 2048),
}

GRANULARITY_LABELS = {
    "per_channel": "Per-channel",
    "2_channels": "2",
    "4_channels": "4",
    "8_channels": "8",
    "16_channels": "16",
    "per_tensor": "Per-tensor",
}


@triton.jit
def _full_gemm_output_scale_kernel(
    x_ptr,
    w_ptr,
    scale_ptr,
    y_ptr,
    M: tl.constexpr,
    K: tl.constexpr,
    N: tl.constexpr,
    CHANNELS_PER_SCALE: tl.constexpr,
    IS_PER_TENSOR: tl.constexpr,
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

    # For an output tile, grouped scales need ceil(BLOCK_N / C) distinct scale
    # values in principle.  This vector load is the reasonable optimized form:
    # one scale per output channel lane, with adjacent lanes reusing the same
    # loaded value through cache/registers.  Per-tensor uses exactly one scalar.
    acc_f = acc.to(tl.float32)
    if IS_PER_TENSOR:
        scales = tl.load(scale_ptr + 0).to(tl.float32)
    else:
        scale_ids = offs_n // CHANNELS_PER_SCALE
        scales = tl.load(scale_ptr + scale_ids, mask=offs_n < N, other=1.0).to(tl.float32)
    out = acc_f * scales[None, :]
    tl.store(
        y_ptr + offs_m[:, None] * N + offs_n[None, :],
        out.to(tl.float16),
        mask=(offs_m[:, None] < M) & (offs_n[None, :] < N),
    )


@triton.jit
def _epilogue_only_output_scale_kernel(
    acc_ptr,
    scale_ptr,
    y_ptr,
    M: tl.constexpr,
    N: tl.constexpr,
    CHANNELS_PER_SCALE: tl.constexpr,
    IS_PER_TENSOR: tl.constexpr,
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
    if IS_PER_TENSOR:
        scales = tl.load(scale_ptr + 0).to(tl.float32)
    else:
        scale_ids = offs_n // CHANNELS_PER_SCALE
        scales = tl.load(scale_ptr + scale_ids, mask=offs_n < N, other=1.0).to(tl.float32)
    tl.store(
        y_ptr + offs_m[:, None] * N + offs_n[None, :],
        (acc * scales[None, :]).to(tl.float16),
        mask=(offs_m[:, None] < M) & (offs_n[None, :] < N),
    )


@triton.jit
def _grouped_moe_output_scale_kernel(
    x_ptr,
    w_ptr,
    scale_ptr,
    y_ptr,
    E: tl.constexpr,
    T: tl.constexpr,
    K: tl.constexpr,
    N: tl.constexpr,
    NUM_SCALES: tl.constexpr,
    CHANNELS_PER_SCALE: tl.constexpr,
    IS_PER_TENSOR: tl.constexpr,
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

    scale_base = pid_e * NUM_SCALES
    if IS_PER_TENSOR:
        scales = tl.load(scale_ptr + scale_base).to(tl.float32)
    else:
        scale_ids = offs_n // CHANNELS_PER_SCALE
        scales = tl.load(
            scale_ptr + scale_base + scale_ids,
            mask=offs_n < N,
            other=1.0,
        ).to(tl.float32)

    y_base = pid_e * T * N
    tl.store(
        y_ptr + y_base + offs_t[:, None] * N + offs_n[None, :],
        (acc.to(tl.float32) * scales[None, :]).to(tl.float16),
        mask=(offs_t[:, None] < T) & (offs_n[None, :] < N),
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


def parse_shapes(value: str) -> List[Tuple[str, int, int]]:
    shapes: List[Tuple[str, int, int]] = []
    for item in parse_str_list(value):
        if item in PRESET_SHAPES:
            k, n = PRESET_SHAPES[item]
            shapes.append((item, k, n))
        else:
            parts = item.split(":")
            if len(parts) != 3:
                raise ValueError("shape must be a preset name or custom_name:K:N")
            name, k_str, n_str = parts
            shapes.append((name, int(k_str), int(n_str)))
    return shapes


def parse_granularity(value: str, n: int) -> Tuple[str, int, int, bool]:
    key = value.strip().lower().replace("-", "_")
    if key in {"per_channel", "channel", "1"}:
        return "per_channel", 1, n, False
    if key in {"per_tensor", "tensor", "all"}:
        return "per_tensor", n, 1, True
    if key.endswith("ch"):
        key = key[:-2]
    channels = min(int(key), n)
    if channels <= 0:
        raise ValueError(f"invalid granularity: {value!r}")
    partitions = triton.cdiv(n, channels)
    return f"{channels}_channels", channels, partitions, partitions == 1


def scale_reduction_for_granularity(num_scales: int, n: int) -> float:
    return float(n) / max(float(num_scales), 1.0)


def expert_scale_summary_rows() -> List[dict]:
    out_dims = [1024, 1024, 2048]
    base = sum(out_dims)
    rows = []
    for label, channels in [
        ("per_channel", 1),
        ("2_channels", 2),
        ("4_channels", 4),
        ("8_channels", 8),
        ("16_channels", 16),
        ("per_tensor", None),
    ]:
        if channels is None:
            count = len(out_dims)
            channels_display = "per_tensor"
        else:
            count = sum(math.ceil(dim / channels) for dim in out_dims)
            channels_display = str(channels)
        rows.append(
            {
                "granularity": label,
                "channels_per_scale": channels_display,
                "scale_count_per_expert": count,
                "scale_reduction_vs_per_channel": f"{base / count:.6f}",
                "fp16_scale_bytes_per_expert": count * 2,
            }
        )
    return rows


def make_scale(num_scales: int, device: str = "cuda") -> torch.Tensor:
    return torch.rand((num_scales,), device=device, dtype=torch.float32) * 0.01 + 0.001


def scale_per_channel(scales: torch.Tensor, n: int, channels_per_scale: int, is_per_tensor: bool) -> torch.Tensor:
    if is_per_tensor:
        return scales[0].expand(n)
    idx = torch.arange(n, device=scales.device) // channels_per_scale
    return scales[idx]


def full_reference(x: torch.Tensor, w: torch.Tensor, scales: torch.Tensor, channels: int, is_per_tensor: bool) -> torch.Tensor:
    acc = torch.matmul(x.to(torch.float32), w.to(torch.float32).t())
    return (acc * scale_per_channel(scales, w.shape[0], channels, is_per_tensor).view(1, -1)).to(torch.float16)


def epilogue_reference(acc: torch.Tensor, scales: torch.Tensor, channels: int, is_per_tensor: bool) -> torch.Tensor:
    return (acc.to(torch.float32) * scale_per_channel(scales, acc.shape[1], channels, is_per_tensor).view(1, -1)).to(torch.float16)


def grouped_reference(x: torch.Tensor, w: torch.Tensor, scales: torch.Tensor, channels: int, is_per_tensor: bool) -> torch.Tensor:
    e, t, _ = x.shape
    n = w.shape[1]
    out = []
    for expert_idx in range(e):
        acc = torch.matmul(x[expert_idx].to(torch.float32), w[expert_idx].to(torch.float32).t())
        out.append((acc * scale_per_channel(scales[expert_idx], n, channels, is_per_tensor).view(1, -1)).to(torch.float16))
    return torch.stack(out, dim=0)


def error_stats(actual: torch.Tensor, expected: torch.Tensor) -> Tuple[float, float, float]:
    actual_f = actual.to(torch.float32)
    expected_f = expected.to(torch.float32)
    abs_err = (actual_f - expected_f).abs()
    rel_err = abs_err / expected_f.abs().clamp_min(1e-6)
    return float(abs_err.max().item()), float(abs_err.mean().item()), float(rel_err.max().item())


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


def run_full_gemm_case(args, shape_name: str, m: int, k: int, n: int, granularity: str) -> dict:
    gran, channels, num_scales, is_per_tensor = parse_granularity(granularity, n)
    x = torch.randint(-8, 8, (m, k), device="cuda", dtype=torch.int8)
    w = torch.randint(-8, 8, (n, k), device="cuda", dtype=torch.int8)
    scales = make_scale(num_scales)
    y = torch.empty((m, n), device="cuda", dtype=torch.float16)
    grid = (triton.cdiv(m, args.block_m), triton.cdiv(n, args.block_n))

    def fn():
        _full_gemm_output_scale_kernel[grid](
            x,
            w,
            scales,
            y,
            M=m,
            K=k,
            N=n,
            CHANNELS_PER_SCALE=channels,
            IS_PER_TENSOR=is_per_tensor,
            BLOCK_M=args.block_m,
            BLOCK_N=args.block_n,
            BLOCK_K=args.block_k,
            num_warps=args.num_warps,
            num_stages=4,
        )

    fn()
    torch.cuda.synchronize()
    max_abs, mean_abs, max_rel = error_stats(y, full_reference(x, w, scales, channels, is_per_tensor))
    median, q20, q80, stdev = measure_ms(fn, args.warmup, args.repeat, args.rounds)
    return base_row(args, "full_gemm", shape_name, m, k, n, 0, 0, gran, channels, num_scales, median, q20, q80, stdev, max_abs, mean_abs, max_rel, "i8_full_gemm_output_scale")


def run_epilogue_case(args, shape_name: str, m: int, k: int, n: int, granularity: str) -> dict:
    gran, channels, num_scales, is_per_tensor = parse_granularity(granularity, n)
    acc = torch.randint(-65536, 65536, (m, n), device="cuda", dtype=torch.int32)
    scales = make_scale(num_scales)
    y = torch.empty((m, n), device="cuda", dtype=torch.float16)
    grid = (triton.cdiv(m, args.block_m), triton.cdiv(n, args.block_n))

    def fn():
        _epilogue_only_output_scale_kernel[grid](
            acc,
            scales,
            y,
            M=m,
            N=n,
            CHANNELS_PER_SCALE=channels,
            IS_PER_TENSOR=is_per_tensor,
            BLOCK_M=args.block_m,
            BLOCK_N=args.block_n,
            num_warps=args.num_warps,
            num_stages=4,
        )

    fn()
    torch.cuda.synchronize()
    max_abs, mean_abs, max_rel = error_stats(y, epilogue_reference(acc, scales, channels, is_per_tensor))
    median, q20, q80, stdev = measure_ms(fn, args.warmup, args.repeat, args.rounds)
    return base_row(args, "epilogue_only", shape_name, m, k, n, 0, 0, gran, channels, num_scales, median, q20, q80, stdev, max_abs, mean_abs, max_rel, "i32_epilogue_output_scale")


def run_grouped_case(args, shape_name: str, tokens: int, active_experts: int, k: int, n: int, granularity: str) -> dict:
    gran, channels, num_scales, is_per_tensor = parse_granularity(granularity, n)
    x = torch.randint(-8, 8, (active_experts, tokens, k), device="cuda", dtype=torch.int8)
    w = torch.randint(-8, 8, (active_experts, n, k), device="cuda", dtype=torch.int8)
    scales = make_scale(active_experts * num_scales).view(active_experts, num_scales)
    y = torch.empty((active_experts, tokens, n), device="cuda", dtype=torch.float16)
    grid = (triton.cdiv(tokens, args.block_m), triton.cdiv(n, args.block_n), active_experts)

    def fn():
        _grouped_moe_output_scale_kernel[grid](
            x,
            w,
            scales,
            y,
            E=active_experts,
            T=tokens,
            K=k,
            N=n,
            NUM_SCALES=num_scales,
            CHANNELS_PER_SCALE=channels,
            IS_PER_TENSOR=is_per_tensor,
            BLOCK_M=args.block_m,
            BLOCK_N=args.block_n,
            BLOCK_K=args.block_k,
            num_warps=args.num_warps,
            num_stages=4,
        )

    fn()
    torch.cuda.synchronize()
    max_abs, mean_abs, max_rel = error_stats(y, grouped_reference(x, w, scales, channels, is_per_tensor))
    median, q20, q80, stdev = measure_ms(fn, args.warmup, args.repeat, args.rounds)
    return base_row(args, "grouped_moe", shape_name, tokens, k, n, active_experts, tokens, gran, channels, num_scales, median, q20, q80, stdev, max_abs, mean_abs, max_rel, "i8_grouped_moe_output_scale")


def base_row(args, scope, shape_name, m, k, n, experts, tokens, gran, channels, num_scales, median, q20, q80, stdev, max_abs, mean_abs, max_rel, kernel_name) -> dict:
    return {
        "scope": scope,
        "shape_name": shape_name,
        "M": m,
        "K": k,
        "N": n,
        "num_active_experts": experts,
        "tokens_per_expert": tokens,
        "granularity": gran,
        "channels_per_scale": channels,
        "num_scales": num_scales,
        "scale_reduction": f"{scale_reduction_for_granularity(num_scales, n):.6f}",
        "latency_ms": f"{median:.6f}",
        "latency_q20_ms": f"{q20:.6f}",
        "latency_q80_ms": f"{q80:.6f}",
        "latency_std_ms": f"{stdev:.6f}",
        "speedup_over_per_channel": "",
        "max_abs_error": f"{max_abs:.6e}",
        "mean_abs_error": f"{mean_abs:.6e}",
        "max_rel_error": f"{max_rel:.6e}",
        "kernel_name": kernel_name,
        "block_m": args.block_m,
        "block_n": args.block_n,
        "block_k": args.block_k,
        "num_warps": args.num_warps,
        "dtype_x": "int8",
        "dtype_w": "int8",
        "dtype_scale": "float32",
    }


def add_speedups(rows: List[dict]) -> None:
    groups: Dict[Tuple, float] = {}
    for row in rows:
        key = (
            row["scope"],
            row["shape_name"],
            row["M"],
            row["K"],
            row["N"],
            row["num_active_experts"],
            row["tokens_per_expert"],
        )
        if row["granularity"] == "per_channel":
            groups[key] = float(row["latency_ms"])
    for row in rows:
        key = (
            row["scope"],
            row["shape_name"],
            row["M"],
            row["K"],
            row["N"],
            row["num_active_experts"],
            row["tokens_per_expert"],
        )
        baseline = groups.get(key)
        row["speedup_over_per_channel"] = f"{baseline / float(row['latency_ms']):.6f}" if baseline else ""


DETAIL_FIELDS = [
    "scope",
    "shape_name",
    "M",
    "K",
    "N",
    "num_active_experts",
    "tokens_per_expert",
    "granularity",
    "channels_per_scale",
    "num_scales",
    "scale_reduction",
    "latency_ms",
    "latency_q20_ms",
    "latency_q80_ms",
    "latency_std_ms",
    "speedup_over_per_channel",
    "max_abs_error",
    "mean_abs_error",
    "max_rel_error",
    "kernel_name",
    "block_m",
    "block_n",
    "block_k",
    "num_warps",
    "dtype_x",
    "dtype_w",
    "dtype_scale",
]


def write_report(output_dir: str, rows: List[dict]) -> None:
    report_path = os.path.join(output_dir, "report.md")
    with open(report_path, "w") as f:
        f.write("# Output-Channel Scale Sharing Benchmark\n\n")
        f.write("This benchmark keeps the integer GEMM body identical across granularities. ")
        f.write("Only the output-scale epilogue is specialized.\n\n")
        f.write("## Interpretation\n\n")
        f.write("- `full_gemm` is the relevant GEMM speedup measurement.\n")
        f.write("- `epilogue_only` isolates scale indexing/loading/apply cost and must not be described as full GEMM speedup.\n")
        f.write("- `grouped_moe` measures multiple active experts with identical launch count across granularities.\n")
        f.write("- Scale-count reduction is metadata reduction, not runtime speedup.\n\n")
        f.write("If `full_gemm` speedup is small, the expected reason is that dot-product work and weight traffic dominate while output-scale epilogue is a small fraction of total time.\n")


def run(args: argparse.Namespace) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    torch.manual_seed(args.seed)
    torch.cuda.set_device(args.device)
    os.makedirs(args.output_dir, exist_ok=True)

    shapes = parse_shapes(args.shapes)
    granularities = parse_str_list(args.granularities)
    m_values = parse_int_list(args.m_values)
    active_values = parse_int_list(args.num_active_experts)
    token_values = parse_int_list(args.tokens_per_expert)
    scopes = set(parse_str_list(args.scopes))

    print(f"[output-scale-bench] output_dir={args.output_dir}")
    print(f"[output-scale-bench] device={torch.cuda.get_device_name(args.device)}")
    print(f"[output-scale-bench] scopes={sorted(scopes)}")
    print(f"[output-scale-bench] shapes={shapes}")
    print(f"[output-scale-bench] granularities={granularities}")
    print(f"[output-scale-bench] warmup={args.warmup} repeat={args.repeat} rounds={args.rounds}")

    rows: List[dict] = []
    for shape_name, k, n in shapes:
        if "full_gemm" in scopes:
            for m in m_values:
                for gran in granularities:
                    rows.append(run_full_gemm_case(args, shape_name, m, k, n, gran))
                    print(f"[output-scale-bench] full_gemm {shape_name} M={m} gran={gran} done")
        if "epilogue_only" in scopes:
            for m in m_values:
                for gran in granularities:
                    rows.append(run_epilogue_case(args, shape_name, m, k, n, gran))
                    print(f"[output-scale-bench] epilogue_only {shape_name} M={m} gran={gran} done")
        if "grouped_moe" in scopes:
            for active in active_values:
                for tokens in token_values:
                    for gran in granularities:
                        rows.append(run_grouped_case(args, shape_name, tokens, active, k, n, gran))
                        print(f"[output-scale-bench] grouped_moe {shape_name} E={active} T={tokens} gran={gran} done")

    add_speedups(rows)
    detail_path = os.path.join(args.output_dir, "output_scale_gemm_detail.csv")
    write_csv(detail_path, rows, DETAIL_FIELDS)
    write_csv(
        os.path.join(args.output_dir, "olmoe_scale_metadata_summary.csv"),
        expert_scale_summary_rows(),
        ["granularity", "channels_per_scale", "scale_count_per_expert", "scale_reduction_vs_per_channel", "fp16_scale_bytes_per_expert"],
    )
    write_report(args.output_dir, rows)
    print(f"[output-scale-bench] detail: {detail_path}")
    print(f"[output-scale-bench] metadata summary: {os.path.join(args.output_dir, 'olmoe_scale_metadata_summary.csv')}")
    print(f"[output-scale-bench] report: {os.path.join(args.output_dir, 'report.md')}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Fair output-channel scale-sharing GEMM/MoE benchmark.")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--scopes", default="full_gemm,epilogue_only,grouped_moe")
    parser.add_argument("--shapes", default="olmoe_gate_up,olmoe_down")
    parser.add_argument("--m_values", default="1,2,4,8,16,32,64,128")
    parser.add_argument("--num_active_experts", default="2,4,8,16")
    parser.add_argument("--tokens_per_expert", default="1,2,4,8")
    parser.add_argument("--granularities", default="per_channel,2,4,8,16,per_tensor")
    parser.add_argument("--warmup", type=int, default=200)
    parser.add_argument("--repeat", type=int, default=1000)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--block_m", type=int, default=16)
    parser.add_argument("--block_n", type=int, default=64)
    parser.add_argument("--block_k", type=int, default=64)
    parser.add_argument("--num_warps", type=int, default=4)
    parser.add_argument("--seed", type=int, default=2)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
