#!/usr/bin/env python3
import argparse
import csv
import math
import os
import statistics
from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple

import torch
import triton
import triton.language as tl


PRESET_SHAPES = {
    # name: (out_features, in_features)
    "olmoe_gate_up": (1024, 2048),
    "olmoe_down": (2048, 1024),
    "qwen_gate_up": (1408, 2048),
    "qwen_down": (2048, 1408),
}


@triton.jit
def _int4_dequant_gemm_kernel(
    x_ptr,
    wq_ptr,
    scale_ptr,
    zero_ptr,
    y_ptr,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    K_PACKED: tl.constexpr,
    GROUP_N: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    acc = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
    scale_ids = offs_n // GROUP_N
    scales = tl.load(scale_ptr + scale_ids, mask=offs_n < N, other=1.0).to(tl.float32)
    zeros = tl.load(zero_ptr + scale_ids, mask=offs_n < N, other=0).to(tl.float32)

    for k0 in range(0, K, BLOCK_K):
        k = k0 + offs_k
        x = tl.load(
            x_ptr + offs_m[:, None] * K + k[None, :],
            mask=(offs_m[:, None] < M) & (k[None, :] < K),
            other=0.0,
        )

        packed_k = k // 2
        w_bytes = tl.load(
            wq_ptr + offs_n[:, None] * K_PACKED + packed_k[None, :],
            mask=(offs_n[:, None] < N) & (k[None, :] < K),
            other=0,
        ).to(tl.int32)

        low = w_bytes & 15
        high = (w_bytes >> 4) & 15
        is_odd = (k & 1) == 1
        w_int = tl.where(is_odd[None, :], high, low).to(tl.float32)
        w = ((w_int - zeros[:, None]) * scales[:, None]).to(tl.float16)

        acc += tl.dot(x, tl.trans(w))

    tl.store(
        y_ptr + offs_m[:, None] * N + offs_n[None, :],
        acc.to(tl.float16),
        mask=(offs_m[:, None] < M) & (offs_n[None, :] < N),
    )


@dataclass(frozen=True)
class BenchCase:
    shape_name: str
    m: int
    n: int
    k: int
    group_size: int


def parse_int_list(value: str) -> List[int]:
    values = []
    for item in value.split(","):
        item = item.strip()
        if item:
            values.append(int(item))
    if not values:
        raise ValueError(f"empty integer list: {value!r}")
    return values


def parse_shapes(value: str) -> List[Tuple[str, int, int]]:
    shapes = []
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        if item in PRESET_SHAPES:
            n, k = PRESET_SHAPES[item]
            shapes.append((item, n, k))
            continue
        parts = item.split(":")
        if len(parts) != 3:
            valid = ",".join(sorted(PRESET_SHAPES))
            raise ValueError(
                f"invalid shape {item!r}; use a preset ({valid}) or name:N:K"
            )
        name, n_str, k_str = parts
        shapes.append((name, int(n_str), int(k_str)))
    if not shapes:
        raise ValueError(f"empty shape list: {value!r}")
    return shapes


def pack_int4(weight_int: torch.Tensor) -> torch.Tensor:
    if weight_int.dtype != torch.uint8:
        raise TypeError("weight_int must be torch.uint8")
    if weight_int.dim() != 2:
        raise ValueError("weight_int must be 2D [N, K]")
    n, k = weight_int.shape
    if k % 2:
        pad = torch.zeros((n, 1), dtype=torch.uint8, device=weight_int.device)
        weight_int = torch.cat([weight_int, pad], dim=1)
    even = weight_int[:, 0::2]
    odd = weight_int[:, 1::2]
    return (even | (odd << 4)).contiguous()


def launch_kernel(
    x: torch.Tensor,
    wq: torch.Tensor,
    scales: torch.Tensor,
    zeros: torch.Tensor,
    y: torch.Tensor,
    group_size: int,
    block_m: int,
    block_n: int,
    block_k: int,
    num_warps: int,
):
    m, k = x.shape
    n = y.shape[1]
    k_packed = wq.shape[1]
    grid = (triton.cdiv(m, block_m), triton.cdiv(n, block_n))
    _int4_dequant_gemm_kernel[grid](
        x,
        wq,
        scales,
        zeros,
        y,
        M=m,
        N=n,
        K=k,
        K_PACKED=k_packed,
        GROUP_N=group_size,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        num_warps=num_warps,
        num_stages=4,
    )


def measure_ms(fn, warmup: int, repeat: int, trials: int) -> Tuple[float, float]:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    trial_ms = []
    for _ in range(trials):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(repeat):
            fn()
        end.record()
        torch.cuda.synchronize()
        trial_ms.append(start.elapsed_time(end) / repeat)
    return statistics.median(trial_ms), min(trial_ms)


def gmean(values: Sequence[float]) -> float:
    vals = [max(float(v), 1e-12) for v in values]
    return math.exp(sum(math.log(v) for v in vals) / len(vals))


def write_csv(path: str, rows: Iterable[dict], fieldnames: Sequence[str]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def run(args: argparse.Namespace) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the Triton benchmark")
    torch.manual_seed(args.seed)
    torch.cuda.set_device(args.device)

    shapes = parse_shapes(args.shapes)
    m_values = parse_int_list(args.m_values)
    group_sizes = parse_int_list(args.group_sizes)
    os.makedirs(args.output_dir, exist_ok=True)

    rows = []
    print(f"[group-scale-bench] output_dir={args.output_dir}")
    print(f"[group-scale-bench] device={args.device} {torch.cuda.get_device_name(args.device)}")
    print(f"[group-scale-bench] shapes={shapes}")
    print(f"[group-scale-bench] m_values={m_values}")
    print(f"[group-scale-bench] group_sizes={group_sizes}")
    print(
        "[group-scale-bench] kernel=int4 packed weight + per-output-group scale/zero "
        "+ fused dequant GEMM"
    )

    for shape_name, n, k in shapes:
        if k % 2:
            raise ValueError(f"K must be even for int4 packing, got {shape_name} K={k}")
        w_int = torch.randint(0, 16, (n, k), device="cuda", dtype=torch.uint8)
        wq = pack_int4(w_int)
        for m in m_values:
            x = torch.randn((m, k), device="cuda", dtype=torch.float16)
            y = torch.empty((m, n), device="cuda", dtype=torch.float16)
            case_rows = []

            for group_size in group_sizes:
                effective_group = min(group_size, n)
                num_scale_groups = triton.cdiv(n, effective_group)
                scales = (
                    torch.rand((num_scale_groups,), device="cuda", dtype=torch.float16)
                    * args.scale_range
                    + args.scale_min
                )
                zeros = torch.full((num_scale_groups,), args.zero_point, device="cuda", dtype=torch.uint8)

                fn = lambda: launch_kernel(
                    x,
                    wq,
                    scales,
                    zeros,
                    y,
                    effective_group,
                    args.block_m,
                    args.block_n,
                    args.block_k,
                    args.num_warps,
                )

                median_ms, best_ms = measure_ms(fn, args.warmup, args.repeat, args.trials)
                tflops = (2.0 * m * n * k) / (median_ms * 1e-3) / 1e12
                row = {
                    "shape_name": shape_name,
                    "M": m,
                    "N": n,
                    "K": k,
                    "group_size": group_size,
                    "effective_group_size": effective_group,
                    "num_scale_groups": num_scale_groups,
                    "median_ms": f"{median_ms:.6f}",
                    "best_ms": f"{best_ms:.6f}",
                    "tflops": f"{tflops:.6f}",
                }
                rows.append(row)
                case_rows.append(row)
                print(
                    f"[group-scale-bench] {shape_name} M={m} N={n} K={k} "
                    f"group={group_size} median_ms={median_ms:.6f} tflops={tflops:.3f}"
                )

            baseline = next(
                float(row["median_ms"]) for row in case_rows if int(row["group_size"]) == group_sizes[0]
            )
            for row in case_rows:
                row["speedup_vs_first_group"] = f"{baseline / float(row['median_ms']):.6f}"

    detail_fields = [
        "shape_name",
        "M",
        "N",
        "K",
        "group_size",
        "effective_group_size",
        "num_scale_groups",
        "median_ms",
        "best_ms",
        "tflops",
        "speedup_vs_first_group",
    ]
    detail_path = os.path.join(args.output_dir, "detail.csv")
    write_csv(detail_path, rows, detail_fields)

    summary_rows = []
    for group_size in group_sizes:
        group_rows = [row for row in rows if int(row["group_size"]) == group_size]
        latencies = [float(row["median_ms"]) for row in group_rows]
        speedups = [float(row["speedup_vs_first_group"]) for row in group_rows]
        summary_rows.append(
            {
                "group_size": group_size,
                "mean_median_ms": f"{statistics.mean(latencies):.6f}",
                "median_median_ms": f"{statistics.median(latencies):.6f}",
                "mean_speedup_vs_group1": f"{statistics.mean(speedups):.6f}",
                "gmean_speedup_vs_group1": f"{gmean(speedups):.6f}",
                "num_cases": len(group_rows),
            }
        )

    summary_fields = [
        "group_size",
        "mean_median_ms",
        "median_median_ms",
        "mean_speedup_vs_group1",
        "gmean_speedup_vs_group1",
        "num_cases",
    ]
    summary_path = os.path.join(args.output_dir, "summary_by_group.csv")
    write_csv(summary_path, summary_rows, summary_fields)

    print(f"[group-scale-bench] detail: {detail_path}")
    print(f"[group-scale-bench] summary: {summary_path}")
    print("[group-scale-bench] use gmean_speedup_vs_group1 or mean_speedup_vs_group1 for the bar plot")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark a Triton fused int4 dequant-GEMM kernel while varying the "
            "output-channel group size used by weight scales."
        )
    )
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument(
        "--shapes",
        default="olmoe_gate_up,olmoe_down",
        help=(
            "Comma-separated presets or name:N:K entries. Presets: "
            + ",".join(sorted(PRESET_SHAPES))
        ),
    )
    parser.add_argument("--m_values", default="1,4,8,16,32,64")
    parser.add_argument("--group_sizes", default="1,16,32,64,128,256,512,1024,2048")
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--repeat", type=int, default=200)
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--block_m", type=int, default=16)
    parser.add_argument("--block_n", type=int, default=64)
    parser.add_argument("--block_k", type=int, default=64)
    parser.add_argument("--num_warps", type=int, default=4)
    parser.add_argument("--scale_min", type=float, default=0.001)
    parser.add_argument("--scale_range", type=float, default=0.02)
    parser.add_argument("--zero_point", type=int, default=8)
    parser.add_argument("--seed", type=int, default=2)
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
