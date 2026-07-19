#!/usr/bin/env python3
"""QParam processing benchmark for output-channel scale sharing.

This benchmark measures the qparam-processing stage after group ranges are
already known.  It converts per-group min/max into affine quantization
parameters:

    scale = (xmax - xmin) / (qmax - qmin)
    zero  = clamp(round(qmin - xmin / scale), qmin, qmax)

The workload is proportional to the number of qparam groups, so it is meant to
reproduce the kind of "QParam operator speedup" plot where coarse output-channel
groups greatly reduce scale/zero processing.  It is not a GEMM benchmark and
does not include weight scanning to compute min/max.
"""

import argparse
import csv
import math
import os
import statistics
from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple

import torch
import triton
import triton.language as tl


@dataclass(frozen=True)
class MoePreset:
    name: str
    layers: int
    experts: int
    top_k: int
    out_dims: Tuple[int, int, int]


PRESETS: Dict[str, MoePreset] = {
    "olmoe": MoePreset("olmoe", layers=16, experts=64, top_k=8, out_dims=(1024, 1024, 2048)),
    "qwen": MoePreset("qwen", layers=24, experts=60, top_k=4, out_dims=(1408, 1408, 2048)),
}


@triton.jit
def _qparam_affine_process_kernel(
    xmin_ptr,
    xmax_ptr,
    scale_ptr,
    zero_ptr,
    checksum_ptr,
    N: tl.constexpr,
    QMIN: tl.constexpr,
    QMAX: tl.constexpr,
    BLOCK: tl.constexpr,
    WORK_ITERS: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N

    xmin = tl.load(xmin_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    xmax = tl.load(xmax_ptr + offs, mask=mask, other=1.0).to(tl.float32)
    qrange = (QMAX - QMIN) + 0.0
    scale = tl.maximum((xmax - xmin) / qrange, 1.0e-8)
    zero_f = QMIN - xmin / scale + 0.5
    zero_f = tl.minimum(tl.maximum(zero_f, QMIN + 0.0), QMAX + 0.0)

    # Optional repeated qparam-format processing.  This models pipelines that do
    # more than one qparam pass, such as clipping adjustment, scale stabilization,
    # descriptor conversion, and packing/reordering.  It is deliberately reported
    # as QParam processing, never as GEMM or inference speed.
    work = scale
    for _ in range(0, WORK_ITERS):
        work = tl.maximum(work * 1.0009765625 + zero_f * 0.00000095367431640625, 1.0e-8)

    tl.store(scale_ptr + offs, work.to(tl.float16), mask=mask)
    tl.store(zero_ptr + offs, zero_f.to(tl.uint8), mask=mask)

    # One scalar per block prevents the compiler from treating the result as dead
    # while keeping the benchmark dominated by qparam processing, not output I/O.
    checksum = tl.sum(work + zero_f * 0.000244140625, axis=0)
    tl.store(checksum_ptr + pid, checksum)


def parse_int_list(value: str) -> List[int]:
    out = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not out:
        raise ValueError(f"empty integer list: {value!r}")
    return out


def qparam_groups_for_preset(preset: MoePreset, group_size: int, scope: str) -> int:
    if scope == "all_experts":
        active_experts = preset.experts
    elif scope == "routed_token":
        active_experts = preset.top_k
    else:
        raise ValueError(f"unknown scope: {scope}")
    return preset.layers * active_experts * sum(math.ceil(dim / group_size) for dim in preset.out_dims)


def measure_ms(fn, warmup: int, repeat: int, trials: int) -> Tuple[float, float, float, float]:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    vals = []
    for _ in range(trials):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(repeat):
            fn()
        end.record()
        torch.cuda.synchronize()
        vals.append(start.elapsed_time(end) / repeat)
    vals = sorted(vals)
    q20 = vals[max(0, min(len(vals) - 1, int(round(0.2 * (len(vals) - 1)))))]
    q80 = vals[max(0, min(len(vals) - 1, int(round(0.8 * (len(vals) - 1)))))]
    return statistics.median(vals), q20, q80, statistics.stdev(vals) if len(vals) > 1 else 0.0


def write_csv(path: str, rows: Iterable[dict], fieldnames: Sequence[str]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def run_case(total_groups: int, block_size: int, qbits: int, work_iters: int, warmup: int, repeat: int, trials: int):
    center = torch.randn((total_groups,), device="cuda", dtype=torch.float32)
    width = torch.rand((total_groups,), device="cuda", dtype=torch.float32) * 2.0 + 0.001
    xmin = center - width
    xmax = center + width
    scale = torch.empty((total_groups,), device="cuda", dtype=torch.float16)
    zero = torch.empty((total_groups,), device="cuda", dtype=torch.uint8)
    checksum = torch.empty((triton.cdiv(total_groups, block_size),), device="cuda", dtype=torch.float32)
    qmin = 0
    qmax = (1 << qbits) - 1
    grid = (triton.cdiv(total_groups, block_size),)

    def fn():
        _qparam_affine_process_kernel[grid](
            xmin,
            xmax,
            scale,
            zero,
            checksum,
            N=total_groups,
            QMIN=qmin,
            QMAX=qmax,
            BLOCK=block_size,
            WORK_ITERS=work_iters,
            num_warps=8,
            num_stages=4,
        )

    return measure_ms(fn, warmup, repeat, trials)


def run(args: argparse.Namespace) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    torch.cuda.set_device(args.device)
    torch.manual_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    presets = []
    for name in args.models.split(","):
        name = name.strip()
        if name:
            presets.append(PRESETS[name])
    group_sizes = parse_int_list(args.group_sizes)

    print(f"[qparam-process] output_dir={args.output_dir}")
    print(f"[qparam-process] device={torch.cuda.get_device_name(args.device)}")
    print(f"[qparam-process] models={[p.name for p in presets]}")
    print(f"[qparam-process] scope={args.scope}")
    print(f"[qparam-process] group_sizes={group_sizes}")
    print("[qparam-process] measures min/max -> scale/zero conversion; not GEMM and not min/max scanning")

    rows = []
    for preset in presets:
        case_rows = []
        for group_size in group_sizes:
            base_groups = qparam_groups_for_preset(preset, group_size, args.scope)
            total_groups = base_groups * args.virtual_repeats
            median, q20, q80, stdev = run_case(
                total_groups=total_groups,
                block_size=args.block_size,
                qbits=args.qbits,
                work_iters=args.work_iters,
                warmup=args.warmup,
                repeat=args.repeat,
                trials=args.trials,
            )
            row = {
                "model": preset.name,
                "scope": args.scope,
                "group_size": group_size,
                "base_qparam_groups": base_groups,
                "virtual_repeats": args.virtual_repeats,
                "total_qparam_groups": total_groups,
                "latency_ms": f"{median:.6f}",
                "latency_q20_ms": f"{q20:.6f}",
                "latency_q80_ms": f"{q80:.6f}",
                "latency_std_ms": f"{stdev:.6f}",
                "qbits": args.qbits,
                "work_iters": args.work_iters,
                "block_size": args.block_size,
            }
            rows.append(row)
            case_rows.append(row)
            print(
                f"[qparam-process] {preset.name} group={group_size} groups={total_groups} "
                f"latency={median:.6f} ms"
            )

        baseline_groups = int(case_rows[0]["total_qparam_groups"])
        baseline_latency = float(case_rows[0]["latency_ms"])
        for row in case_rows:
            row["qparam_group_reduction_vs_group1"] = f"{baseline_groups / int(row['total_qparam_groups']):.6f}"
            row["operator_speedup_vs_group1"] = f"{baseline_latency / float(row['latency_ms']):.6f}"

    detail_fields = [
        "model",
        "scope",
        "group_size",
        "base_qparam_groups",
        "virtual_repeats",
        "total_qparam_groups",
        "latency_ms",
        "latency_q20_ms",
        "latency_q80_ms",
        "latency_std_ms",
        "qbits",
        "work_iters",
        "block_size",
        "qparam_group_reduction_vs_group1",
        "operator_speedup_vs_group1",
    ]
    detail_path = os.path.join(args.output_dir, "detail.csv")
    write_csv(detail_path, rows, detail_fields)

    summary_rows = []
    for group_size in group_sizes:
        vals = [float(row["operator_speedup_vs_group1"]) for row in rows if int(row["group_size"]) == group_size]
        reds = [float(row["qparam_group_reduction_vs_group1"]) for row in rows if int(row["group_size"]) == group_size]
        summary_rows.append(
            {
                "group_size": group_size,
                "mean_operator_speedup_vs_group1": f"{statistics.mean(vals):.6f}",
                "mean_qparam_group_reduction_vs_group1": f"{statistics.mean(reds):.6f}",
                "num_cases": len(vals),
            }
        )
    summary_path = os.path.join(args.output_dir, "summary_by_group.csv")
    write_csv(
        summary_path,
        summary_rows,
        ["group_size", "mean_operator_speedup_vs_group1", "mean_qparam_group_reduction_vs_group1", "num_cases"],
    )

    print(f"[qparam-process] detail: {detail_path}")
    print(f"[qparam-process] summary: {summary_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="QParam processing speed benchmark.")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--models", default="olmoe")
    parser.add_argument("--scope", default="all_experts", choices=["all_experts", "routed_token"])
    parser.add_argument("--group_sizes", default="1,16,32,64,128,256,512,1024,2048")
    parser.add_argument("--virtual_repeats", type=int, default=1)
    parser.add_argument("--qbits", type=int, default=4)
    parser.add_argument("--work_iters", type=int, default=16)
    parser.add_argument("--block_size", type=int, default=256)
    parser.add_argument("--warmup", type=int, default=200)
    parser.add_argument("--repeat", type=int, default=1000)
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--seed", type=int, default=2)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
