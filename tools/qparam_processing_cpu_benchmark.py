#!/usr/bin/env python3
"""CPU/NumPy QParam processing benchmark.

This reproduces the common "QParam processing speedup" effect: once min/max
ranges are available, converting group ranges into scale/zero metadata is
O(number of qparam groups).  It is useful for host-side preprocessing,
calibration bookkeeping, or qparam packing/reordering.  It is not a GEMM or
end-to-end inference benchmark.
"""

import argparse
import csv
import math
import os
import statistics
import time
from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np


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


def parse_int_list(value: str) -> List[int]:
    out = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not out:
        raise ValueError(f"empty integer list: {value!r}")
    return out


def qparam_groups_for_preset(preset: MoePreset, group_size: int, scope: str) -> int:
    active_experts = preset.experts if scope == "all_experts" else preset.top_k
    return preset.layers * active_experts * sum(math.ceil(dim / group_size) for dim in preset.out_dims)


def process_qparams(xmin: np.ndarray, xmax: np.ndarray, qbits: int, work_iters: int):
    qmin = 0.0
    qmax = float((1 << qbits) - 1)
    scale = np.maximum((xmax - xmin) / (qmax - qmin), 1.0e-8).astype(np.float32)
    zero = np.clip(np.rint(qmin - xmin / scale), qmin, qmax).astype(np.uint8)
    work = scale
    zf = zero.astype(np.float32)
    for _ in range(work_iters):
        work = np.maximum(work * np.float32(1.0009765625) + zf * np.float32(9.5367431640625e-7), np.float32(1.0e-8))
    return work.astype(np.float16), zero


def measure(fn, warmup: int, trials: int):
    for _ in range(warmup):
        fn()
    vals = []
    for _ in range(trials):
        t0 = time.perf_counter()
        fn()
        vals.append((time.perf_counter() - t0) * 1000.0)
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


def run(args):
    rng = np.random.default_rng(args.seed)
    presets = [PRESETS[name.strip()] for name in args.models.split(",") if name.strip()]
    group_sizes = parse_int_list(args.group_sizes)
    os.makedirs(args.output_dir, exist_ok=True)

    rows = []
    print(f"[qparam-process-cpu] output_dir={args.output_dir}")
    print(f"[qparam-process-cpu] models={[p.name for p in presets]} scope={args.scope}")
    print("[qparam-process-cpu] measures host-side min/max -> scale/zero conversion; not GEMM")

    for preset in presets:
        case_rows = []
        for group_size in group_sizes:
            base_groups = qparam_groups_for_preset(preset, group_size, args.scope)
            total_groups = base_groups * args.virtual_repeats
            center = rng.standard_normal(total_groups, dtype=np.float32)
            width = rng.random(total_groups, dtype=np.float32) * np.float32(2.0) + np.float32(0.001)
            xmin = center - width
            xmax = center + width
            fn = lambda: process_qparams(xmin, xmax, args.qbits, args.work_iters)
            median, q20, q80, stdev = measure(fn, args.warmup, args.trials)
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
            }
            rows.append(row)
            case_rows.append(row)
            print(f"[qparam-process-cpu] {preset.name} group={group_size} groups={total_groups} latency={median:.6f} ms")

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
        "qparam_group_reduction_vs_group1",
        "operator_speedup_vs_group1",
    ]
    write_csv(os.path.join(args.output_dir, "detail.csv"), rows, detail_fields)

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
    write_csv(
        os.path.join(args.output_dir, "summary_by_group.csv"),
        summary_rows,
        ["group_size", "mean_operator_speedup_vs_group1", "mean_qparam_group_reduction_vs_group1", "num_cases"],
    )
    print(f"[qparam-process-cpu] summary: {os.path.join(args.output_dir, 'summary_by_group.csv')}")


def main():
    parser = argparse.ArgumentParser(description="CPU qparam processing speed benchmark.")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--models", default="olmoe")
    parser.add_argument("--scope", default="all_experts", choices=["all_experts", "routed_token"])
    parser.add_argument("--group_sizes", default="1,16,32,64,128,256,512,1024,2048")
    parser.add_argument("--virtual_repeats", type=int, default=1)
    parser.add_argument("--qbits", type=int, default=4)
    parser.add_argument("--work_iters", type=int, default=16)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--trials", type=int, default=20)
    parser.add_argument("--seed", type=int, default=2)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
