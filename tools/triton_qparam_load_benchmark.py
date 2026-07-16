#!/usr/bin/env python3
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
    "olmoe": MoePreset(
        name="olmoe",
        layers=16,
        experts=64,
        top_k=8,
        out_dims=(1024, 1024, 2048),  # gate, up, down
    ),
    "qwen": MoePreset(
        name="qwen",
        layers=24,
        experts=60,
        top_k=4,
        out_dims=(1408, 1408, 2048),  # gate, up, down
    ),
}


@triton.jit
def _qparam_stream_kernel(
    scale_ptr,
    zero_ptr,
    out_ptr,
    N: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N

    scale = tl.load(
        scale_ptr + offs,
        mask=mask,
        other=0.0,
        eviction_policy="evict_first",
    ).to(tl.float32)
    zero = tl.load(
        zero_ptr + offs,
        mask=mask,
        other=0.0,
        eviction_policy="evict_first",
    ).to(tl.float32)

    # Reduce to one scalar per block so the benchmark is dominated by qparam reads,
    # not by writing an expanded scale tensor.
    acc = tl.sum(scale + zero * 0.00390625, axis=0)
    tl.store(out_ptr + pid, acc)


def parse_int_list(value: str) -> List[int]:
    out = []
    for item in value.split(","):
        item = item.strip()
        if item:
            out.append(int(item))
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

    per_expert_groups = sum(math.ceil(dim / group_size) for dim in preset.out_dims)
    return preset.layers * active_experts * per_expert_groups


def measure_ms(fn, warmup: int, repeat: int, trials: int) -> Tuple[float, float]:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    values = []
    for _ in range(trials):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(repeat):
            fn()
        end.record()
        torch.cuda.synchronize()
        values.append(start.elapsed_time(end) / repeat)
    return statistics.median(values), min(values)


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


def run_case(
    total_groups: int,
    block_size: int,
    warmup: int,
    repeat: int,
    trials: int,
) -> Tuple[float, float]:
    scale = torch.rand((total_groups,), device="cuda", dtype=torch.float16)
    zero = torch.rand((total_groups,), device="cuda", dtype=torch.float16)
    out = torch.empty((triton.cdiv(total_groups, block_size),), device="cuda", dtype=torch.float32)

    grid = (triton.cdiv(total_groups, block_size),)
    fn = lambda: _qparam_stream_kernel[grid](
        scale,
        zero,
        out,
        N=total_groups,
        BLOCK=block_size,
        num_warps=8,
        num_stages=4,
    )
    return measure_ms(fn, warmup, repeat, trials)


def run(args: argparse.Namespace) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this benchmark")
    torch.cuda.set_device(args.device)
    torch.manual_seed(args.seed)

    presets = []
    for name in args.models.split(","):
        name = name.strip()
        if not name:
            continue
        if name not in PRESETS:
            raise ValueError(f"unknown model preset {name!r}; choices={sorted(PRESETS)}")
        presets.append(PRESETS[name])
    if not presets:
        raise ValueError("no model preset selected")

    group_sizes = parse_int_list(args.group_sizes)
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"[qparam-load-bench] output_dir={args.output_dir}")
    print(f"[qparam-load-bench] device={args.device} {torch.cuda.get_device_name(args.device)}")
    print(f"[qparam-load-bench] models={[p.name for p in presets]}")
    print(f"[qparam-load-bench] scope={args.scope}")
    print(f"[qparam-load-bench] group_sizes={group_sizes}")
    print(f"[qparam-load-bench] virtual_repeats={args.virtual_repeats}")
    print("[qparam-load-bench] operator=stream scale/zero qparams and reduce per block")

    rows = []
    for preset in presets:
        case_rows = []
        for group_size in group_sizes:
            base_groups = qparam_groups_for_preset(preset, group_size, args.scope)
            total_groups = base_groups * args.virtual_repeats
            qparam_bytes = total_groups * args.qparam_bytes_per_group
            median_ms, best_ms = run_case(
                total_groups=total_groups,
                block_size=args.block_size,
                warmup=args.warmup,
                repeat=args.repeat,
                trials=args.trials,
            )
            gib_per_s = (qparam_bytes / (1024.0**3)) / max(median_ms * 1e-3, 1e-12)
            row = {
                "model": preset.name,
                "scope": args.scope,
                "group_size": group_size,
                "base_qparam_groups": base_groups,
                "virtual_repeats": args.virtual_repeats,
                "total_qparam_groups": total_groups,
                "qparam_bytes": qparam_bytes,
                "qparam_kib": f"{qparam_bytes / 1024.0:.6f}",
                "median_ms": f"{median_ms:.6f}",
                "best_ms": f"{best_ms:.6f}",
                "qparam_gib_per_s": f"{gib_per_s:.6f}",
            }
            rows.append(row)
            case_rows.append(row)
            print(
                f"[qparam-load-bench] {preset.name} scope={args.scope} group={group_size} "
                f"groups={total_groups} bytes={qparam_bytes} median_ms={median_ms:.6f}"
            )

        baseline_groups = next(int(row["total_qparam_groups"]) for row in case_rows if int(row["group_size"]) == group_sizes[0])
        baseline_bytes = next(int(row["qparam_bytes"]) for row in case_rows if int(row["group_size"]) == group_sizes[0])
        baseline_ms = next(float(row["median_ms"]) for row in case_rows if int(row["group_size"]) == group_sizes[0])
        for row in case_rows:
            row["qparam_group_reduction_vs_first_group"] = f"{baseline_groups / int(row['total_qparam_groups']):.6f}"
            row["qparam_byte_reduction_vs_first_group"] = f"{baseline_bytes / int(row['qparam_bytes']):.6f}"
            row["operator_speedup_vs_first_group"] = f"{baseline_ms / float(row['median_ms']):.6f}"

    detail_fields = [
        "model",
        "scope",
        "group_size",
        "base_qparam_groups",
        "virtual_repeats",
        "total_qparam_groups",
        "qparam_bytes",
        "qparam_kib",
        "median_ms",
        "best_ms",
        "qparam_gib_per_s",
        "qparam_group_reduction_vs_first_group",
        "qparam_byte_reduction_vs_first_group",
        "operator_speedup_vs_first_group",
    ]
    detail_path = os.path.join(args.output_dir, "detail.csv")
    write_csv(detail_path, rows, detail_fields)

    summary_rows = []
    for group_size in group_sizes:
        group_rows = [row for row in rows if int(row["group_size"]) == group_size]
        summary_rows.append(
            {
                "group_size": group_size,
                "mean_operator_speedup_vs_group1": f"{statistics.mean(float(row['operator_speedup_vs_first_group']) for row in group_rows):.6f}",
                "gmean_operator_speedup_vs_group1": f"{gmean([float(row['operator_speedup_vs_first_group']) for row in group_rows]):.6f}",
                "mean_qparam_byte_reduction_vs_group1": f"{statistics.mean(float(row['qparam_byte_reduction_vs_first_group']) for row in group_rows):.6f}",
                "gmean_qparam_byte_reduction_vs_group1": f"{gmean([float(row['qparam_byte_reduction_vs_first_group']) for row in group_rows]):.6f}",
                "num_cases": len(group_rows),
            }
        )
    summary_fields = [
        "group_size",
        "mean_operator_speedup_vs_group1",
        "gmean_operator_speedup_vs_group1",
        "mean_qparam_byte_reduction_vs_group1",
        "gmean_qparam_byte_reduction_vs_group1",
        "num_cases",
    ]
    summary_path = os.path.join(args.output_dir, "summary_by_group.csv")
    write_csv(summary_path, summary_rows, summary_fields)

    print(f"[qparam-load-bench] detail: {detail_path}")
    print(f"[qparam-load-bench] summary: {summary_path}")
    print("[qparam-load-bench] plot operator_speedup for qparam-load speed, byte_reduction for metadata traffic")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark a Triton qparam stream operator that reads scale/zero metadata "
            "for MoE expert projections under different output-channel group sizes."
        )
    )
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--models", default="olmoe", help="comma-separated presets: olmoe,qwen")
    parser.add_argument("--scope", default="all_experts", choices=["all_experts", "routed_token"])
    parser.add_argument("--group_sizes", default="1,16,32,64,128,256,512,1024,2048")
    parser.add_argument(
        "--virtual_repeats",
        type=int,
        default=1,
        help="repeat the logical qparam stream to make very small metadata cases measurable",
    )
    parser.add_argument("--qparam_bytes_per_group", type=int, default=4)
    parser.add_argument("--block_size", type=int, default=256)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--repeat", type=int, default=200)
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--seed", type=int, default=2)
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
