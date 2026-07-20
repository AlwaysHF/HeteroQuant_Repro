#!/usr/bin/env python3
"""Structure-level MoE FFN benchmark: DuQuant gate/up transform vs OSE branch.

This benchmark does not use real model weights.  It uses random tensors with the
project's MoE FFN shapes and measures two inference structures with the same
Triton FP16 GEMM backend:

  DuQuant-like:
    x -> block rotation/permutation transform -> gate/up GEMMs -> silu*up -> down

  Ours/OSE:
    x -> gate/up main GEMMs
      -> top-k input-channel OSE two-projection branch added to gate/up
      -> silu*up -> down

The goal is to isolate whether removing the high-dimensional gate/up DuQuant
activation transform can offset the OSE branch overhead.  It is intentionally a
structure benchmark, not a packed INT4/INT8 deployment kernel.
"""

import argparse
import csv
import math
import os
import statistics
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import triton
import triton.language as tl


@dataclass(frozen=True)
class MoEPreset:
    hidden: int
    intermediate: int
    num_experts: int
    top_k: int


PRESETS: Dict[str, MoEPreset] = {
    "olmoe": MoEPreset(hidden=2048, intermediate=1024, num_experts=64, top_k=8),
    "qwen": MoEPreset(hidden=2048, intermediate=1408, num_experts=60, top_k=4),
}


@triton.jit
def _grouped_fp16_gemm_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    E: tl.constexpr,
    M: tl.constexpr,
    K: tl.constexpr,
    N: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_e = tl.program_id(2)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    acc = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)

    a_base = pid_e * M * K
    b_base = pid_e * K * N
    for k0 in range(0, K, BLOCK_K):
        k = k0 + offs_k
        a = tl.load(
            a_ptr + a_base + offs_m[:, None] * K + k[None, :],
            mask=(offs_m[:, None] < M) & (k[None, :] < K),
            other=0.0,
        )
        b = tl.load(
            b_ptr + b_base + k[:, None] * N + offs_n[None, :],
            mask=(k[:, None] < K) & (offs_n[None, :] < N),
            other=0.0,
        )
        acc += tl.dot(a, b, out_dtype=tl.float32)

    tl.store(
        c_ptr + pid_e * M * N + offs_m[:, None] * N + offs_n[None, :],
        acc.to(tl.float16),
        mask=(offs_m[:, None] < M) & (offs_n[None, :] < N),
    )


@triton.jit
def _duquant_block_rotate_kernel(
    x_ptr,
    r_ptr,
    y_ptr,
    E: tl.constexpr,
    M: tl.constexpr,
    H: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_b = tl.program_id(1)
    pid_e = tl.program_id(2)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_i = tl.arange(0, BLOCK_H)
    offs_o = tl.arange(0, BLOCK_H)
    h0 = pid_b * BLOCK_H

    x = tl.load(
        x_ptr + pid_e * M * H + offs_m[:, None] * H + h0 + offs_i[None, :],
        mask=(offs_m[:, None] < M) & (h0 + offs_i[None, :] < H),
        other=0.0,
    )
    r = tl.load(
        r_ptr + pid_e * BLOCK_H * BLOCK_H + offs_i[:, None] * BLOCK_H + offs_o[None, :],
        mask=(offs_i[:, None] < BLOCK_H) & (offs_o[None, :] < BLOCK_H),
        other=0.0,
    )
    out = tl.dot(x, r, out_dtype=tl.float32)
    tl.store(
        y_ptr + pid_e * M * H + offs_m[:, None] * H + h0 + offs_o[None, :],
        out.to(tl.float16),
        mask=(offs_m[:, None] < M) & (h0 + offs_o[None, :] < H),
    )


@triton.jit
def _permute_hidden_kernel(
    x_ptr,
    perm_ptr,
    y_ptr,
    E: tl.constexpr,
    M: tl.constexpr,
    H: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    total = E * M * H
    mask = offs < total
    h = offs % H
    mt = offs // H
    e = mt // M
    m = mt - e * M
    src_h = tl.load(perm_ptr + e * H + h, mask=mask, other=0)
    vals = tl.load(x_ptr + e * M * H + m * H + src_h, mask=mask, other=0.0)
    tl.store(y_ptr + offs, vals, mask=mask)


@triton.jit
def _silu_mul_kernel(
    gate_ptr,
    up_ptr,
    out_ptr,
    TOTAL: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < TOTAL
    gate = tl.load(gate_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    up = tl.load(up_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    silu = gate * tl.sigmoid(gate)
    tl.store(out_ptr + offs, (silu * up).to(tl.float16), mask=mask)


@triton.jit
def _ose_two_proj_kernel(
    x_ptr,
    gate_idx_ptr,
    up_idx_ptr,
    gate_w_ptr,
    up_w_ptr,
    gate_ptr,
    up_ptr,
    E: tl.constexpr,
    M: tl.constexpr,
    H: tl.constexpr,
    I: tl.constexpr,
    TOPK: tl.constexpr,
    ADD_TO_MAIN: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_e = tl.program_id(2)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    k_mask = offs_k < TOPK

    gate_idx = tl.load(gate_idx_ptr + pid_e * TOPK + offs_k, mask=k_mask, other=0)
    up_idx = tl.load(up_idx_ptr + pid_e * TOPK + offs_k, mask=k_mask, other=0)

    x_gate = tl.load(
        x_ptr + pid_e * M * H + offs_m[:, None] * H + gate_idx[None, :],
        mask=(offs_m[:, None] < M) & k_mask[None, :],
        other=0.0,
    )
    x_up = tl.load(
        x_ptr + pid_e * M * H + offs_m[:, None] * H + up_idx[None, :],
        mask=(offs_m[:, None] < M) & k_mask[None, :],
        other=0.0,
    )
    wg = tl.load(
        gate_w_ptr + pid_e * TOPK * I + offs_k[:, None] * I + offs_n[None, :],
        mask=k_mask[:, None] & (offs_n[None, :] < I),
        other=0.0,
    )
    wu = tl.load(
        up_w_ptr + pid_e * TOPK * I + offs_k[:, None] * I + offs_n[None, :],
        mask=k_mask[:, None] & (offs_n[None, :] < I),
        other=0.0,
    )
    gate_acc = tl.dot(x_gate, wg, out_dtype=tl.float32)
    up_acc = tl.dot(x_up, wu, out_dtype=tl.float32)

    out_offs = pid_e * M * I + offs_m[:, None] * I + offs_n[None, :]
    if ADD_TO_MAIN:
        gate_base = tl.load(
            gate_ptr + out_offs,
            mask=(offs_m[:, None] < M) & (offs_n[None, :] < I),
            other=0.0,
        ).to(tl.float32)
        up_base = tl.load(
            up_ptr + out_offs,
            mask=(offs_m[:, None] < M) & (offs_n[None, :] < I),
            other=0.0,
        ).to(tl.float32)
        gate_acc += gate_base
        up_acc += up_base

    mask = (offs_m[:, None] < M) & (offs_n[None, :] < I)
    tl.store(gate_ptr + out_offs, gate_acc.to(tl.float16), mask=mask)
    tl.store(up_ptr + out_offs, up_acc.to(tl.float16), mask=mask)


@triton.jit
def _zero_kernel(ptr, TOTAL: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < TOTAL
    tl.store(ptr + offs, tl.zeros((BLOCK,), tl.float32), mask=mask)


@triton.jit
def _scatter_weighted_add_kernel(
    down_ptr,
    token_idx_ptr,
    route_weight_ptr,
    final_ptr,
    E: tl.constexpr,
    M: tl.constexpr,
    H: tl.constexpr,
    TOTAL_TOKENS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_e = tl.program_id(2)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_h = pid_h * BLOCK_N + tl.arange(0, BLOCK_N)
    valid_m = offs_m < M
    token_idx = tl.load(token_idx_ptr + pid_e * M + offs_m, mask=valid_m, other=-1)
    rw = tl.load(route_weight_ptr + pid_e * M + offs_m, mask=valid_m, other=0.0).to(tl.float32)
    vals = tl.load(
        down_ptr + pid_e * M * H + offs_m[:, None] * H + offs_h[None, :],
        mask=valid_m[:, None] & (offs_h[None, :] < H),
        other=0.0,
    ).to(tl.float32)
    mask = (token_idx[:, None] >= 0) & (token_idx[:, None] < TOTAL_TOKENS) & (offs_h[None, :] < H)
    tl.atomic_add(final_ptr + token_idx[:, None] * H + offs_h[None, :], vals * rw[:, None], sem="relaxed", mask=mask)


def parse_csv_ints(value: str) -> List[int]:
    out = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not out:
        raise ValueError(f"empty integer list: {value!r}")
    return out


def parse_csv_strings(value: str) -> List[str]:
    out = [item.strip() for item in value.split(",") if item.strip()]
    if not out:
        raise ValueError(f"empty list: {value!r}")
    return out


def ceil_pow2(value: int) -> int:
    return 1 << (max(1, int(value)) - 1).bit_length()


def make_small_normal(shape: Sequence[int], device: str = "cuda") -> torch.Tensor:
    return (torch.randn(*shape, device=device, dtype=torch.float16) * 0.02).contiguous()


def make_rotation_bank(active_experts: int, stages: int, block_size: int, device: str = "cuda") -> torch.Tensor:
    stage_tensors = []
    for _ in range(stages):
        expert_tensors = []
        for _ in range(active_experts):
            mat = torch.randn((block_size, block_size), device=device, dtype=torch.float32)
            q, _ = torch.linalg.qr(mat)
            expert_tensors.append(q.to(torch.float16))
        stage_tensors.append(torch.stack(expert_tensors, dim=0))
    return torch.stack(stage_tensors, dim=0).contiguous()


def make_permutation_bank(active_experts: int, hidden: int, device: str = "cuda") -> torch.Tensor:
    perms = [torch.randperm(hidden, device=device, dtype=torch.int64) for _ in range(active_experts)]
    return torch.stack(perms, dim=0).contiguous()


def make_ose_indices(active_experts: int, hidden: int, topk: int, device: str = "cuda") -> Tuple[torch.Tensor, torch.Tensor]:
    gate = []
    up = []
    for _ in range(active_experts):
        gate.append(torch.randperm(hidden, device=device, dtype=torch.int64)[:topk])
        up.append(torch.randperm(hidden, device=device, dtype=torch.int64)[:topk])
    return torch.stack(gate, dim=0).contiguous(), torch.stack(up, dim=0).contiguous()


def zero_out_ose_rows(weight: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    out = weight.clone()
    for e in range(weight.shape[0]):
        out[e, indices[e], :] = 0
    return out.contiguous()


def gather_ose_weight_rows(weight: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    rows = []
    for e in range(weight.shape[0]):
        rows.append(weight[e].index_select(0, indices[e]))
    return torch.stack(rows, dim=0).contiguous()


def make_token_metadata(active_experts: int, tokens_per_expert: int, total_tokens: int, device: str = "cuda") -> Tuple[torch.Tensor, torch.Tensor]:
    token_ids = torch.randint(0, total_tokens, (active_experts, tokens_per_expert), device=device, dtype=torch.int64)
    route = torch.rand((active_experts, tokens_per_expert), device=device, dtype=torch.float32)
    return token_ids.contiguous(), route.contiguous()


def launch_gemm(a, b, c, e: int, m: int, k: int, n: int, args) -> None:
    grid = (triton.cdiv(m, args.block_m), triton.cdiv(n, args.block_n), e)
    _grouped_fp16_gemm_kernel[grid](
        a,
        b,
        c,
        E=e,
        M=m,
        K=k,
        N=n,
        BLOCK_M=args.block_m,
        BLOCK_N=args.block_n,
        BLOCK_K=args.block_k,
        num_warps=args.num_warps,
        num_stages=4,
    )


def launch_duquant_transform(x, tmp_a, tmp_b, rotations, perms, e: int, m: int, hidden: int, args):
    if hidden % args.duquant_block_size != 0:
        raise ValueError(f"hidden={hidden} must be divisible by block_size={args.duquant_block_size}")
    src = x
    dst = tmp_a
    scratch = tmp_b
    for stage in range(args.duquant_rotation_stages):
        r = rotations[stage]
        grid = (triton.cdiv(m, args.block_m), hidden // args.duquant_block_size, e)
        _duquant_block_rotate_kernel[grid](
            src,
            r,
            dst,
            E=e,
            M=m,
            H=hidden,
            BLOCK_M=args.block_m,
            BLOCK_H=args.duquant_block_size,
            num_warps=args.num_warps,
            num_stages=4,
        )
        src = dst
        dst = scratch if dst is tmp_a else tmp_a
        scratch = tmp_a if scratch is tmp_b else tmp_b
        if args.include_duquant_permutation and stage < args.duquant_rotation_stages - 1:
            grid_perm = (triton.cdiv(e * m * hidden, args.elem_block),)
            _permute_hidden_kernel[grid_perm](
                src,
                perms,
                dst,
                E=e,
                M=m,
                H=hidden,
                BLOCK=args.elem_block,
                num_warps=4,
            )
            src = dst
            dst = scratch if dst is tmp_a else tmp_a
            scratch = tmp_a if scratch is tmp_b else tmp_b
    return src


def launch_ose_branch(x, gate_idx, up_idx, gate_w, up_w, gate, up, e: int, m: int, hidden: int, intermediate: int, topk: int, add_to_main: bool, args) -> None:
    grid = (triton.cdiv(m, args.block_m), triton.cdiv(intermediate, args.block_n), e)
    _ose_two_proj_kernel[grid](
        x,
        gate_idx,
        up_idx,
        gate_w,
        up_w,
        gate,
        up,
        E=e,
        M=m,
        H=hidden,
        I=intermediate,
        TOPK=topk,
        ADD_TO_MAIN=add_to_main,
        BLOCK_M=args.block_m,
        BLOCK_N=args.block_n,
        BLOCK_K=ceil_pow2(topk),
        num_warps=args.num_warps,
        num_stages=4,
    )


def launch_silu_mul(gate, up, inter, total: int, args) -> None:
    grid = (triton.cdiv(total, args.elem_block),)
    _silu_mul_kernel[grid](gate, up, inter, TOTAL=total, BLOCK=args.elem_block, num_warps=4)


def launch_scatter(down, token_ids, route_weights, final, e: int, m: int, hidden: int, total_tokens: int, args) -> None:
    grid_zero = (triton.cdiv(total_tokens * hidden, args.elem_block),)
    _zero_kernel[grid_zero](final, TOTAL=total_tokens * hidden, BLOCK=args.elem_block, num_warps=4)
    grid = (triton.cdiv(m, args.block_m), triton.cdiv(hidden, args.block_n), e)
    _scatter_weighted_add_kernel[grid](
        down,
        token_ids,
        route_weights,
        final,
        E=e,
        M=m,
        H=hidden,
        TOTAL_TOKENS=total_tokens,
        BLOCK_M=args.block_m,
        BLOCK_N=args.block_n,
        num_warps=args.num_warps,
    )


def measure_ms(fn: Callable[[], None], warmup: int, repeat: int, rounds: int) -> Tuple[float, float, float, float]:
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


def torch_duquant_transform(x: torch.Tensor, rotations: torch.Tensor, perms: torch.Tensor, args) -> torch.Tensor:
    e, m, hidden = x.shape
    block = args.duquant_block_size
    y = x
    for stage in range(args.duquant_rotation_stages):
        r = rotations[stage].to(torch.float32)
        y = y.reshape(e, m, hidden // block, block).to(torch.float32)
        y = torch.einsum("embk,ekr->embr", y, r).reshape(e, m, hidden).to(torch.float16)
        if args.include_duquant_permutation and stage < args.duquant_rotation_stages - 1:
            y = y.gather(2, perms[:, None, :].expand(e, m, hidden))
    return y.contiguous()


def torch_gemm(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return torch.matmul(a.to(torch.float32), b.to(torch.float32)).to(torch.float16)


def torch_silu_mul(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    return (torch.nn.functional.silu(gate.to(torch.float32)) * up.to(torch.float32)).to(torch.float16)


def torch_ose_add(x, gate_idx, up_idx, gate_w, up_w, gate, up) -> Tuple[torch.Tensor, torch.Tensor]:
    gate_out = gate.to(torch.float32)
    up_out = up.to(torch.float32)
    e, m, _ = x.shape
    for expert in range(e):
        xg = x[expert].index_select(1, gate_idx[expert])
        xu = x[expert].index_select(1, up_idx[expert])
        gate_out[expert] += torch.matmul(xg.to(torch.float32), gate_w[expert].to(torch.float32))
        up_out[expert] += torch.matmul(xu.to(torch.float32), up_w[expert].to(torch.float32))
    return gate_out.to(torch.float16), up_out.to(torch.float16)


def torch_scatter(down, token_ids, route_weights, total_tokens: int) -> torch.Tensor:
    e, m, hidden = down.shape
    final = torch.zeros((total_tokens, hidden), device=down.device, dtype=torch.float32)
    for expert in range(e):
        final.index_add_(
            0,
            token_ids[expert],
            down[expert].to(torch.float32) * route_weights[expert, :, None],
        )
    return final


def error_stats(actual: torch.Tensor, expected: torch.Tensor) -> Tuple[float, float]:
    diff = (actual.to(torch.float32) - expected.to(torch.float32)).abs()
    rel = diff / expected.to(torch.float32).abs().clamp_min(1e-6)
    return float(diff.max().item()), float(rel.max().item())


def write_csv(path: str, rows: Iterable[dict], fieldnames: Sequence[str]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def build_case_tensors(preset: MoEPreset, active_experts: int, tokens_per_expert: int, args) -> dict:
    e = active_experts
    m = tokens_per_expert
    h = preset.hidden
    i = preset.intermediate
    topk = min(args.ose_topk, h)

    x = make_small_normal((e, m, h))
    w_gate_full = make_small_normal((e, h, i))
    w_up_full = make_small_normal((e, h, i))
    w_down = make_small_normal((e, i, h))
    gate_idx, up_idx = make_ose_indices(e, h, topk)
    gate_ose_w = gather_ose_weight_rows(w_gate_full, gate_idx)
    up_ose_w = gather_ose_weight_rows(w_up_full, up_idx)
    w_gate_main = zero_out_ose_rows(w_gate_full, gate_idx)
    w_up_main = zero_out_ose_rows(w_up_full, up_idx)

    rotations = make_rotation_bank(e, args.duquant_rotation_stages, args.duquant_block_size)
    perms = make_permutation_bank(e, h)
    total_tokens = max(args.total_tokens, e * m // max(preset.top_k, 1), 1)
    token_ids, route_weights = make_token_metadata(e, m, total_tokens)

    return {
        "x": x,
        "w_gate_full": w_gate_full,
        "w_up_full": w_up_full,
        "w_gate_main": w_gate_main,
        "w_up_main": w_up_main,
        "w_down": w_down,
        "gate_idx": gate_idx,
        "up_idx": up_idx,
        "gate_ose_w": gate_ose_w,
        "up_ose_w": up_ose_w,
        "rotations": rotations,
        "perms": perms,
        "token_ids": token_ids,
        "route_weights": route_weights,
        "total_tokens": total_tokens,
    }


def make_runtime_buffers(active_experts: int, tokens_per_expert: int, preset: MoEPreset, final_tokens: int, args) -> dict:
    e = active_experts
    m = tokens_per_expert
    h = preset.hidden
    i = preset.intermediate
    return {
        "x_tmp_a": torch.empty((e, m, h), device="cuda", dtype=torch.float16),
        "x_tmp_b": torch.empty((e, m, h), device="cuda", dtype=torch.float16),
        "x_tmp_gate": torch.empty((e, m, h), device="cuda", dtype=torch.float16),
        "x_tmp_up": torch.empty((e, m, h), device="cuda", dtype=torch.float16),
        "gate": torch.empty((e, m, i), device="cuda", dtype=torch.float16),
        "up": torch.empty((e, m, i), device="cuda", dtype=torch.float16),
        "inter": torch.empty((e, m, i), device="cuda", dtype=torch.float16),
        "down": torch.empty((e, m, h), device="cuda", dtype=torch.float16),
        "final": torch.empty((max(final_tokens, 1), h), device="cuda", dtype=torch.float32),
    }


def build_variant_fn(variant: str, tensors: dict, buffers: dict, preset: MoEPreset, active_experts: int, tokens_per_expert: int, include_scatter: bool, args) -> Tuple[Callable[[], None], str]:
    e = active_experts
    m = tokens_per_expert
    h = preset.hidden
    i = preset.intermediate
    topk = min(args.ose_topk, h)

    x = tensors["x"]
    w_down = tensors["w_down"]
    gate = buffers["gate"]
    up = buffers["up"]
    inter = buffers["inter"]
    down = buffers["down"]

    def finish_ffn():
        launch_silu_mul(gate, up, inter, e * m * i, args)
        launch_gemm(inter, w_down, down, e, m, i, h, args)
        if include_scatter:
            launch_scatter(
                down,
                tensors["token_ids"],
                tensors["route_weights"],
                buffers["final"],
                e,
                m,
                h,
                tensors["total_tokens"],
                args,
            )

    if variant == "plain_no_ose":
        def fn():
            launch_gemm(x, tensors["w_gate_full"], gate, e, m, h, i, args)
            launch_gemm(x, tensors["w_up_full"], up, e, m, h, i, args)
            finish_ffn()
        return fn, "2x_gemm+silu_mul+down_gemm"

    if variant == "duquant_once":
        def fn():
            x_du = launch_duquant_transform(
                x,
                buffers["x_tmp_a"],
                buffers["x_tmp_b"],
                tensors["rotations"],
                tensors["perms"],
                e,
                m,
                h,
                args,
            )
            launch_gemm(x_du, tensors["w_gate_full"], gate, e, m, h, i, args)
            launch_gemm(x_du, tensors["w_up_full"], up, e, m, h, i, args)
            finish_ffn()
        return fn, "duquant_transform_once+2x_gemm+silu_mul+down_gemm"

    if variant == "duquant_twice":
        def fn():
            x_gate = launch_duquant_transform(
                x,
                buffers["x_tmp_a"],
                buffers["x_tmp_b"],
                tensors["rotations"],
                tensors["perms"],
                e,
                m,
                h,
                args,
            )
            x_up = launch_duquant_transform(
                x,
                buffers["x_tmp_gate"],
                buffers["x_tmp_up"],
                tensors["rotations"],
                tensors["perms"],
                e,
                m,
                h,
                args,
            )
            launch_gemm(x_gate, tensors["w_gate_full"], gate, e, m, h, i, args)
            launch_gemm(x_up, tensors["w_up_full"], up, e, m, h, i, args)
            finish_ffn()
        return fn, "duquant_transform_twice+2x_gemm+silu_mul+down_gemm"

    if variant == "ours_ose":
        def fn():
            launch_gemm(x, tensors["w_gate_main"], gate, e, m, h, i, args)
            launch_gemm(x, tensors["w_up_main"], up, e, m, h, i, args)
            launch_ose_branch(
                x,
                tensors["gate_idx"],
                tensors["up_idx"],
                tensors["gate_ose_w"],
                tensors["up_ose_w"],
                gate,
                up,
                e,
                m,
                h,
                i,
                topk,
                True,
                args,
            )
            finish_ffn()
        return fn, "2x_main_gemm+ose_two_proj_add+silu_mul+down_gemm"

    if variant == "component_duquant_transform":
        def fn():
            launch_duquant_transform(
                x,
                buffers["x_tmp_a"],
                buffers["x_tmp_b"],
                tensors["rotations"],
                tensors["perms"],
                e,
                m,
                h,
                args,
            )
        return fn, "duquant_transform_only"

    if variant == "component_ose_branch":
        def fn():
            launch_ose_branch(
                x,
                tensors["gate_idx"],
                tensors["up_idx"],
                tensors["gate_ose_w"],
                tensors["up_ose_w"],
                gate,
                up,
                e,
                m,
                h,
                i,
                topk,
                False,
                args,
            )
        return fn, "ose_two_proj_only"

    raise ValueError(f"unknown variant: {variant}")


def reference_for_variant(variant: str, tensors: dict, preset: MoEPreset, include_scatter: bool, args) -> Optional[torch.Tensor]:
    if variant.startswith("component_"):
        return None
    x = tensors["x"]
    if variant in {"duquant_once", "duquant_twice"}:
        x_gate = torch_duquant_transform(x, tensors["rotations"], tensors["perms"], args)
        x_up = x_gate if variant == "duquant_once" else torch_duquant_transform(x, tensors["rotations"], tensors["perms"], args)
        gate = torch_gemm(x_gate, tensors["w_gate_full"])
        up = torch_gemm(x_up, tensors["w_up_full"])
    elif variant == "ours_ose":
        gate = torch_gemm(x, tensors["w_gate_main"])
        up = torch_gemm(x, tensors["w_up_main"])
        gate, up = torch_ose_add(
            x,
            tensors["gate_idx"],
            tensors["up_idx"],
            tensors["gate_ose_w"],
            tensors["up_ose_w"],
            gate,
            up,
        )
    elif variant == "plain_no_ose":
        gate = torch_gemm(x, tensors["w_gate_full"])
        up = torch_gemm(x, tensors["w_up_full"])
    else:
        raise ValueError(f"unknown variant: {variant}")

    inter = torch_silu_mul(gate, up)
    down = torch_gemm(inter, tensors["w_down"])
    if include_scatter:
        return torch_scatter(down, tensors["token_ids"], tensors["route_weights"], tensors["total_tokens"])
    return down


def actual_for_variant(variant: str, buffers: dict, include_scatter: bool) -> Optional[torch.Tensor]:
    if variant.startswith("component_"):
        return None
    return buffers["final"] if include_scatter else buffers["down"]


def run_case(scope: str, preset_name: str, active_experts: int, tokens_per_expert: int, variant: str, args) -> dict:
    preset = PRESETS[preset_name]
    include_scatter = scope == "moe_layer" and args.include_scatter
    tensors = build_case_tensors(preset, active_experts, tokens_per_expert, args)
    buffers = make_runtime_buffers(active_experts, tokens_per_expert, preset, tensors["total_tokens"], args)
    fn, kernel_sequence = build_variant_fn(
        variant,
        tensors,
        buffers,
        preset,
        active_experts,
        tokens_per_expert,
        include_scatter,
        args,
    )

    fn()
    torch.cuda.synchronize()

    max_abs = 0.0
    max_rel = 0.0
    if args.validate and not variant.startswith("component_"):
        ref = reference_for_variant(variant, tensors, preset, include_scatter, args)
        actual = actual_for_variant(variant, buffers, include_scatter)
        max_abs, max_rel = error_stats(actual, ref)

    median, q20, q80, stdev = measure_ms(fn, args.warmup, args.repeat, args.rounds)
    return {
        "scope": scope,
        "preset": preset_name,
        "variant": variant,
        "active_experts": active_experts,
        "tokens_per_expert": tokens_per_expert,
        "total_routed_tokens": active_experts * tokens_per_expert,
        "total_tokens": tensors["total_tokens"] if include_scatter else "",
        "hidden_size": preset.hidden,
        "intermediate_size": preset.intermediate,
        "num_experts_model": preset.num_experts,
        "top_k_model": preset.top_k,
        "ose_topk": min(args.ose_topk, preset.hidden),
        "duquant_block_size": args.duquant_block_size,
        "duquant_rotation_stages": args.duquant_rotation_stages,
        "include_duquant_permutation": int(args.include_duquant_permutation),
        "include_scatter": int(include_scatter),
        "latency_ms": f"{median:.6f}",
        "latency_q20_ms": f"{q20:.6f}",
        "latency_q80_ms": f"{q80:.6f}",
        "latency_std_ms": f"{stdev:.6f}",
        "speedup_over_duquant_once": "",
        "speedup_over_duquant_twice": "",
        "delta_over_plain_ms": "",
        "max_abs_error": f"{max_abs:.6e}",
        "max_rel_error": f"{max_rel:.6e}",
        "kernel_sequence": kernel_sequence,
        "block_m": args.block_m,
        "block_n": args.block_n,
        "block_k": args.block_k,
        "num_warps": args.num_warps,
        "dtype": "fp16_gemm_fp32_accum",
    }


FIELDS = [
    "scope",
    "preset",
    "variant",
    "active_experts",
    "tokens_per_expert",
    "total_routed_tokens",
    "total_tokens",
    "hidden_size",
    "intermediate_size",
    "num_experts_model",
    "top_k_model",
    "ose_topk",
    "duquant_block_size",
    "duquant_rotation_stages",
    "include_duquant_permutation",
    "include_scatter",
    "latency_ms",
    "latency_q20_ms",
    "latency_q80_ms",
    "latency_std_ms",
    "speedup_over_duquant_once",
    "speedup_over_duquant_twice",
    "delta_over_plain_ms",
    "max_abs_error",
    "max_rel_error",
    "kernel_sequence",
    "block_m",
    "block_n",
    "block_k",
    "num_warps",
    "dtype",
]


def add_derived_metrics(rows: List[dict]) -> None:
    grouped: Dict[Tuple[str, str, str, str], Dict[str, float]] = {}
    for row in rows:
        key = (row["scope"], row["preset"], row["active_experts"], row["tokens_per_expert"])
        grouped.setdefault(key, {})[row["variant"]] = float(row["latency_ms"])
    for row in rows:
        key = (row["scope"], row["preset"], row["active_experts"], row["tokens_per_expert"])
        latency = float(row["latency_ms"])
        du_once = grouped.get(key, {}).get("duquant_once")
        du_twice = grouped.get(key, {}).get("duquant_twice")
        plain = grouped.get(key, {}).get("plain_no_ose")
        if du_once:
            row["speedup_over_duquant_once"] = f"{du_once / latency:.6f}"
        if du_twice:
            row["speedup_over_duquant_twice"] = f"{du_twice / latency:.6f}"
        if plain is not None:
            row["delta_over_plain_ms"] = f"{latency - plain:.6f}"


def write_report(output_dir: str, rows: List[dict]) -> None:
    report_path = os.path.join(output_dir, "report.md")
    full_rows = [row for row in rows if not row["variant"].startswith("component_")]
    with open(report_path, "w") as f:
        f.write("# MoE FFN DuQuant-vs-OSE Structure Benchmark\n\n")
        f.write("This is a synthetic-weights structure benchmark.  All full variants use the same Triton FP16 GEMM kernel, FP32 accumulation, and FP16 intermediate/output tensors.\n\n")
        f.write("## Variants\n\n")
        f.write("- `duquant_once`: optimized DuQuant-like baseline.  The gate/up activation transform is computed once and reused by both gate and up GEMMs.\n")
        f.write("- `duquant_twice`: less optimized path matching a separate transform before gate and up.\n")
        f.write("- `plain_no_ose`: no gate/up DuQuant transform and no OSE branch.\n")
        f.write("- `ours_ose`: no gate/up DuQuant transform; add top-k input-channel OSE gate/up branch before `silu(gate) * up`.\n")
        f.write("- `component_duquant_transform` and `component_ose_branch`: isolated component timings.\n\n")
        f.write("## Important Scope\n\n")
        f.write("This benchmark proves structural runtime differences under a shared GEMM backend.  It does not prove packed INT4/INT8 deployment latency.\n\n")
        f.write("## Compact Summary\n\n")
        f.write("| scope | preset | E | T/expert | duquant_once ms | ours_ose ms | ours speedup vs duquant_once |\n")
        f.write("|---|---|---:|---:|---:|---:|---:|\n")
        keys = sorted({(r["scope"], r["preset"], r["active_experts"], r["tokens_per_expert"]) for r in full_rows})
        row_map = {(r["scope"], r["preset"], r["active_experts"], r["tokens_per_expert"], r["variant"]): r for r in full_rows}
        for key in keys:
            du = row_map.get((*key, "duquant_once"))
            ours = row_map.get((*key, "ours_ose"))
            if not du or not ours:
                continue
            f.write(
                f"| {key[0]} | {key[1]} | {key[2]} | {key[3]} | "
                f"{float(du['latency_ms']):.6f} | {float(ours['latency_ms']):.6f} | "
                f"{float(ours['speedup_over_duquant_once']):.4f} |\n"
            )


def run(args: argparse.Namespace) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this benchmark")
    torch.manual_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    scopes = parse_csv_strings(args.scopes)
    presets = parse_csv_strings(args.presets)
    variants = parse_csv_strings(args.variants)
    tokens_values = parse_csv_ints(args.tokens_per_expert)
    active_values = parse_csv_ints(args.active_experts)

    for preset in presets:
        if preset not in PRESETS:
            raise ValueError(f"unknown preset {preset!r}; valid: {', '.join(sorted(PRESETS))}")
    if args.duquant_rotation_stages < 1:
        raise ValueError("--duquant_rotation_stages must be >= 1")
    if args.duquant_block_size <= 0:
        raise ValueError("--duquant_block_size must be positive")

    rows: List[dict] = []
    total_cases = 0
    for scope in scopes:
        if scope not in {"single_expert", "moe_layer"}:
            raise ValueError(f"unknown scope: {scope}")
        for preset_name in presets:
            max_experts = PRESETS[preset_name].num_experts
            active_iter = [1] if scope == "single_expert" else [e for e in active_values if e <= max_experts]
            total_cases += len(tokens_values) * len(active_iter) * len(variants)

    done = 0
    for scope in scopes:
        for preset_name in presets:
            if scope == "single_expert":
                active_iter = [1]
            else:
                active_iter = active_values
            for active_experts in active_iter:
                max_experts = PRESETS[preset_name].num_experts
                if active_experts > max_experts:
                    print(
                        f"[moe-ffn-bench] skip scope={scope} preset={preset_name} "
                        f"E={active_experts}; model has only {max_experts} experts",
                        flush=True,
                    )
                    continue
                for tokens_per_expert in tokens_values:
                    for variant in variants:
                        done += 1
                        print(
                            f"[moe-ffn-bench] {done}/{total_cases} scope={scope} preset={preset_name} "
                            f"E={active_experts} T={tokens_per_expert} variant={variant}",
                            flush=True,
                        )
                        rows.append(run_case(scope, preset_name, active_experts, tokens_per_expert, variant, args))

    add_derived_metrics(rows)
    detail_path = os.path.join(args.output_dir, "moe_ffn_duquant_vs_ose_detail.csv")
    write_csv(detail_path, rows, FIELDS)
    write_report(args.output_dir, rows)
    print(f"[moe-ffn-bench] detail: {detail_path}")
    print(f"[moe-ffn-bench] report: {os.path.join(args.output_dir, 'report.md')}")


def main():
    parser = argparse.ArgumentParser(description="Triton MoE FFN DuQuant-vs-OSE structure benchmark.")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--scopes", default="single_expert,moe_layer")
    parser.add_argument("--presets", default="olmoe")
    parser.add_argument("--variants", default="duquant_once,ours_ose,plain_no_ose,duquant_twice,component_duquant_transform,component_ose_branch")
    parser.add_argument("--tokens_per_expert", default="1,2,4,8,16,32")
    parser.add_argument("--active_experts", default="8,16,64")
    parser.add_argument("--total_tokens", type=int, default=0)
    parser.add_argument("--ose_topk", type=int, default=64)
    parser.add_argument("--duquant_block_size", type=int, default=128)
    parser.add_argument("--duquant_rotation_stages", type=int, default=2)
    parser.add_argument("--include_duquant_permutation", action="store_true", default=True)
    parser.add_argument("--no_include_duquant_permutation", dest="include_duquant_permutation", action="store_false")
    parser.add_argument("--include_scatter", action="store_true", default=True)
    parser.add_argument("--no_include_scatter", dest="include_scatter", action="store_false")
    parser.add_argument("--validate", action="store_true", default=True)
    parser.add_argument("--no_validate", dest="validate", action="store_false")
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--repeat", type=int, default=200)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--block_m", type=int, default=16)
    parser.add_argument("--block_n", type=int, default=64)
    parser.add_argument("--block_k", type=int, default=64)
    parser.add_argument("--num_warps", type=int, default=4)
    parser.add_argument("--elem_block", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=2)
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
