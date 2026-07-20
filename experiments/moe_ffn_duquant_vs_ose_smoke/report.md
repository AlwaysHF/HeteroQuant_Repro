# MoE FFN DuQuant-vs-OSE Structure Benchmark

This is a synthetic-weights structure benchmark.  All full variants use the same Triton FP16 GEMM kernel, FP32 accumulation, and FP16 intermediate/output tensors.

## Variants

- `duquant_once`: optimized DuQuant-like baseline.  The gate/up activation transform is computed once and reused by both gate and up GEMMs.
- `duquant_twice`: less optimized path matching a separate transform before gate and up.
- `plain_no_ose`: no gate/up DuQuant transform and no OSE branch.
- `ours_ose`: no gate/up DuQuant transform; add top-k input-channel OSE gate/up branch before `silu(gate) * up`.
- `component_duquant_transform` and `component_ose_branch`: isolated component timings.

## Important Scope

This benchmark proves structural runtime differences under a shared GEMM backend.  It does not prove packed INT4/INT8 deployment latency.

## Compact Summary

| scope | preset | E | T/expert | duquant_once ms | ours_ose ms | ours speedup vs duquant_once |
|---|---|---:|---:|---:|---:|---:|
| moe_layer | olmoe | 2 | 1 | 0.849557 | 0.785035 | 1.0822 |
| single_expert | olmoe | 1 | 1 | 0.520213 | 0.373179 | 1.3940 |
