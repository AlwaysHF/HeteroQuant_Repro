# Output-Channel Scale Sharing Benchmark

This benchmark keeps the integer GEMM body identical across granularities. Only the output-scale epilogue is specialized.

## Interpretation

- `full_gemm` is the relevant GEMM speedup measurement.
- `epilogue_only` isolates scale indexing/loading/apply cost and must not be described as full GEMM speedup.
- `grouped_moe` measures multiple active experts with identical launch count across granularities.
- Scale-count reduction is metadata reduction, not runtime speedup.

If `full_gemm` speedup is small, the expected reason is that dot-product work and weight traffic dominate while output-scale epilogue is a small fraction of total time.
