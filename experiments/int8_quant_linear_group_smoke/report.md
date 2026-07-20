# INT8 Quantized Linear Group-Size Benchmark

This benchmark uses real INT8 dot-product kernels (`tl.dot` with INT32 accumulation). It measures the runtime after activation and weight tensors have already been quantized.

## Scopes

- `single_gemm`: one quantized Linear projection, including INT8 GEMM and output scaling.
- `epilogue_only`: only INT32-to-FP16 scale application; this is an upper bound on group-size runtime benefit.
- `moe_expert`: gate, up, and down grouped expert projections for a routed MoE MLP shape.
- `moe_epilogue_only`: epilogue-only version of the three MoE projections.

Group size changes only output-channel weight scale sharing. If full GEMM speedup is small while epilogue-only speedup is larger, the reason is that GEMM math and weight traffic dominate the full layer.
