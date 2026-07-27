# Post-Quant MoE FFN DuQuant-vs-OSE Benchmark

This benchmark measures a synthetic post-quant W8A8 MoE runtime path. Calibration, DuQuant search, weight-side transforms, OSE channel selection, weight quantization, and packing are excluded.

`duquant_once` applies the gate/up DuQuant activation transform once and reuses it for both gate and up. `ours_ose_w8a8` removes that gate/up transform and adds a W8A8 top-k input-channel OSE branch. The optional down DuQuant transform is a common cost and is included when `include_down_duquant_transform=1`.

## Summary

| scope | preset | E | tokens/expert | total tokens | duquant_once ms | plain_no_ose ms | ours_ose_w8a8 ms | ours speedup vs duquant_once |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| routed_moe_layer | olmoe | 1 | 1 | 1 | 1.057617 | 0.846019 | 0.970544 | 1.0897 |
| routed_moe_layer | olmoe | 1 | 2 | 1 | 1.052822 | 0.840849 | 0.952164 | 1.1057 |
| routed_moe_layer | olmoe | 1 | 4 | 1 | 1.065934 | 0.861607 | 0.955604 | 1.1155 |
| routed_moe_layer | olmoe | 1 | 8 | 1 | 1.049329 | 0.844069 | 0.947871 | 1.1070 |
| routed_moe_layer | olmoe | 1 | 16 | 2 | 1.042174 | 0.835262 | 0.944592 | 1.1033 |
| routed_moe_layer | olmoe | 1 | 32 | 4 | 1.061140 | 0.842620 | 0.960002 | 1.1054 |
| routed_moe_layer | olmoe | 8 | 1 | 1 | 1.051956 | 0.842338 | 0.964097 | 1.0911 |
| routed_moe_layer | olmoe | 8 | 2 | 2 | 1.076632 | 0.841196 | 0.952456 | 1.1304 |
| routed_moe_layer | olmoe | 8 | 4 | 4 | 1.061654 | 0.848169 | 0.957323 | 1.1090 |
| routed_moe_layer | olmoe | 8 | 8 | 8 | 1.060022 | 0.843616 | 0.944654 | 1.1221 |
| routed_moe_layer | olmoe | 8 | 16 | 16 | 1.070568 | 0.849158 | 0.951812 | 1.1248 |
| routed_moe_layer | olmoe | 8 | 32 | 32 | 1.068876 | 0.859248 | 0.949571 | 1.1256 |
| routed_moe_layer | olmoe | 16 | 1 | 2 | 1.095410 | 1.000188 | 1.130897 | 0.9686 |
| routed_moe_layer | olmoe | 16 | 2 | 4 | 1.068596 | 0.850383 | 0.947688 | 1.1276 |
| routed_moe_layer | olmoe | 16 | 4 | 8 | 1.067367 | 0.849480 | 0.980470 | 1.0886 |
| routed_moe_layer | olmoe | 16 | 8 | 16 | 1.059216 | 0.838622 | 0.942677 | 1.1236 |
| routed_moe_layer | olmoe | 16 | 16 | 32 | 1.058355 | 0.840704 | 0.953945 | 1.1095 |
| routed_moe_layer | olmoe | 16 | 32 | 64 | 1.063187 | 0.847009 | 0.955278 | 1.1130 |
| routed_moe_layer | olmoe | 64 | 1 | 8 | 1.109123 | 1.113800 | 1.162056 | 0.9544 |
| routed_moe_layer | olmoe | 64 | 2 | 16 | 1.160275 | 1.134587 | 1.169907 | 0.9918 |
| routed_moe_layer | olmoe | 64 | 4 | 32 | 1.200355 | 1.130061 | 1.249296 | 0.9608 |
| routed_moe_layer | olmoe | 64 | 8 | 64 | 1.305287 | 1.190754 | 1.263548 | 1.0330 |
| routed_moe_layer | olmoe | 64 | 16 | 128 | 1.295419 | 1.237046 | 1.303430 | 0.9939 |
| routed_moe_layer | olmoe | 64 | 32 | 256 | 1.579567 | 1.384304 | 1.575902 | 1.0023 |
