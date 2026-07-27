# Post-Quant MoE FFN DuQuant-vs-OSE Benchmark

This benchmark measures a synthetic post-quant W8A8 MoE runtime path. Calibration, DuQuant search, weight-side transforms, OSE channel selection, weight quantization, and packing are excluded.

`duquant_once` applies the gate/up DuQuant activation transform once and reuses it for both gate and up. `ours_ose_w8a8` removes that gate/up transform and adds a W8A8 top-k input-channel OSE branch. The optional down DuQuant transform is a common cost and is included when `include_down_duquant_transform=1`.

## Summary

| scope | preset | E | tokens/expert | total tokens | duquant_once ms | plain_no_ose ms | ours_ose_w8a8 ms | ours speedup vs duquant_once |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| routed_moe_layer | olmoe | 1 | 1 | 1 | 1.063691 | 0.842192 | 0.949388 | 1.1204 |
| routed_moe_layer | olmoe | 1 | 2 | 1 | 1.054180 | 0.839722 | 0.941001 | 1.1203 |
| routed_moe_layer | olmoe | 1 | 4 | 1 | 1.052731 | 0.839344 | 0.949612 | 1.1086 |
| routed_moe_layer | olmoe | 1 | 8 | 1 | 1.057738 | 0.841394 | 0.949759 | 1.1137 |
| routed_moe_layer | olmoe | 1 | 16 | 2 | 1.045616 | 0.830847 | 0.945887 | 1.1054 |
| routed_moe_layer | olmoe | 1 | 32 | 4 | 1.051943 | 0.836652 | 0.942609 | 1.1160 |
| routed_moe_layer | olmoe | 8 | 1 | 1 | 1.057127 | 0.840411 | 0.954788 | 1.1072 |
| routed_moe_layer | olmoe | 8 | 2 | 2 | 1.059229 | 0.833524 | 0.946568 | 1.1190 |
| routed_moe_layer | olmoe | 8 | 4 | 4 | 1.061842 | 0.841103 | 0.949347 | 1.1185 |
| routed_moe_layer | olmoe | 8 | 8 | 8 | 1.057794 | 0.839328 | 0.936559 | 1.1294 |
| routed_moe_layer | olmoe | 8 | 16 | 16 | 1.059717 | 0.852745 | 0.949992 | 1.1155 |
| routed_moe_layer | olmoe | 8 | 32 | 32 | 1.057756 | 0.841186 | 0.954834 | 1.1078 |
| routed_moe_layer | olmoe | 16 | 1 | 2 | 1.046209 | 0.831614 | 0.945890 | 1.1061 |
| routed_moe_layer | olmoe | 16 | 2 | 4 | 1.078342 | 0.840835 | 0.935195 | 1.1531 |
| routed_moe_layer | olmoe | 16 | 4 | 8 | 1.053301 | 0.843023 | 0.952701 | 1.1056 |
| routed_moe_layer | olmoe | 16 | 8 | 16 | 1.055507 | 0.836018 | 0.948401 | 1.1129 |
| routed_moe_layer | olmoe | 16 | 16 | 32 | 1.061958 | 0.841744 | 0.973508 | 1.0909 |
| routed_moe_layer | olmoe | 16 | 32 | 64 | 1.364025 | 0.844942 | 0.953608 | 1.4304 |
| routed_moe_layer | olmoe | 64 | 1 | 8 | 1.078835 | 1.033212 | 1.051409 | 1.0261 |
| routed_moe_layer | olmoe | 64 | 2 | 16 | 1.099070 | 1.042431 | 1.052502 | 1.0442 |
| routed_moe_layer | olmoe | 64 | 4 | 32 | 1.304496 | 1.146297 | 1.081366 | 1.2063 |
| routed_moe_layer | olmoe | 64 | 8 | 64 | 1.158734 | 1.081848 | 1.141188 | 1.0154 |
| routed_moe_layer | olmoe | 64 | 16 | 128 | 1.216362 | 1.126341 | 1.198490 | 1.0149 |
| routed_moe_layer | olmoe | 64 | 32 | 256 | 1.439606 | 1.253900 | 1.385104 | 1.0393 |
