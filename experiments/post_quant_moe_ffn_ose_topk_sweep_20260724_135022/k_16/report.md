# Post-Quant MoE FFN DuQuant-vs-OSE Benchmark

This benchmark measures a synthetic post-quant W8A8 MoE runtime path. Calibration, DuQuant search, weight-side transforms, OSE channel selection, weight quantization, and packing are excluded.

`duquant_once` applies the gate/up DuQuant activation transform once and reuses it for both gate and up. `ours_ose_w8a8` removes that gate/up transform and adds a W8A8 top-k input-channel OSE branch. The optional down DuQuant transform is a common cost and is included when `include_down_duquant_transform=1`.

## Summary

| scope | preset | E | tokens/expert | total tokens | duquant_once ms | plain_no_ose ms | ours_ose_w8a8 ms | ours speedup vs duquant_once |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| routed_moe_layer | olmoe | 1 | 1 | 1 | 1.052243 | 0.834933 | 0.943501 | 1.1153 |
| routed_moe_layer | olmoe | 1 | 2 | 1 | 1.056397 | 0.836478 | 0.933827 | 1.1313 |
| routed_moe_layer | olmoe | 1 | 4 | 1 | 1.057148 | 0.837747 | 0.943844 | 1.1200 |
| routed_moe_layer | olmoe | 1 | 8 | 1 | 1.096103 | 0.830515 | 0.946179 | 1.1585 |
| routed_moe_layer | olmoe | 1 | 16 | 2 | 1.048732 | 0.830685 | 0.934287 | 1.1225 |
| routed_moe_layer | olmoe | 1 | 32 | 4 | 1.050272 | 0.826032 | 0.932826 | 1.1259 |
| routed_moe_layer | olmoe | 8 | 1 | 1 | 1.053463 | 0.831147 | 0.941438 | 1.1190 |
| routed_moe_layer | olmoe | 8 | 2 | 2 | 1.091288 | 0.872664 | 1.085788 | 1.0051 |
| routed_moe_layer | olmoe | 8 | 4 | 4 | 1.290864 | 0.822944 | 1.071962 | 1.2042 |
| routed_moe_layer | olmoe | 8 | 8 | 8 | 1.045536 | 0.839111 | 0.938584 | 1.1139 |
| routed_moe_layer | olmoe | 8 | 16 | 16 | 1.043230 | 0.838709 | 0.951010 | 1.0970 |
| routed_moe_layer | olmoe | 8 | 32 | 32 | 1.047187 | 0.839500 | 0.936638 | 1.1180 |
| routed_moe_layer | olmoe | 16 | 1 | 2 | 1.050130 | 0.846215 | 0.937346 | 1.1203 |
| routed_moe_layer | olmoe | 16 | 2 | 4 | 1.046400 | 0.831743 | 0.953500 | 1.0974 |
| routed_moe_layer | olmoe | 16 | 4 | 8 | 1.050186 | 0.837825 | 1.303924 | 0.8054 |
| routed_moe_layer | olmoe | 16 | 8 | 16 | 1.049506 | 0.834021 | 0.937687 | 1.1193 |
| routed_moe_layer | olmoe | 16 | 16 | 32 | 1.052638 | 0.828970 | 0.936799 | 1.1237 |
| routed_moe_layer | olmoe | 16 | 32 | 64 | 1.045954 | 0.834283 | 0.940791 | 1.1118 |
| routed_moe_layer | olmoe | 64 | 1 | 8 | 1.159281 | 1.116389 | 1.120470 | 1.0346 |
| routed_moe_layer | olmoe | 64 | 2 | 16 | 1.163866 | 1.115429 | 1.114534 | 1.0443 |
| routed_moe_layer | olmoe | 64 | 4 | 32 | 1.193732 | 1.124390 | 1.146247 | 1.0414 |
| routed_moe_layer | olmoe | 64 | 8 | 64 | 1.285801 | 1.164516 | 1.170381 | 1.0986 |
| routed_moe_layer | olmoe | 64 | 16 | 128 | 1.289164 | 1.225860 | 1.294509 | 0.9959 |
| routed_moe_layer | olmoe | 64 | 32 | 256 | 1.572113 | 1.374793 | 1.470475 | 1.0691 |
