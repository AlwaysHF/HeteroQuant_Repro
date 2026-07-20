# Post-Quant MoE FFN DuQuant-vs-OSE Benchmark

This benchmark measures a synthetic post-quant W8A8 FFN runtime path. Calibration, DuQuant search, weight-side transforms, OSE channel selection, weight quantization, and packing are excluded.

`duquant_once` applies the gate/up DuQuant activation transform once and reuses it for both gate and up. `ours_ose_w8a8` removes that gate/up transform and adds a W8A8 top-k input-channel OSE branch. The optional down DuQuant transform is a common cost and is included when `include_down_duquant_transform=1`.

## Summary

| preset | E | tokens/expert | duquant_once ms | plain_no_ose ms | ours_ose_w8a8 ms | ours speedup vs duquant_once |
|---|---:|---:|---:|---:|---:|---:|
| olmoe | 1 | 1 | 0.843750 | 0.634011 | 0.738023 | 1.1433 |
| olmoe | 1 | 2 | 0.839441 | 0.636725 | 0.734057 | 1.1436 |
| olmoe | 1 | 4 | 0.840262 | 0.631487 | 0.738848 | 1.1373 |
| olmoe | 1 | 8 | 0.845990 | 0.638673 | 0.742573 | 1.1393 |
| olmoe | 1 | 16 | 0.862881 | 0.637807 | 0.755877 | 1.1416 |
| olmoe | 1 | 32 | 0.846426 | 0.638221 | 0.742455 | 1.1400 |
| olmoe | 8 | 1 | 0.839697 | 0.631892 | 0.737092 | 1.1392 |
| olmoe | 8 | 2 | 0.848197 | 0.634633 | 0.738577 | 1.1484 |
| olmoe | 8 | 4 | 0.846833 | 0.631749 | 0.739798 | 1.1447 |
| olmoe | 8 | 8 | 0.875655 | 0.643192 | 0.744268 | 1.1765 |
| olmoe | 8 | 16 | 0.840685 | 0.625332 | 0.737466 | 1.1400 |
| olmoe | 8 | 32 | 0.924363 | 0.646340 | 0.737952 | 1.2526 |
| olmoe | 16 | 1 | 0.851183 | 0.642440 | 0.741068 | 1.1486 |
| olmoe | 16 | 2 | 0.839862 | 0.634788 | 0.747185 | 1.1240 |
| olmoe | 16 | 4 | 0.846857 | 0.637502 | 0.737433 | 1.1484 |
| olmoe | 16 | 8 | 0.894158 | 0.642782 | 0.745220 | 1.1999 |
| olmoe | 16 | 16 | 0.888703 | 0.632558 | 0.743847 | 1.1947 |
| olmoe | 16 | 32 | 0.844116 | 0.635709 | 0.742092 | 1.1375 |
| olmoe | 64 | 1 | 0.896048 | 0.844948 | 0.870150 | 1.0298 |
| olmoe | 64 | 2 | 0.905346 | 0.850971 | 0.875833 | 1.0337 |
| olmoe | 64 | 4 | 0.918930 | 0.861373 | 0.888093 | 1.0347 |
| olmoe | 64 | 8 | 0.942850 | 0.881594 | 0.911659 | 1.0342 |
| olmoe | 64 | 16 | 0.991032 | 0.915002 | 0.961888 | 1.0303 |
| olmoe | 64 | 32 | 1.139081 | 0.991717 | 1.076016 | 1.0586 |
