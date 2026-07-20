# Post-Quant MoE FFN DuQuant-vs-OSE Benchmark

This benchmark measures a synthetic post-quant W8A8 FFN runtime path. Calibration, DuQuant search, weight-side transforms, OSE channel selection, weight quantization, and packing are excluded.

`duquant_once` applies the gate/up DuQuant activation transform once and reuses it for both gate and up. `ours_ose_w8a8` removes that gate/up transform and adds a W8A8 top-k input-channel OSE branch. The optional down DuQuant transform is a common cost and is included when `include_down_duquant_transform=1`.

## Summary

| preset | E | tokens/expert | duquant_once ms | plain_no_ose ms | ours_ose_w8a8 ms | ours speedup vs duquant_once |
|---|---:|---:|---:|---:|---:|---:|
| olmoe | 1 | 1 | 0.872496 | 0.639131 | 0.743031 | 1.1742 |
| olmoe | 1 | 2 | 0.849767 | 0.636754 | 0.738255 | 1.1510 |
| olmoe | 1 | 4 | 0.850734 | 0.641884 | 0.739552 | 1.1503 |
| olmoe | 1 | 8 | 0.850956 | 0.643246 | 0.744846 | 1.1425 |
| olmoe | 1 | 16 | 0.876237 | 0.641648 | 0.740309 | 1.1836 |
| olmoe | 1 | 32 | 0.854817 | 0.641687 | 0.749429 | 1.1406 |
| olmoe | 8 | 1 | 0.872321 | 0.640046 | 0.748984 | 1.1647 |
| olmoe | 8 | 2 | 1.111592 | 0.640892 | 0.751366 | 1.4794 |
| olmoe | 8 | 4 | 0.853111 | 0.639316 | 0.746340 | 1.1431 |
| olmoe | 8 | 8 | 0.850827 | 0.640662 | 0.743166 | 1.1449 |
| olmoe | 8 | 16 | 0.852364 | 0.638177 | 0.743801 | 1.1460 |
| olmoe | 8 | 32 | 0.854655 | 0.644541 | 0.740262 | 1.1545 |
| olmoe | 16 | 1 | 0.859073 | 0.646043 | 0.746078 | 1.1515 |
| olmoe | 16 | 2 | 0.846700 | 0.638817 | 0.739587 | 1.1448 |
| olmoe | 16 | 4 | 0.850640 | 0.640360 | 0.739330 | 1.1506 |
| olmoe | 16 | 8 | 0.854110 | 0.643693 | 0.746190 | 1.1446 |
| olmoe | 16 | 16 | 0.848059 | 0.638190 | 0.744982 | 1.1384 |
| olmoe | 16 | 32 | 0.859846 | 0.648420 | 0.743962 | 1.1558 |
| olmoe | 64 | 1 | 0.895549 | 0.844788 | 0.868798 | 1.0308 |
| olmoe | 64 | 2 | 0.901898 | 0.849452 | 0.874752 | 1.0310 |
| olmoe | 64 | 4 | 0.916103 | 0.859423 | 0.885990 | 1.0340 |
| olmoe | 64 | 8 | 0.940661 | 0.879743 | 0.909536 | 1.0342 |
| olmoe | 64 | 16 | 0.989525 | 0.913132 | 0.959850 | 1.0309 |
| olmoe | 64 | 32 | 1.132060 | 0.984217 | 1.069223 | 1.0588 |
