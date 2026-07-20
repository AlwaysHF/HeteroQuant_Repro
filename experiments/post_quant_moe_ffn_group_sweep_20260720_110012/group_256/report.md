# Post-Quant MoE FFN DuQuant-vs-OSE Benchmark

This benchmark measures a synthetic post-quant W8A8 FFN runtime path. Calibration, DuQuant search, weight-side transforms, OSE channel selection, weight quantization, and packing are excluded.

`duquant_once` applies the gate/up DuQuant activation transform once and reuses it for both gate and up. `ours_ose_w8a8` removes that gate/up transform and adds a W8A8 top-k input-channel OSE branch. The optional down DuQuant transform is a common cost and is included when `include_down_duquant_transform=1`.

## Summary

| preset | E | tokens/expert | duquant_once ms | plain_no_ose ms | ours_ose_w8a8 ms | ours speedup vs duquant_once |
|---|---:|---:|---:|---:|---:|---:|
| olmoe | 1 | 1 | 0.880609 | 0.644998 | 0.748326 | 1.1768 |
| olmoe | 1 | 2 | 0.851846 | 0.642540 | 0.744418 | 1.1443 |
| olmoe | 1 | 4 | 0.855810 | 0.647019 | 0.750665 | 1.1401 |
| olmoe | 1 | 8 | 0.880402 | 0.644640 | 0.747218 | 1.1782 |
| olmoe | 1 | 16 | 0.855794 | 0.648068 | 0.750919 | 1.1397 |
| olmoe | 1 | 32 | 0.876320 | 0.642782 | 0.768580 | 1.1402 |
| olmoe | 8 | 1 | 0.854268 | 0.642089 | 0.744050 | 1.1481 |
| olmoe | 8 | 2 | 0.857328 | 0.642954 | 0.756872 | 1.1327 |
| olmoe | 8 | 4 | 0.858681 | 0.646334 | 0.751260 | 1.1430 |
| olmoe | 8 | 8 | 0.880459 | 0.665487 | 0.746464 | 1.1795 |
| olmoe | 8 | 16 | 0.875249 | 0.642971 | 0.742482 | 1.1788 |
| olmoe | 8 | 32 | 0.858693 | 0.648108 | 0.751628 | 1.1424 |
| olmoe | 16 | 1 | 0.858851 | 0.653750 | 0.745410 | 1.1522 |
| olmoe | 16 | 2 | 0.854350 | 0.642202 | 0.750923 | 1.1377 |
| olmoe | 16 | 4 | 0.865095 | 0.654489 | 0.746339 | 1.1591 |
| olmoe | 16 | 8 | 0.861346 | 0.644877 | 0.756127 | 1.1392 |
| olmoe | 16 | 16 | 0.851544 | 0.640399 | 0.747738 | 1.1388 |
| olmoe | 16 | 32 | 0.857836 | 0.648344 | 0.755284 | 1.1358 |
| olmoe | 64 | 1 | 0.894171 | 0.843468 | 0.868644 | 1.0294 |
| olmoe | 64 | 2 | 0.901442 | 0.849018 | 0.874609 | 1.0307 |
| olmoe | 64 | 4 | 0.916660 | 0.859959 | 0.886638 | 1.0339 |
| olmoe | 64 | 8 | 0.940843 | 0.879396 | 0.909468 | 1.0345 |
| olmoe | 64 | 16 | 0.989108 | 0.912927 | 0.959664 | 1.0307 |
| olmoe | 64 | 32 | 1.133906 | 0.985310 | 1.071418 | 1.0583 |
