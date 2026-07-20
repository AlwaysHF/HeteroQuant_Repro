# Post-Quant MoE FFN DuQuant-vs-OSE Benchmark

This benchmark measures a synthetic post-quant W8A8 FFN runtime path. Calibration, DuQuant search, weight-side transforms, OSE channel selection, weight quantization, and packing are excluded.

`duquant_once` applies the gate/up DuQuant activation transform once and reuses it for both gate and up. `ours_ose_w8a8` removes that gate/up transform and adds a W8A8 top-k input-channel OSE branch. The optional down DuQuant transform is a common cost and is included when `include_down_duquant_transform=1`.

## Summary

| preset | E | tokens/expert | duquant_once ms | plain_no_ose ms | ours_ose_w8a8 ms | ours speedup vs duquant_once |
|---|---:|---:|---:|---:|---:|---:|
| olmoe | 1 | 1 | 0.849750 | 0.642337 | 0.748807 | 1.1348 |
| olmoe | 1 | 2 | 0.865086 | 0.642321 | 0.748703 | 1.1554 |
| olmoe | 1 | 4 | 0.847066 | 0.637206 | 0.736292 | 1.1504 |
| olmoe | 1 | 8 | 0.855347 | 0.644739 | 0.744024 | 1.1496 |
| olmoe | 1 | 16 | 0.856100 | 0.641985 | 0.742750 | 1.1526 |
| olmoe | 1 | 32 | 0.848997 | 0.639787 | 0.742258 | 1.1438 |
| olmoe | 8 | 1 | 0.847973 | 0.641833 | 0.746824 | 1.1354 |
| olmoe | 8 | 2 | 0.847838 | 0.639090 | 0.744958 | 1.1381 |
| olmoe | 8 | 4 | 0.857377 | 0.638923 | 0.742870 | 1.1541 |
| olmoe | 8 | 8 | 0.850580 | 0.643915 | 0.747109 | 1.1385 |
| olmoe | 8 | 16 | 0.853298 | 0.646783 | 0.746382 | 1.1432 |
| olmoe | 8 | 32 | 0.855895 | 0.640913 | 0.746913 | 1.1459 |
| olmoe | 16 | 1 | 0.853273 | 0.648119 | 0.749409 | 1.1386 |
| olmoe | 16 | 2 | 0.852423 | 0.641641 | 0.748988 | 1.1381 |
| olmoe | 16 | 4 | 0.855335 | 0.640961 | 0.747142 | 1.1448 |
| olmoe | 16 | 8 | 0.869830 | 0.647138 | 0.745222 | 1.1672 |
| olmoe | 16 | 16 | 0.860377 | 0.646704 | 0.749641 | 1.1477 |
| olmoe | 16 | 32 | 0.853398 | 0.641126 | 0.744289 | 1.1466 |
| olmoe | 64 | 1 | 0.892947 | 0.842692 | 0.867538 | 1.0293 |
| olmoe | 64 | 2 | 0.900423 | 0.848586 | 0.874264 | 1.0299 |
| olmoe | 64 | 4 | 0.915990 | 0.859891 | 0.886562 | 1.0332 |
| olmoe | 64 | 8 | 0.940038 | 0.879578 | 0.909643 | 1.0334 |
| olmoe | 64 | 16 | 0.988973 | 0.912930 | 0.959602 | 1.0306 |
| olmoe | 64 | 32 | 1.131489 | 0.985447 | 1.070304 | 1.0572 |
