# MoE Layer MAC/Latency With Ratios

Speedup is computed against the aggregated DuQuant row. MAC is normalized per routed token-expert for the benchmarked MoE FFN path.

| Method | MAC | MAC Speedup | E=1 Med / Speedup | E=8 Med / Speedup | E=16 Med / Speedup | E=64 Med / Speedup |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| DuQuant | 7,077,888 | 1.000x | 1.056250 / 1.000x | 1.058511 / 1.000x | 1.053275 / 1.000x | 1.209987 / 1.000x |
| Ours k=16 | 6,586,368 | 1.075x | 0.938894 / 1.125x | 0.946224 / 1.119x | 0.939239 / 1.121x | 1.158314 / 1.045x |
| Ours k=32 | 6,619,136 | 1.069x | 0.946873 / 1.116x | 0.939538 / 1.127x | 0.946838 / 1.112x | 1.198440 / 1.010x |
| Ours k=64 | 6,684,672 | 1.059x | 0.947638 / 1.115x | 0.949669 / 1.115x | 0.950551 / 1.108x | 1.111277 / 1.089x |
| Ours k=128 | 6,815,744 | 1.038x | 0.953884 / 1.107x | 0.952134 / 1.112x | 0.954611 / 1.103x | 1.256422 / 0.963x |
| Ours k=256 | 7,077,888 | 1.000x | 0.943697 / 1.119x | 0.948308 / 1.116x | 0.945815 / 1.114x | 1.304361 / 0.928x |
