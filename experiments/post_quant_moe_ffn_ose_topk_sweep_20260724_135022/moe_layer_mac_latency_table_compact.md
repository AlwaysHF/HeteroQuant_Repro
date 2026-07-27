# Compact MoE Layer MAC and Median Latency Table

DuQuant is aggregated across all k sweep runs because k only applies to Ours. MAC is normalized per routed token-expert for the benchmarked MoE FFN path.

| Method | MoE FFN MAC / routed token-expert | E=1 Med (ms) | E=8 Med (ms) | E=16 Med (ms) | E=64 Med (ms) |
| --- | --- | --- | --- | --- | --- |
| DuQuant | 7,077,888 | 1.056250 | 1.058511 | 1.053275 | 1.209987 |
| Ours k=16 | 6,586,368 | 0.938894 | 0.946224 | 0.939239 | 1.158314 |
| Ours k=32 | 6,619,136 | 0.946873 | 0.939538 | 0.946838 | 1.198440 |
| Ours k=64 | 6,684,672 | 0.947638 | 0.949669 | 0.950551 | 1.111277 |
| Ours k=128 | 6,815,744 | 0.953884 | 0.952134 | 0.954611 | 1.256422 |
| Ours k=256 | 7,077,888 | 0.943697 | 0.948308 | 0.945815 | 1.304361 |
