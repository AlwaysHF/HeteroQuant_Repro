# MoE Layer MAC and Median Latency Table

MAC is normalized per routed token-expert and includes the benchmarked MoE FFN path: gate/up/down main GEMMs plus the method-specific online overhead and the common down DuQuant transform. Router/attention/norm/residual are not included.

| Method | MoE FFN MAC / routed token-expert | E=1 Med (ms) | E=8 Med (ms) | E=16 Med (ms) | E=64 Med (ms) |
| --- | --- | --- | --- | --- | --- |
| DuQuant (k=16 ref) | 7,077,888 | 1.054320 | 1.050325 | 1.049818 | 1.239766 |
| Ours k=16 | 6,586,368 | 0.938894 | 0.946224 | 0.939239 | 1.158314 |
| DuQuant (k=32 ref) | 7,077,888 | 1.055000 | 1.050276 | 1.052031 | 1.134491 |
| Ours k=32 | 6,619,136 | 0.946873 | 0.939538 | 0.946838 | 1.198440 |
| DuQuant (k=64 ref) | 7,077,888 | 1.053456 | 1.058511 | 1.058733 | 1.187548 |
| Ours k=64 | 6,684,672 | 0.947638 | 0.949669 | 0.950551 | 1.111277 |
| DuQuant (k=128 ref) | 7,077,888 | 1.055219 | 1.065265 | 1.065277 | 1.247887 |
| Ours k=128 | 6,815,744 | 0.953884 | 0.952134 | 0.954611 | 1.256422 |
| DuQuant (k=256 ref) | 7,077,888 | 1.057313 | 1.058203 | 1.052219 | 1.244879 |
| Ours k=256 | 7,077,888 | 0.943697 | 0.948308 | 0.945815 | 1.304361 |
