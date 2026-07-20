# Post-Quant MoE FFN Group Sweep Summary

Root: `/home/lwk/HeteroQuant_Repro/experiments/post_quant_moe_ffn_group_sweep_20260720_110012`

`main_weight_group_size` changes output-channel scale sharing for the main gate/up/down INT8 weights. `ose_weight_group_size` stayed at the script default unless overridden.

| group | cases | speedup min | speedup median | speedup mean | speedup max | E1 med | E8 med | E16 med | E64 med |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 24 | 1.029763 | 1.140000 | 1.125865 | 1.252606 | 1.140800 | 1.146551 | 1.148487 | 1.033955 |
| 16 | 24 | 1.030791 | 1.144848 | 1.136672 | 1.479428 | 1.150692 | 1.150243 | 1.147692 | 1.032510 |
| 64 | 24 | 1.029651 | 1.136354 | 1.113548 | 1.152989 | 1.141794 | 1.140038 | 1.139775 | 1.032357 |
| 256 | 24 | 1.029387 | 1.139865 | 1.121836 | 1.179506 | 1.142246 | 1.145560 | 1.138991 | 1.032271 |
| 1024 | 24 | 1.029765 | 1.144540 | 1.149398 | 1.655273 | 1.148395 | 1.165267 | 1.144268 | 1.032303 |
| 2048 | 24 | 1.029518 | 1.140212 | 1.115785 | 1.157385 | 1.140674 | 1.145712 | 1.140915 | 1.032515 |

Notes:
- Speedup is `duquant_once_latency / ours_ose_w8a8_latency`; larger is better for ours.
- `group=1024` is per-tensor for OlmoE gate/up but still two scale groups for down; `group=2048` is per-tensor for gate/up/down.
- Max values for group 16 and 1024 include isolated latency spikes in `duquant_once`; median is the safer comparison.
