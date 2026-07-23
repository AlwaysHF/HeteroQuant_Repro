# route_range Cross Ablation: TopK and Group Size

Metric: route_range / routed_range. OSE topK uses smooth_scale. PPL datasets: WikiText2 and C4.

## TopK Sweep at group_size=1

| group_size | topK | WikiText2 PPL | C4 PPL |
|---:|---:|---:|---:|
| 1 | 0 | 7.0522 | 11.2972 |
| 1 | 16 | 6.9504 | 11.1879 |
| 1 | 32 | 6.9304 | 11.1730 |
| 1 | 64 | 6.9024 | 11.1318 |
| 1 | 128 | 6.8968 | 11.1247 |
| 1 | 256 | 6.8808 | 11.1159 |

## Group-Size Sweep at topK=64

| group_size | topK | WikiText2 PPL | C4 PPL |
|---:|---:|---:|---:|
| 1 | 64 | 6.9024 | 11.1318 |
| 16 | 64 | 6.9202 | 11.1631 |
| 32 | 64 | 6.9781 | 11.2203 |
| 64 | 64 | 7.0227 | 11.2929 |
| 128 | 64 | 7.0256 | 11.3025 |
| 256 | 64 | 7.0520 | 11.3482 |
| 512 | 64 | 7.1353 | 11.3840 |
| 1024 | 64 | 7.2155 | 11.4724 |
| 2048 | 64 | 7.2277 | 11.5132 |

## Notes

- Best in the topK sweep is topK=256: WikiText2 6.8808, C4 11.1159.
- Best in the group-size sweep is group_size=1: WikiText2 6.9024, C4 11.1318.
- Larger group sizes monotonically worsen C4 in the completed group-size sweep; WikiText2 also trends worse, with a small local fluctuation around 128/256.
