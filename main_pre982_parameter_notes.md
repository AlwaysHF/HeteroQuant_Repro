# main.py 前 982 行参数、模型、变量整理

本文只整理 `/home/lwk/HeteroQuant_Repro/main.py` 从文件开头到 `if args.smooth:` 附近，也就是主流水线正式进入 Stage 1 之前已经准备好的内容。理解它时可以把 `args` 看成一个贯穿全流程的配置容器，把 `lm` 看成模型、tokenizer、推理接口的包装对象，把 `dataloader` 看成 Stage 1/Stage 2 的校准样本入口。

## 总体分层

前 982 行可以分成五层：

| 层级 | 代码区域 | 作用 |
| --- | --- | --- |
| 运行环境 | imports、线程设置、CUDA benchmark | 准备 PyTorch、随机数、路径、日志、量化相关模块 |
| 工具函数 | cache helper、MoE outlier helper、MoE quant plan parser | 给后面的 Smooth / DuQuant 准备复用逻辑 |
| Stage 封装 | `run_smooth_lm_stage`、`prepare_moe_outlier_scores`、`run_duquant_stage`、`evaluation block` | 把三个主阶段包装成函数，主流程只负责调度 |
| 参数与模型初始化 | `argparse` 到 `logger.info(args_for_log)` | 解析命令行、校验、设随机种子、建目录、加载模型、识别模型家族 |
| 量化参数与校准数据 | `args.*_quant_params` 到 `if args.smooth:` 之前 | 组装量化器配置，必要时加载 calibration dataloader |

## 全局统一使用的参数和变量

这些变量不是某一个阶段私有的，它们决定整次运行的身份、模型、日志、缓存、随机性或设备迁移。

| 名称 | 来源 | 作用 | 主要消费者 |
| --- | --- | --- | --- |
| `args` | `parser.parse_args()` | 全局配置容器，后续还不断追加派生字段 | 所有 stage、`LMClass`、`duquant`、`evaluate`、量化层类 |
| `args.model` | CLI | 模型路径或 HuggingFace 名称 | `LMClass`、校准集/tokenizer、cache key、评测 |
| `args.model_name` | CLI 或由 `args.model` 推导 | 模型简称，也是模型家族判断依据 | `LMClass`、`convert_device`、`duquant` |
| `args.model_name` | 加载模型后推导 | 规范化模型家族名，如 `qwen2_moe`、`olmoe`、`pangumoe` | cache 文件名、评测 testloader cache |
| `args.seed` | CLI | 随机种子，保证 calibration 抽样和初始化可复现 | `random`、`numpy`、`torch`、`get_loaders`、cache key |
| `args.seq_length` | CLI，评测时会被改成 2048 | 校准阶段序列长度；`lm.seqlen` 也会设成这个值 | dataloader、Smooth stats、DuQuant 输入缓存 |
| `args.batch_size` | CLI | 构造 `LMClass` 时保存为 `batch_size_per_gpu` | `LMClass` / lm_eval 接口 |
| `args.cache_dir` | CLI | 存 dataloader、smooth stats、testloader cache | Stage 1、Stage 2 校准数据准备、Stage 3 PPL |
| `args.output_dir` | CLI 后追加模型和 bits 后缀 | 日志输出目录 | logger、Stage 3 logging |
| `args.attn_implementation` | 固定赋值 `"eager"` | 加载模型时指定 attention 实现，同时进入 smooth cache key | `LMClass`、Smooth stats cache |
| `args.quant_method` | 固定赋值 `"duquant"` | 告诉量化器使用 DuQuant 初始化/旋转逻辑 | 各类 `*_quant_params`、`QuantLinear`、`UniformAffineQuantizer` |
| `lm` | `LMClass(args)` | 包装模型、tokenizer、seqlen、lm_eval 所需接口 | Stage 1、Stage 2、Stage 3 |
| `lm.model` | `LMClass` 内部加载 | 真正的 causal LM 模型 | Smooth、DuQuant、Evaluation |
| `lm.tokenizer` | `LMClass` 内部加载 | 文本编码解码 | lm_eval / 数据加载相关 |
| `lm.seqlen` | 先来自模型 config，后设为 `args.seq_length` | 当前运行使用的序列长度 | dataloader、DuQuant 输入张量、Evaluation |
| `logger` | `utils.create_logger(output_dir)` | 统一记录参数、耗时、结果 | 所有 stage |
| `output_dir` | `Path(args.output_dir)` | logger 创建参数 | logging |

## 全局派生和校验变量

这些字段由 CLI 参数推导出来，后续会被多个模块读取。

| 名称 | 产生位置 | 含义 | 使用方式 |
| --- | --- | --- | --- |
| `args.removed_activation_energy_score` | 初始化为空 `{}`；Stage 1 或辅助统计阶段可覆盖 | MoE expert 输入通道的激活能量，用于 `removed_activation_weight_error_mode` outlier score | 量化层选择 outlier 输入列 |
| `args.moe_outlier_scores` | 初始化为空 `{}`；Smooth 可生成 | module-level MoE outlier score | 量化层选择同层共享 top-k outlier 列 |
| `args.moe_quant_plan_dict` | 从 `--moe_quant_plan` JSON 解析 | 对特定 MoE expert projection 指定不同 `abits/wbits/group_size` | `int_olmoe_layer.py`、`int_qwen2_moe_layer.py` |
| `args.expert_ratio` | 先默认 `8/64`，模型加载后可更新 | 每 token 激活专家数 / 总专家数 | 推导 `args.expert_token_num` |
| `args.expert_token_num` | 由 `expert_ratio * seq_length * expert_token_num_ratio` 得到 | 每个 expert 最多收集多少校准 token | DuQuant 的 MoE expert 校准 |
| `args_for_log` | `argparse.Namespace(**vars(args))` | 日志用参数副本；大 plan dict 会被替换成简短字符串 | 只用于日志 |

前置校验包括：

- `moe_outlier_topk >= 0`。
- `moe_outlier_score == "smooth_scale"` 时必须开启 `--smooth`。
- `fast_moe_calib_tokens > 0`。
- `moe_quant_plan` 中每个 entry 必须能得到 `abits` 和 `wbits`，且 `wbits <= abits`。

## Stage 1: Smooth_LM 专属或主要使用

Stage 1 只有在 `args.smooth` 为真时执行。它的核心任务是收集或读取激活统计量，然后调用 `smooth_lm(...)` 修改模型权重分布，让后续量化更容易。

| 参数/变量 | 含义 | 在 Stage 1 中的作用 |
| --- | --- | --- |
| `args.smooth` | 是否启用 Smooth_LM | 控制是否调用 `run_smooth_lm_stage` |
| `args.alpha` | activation scale 与 weight scale 的平衡指数 | 传给 `get_scale` / `smooth_lm` |
| `args.fc1_scale_merge` | MoE fc1/gate/up 平滑尺度的聚合策略 | 决定使用 `max/sum/act_mean/act_p99` 等路径 |
| `args.otsu_ratio` | Otsu/outlier 判断相关比例 | `moe_down_smooth_mode="otsu"` 时参与 down_proj 平滑 |
| `args.otsu_smooth_rate` | Otsu outlier 平滑强度 | 同上 |
| `args.moe_down_smooth_mode` | MoE down_proj 平滑模式 | `"duquant"` 使用 scale，`"otsu"` 使用 outlier mask |
| `args.act_mean_beta` | `act_mean` 平滑时的放大系数 | `build_moe_fc1_smooth_scales` 中将 raw stat 归一化后乘 beta |
| `args.disable_act_stats_cache` | 禁用 Smooth stats cache | 强制不保存/不读 cache |
| `args.refresh_act_stats_cache` | 刷新 Smooth stats cache | 即使 cache 存在也重新计算 |
| `args.moe_outlier_score` | 是否用 smooth scale 作为同层共享 outlier score | `"smooth_scale"` 时由 MoE fc1 smooth scales 展开成 module-level `moe_outlier_scores` |
| `args.moe_outlier_topk` | outlier 列数量 | 若大于 0 且 score 需要激活能量，会触发额外统计 |
| `args.moe_outlier_score` | outlier 分数来源 | `"removed_activation_weight_error_mode"` 时需要 `removed_activation_energy_score` |
| `dataloader` | calibration dataloader | 用来跑模型收集 activation stats |

Stage 1 内部会产生的主要临时统计：

| 变量 | 含义 |
| --- | --- |
| `act_scales` | 每个相关模块输入 activation 的通道尺度 |
| `act_per_channel_scales` | activation 每通道最大/最小值，Otsu 平滑会用 |
| `act_samples` | 每个 MoE expert/模块实际收到多少 token |
| `weight_scores` | 权重侧分数，用于某些 `fc1_scale_merge` 策略 |
| `router_logits` | router 分布统计，用于某些 `fc1_scale_merge` 策略 |
| `moe_act_means` / `moe_act_p99s` | `act_mean` / `act_p99` 策略下的 MoE activation 统计 |
| `removed_activation_energy_score` | expert 输入通道激活平方均值，给 `removed_activation_weight_error_mode` outlier score 用 |

这些统计量的 cache key 由 `model/model_name/calib_dataset/nsamples/seq_length/seed/attn_implementation/stat_name` 共同决定，所以同一模型、同一校准配置下可以复用。

## Stage 2: DuQuant 专属或主要使用

Stage 2 调用 `run_duquant_stage(lm, args, dataloader, logger)`，实际进入 `quantize/duquant.py`。它的核心任务是逐层把原始 decoder layer 替换成量化 layer，收集每层校准输入，初始化 DuQuant 的旋转/排列/scale，然后把权重量化到位。

### Stage 2 主控制参数

| 参数/变量 | 含义 | 用途 |
| --- | --- | --- |
| `args.wbits` | 普通权重量化 bit | 进入 `weight_quant_params` |
| `args.abits` | 普通激活量化 bit | 进入 `act_quant_params`，也给 Q/K/V matmul 量化用 |
| `args.router_wbits` | MoE router 权重量化 bit | 进入 `router_weight_quant_params` |
| `args.router_abits` | MoE router 激活量化 bit | 进入 `router_act_quant_params` |
| `args.symmetric` | 权重量化是否对称 | 普通 weight/router weight/outlier bank 使用 |
| `args.w_dynamic_method` | 普通权重动态量化策略 | 传入 weight quantizer |
| `args.a_dynamic_method` | 激活动态量化策略 | 传入 act/q/k/v quantizer |
| `args.router_w_dynamic_method` | router 权重动态量化策略 | 传入 router weight quantizer |
| `args.group_size` | 权重量化 group size | 普通 weight/router weight |
| `args.act_group_size` | 激活量化 group size | 普通 act/router act |
| `args.swc` | static weight clipping ratio | 权重量化 clipping |
| `args.lac` | activation clipping ratio | 激活量化 clipping |
| `args.block_size` | DuQuant rotation block size | `UniformAffineQuantizer` 中决定旋转/分块规模 |
| `args.max_rotation_step` | rotation 搜索最大步数 | DuQuant rotation |
| `args.permutation_times` | permutation 轮数 | DuQuant permutation |
| `args.scale_search_steps` | weight scale 搜索候选数 | 权重量化 scale search |
| `args.disable_moe_gate_up_duquant_rotation` | 是否关闭 MoE gate/up 的 DuQuant rotation | 传给 MoE MLP 中 gate/up 的 `QuantLinear(rotate=...)` |
| `args.expert_token_num_ratio` | expert 校准 token 数倍率 | 推导 `expert_token_num` |
| `args.expert_token_num` | 每个 expert 的校准 token 上限 | `duquant.py` 收集 expert 输入时截断 |
| `args.fast_moe_down_calibration` | 是否跳过 routed expert-token 收集 | 用 `mlp.gate` tokens 快速校准所有 expert |
| `args.fast_moe_calib_tokens` | 快速 MoE 校准 token 数 | fast path 中选多少 `mlp.gate` token |

### Stage 2 outlier 和异构精度参数

| 参数/变量 | 含义 | 用途 |
| --- | --- | --- |
| `args.moe_outlier_topk` | 每个 MoE expert gate/up 拆出的 outlier 输入列数 | `int_olmoe_layer.py` / `int_qwen2_moe_layer.py` 创建 outlier bank |
| `args.moe_outlier_quant` | outlier bank 的量化模式 | `same/fp16/w16a16/w8a8/w4a4` |
| `args.moe_outlier_score` | outlier 列选择分数 | 可用 `smooth_scale/weight_max/weight_error` |
| `args.removed_activation_energy_score` | 激活能量统计 | `removed_activation_weight_error_mode` 分数需要 |
| `args.moe_outlier_score` | 是否同层共享 outlier score | `"smooth_scale"` 时读取 `moe_outlier_scores` |
| `args.moe_outlier_scores` | Stage 1 生成的层级 score | 供每个 gate_proj/up_proj 按 module name 读取 top-k score |
| `args.moe_quant_plan_dict` | per-module 量化计划 | 覆盖特定 expert `gate_proj/up_proj/down_proj` 的 bits/group size |

### Stage 2 组装出的量化参数字典

这些字典都在 982 行之前创建，但真正被 Stage 2 的 `QuantLinear` / `QuantMatMul` / `UniformAffineQuantizer` 消费。

| 字典 | 目标模块 | 关键字段 |
| --- | --- | --- |
| `args.weight_quant_params` | 普通 Linear 权重默认配置 | `n_bits=wbits`、`per_channel_axes=[0]`、`symmetric`、`group_size`、`swc`、`block_size`、`max_rotation_step`、`permutation_times`、`scale_search_steps` |
| `args.act_quant_params` | 普通 Linear 输入激活默认配置 | `n_bits=abits`、`lac`、`act_group_size`、`a_dynamic_method`、DuQuant rotation 参数 |
| `args.router_weight_quant_params` | MoE router/gate 权重 | `n_bits=router_wbits`、`router_top_k=num_experts_per_tok`、其他类似 weight |
| `args.router_act_quant_params` | MoE router/gate 激活 | `n_bits=router_abits`、其他类似 act |
| `args.q_quant_params` | attention score matmul 中 Q 的量化 | `n_bits=abits`、可 rotation |
| `args.k_quant_params` | attention score matmul 中 K 的量化 | `n_bits=abits` |
| `args.v_quant_params` | attention value matmul 中 V 的量化 | `n_bits=abits` |
| `args.p_quant_params` | softmax probability 的量化配置 | 固定 `n_bits=16`、`metric="fix0to1"`，基本等于不压低精度 |

注意：`duquant.py` 每处理一层时还会把通用字典复制成模块级字典：

- `args.q_weight_quant_params` / `args.q_act_quant_params`
- `args.k_weight_quant_params` / `args.k_act_quant_params`
- `args.v_weight_quant_params` / `args.v_act_quant_params`
- `args.gate_weight_quant_params` / `args.gate_act_quant_params`
- `args.up_weight_quant_params` / `args.up_act_quant_params`
- `args.down_weight_quant_params` / `args.down_act_quant_params`
- `args.o_weight_quant_params` / `args.o_act_quant_params`

这些复制是在 Stage 2 内部做的，所以 982 行之前还看不到它们，但量化 layer 构造时会读取它们。

## Stage 3: Evaluation 专属或主要使用

Stage 3 调用 `evaluation block(...)`，内部进入 `utils.evaluate(lm, args, logger)`。它不需要 calibration dataloader，而是根据评测参数重新加载测试集或调用 lm-eval。

| 参数/变量 | 含义 | 用途 |
| --- | --- | --- |
| `args.eval_ppl` | 是否评测 perplexity | 为真时读取 `test_dataset` 并计算 PPL |
| `args.test_dataset` | PPL 测试集，默认 `wikitext2` | 可逗号分隔多个数据集 |
| `args.tasks` | lm-eval 任务名字符串 | 非空时调用 `evaluator.simple_evaluate` |
| `args.num_fewshot` | few-shot 数量 | 传给 lm-eval |
| `args.cache_dir` | testloader cache 目录 | PPL 数据缓存 |
| `args.model_name` | testloader cache 文件名的一部分 | 区分模型家族 |
| `args.seed` | testloader 构造随机种子 | `get_loaders` |
| `lm` | 量化后的模型包装对象 | 被迁移到设备并执行评测 |

重要细节：`evaluate` 开头会把 `lm.seqlen` 和 `args.seq_length` 都改成 `2048`，所以 Stage 3 的推理长度固定为 2048，不沿用校准阶段的 4096。

## Calibration dataset setup 里的变量

这一段处于 Stage 1/2 之前，但只在需要量化时执行。

| 变量 | 含义 | 后续用途 |
| --- | --- | --- |
| `dataloader` | 校准样本列表/迭代器 | Stage 1 收集 activation stats；Stage 2 捕获每层输入 |
| `cache_dataloader` | 校准 dataloader 缓存路径 | 避免重复 `get_loaders` |
| `dataloader_start` / `dataloader_seconds` | 加载校准集耗时 | logging |
| `calibration_start` | Stage 1 + Stage 2 总校准计时起点 | logging |

触发条件是：

```python
if args.wbits < 16 or args.abits < 16:
```

也就是说，只要权重或激活任一侧低于 16 bit，就进入量化准备。如果两者都是 16 bit，则跳过 Stage 1/2，只做 Stage 3 evaluation。

## 模型对象准备

`LMClass(args)` 做了几件关键事情：

- 保存 `args`、`model_name`、`batch_size_per_gpu`。
- 根据 `args.model_name` 选择模型加载路径：
  - `olmoe`：加载本仓库 `models.olmoe.modeling_olmoe.OlmoeForCausalLM`。
  - `qwen`：加载 Transformers 的 `Qwen2MoeForCausalLM`。
  - `pangumoe`：加载本仓库 Pangu MoE 模型，并把 `args.block_size` 改成 `64`。
  - 其他：走通用 `AutoModelForCausalLM`。
- 加载 tokenizer 和 generation config。
- 设置 `self.seqlen = self.model.config.max_position_embeddings`，随后 `main.py` 又改成 `args.seq_length`。
- 设置模型 eval。

`main.py` 在模型加载后继续做：

- 根据 `args.model_name` 再设 `args.model_name`。
- 如果模型 config 有 `num_experts` 和 `num_experts_per_tok`，更新 `expert_ratio` 与 `expert_token_num`。
- `lm.model.eval()`。
- 所有参数 `requires_grad = False`，说明这是后训练量化/评测流程，不做反向训练。

## 容易混淆的边界

- `982` 行之前不是“只解析参数”。它已经完成模型加载、日志创建、量化器配置字典组装，并在需要量化时加载了 calibration dataloader。
- `if args.smooth:` 附近是 Stage 1 的入口。严格说，`if args.wbits < 16 or args.abits < 16:` 开始后就已经进入量化流程的外层。
- `args.weight_quant_params` / `args.act_quant_params` 是默认模板；真正每层用时，Stage 2 会复制成 `q/k/v/gate/up/down/o` 的专属参数。
- `args.seq_length` 在校准阶段和评测阶段含义不完全一样：校准阶段来自 CLI，评测阶段被固定改为 2048。
- `moe_quant_plan_dict` 不是 Smooth 参数，它主要影响 Stage 2 里 MoE expert projection 的异构量化精度。
- `removed_activation_energy_score` 如果 `--smooth` 开启，可能在 Stage 1 顺便收集并按 smooth scale 修正；现在 baseline 分数只保留 `weight_max/weight_error`，并统一在 DuQuant 前通过 `prepare_moe_outlier_scores` 生成。

## 一句话心智模型

前 982 行的主线是：

```text
命令行参数 -> 校验/派生参数 -> 建日志和缓存目录 -> 加载模型并识别模型家族
-> 组装量化器配置字典 -> 准备 calibration dataloader
-> 等待进入 Stage 1 Smooth、Stage 2 DuQuant、Stage 3 Evaluation
```

