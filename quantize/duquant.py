import torch
import torch.nn as nn
from models.int_olmoe_layer import QuantOlmoeDecoderLayer
from models.int_qwen2_moe_layer import QuantQwen2MoeDecoderLayer
from quantize.int_linear import QuantLinear
import copy
import gc
import ctypes
import os
from quantize.utils import *
import functools
from collections import defaultdict


_LIBC = None


def trim_cpu_memory():
    """Return freed CPU heap pages to the OS when running under a tight cgroup."""
    global _LIBC
    gc.collect()
    if os.name != "posix":
        return
    try:
        if _LIBC is None:
            _LIBC = ctypes.CDLL("libc.so.6")
        _LIBC.malloc_trim(0)
    except Exception:
        pass


def get_named_linears(module):
    return {name: m for name, m in module.named_modules() if isinstance(m, (QuantLinear, nn.Linear)) and ("mlp.gate" in name or "up_proj" in name or "block_sparse_moe.gate" in name or "w3" in name) and "gate_proj" not in name}


def stat_tensor(name, tensor, act_samples):
    if ("mlp.gate" in name or "up_proj" in name or "block_sparse_moe.gate" in name or "w3" in name) and "gate_proj" not in name:
        tensor = tensor.view(-1, tensor.shape[-1])
        num = tensor.shape[0]
        if name in act_samples:
            act_samples[name] += num
        else:
            act_samples[name] = num

# get input features of all linear layers for linear layer back to fp16
def cache_input_hook(m, x, y, name, feat_dict, act_samples, expert_token_num):
    x = x[0]
    x = x.detach().cpu()
    update_tag = True
    if (expert_token_num > 0 and name in act_samples and act_samples[name] >= expert_token_num):
        update_tag = False
    if update_tag:
        feat_dict[name].append(x)
        stat_tensor(name, x, act_samples)


used_device = "cuda"

def duquant(
    lm,
    args,
    dataloader,
    logger=None,
):
    logger.info("Starting ...")
    
    # move embedding layer and first layer to target device
    model = lm.model
    dev = used_device

    logger.info(f"dev : {dev}, lm.model.device : {lm.model.device}")
    use_cache = model.config.use_cache
    model.config.use_cache = False
    is_llama = False
    expert_num = int(getattr(model.config, "num_experts", 64))
    if "olmoe" in args.model_name.lower():
        is_llama = True
        is_MOE = True
        args.sp_model_name = "olmoe"
        layers = model.model.layers
        model.model.embed_tokens = model.model.embed_tokens.to(dev)
        model.model.norm = model.model.norm.to(dev)
        DecoderLayer = QuantOlmoeDecoderLayer
        layer_name_prefix = "model.layers"
    elif "qwen" in args.model_name.lower():
        is_llama = True
        is_MOE = True
        args.sp_model_name = "qwen2_moe"
        layers = model.model.layers
        model.model.embed_tokens = model.model.embed_tokens.to(dev)
        model.model.norm = model.model.norm.to(dev)
        DecoderLayer = QuantQwen2MoeDecoderLayer
        layer_name_prefix = "model.layers"
    else:
        raise ValueError("Only support for olmoe/qwen2_moe now")
    
    
    layers[0] = layers[0].to(dev)
    dtype = torch.float16
    inps = torch.zeros(
        (args.nsamples, lm.seqlen, model.config.hidden_size), dtype=dtype, device=dev
    )
    cache = {"i": 0}

    # catch the first layer input
    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
            self.is_llama = False

        def forward(self, inp, **kwargs):
            inps[cache["i"]] = inp
            cache["i"] += 1
            cache["attention_mask"] = kwargs["attention_mask"]
            if self.is_llama:
                cache["position_ids"] = kwargs["position_ids"]
            raise ValueError

    layers[0] = Catcher(layers[0])
    layers[0].is_llama = is_llama
    input_ids = []

    with torch.no_grad():
        for batch in dataloader:
            if cache["i"] >= args.nsamples:
                break
            try:
                input_ids.append(batch[0])
                model(batch[0].to(dev))
            except ValueError:
                pass
    
    # move embedding layer and first layer to cpu
    layers[0] = layers[0].module
    layers[0] = layers[0].cpu()
    if "olmoe" in args.model_name.lower() or "qwen" in args.model_name.lower():
        model.model.embed_tokens = model.model.embed_tokens.cpu()
        model.model.norm = model.model.norm.cpu()
    else:
        raise ValueError("Only support for olmoe/qwen2_moe now")
    torch.cuda.empty_cache()
    
    rotate_inps = copy.copy(inps).mean(dim=0).unsqueeze(0)
    fp_inps = copy.deepcopy(inps)
    attention_mask = cache["attention_mask"]
    if attention_mask is None:
        logger.info(
            "No attention mask caught from the first layer."
            " Seems that model's attention works without a mask."
        )
    if is_llama:
        position_ids = cache["position_ids"]
    else:
        position_ids = None

    for i in range(len(layers)):
        dev = used_device

        for name in ['q', 'k', 'v', 'gate', 'up', 'down', 'o']:
            exec(f"args.{name}_weight_quant_params = copy.copy(args.weight_quant_params)")
            exec(f"args.{name}_act_quant_params = copy.copy(args.act_quant_params)")
        args.q_quant_params = copy.copy(args.act_quant_params)
        args.k_quant_params = copy.copy(args.act_quant_params)

        logger.info(f"=== Start quantize layer {i} ===")
        layer = layers[i]
        if "moe" in args.model_name.lower():
            qlayer = DecoderLayer(lm.model.config, i, layer, args)
        else:
            qlayer = DecoderLayer(lm.model.config, layer, args)

        # print(qlayer)
        qlayer = qlayer.to(dev)

        set_init_duquant_params_state(qlayer, True)
        rotate_inps = copy.copy(fp_inps).mean(dim=0).unsqueeze(0)

        fp_inps_back = copy.copy(fp_inps)


        set_quant_state(qlayer, weight_quant=False, act_quant=False)
        with torch.no_grad():
            with torch.cuda.amp.autocast():
                for j in range(args.nsamples):
                    fp_inps[j] = qlayer(
                        fp_inps[j].unsqueeze(0).to(dev),
                        attention_mask=attention_mask.to(dev),
                        position_ids=position_ids.to(dev),
                    )[0]

        set_quant_state(qlayer, weight_quant=False, act_quant=True)  # weight will be manually quantized before forward

        qlayer.half()

        logger.info("duquant begin")
        input_feat = defaultdict(list)
        fast_moe_down_calibration = is_MOE and args.fast_moe_down_calibration
        set_init_duquant_params_state(qlayer, False)
        set_quant_state(qlayer, weight_quant=True, act_quant=True)
        with torch.no_grad():
            with torch.cuda.amp.autocast():
                set_registered_x_none(qlayer)
                
                if is_MOE: # stage1: calibrate the QKV layers
                    moe_ffn_modules = [qlayer.mlp.experts]
                    if hasattr(qlayer.mlp, "shared_expert"):
                        moe_ffn_modules.append(qlayer.mlp.shared_expert)
                    for moe_ffn_module in moe_ffn_modules:
                        set_init_duquant_params_state(moe_ffn_module, True) # experts: fp forward, no init_duquant_params
                        set_quant_state(moe_ffn_module, weight_quant=False, act_quant=False)
                        set_calibration_state(moe_ffn_module, False)

                ### cache input sample of experts with quantized block
                named_linears = get_named_linears(qlayer)
                if fast_moe_down_calibration:
                    named_linears = {name: module for name, module in named_linears.items() if name == "mlp.gate" or name.endswith("block_sparse_moe.gate")}
                    logger.info(
                        f"fast MoE down calibration enabled: skip routed expert-token collection, "
                        f"cache only mlp.gate tokens, fast_moe_calib_tokens={args.fast_moe_calib_tokens}"
                    )
                actual_act_samples = {}
                handles = []
                for name in named_linears:
                    handles.append(named_linears[name].register_forward_hook(functools.partial(cache_input_hook, name=name, feat_dict=input_feat, act_samples=actual_act_samples, expert_token_num=args.expert_token_num)))
                
                    
                for k in range(rotate_inps.shape[0]): # stage1: calibrate the QKV layers for 1 time
                    qlayer(rotate_inps[k].unsqueeze(0).to(dev), attention_mask=attention_mask.to(dev),position_ids=position_ids.to(dev))

                set_calibration_state(qlayer, False) # stop weight calibration for all layers
                if args.expert_token_num > 0 and not fast_moe_down_calibration: # collect more calibration data for experts with quantized QKV layers
                    for k in range(fp_inps_back.shape[0]):
                        qlayer(fp_inps_back[k].unsqueeze(0).to(dev), attention_mask=attention_mask.to(dev),position_ids=position_ids.to(dev))

                for h in handles:
                    h.remove()
                input_feat = {k: torch.cat(v, dim=0) for k, v in input_feat.items()}
                
                if is_MOE:
                    logger.info("========================")
                    for k, v in input_feat.items():
                        logger.info(f"k: {k}, v.shape: {v.shape}, v.dtype: {v.dtype}, v.device: {v.device}") # k: mlp.gate, v.shape: torch.Size([2048, 2048])

                if is_MOE:
                    moe_ffn_modules = [qlayer.mlp.experts]
                    if hasattr(qlayer.mlp, "shared_expert"):
                        moe_ffn_modules.append(qlayer.mlp.shared_expert)
                    for moe_ffn_module in moe_ffn_modules:
                        set_init_duquant_params_state(moe_ffn_module, False) # quant forward, do init_duquant_params
                        set_quant_state(moe_ffn_module, weight_quant=True, act_quant=True)
                        set_calibration_state(moe_ffn_module, True)

                    if fast_moe_down_calibration:
                        if "mlp.gate" not in input_feat:
                            raise KeyError("missing mlp.gate input features for fast MoE down calibration")
                        select_token_num = min(input_feat["mlp.gate"].shape[0], args.fast_moe_calib_tokens)
                        moe_calib_tokens = input_feat["mlp.gate"][:select_token_num].to(dev)
                        logger.info(
                            f"fast stage3: calibrate all MoE experts with unrouted mlp.gate tokens, "
                            f"tokens={select_token_num}"
                        )
                        for k in range(expert_num):
                            logger.info(f"fast calibration qlayer.mlp.experts[{k}] with unrouted mlp.gate tokens ...")
                            if hasattr(qlayer.mlp, "forward_expert"):
                                qlayer.mlp.forward_expert(k, moe_calib_tokens)
                            else:
                                qlayer.mlp.experts[k](moe_calib_tokens)
                        if hasattr(qlayer.mlp, "forward_shared_expert"):
                            logger.info("fast calibration qlayer.mlp.shared_expert with unrouted mlp.gate tokens ...")
                            qlayer.mlp.forward_shared_expert(moe_calib_tokens)
                    else:
                        # stage2: calibrate the activated experts with special tokens for 1 time
                        logger.info(f"calibrate the activated experts with special tokens")
                        for k in range(expert_num):
                            if f"mlp.experts.{k}.up_proj" in input_feat:
                                logger.info(f"calibration qlayer.mlp.experts[{k}] with input_feat ...")
                                if args.expert_token_num > 0: # 4096
                                    select_expert_token_num = min(input_feat[f"mlp.experts.{k}.up_proj"].shape[0], args.expert_token_num)
                                else: # 0
                                    select_expert_token_num = input_feat[f"mlp.experts.{k}.up_proj"].shape[0]
                                if hasattr(qlayer.mlp, "forward_expert"):
                                    qlayer.mlp.forward_expert(k, input_feat[f"mlp.experts.{k}.up_proj"][:select_expert_token_num].to(dev))
                                else:
                                    qlayer.mlp.experts[k](input_feat[f"mlp.experts.{k}.up_proj"][:select_expert_token_num].to(dev))

                        if hasattr(qlayer.mlp, "forward_shared_expert") and "mlp.shared_expert.up_proj" in input_feat:
                            logger.info("calibration qlayer.mlp.shared_expert with input_feat ...")
                            if args.expert_token_num > 0:
                                select_shared_token_num = min(input_feat["mlp.shared_expert.up_proj"].shape[0], args.expert_token_num)
                            else:
                                select_shared_token_num = input_feat["mlp.shared_expert.up_proj"].shape[0]
                            qlayer.mlp.forward_shared_expert(input_feat["mlp.shared_expert.up_proj"][:select_shared_token_num].to(dev))

                        # stage3: recalibration the left experts with the whole tokens for 1 time
                        logger.info(f"recalibration the left experts with the whole tokens")
                        for k in range(expert_num):
                            gate_act_quantizer = qlayer.mlp.experts[k].gate_proj.act_quantizer
                            if gate_act_quantizer.rotate is True:
                                need_recalibration = gate_act_quantizer.permutation_list == []
                            else:
                                need_recalibration = gate_act_quantizer.scales is None
                            if need_recalibration: # not calibrated yet
                                logger.info(f"recalibration qlayer.mlp.experts[{k}] with input_feat ...")
                                if "olmoe" in args.model_name.lower() or "qwen" in args.model_name.lower():
                                    if hasattr(qlayer.mlp, "forward_expert"):
                                        qlayer.mlp.forward_expert(k, input_feat["mlp.gate"][:args.seq_length].to(dev))
                                    else:
                                        qlayer.mlp.experts[k](input_feat["mlp.gate"][:args.seq_length].to(dev))
                        if hasattr(qlayer.mlp, "forward_shared_expert"):
                            gate_act_quantizer = qlayer.mlp.shared_expert.gate_proj.act_quantizer
                            if gate_act_quantizer.rotate is True:
                                need_recalibration = gate_act_quantizer.permutation_list == []
                            else:
                                need_recalibration = gate_act_quantizer.scales is None
                            if need_recalibration:
                                logger.info("recalibration qlayer.mlp.shared_expert with unrouted mlp.gate tokens ...")
                                qlayer.mlp.forward_shared_expert(input_feat["mlp.gate"][:args.seq_length].to(dev))

        qlayer.register_duquant_params()
        set_init_duquant_params_state(qlayer, True) # quant forward, init_duquant_params done
        logger.info("duquant done")
        del input_feat

        # Drop the original layer from the model before quant_inplace creates
        # dequantized weight buffers. This avoids a full-layer CPU memory spike
        # near the end of large MoE models such as Qwen1.5-MoE-A2.7B.
        layers[i] = nn.Identity()
        del layer
        torch.cuda.empty_cache()
        trim_cpu_memory()

        qlayer.half()
        set_calibration_state(qlayer, False)
        quant_inplace(qlayer)
        set_quant_state(qlayer, weight_quant=False, act_quant=True)
        torch.cuda.empty_cache()
        trim_cpu_memory()
    
        layers[i] = qlayer.to("cpu")
        torch.cuda.empty_cache()
        trim_cpu_memory()

    del inps
    del fp_inps
    del fp_inps_back
    torch.cuda.empty_cache()
    trim_cpu_memory()
    model.config.use_cache = use_cache

    logger.info(f"{args.output_dir.split('/')[-1]}")


    return model

