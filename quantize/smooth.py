import torch
import torch.nn as nn

from models.olmoe.modeling_olmoe import OlmoeDecoderLayer
from models.pangu_moe.modeling_pangu_moe import PanguProMoEDecoderLayer
try:
    from transformers.models.qwen2_moe.modeling_qwen2_moe import Qwen2MoeDecoderLayer
except Exception:
    class Qwen2MoeDecoderLayer(nn.Module):
        pass

import logging

from quantize.outlier_finder import otsu

@torch.no_grad()
def calcu_outlier_mask(per_channel_max, per_channel_min, dev, ratio=0.65, smooth_rate=0.7, alpha=1.5):
    total_channel = per_channel_max.size()[0]
    per_channel_max = per_channel_max.to(dev)
    per_channel_min = per_channel_min.to(dev)

    outlier_mask = torch.ones(total_channel).to(dev)
    outlier_mask_pos = torch.ones(total_channel).to(dev)
    outlier_mask_neg = torch.ones(total_channel).to(dev)

    per_channel_min = per_channel_min.to(torch.float32)
    q1_neg = torch.quantile(per_channel_min, 1-ratio)
    q3_neg = torch.quantile(per_channel_min, ratio)
    IQR_neg = q3_neg - q1_neg
    outlier_neg = q1_neg - alpha*IQR_neg

    per_channel_max = per_channel_max.to(torch.float32)
    q1_pos = torch.quantile(per_channel_max, 1-ratio)
    q3_pos = torch.quantile(per_channel_max, ratio)
    IQR_pos = q3_pos -q1_pos
    outlier_pos = q3_pos+ alpha*IQR_pos

    if per_channel_max[per_channel_max > outlier_pos].numel()>1:
        logging.info("pos first outlier={} with outlier_pos={}".format(per_channel_max[per_channel_max > outlier_pos].numel(), outlier_pos))
        _,pos_outliers,outlier_pos=otsu(per_channel_max[per_channel_max > outlier_pos])
        logging.info("pos later outlier={} with outlier_pos={}".format(per_channel_max[per_channel_max > outlier_pos].numel(), outlier_pos))

    if per_channel_min[per_channel_min < outlier_neg].numel()>1:
        logging.info("neg first outlier={} with outlier_neg={}".format(per_channel_min[per_channel_min < outlier_neg].numel(), outlier_neg))
        _,neg_outliers,outlier_neg = otsu(per_channel_min[per_channel_min < outlier_neg],pos=False)
        logging.info("neg later outlier={} with outlier_neg={}".format(per_channel_min[per_channel_min < outlier_neg].numel(), outlier_neg))

    outlier_mask_pos[per_channel_max > outlier_pos] = 0 # mask outlier value
    outlier_mask_neg[per_channel_min < outlier_neg] = 0 # mask outlier value

    max_value = torch.max(per_channel_max[outlier_mask_pos == 1]) # get normal max value
    min_value = torch.min(per_channel_min[outlier_mask_neg == 1]) # get normal min value

    outlier_mask_pos[outlier_mask_pos == 0] = (per_channel_max.to(torch.float32)[outlier_mask_pos == 0]/max_value)**smooth_rate
    outlier_mask_neg[outlier_mask_neg == 0] = (per_channel_min.to(torch.float32)[outlier_mask_neg == 0]/min_value)**smooth_rate

    if min_value != 0:
        outlier_mask = torch.max(outlier_mask_pos,outlier_mask_neg)
    else:
        outlier_mask = outlier_mask_pos
    outlier_mask[outlier_mask==0] = 1
    logging.info("outlier channel:{}".format(torch.sum(outlier_mask!=1)))

    outlier_mask_pos = outlier_mask_pos.to("cpu")
    outlier_mask_neg = outlier_mask_neg.to("cpu")
    per_channel_max = per_channel_max.to("cpu")
    per_channel_min = per_channel_min.to("cpu")
    
    del outlier_mask_pos
    del outlier_mask_neg
    del per_channel_max
    del per_channel_min
    return outlier_mask


@torch.no_grad()
def get_scale(fcs, act_scales, alpha=0.5):
    if not isinstance(fcs, list):
        fcs = [fcs]
    device, dtype = fcs[0].weight.device, fcs[0].weight.dtype
    act_scales = act_scales.to(device=device, dtype=dtype).clamp(min=1e-5)
    weight_scales = torch.cat([fc.weight.abs().max(dim=0, keepdim=True)[0] for fc in fcs], dim=0)
    weight_scales = weight_scales.max(dim=0)[0].clamp(min=1e-5)
    scales = (act_scales.pow(alpha) / weight_scales.pow(1-alpha)).clamp(min=1e-5)
    return scales



@torch.no_grad()
def build_moe_fc1_smooth_scales(moe_act_stats, act_mean_beta=2.0, model=None):
    smooth_scales = {}
    beta = float(act_mean_beta)
    modules = dict(model.named_modules()) if model is not None else {}
    for stat_key, stat_value in moe_act_stats.items():
        if ".mlp.gate" not in stat_key:
            continue
        ref_module = modules.get(stat_key)
        if ref_module is not None and hasattr(ref_module, "weight"):
            scale = stat_value.to(
                device=ref_module.weight.device,
                dtype=ref_module.weight.dtype,
            ).flatten().clamp(min=1e-5)
        else:
            scale = stat_value.detach().float().cpu().flatten().clamp(min=1e-5)
        scale = (scale / scale.mean().clamp(min=1e-5) * beta).clamp(min=1e-5)
        smooth_scales[stat_key] = scale.detach().cpu()
    return smooth_scales


def _get_moe_fc1_smooth_scale(moe_fc1_smooth_scales, sp_name, ref_module):
    if moe_fc1_smooth_scales is None or sp_name not in moe_fc1_smooth_scales:
        raise KeyError(f"missing MoE fc1 smooth scale for {sp_name}")
    return moe_fc1_smooth_scales[sp_name].to(
        device=ref_module.weight.device,
        dtype=ref_module.weight.dtype,
    )


@torch.no_grad()
def get_moe_down_smooth_scale(down_proj, scales, act_per_channel_scales, sp_name, dev, alpha, otsu_ratio, otsu_smooth_rate, mode="duquant"):
    if mode == "duquant":
        return get_scale(down_proj, scales[sp_name], alpha)
    if mode == "otsu":
        return calcu_outlier_mask(
            act_per_channel_scales[0][sp_name],
            act_per_channel_scales[1][sp_name],
            dev,
            ratio=otsu_ratio,
            smooth_rate=otsu_smooth_rate,
        )
    raise ValueError(f"unknown moe_down_smooth_mode: {mode}")


@torch.no_grad()
def smooth_ln_fcs(ln, fcs, scales):
    if not isinstance(fcs, list):
        fcs = [fcs]

    device, dtype = fcs[0].weight.device, fcs[0].weight.dtype
    scales = scales.to(device).to(dtype)
    
    ln.weight.div_(scales)
    if hasattr(ln, 'bias') and ln.bias is not None:
        ln.bias.div_(scales)

    for fc in fcs:
        fc.weight.mul_(scales.view(1, -1))

    for p in ln.parameters():
        assert torch.isnan(p).sum() == 0
    for fc in fcs:
        for p in fc.parameters():
            assert torch.isnan(p).sum() == 0


@torch.no_grad()
def scale_fc_fc(fc1, fc2, scales):
    assert isinstance(fc1, nn.Linear)
    assert isinstance(fc2, nn.Linear)
    
    device, dtype = fc2.weight.device, fc2.weight.dtype
    scales = scales.to(device).to(dtype)
    
    fc1.weight[-scales.size(0):].div_(scales.view(-1, 1))
    if fc1.bias is not None:
        fc1.bias.div_(scales.view(-1))

    fc2.weight.mul_(scales.view(1, -1))

    for p in fc1.parameters():
        assert torch.isnan(p).sum() == 0
    for p in fc2.parameters():
        assert torch.isnan(p).sum() == 0


@torch.no_grad()
def smooth_lm(
    model,
    scales,
    act_per_channel_scales,
    act_samples,
    weight_scores,
    router_logits,
    fc1_scale_merge="act_mean",
    alpha=0.6,
    otsu_ratio=0.8,
    otsu_smooth_rate=0.7,
    moe_down_smooth_mode="duquant",
    moe_fc1_smooth_scales=None,
    logger=None,
):
    if fc1_scale_merge not in ("act_mean", "act_p99"):
        raise ValueError(
            f"unsupported fc1_scale_merge={fc1_scale_merge!r}; "
            "current MoE fc1 smoothing expects prebuilt act_mean/act_p99 scales"
        )

    for name, module in model.named_modules():
        if isinstance(module, (OlmoeDecoderLayer)):
            expert_num = 64
            select_expert_num = 8
            share_expert_num = 0
            dev = module.self_attn.q_proj.weight.device
            logger.info(f"[smooth_lm] name: {name}")
            
            logger.info("smooth qkv")
            attn_ln = module.input_layernorm
            qkv = [module.self_attn.q_proj, module.self_attn.k_proj, module.self_attn.v_proj]
            sp_name = name + '.self_attn.q_proj'
            logger.info(f"[smooth_lm] sp_name: {sp_name}")
            logger.info("get_scale")
            sm_scales = get_scale(qkv, scales[sp_name], alpha)
            logger.info("scale={},max={},min={},scale.shape: {}".format(sm_scales,sm_scales.max(), sm_scales.min(), sm_scales.shape))
            smooth_ln_fcs(attn_ln, qkv, sm_scales)

            if module.self_attn.v_proj.weight.shape == module.self_attn.o_proj.weight.shape:
                logging.info("smooth vo")
                prev_op = module.self_attn.v_proj
                layers = [module.self_attn.o_proj]
                sp_name = name + '.self_attn.o_proj'
                logger.info(f"[smooth_lm] sp_name: {sp_name}")
                logger.info("get_scale")
                sm_scales = get_scale(layers[0], scales[sp_name], alpha)
                logger.info("scale={},max={},min={},scale.shape: {}".format(sm_scales,sm_scales.max(), sm_scales.min(), sm_scales.shape))
                scale_fc_fc(prev_op, layers[0], sm_scales)
            

            logging.info(f"smooth mlp.gate, up_proj/gate_proj in mlp.experts")
            ffn_ln = module.post_attention_layernorm
            fc1 = [module.mlp.gate] + [module.mlp.experts[i].gate_proj for i in range(expert_num)] + [module.mlp.experts[i].up_proj for i in range(expert_num)]
            sp_name = name + '.mlp.gate'
            logger.info(f"[smooth_lm] sp_name: {sp_name}")
            logger.info(f"use prebuilt MoE fc1 smooth scale from {fc1_scale_merge}")
            sm_scales = _get_moe_fc1_smooth_scale(moe_fc1_smooth_scales, sp_name, module.mlp.gate)
            logger.info("scale={},max={},min={},scale.shape: {}".format(sm_scales,sm_scales.max(), sm_scales.min(), sm_scales.shape))
            smooth_ln_fcs(ffn_ln, fc1, sm_scales)

            for i in range(expert_num):
                logging.info(f"smooth down_proj in mlp.experts.{i}")
                prev_op = module.mlp.experts[i].up_proj
                layers = [module.mlp.experts[i].down_proj]
                sp_name = name + f'.mlp.experts.{i}.down_proj'
                if sp_name in scales.keys():
                    logger.info(f"[smooth_lm] sp_name: {sp_name}")
                    logger.info(f"get_moe_down_smooth_scale mode={moe_down_smooth_mode}")
                    sm_scales = get_moe_down_smooth_scale(layers[0], scales, act_per_channel_scales, sp_name, dev, alpha, otsu_ratio, otsu_smooth_rate, mode=moe_down_smooth_mode)
                    logger.info("scale={},max={},min={},scale.shape: {}".format(sm_scales,sm_scales.max(), sm_scales.min(), sm_scales.shape))
                    scale_fc_fc(prev_op, layers[0], sm_scales)

        elif isinstance(module, (Qwen2MoeDecoderLayer)):
            expert_num = module.mlp.num_experts
            select_expert_num = module.mlp.top_k
            share_expert_num = 1
            dev = module.self_attn.q_proj.weight.device
            logger.info(f"[smooth_lm] name: {name}")

            logger.info("smooth qkv")
            attn_ln = module.input_layernorm
            qkv = [module.self_attn.q_proj, module.self_attn.k_proj, module.self_attn.v_proj]
            sp_name = name + '.self_attn.q_proj'
            logger.info(f"[smooth_lm] sp_name: {sp_name}")
            sm_scales = get_scale(qkv, scales[sp_name], alpha)
            logger.info("scale={},max={},min={},scale.shape: {}".format(sm_scales, sm_scales.max(), sm_scales.min(), sm_scales.shape))
            smooth_ln_fcs(attn_ln, qkv, sm_scales)

            if module.self_attn.v_proj.weight.shape == module.self_attn.o_proj.weight.shape:
                logger.info("smooth vo")
                prev_op = module.self_attn.v_proj
                layers = [module.self_attn.o_proj]
                sp_name = name + '.self_attn.o_proj'
                logger.info(f"[smooth_lm] sp_name: {sp_name}")
                sm_scales = get_scale(layers[0], scales[sp_name], alpha)
                logger.info("scale={},max={},min={},scale.shape: {}".format(sm_scales, sm_scales.max(), sm_scales.min(), sm_scales.shape))
                scale_fc_fc(prev_op, layers[0], sm_scales)

            logger.info("smooth qwen2_moe mlp.gate/shared_expert_gate and expert/shared gate/up")
            ffn_ln = module.post_attention_layernorm
            fc1 = [module.mlp.gate, module.mlp.shared_expert_gate]
            fc1 += [module.mlp.experts[i].gate_proj for i in range(expert_num)]
            fc1 += [module.mlp.experts[i].up_proj for i in range(expert_num)]
            fc1 += [module.mlp.shared_expert.gate_proj, module.mlp.shared_expert.up_proj]
            sp_name = name + '.mlp.gate'
            logger.info(f"[smooth_lm] sp_name: {sp_name}")
            logger.info(f"use prebuilt MoE fc1 smooth scale from {fc1_scale_merge}")
            sm_scales = _get_moe_fc1_smooth_scale(moe_fc1_smooth_scales, sp_name, module.mlp.gate)
            logger.info("scale={},max={},min={},scale.shape: {}".format(sm_scales, sm_scales.max(), sm_scales.min(), sm_scales.shape))
            smooth_ln_fcs(ffn_ln, fc1, sm_scales)

            for i in range(expert_num):
                logger.info(f"smooth down_proj in mlp.experts.{i}")
                prev_op = module.mlp.experts[i].up_proj
                layers = [module.mlp.experts[i].down_proj]
                sp_name = name + f'.mlp.experts.{i}.down_proj'
                if sp_name in scales.keys():
                    logger.info(f"[smooth_lm] sp_name: {sp_name}")
                    logger.info(f"get_moe_down_smooth_scale mode={moe_down_smooth_mode}")
                    sm_scales = get_moe_down_smooth_scale(layers[0], scales, act_per_channel_scales, sp_name, dev, alpha, otsu_ratio, otsu_smooth_rate, mode=moe_down_smooth_mode)
                    logger.info("scale={},max={},min={},scale.shape: {}".format(sm_scales, sm_scales.max(), sm_scales.min(), sm_scales.shape))
                    scale_fc_fc(prev_op, layers[0], sm_scales)

            prev_op = module.mlp.shared_expert.up_proj
            layers = [module.mlp.shared_expert.down_proj]
            sp_name = name + '.mlp.shared_expert.down_proj'
            if sp_name in scales.keys():
                logger.info(f"[smooth_lm] sp_name: {sp_name}")
                logger.info(f"get_moe_down_smooth_scale mode={moe_down_smooth_mode}")
                sm_scales = get_moe_down_smooth_scale(layers[0], scales, act_per_channel_scales, sp_name, dev, alpha, otsu_ratio, otsu_smooth_rate, mode=moe_down_smooth_mode)
                logger.info("scale={},max={},min={},scale.shape: {}".format(sm_scales, sm_scales.max(), sm_scales.min(), sm_scales.shape))
                scale_fc_fc(prev_op, layers[0], sm_scales)

        elif isinstance(module, (PanguProMoEDecoderLayer)):
            expert_num = 64
            select_expert_num = 8
            share_expert_num = 1
            dev = module.self_attn.q_proj.weight.device
            logger.info(f"[smooth_lm] name: {name}")
            
            logger.info("smooth qkv")
            attn_ln = module.input_layernorm
            qkv = [module.self_attn.q_proj, module.self_attn.k_proj, module.self_attn.v_proj]
            sp_name = name + '.self_attn.q_proj'
            logger.info(f"[smooth_lm] sp_name: {sp_name}")
            logger.info("get_scale")
            sm_scales = get_scale(qkv, scales[sp_name], alpha)
            logger.info("scale={},max={},min={},scale.shape: {}".format(sm_scales,sm_scales.max(), sm_scales.min(), sm_scales.shape))
            smooth_ln_fcs(attn_ln, qkv, sm_scales)

            if module.self_attn.v_proj.weight.shape == module.self_attn.o_proj.weight.shape:
                logging.info("smooth vo")
                prev_op = module.self_attn.v_proj
                layers = [module.self_attn.o_proj]
                sp_name = name + '.self_attn.o_proj'
                logger.info(f"[smooth_lm] sp_name: {sp_name}")
                logger.info("get_scale")
                sm_scales = get_scale(layers[0], scales[sp_name], alpha)
                logger.info("scale={},max={},min={},scale.shape: {}".format(sm_scales,sm_scales.max(), sm_scales.min(), sm_scales.shape))
                scale_fc_fc(prev_op, layers[0], sm_scales)
            

            logging.info(f"smooth mlp.gate, up_proj/gate_proj in mlp.experts")
            ffn_ln = module.post_attention_layernorm
            fc1 = [module.mlp.gate] + [module.mlp.experts[i].gate_proj for i in range(expert_num)] + [module.mlp.experts[i].up_proj for i in range(expert_num)] + [module.mlp.shared_expert.gate_proj, module.mlp.shared_expert.up_proj]
            sp_name = name + '.mlp.gate'
            logger.info(f"[smooth_lm] sp_name: {sp_name}")
            logger.info(f"use prebuilt MoE fc1 smooth scale from {fc1_scale_merge}")
            sm_scales = _get_moe_fc1_smooth_scale(moe_fc1_smooth_scales, sp_name, module.mlp.gate)
            logger.info("scale={},max={},min={},scale.shape: {}".format(sm_scales,sm_scales.max(), sm_scales.min(), sm_scales.shape))
            smooth_ln_fcs(ffn_ln, fc1, sm_scales)

            for i in range(expert_num):
                logging.info(f"smooth down_proj in mlp.experts.{i}")
                prev_op = module.mlp.experts[i].up_proj
                layers = [module.mlp.experts[i].down_proj]
                sp_name = name + f'.mlp.experts.{i}.down_proj'
                if sp_name in scales.keys():
                    logger.info(f"[smooth_lm] sp_name: {sp_name}")
                    logger.info(f"get_moe_down_smooth_scale mode={moe_down_smooth_mode}")
                    sm_scales = get_moe_down_smooth_scale(layers[0], scales, act_per_channel_scales, sp_name, dev, alpha, otsu_ratio, otsu_smooth_rate, mode=moe_down_smooth_mode)
                    logger.info("scale={},max={},min={},scale.shape: {}".format(sm_scales,sm_scales.max(), sm_scales.min(), sm_scales.shape))
                    scale_fc_fc(prev_op, layers[0], sm_scales)

            prev_op = module.mlp.shared_expert.up_proj
            layers = [module.mlp.shared_expert.down_proj]
            sp_name = name + f'.mlp.shared_expert.down_proj'
            logger.info(f"[smooth_lm] sp_name: {sp_name}")
            logger.info(f"get_moe_down_smooth_scale mode={moe_down_smooth_mode}")
            sm_scales = get_moe_down_smooth_scale(layers[0], scales, act_per_channel_scales, sp_name, dev, alpha, otsu_ratio, otsu_smooth_rate, mode=moe_down_smooth_mode)
            logger.info("scale={},max={},min={},scale.shape: {}".format(sm_scales,sm_scales.max(), sm_scales.min(), sm_scales.shape))
            scale_fc_fc(prev_op, layers[0], sm_scales)
