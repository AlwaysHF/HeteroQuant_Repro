from quantize.int_linear import QuantLinear
import torch
import torch.nn as nn
from quantize.int_matmul import QuantMatMul
from quantize.quantizer import UniformAffineQuantizer

def set_registered_x_none(model):
    for name, module in model.named_modules():
        if isinstance(module, QuantLinear):
            module.weight_quantizer.registered_x = None
            module.act_quantizer.registered_x = None


@torch.no_grad()
def set_init_duquant_params_state(model, mode):
    if isinstance(mode, bool):
        mode = torch.tensor(mode)
    for name, module in model.named_modules():
        if hasattr(module, "init_duquant_params"):
            module.init_duquant_params = mode


@torch.no_grad()
def set_calibration_state(model, mode):
    if isinstance(mode, bool):
        mode = torch.tensor(mode)
    for name, module in model.named_modules():
        if isinstance(module, QuantLinear):
            module.weight_quantizer.do_calibration = mode



@torch.no_grad()
def quant_inplace(model):
    for name, module in model.named_modules():
        if isinstance(module, QuantLinear):
            module.weight = module.weight_quantizer(module.weight, return_no_quant=False)
        elif hasattr(module, "quantize_weight_inplace"):
            module.quantize_weight_inplace()

@torch.no_grad()
def quant_soft_inplace(model):
    for name, module in model.named_modules():
        if isinstance(module, QuantLinear):
            module.weight = module.weight_quantizer(module.weight, return_no_quant=True)

def set_quant_state(self, weight_quant: bool = False, act_quant: bool = False):
    # setting weight quantization here does not affect actual forward pass
    self.use_weight_quant = weight_quant
    self.use_act_quant = act_quant
    for m in self.modules():
        if isinstance(m, (QuantLinear, QuantMatMul)):
            m.set_quant_state(weight_quant, act_quant)
        elif hasattr(m, "quantize_weight_inplace") and hasattr(m, "set_quant_state"):
            m.set_quant_state(weight_quant, act_quant)
