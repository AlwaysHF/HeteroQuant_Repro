import torch
from math import inf
import logging
from termcolor import colored
import sys
import os
import time
import pickle
from tqdm import tqdm
import math
import torch.nn as nn

import functools
import torch.nn.functional as F
from collections import defaultdict

def _sync_cuda_for_timing():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


@torch.no_grad()
def ampscaler_get_grad_norm(parameters, norm_type: float = 2.0) -> torch.Tensor:
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = [p for p in parameters if p.grad is not None]
    norm_type = float(norm_type)
    if len(parameters) == 0:
        return torch.tensor(0.)
    device = parameters[0].grad.device
    if norm_type == inf:
        total_norm = max(p.grad.detach().abs().max().to(device) for p in parameters)
    else:
        total_norm = torch.norm(torch.stack([torch.norm(p.grad.detach(),
                                                        norm_type).to(device) for p in parameters]), norm_type)
    return total_norm

class NativeScalerWithGradNormCount:
    state_dict_key = "amp_scaler"

    def __init__(self):
        self._scaler = torch.cuda.amp.GradScaler()

    def __call__(self, loss, optimizer, clip_grad=None, parameters=None, create_graph=False, update_grad=True,retain_graph=False):
        self._scaler.scale(loss).backward(create_graph=create_graph, retain_graph=retain_graph)
        if update_grad:
            if clip_grad is not None:
                assert parameters is not None
                self._scaler.unscale_(optimizer)  # unscale the gradients of optimizer's assigned params in-place
                norm = torch.nn.utils.clip_grad_norm_(parameters, clip_grad)
            else:
                self._scaler.unscale_(optimizer)
                norm = ampscaler_get_grad_norm(parameters)
            self._scaler.step(optimizer)
            self._scaler.update()
        else:
            norm = None
        return norm

    def state_dict(self):
        return self._scaler.state_dict()

    def load_state_dict(self, state_dict):
        self._scaler.load_state_dict(state_dict)


def create_logger(output_dir, dist_rank=0, name=''):
    # create logger
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    # create formatter
    fmt = '[%(asctime)s %(name)s] (%(filename)s %(lineno)d): %(levelname)s %(message)s'
    color_fmt = colored('[%(asctime)s %(name)s]', 'green') + \
                colored('(%(filename)s %(lineno)d)', 'yellow') + ': %(levelname)s %(message)s'

    # create console handlers for master process
    if dist_rank == 0:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.DEBUG)
        console_handler.setFormatter(
            logging.Formatter(fmt=color_fmt, datefmt='%Y-%m-%d %H:%M:%S'))
        logger.addHandler(console_handler)

    # create file handlers
    file_handler = logging.FileHandler(os.path.join(output_dir, f'log_rank{dist_rank}_{int(time.time())}.txt'), mode='a')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter(fmt=fmt, datefmt='%Y-%m-%d %H:%M:%S'))
    logger.addHandler(file_handler)

    return logger


def exchange_row_col(_tensor, i, j):
    tensor = _tensor.detach().clone()
    assert isinstance(tensor, torch.Tensor)
    indices_row = torch.arange(tensor.size(0))
    indices_row[i], indices_row[j] = indices_row[j].item(), indices_row[i].item()
    tensor = tensor[indices_row]

    indices_col = torch.arange(tensor.size(1))
    indices_col[i], indices_col[j] = indices_col[j].item(), indices_col[i].item()
    tensor = tensor[:, indices_col]
    return tensor

Rot = {}
def get_rot(n, device='cpu'):
    try:
        if Rot.get(n) is None:
            Rot[n] = pickle.load(open("Rot.pkl", "rb"))[n]
        R = Rot[n].to(device)
        random_matrix = torch.randn(n-1, n-1).to(device)
        q, r = torch.linalg.qr(random_matrix)
        q = torch.cat([torch.zeros(n-1, 1).to(device), q], dim=1)
        q = torch.cat([torch.zeros(1, n).to(device), q], dim=0)
        q[0, 0] = 1
        R = torch.matmul(R,q)
        return R
    except Exception as e:
        print(e)
        assert False, 'No such rotate matrix'


def get_hadamard(n): 
    if n == 1:
        return torch.tensor([[1.]], dtype=torch.float32)
    else:
        assert n % 1 == 0, "The size should be divided by 2."
        H_n_minus_1 = get_hadamard(n//2)
        return torch.cat([torch.cat([H_n_minus_1, H_n_minus_1], dim=1),
                          torch.cat([H_n_minus_1, -H_n_minus_1], dim=1)], dim=0) / math.sqrt(2)




use_GPU = torch.cuda.is_available()
used_device = "cuda"
if not use_GPU:
    import torch_npu
    used_device = "npu"


device_map_olmoe = {'model.embed_tokens': 0,
              'model.layers.0': 0, 'model.layers.1': 0, 'model.layers.2': 0, 'model.layers.3': 0, 'model.layers.4': 0, 'model.layers.5': 0, 
              'model.layers.6': 0, 'model.layers.7': 0, 
              'model.layers.8': 0, 'model.layers.9': 0, 'model.layers.10': 0, 'model.layers.11': 0,
              'model.layers.12': 0, 'model.layers.13': 0, 'model.layers.14': 0, 'model.layers.15': 0, 
              'model.norm':0, "lm_head":0}

device_map_pangumoe = {'model.embed_tokens': 1,
              'model.layers.0': 0, 'model.layers.1': 0, 'model.layers.2': 0, 'model.layers.3': 0, 'model.layers.4': 0, 'model.layers.5': 0, 
              'model.layers.6': 1, 'model.layers.7': 1, 'model.layers.8': 1, 'model.layers.9': 1, 'model.layers.10': 1, 'model.layers.11': 1,
              'model.layers.12': 2, 'model.layers.13': 2, 'model.layers.14': 2, 'model.layers.15': 2, 'model.layers.16': 2, 'model.layers.17': 2,
              'model.layers.18': 3, 'model.layers.19': 3, 'model.layers.20': 3, 'model.layers.21': 3, 'model.layers.22': 3, 'model.layers.23': 3,
              'model.layers.24': 4, 'model.layers.25': 4, 'model.layers.26': 4, 'model.layers.27': 4, 'model.layers.28': 4, 'model.layers.29': 4,
              'model.layers.30': 5, 'model.layers.31': 5, 'model.layers.32': 5, 'model.layers.33': 5, 'model.layers.34': 5, 'model.layers.35': 5,
              'model.layers.36': 6, 'model.layers.37': 6, 'model.layers.38': 6, 'model.layers.39': 6, 'model.layers.40': 6, 'model.layers.41': 6,
              'model.layers.42': 7, 'model.layers.43': 7, 'model.layers.44': 7, 'model.layers.45': 7, 'model.layers.46': 7, 'model.layers.47': 7,
              'model.norm':7, "lm_head":6}


from accelerate.big_modeling import dispatch_model


def convert_device(lm, args):
    if "pangumoe" in args.net.lower():
        dispatch_model(lm.model, device_map=device_map_pangumoe)
    elif "olmoe" in args.net.lower():
        dispatch_model(lm.model, device_map=device_map_olmoe)
    else:
        lm.model = lm.model.to(used_device)


from lm_eval import evaluator
from datautils import get_loaders


@torch.no_grad()
def evaluate(lm, args, logger):
    lm.seqlen = 2048 # use fixed seq_length for inference
    args.seq_length = 2048 # use fixed seq_length for inference

    results = {}
    convert_device(lm, args)

    if args.eval_ppl:
        test_datasets = getattr(args, "test_dataset", "wikitext2,c4")
        if isinstance(test_datasets, str):
            test_datasets = [item.strip() for item in test_datasets.split(",") if item.strip()]
        if not test_datasets:
            test_datasets = ["wikitext2"]
        for dataset in test_datasets:
            cache_testloader = f'{args.cache_dir}/testloader_{args.model_family}_{dataset}_{args.seq_length}_all.cache'
            if os.path.exists(cache_testloader):
                testloader = torch.load(cache_testloader)
            else:
                dataloader, testloader = get_loaders(
                    dataset,
                    seed=args.seed,
                    model=args.model,
                    seqlen=lm.seqlen,
                )
                torch.save(testloader, cache_testloader)
            if "c4" in dataset:
                testenc = testloader
            else:
                testenc = testloader.input_ids

            nsamples = testenc.numel() // lm.seqlen
            use_cache = lm.model.config.use_cache
            lm.model.config.use_cache = False
            lm.model.eval()
            testenc = testenc.to(lm.device)
            nlls = []
            loss_fct = nn.CrossEntropyLoss()
            _sync_cuda_for_timing()
            inference_start = time.perf_counter()
            with torch.inference_mode():
                for i in tqdm(range(nsamples)):
                    batch = testenc[:, (i * lm.seqlen) : ((i + 1) * lm.seqlen)]
                    outputs = lm.model.model(batch)
                    hidden_states = outputs[0]
                    logits = lm.model.lm_head(hidden_states)
                    shift_logits = logits[:, :-1, :]
                    shift_labels = testenc[:, (i * lm.seqlen) : ((i + 1) * lm.seqlen)][
                        :, 1:
                    ].to(lm.model.lm_head.weight.device)
                    loss = loss_fct(
                        shift_logits.view(-1, shift_logits.size(-1)),
                        shift_labels.view(-1),
                    )
                    neg_log_likelihood = loss.float() * lm.seqlen
                    nlls.append(neg_log_likelihood)
                eval_samples = len(nlls)
                ppl = torch.exp(torch.stack(nlls).sum() / (eval_samples * lm.seqlen))
            _sync_cuda_for_timing()
            inference_seconds = time.perf_counter() - inference_start
            logger.info(f'{dataset} : {ppl.item()}')
            logger.info(f'[timing] inference_seconds.{dataset}: {inference_seconds:.6f}')
            lm.model.config.use_cache = use_cache
            results[dataset] = ppl.item()
            results[f'{dataset}_inference_seconds'] = inference_seconds



    if args.tasks != "":
        task_specs = [task.strip() for task in args.tasks.split(",") if task.strip()]
        if not task_specs:
            return results

        task_string = ",".join(task_specs)
        task_lm = lm

        _sync_cuda_for_timing()
        task_start = time.perf_counter()
        try:
            t_results = evaluator.simple_evaluate(
                task_lm,
                tasks=task_string,
                num_fewshot=args.num_fewshot,
                limit=None,
            )
        except Exception:
            _sync_cuda_for_timing()
            task_seconds = time.perf_counter() - task_start
            logger.exception(f'tasks {task_string} failed after {task_seconds:.6f}s')
            return results
        _sync_cuda_for_timing()
        task_seconds = time.perf_counter() - task_start
        logger.info(f'[timing] inference_seconds.task.all: {task_seconds:.6f}')

        logger.info("original t_results")
        logger.info(t_results)
        del t_results['versions']
        del t_results['config']

        for key in t_results['results'].keys(): # piqa, winogrande
            if 'acc_stderr' in t_results['results'][key]:
                del t_results['results'][key]['acc_stderr']
            if 'acc_norm_stderr' in t_results['results'][key]:
                del t_results['results'][key]['acc_norm_stderr']
                
            if 'acc_norm' in t_results['results'][key]:
                if t_results['results'][key]['acc'] > t_results['results'][key]['acc_norm']:
                    del t_results['results'][key]['acc_norm']
                else:
                    del t_results['results'][key]['acc']
        logger.info("clear t_results")
        logger.info(t_results)
        
        logger.info("sorted t_results")
        t_results = dict(sorted(t_results['results'].items()))
        logger.info(t_results)

        acc_sum = 0
        acc_count = 0
        for res in t_results.values():
            if 'acc' in res:
                acc = res['acc']
            else:
                acc = res['acc_norm']
            acc_sum += acc
            acc_count += 1
        acc_avg = acc_sum / acc_count
        t_results['acc_avg'] = acc_avg
        t_results = dict(sorted(t_results.items()))

        logger.info("final t_results")
        logger.info(t_results)

        results.update(t_results)
        logger.info(results)
        # pprint(results)
        # for test of MMLU
        if 'hendrycksTest' in args.tasks:
            all_cors = []
            all_cors_norm = []
            subcat_cors = {subcat: [] for subcat_lists in subcategories.values() for subcat in subcat_lists}
            cat_cors = {cat: [] for cat in categories}
            cat_cors_norm = {cat: [] for cat in categories}
            for key in t_results['results'].keys():
                if not 'hendrycksTest' in key:
                    continue
                subject = key.split('-')[-1]
                cors = t_results['results'][key]['acc']
                cors_norm = t_results['results'][key]['acc_norm']
                subcats = subcategories[subject]
                for subcat in subcats:
                    subcat_cors[subcat].append(cors)
                    for key in categories.keys():
                        if subcat in categories[key]:
                            cat_cors[key].append(cors)
                            cat_cors_norm[key].append(cors_norm)
                    all_cors.append(cors)
                    all_cors_norm.append(cors_norm)
                    
            for cat in cat_cors:
                cat_acc = np.mean(cat_cors[cat])
                logger.info("Average accuracy {:.4f} - {}".format(cat_acc, cat))
            weighted_acc = np.mean(all_cors)
            logger.info("Average accuracy: {:.4f}".format(weighted_acc))               

    return results



# score means the maximum of max/mean in all columns
def get_weight_scores(model):
    print("get_weight_scores")
    weight_scores = {}

    for name, module in tqdm(model.named_modules()):
        if isinstance(module, (nn.Linear)):
            if ("mlp.gate" in name or "up_proj" in name or "block_sparse_moe.gate" in name or "w3" in name) and "gate_proj" not in name:
                weight = module.weight.data
                with torch.no_grad():
                    abs_weight = weight.abs()
                    score = abs_weight.max(dim=0).values / abs_weight.mean(dim=0)
                    score = score.max().item()
                    weight_scores[name] = score # cin
                
    for name, module in tqdm(model.named_modules()):
        if isinstance(module, (nn.Linear)):
            if "gate_proj" in name or "w1" in name:
                if "gate_proj" in name:
                    name_up = name.replace("gate_proj", "up_proj")
                if "w1" in name:
                    name_up = name.replace("w1", "w3")
                weight = module.weight.data
                with torch.no_grad():
                    abs_weight = weight.abs()
                    score = abs_weight.max(dim=0).values / abs_weight.mean(dim=0)
                    score = score.max().item()
                    weight_scores[name_up] = max(score, weight_scores[name_up])
    return weight_scores



def get_router_logits(model, dataloader, num_samples=128):
    print("get_router_logits")
    model.eval()
    device = next(model.parameters()).device
    router_logits_list = {}

    def stat_tensor(name, tensor): # router_logits
        routing_weights = F.softmax(tensor, dim=1, dtype=torch.float) # (batch * sequence_length, n_experts)

        if name in router_logits_list:
            router_logits_list[name] += routing_weights.mean(dim=0) # (n_experts)
        else:
            router_logits_list[name] = routing_weights.mean(dim=0)

    def stat_input_hook(m, x, y, name):
        if isinstance(y, tuple):
            y = y[-1]
        stat_tensor(name, y)

    hooks = []
    for name, m in model.named_modules():
        if isinstance(m, (nn.Linear)):
            if ("mlp.gate" in name or "block_sparse_moe.gate" in name) and "gate_proj" not in name:
                hooks.append(
                    m.register_forward_hook(
                        functools.partial(stat_input_hook, name=name)))

    for i in tqdm(range(num_samples)):
        model(dataloader[i][0].to(device))

    for h in hooks:
        h.remove()

    for key in router_logits_list.keys():
        router_logits_list[key] *= 1.0/num_samples
    return router_logits_list


def get_router_selected_experts(model, dataloader, top_k=8, num_samples=128, net="olmoe"):
    print("get_router_selected_experts")
    model.eval()
    device = next(model.parameters()).device
    selected_experts_list = defaultdict(list)

    def stat_tensor(name, tensor): # selected_experts
        routing_weights = F.softmax(tensor, dim=1, dtype=torch.float) # (batch * sequence_length, n_experts)
        _, selected_experts = torch.topk(routing_weights, top_k, dim=-1)
        selected_experts_list[name].append(selected_experts.to('cpu'))

    def stat_input_hook(m, x, y, name):
        if isinstance(y, tuple):
            y = y[-1]
        stat_tensor(name, y)

    hooks = []
    for name, m in model.named_modules():
        if isinstance(m, (nn.Linear)):
            if ("mlp.gate" in name or "block_sparse_moe.gate" in name) and "gate_proj" not in name:
                hooks.append(
                    m.register_forward_hook(
                        functools.partial(stat_input_hook, name=name)))

    for i in tqdm(range(num_samples)):
        model(dataloader[i][0].to(device))

    for h in hooks:
        h.remove()

    selected_experts_list = {k: torch.cat(v, dim=0) for k, v in selected_experts_list.items()}
    return selected_experts_list


def get_act_samples(model, dataloader, num_samples=128):
    print("get_act_samples")
    model.eval()
    device = next(model.parameters()).device
    act_samples = {}

    def stat_tensor(name, tensor):
        if ("mlp.gate" in name or "up_proj" in name or "block_sparse_moe.gate" in name or "w3" in name) and "gate_proj" not in name:
            # logging.info(f"[get_act_samples.stat_tensor] name: {name}")
            tensor = tensor.view(-1, tensor.shape[-1])
            num = tensor.shape[0]
            if name in act_samples:
                act_samples[name] += num
            else:
                act_samples[name] = num

    def stat_input_hook(m, x, y, name):
        if isinstance(x, tuple):
            x = x[0]
        stat_tensor(name, x)

    hooks = []
    for name, m in model.named_modules():
        if isinstance(m, (nn.Linear)):
            hooks.append(
                m.register_forward_hook(
                    functools.partial(stat_input_hook, name=name)))

    for i in tqdm(range(num_samples)):
        model(dataloader[i][0].to(device))

    for h in hooks:
        h.remove()

    return act_samples


def get_act_scales(model, dataloader, num_samples=128):
    print("get_act_scales")
    model.eval()
    device = next(model.parameters()).device
    act_scales = {}

    def stat_tensor(name, tensor):
        if any([_ in name for _ in ["q_proj", "o_proj", "up_proj", "down_proj", "mlp.gate", "block_sparse_moe.gate", "w2", "w3"]]):
            # logging.info(f"[get_act_scales.stat_tensor] name: {name}")
            if tensor.numel() > 0:
                hidden_dim = tensor.shape[-1]
                tensor = tensor.view(-1, hidden_dim).abs().detach()
                comming_max = torch.max(tensor, dim=0)[0].float().cpu()
                if name in act_scales:
                    act_scales[name] = torch.max(act_scales[name], comming_max)
                else:
                    act_scales[name] = comming_max

    def stat_input_hook(m, x, y, name):
        if isinstance(x, tuple):
            x = x[0]
        stat_tensor(name, x)

    hooks = []
    for name, m in model.named_modules():
        if isinstance(m, (nn.Linear)):
            hooks.append(
                m.register_forward_hook(
                    functools.partial(stat_input_hook, name=name)))

    for i in tqdm(range(num_samples)):
        model(dataloader[i][0].to(device))

    for h in hooks:
        h.remove()

    return act_scales



def get_moe_act_means(model, dataloader, num_samples=128):
    print("get_moe_act_means")
    model.eval()
    device = next(model.parameters()).device
    act_sums = {}
    act_counts = {}

    def stat_tensor(name, tensor):
        if "mlp.gate" in name and "gate_proj" not in name:
            if tensor.numel() > 0:
                hidden_dim = tensor.shape[-1]
                tensor = tensor.view(-1, hidden_dim).abs().detach().float()
                coming_sum = tensor.sum(dim=0).cpu()
                coming_count = tensor.shape[0]
                if name in act_sums:
                    act_sums[name] += coming_sum
                    act_counts[name] += coming_count
                else:
                    act_sums[name] = coming_sum
                    act_counts[name] = coming_count

    def stat_input_hook(m, x, y, name):
        if isinstance(x, tuple):
            x = x[0]
        stat_tensor(name, x)

    hooks = []
    for name, m in model.named_modules():
        if isinstance(m, (nn.Linear)) and "mlp.gate" in name and "gate_proj" not in name:
            hooks.append(
                m.register_forward_hook(
                    functools.partial(stat_input_hook, name=name)))

    for i in tqdm(range(num_samples)):
        model(dataloader[i][0].to(device))

    for h in hooks:
        h.remove()

    return {name: act_sums[name] / max(act_counts[name], 1) for name in act_sums}


def get_moe_act_p99s(model, dataloader, num_samples=128):
    print("get_moe_act_p99s")
    model.eval()
    device = next(model.parameters()).device
    act_p99s = {}

    def stat_tensor(name, tensor):
        if "mlp.gate" in name and "gate_proj" not in name:
            if tensor.numel() > 0:
                hidden_dim = tensor.shape[-1]
                tensor = tensor.view(-1, hidden_dim).abs().detach().float()
                coming_p99 = torch.quantile(tensor, 0.99, dim=0).cpu()
                if name in act_p99s:
                    act_p99s[name] = torch.max(act_p99s[name], coming_p99)
                else:
                    act_p99s[name] = coming_p99

    def stat_input_hook(m, x, y, name):
        if isinstance(x, tuple):
            x = x[0]
        stat_tensor(name, x)

    hooks = []
    for name, m in model.named_modules():
        if isinstance(m, (nn.Linear)) and "mlp.gate" in name and "gate_proj" not in name:
            hooks.append(
                m.register_forward_hook(
                    functools.partial(stat_input_hook, name=name)))

    for i in tqdm(range(num_samples)):
        model(dataloader[i][0].to(device))

    for h in hooks:
        h.remove()

    return act_p99s



def get_smooth_activation_stats(
    model,
    dataloader,
    num_samples=128,
    collect_moe_act_mean=False,
    collect_moe_act_p99=False,
    collect_act_scales=True,
    collect_act_per_channel_scales=True,
):
    print("get_smooth_activation_stats")
    model.eval()
    device = next(model.parameters()).device
    act_sums = {}
    act_counts = {}
    act_p99s = {}
    act_scales = {}
    act_per_channel_max = {}
    act_per_channel_min = {}

    def should_record_activation(name):
        return any(
            item in name
            for item in [
                "q_proj",
                "o_proj",
                "up_proj",
                "down_proj",
                "mlp.gate",
                "block_sparse_moe.gate",
                "w2",
                "w3",
            ]
        )

    def is_moe_gate(name):
        return "mlp.gate" in name and "gate_proj" not in name

    def stat_tensor(name, tensor):
        if tensor.numel() == 0:
            return

        record_activation = should_record_activation(name)
        record_moe_gate = is_moe_gate(name)
        if not record_activation and not record_moe_gate:
            return

        hidden_dim = tensor.shape[-1]
        tensor = tensor.view(-1, hidden_dim).detach()

        if collect_act_scales and record_activation:
            coming_max = torch.max(tensor.abs(), dim=0)[0].float().cpu()
            if name in act_scales:
                act_scales[name] = torch.max(act_scales[name], coming_max)
            else:
                act_scales[name] = coming_max

        if collect_act_per_channel_scales and record_activation:
            tensor_float = tensor.float()
            coming_max = torch.max(tensor_float, dim=0)[0].cpu()
            if name in act_per_channel_max:
                act_per_channel_max[name] = act_per_channel_max[name] * 0.9 + coming_max * 0.1
            else:
                act_per_channel_max[name] = coming_max

            coming_min = torch.min(tensor_float, dim=0)[0].cpu()
            if name in act_per_channel_min:
                act_per_channel_min[name] = act_per_channel_min[name] * 0.9 + coming_min * 0.1
            else:
                act_per_channel_min[name] = coming_min

        if record_moe_gate and (collect_moe_act_mean or collect_moe_act_p99):
            tensor_abs = tensor.abs().float()
            if collect_moe_act_mean:
                coming_sum = tensor_abs.sum(dim=0).cpu()
                coming_count = tensor_abs.shape[0]
                if name in act_sums:
                    act_sums[name] += coming_sum
                    act_counts[name] += coming_count
                else:
                    act_sums[name] = coming_sum
                    act_counts[name] = coming_count
            if collect_moe_act_p99:
                coming_p99 = torch.quantile(tensor_abs, 0.99, dim=0).cpu()
                if name in act_p99s:
                    act_p99s[name] = torch.max(act_p99s[name], coming_p99)
                else:
                    act_p99s[name] = coming_p99

    def stat_input_hook(m, x, y, name):
        if isinstance(x, tuple):
            x = x[0]
        stat_tensor(name, x)

    hooks = []
    for name, m in model.named_modules():
        if isinstance(m, nn.Linear):
            if should_record_activation(name) or is_moe_gate(name):
                hooks.append(
                    m.register_forward_hook(
                        functools.partial(stat_input_hook, name=name)))

    with torch.no_grad():
        for i in tqdm(range(num_samples)):
            model(dataloader[i][0].to(device))

    for h in hooks:
        h.remove()

    stats = {}
    if collect_moe_act_mean:
        stats["moe_act_means"] = {
            name: act_sums[name] / max(act_counts[name], 1)
            for name in act_sums
        }
    if collect_moe_act_p99:
        stats["moe_act_p99s"] = act_p99s
    if collect_act_scales:
        stats["act_scales"] = act_scales
    if collect_act_per_channel_scales:
        stats["act_per_channel_scales"] = (act_per_channel_max, act_per_channel_min)
    return stats


def get_moe_expert_act_square_means(model, dataloader, num_samples=128):
    print("get_moe_expert_act_square_means")
    model.eval()
    device = next(model.parameters()).device
    act_sums = {}
    act_counts = {}

    def should_record(name):
        if "mlp.gate" in name and "gate_proj" not in name:
            return True
        return ".mlp.experts." in name and name.endswith(".up_proj")

    def stat_tensor(name, tensor):
        if should_record(name) and tensor.numel() > 0:
            hidden_dim = tensor.shape[-1]
            tensor = tensor.view(-1, hidden_dim).detach().float()
            coming_sum = tensor.pow(2).sum(dim=0).cpu()
            coming_count = tensor.shape[0]
            if name in act_sums:
                act_sums[name] += coming_sum
                act_counts[name] += coming_count
            else:
                act_sums[name] = coming_sum
                act_counts[name] = coming_count

    def stat_input_hook(m, x, y, name):
        if isinstance(x, tuple):
            x = x[0]
        stat_tensor(name, x)

    hooks = []
    for name, m in model.named_modules():
        if isinstance(m, (nn.Linear)) and should_record(name):
            hooks.append(
                m.register_forward_hook(
                    functools.partial(stat_input_hook, name=name)))

    for i in tqdm(range(num_samples)):
        model(dataloader[i][0].to(device))

    for h in hooks:
        h.remove()

    return {name: act_sums[name] / max(act_counts[name], 1) for name in act_sums}


def get_act_per_channel_scales(model, dataloader, num_samples=128):
    print("get_act_per_channel_scales")
    model.eval()
    device = next(model.parameters()).device
    act_per_channel_max = {}
    act_per_channel_min = {}

    def stat_tensor(name, tensor):
        if any([_ in name for _ in ["q_proj", "o_proj", "up_proj", "down_proj", "mlp.gate", "block_sparse_moe.gate", "w2", "w3"]]):
            # logging.info(f"[get_act_per_channel_scales.stat_tensor] name: {name}")
            if tensor.numel() > 0:
                hidden_dim = tensor.shape[-1]
                tensor = tensor.view(-1, hidden_dim).detach().float()
                comming_max = torch.max(tensor, dim=0)[0].cpu()
                if name in act_per_channel_max:
                    # act_per_channel_max[name] = torch.max(act_per_channel_max[name], comming_max)
                    act_per_channel_max[name] = act_per_channel_max[name] * 0.9 + comming_max * 0.1
                else:
                    act_per_channel_max[name] = comming_max

                comming_min = torch.min(tensor, dim=0)[0].cpu()
                if name in act_per_channel_min:
                    # act_per_channel_min[name] = torch.min(act_per_channel_min[name], comming_min)
                    act_per_channel_min[name] = act_per_channel_min[name] * 0.9 + comming_min * 0.1
                else:
                    act_per_channel_min[name] = comming_min


    def stat_input_hook(m, x, y, name):
        if isinstance(x, tuple):
            x = x[0]
        stat_tensor(name, x)

    hooks = []
    for name, m in model.named_modules():
        if isinstance(m, (nn.Linear)):
            hooks.append(
                m.register_forward_hook(
                    functools.partial(stat_input_hook, name=name)))

    for i in tqdm(range(num_samples)):
        model(dataloader[i][0].to(device))

    for h in hooks:
        h.remove()

    return act_per_channel_max, act_per_channel_min



