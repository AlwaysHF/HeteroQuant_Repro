import pdb
from transformers import AutoTokenizer
from datasets import load_dataset
import numpy as np
import torch
import random
import glob
import os


def set_seed(seed):
    np.random.seed(seed)
    torch.random.manual_seed(seed)


def get_tokenizer(model):
    try:
        return AutoTokenizer.from_pretrained(model, use_fast=False)
    except ValueError:
        return AutoTokenizer.from_pretrained(model, use_fast=True)





def _hf_dataset_cache_roots():
    roots = []
    if os.environ.get("HF_DATASETS_CACHE"):
        roots.append(os.environ["HF_DATASETS_CACHE"])
    if os.environ.get("HF_HOME"):
        roots.append(os.path.join(os.environ["HF_HOME"], "datasets"))
    roots.append(os.path.expanduser("~/.cache/huggingface/datasets"))
    deduped = []
    for root in roots:
        root = os.path.abspath(os.path.expanduser(root))
        if root not in deduped:
            deduped.append(root)
    return deduped


def _load_c4_from_arrow_cache(split):
    try:
        from datasets import Dataset, concatenate_datasets
    except Exception:
        return None

    patterns = []
    for root in _hf_dataset_cache_roots():
        base = os.path.join(root, "allenai___c4", "default-*", "0.0.0", "*")
        if split == "train":
            patterns.append(os.path.join(base, "c4-train-*.arrow"))
        else:
            patterns.append(os.path.join(base, "c4-validation*.arrow"))

    paths = []
    for pattern in patterns:
        paths.extend(glob.glob(pattern))
    paths = sorted(set(paths))
    if not paths:
        return None

    datasets = [Dataset.from_file(path) for path in paths]
    if len(datasets) == 1:
        return datasets[0]
    return concatenate_datasets(datasets)


def _load_c4_split(split):
    data_files = {
        "train": {"train": "en/c4-train.00000-of-01024.json.gz"},
        "validation": {"validation": "en/c4-validation.00000-of-00008.json.gz"},
    }
    try:
        return load_dataset("allenai/c4", data_files=data_files[split], split=split)
    except Exception as exc:
        fallback = _load_c4_from_arrow_cache(split)
        if fallback is not None:
            print(f"load allenai/c4 {split} from local arrow cache after load_dataset failed: {exc}")
            return fallback
        raise


def get_pile(nsamples, seed, seqlen, model):
    print("get_pile")
    traindata = load_dataset("json", data_files='/cpfs01/user/chenmengzhao/prompt_quantization/val.jsonl.zst', split="train")

    tokenizer = get_tokenizer(model)
    trainenc = tokenizer("\n\n".join(traindata['text'][:1000]), return_tensors='pt')

    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))
    return trainloader, None


def get_wikitext2(nsamples, seed, seqlen, model):
    print("get_wikitext2")
    traindata = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train') # hugging-face hub path: 'wikitext'
    testdata = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test') # hugging-face hub path: 'wikitext'

    tokenizer = get_tokenizer(model)
    trainenc = tokenizer("\n\n".join(traindata['text']), return_tensors='pt')
    testenc = tokenizer("\n\n".join(testdata['text']), return_tensors='pt')

    
    random.seed(seed)
    trainloader = []
    # print('trainenc', trainenc.input_ids)
    for _ in range(nsamples):
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))
    return trainloader, testenc

def get_ptb(nsamples, seed, seqlen, model):
    print("get_ptb")
    traindata = load_dataset('ptb_text_only', 'penn_treebank', split='train')
    valdata = load_dataset('ptb_text_only', 'penn_treebank', split='validation')


    tokenizer = get_tokenizer(model)

    trainenc = tokenizer("\n\n".join(traindata['sentence']), return_tensors='pt')
    testenc = tokenizer("\n\n".join(valdata['sentence']), return_tensors='pt')

    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))
    return trainloader, testenc

def get_c4(nsamples, seed, seqlen, model):
    traindata = _load_c4_split('train') # hugging-face hub path: 'allenai/c4'
    valdata = _load_c4_split('validation') # hugging-face hub path: 'allenai/c4'


    tokenizer = get_tokenizer(model)

    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        while True:
            i = random.randint(0, len(traindata) - 1)
            trainenc = tokenizer(traindata[i]['text'], return_tensors='pt')
            if trainenc.input_ids.shape[1] > seqlen:
                break
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))

    random.seed(20)
    valenc = []
    for _ in range(256):
        while True:
            i = random.randint(0, len(valdata) - 1)
            tmp = tokenizer(valdata[i]['text'], return_tensors='pt')
            if tmp.input_ids.shape[1] > seqlen:
                break
        i = random.randint(0, tmp.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        valenc.append(tmp.input_ids[:, i:j])
    valenc = torch.hstack(valenc)

    return trainloader, valenc 

def get_ptb_new(nsamples, seed, seqlen, model):
    print("get_ptb_new")
    traindata = load_dataset('ptb_text_only', 'penn_treebank', split='train')
    testdata  = load_dataset('ptb_text_only', 'penn_treebank', split='test')


    tokenizer = get_tokenizer(model)

    trainenc = tokenizer(" ".join(traindata["sentence"]), return_tensors="pt")
    testenc = tokenizer(" ".join(testdata ["sentence"]), return_tensors="pt")

    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))
    return trainloader, testenc


def get_c4_new(nsamples, seed, seqlen, model):
    print("get_c4_new")
    traindata = _load_c4_split('train')
    valdata = _load_c4_split('validation')

    tokenizer = get_tokenizer(model)
    
    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        while True:
            i = random.randint(0, len(traindata) - 1)
            trainenc = tokenizer(traindata[i]["text"], return_tensors="pt")
            if trainenc.input_ids.shape[1] > seqlen:
                break
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))

    valenc = tokenizer(" ".join(valdata[:1100]["text"]), return_tensors="pt")
    valenc = valenc.input_ids[:, : (256 * seqlen)]
    return trainloader, valenc


def get_loaders(
    name, nsamples=128, seed=0, seqlen=2048, model='',
):
    if 'wikitext2' in name:
        return get_wikitext2(nsamples, seed, seqlen, model)
    if 'pile' in name:
        return get_pile(nsamples, seed, seqlen, model)
    if 'ptb' in name:
        if 'new' in name:
            return get_ptb_new(nsamples, seed, seqlen, model)  
        return get_ptb(nsamples, seed, seqlen, model)
    if 'c4' in name:
        if 'new' in name:
            return get_c4_new(nsamples, seed, seqlen, model)  
        return get_c4(nsamples, seed, seqlen, model)
    if 'mix' in name:
        wiki_train,wiki_val=get_wikitext2(nsamples//3, seed, seqlen, model)
        ptb_train,ptb_val=get_ptb(nsamples//3, seed, seqlen, model)
        c4_train,c4_val=get_c4(nsamples//3, seed, seqlen, model)
        train=wiki_train+ptb_train+c4_train
        val=None
        return train,val
