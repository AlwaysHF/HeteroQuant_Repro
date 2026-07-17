#!/usr/bin/env python3
import argparse
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from datasets import load_from_disk
from transformers import AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def parse_int_list(value):
    out = []
    for item in value.split(","):
        item = item.strip()
        if item:
            out.append(int(item))
    if not out:
        raise ValueError(f"empty integer list: {value!r}")
    return out


def load_texts(dataset_path, max_docs):
    dataset_path = os.path.abspath(os.path.expanduser(dataset_path))
    ds = load_from_disk(dataset_path)
    if hasattr(ds, "keys"):
        split = "train" if "train" in ds else next(iter(ds.keys()))
        ds = ds[split]
    texts = []
    for row in ds:
        text = row.get("text", "")
        if text and text.strip():
            texts.append(text.strip())
        if len(texts) >= max_docs:
            break
    if not texts:
        raise RuntimeError(f"no non-empty text found in {dataset_path}")
    return "\n\n".join(texts)


def get_layer_mlp(model, layer_idx):
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers[layer_idx].mlp
    if hasattr(model, "base_model") and hasattr(model.base_model, "model"):
        return model.base_model.model.layers[layer_idx].mlp
    raise RuntimeError("could not locate model.model.layers[*].mlp")


def collect_expert_inputs(args):
    experts = parse_int_list(args.experts)
    device = torch.device(args.device)
    dtype = torch.float16 if device.type == "cuda" else torch.float32

    print(f"[activation-surface] loading tokenizer: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"[activation-surface] loading model on {device}")
    from models.olmoe.modeling_olmoe import OlmoeForCausalLM

    model = OlmoeForCausalLM.from_pretrained(
        args.model,
        torch_dtype=dtype,
        device_map="cpu",
        trust_remote_code=True,
        attn_implementation=args.attn_implementation,
    )
    model.to(device)
    model.eval()
    model.config.use_cache = False

    text = load_texts(args.dataset_path, args.max_docs)
    enc = tokenizer(text, return_tensors="pt", truncation=False)
    input_ids = enc.input_ids[0]
    if input_ids.numel() < args.seq_length:
        raise RuntimeError(
            f"not enough tokens: have {input_ids.numel()}, need at least {args.seq_length}"
        )

    mlp = get_layer_mlp(model, args.layer)
    top_k = int(getattr(mlp, "top_k", getattr(model.config, "num_experts_per_tok", 1)))
    collected = {expert: [] for expert in experts}
    counts = {expert: 0 for expert in experts}

    def gate_hook(_module, inputs, output):
        hidden = inputs[0].detach()
        logits = output.detach().float()
        selected = torch.topk(F.softmax(logits, dim=-1), top_k, dim=-1).indices
        for expert in experts:
            need = args.tokens_per_expert - counts[expert]
            if need <= 0:
                continue
            mask = (selected == expert).any(dim=-1)
            if not torch.any(mask):
                continue
            values = hidden[mask][:need].float().cpu()
            if values.numel() > 0:
                collected[expert].append(values)
                counts[expert] += int(values.shape[0])

    handle = mlp.gate.register_forward_hook(gate_hook)
    try:
        with torch.no_grad():
            start = 0
            chunk_id = 0
            while start + args.seq_length <= input_ids.numel():
                chunk = input_ids[start : start + args.seq_length].unsqueeze(0).to(device)
                print(
                    f"[activation-surface] chunk={chunk_id} token_range=[{start},{start + args.seq_length}) "
                    f"counts={counts}"
                )
                model(chunk, use_cache=False)
                if all(counts[expert] >= args.tokens_per_expert for expert in experts):
                    break
                chunk_id += 1
                start += args.seq_length
                if chunk_id >= args.max_chunks:
                    break
    finally:
        handle.remove()

    arrays = {}
    for expert in experts:
        if collected[expert]:
            arr = torch.cat(collected[expert], dim=0)[: args.tokens_per_expert].numpy()
        else:
            arr = np.zeros((0, int(model.config.hidden_size)), dtype=np.float32)
        if arr.shape[0] < args.tokens_per_expert:
            print(
                f"[activation-surface] warning: expert {expert} only collected "
                f"{arr.shape[0]} tokens"
            )
        arrays[f"expert_{expert}"] = arr
    return arrays, experts, int(model.config.hidden_size)


def style_3d_axis(ax):
    ax.set_xlabel("Channel", labelpad=5)
    ax.set_ylabel("Token", labelpad=5)
    ax.set_zlabel("")
    ax.view_init(elev=29, azim=-62)
    ax.tick_params(axis="both", labelsize=8)
    ax.tick_params(axis="z", labelsize=8)
    ax.set_facecolor("white")
    ax.xaxis.pane.set_facecolor((1.0, 1.0, 1.0, 1.0))
    ax.yaxis.pane.set_facecolor((1.0, 1.0, 1.0, 1.0))
    ax.zaxis.pane.set_facecolor((1.0, 1.0, 1.0, 1.0))
    ax.xaxis.pane.set_edgecolor((0.86, 0.86, 0.86, 1.0))
    ax.yaxis.pane.set_edgecolor((0.86, 0.86, 0.86, 1.0))
    ax.zaxis.pane.set_edgecolor((0.86, 0.86, 0.86, 1.0))


def add_corner_annotation(ax, expert, args):
    """
    在每个 3D 子图左上角添加注释：
    OLMoE
    Layer x
    Expert y

    可通过参数调节位置、字号、颜色。
    """
    text = f"OLMoE\nLayer {args.layer}\nExpert {expert}"
    ax.text2D(
        args.corner_annot_x,
        args.corner_annot_y,
        text,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=args.corner_annot_fontsize,
        fontweight=args.corner_annot_fontweight,
        color=args.corner_annot_color,
    )


def plot_surface_figure(arrays, experts, args):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(15.2, 5.25), facecolor="white")
    cmap = plt.get_cmap(args.cmap)
    global_vmax = 0.0
    transformed = {}

    for expert in experts:
        arr = arrays[f"expert_{expert}"]
        arr = arr[: args.tokens_per_expert]
        if arr.size == 0:
            z = np.zeros((args.tokens_per_expert, 1), dtype=np.float32)
        else:
            z = np.log1p(np.abs(arr))
        transformed[expert] = z
        if z.size:
            global_vmax = max(global_vmax, float(np.percentile(z, args.clip_percentile)))
    global_vmax = max(global_vmax, 1e-6)

    for i, expert in enumerate(experts, start=1):
        ax = fig.add_subplot(1, len(experts), i, projection="3d")
        z_full = transformed[expert]
        z = z_full[:, :: args.channel_stride]
        tokens = np.arange(z.shape[0])
        channels = np.arange(z.shape[1]) * args.channel_stride
        x, y = np.meshgrid(channels, tokens)
        z = np.clip(z, 0.0, global_vmax)

        ax.plot_surface(
            x,
            y,
            z,
            cmap=cmap,
            linewidth=0,
            antialiased=True,
            rstride=1,
            cstride=1,
            alpha=0.96,
        )

        ax.set_xlim(0, max(1, args.hidden_size_override or x.max()))
        ax.set_ylim(0, max(1, args.tokens_per_expert))
        ax.set_zlim(0, global_vmax)

        style_3d_axis(ax)

        # 左上角注释：OLMoE / Layer / Expert
        add_corner_annotation(ax, expert, args)

        # 右侧竖直 z 轴说明
        ax.text2D(
            1.02,
            0.52,
            "log1p(|x|)",
            transform=ax.transAxes,
            rotation=90,
            ha="left",
            va="center",
            fontsize=11,
            color="#1F1F1F",
        )

    fig.tight_layout(pad=1.0, rect=(0.0, 0.05, 1.0, 1.0))
    png_path = output_dir / f"input_activation_surface_layer{args.layer}.png"
    pdf_path = output_dir / f"input_activation_surface_layer{args.layer}.pdf"
    fig.savefig(png_path, dpi=args.dpi, bbox_inches="tight")
    if args.save_pdf:
        fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[activation-surface] saved: {png_path}")
    if args.save_pdf:
        print(f"[activation-surface] saved: {pdf_path}")


def plot_sorted_curve(arrays, experts, args):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7.0, 4.2), facecolor="white")
    colors = ["#D1495B", "#3C8D7B", "#7A5AA6", "#E07A5F", "#487DBA"]
    for idx, expert in enumerate(experts):
        arr = arrays[f"expert_{expert}"]
        if arr.size == 0:
            continue
        score = np.percentile(np.abs(arr), 99, axis=0)
        score = score / max(float(np.mean(score)), 1e-12)
        sorted_score = np.sort(score)[::-1]
        ax.plot(
            np.arange(sorted_score.shape[0]),
            sorted_score,
            linewidth=1.45,
            color=colors[idx % len(colors)],
            label=f"Expert {expert}",
        )
    ax.set_xlabel("Channel rank")
    ax.set_ylabel("Normalized p99(|activation|)")
    ax.set_title(f"Layer {args.layer} input-channel concentration", fontweight="bold")
    ax.grid(axis="y", alpha=0.25, linestyle="--")
    ax.legend(frameon=True)
    ax.set_xlim(0, args.hidden_size_override or None)
    fig.tight_layout()

    png_path = output_dir / f"input_activation_sorted_p99_layer{args.layer}.png"
    pdf_path = output_dir / f"input_activation_sorted_p99_layer{args.layer}.pdf"
    fig.savefig(png_path, dpi=args.dpi, bbox_inches="tight")
    if args.save_pdf:
        fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[activation-surface] saved: {png_path}")
    if args.save_pdf:
        print(f"[activation-surface] saved: {pdf_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Collect routed OLMoE expert input activations and draw channel-token surfaces."
    )
    parser.add_argument("--model", default="/home/lwk/HeteroQuant_Repro/local_models/olmoe_compat")
    parser.add_argument("--dataset_path", default="/cephfs/shared/lwk/notebook04/datasets/wiki2_raw_v1")
    parser.add_argument("--output_dir", default="/home/lwk/HeteroQuant_Repro/figures/input_activation_surface")
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--experts", default="8,6,38")
    parser.add_argument("--tokens_per_expert", type=int, default=200)
    parser.add_argument("--seq_length", type=int, default=4096)
    parser.add_argument("--max_chunks", type=int, default=4)
    parser.add_argument("--max_docs", type=int, default=2000)
    parser.add_argument("--channel_stride", type=int, default=1)
    parser.add_argument("--clip_percentile", type=float, default=99.95)
    parser.add_argument("--cmap", default="coolwarm")
    parser.add_argument("--dpi", type=int, default=220)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--attn_implementation", default="eager")
    parser.add_argument("--hidden_size_override", type=int, default=2048)
    parser.add_argument("--load_npz", default="")
    parser.add_argument("--save_pdf", action="store_true")

    # ========= 左上角注释可调参数 =========
    parser.add_argument("--corner_annot_x", type=float, default=0.03)
    parser.add_argument("--corner_annot_y", type=float, default=0.97)
    parser.add_argument("--corner_annot_fontsize", type=float, default=13)
    parser.add_argument("--corner_annot_fontweight", default="bold")
    parser.add_argument("--corner_annot_color", default="#4A4A4A")  # 深灰色
    # ===================================

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    experts = parse_int_list(args.experts)

    if args.load_npz:
        data = np.load(args.load_npz)
        arrays = {key: data[key] for key in data.files if key.startswith("expert_")}
        print(f"[activation-surface] loaded cached activations: {args.load_npz}")
    else:
        arrays, experts, hidden_size = collect_expert_inputs(args)
        args.hidden_size_override = hidden_size
        npz_path = output_dir / f"input_activation_layer{args.layer}_experts_{'_'.join(map(str, experts))}.npz"
        np.savez_compressed(npz_path, **arrays)
        print(f"[activation-surface] saved cached activations: {npz_path}")

    plot_surface_figure(arrays, experts, args)
    plot_sorted_curve(arrays, experts, args)


if __name__ == "__main__":
    main()