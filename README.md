# HeteroQuant Repro

This folder is a trimmed HeteroQuant reproduction project built on the DuQuant runtime path.

It keeps only the runtime code needed for the routed_range HeteroQuant
configuration used by the OLMoE and Qwen reproductions.

## Models

Model weights are symlinks, not copies. The scripts expect:

- `local_models/olmoe_compat`
- `local_models/Qwen1.5-MoE-A2.7B`


## Runtime Artifacts

The repository does not track model weights, cached datasets, experiment outputs,
or `Rot.pkl`.

Before running the scripts on a fresh clone, prepare these local paths:

- `local_models/olmoe_compat`
- `local_models/Qwen1.5-MoE-A2.7B`
- `Rot.pkl` in the project root

In the current environment, `Rot.pkl` is copied from the original runtime tree.
It is intentionally excluded from normal Git because it is a large binary file.
Use Git LFS or external storage if this file must be distributed through GitHub.

## Reproduce

Full OLMoE 6.91 style run:

```bash
cd /home/lwk/HeteroQuant_Repro
bash scripts/reproduce_olmoe_691.sh
```

Expected key result:

- `wikitext2 : 6.906770706176758`

Full Qwen 7.44 style run:

```bash
cd /home/lwk/HeteroQuant_Repro
bash scripts/reproduce_qwen_744.sh
```

Expected key result:

- `wikitext2 : 7.440934181213379`

For a faster wiki-only check, clear the task list and restrict the PPL dataset:

```bash
TASKS="" TEST_DATASET=wikitext2 bash scripts/reproduce_olmoe_691.sh
TASKS="" TEST_DATASET=wikitext2 bash scripts/reproduce_qwen_744.sh
```

## Notes

- `DUQUANT_TORCH_NUM_THREADS=8` and
  `DUQUANT_TORCH_NUM_INTEROP_THREADS=8` are set by the scripts to avoid the
  slow CPU thread oversubscription seen in the original environment.
- Profiling/debug stop code is intentionally not included in this project.
