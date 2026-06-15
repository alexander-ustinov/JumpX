# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this project is

JumpX trains lightweight "lens" adapters that let a frozen Qwen3.5-9B base model **skip a contiguous span of transformer layers** at inference time while preserving output quality. The base model's weights are never trained; only a small set of `nn.Linear` lenses are learned via knowledge distillation from the full (non-skipping) model.

The core idea (`model.py`):
- A `JumpQwen` runs the base model layer-by-layer. When it reaches a "start" layer in `target_pairs` (e.g. `[[22, 28]]`), it applies the trained lens to the hidden state and then jumps the layer index straight to the "end" layer — every layer in between is never executed.
- Each lens is a `d_model × d_model` linear layer initialized to **identity** (`nn.init.eye_` weight, zero bias), so an untrained model ≈ the full model on the skipped span.
- Lenses are applied to `hiddens.detach()` — gradients never flow back into the frozen base model.
- Training is pure distillation: KL divergence between the full model's logits (teacher) and the layer-skipping model's logits (student), `reduction="batchmean"`.

## Running

There is **no `requirements.txt`/`pyproject.toml`**; dependencies (`torch`, `transformers`, `datasets`, `omegaconf`, `fire`, `tqdm`) are assumed present in the environment. Python 3.11.

Training is distributed-only — `JumpExperiment` calls `dist.init_process_group("nccl")` unconditionally, so even single-GPU runs must go through `torchrun`:

```bash
# Train (distillation). config.yaml is the OmegaConf config.
torchrun --nproc_per_node=8 train.py --config config.yaml

# Baseline eval of the full model (no jumps) on MMLU + QuALITY
torchrun --nproc_per_node=8 eval_baseline.py

# Single-GPU inference / generation demo from a trained checkpoint
python inference.py            # edit CKPT_PATH / START_LAYER / END_LAYER at top of file

# Sanity check: manual layer-by-layer forward must match llm.model(...)
python common/verify_forward.py
```

`train.py` uses `fire`, so any `config.*` value can also be overridden on the CLI. There is no test suite; `common/verify_forward.py` is the closest thing to a correctness test and should be re-run whenever `JumpQwen.forward` or the manual forward logic changes.

## Architecture / control flow

- `train.py` → `JumpTrainer(config)` (`trainer.py`) → `JumpExperiment(config)` (`src/tuned_exp.py`).
- `JumpExperiment` owns all the heavy state: it loads the frozen base model (`llm.model` + `llm.lm_head`), builds `JumpQwen`, wraps **only the lenses** in `DistributedDataParallel`, splits params into Muon (ndim≥2) vs AdamW (ndim<2) groups, and builds the streaming dataloader.
- `JumpTrainer` runs the optimization loop with two optimizers — **Muon** for matrix params, **AdamW** for the rest — gradient accumulation (`accum_steps`), grad clipping at 1.0, periodic checkpointing, and inline MMLU/QuALITY eval.
- `data_stream.py` `StreamingDataset` streams SlimPajama-627B-DC shards from the HF hub (mixture weights in `DATA_PROBS`), tokenizes, packs into fixed `seq_len` blocks, and has a retry/reseed fallback for flaky streaming (logged to `streaming_fallbacks/`).

### Manual forward must mirror Qwen internals
`JumpQwen.forward` and `verify_forward.py` reimplement the base model's forward pass by hand: building the causal mask, computing RoPE position embeddings via `text_model.rotary_emb`, and selectively passing `attention_mask=None` for `linear_attention` layer types (read from `config.layer_types`). If the upstream Qwen architecture changes, this hand-rolled loop is the first thing that breaks.

### Checkpoints
`save_checkpoint` writes **only `lenses.module.state_dict()`** to `checkpoints/step_{step}.pt` — not the base model. `JumpQwen.from_checkpoint(ckpt, start_layer, end_layer, base_model, lm_head)` reconstructs a model by loading those lens weights; the start/end layers are **not stored in the checkpoint** and must be supplied by the caller (note the convention in `inference.py`'s filename `step_5000_17_21.pt` encoding layers 17→21).

## Known gotchas / unresolved references

- **The `X` package does not exist in this repo.** `trainer.py` and `eval_baseline.py` import `from X.eval.quality_loader import load_quality` and `from X.eval.utils import prepare_mmlu`. `train.py`/`eval_baseline.py` insert `parent.parent` (the `jumplens/` dir) onto `sys.path`, so an external `X/` package providing `eval.quality_loader` and `eval.utils` is expected to live there. This must be supplied for training/eval to run.
- `config.yaml` hardcodes an absolute `eval_log_file` path (`/home/jovyan/...`) — update it for the local environment.
- `prepare_mmlu` exists both in the (missing) `X.eval.utils` and locally in `eval/eval_utils.py`; the local copy is the readable reference for prompt formatting. `make_quality_prompt` is imported from the local `eval/eval_utils.py`.
- Code comments and docstrings are partly in Russian.
