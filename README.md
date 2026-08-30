# Sarcasm Understanding via Multimodal and Contextual Analysis

Evaluation framework for a five-condition study of sarcasm understanding, crossing **input
modality** (audio+text vs. text-only) with **conversational context** (absent, text, audio+text).
Twelve models, 867 evaluation items, 29 runs.

> **Dataset:** [`darksyntax0/conversational-sarcasm-benchmark`](https://huggingface.co/datasets/darksyntax0/conversational-sarcasm-benchmark)
> on HuggingFace. This repository contains **code only**.

## Experimental design

|                          | no context | context as **text** | context as **audio+text** |
|--------------------------|------------|---------------------|---------------------------|
| **target: audio + text** | Stage 1    | **Stage 2.5**       | Stage 2                   |
| **target: text only**    | Stage 3    | Stage 4             | *n/a*                     |

Stage 2.5 is a hybrid condition: it holds contextual information constant with respect to Stage 2
and removes only the context *audio* channel, isolating the cost attributable to processing a
second audio stream. It is also the only multimodal condition in which all five audio models are
treated identically (see the Audio-Flamingo-3 note below).

Contrasts: `S1↔S2` and `S3↔S4` isolate context (within-model); `S2↔S2.5` isolates the
context-audio channel; `S2.5↔S4` is the tightest modality contrast, since both receive context as
text and differ only in the target channel.

## Repository layout

```
stage1/              Stage 1  — target audio + text
stage2/              Stage 2  — context + target, both audio + text
stage2_5/            Stage 2.5— target audio + text, context as text
stage3_4/            Stages 3 & 4 — text-only (stage selected by --stage)
reference_scripts/   Standalone single-example scripts per audio model
scoring/             Label normaliser + unified scorer (reproduces paper tables)
scripts/             verify_setup.py pre-flight check
envs/                Pinned requirements for each conda environment
docs/                Full methodology documentation
```

## Hardware and precision requirements

Developed and run on **4 × NVIDIA Tesla V100-PCIE-32GB** (compute capability 7.0) with driver
460.91.03. Three constraints follow from this and will affect any replication:

- **CUDA 11.2 driver ceiling.** All environments use `torch 2.7.1+cu118`. CUDA 12.x/13.x wheels
  will not run on this driver; installing them is the single most likely setup failure.
- **flash-attention-2 is unsupported on Volta** (requires Ampere+). All models use `eager`
  attention.
- **Text models require bfloat16.** All seven text LLMs overflow numerically in float16 on this
  hardware and emit degenerate output (runs of `!` tokens). Audio models are stable in float16.

## Environments

Six conda environments, deliberately pinned to **different** `transformers` versions — these are
not accidental drift, and unifying them will break specific models:

| Environment | transformers | Used by |
|---|---|---|
| `qwen2-audio-env` | 5.16.0.dev0 | Qwen2-Audio, Qwen2.5, Llama-3.1, Qwen3, Gemma-2 |
| `audio-flamingo`  | 5.16.0.dev0 | Audio-Flamingo-3 |
| `voxtral`         | 5.14.1      | Voxtral-Mini, Mistral-7B (needs `mistral-common`) |
| `phi4-env`        | 4.48.0      | Phi-4-multimodal, Phi-3-mini |
| `phi4_mini_text`  | 4.49.0      | Phi-4-mini-instruct |
| `audio-reasoner`  | 4.48.0      | Audio-Reasoner (uses `ms-swift`, not raw transformers) |

```bash
conda create -n qwen2-audio-env python=3.10 -y
conda activate qwen2-audio-env
pip install torch==2.7.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu118
pip install -r envs/qwen2-audio-env.requirements.txt
```

Repeat per environment. `voxtral` uses Python 3.11; the rest use 3.10.

## Setup

```bash
# 1. Dataset
huggingface-cli download darksyntax0/conversational-sarcasm-benchmark \
    --repo-type dataset --local-dir ./data

# 2. Audio-Reasoner checkpoint (Stages 1, 2, 2.5 only)
huggingface-cli download zhifeixie/Audio-Reasoner --local-dir ./checkpoints/Audio-Reasoner-7B

# 3. Environment variables
export CONDA_ROOT=$HOME/miniconda          # root holding envs/<name>
export CONDA3_ROOT=$HOME/miniconda3        # only if voxtral lives in a second install
export SARCASM_DATA_ROOT=$PWD/data/samples
export AUDIO_REASONER_CKPT=$PWD/checkpoints/Audio-Reasoner-7B
export HF_HOME=/path/with/space            # model weights are large

# 4. Pre-flight check (no GPU, no inference)
python scripts/verify_setup.py
```

## Running

Every stage follows the same two-phase protocol: a small validation subset first, then the full
run behind an explicit confirmation flag. The full run is **never** auto-chained from validation.

```bash
conda activate benchmark-orchestrator

# Stage 1
cd stage1
python -m orchestrator.run_pilot                          # 18-sample deterministic pilot
python -m orchestrator.run_full --i-reviewed-the-pilot     # full 867

# Stage 2 / Stage 2.5
cd stage2      && python run_stage2.py    --limit 20
               && python run_stage2.py    --full --i-reviewed-the-validation
cd stage2_5    && python run_stage2_5.py  --limit 20
               && python run_stage2_5.py  --full --i-reviewed-the-validation

# Stages 3 and 4
cd stage3_4
python run_text_stage.py --stage 3 --limit 5
python run_text_stage.py --stage 3 --full --i-reviewed-the-validation
python run_text_stage.py --stage 4 --full --i-reviewed-the-validation
```

Each model runs as a **single long-lived subprocess** that loads once and iterates the manifest,
with per-sample timeouts, crash-safe incremental writes, resume-on-restart, a stall watchdog, and
dynamic GPU assignment queried immediately before each launch. Interrupted runs resume by
re-invoking the same command.

## Scoring

```bash
python scoring/score_all.py \
    --results-root stage1/outputs/results/full \
    --ground-truth stage1/outputs/manifests/full_ground_truth.jsonl
```

Scoring reads stored `results.jsonl` only — no GPU, no re-inference. Absent, unparseable, or
out-of-inventory predictions are scored **incorrect** rather than excluded, so all conditions are
evaluated on an identical item set.

## Model-specific notes

These cost real debugging time and are preserved here deliberately:

- **Qwen2-Audio** — the processor kwarg is `audio=` (singular) in transformers 5.x. The older
  `audios=` is *silently ignored*, so the model receives no audio yet still produces fluent,
  hallucinated output. Every audio script therefore asserts that audio tensors are present before
  generation.
- **Audio-Flamingo-3** — its processor enforces one audio per prompt (`len(text) == len(audio)`)
  and flattens conversations to a single prompt, so **two audio clips are impossible** in any
  arrangement. In Stage 2 it runs a documented reduced-modality variant, definitionally identical
  to Stage 2.5.
- **Phi-4-multimodal** — raises `Tensor.item() cannot be called on meta tensors` under
  transformers 5.x; pin that environment to 4.48.0.
- **Audio-Reasoner** — runs through `ms-swift`'s `PtEngine`, not raw transformers, and its
  `<THINK>/<RESPONSE>` structure never yields bare JSON; a `FINAL_JSON:` marker plus a
  CoT-section fallback is used.
- **Llama-3.1** — a gated repo. Pre-download it, then set `HF_HUB_OFFLINE=1`; if offline loading
  fails, check that `refs/main` exists in the HF cache directory.

## Output format

One JSON object per sample per model in `results.jsonl`:

```json
{"sample_id": "...", "model_name": "...", "stage": "2.5",
 "is_sarcastic": true, "predicted_category": "IRONY",
 "sarcasm_source": "...", "sarcasm_target": "...", "confidence": 0.8,
 "explanation": "...", "reasoning_steps": ["..."],
 "raw_output": "...", "parse_error": null, "salvaged": false,
 "latency_seconds": 5.1, "gpu_mem_reserved_mb": 16900.0, "timestamp": "..."}
```

`raw_output` is always retained, so parsing decisions are auditable and re-executable without
re-running inference. Stage 1 uses a four-field prediction subset (category, confidence,
explanation, reasoning_steps).

## License

Code in this repository: **MIT** (see `LICENSE`).
The dataset is distributed separately under **CC BY 4.0** on HuggingFace.

## Citation

See `CITATION.cff`.
