# Stage 3 & Stage 4 Sarcasm Benchmark — Framework Documentation

**The text-only control arm of the four-stage experiment.**

| | TARGET ONLY | CONTEXT + TARGET |
|---|---|---|
| **AUDIO + TEXT** | Stage 1 | Stage 2 |
| **TEXT ONLY** | **STAGE 3 (this document)** | **STAGE 4 (this document)** |

- **Stage 3:** `target.txt` → sarcasm analysis (no context, no audio)
- **Stage 4:** `context.txt` + `target.txt` → sarcasm analysis (no audio)

The 2×2 design supports four contrasts: **S1↔S2** (context effect, multimodal), **S3↔S4** (context
effect, text-only), **S1↔S3** (modality effect, target-only), **S2↔S4** (modality effect,
contextual).

**Location:** `stage3_4/`
**Status:** Complete — 7 models × 867 samples × 2 stages = **12,138 predictions**, all exit code 0.
Stages 1 and 2 were treated as frozen and were never modified.

---

## 1. Dataset, split, and ground truth

Stages 3 & 4 use the **identical 867-sample split** as Stages 1 & 2. `samples.py` applies the same
validity predicate — all five artifacts present (`context.wav`, `context.txt`, `target.wav`,
`target.txt`, `sarcasm_analysis.json`) and both transcripts non-empty — so all four stages evaluate
exactly the same items. The audio files must merely *exist* to keep the split identical; **no audio
is ever read** in Stage 3/4.

**Ground truth (unchanged, not re-derived):**
```json
{"category": "IRONY", "sarcasm_source": "Speaker", "sarcasm_target": "Preference shift",
 "explanation": "...", "sample_id": "-C1L4vuYaeM_000"}
```
Label distribution (867): SARCASM 484, IRONY 256, SATIRE 45, OVERSTATEMENT 35, UNDERSTATEMENT 24,
RHETORICAL_QUESTION 23.

---

## 2. Directory structure

```
sarcasm_ICL/stage_text/
├── samples.py            (58)  same-867 split; loads context.txt / target.txt / GT only
├── text_prompt.py        (74)  Stage 3 & Stage 4 prompts + output schema
├── json_extract.py      (224)  VENDORED read-only copy of the proven parser
├── run_text_stage.py    (173)  orchestrator: --stage {3,4}, manifest, idle-GPU scheduler, watchdog
├── configs/text_models.yaml (53)  env / script / hf_id / timeout / max_new_tokens / tier
├── infer/
│   ├── _text_base.py    (180)  shared loop: SIGALRM timeout, resume, fsync, parse + field salvage,
│   │                           per-row reproducibility record, shared prompt construction
│   ├── _common_text.py   (67)  shared causal-LM load + chat-template generation
│   ├── qwen2_5_7b.py      (5)  ─┐
│   ├── llama3_1_8b.py     (5)   │ per-model config only (model id, env, dtype,
│   ├── mistral_7b_v03.py  (6)   │ trust_remote_code, template kwargs)
│   ├── phi4_mini.py       (5)   │
│   ├── qwen3_8b.py        (9)   │ (+ enable_thinking=False)
│   ├── gemma2_9b.py       (7)   │ (+ bfloat16 note)
│   └── phi3_mini.py       (5)  ─┘
└── results/
    ├── manifests/  manifest_{pilot,full}.jsonl, ground_truth_{pilot,full}.jsonl
    ├── stage3/<model>/  results.jsonl, run.log, run_meta.json
    └── stage4/<model>/  results.jsonl, run.log, run_meta.json
```

Stage 3 and Stage 4 share one codebase; the stage is a **runtime flag** (`--stage 3|4`) that selects
the prompt. This guarantees that everything except the presence of the context utterance is
identical between the two stages.

---

## 3. Models and environments

Seven instruction-tuned text LLMs. Environment assignment follows the existing mapping; only one
new environment was created (`phi4_mini_text`), and **no existing audio environment was modified**,
because Stages 1 & 2 depend on their current configuration.

| # | Model | HF checkpoint | Params | Conda env | transformers | dtype | Tier |
|---|---|---|---|---|---|---|---|
| 1 | Qwen2.5-7B-Instruct | `Qwen/Qwen2.5-7B-Instruct` | 7.6 B | `qwen2-audio-env` | 5.16.0.dev0 | bfloat16 | 1 |
| 2 | Llama-3.1-8B-Instruct | `meta-llama/Llama-3.1-8B-Instruct` | 8.0 B | `qwen2-audio-env` | 5.16.0.dev0 | bfloat16 | 1 |
| 3 | Mistral-7B-Instruct-v0.3 | `mistralai/Mistral-7B-Instruct-v0.3` | 7.3 B | `voxtral` | 5.14.1 | bfloat16 | 1 |
| 4 | Phi-4-mini-instruct | `microsoft/Phi-4-mini-instruct` | 3.8 B | **`phi4_mini_text`** (new) | 4.49.0 | bfloat16 | 1 |
| 5 | Qwen3-8B | `Qwen/Qwen3-8B` | 8.2 B | `qwen2-audio-env` | 5.16.0.dev0 | bfloat16 | 1 |
| 6 | Gemma-2-9B-it | `google/gemma-2-9b-it` | 9.2 B | `qwen2-audio-env` | 5.16.0.dev0 | bfloat16 | 2 |
| 7 | Phi-3-mini-4k-instruct | `microsoft/Phi-3-mini-4k-instruct` | 3.8 B | `phi4-env` | 4.48.0 | bfloat16 | optional |

**Checkpoint-name disambiguation** (stated explicitly; no silent substitution): "Qwen3-8B-Instruct"
resolves to **`Qwen/Qwen3-8B`** (Qwen3 publishes no separate `-Instruct` repo; the base is
instruction-tuned); "Gemma-2-9B-Instruct" resolves to **`google/gemma-2-9b-it`** (Google's
instruction suffix); "Phi-4-Mini-Instruct" resolves to `microsoft/Phi-4-mini-instruct`.

**New environment:** `phi4_mini_text` = Python 3.10 + `torch 2.7.1+cu118` + **`transformers==4.49.0`**
+ accelerate/sentencepiece/protobuf/einops.

### Model-specific handling

- **All models — bfloat16, not float16.** Every model was released in bf16 and **overflows in fp16
  on this hardware**, producing degenerate output (streams of `!` tokens interleaved with real
  text). Verified directly: identical prompt, bf16 → valid JSON, fp16 → garbage. All seven run in
  bf16 (emulated on Volta, numerically correct).
- **Qwen3-8B** — `enable_thinking=False` in the chat template; the default thinking mode emits
  `<think>…</think>` preambles that would make outputs non-comparable to the other models.
- **Gemma-2 / Mistral** — no system role in their chat templates. Handled uniformly (below).
- **Phi-3-mini / Phi-4-mini** — `trust_remote_code=True`.
- **Attention** — `eager` everywhere (flash-attention-2 requires Ampere+; hardware is Volta).

**Uniform message structure.** To keep the comparison fair, every model receives the *complete,
self-contained task* in a **single user turn with no system message**, identical wording, and
**greedy decoding** (`do_sample=False`). Only the model's own chat template differs.

---

## 4. Prompt design and fair-comparison guardrails

Both prompts are built by `text_prompt.py` and share an identical task/schema block; they differ
**only** in whether the context utterance is present.

**Stage 3:**
```
You are analyzing a spoken utterance (given as its transcription) for sarcasm.

TARGET UTTERANCE:
{target_text}

Analyze the TARGET utterance for sarcasm. <shared task + schema block>
```

**Stage 4:**
```
You are analyzing a spoken conversation (given as transcriptions) for sarcasm.

CONTEXT UTTERANCE (what was said just before):
{context_text}

TARGET UTTERANCE (the line to analyze):
{target_text}

Analyze the TARGET utterance using the provided conversational CONTEXT. <shared task + schema block>
```

The shared block asks for the same five judgements as Stages 1–2 (sarcastic yes/no, category,
source, target, explanation), lists the six categories plus `OTHER`, imposes strict-JSON rules, and
shows a **placeholder-only** format template (a realistic worked example was previously copied
verbatim by a model, so all example values are explicit placeholders).

**Guardrails enforced structurally, not by convention:**
- Stage 3 **never** receives context — `build_stage3_prompt()` does not accept a context argument.
- Stage 4 **never** receives audio — no audio path exists anywhere in this codebase.
- No ground-truth field ever enters a prompt; ground truth is written to a **separate**
  `ground_truth_*.jsonl` that the model-side code never reads.
- No information beyond the defined context utterance (no future turns, no full transcript).
- Dataset split, task wording, category set, output schema, and decoding are unchanged from
  Stages 1–2.

---

## 5. Output schema

Preserves the Stage 1/2 semantic fields (superset of Stage 1's), enabling cross-stage evaluation:

```json
{"is_sarcastic": true, "predicted_category": "IRONY", "sarcasm_source": "Speaker",
 "sarcasm_target": "Preference shift", "confidence": 0.8,
 "explanation": "one sentence", "reasoning_steps": ["step one", "step two"]}
```

Field mapping to ground truth: `predicted_category` → `category`, `sarcasm_source` →
`sarcasm_source`, `sarcasm_target` → `sarcasm_target`, `explanation` → `explanation`. No new
ground-truth schema was introduced. `confidence` is a self-reported generative-model number, **not
a calibrated probability**.

**Per-row record** (`results/stage{3,4}/<model>/results.jsonl`): `sample_id`, `model_name`,
**`stage`**, the seven prediction fields, `raw_output`, `parse_error`, `salvaged`, `error`,
`latency_seconds`, `gpu_mem_allocated_mb`, `gpu_mem_reserved_mb`, `timestamp`, plus the full
reproducibility block — `model_id`, `conda_env`, `transformers_version`, `torch_version`, `dtype`,
`attn_implementation`, `gen_max_new_tokens`, `gen_do_sample`, `gen_template_kwargs`.

---

## 6. Execution architecture and JSON robustness

Deliberately mirrors Stage 1/2 rather than inventing new orchestration: **one long-lived subprocess
per model** (load once, iterate the manifest), per-sample `SIGALRM` timeout, append + `flush` +
`fsync` per row, resume by skipping already-written `sample_id`s, a process-level stall watchdog,
and live `nvidia-smi` idle-GPU selection with per-run GPU reservation (≥18 GB free, ≤40% util) so
models never double-book a card on the shared cluster.

**JSON recovery ladder** (vendored parser + field salvage): strict JSON → Python dict-literal →
backslash-unescape repair → balanced-brace scan → bracket/key-quote repair → **field-level salvage**
(regex extraction of the scalar prediction fields when the whole object is unrecoverable). Salvaged
rows are flagged `"salvaged": true` so the provenance is auditable and can be reported separately.

---

## 7. Validation before the full run

A deterministic 5-sample subset of the real dataset was run through the complete pipeline for
**all 7 models × both stages** before any full run. Every model was checked for: environment,
model load, tokenizer, transcript loading, prompt construction, generation, structured-output
parsing, and result persistence.

**Result: 7/7 models passed both stages** — 5/5 usable, 0 errors, valid categories, all fields
populated, ~4–9 s/sample.

Two real defects were found **and fixed during validation** (both would have silently corrupted a
full run):
1. **fp16 numerical overflow** across all text models → switched to bf16 (§3).
2. **Llama-3.1 gated-repo 403** at load time. The 15 GB cache was complete but missing the
   `refs/main` pointer that offline resolution needs. Repaired the pointer and set
   `HF_HUB_OFFLINE=1` for all runs — all seven checkpoints are pre-cached, so runs are now
   hub-independent and more reproducible.

---

## 8. Results (full runs, 867 samples per model per stage)

### Stage 3 — target text only

| Model | Usable | Clean JSON | Salvaged | is_sarc / src / tgt | Timeouts | Parse err | Mean latency | GPU-h |
|---|---|---|---|---|---|---|---|---|
| phi4_mini | 867 (100%) | 867 | 0 | 867 / 831 / 831 | 0 | 0 | 4.4 s | 1.07 |
| qwen3_8b | 867 (100%) | 867 | 0 | 867 / 867 / 867 | 0 | 0 | 6.7 s | 1.62 |
| llama3_1_8b | 867 (100%) | 864 | 3 | 867 / 867 / 866 | 0 | 0 | 6.9 s | 1.65 |
| phi3_mini | 867 (100%) | 863 | 4 | 867 / 867 / 867 | 0 | 0 | 6.7 s | 1.60 |
| mistral_7b_v03 | 867 (100%) | 847 | 20 | 867 / 843 / 844 | 0 | 0 | 13.3 s | 3.19 |
| qwen2_5_7b | 867 (100%) | 842 | 25 | 865 / 867 / 867 | 0 | 0 | 4.2 s | 1.02 |
| gemma2_9b | 867 (100%) | 776 | 91 | 867 / 867 / 867 | 0 | 0 | 7.3 s | 1.75 |

### Stage 4 — context + target text

| Model | Usable | Clean JSON | Salvaged | is_sarc / src / tgt | Timeouts | Parse err | Mean latency | GPU-h |
|---|---|---|---|---|---|---|---|---|
| phi4_mini | 867 (100%) | 867 | 0 | 867 / 855 / 855 | 0 | 0 | 4.6 s | 1.11 |
| qwen3_8b | 867 (100%) | 867 | 0 | 867 / 867 / 867 | 0 | 0 | 11.8 s | 2.85 |
| phi3_mini | 867 (100%) | 867 | 0 | 867 / 867 / 867 | 0 | 0 | 8.3 s | 1.99 |
| llama3_1_8b | 867 (100%) | 863 | 4 | 867 / 867 / 865 | 0 | 0 | 7.1 s | 1.71 |
| mistral_7b_v03 | 867 (100%) | 844 | 23 | 867 / 855 / 856 | 0 | 0 | 8.0 s | 1.92 |
| qwen2_5_7b | 867 (100%) | 803 | 64 | 855 / 867 / 864 | 0 | 0 | 5.0 s | 1.21 |
| gemma2_9b | 867 (100%) | 700 | 167 | 867 / 867 / 867 | 0 | 0 | 7.4 s | 1.78 |

**Totals:** 12,138 inferences, **100% usable**, **0 timeouts**, **0 unrecoverable parse errors**,
≈ **24.5 GPU-hours** (Stage 3 ≈ 11.9, Stage 4 ≈ 12.6).

**Observations for the write-up.**
- **Gemma-2 is the weakest JSON formatter** (91 salvaged in Stage 3, 167 in Stage 4 = 19%). All
  rows still yielded usable predictions, but the `salvaged` flag should be reported for
  transparency, and salvaged rows lack `reasoning_steps`.
- **Salvage rates rise from Stage 3 to Stage 4 for most models** (qwen2.5 25→64, gemma2 91→167),
  consistent with longer context prompts degrading strict format adherence — a formatting effect
  that should not be conflated with task accuracy.
- Prediction fields are near-fully populated in both stages, so all four cross-stage comparisons
  have complete data on category *and* source/target/explanation.

---

## 9. Reproducibility

- **Stages 1 & 2 untouched.** Stage 3/4 lives in its own tree, writes only under
  `stage_text/results/`, and reuses Stage 1/2 logic solely through a **vendored read-only copy** of
  the JSON parser — no imports from, and no edits to, the frozen trees.
- **Only one new environment** (`phi4_mini_text`); no existing audio environment was modified or
  upgraded.
- **Deterministic**: sorted sample-id traversal; greedy decoding; pilot subset = first N valid ids.
- **Offline model loading** (`HF_HUB_OFFLINE=1`) — runs do not depend on hub availability or gating.
- **Full configuration recorded on every row** (model id, env, transformers/torch versions, dtype,
  attention implementation, generation parameters, stage), plus `run_meta.json` per model per stage.
- **Hardware:** 4 × Tesla V100-PCIE-32GB (CC 7.0), driver 460.91.03, `torch 2.7.1+cu118`, bf16,
  eager attention.

---

## 10. How to reproduce

```bash
conda activate benchmark-orchestrator
cd stage3_4

# validation subset (both stages, all models)
python run_text_stage.py --stage 3 --limit 5
python run_text_stage.py --stage 4 --limit 5

# full runs (tripwire-gated)
python run_text_stage.py --stage 3 --full --i-reviewed-the-validation
python run_text_stage.py --stage 4 --full --i-reviewed-the-validation
```

---

## 11. Next step — cross-stage evaluation

All four stages are now complete (S1: 867×5, S2: 867×5, S3: 867×7, S4: 867×7). The remaining
component is a **shared scoring layer** applying one category normaliser (models emit `'satire'`,
`'SARCASTIC'`, etc.) across all four stages, producing accuracy / macro-F1 / per-class metrics plus
`is_sarcastic`, source/target, and explanation-similarity scores, to yield the four planned
contrasts: S1↔S2, S3↔S4, S1↔S3, S2↔S4. Note that S1 predicts category only, so S1↔S3 comparisons
should be restricted to the category field.
