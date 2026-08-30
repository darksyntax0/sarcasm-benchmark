# A Four-Stage Benchmark for Multimodal Sarcasm Understanding — Methodology

**Complete methodology for Stages 1–4.** This document is the single reference for the
experimental design, dataset, models, prompting, execution architecture, validation protocol,
and reproducibility guarantees of the whole benchmark. Per-stage implementation detail lives in
`STAGE1_FRAMEWORK.md`, `STAGE2_FRAMEWORK.md`, and `STAGE3_4_FRAMEWORK.md`.

---

## 1. Experimental design

The benchmark is a **2 × 2 factorial design** crossing *modality* (audio+text vs. text-only)
with *conversational context* (target-only vs. context+target). Every cell is evaluated on the
same 867 items with the same task, label set, and output schema.

|  | **TARGET ONLY** | **CONTEXT + TARGET** |
|---|---|---|
| **AUDIO + TEXT** | **Stage 1** — `target.wav` + `target.txt` | **Stage 2** — `context.wav` + `context.txt` + `target.wav` + `target.txt` |
| **TEXT ONLY** | **Stage 3** — `target.txt` | **Stage 4** — `context.txt` + `target.txt` |

**Planned contrasts:**

| Contrast | Isolates | Research question |
|---|---|---|
| S1 ↔ S2 | context, modality held at audio+text | Does conversational context improve sarcasm understanding in the multimodal speech setting? |
| S3 ↔ S4 | context, modality held at text-only | Does the same context effect hold when only transcription is available? |
| S1 ↔ S3 | modality, context held at target-only | Does the speech signal add information beyond its transcription? |
| S2 ↔ S4 | modality, context held at context+target | Does multimodal contextual understanding exceed text-only contextual understanding? |

Stages 1 and 2 use five **audio-language models**; Stages 3 and 4 use seven **text-only
instruction LLMs**. Within a modality arm the model set is held fixed across the context
manipulation, so S1↔S2 and S3↔S4 are within-model comparisons. S1↔S3 and S2↔S4 are
cross-architecture comparisons and must be interpreted at the level of the modality arm rather
than of matched model pairs.

**Scale:** 4 stages × 867 samples × (5 or 7) models = **20,808 model inferences**, ≈ **57.1
GPU-hours**.

---

## 2. Dataset

### 2.1 Source and unit of analysis

The corpus comprises 1,106 sample directories derived from conversational video, each a
`{video_id}_{clip_index}` unit containing up to five artifacts: `target.wav` / `target.txt`
(the utterance to analyse), `context.wav` / `context.txt` (the immediately preceding utterance),
and `sarcasm_analysis.json` (ground truth).

### 2.2 Validity predicate and the shared split

A single validity predicate defines the evaluation set, computed fresh at run time and never
hardcoded. A sample is valid iff all five artifacts are present, both transcripts are non-empty
after whitespace stripping, and the ground-truth JSON parses with a category in the canonical
label set.

| Filter (applied in order) | Excluded | Remaining |
|---|---|---|
| Total sample directories | — | 1,106 |
| Missing `target.wav` | 32 | 1,074 |
| Missing or empty `target.txt` | 164 | 910 |
| Missing `sarcasm_analysis.json` | 43 | 867 |
| Malformed JSON / schema / label | 0 | **867** |

**All four stages use these same 867 items.** The text stages require the audio files to *exist*
solely to preserve split identity; they never read them. This is what makes the cross-stage
contrasts item-matched rather than merely distributionally comparable.

### 2.3 Ground truth

```json
{"category": "IRONY", "sarcasm_source": "Speaker", "sarcasm_target": "Preference shift",
 "explanation": "...", "sample_id": "-C1L4vuYaeM_000"}
```

Verified 100% schema-uniform across all 867 items, with zero parse failures.

**Label distribution (n = 867).** The task is a six-way classification over a markedly imbalanced
label set — a property that makes macro-F1 the primary metric and renders per-class support a
necessary reporting element.

| Label | n | % |
|---|---|---|
| SARCASM | 484 | 55.8 |
| IRONY | 256 | 29.5 |
| SATIRE | 45 | 5.2 |
| OVERSTATEMENT | 35 | 4.0 |
| UNDERSTATEMENT | 24 | 2.8 |
| RHETORICAL_QUESTION | 23 | 2.7 |

Audio is uniformly mono, 16 kHz, 16-bit PCM (verified across all files).

---

## 3. Models and execution environments

Twelve model configurations across two modality arms. Each model executes inside its own conda
environment, invoked **by interpreter path directly** rather than via `conda activate` or
`conda run`, both of which proved fragile for unattended multi-hour runs.

### 3.1 Audio-language models (Stages 1–2)

| Model | Checkpoint | Environment | transformers | Notes |
|---|---|---|---|---|
| Qwen2-Audio-7B-Instruct | `Qwen/Qwen2-Audio-7B-Instruct` | `qwen2-audio-env` | 5.16.0.dev0 | HF-native |
| Audio-Flamingo-3 | `nvidia/audio-flamingo-3-hf` | `audio-flamingo` | 5.16.0.dev0 | single-audio only (§7.2) |
| Phi-4-multimodal-instruct | `microsoft/Phi-4-multimodal-instruct` | `phi4-env` | 4.48.0 | `trust_remote_code` |
| Voxtral-Mini-3B-2507 | `mistralai/Voxtral-Mini-3B-2507` | `voxtral` | 5.14.1 | `mistral-common` |
| Audio-Reasoner-7B | local checkpoint | `audio-reasoner` | 4.48.0 | ms-swift `PtEngine`; CoT |

### 3.2 Text-only instruction LLMs (Stages 3–4)

| Model | Checkpoint | Params | Environment | transformers |
|---|---|---|---|---|
| Qwen2.5-7B-Instruct | `Qwen/Qwen2.5-7B-Instruct` | 7.6 B | `qwen2-audio-env` | 5.16.0.dev0 |
| Llama-3.1-8B-Instruct | `meta-llama/Llama-3.1-8B-Instruct` | 8.0 B | `qwen2-audio-env` | 5.16.0.dev0 |
| Mistral-7B-Instruct-v0.3 | `mistralai/Mistral-7B-Instruct-v0.3` | 7.3 B | `voxtral` | 5.14.1 |
| Phi-4-mini-instruct | `microsoft/Phi-4-mini-instruct` | 3.8 B | `phi4_mini_text` | 4.49.0 |
| Qwen3-8B | `Qwen/Qwen3-8B` | 8.2 B | `qwen2-audio-env` | 5.16.0.dev0 |
| Gemma-2-9B-it | `google/gemma-2-9b-it` | 9.2 B | `qwen2-audio-env` | 5.16.0.dev0 |
| Phi-3-mini-4k-instruct | `microsoft/Phi-3-mini-4k-instruct` | 3.8 B | `phi4-env` | 4.48.0 |

**Checkpoint disambiguation.** Three requested model names have no verbatim HF repository and were
resolved to canonical checkpoints, stated explicitly to preclude any appearance of silent
substitution: *Qwen3-8B-Instruct* → `Qwen/Qwen3-8B` (Qwen3 publishes no separate `-Instruct`
variant; the base model is instruction-tuned); *Gemma-2-9B-Instruct* → `google/gemma-2-9b-it`;
*Phi-4-Mini-Instruct* → `microsoft/Phi-4-mini-instruct`.

### 3.3 Hardware and numerical precision

4 × NVIDIA Tesla V100-PCIE-32GB (compute capability 7.0, Volta), driver 460.91.03 (CUDA 11.2
ceiling), `torch 2.7.1+cu118` in every environment. Two hardware constraints shape the
configuration and are reported because they materially affect replication:

1. **flash-attention-2 is unusable on Volta** (requires Ampere+). All models use `eager`
   attention.
2. **Text models require bfloat16.** All seven text LLMs are released in bf16 and **overflow
   numerically in fp16 on this hardware**, emitting degenerate output (runs of `!` tokens
   interleaved with partial text). This was diagnosed by controlled comparison — identical prompt
   and decoding, bf16 producing valid JSON and fp16 producing degenerate text — and all text
   models therefore run in bf16. Audio models run in fp16, in which they are numerically stable.

---

## 4. Task formulation and prompting

### 4.1 Shared task

Across all four stages the model performs one task: given the target utterance (and, in Stages 2
and 4, the preceding context utterance), decide whether the target is sarcastic, assign one of six
sarcasm categories (or `OTHER`), identify the sarcasm source and target, and produce an
explanation. Category names, task wording, strict-JSON requirements, and decoding are held
constant; only the evidence presented to the model varies by cell.

### 4.2 Stage-specific prompt content

| Stage | Prompt contains | Prompt never contains |
|---|---|---|
| 1 | target audio + target transcript | context (either modality) |
| 2 | context audio + context transcript + target audio + target transcript | — |
| 3 | target transcript | context, any audio |
| 4 | context transcript + target transcript | any audio |

Stage 3 and Stage 4 prompts are produced by the same module and differ *only* by the presence of a
`CONTEXT UTTERANCE:` block; the stage is a runtime flag, so no other text differs between them.
Stages 1 and 2 likewise share a task/schema block.

### 4.3 Fair-comparison guardrails

These are enforced structurally rather than by convention, so that a cell cannot become more
informative than its design specifies:

- `build_stage3_prompt()` **does not accept a context argument** — Stage 3 cannot receive context.
- No audio path exists anywhere in the Stage 3/4 codebase — the text stages cannot receive audio.
- Ground truth is written to a **separate file that model-side code never reads**; no label,
  source, target, or reference explanation can enter a prompt.
- No information beyond the single defined context utterance is exposed (no later turns, no full
  transcript).
- **Greedy decoding** (`do_sample=False`) everywhere; no sampling variance.
- Every model receives the complete self-contained task in a **single user turn with no system
  message**, so message structure is identical across architectures with differing chat templates
  (Gemma-2 and Mistral, for example, define no system role).

### 4.4 Prompt-contamination control

An initial prompt revision included a realistic worked example of the required JSON. Audio-
Flamingo-3 reproduced that example verbatim on **100% of pilot samples**, ignoring the audio
entirely. All format templates were consequently rewritten to contain **only explicit
placeholders** (`"<one category name above>"`), with an instruction not to copy them. This
qualifies as a validity threat that was detected and eliminated before any full run.

### 4.5 Output schema

Stages 2, 3 and 4 share a seven-field schema; Stage 1 uses a four-field subset.

```json
{"is_sarcastic": true, "predicted_category": "IRONY", "sarcasm_source": "Speaker",
 "sarcasm_target": "Preference shift", "confidence": 0.8,
 "explanation": "one sentence", "reasoning_steps": ["step one", "step two"]}
```

| Prediction field | Ground-truth field | Stage 1 | Stages 2–4 |
|---|---|---|---|
| `predicted_category` | `category` | ✓ | ✓ |
| `explanation` | `explanation` | ✓ | ✓ |
| `sarcasm_source` | `sarcasm_source` | — | ✓ |
| `sarcasm_target` | `sarcasm_target` | — | ✓ |
| `is_sarcastic`, `confidence`, `reasoning_steps` | (no GT counterpart) | partial | ✓ |

**Consequence for analysis:** the **S1 ↔ S3 contrast must be restricted to `predicted_category`**,
the only prediction field common to both. S2 ↔ S4 supports the full field set. No new ground-truth
schema was introduced at any stage.

---

## 5. Execution architecture

All four stages share one orchestration design, reimplemented per stage rather than shared by
import so that each stage remains independently reproducible.

**One long-lived subprocess per model.** Each model loads once and iterates the entire manifest.
The rejected alternative — environment activation per sample — would have reloaded a multi-gigabyte
checkpoint 867 times per model.

**Reliability mechanisms:**

- **Per-sample timeout.** `SIGALRM` wrapped tightly around the single generation call, cancelled in
  a `finally` block so it cannot leak into the next sample's timing. One pathological sample cannot
  stall a run.
- **Crash-safe incremental output.** Each result is appended and `flush()` + `fsync()`-ed
  immediately, so partial progress survives an OOM-kill or node event and is inspectable mid-run.
- **Resume.** On restart a model skips `sample_id`s already present in its output file.
- **Process-level stall watchdog.** The orchestrator terminates a subprocess whose results file has
  stopped growing, catching true hangs (e.g. CUDA deadlock) that an in-process signal cannot
  interrupt; recovery is re-invocation, which resumes.
- **Dynamic GPU scheduling.** `nvidia-smi` is queried immediately before every launch — never
  cached — to select a GPU meeting free-memory and utilisation thresholds, with per-run reservation
  so two models cannot race onto the same device. This is necessary because the cluster is shared
  with other users whose jobs appear and disappear mid-run.
- **Offline model loading** (`HF_HUB_OFFLINE=1`) for the text stages: all checkpoints are
  pre-cached, so runs neither depend on hub availability nor are affected by repository gating.

**Silent-failure guard (audio stages).** Because one processor API change caused audio to be
accepted-and-ignored while still producing fluent output (§7.1), every audio model script asserts,
after constructing processor inputs and before generation, that audio tensors are actually present
with plausible shape. Violations are recorded as a distinct `audio_input_error` rather than passing
silently. This is the mechanism that lets us assert the audio stages genuinely consumed audio.

---

## 6. Output parsing

Generative models frequently emit non-strict JSON. Rather than discard such samples, recovery
proceeds through an ordered ladder, and the recovery path used is recorded per sample:

1. **Strict JSON** (`json.loads`).
2. **Python dict-literal** (`ast.literal_eval`) — single-quoted dict-repr output.
3. **Backslash-unescape repair** — models that double-escape quotes.
4. **Balanced-brace scanning** — extracts a trailing JSON object from surrounding prose or markup;
   essential for Audio-Reasoner, whose `<THINK>…</THINK><RESPONSE>…</RESPONSE>` structure is
   trained-in and never disappears. A `FINAL_JSON:` marker fast-path is tried first for that model.
5. **Bracket / key-quote repair** — closes dropped trailing delimiters, restores a missing opening
   quote on a key.
6. **Field-level salvage** (Stages 2–4) — when the object as a whole is unrecoverable, the scalar
   prediction fields are extracted directly. Applies to models emitting dict-repr output whose
   free-text fields contain nested apostrophes that defeat both parsers while the scalar fields
   remain unambiguous.

Rows recovered at step 6 are flagged **`"salvaged": true`**, so salvage-dependent results can be
reported separately or excluded in a sensitivity analysis. The complete `raw_output` is retained
for every sample regardless of parse path, making all parsing decisions auditable and re-runnable
post hoc. Category strings are normalised at scoring time for case and spacing, plus one
evidence-based alias (`SARCASTIC` → `SARCASM`, an adjective/noun form observed in real output).

---

## 7. Validation protocol and diagnosed defects

### 7.1 Protocol

No stage proceeded to a full run without passing a small-scale validation on **real dataset
samples**, checking environment, model load, tokenizer/processor, artifact loading, prompt
construction, generation, structured-output parsing, and result persistence. Full runs are gated
behind an explicit confirmation flag and never auto-chain from validation.

For the audio stages, validation additionally performed **tensor-level verification that the
intended audio actually reached the model** — a stricter criterion than "generation succeeded",
adopted because of the defect below.

### 7.2 Defects detected during validation

Every item below was found *before* the corresponding full run and would have silently corrupted
results:

| # | Stage(s) | Defect | Resolution |
|---|---|---|---|
| 1 | 1, 2 | Qwen2-Audio processor kwarg renamed `audios=` → `audio=`; the obsolete name is **silently ignored**, so the model received **no audio** yet produced fluent, plausible output | Corrected kwarg; added the tensor-presence assertion now applied to all audio models |
| 2 | 1, 2 | Audio-Flamingo-3 loaded without an explicit dtype → fp32 → CPU offload → every sample timed out | Explicit fp16 + feature-tensor dtype cast |
| 3 | 1 | Phi-4-multimodal raised `Tensor.item() cannot be called on meta tensors` under transformers 5.16.0.dev0 | Pinned that environment to transformers 4.48.0 |
| 4 | 1 | Audio-Reasoner truncated before emitting an answer under a 512-token budget | Raised to 900 tokens; added `FINAL_JSON:` marker and CoT-section fallback |
| 5 | 1, 2 | Prompt's realistic worked example copied verbatim by AF3 on 100% of pilot samples | Replaced all examples with explicit placeholders (§4.4) |
| 6 | 2 | **Audio-Flamingo-3 cannot accept two audio inputs.** Its processor enforces `len(text) == len(audio)` and `apply_chat_template` flattens any conversation to one prompt, so both single-turn and multi-turn two-audio requests are rejected | See §7.3 — reported as a limitation, not worked around |
| 7 | 2 | Over-strict validation assertion: Voxtral chunks audio into 30 s windows, so feature rows equal total chunks, not exactly two | Assertion corrected to ≥ 2 (a harness defect, not a model defect) |
| 8 | 3, 4 | **All seven text models overflow in fp16**, emitting degenerate output | Switched to bf16 after controlled comparison (§3.3) |
| 9 | 3, 4 | Llama-3.1 failed to load: gated-repo 403. Cache was complete but missing the `refs/main` pointer offline resolution requires | Repaired pointer; enabled `HF_HUB_OFFLINE=1` for all text runs |

### 7.3 Audio-Flamingo-3 in Stage 2 — a reported limitation

Defect 6 is a genuine capability limit of the deployed model, not an implementation shortcoming: no
prompt arrangement admits two audio clips. Rather than exclude the model or disguise the gap,
Stage 2 runs AF3 in an explicitly documented **reduced-modality variant** — target audio plus
context supplied *as transcription text*. The prompt states this truthfully, and the model registry
records `native_two_audio: false`.

**AF3's Stage 2 results are therefore not directly comparable to the four native two-audio
models and must be footnoted wherever Stage 2 is reported.**

---

## 8. Run completion and reliability

The tables below report **pipeline reliability**, not task accuracy: how many samples produced a
usable prediction, and by which recovery path. Accuracy against ground truth is produced by the
cross-stage scoring layer (§10).

### 8.1 Stage 1 — target audio + text (5 models)

| Model | Usable | Timeouts | Parse errors | Mean latency | GPU-h |
|---|---|---|---|---|---|
| voxtral | 866 / 867 (99.9%) | 0 | 1 | 3.1 s | 0.75 |
| phi4_multimodal | 864 / 867 (99.7%) | 0 | 3 | 13.3 s | 3.21 |
| audio_reasoner | 837 / 867 (96.5%) | 0 | 0 | 34.2 s | 8.24 |
| qwen2_audio | 799 / 867 (92.2%) | 0 | 68 | 4.4 s | 1.06 |
| audio_flamingo3 | 252 / 867 (29.1%) | 0 | 613 | 6.2 s | 1.50 |

### 8.2 Stage 2 — context + target audio + text (5 models)

| Model | Usable | Salvaged | Audio errors | Mean latency | GPU-h |
|---|---|---|---|---|---|
| qwen2_audio | 867 / 867 (100%) | 709 | 0 | 5.1 s | 1.22 |
| voxtral_mini | 861 / 867 (99.3%) | 1 | 0 | 5.8 s | 1.39 |
| phi4_multimodal | 858 / 867 (99.0%) | 3 | 0 | 11.5 s | 2.77 |
| audio_flamingo3 † | 847 / 867 (97.7%) | 22 | 20 | 6.0 s | 1.45 |
| audio_reasoner | 804 / 867 (92.7%) | 0 | 0 | 46.0 s | 11.09 |

† reduced-modality variant (§7.3).

### 8.3 Stage 3 — target text only (7 models)

| Model | Usable | Salvaged | Mean latency | GPU-h |
|---|---|---|---|---|
| phi4_mini | 867 / 867 (100%) | 0 | 4.4 s | 1.07 |
| qwen3_8b | 867 / 867 (100%) | 0 | 6.7 s | 1.62 |
| llama3_1_8b | 867 / 867 (100%) | 3 | 6.9 s | 1.65 |
| phi3_mini | 867 / 867 (100%) | 4 | 6.7 s | 1.60 |
| mistral_7b_v03 | 867 / 867 (100%) | 20 | 13.3 s | 3.19 |
| qwen2_5_7b | 867 / 867 (100%) | 25 | 4.2 s | 1.02 |
| gemma2_9b | 867 / 867 (100%) | 91 | 7.3 s | 1.75 |

### 8.4 Stage 4 — context + target text (7 models)

| Model | Usable | Salvaged | Mean latency | GPU-h |
|---|---|---|---|---|
| phi4_mini | 867 / 867 (100%) | 0 | 4.6 s | 1.11 |
| qwen3_8b | 867 / 867 (100%) | 0 | 11.8 s | 2.85 |
| phi3_mini | 867 / 867 (100%) | 0 | 8.3 s | 1.99 |
| llama3_1_8b | 867 / 867 (100%) | 4 | 7.1 s | 1.71 |
| mistral_7b_v03 | 867 / 867 (100%) | 23 | 8.0 s | 1.92 |
| qwen2_5_7b | 867 / 867 (100%) | 64 | 5.0 s | 1.21 |
| gemma2_9b | 867 / 867 (100%) | 167 | 7.4 s | 1.78 |

### 8.5 Aggregate

| | Stage 1 | Stage 2 | Stage 3 | Stage 4 | Total |
|---|---|---|---|---|---|
| Models | 5 | 5 | 7 | 7 | — |
| Inferences | 4,335 | 4,335 | 6,069 | 6,069 | **20,808** |
| Timeouts | 0 | 0 | 0 | 0 | **0** |
| GPU-hours | 14.8 | 17.9 | 11.9 | 12.6 | **57.1** |

Zero timeouts occurred across all 20,808 inferences. Stages 3 and 4 additionally recorded zero
unrecoverable parse failures.

---

## 9. Threats to validity

Reported explicitly, as each bears on how the results may legitimately be interpreted.

1. **Audio-Flamingo-3 Stage 2 modality asymmetry.** AF3 never heard context audio (§7.3); its
   Stage 2 numbers measure a different condition from the other four models.

2. **Stage 1 parser-version artefact (recommended correction).** The Stage 1 full run executed
   before the bracket/key-quote repair layer (§6, step 5) was added. Re-parsing the stored
   `raw_output` with the current parser — requiring **no re-inference and no GPU time** — recovers
   **398 of Audio-Flamingo-3's 613** parse failures, moving it from 252/867 (29.1%) to **650/867
   (75.0%)**; qwen2_audio would gain 4 (799 → 803). This has deliberately **not** been applied,
   because Stage 1 is frozen. AF3's Stage 1 figure as printed therefore conflates output-formatting
   recovery with model capability. Recommended remedy: perform the re-parse into a *separate*
   results file and report both figures, disclosing the parser revision.

3. **Salvage dependence.** Two results lean heavily on field-level salvage: qwen2_audio in Stage 2
   (709/867 = 82%) and gemma2_9b in Stage 4 (167/867 = 19%). Salvaged rows recover the scalar
   prediction fields but not `reasoning_steps`. Report salvage rates alongside accuracy, and
   consider a sensitivity analysis restricted to cleanly parsed rows.

4. **Format adherence degrades with context length.** Salvage rates rise from Stage 3 to Stage 4
   for most text models (qwen2_5_7b 25 → 64; gemma2_9b 91 → 167). This is a *formatting* effect of
   longer prompts and must not be conflated with a change in task accuracy.

5. **Self-reported confidence is not calibrated.** `confidence` is a number a generative model
   emits about itself, not a calibrated posterior; it should be described as such wherever shown.

6. **Explanation similarity is a proxy.** Where embedding cosine similarity is used against
   reference explanations, it measures semantic overlap, not correctness: wording may diverge while
   reasoning agrees, and vice versa.

7. **Cross-architecture modality contrasts.** S1↔S3 and S2↔S4 compare different model families, so
   observed differences reflect the modality arm as a whole rather than matched pairs. S1↔S3 is
   further restricted to `predicted_category` (§4.5).

8. **Label imbalance.** Two classes account for 85.3% of items; three have fewer than 40. Macro-F1
   and per-class support are required for meaningful reporting, and small-support classes carry
   wide uncertainty.

9. **Shared-cluster timing.** Latency and GPU-hour figures were measured on GPUs concurrently used
   by other workloads and are indicative of relative cost, not benchmark-grade throughput.

---

## 10. Reproducibility

**Stage isolation.** Each stage lives in its own tree and writes only to its own results
directory. Later stages reuse earlier logic through **vendored read-only copies** (notably the JSON
parser), never by import, so no later stage can alter an earlier one's behaviour. Stages 1 and 2
were frozen before Stages 3 and 4 were written, and no file in either was modified.

```
benchmark_framework/outputs/results/full/<model>/      # Stage 1
sarcasm_ICL/stage2/results/stage2/<model>/             # Stage 2
sarcasm_ICL/stage_text/results/stage3/<model>/         # Stage 3
sarcasm_ICL/stage_text/results/stage4/<model>/         # Stage 4
```

**Recorded per sample:** `sample_id`, model name, stage, all prediction fields, complete
`raw_output`, `parse_error`, `salvaged`, `error`, `latency_seconds`, GPU memory (allocated and
reserved), UTC timestamp, and — for the text stages — the full configuration block: `model_id`,
`conda_env`, `transformers_version`, `torch_version`, `dtype`, `attn_implementation`,
`gen_max_new_tokens`, `gen_do_sample`, `gen_template_kwargs`.

**Recorded per run:** GPU index, start/end timestamps, exit code, rows written (`run_meta.json`).

**Determinism:** sorted sample-id traversal; greedy decoding throughout; fixed-seed pilot subsets;
offline checkpoint loading for the text stages.

**Environment declarations:** every model's environment, interpreter path, checkpoint, timeout, and
token budget is declared in a per-stage YAML registry, validated fail-fast before any GPU work
begins.

**Reproduction commands:**

```bash
conda activate benchmark-orchestrator

# Stage 1
cd benchmark_framework
python -m orchestrator.run_pilot
python -m orchestrator.run_full --i-reviewed-the-pilot

# Stage 2
cd sarcasm_ICL/stage2
python run_stage2.py --limit 20
python run_stage2.py --full --i-reviewed-the-validation

# Stages 3 and 4
cd sarcasm_ICL/stage_text
python run_text_stage.py --stage 3 --limit 5
python run_text_stage.py --stage 4 --limit 5
python run_text_stage.py --stage 3 --full --i-reviewed-the-validation
python run_text_stage.py --stage 4 --full --i-reviewed-the-validation
```

---

## 11. Analysis stage (pending)

All four stages have completed inference. The remaining component is a **shared scoring layer**
applying one category normaliser across all stages and computing, per model and per stage:

- accuracy, macro-F1, weighted-F1, per-class precision/recall/F1 with support, 7 × 7 confusion
  matrices;
- `is_sarcastic` binary accuracy and source/target agreement (Stages 2–4);
- explanation similarity against reference explanations;
- salvage-rate-stratified sensitivity analyses.

It should then emit the four planned contrasts (S1↔S2, S3↔S4, S1↔S3, S2↔S4), observing the
restrictions in §4.5 and §9. Because scoring reads stored results only, it consumes no GPU time and
can be re-run freely as normalisation rules evolve.

---

## 12. Document map

| Document | Scope |
|---|---|
| `BENCHMARK_METHODOLOGY.md` | this document — unified methodology, all four stages |
| `STAGE1_FRAMEWORK.md` | Stage 1 implementation detail |
| `STAGE2_FRAMEWORK.md` | Stage 2 implementation detail |
| `STAGE3_4_FRAMEWORK.md` | Stages 3 & 4 implementation detail (one shared codebase) |
