# Stage 1 Sarcasm Benchmark — Framework Documentation

**Task:** Target Audio + Target Transcription → sarcasm analysis. The model receives the target
utterance's **audio clip and its transcription** (no conversational context) and classifies the
sarcasm type, with a confidence, explanation, and reasoning steps.

**Position in the four-stage design:**

| | TARGET ONLY | CONTEXT + TARGET |
|---|---|---|
| **AUDIO + TEXT** | **STAGE 1 (this document)** | STAGE 2 |
| **TEXT ONLY** | STAGE 3 | STAGE 4 |

**Location:** `stage1/`
**Status:** Complete (867/867 samples × 5 models). **Frozen** — the reference baseline for all
later stages; no file in this tree is modified by Stage 2/3/4.

---

## 1. Dataset and split

**Source:** `$SARCASM_DATA_ROOT/` — 1,106 sample directories, each
potentially containing `target.wav`, `target.txt`, `context.wav`, `context.txt`,
`sarcasm_analysis.json`.

**Validity predicate** (computed fresh by `bench_common/dataset.py`, never hardcoded). A sample is
valid iff:
1. `target.wav` exists,
2. `target.txt` exists **and is non-empty after strip()**,
3. `sarcasm_analysis.json` exists, parses, has all required keys, and its `category` is one of the
   six canonical labels.

**Exclusion waterfall (evaluated in the order above):**

| Reason | Count |
|---|---|
| Total sample directories | 1,106 |
| Missing `target.wav` | −32 |
| Missing or empty `target.txt` | −164 |
| Missing `sarcasm_analysis.json` | −43 |
| Invalid JSON / schema / category | −0 |
| **Valid Stage 1 samples** | **867** |

This same 867-sample split is reused verbatim by Stages 2, 3, and 4, so all four stages are
evaluated on identical items.

**Audio format** (verified across all files): mono, 16 kHz, 16-bit PCM.

**Ground-truth schema** (`sarcasm_analysis.json`, 100% uniform across all 867):
```json
{"category": "IRONY", "sarcasm_source": "Speaker", "sarcasm_target": "Preference shift",
 "explanation": "...", "sample_id": "-C1L4vuYaeM_000"}
```

**Label distribution (867):** SARCASM 484 (55.8%), IRONY 256 (29.5%), SATIRE 45 (5.2%),
OVERSTATEMENT 35 (4.0%), UNDERSTATEMENT 24 (2.8%), RHETORICAL_QUESTION 23 (2.7%).

---

## 2. Directory structure

```
benchmark_framework/
├── bench_common/            stdlib-only; importable by the orchestrator AND all 5 model envs
│   ├── constants.py    (13)  VALID_CATEGORIES, REQUIRED_GT_KEYS, REQUIRED_PRED_KEYS
│   ├── schema.py       (97)  ManifestRecord / GroundTruthRecord / ResultRecord dataclasses
│   ├── dataset.py     (116)  validity predicate + exclusion report (single source of truth)
│   ├── prompt.py       (45)  build_stage1_prompt() — identical text for all 5 models
│   ├── json_extract.py(208)  robust JSON recovery ladder
│   └── io_utils.py     (35)  append+flush+fsync JSONL, resume helper
├── model_scripts/           each runs ONLY inside its own model's conda interpreter
│   ├── _model_infer_base.py  (151) shared loop: SIGALRM timeout, resume, mem/timing, parse
│   ├── qwen2_audio_infer.py   (79)
│   ├── audio_flamingo3_infer.py (68)
│   ├── phi4_infer.py          (75)
│   ├── voxtral_infer.py       (68)
│   └── audio_reasoner_infer.py(129) ms-swift PtEngine + FINAL_JSON adapter
├── orchestrator/            runs ONLY in the benchmark-orchestrator env
│   ├── config.py        (58)  YAML registry loader + fail-fast path validation
│   ├── gpu_selector.py  (80)  live nvidia-smi idle-GPU selection
│   ├── manifest_builder.py(63) manifest + ground-truth writer; deterministic pilot subset
│   ├── subprocess_runner.py(251) parallel GPU scheduler + stall watchdog + run_meta
│   ├── run_pilot.py    (105)  Phase A entry point
│   ├── run_full.py     (101)  Phase B entry point (tripwire-gated)
│   ├── metrics/  classification.py (135), explanation_similarity.py (69),
│   │             performance.py (57), aggregate.py (105)
│   └── report/   generate_report.py (129)
├── configs/model_registry.yaml (37)
├── env/environment.yml         orchestrator env spec
└── outputs/
    ├── manifests/   full_manifest.jsonl, full_ground_truth.jsonl (+ pilot variants)
    ├── results/{pilot,full}/<model>/  results.jsonl, run.log, run_meta.json
    ├── metrics/     <model>_{pilot,full}_metrics.json, comparison CSVs
    └── reports/     <model>_report.md + figures/*.png
```

---

## 3. Models, environments, and required corrections

All five models run in **separate conda environments**, invoked by their interpreter path directly
(never `conda activate` / `conda run`, which proved fragile for unattended multi-hour runs).

| Model | HF checkpoint | Conda env | transformers | Correction required to run |
|---|---|---|---|---|
| Qwen2-Audio-7B-Instruct | `Qwen/Qwen2-Audio-7B-Instruct` | `qwen2-audio-env` | 5.16.0.dev0 | Processor kwarg renamed `audios=` → **`audio=`**; the old name is **silently ignored**, feeding the model *no audio* while still producing fluent (hallucinated) output. Fixed. |
| Audio-Flamingo-3 | `nvidia/audio-flamingo-3-hf` | `audio-flamingo` | 5.16.0.dev0 | Loaded without dtype → fp32 → CPU offload → every sample timed out. Fixed with `torch_dtype=float16` + casting `input_features` to `model.dtype`. |
| Phi-4-multimodal-instruct | `microsoft/Phi-4-multimodal-instruct` | `phi4-env` | **4.48.0** | Under transformers 5.16.0.dev0 the bundled remote code raised `Tensor.item() cannot be called on meta tensors`. Resolved by pinning that env to transformers 4.48.0. Attention set to `eager` (flash-attn unavailable on V100). |
| Voxtral-Mini-3B-2507 | `mistralai/Voxtral-Mini-3B-2507` | `voxtral` (miniconda3) | 5.14.1 | None — worked as documented. |
| Audio-Reasoner-7B | local `Audio-Reasoner/checkpoints/Audio-Reasoner-7B` | `audio-reasoner` | 4.48.0 | Runs via **ms-swift `PtEngine`**, not raw transformers. Needed local checkpoint path, non-streaming config, a `FINAL_JSON:` marker instruction, and a token budget raised 512 → 900 (its fixed `<THINK>` CoT structure otherwise truncates before the answer). |

**Silent-failure guard.** Because of the Qwen2-Audio kwarg bug — which produced *plausible output
with no audio attached* — every model script asserts, after building processor inputs and before
`generate()`, that an audio tensor is actually present and non-trivial. A failure is recorded as a
distinct `audio_input_error`, never silently ignored.

---

## 4. Prompt and prediction schema

One prompt text, built by `bench_common/prompt.py`, is shared **verbatim** by all five models;
only the audio-attachment mechanics differ per model. The prompt states the transcript, lists the
six categories plus `OTHER`, imposes strict-JSON rules, and shows a **placeholder-only** format
template (an earlier version with a realistic worked example was copied verbatim by
Audio-Flamingo-3 and had to be replaced).

**Prediction schema (Stage 1):**
```json
{"predicted_category": "SARCASM", "confidence": 0.85,
 "explanation": "one sentence", "reasoning_steps": ["step one", "step two"]}
```
`predicted_category` maps to ground-truth `category`; `explanation` maps to ground-truth
`explanation`. (Stages 2–4 extend this with `is_sarcastic`, `sarcasm_source`, `sarcasm_target`;
Stage 1 predicts category only, which is the field used for the cross-stage comparison.)

**Caveat for reporting:** `confidence` is a self-reported number from a generative model, **not a
calibrated probability**, and must be described as such wherever it appears.

---

## 5. Execution architecture

**One long-lived subprocess per model.** Each model loads once and iterates the entire manifest.
(The rejected alternative — activating an environment per sample — would have reloaded a multi-GB
checkpoint 867 times per model.)

- **Per-sample timeout:** `signal.alarm` (SIGALRM) wrapped tightly around the single
  `generate()`/`infer()` call, so one hung sample cannot stall a run; the alarm is cancelled in a
  `finally` block so it never leaks into the next sample's timing.
- **Crash-safe incremental output:** each result is appended and `flush()`+`fsync()`'d
  immediately, so partial progress survives an OOM-kill or node event and is inspectable mid-run.
- **Resume:** on restart a model skips `sample_id`s already present in its results file.
- **Process-level stall watchdog:** the orchestrator kills a subprocess whose results file has not
  grown for N minutes (catches true hangs, e.g. a CUDA deadlock, that SIGALRM cannot interrupt);
  recovery is simply re-invoking the same command, which resumes.
- **GPU scheduling:** `nvidia-smi` is queried **immediately before every launch** (never cached) to
  pick a GPU with ≥20 GB free and ≤30% utilisation; GPUs are reserved per run so two models cannot
  race onto the same card. This matters because the cluster is shared with other users.

**Two-phase rollout (mandatory).** Phase A is a deterministic pilot (18 samples, `seed=42`, so
re-runs pick the same items). Phase B is the full 867-sample run and is a **separate script**
requiring an explicit `--i-reviewed-the-pilot` flag — the pilot never auto-chains into the full run.

---

## 6. JSON robustness

Audio LLMs frequently emit non-strict JSON. Recovery proceeds in layers, most-to-least strict:

1. **Strict JSON** (`json.loads`).
2. **Python dict-literal** (`ast.literal_eval`) — single-quoted dict-repr output.
3. **Backslash-unescape repair** — some models double-escape quotes (`\"` where `"` was meant).
4. **Balanced-brace scanning** — extracts the trailing JSON object from surrounding prose or tags
   (essential for Audio-Reasoner, whose `<THINK>…</THINK><RESPONSE>…</RESPONSE>` wrapper never
   disappears); a `FINAL_JSON:` marker fast-path is tried first for that model.
5. **Bracket / key-quote repair** — closes dropped trailing `]`/`}` and restores a missing opening
   quote on a key.

Anything still unparseable is recorded as `parse_error` with the full `raw_output` retained for
audit. Category strings are normalised at scoring time (case/spacing, plus the observed
`SARCASTIC` → `SARCASM` adjective/noun alias).

---

## 7. Results (full run, 867 samples)

| Model | Usable predictions | Timeouts | Parse errors | Audio errors | Mean latency | GPU-hours |
|---|---|---|---|---|---|---|
| voxtral | 866 / 867 (99.9%) | 0 | 1 | 0 | 3.1 s | 0.75 |
| phi4_multimodal | 864 / 867 (99.7%) | 0 | 3 | 0 | 13.3 s | 3.20 |
| audio_reasoner | 837 / 867 (96.5%) | 0 | 0 | 0 | 34.2 s | 8.24 |
| qwen2_audio | 799 / 867 (92.2%) | 0 | 68 | 0 | 4.4 s | 1.06 |
| audio_flamingo3 | 252 / 867 (29.1%) | 0 | 613 | 0 | 6.2 s | 1.49 |

Total ≈ **14.7 GPU-hours**. Zero timeouts and zero audio-input errors across all 4,335 inferences.

**Notes for the write-up.**
- *audio_reasoner*: 30 samples yielded no matchable category — its chain-of-thought did not surface
  one and the structured fallback could not extract it. Inherent to the CoT model, not a harness
  error.
- *audio_flamingo3*: see the reproducibility caveat in §8 — its low usable rate is substantially a
  **parser-version artefact**, not purely model behaviour.

---

## 8. Reproducibility notes and known caveats

**Recorded per run:** model name, interpreter path, conda env, checkpoint, generation parameters
(`max_new_tokens`, `timeout_seconds`), GPU index, start/end timestamps, exit code, rows written
(`run_meta.json`), plus per-sample latency, GPU memory, timestamp, and full `raw_output`.

**Hardware/software envelope:** 4 × Tesla V100-PCIE-32GB (compute capability 7.0), driver
460.91.03 (CUDA 11.2 ceiling), `torch 2.7.1+cu118` in every environment. flash-attention-2 is
unavailable on Volta, so all models use `eager`/`sdpa` attention.

**⚠ Parser-version caveat (Audio-Flamingo-3).** The Stage 1 full run executed with the JSON parser
as it existed at launch. The bracket/key-quote repair layer (§6, item 5) was added afterwards.
Re-parsing the stored `raw_output` with the current parser — **no re-inference, no GPU time** —
recovers **398 of AF3's 613** parse failures, moving it from 252/867 (29.1%) to **650/867 (75.0%)**;
qwen2_audio would gain 4 (799 → 803). Because Stage 1 is frozen, this has **not** been applied.
For the paper, either (a) report the numbers above as-is and state this caveat explicitly, or
(b) perform a post-hoc re-parse pass into a *separate* results file and report both. Option (b) is
recommended, since AF3's headline number otherwise conflates output-formatting recovery with model
capability. Nothing in the recorded raw outputs changes either way.

**Isolation:** Stage 1 writes only under `benchmark_framework/outputs/`. Stages 2–4 reuse its
*philosophy* and a vendored read-only copy of `json_extract.py`, never importing or editing this
tree, so Stage 1 remains exactly reproducible.

---

## 9. How to reproduce

```bash
conda activate benchmark-orchestrator
cd stage1

python -m orchestrator.run_pilot                      # Phase A: 18-sample deterministic pilot
python -m orchestrator.run_full --i-reviewed-the-pilot  # Phase B: full 867-sample run
```
Per-model environment, checkpoint, timeout, and token budget are declared in
`configs/model_registry.yaml`.
