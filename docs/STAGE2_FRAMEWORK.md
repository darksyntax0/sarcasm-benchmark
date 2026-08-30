# Stage 2 Sarcasm Benchmark — Framework Documentation

**Task:** Context Audio + Context Transcription + Target Audio + Target Transcription → sarcasm
analysis. Each request carries **both** clips and **both** transcriptions in a single inference
call, and the model predicts whether the TARGET utterance is sarcastic (using the CONTEXT),
its sarcasm category, who is being sarcastic, who/what is targeted, and an explanation.

**Location:** `stage2/`
**Built:** 2026-08-21. Stage 1 (`benchmark_framework/`) was treated as **frozen** — never
modified, imported for edit, or overwritten. Stage 2 is fully self-contained.

---

## 1. Relationship to Stage 1 (and how it stays frozen)

| | Stage 1 (frozen) | Stage 2 (this) |
|---|---|---|
| Input | target audio + target text only | **context + target** audio + text, one request |
| Location | `benchmark_framework/` | `sarcasm_ICL/stage2/` |
| Dataset | same 867 valid samples | same 867 (now also require `context.*` present) |
| GT schema | `category, sarcasm_source, sarcasm_target, explanation, sample_id` | identical (unchanged) |
| Prediction | `predicted_category, confidence, explanation, reasoning_steps` | **superset** (adds `is_sarcastic, sarcasm_source, sarcasm_target`) |
| Results | `benchmark_framework/outputs/results/full/` | `sarcasm_ICL/stage2/results/stage2/` |

**Isolation guarantees:** Stage 2 never writes into `benchmark_framework/`. The one piece of
Stage 1 logic reused — the robust JSON parser — was **copied** (vendored) into
`stage2/json_extract.py`, not imported, so Stage 2 has no live dependency on Stage 1's tree and
editing it can never affect Stage 1. Stage 1's full run finished at 867/867 during Stage 2 build
and was verified alive/untouched throughout.

---

## 2. Directory structure

```
sarcasm_ICL/
├── run_audio_flamingo3.py     ← your 5 original candidate scripts (reference; untouched)
├── run_audio_reasoner.py
├── run_phi4_multimodal.py
├── run_qwen2_audio.py
├── run_voxtral_mini.py
└── stage2/                    ← the Stage 2 framework
    ├── samples.py             (71 ln)  stdlib-only dataset access; iter_valid_samples(), load_sample()
    ├── stage2_prompt.py       (66 ln)  shared task text + output schema (VALID_CATEGORIES, REQUIRED_PRED_KEYS)
    ├── json_extract.py        (224 ln) VENDORED robust JSON parser (json→ast→unescape→bracket/key repair)
    ├── run_stage2.py          (172 ln) orchestrator: manifest build + idle-GPU scheduler + watchdog
    ├── configs/
    │   └── stage2_models.yaml (34 ln)  explicit env/script/params/native_two_audio per model
    ├── infer/
    │   ├── _stage2_base.py     (160 ln) shared loop: timeout, resume, mem/timing, parse+salvage
    │   ├── qwen2_audio.py      (43 ln)  per-model inference (2 audios)
    │   ├── voxtral_mini.py     (43 ln)  per-model inference (2 audios)
    │   ├── phi4_multimodal.py  (41 ln)  per-model inference (2 audios)
    │   ├── audio_reasoner.py   (66 ln)  per-model inference (2 audios, ms-swift CoT)
    │   └── audio_flamingo3.py  (48 ln)  per-model inference (TARGET audio + text context)
    ├── validate/              ← Step-4 small-dataset validators + verdict JSONs
    │   ├── validate_<model>.py (×5)
    │   ├── run_all.sh
    │   └── out/<model>.json    tensor-level "both audios reached the model" verdicts
    └── results/               ← ALL Stage-2 outputs (never touches Stage 1)
        ├── manifests/         manifest_{pilot,full}.jsonl + ground_truth_{pilot,full}.jsonl
        ├── stage2/<model>/    results.jsonl (one line per sample) + run.log + run_meta.json
        └── stage2_full.log    orchestrator log
```

---

## 3. The five models, environments, and the fixes that were required

Each model runs in its own conda env via that env's python interpreter directly (no
`conda activate` at runtime). The candidate scripts you provided were treated as **unverified
prototypes**; each was checked against the *actually installed* API and given the **smallest**
necessary correction:

| Model | Conda env / transformers | Fix applied vs your candidate |
|---|---|---|
| **qwen2_audio** | `qwen2-audio-env` / 5.16.0.dev0 | Candidate used `processor(..., audios=)`; this build renamed the kwarg to **`audio=`** (singular). The old name is **silently ignored** → model gets no audio. Changed to `audio=`. |
| **voxtral_mini** | `voxtral` (miniconda3) / 5.14.1 | No code fix (multi-audio is natively supported). Validation assertion corrected: Voxtral chunks clips into 30 s windows, so audio-feature rows == total chunks (**≥2**), not exactly 2. |
| **phi4_multimodal** | `phi4-env` / 4.48.0 | Candidate defaulted to `flash_attention_2` (**not installed**) → use `eager`. Load audio via `librosa@16k` (float32) instead of `sf.read` (float64). `audios=[(wav,sr),…]` plural is correct for Phi-4. |
| **audio_reasoner** | `audio-reasoner` / swift 3.0.0 | Point at the **local** checkpoint `…/Audio-Reasoner/checkpoints/Audio-Reasoner-7B`; non-streaming inference; CoT output coaxed to JSON via a `FINAL_JSON:` marker + structured fallback. |
| **audio_flamingo3** | `audio-flamingo` / 5.16.0.dev0 | Candidate omitted dtype → fp32/CPU-offload/timeouts → added `torch_dtype=float16`. **See §4 — native two-audio is impossible in this build.** |

**Two-audio was tensor-verified** for 4 of 5 models (the validator inspects the processor output
and confirms *both* clips are encoded, guarding against the silent-audio-drop class of bug):
- qwen2_audio: `input_features` shape `(2,128,3000)`
- voxtral_mini: `input_features` rows ≥ 2
- phi4_multimodal: `audio_embed_sizes` length 2
- audio_reasoner: ms-swift template `input_features` rows == 2

---

## 4. Audio-Flamingo-3 — the one hard limitation (reported, not hidden)

AF3's installed processor (`processing_audioflamingo3.py`) enforces
`len(text) == len(audio)` in `validate_inputs`, and `apply_chat_template` always flattens a
conversation to a **single** prompt string. Therefore **any** request with two audios — whether
both in one turn or split across turns — raises `Got 1 text but 2 audios; they must match 1:1.`
This build's AF3 is architecturally **one-audio-per-prompt**; native two-audio input is
**not supported**.

**Decision (yours):** AF3 runs a **reduced-modality variant** — it receives the **TARGET audio**
plus the **CONTEXT as transcription text only** (no context audio). The prompt states this truth
(`context_audio_available=False`), and `native_two_audio: false` is recorded in the config.
**AF3's Stage-2 numbers are therefore not directly comparable** to the four native two-audio
models; this must be footnoted in any comparison/paper.

---

## 5. Standardized task prompt & output schema

The **semantic** task is identical across all five models (`stage2_prompt.build_task_text`);
only each model's low-level chat/placeholder syntax differs (handled inside each `infer/` script).
The model is asked to return a single JSON object:

```json
{"is_sarcastic": true, "predicted_category": "IRONY", "sarcasm_source": "Speaker",
 "sarcasm_target": "Preference shift", "confidence": 0.8,
 "explanation": "one sentence", "reasoning_steps": ["step one", "step two"]}
```

- `predicted_category` ∈ {SARCASM, IRONY, SATIRE, OVERSTATEMENT, UNDERSTATEMENT,
  RHETORICAL_QUESTION, OTHER} — aligns with GT `category`.
- `sarcasm_source` / `sarcasm_target` align with GT `sarcasm_source` / `sarcasm_target`.
- `explanation` aligns with GT `explanation`.
- `is_sarcastic`, `confidence`, `reasoning_steps` are extra signals (confidence is the model's
  self-report, not a calibrated probability). **No new ground-truth schema was invented.**

---

## 6. JSON robustness (parser + salvage)

Audio LLMs frequently emit non-strict JSON. The pipeline recovers predictions in layers, most to
least strict, each transparently flagged:

1. **Strict JSON** (`json.loads`).
2. **Python dict-literal** (`ast.literal_eval`) — single-quoted dicts.
3. **Backslash-unescape repair** — for spuriously `\"`-escaped quotes.
4. **Bracket / key-quote repair** — closes dropped trailing `]`/`}` and inserts a missing opening
   quote on a key (e.g. `,reasoning_steps":` → `,"reasoning_steps":`). Recovered Audio-Flamingo-3.
5. **Field-level salvage** (`salvage_stage2_fields`) — last resort when the whole object is
   unparseable (Qwen2-Audio emits single-quoted dicts whose *explanation/reasoning* contain nested
   apostrophes that break json **and** ast, while the scalar fields stay clean). Regex-extracts
   `is_sarcastic, predicted_category, sarcasm_source, sarcasm_target, confidence`. Records flagged
   `"salvaged": true` so the provenance is auditable. This took Qwen2-Audio from 3/20 usable to
   20/20 on the validation set.

Every result row keeps the full `raw_output` for audit regardless of parse path.

---

## 7. Validation performed (Steps 3–4, before any full run)

A deterministic small subset of real dataset samples was run through the whole pipeline
(you requested ≥20). Two rounds: the first surfaced real issues, all fixed; the second was clean.

**Final 20-sample validation:**

| Model | Usable | Native 2-audio | is_sarc / cat / src / tgt | Notes |
|---|---|---|---|---|
| voxtral_mini | 20/20 clean | ✅ | 20 / 20 / 20 / 20 | clean |
| phi4_multimodal | 20/20 clean | ✅ | 20 / 20 / 20 / 20 | clean |
| audio_flamingo3 | 20/20 clean | ⚠️ target-only | 20 / 20 / 20 / 20 | reduced modality |
| audio_reasoner | 20/20 clean | ✅ | 20 / 20 / 16 / 17 | CoT + fallback; ~37 s/sample |
| qwen2_audio | 20/20 usable | ✅ | 20 / 20 / 20 / 20 | 3 clean + 17 field-salvaged |

Issues found and fixed during validation: qwen2 silent-audio kwarg; AF3 fp32/offload; phi4
flash-attn default; voxtral chunk-count assertion; JSON key-quote repair; token-budget truncation;
qwen2 field-salvage; audio_reasoner fallback field population.

---

## 8. Orchestrator & how to run

`run_stage2.py` builds the manifest + ground-truth files, then launches each model as a
long-lived subprocess pinned to an idle GPU (live `nvidia-smi` query, ≥20 GB free & ≤30 % util),
running as many in parallel as there are free GPUs and queueing the rest. Per-model: one model
load, loop all samples, append+`fsync` each result (crash-safe), **resume** by skipping
sample_ids already written. A stall-watchdog kills a subprocess whose results file stops growing.

```bash
# environment for the orchestrator process (only needs pyyaml+stdlib; models run in their own envs)
conda activate benchmark-orchestrator
cd stage2

python run_stage2.py --build-manifest-only            # build manifest/GT only
python run_stage2.py --limit 20                       # small validation subset, all models
python run_stage2.py --limit 20 --models qwen2_audio  # subset of models
python run_stage2.py --full --i-reviewed-the-validation   # FULL 867-sample run (tripwire required)
```

Per-model config (env, script, `timeout_seconds`, `max_new_tokens`, `native_two_audio`) lives in
`configs/stage2_models.yaml`.

---

## 9. Output files

Per model, under `results/stage2/<model>/`:
- `results.jsonl` — one row per sample: `sample_id, model_name, is_sarcastic, predicted_category,
  sarcasm_source, sarcasm_target, confidence, explanation, reasoning_steps, raw_output,
  parse_error, salvaged, audio_input_error, error, latency_seconds, gpu_mem_*_mb, timestamp`.
- `run.log` — the model subprocess's stdout/stderr.
- `run_meta.json` — gpu index, start/end, exit code, rows written.

Ground truth for scoring: `results/manifests/ground_truth_full.jsonl`
(`sample_id, gt_category, gt_sarcasm_source, gt_sarcasm_target, gt_explanation`).

---

## 10. Reproducibility notes (for paper / repo release)

- **Stage 1 frozen and independent.** Stage 2 shares only vendored, read-only code.
- **Exact model config** recorded in `configs/stage2_models.yaml` (envs, checkpoints, token/timeouts).
- **Deterministic** dataset traversal (sorted sample ids); pilot subset = first N valid.
- **Model-specific logic isolated** — each `infer/<model>.py` owns its processor/chat-template.
- **Honest provenance** — `salvaged` flag, `native_two_audio` flag, AF3 reduced-modality caveat.
- **Same 867 samples** as Stage 1 (context files were present for all 867 valid samples).

---

## 11. Full run status

Launched 2026-08-21 in tmux session **`stage2_full`**, all 5 models, 867 samples, parallel across
all 4 GPUs (Stage 1 having finished). Wall-clock is bounded by audio_reasoner's CoT
(~37 s/sample ≈ 9 h); the other four finish in ~1–2.5 h and AF3 slots onto the first freed GPU.

```bash
tmux attach -t stage2_full            # Ctrl+B then D to detach
tmux capture-pane -t stage2_full -p | tail -30
tail -f sarcasm_ICL/stage2/results/stage2_full.log
```

A scoring/report step (classification metrics vs GT, explanation similarity, per-model comparison)
can be added after the run completes, mirroring Stage 1's reporter but reading Stage-2 results and
including the `is_sarcastic` / `sarcasm_source` / `sarcasm_target` fields.
