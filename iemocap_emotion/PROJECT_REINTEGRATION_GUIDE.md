# Project Reintegration Guide

**Purpose of this document:** a complete, evidence-based handover so you can regain full ownership of this project — understand every decision, defend it, and know exactly which parts of it you can currently trust.

**How to read this document:** every claim below is tagged as one of:
- **[VERIFIED]** — I read the actual file/output and confirmed it.
- **[PLANNED, NOT DONE]** — described somewhere (README, docstring, comment) but no evidence it was executed.
- **[ATTEMPTED/ABANDONED]** — started, then changed or dropped.
- **[INFERRED]** — my best reading of the evidence, but not directly confirmed — flagged so you don't repeat it as fact without checking.
- **[NEEDS YOUR INPUT]** — a decision or piece of context only you have.

This audit did not modify, run, or train anything. All numbers below were read directly from files on disk.

---

## 1. Project in plain language

**The problem.** When something goes wrong in a cockpit, the Cockpit Voice Recorder (CVR) captures everything the crew says. After an incident, investigators listen to that audio to reconstruct what happened — but a lot of the *emotional* information in the crew's voices (stress, fear, anger, confusion) is currently interpreted informally by a human listener, not measured systematically. An automated system that could flag "this utterance sounds like high stress/anger" from voice + words could help investigators triage long recordings faster, or in the future support real-time crew-state monitoring.

**Why it matters here.** Nobody has built and validated a text+audio emotion model specifically on aviation cockpit speech, because that data (real CVR recordings) is restricted and not available to this project (referred to throughout the code as "pending GCAA data access" — GCAA is presumably the aviation authority you'd need clearance from). So the actual research strategy is: **build and validate the full pipeline on a public, well-studied speech-emotion dataset (IEMOCAP) first, prove it works and prove you understand its limitations, and design the system so a real CVR corpus could be substituted in later** without redesigning everything.

**Central research question (as the code implies it, not as a quoted thesis statement — I did not find a written thesis statement anywhere in the repo):** *Can a parameter-efficient (LoRA) fine-tuned multimodal (text + audio) model reliably classify five emotion categories from spontaneous, session-independent speech, and can retrieval-based evidence (kNN over training examples) be used to flag likely misclassifications for human review, in a way that could transfer to a safety-critical audio-triage setting like CVR analysis?**

**Intended contribution / novelty** (inferred from what was actually built, not from any explicit "our contribution is..." statement in the repo):
1. A cross-modal attention fusion model (RoBERTa + WavLM, both LoRA fine-tuned) evaluated with proper Leave-One-Session-Out (LOSO) cross-validation on IEMOCAP — not just a single train/test split.
2. A cross-corpus check on MELD (a different, noisier, multi-speaker TV-dialogue dataset) to see whether performance holds up outside IEMOCAP.
3. An "agentic" layer on top of the trained classifier — a retrieval-augmented explanation and alerting system (kNN neighbor evidence + rule-based alert flags) that is explicitly framed as a decision-support prototype for a future CVR pipeline, not just a classifier.

**What the final system is expected to do:** given an utterance (audio clip + transcript), predict one of 5 emotions (anger, fear, joy, neutral, sadness), report retrieved similar training examples as evidence, and raise a review alert when the retrieval evidence disagrees with the prediction — with placeholder fields (`flight_phase`, `utterance_timecode`) already wired into the output structure for future CVR context, currently always empty.

**What the current evidence supports:**
- [VERIFIED] The fusion model beats single-modality models on IEMOCAP under proper 5-fold LOSO cross-validation (Macro F1 0.694 vs text 0.629 vs audio 0.526 — see §7).
- [VERIFIED] The same architecture, retrained on MELD, does **not** show the same fusion advantage (fusion Macro F1 0.499 vs text 0.500 — fusion does not beat text-only on MELD).
- [VERIFIED] A simple retrieval-agreement heuristic correlates strongly with classifier correctness on IEMOCAP fold 1 (84% agreement when correct vs 20% when wrong) — this is a real, measured signal, not just a design assumption.
- [VERIFIED] The alert rule built on that signal, on IEMOCAP fold 1, catches about 28–34% of actual misclassifications at a precision of 42–45% (i.e., more than half of its alerts are false alarms) — explicitly documented in the code as **not sufficient as a standalone safety gate**.

**What the current evidence does NOT support (important for you to say out loud to a reviewer):**
- No claim about performance on real CVR/aviation audio — there is zero aviation data anywhere in this project. Everything is IEMOCAP (acted/elicited emotional dialogue) and MELD (TV show dialogue).
- No claim that the retrieval/alert layer generalizes beyond IEMOCAP fold 1 — it has literally never been run against folds 2–5, despite the underlying classifier being fully trained and tested on all 5 folds. See §11, Critical issue #1.
- No statistically tested claim that fusion significantly beats text/audio on IEMOCAP — the significance-testing script (`analyse_results.py`) exists and is well-designed, but its saved output only ever covers MELD, not IEMOCAP (see §11, Critical issue #2). The IEMOCAP fusion-vs-text gap (0.694 vs 0.629 macro F1, ±0.05/±0.05 std across folds) looks meaningful but has not been formally significance-tested and saved.
- No claim survives about the "6th emotion class" (disgust) — the README and some code comments still describe a 6-class system, but the class was fully dropped, not merged, in the actual running code (5 classes only). This is a documentation error, not a modeling one, but you must not describe this as 6-class work to anyone.

---

## 2. Research design

### Datasets
- **Primary: IEMOCAP** (`data_splits/data/IEMOCAP_full_release/`, root-level) — 5 acted/elicited dyadic-dialogue sessions, 10 unique speakers (2 per session, never repeated across sessions). [VERIFIED]
- **Secondary/cross-corpus: MELD** (`src/MELD.Raw/`, extracted from `MELD.Raw.tar.gz`) — TV show ("Friends") dialogue, multi-speaker, naturalistic/scripted acting, native train/dev/test split (not session-based). [VERIFIED]

### Emotion classes and why
Final classes: **anger, fear, joy, neutral, sadness** (5 classes). [VERIFIED — `src/config.py:150-152`]

Original IEMOCAP labels and how each was handled (`src/config.py:114-142`):
| Raw IEMOCAP label | Mapped to | Rationale (from code comments) |
|---|---|---|
| ang | anger | kept |
| sad | sadness | kept |
| fea | fear | kept, despite very low support (~40 utterances total across all 5 sessions) |
| neu | neutral | kept |
| hap, exc | joy (merged) | happiness and excitement merged into one "joy" class |
| dis | **dropped (None)** | comment: only 2 samples across all 5 sessions — too rare to model |
| fru | **dropped (None)** | comment: IEMOCAP-specific ("frustration"), not present in MELD/CREMA-D, so keeping it would break cross-corpus comparability |
| sur | **dropped (None)** | comment: inconsistently annotated |
| oth, xxx | **dropped (None)** | annotation artefacts / no-consensus labels |

MELD label map (`src/config.py`) keeps the same 5 names, drops `disgust` and `surprise` (both present in MELD but not survivable in the IEMOCAP-derived label space) — this is what makes the two datasets label-space-compatible for the cross-corpus comparison. [VERIFIED]

**[NEEDS YOUR INPUT]**: whether "fear" should stay in the label set at all is a genuinely defensible-both-ways decision given ~40 total samples — see §8 and §10 for the full argument either way. The code currently keeps it; you should be ready to justify that choice explicitly, not just point at the code.

### Dataset exclusions
- Disgust: dropped from both datasets (too rare in IEMOCAP: 2 samples).
- Excitement: merged into joy, not dropped — worth being precise about this distinction with an examiner (excitement isn't "excluded," it's folded into a broader positive-affect category).
- Frustration, surprise, other/no-consensus: dropped, mainly for cross-corpus label-space compatibility and annotation-quality reasons.
- Total retained IEMOCAP utterances: **5,571** (5,572 rows in `src/data_splits/iemocap_metadata.csv` minus 1 header row). [VERIFIED via `wc -l`]

### Train/val/test protocol — IEMOCAP
**Leave-One-Session-Out (LOSO), 5-fold.** For fold *k*: test = Session *k*, validation = Session (*k* mod 5) + 1, train = the remaining 3 sessions. [VERIFIED — `src/data_preprocessing.py::get_split_dfs`, `src/config.py`]

This is **session-independent**, and because each IEMOCAP session contains exactly 2 unique speakers who never appear in any other session, this is also **speaker-independent** — no speaker's voice appears in both train and test for any fold. This is the correct protocol for a "will this generalize to a person we haven't heard before" claim, which is directly relevant to a CVR use case (you will never have prior recordings of the exact crew member in a real deployment). [VERIFIED — protocol matches speaker roster structure of IEMOCAP]

### Train/val/test protocol — MELD
Native split (train/dev/test as provided by MELD, not session-based) — dev used as validation, test held out. [VERIFIED — `src/meld_preprocessing.py`]

### Cross-validation design
5-fold LOSO on IEMOCAP, run to completion for all three model types (text, audio, fusion) — all 15 fold×model combinations have checkpoints, logs, and metrics on disk. [VERIFIED — see §7]

### Cross-corpus evaluation design
MELD is **not** used to cross-validate the IEMOCAP-trained model. Instead, **separate models were trained from scratch on MELD's own native split** (`checkpoints/{model}/meld_best.pt`, root-level). This means the "cross-corpus" element of this project is currently "does the same architecture/training recipe work on a second, harder dataset," not "does a model trained on IEMOCAP transfer zero-shot to MELD." **[NEEDS YOUR INPUT / clarify with supervisor]**: if your intended claim is about *transfer* (IEMOCAP→MELD zero-shot or fine-tuned), that experiment has not been run — what exists is a parallel, independently-trained MELD model. Confirm which framing your supervisor expects before writing this up.

### Text-only / audio-only / multimodal experiments
All three exist as fully independent trained models for both datasets: `RoBERTaLoRA` (text), `WavLMLoRA` (audio), `CrossModalAttentionFusion` (both, with cross-attention). [VERIFIED]

### Baselines and comparison models
There is no baseline outside this project's own three models (e.g., no comparison to a published SOTA IEMOCAP or MELD number, no majority-class baseline reported in the metrics files I found). The three models act as baselines for each other (text vs audio vs fusion). **This is a gap** — see §11.

### Evaluation metrics and why
From `src/metrics.py::compute_metrics` [VERIFIED]:
- **Macro F1** — the primary reported metric. Averages F1 across classes unweighted by support, so it doesn't let the majority class (neutral) dominate the score — important given the extreme class imbalance (fear ≈ 40 samples vs neutral in the thousands).
- **Weighted F1** — F1 averaged weighted by class support; included so you can see performance from a "typical utterance" perspective, contrasted against Macro F1's "typical class" perspective.
- **Weighted accuracy** (= plain accuracy) and **unweighted accuracy** (= macro recall) — same idea applied to accuracy.
- **Per-class F1** — necessary because Macro F1 alone hides which classes are failing (this is how the fear-class high-variance problem was discovered — see §7, §9).
- **Confusion matrix** — for qualitative error analysis (e.g., the anger→neutral confusion pattern discussed throughout `agentic_ai/`).
- **ECE (Expected Calibration Error)**, 10 equal-width bins, computed separately in `src/plot_calibration.py`/`src/temperature_scaling.py` — measures whether the model's confidence scores are trustworthy (a 90%-confidence prediction should be right ~90% of the time). This matters specifically for a decision-support tool: if confidence is miscalibrated, a human reviewer trusting "the model was 95% sure" could be misled.

### Class imbalance handling
- Inverse-frequency class weights in the loss, computed **per LOSO fold on the training split only** (never on val/test — no leakage), capped at `max_class_weight=5.0` to avoid the rare `fear` class dominating gradients. [VERIFIED — `src/config.py`, `src/data_preprocessing.py::compute_class_weights`]
- Label smoothing (0.1) also applied globally as an additional regularizer. [VERIFIED — `src/config.py`]
- Fear is explicitly treated as a known-difficult, low-support class throughout the codebase (comments, and the "risk-weighted F1" prototype metric in `agentic_ai/risk_weighted_f1.py` that up-weights anger/fear specifically because they're operationally more important to catch, at a documented cost to raw accuracy).
- **What is not done**: no oversampling/SMOTE/synthetic-data approach for fear was attempted (no evidence of this anywhere in the code) — only loss reweighting. **[NEEDS YOUR INPUT]**: worth deciding whether to explicitly justify "we chose reweighting over oversampling because X" or acknowledge it as unexplored.

### Leakage / methodological risks (see §11 for the full ranked list)
- Class weights: no leakage (confirmed above).
- Checkpoint selection: selects on validation Macro F1, evaluates on a disjoint held-out test session — correct, no leakage. [VERIFIED — `src/train.py`]
- Audio caching (`audio_cache/`, keyed by `{utterance_id}_{max_samples}`): cache key doesn't include the fold, but since audio preprocessing (RMS normalize, resample, pad/truncate) doesn't depend on which fold an utterance belongs to, this is not a leakage risk, just a reuse optimization.
- The **agentic RAG layer's kNN index is built from that fold's *training* set only** — explicitly verified at build time via an assertion in `evaluate_agent.py` that the index and eval-split utterance IDs have zero overlap. [VERIFIED — this is a well-designed leak-check, not just an assumption]

---

## 3. Complete project timeline

I want to be direct about a limitation here: **file modification timestamps and directory structure are the only chronological evidence available to me** — I do not have access to git history (this is not a git repo, per the environment info) or your prior conversations with me from other sessions beyond what appears in this conversation's context. So this timeline is reconstructed, not authoritative, and several dates are approximate.

| Order | What (evidence) | When (file dates, approximate) | Why (inferred or documented) | Result | Status |
|---|---|---|---|---|---|
| 1 | IEMOCAP metadata build, label mapping design (`src/data_preprocessing.py`, `src/config.py`) | earliest — no exact date available | Needed before any training could start | 5,571 utterances retained, 5-class label space | [VERIFIED — output exists] Retained |
| 2 | Text/Audio/Fusion model architectures built with LoRA (`src/models_*.py`) | before training runs | Parameter-efficient fine-tuning chosen over full fine-tuning, likely for compute/memory reasons on available hardware | Working models | [VERIFIED] Retained |
| 3 | IEMOCAP 5-fold LOSO training, all 3 model types | `src/checkpoints/fusion/fold*.pt` dated **2026-07-01, ~16:45–19:10** (roughly 2h25m span across 5 folds, fusion model) | Core experiment: proper cross-validation on primary dataset | Fusion Macro F1 0.694±0.054 (test), beats text (0.629) and audio (0.526) — see §7 | [VERIFIED] Retained, this is the project's main quantitative result |
| 4 | `agentic_ai/` layer built: EmotionAgent, retrieval index, AlertAgent, orchestrator | after IEMOCAP training (depends on a trained fusion checkpoint existing) | Framed throughout as building toward a CVR decision-support prototype, not just a classifier | Working prototype, evaluated only on fold 1 | [VERIFIED] Retained, but **fold-1-only** — see §11 Critical #1 |
| 5 | MELD dataset acquired, extracted, preprocessed (`src/MELD.Raw.tar.gz`, `src/meld_preprocessing.py`) | unclear exact date, but MELD checkpoints dated **2026-07-20, 11:43** — about 3 weeks after IEMOCAP fusion training | Likely intended as a generalization/cross-corpus check, given IEMOCAP's small size and single-domain (acted dialogue) nature | MELD fusion Macro F1 0.499, text 0.500 — fusion does **not** beat text here | [VERIFIED] Retained as a finding, but this is a **different framing of "cross-corpus"** than transfer — see §2 caveat |
| 6 | Calibration/ECE analysis, temperature scaling, McNemar significance testing — but only saved for MELD (`src/temperature_scaling.py`, `src/analyse_results.py`, `outputs/analysis/`) | after MELD training | Assessing whether model confidence is trustworthy, and whether fusion's improvement (or lack of it, on MELD) is statistically real | text vs fusion on MELD: statistically significant difference (McNemar χ²=10.04, p=0.0015) but fusion is *worse*, not better, on macro accuracy terms; audio significantly worse than both | [VERIFIED] Retained. **[ATTEMPTED/ABANDONED-BY-OMISSION]**: the equivalent IEMOCAP significance testing was never actually produced and saved, despite the script supporting it and the underlying prediction CSVs existing — see §11 Critical #2. This looks like an unfinished step, not a deliberate decision. |
| 7 | Alert-rule design and threshold tuning for the agentic layer (`agentic_ai/alert_agent.py`, `evaluate_alert.py`) | after step 4/6 | Empirical: agreement=0.84 when correct vs 0.20 when wrong (fold1) motivated a simple threshold rule | t=0.6 chosen deliberately over the F1-optimal t=0.9, because t=0.9 produced unacceptable false-alarm rate (37.6% FPR) — explicitly reasoned about in the docstring, a genuinely good methodological decision | [VERIFIED] Retained |
| 8 | Stage-2 alert rule extension ("neutral_low_agreement") — I built and evaluated this directly in this conversation history, at your request | this session and the one before it | The stage-2 threshold at 0.6 (same as primary) produced literally zero additional alerts (mathematically redundant); re-run at t=0.8 produced 59 additional alerts (18 TP/41 FP), a modest recall gain (0.282→0.342) at a precision cost (0.450→0.415) | [VERIFIED — recomputed in-session] Retained at t=0.8 as the current default, but this is a very recent, lightly-validated change — **only tested on fold 1** | Recent, provisional |
| 9 | HITL pilot utterance selection, case-study utterance selection (`agentic_ai/outputs/hitl_pilot_utterances.csv`, `case_study_utterances.csv`) | this session and the one before it | Preparing concrete example utterances (with real transcripts, audio paths, model outputs) for a human-in-the-loop pilot study and illustrative case studies | 10 utterances selected for HITL pilot (2 per class, correct/wrong pairs); 6 utterances selected for 2 narrative case-study scenarios | [VERIFIED] Retained, but note: this is *utterance selection/curation for a future study*, not a completed HITL study — no human annotators have actually reviewed anything yet, as far as this repo shows | In progress / setup only |
| 10 | Documentation drift (README, code comments) never corrected | ongoing, unclear when it started | README describes 6-class system and includes disgust in its "kept" mapping table — this directly contradicts the actual 5-class code | N/A | [ATTEMPTED/ABANDONED] The disgust-inclusion approach was apparently tried at some point (why else would the README describe it this specifically) and then reversed in code, without updating the README | Needs correction |

**What I could not reconstruct**: the exact order of steps 1–3 relative to each other with real dates (only fold 1's fusion checkpoint timestamp is directly informative), whether disgust was ever actually trained-with-included at any point (I found no checkpoint or metrics evidence of a 6-class run — only the stale README/comment text), and the reasoning behind choosing RoBERTa-base/WavLM-base-plus specifically over other options (no comparison experiments with alternative backbones exist in the repo).

---

## 4. Current pipeline (end-to-end, current running state)

This describes the **IEMOCAP LOSO pipeline**, run from inside the `src/` directory (this matters because config paths are relative — see §5 and §11).

| Stage | File / function | Input | Output |
|---|---|---|---|
| 1. Dataset acquisition | manual — IEMOCAP corpus placed at `data_splits/data/IEMOCAP_full_release/` (root-level); MELD via `src/MELD.Raw.tar.gz`, extracted by `src/meld_preprocessing.py::extract_inner_tarballs` | raw corpus files | extracted session/dialogue folders |
| 2. Metadata build / label mapping | `src/data_preprocessing.py::build_metadata()`, using `cfg.label_map` from `src/config.py` | `IEMOCAP_full_release/Session{1-5}/dialog/{EmoEvaluation,transcriptions}/*.txt` | `src/data_splits/iemocap_metadata.csv` (5,571 rows) |
| 3. Split generation | `src/data_preprocessing.py::get_split_dfs(df, fold)` | metadata CSV, fold number | in-memory train/val/test DataFrames (session-based LOSO) |
| 4. Text tokenization | `src/dataset.py::TextDataset`, using `RobertaTokenizer` (`roberta-base`, `max_text_tokens=128`) | transcript text | `input_ids`, `attention_mask` |
| 5. Audio loading/processing | `src/data_preprocessing.py::load_and_preprocess_audio` (mono, resample to 16kHz, RMS-normalize to -20dBFS, pad/truncate to 6.0s = 96,000 samples); cached via `precompute_audio_cache` to `src/data_splits/audio_cache/*.pt` | raw `.wav` file | waveform tensor + attention mask |
| 6. Model architecture | `src/models_text.py::RoBERTaLoRA`, `src/models_audio.py::WavLMLoRA`, `src/models_fusion.py::CrossModalAttentionFusion` | tokenized text / processed audio | 5-class logits |
| 7. LoRA fine-tuning | `peft.LoraConfig` inside each model file (rank=8, alpha=16, dropout=0.05; text targets `["query","value"]`, audio targets `["q_proj","v_proj"]`) | pretrained RoBERTa-base / WavLM-base-plus weights | LoRA-adapted model with frozen backbone + small trainable adapters |
| 8. Training loop | `src/train.py::main()` → `train_one_fold()` (exact function boundary as summarized by the audit; I did not personally read every line of `train.py`, so treat internal function names as [INFERRED] unless you check them directly) | train/val DataLoaders, `cfg` hyperparameters | best-val-Macro-F1 checkpoint |
| 9. Validation / checkpoint selection | inside the training loop — saves whenever val Macro F1 improves, early-stops after 5 epochs with no improvement, reloads best checkpoint before final test eval | val split each epoch | `src/checkpoints/{model}/fold{k}_best.pt` |
| 10. Testing | same script, post-training, on the held-out test session | test split, best checkpoint | `src/outputs/{model}/fold{k}_test_metrics.json`, `fold{k}_test_predictions.csv`, `fold{k}_confusion_matrix.npy` |
| 11. Fusion | `src/models_fusion.py::CrossModalAttentionFusion` — cross-attention (text as Query, audio as Key/Value), then concatenation of `[z_text; z_cross; z_audio]` (768-dim) → classifier head | text + audio tensors | 5-class logits, plus `embed()` for the 768-dim pre-classifier vector used by the RAG layer |
| 12. Metrics aggregation | `src/metrics.py::aggregate_fold_results` (only if all 5 folds run in one invocation) | 5 folds' `test_metrics.json` | `src/outputs/{model}/all_folds_summary.json` (mean ± std) |
| 13. Result visualization/analysis (**MELD only, currently**) | `src/analyse_results.py`, `src/plot_calibration.py`, `src/plot_confusion_matrices_meld.py`, `src/temperature_scaling.py` | `outputs/{model}/meld_test_predictions*.csv` (root-level) | `outputs/analysis/*.png`, `*.txt` |
| 14. Agentic layer (fold 1 only) | `agentic_ai/build_index.py` → `agentic_ai/agent.py::EmotionAgent` → `agentic_ai/alert_agent.py::AlertAgent` → `agentic_ai/orchestrator.py::SupervisorAgent` | fold-1 fusion checkpoint, fold-1 training set (for the kNN index) | `agentic_ai/cache/knn_fold1.npz`, `agentic_ai/outputs/fold1_*` CSVs/JSONs |

---

## 5. Codebase map

**Critical structural note before the table**: this project has **two parallel directory trees with identical names** because `src/config.py` uses relative paths (`./outputs`, `./checkpoints`, `./logs`, `./data_splits`) that resolve differently depending on whether you run scripts from the project root or from inside `src/`. The IEMOCAP LOSO work lives under `src/{outputs,checkpoints,logs,data_splits}/`; the MELD work lives under the project-root `{outputs,checkpoints,logs,data_splits}/`. **This is the single most important structural fact to internalize before touching any file path in this project.** [VERIFIED]

| File / folder | Purpose | Inputs | Outputs | Main functions/classes | Status | Warnings |
|---|---|---|---|---|---|---|
| `README.md` | Project intro, setup steps, IEMOCAP LOSO fold table, label-mapping table | — | — | — | Present, **stale** | Says "6-class", includes disgust as kept — contradicts actual 5-class code. Does not mention MELD or `agentic_ai/` at all. |
| `requirements.txt` | Python dependencies | — | — | — | Present, **incomplete** | Missing `scipy`, `statsmodels` (both actually used); `matplotlib`/`seaborn` marked optional but are hard requirements for 3 scripts that were run successfully |
| `src/config.py` | Single source of truth for all hyperparameters, paths, label maps | — | — | `Config` dataclass, `LABEL_TO_IDX`/`IDX_TO_LABEL`, `NUM_CLASSES` | Working, drives everything | Comment on line ~318 says `NUM_CLASSES # 6` — stale comment, actual value is 5. Read this first. |
| `src/data_preprocessing.py` | IEMOCAP metadata build, LOSO split logic, audio loading/caching, class weights | raw IEMOCAP corpus | `iemocap_metadata.csv`, `audio_cache/*.pt` | `build_metadata`, `get_split_dfs`, `compute_class_weights`, `load_and_preprocess_audio` | Working | `load_metadata()` validates label indices against current config and **raises** if stale — good, but means old metadata CSVs will hard-fail after a config change until regenerated |
| `src/meld_preprocessing.py` | MELD extraction (tarball→ffmpeg→metadata), label mapping to shared space | `MELD.Raw.tar.gz`, ffmpeg | `data_splits/meld/*.csv`, `data_splits/meld_audio/**/*.wav` | `extract_inner_tarballs`, `extract_audio_ffmpeg`, `build_meld_metadata` | Working, fully run | ~10.8GB of raw MELD tarballs left on disk — cleanup candidate |
| `src/dataset.py` | PyTorch `Dataset` wrappers | metadata DataFrame | batches | `TextDataset`, `AudioDataset`, `MultimodalDataset`, `build_dataloader` | Working | Seeded via `cfg.seed` (42) — good reproducibility practice |
| `src/models_text.py` | Text model | tokenized text | 5-class logits, or 3072-dim pooled embedding | `RoBERTaLoRA` | Working | Docstring likely still says "6-class" in places — verify before quoting |
| `src/models_audio.py` | Audio model | processed audio | 5-class logits, or frame embeddings | `WavLMLoRA` | Working | CNN feature extractor explicitly frozen; vectorized mask subsampling (rewritten from a slow loop, per in-code comment) |
| `src/models_fusion.py` | Fusion model, the one actually used downstream by `agentic_ai/` | text + audio | 5-class logits, `embed()` → 768-dim vector | `CrossModalAttentionFusion` | Working | `_subsample_mask` here is a **non-vectorized Python loop**, unlike the vectorized version in `models_audio.py` — inconsistent, minor perf issue, not a correctness bug |
| `src/train.py` | Main training entry point | CLI args, `cfg` | checkpoints, logs, metrics JSON/CSV | `main()` | Working, **the true entry point for every model-training experiment** | No `--resume` support — an interrupted fold restarts from scratch. Must be run once per model type; fold aggregation only happens if all 5 folds are run in the same invocation. |
| `src/metrics.py` | Metric computation, fold aggregation | predictions/labels | dict of metrics; `all_folds_summary.json` | `compute_metrics`, `aggregate_fold_results` | Working | No file I/O itself — called by `train.py` |
| `src/temperature_scaling.py` | MELD-only calibration correction | `checkpoints/{model}/meld_best.pt` (root) | `outputs/{model}/meld_*_predictions_probs.csv`, `outputs/analysis/calibration_plots_after_scaling.png` | — | Working, **MELD only** | Hardcodes root-level relative paths — will silently target the wrong tree if run from `src/` |
| `src/plot_calibration.py` | MELD-only ECE/reliability diagrams | same as above | `outputs/analysis/calibration_plots.png` | — | Working, MELD only | Same path-resolution risk as above |
| `src/plot_confusion_matrices_meld.py` | MELD confusion matrix plot | **hardcoded numbers in the script itself, not read from any file** | `outputs/analysis/confusion_matrices_meld.png` | — | Working, but **fragile** | If MELD models are retrained, this plot will NOT update automatically — someone must manually re-copy the new numbers into the script |
| `src/analyse_results.py` | Per-class F1 tables, confusion matrices, McNemar/Wilcoxon significance tests | `outputs/{model}/*predictions*.csv` under `OUTPUTS_ROOT="./outputs"` | `outputs/analysis/results_analysis.txt` | — | **Partially working — only MELD output ever saved, despite IEMOCAP support existing in the code** | Because `OUTPUTS_ROOT` is relative and this was run from a context where it resolved to the root `outputs/` (MELD-only), the IEMOCAP fold CSVs in `src/outputs/` were never picked up. See §11 Critical #2. |
| `agentic_ai/__init__.py` | package marker | — | — | — | trivial | — |
| `agentic_ai/result.py` | Typed containers | — | — | `RetrievedExample`, `AgentResult` | trivial, complete | — |
| `agentic_ai/agent.py` | Agent 3: single-utterance classify + retrieve | audio path + text, fold-1 checkpoint, fold-1 kNN index | `AgentResult` | `EmotionAgent` | Working, **fold=1 default and, in practice, the only fold ever used** | Default `fold=1` in constructor — nothing stops you from passing `fold=2..5`, but the required kNN cache files for those folds don't exist yet |
| `agentic_ai/rag.py` | kNN retrieval over a precomputed index | query embedding, `.npz` index | top-k neighbors | `KNNRetriever` | Working, explicitly a placeholder for a future CVR corpus | — |
| `agentic_ai/build_index.py` | Builds the kNN index from a fold's training set | fold's train split, fusion checkpoint | `agentic_ai/cache/knn_fold{k}.npz` | — | Working, **only run for fold 1** (`knn_fold1.npz` is the only cache file present) | To use folds 2–5 in the agentic layer, this must be run for those folds first — has not been done |
| `agentic_ai/alert_agent.py` | Rule-based misclassification-alert logic | `AgentResult` | `AlertResult` | `AlertAgent` | Working, empirically justified on fold-1 data only | Stage-2 threshold recently changed to 0.8 (this session) — provisional, fold-1-only validation |
| `agentic_ai/orchestrator.py` | Sequences EmotionAgent → AlertAgent | audio + text | `SupervisorResult` | `SupervisorAgent`, `MODEL_VERSION` constant | Working, deliberately minimal | `MODEL_VERSION` hardcodes "fold1" in the string — must be manually updated if a different fold is ever used operationally |
| `agentic_ai/evaluate_agent.py` | Full-split evaluation of `EmotionAgent` | a chosen fold/split | `fold{k}_{split}_agent_eval.csv/json` | — | Working, supports `--fold`/`--split`, but has only ever been invoked for fold 1 | Re-running this for folds 2-5 is a legitimate, low-cost next step (see §13) |
| `agentic_ai/evaluate_alert.py` | Alert-rule contingency-table evaluation (val vs test) | pre-computed eval CSVs | printed report | — | Working, fold-1 only | Explicitly distinguishes "reportable" (test) vs "not reportable" (val, threshold-tuning) results — good practice, follow this convention in your own writing |
| `agentic_ai/risk_weighted_f1.py` | Custom risk-weighted F1 (anger/fear up-weighted) | `agentic_ai/outputs/fold1_test_agent_eval.csv` | `fold1_risk_weighted_f1.txt` | — | Working, **hardcoded to fold 1, no CLI arg at all** | Risk weights explicitly marked "under review" in the code/output — do not present these as final without revisiting the weight choices |
| `agentic_ai/plot_roc_pr.py` | ROC/PR curves for the alert signal | fold-1 val/test CSVs | `roc_pr_curves.png` | — | Working, **hardcoded to fold 1** | Same as above |
| `agentic_ai/test_agent.py`, `agentic_ai/test_orchestrator.py` | Manual smoke tests (3 hardcoded example utterances) | — | printed output | — | Working as smoke tests, **not automated unit tests** (no `assert`) | Misleading filename prefix if you expect pytest-style tests |
| `outputs/analysis/` (root) | Saved MELD calibration/significance results | — | `.png`/`.txt` files | — | Present, MELD-only | No IEMOCAP equivalent directory exists anywhere |
| `checkpoints/`, `logs/`, `outputs/`, `data_splits/meld*` (root) | MELD run artifacts | — | — | — | Present, complete for MELD | Same names as the `src/`-level IEMOCAP trees — do not confuse them |
| `src/checkpoints/`, `src/logs/`, `src/outputs/`, `src/data_splits/` | IEMOCAP LOSO run artifacts | — | — | — | Present, complete for all 5 folds × 3 models | This is the tree the README actually describes, despite the README's folder diagram implying root-level |

**Where things are controlled (your direct questions, answered explicitly):**
- **Dataset paths**: `src/config.py` (`iemocap_root`, `meld_root`, and the relative-path fields `output_dir`/`checkpoint_dir`/`log_dir`/`splits_dir`/`meld_audio_dir`/`meld_splits_dir`). **You must know that these relative paths are why the two directory trees exist** — this is the #1 thing to understand about this codebase's file layout.
- **Random seeds**: `Config.seed = 42` (`src/config.py`), consumed by `dataset.py::build_dataloader` (`torch.Generator`, `seed_worker`) and presumably by `train.py` for model init (**[INFERRED]** — I did not personally verify every seeding call site in `train.py`; worth a direct check if exact bit-for-bit reproducibility matters to you).
- **Checkpoints/results storage**: see the dual-tree note above — `{src/}checkpoints/`, `{src/}outputs/`, `{src/}logs/`.
- **True entry point per experiment**: IEMOCAP training → `python train.py --model {text|audio|fusion} [--fold k]` run from inside `src/`. MELD training → same script with `--dataset meld`, but evidently run from the project root (see checkpoint locations). Agentic-layer evaluation → `python agentic_ai/evaluate_agent.py [--fold k] [--split ...]`, run from the project root (it manipulates `sys.path` internally to reach `src/`).
- **Files to understand first** (my recommendation, expanded on in §12): `src/config.py`, then `src/data_preprocessing.py::get_split_dfs`, then `src/models_fusion.py`, then `agentic_ai/agent.py` and `agentic_ai/alert_agent.py`.

---

## 6. Experiment inventory

| Experiment | Modality | Dataset | Model | Split/fold | Key settings | Status | Main result | Result location |
|---|---|---|---|---|---|---|---|---|
| IEMOCAP-text-LOSO | Text | IEMOCAP | RoBERTaLoRA | folds 1-5 | LoRA r=8/α=16, lr=3e-4, 30 epochs max, early-stop patience 5 | **Complete** | Macro F1 0.629 ± 0.052 | `src/outputs/text/all_folds_summary.json`, `fold{1-5}_test_metrics.json` |
| IEMOCAP-audio-LOSO | Audio | IEMOCAP | WavLMLoRA | folds 1-5 | same LoRA settings, CNN frozen | **Complete** | Macro F1 0.526 ± 0.021 | `src/outputs/audio/all_folds_summary.json` |
| IEMOCAP-fusion-LOSO | Text+Audio | IEMOCAP | CrossModalAttentionFusion | folds 1-5 | cross-attention, 4 heads | **Complete** | Macro F1 0.694 ± 0.054 | `src/outputs/fusion/all_folds_summary.json` |
| MELD-text-native | Text | MELD | RoBERTaLoRA | native (train/dev/test) | same LoRA settings | **Complete** | Macro F1 0.500 (test) | `outputs/text/meld_test_metrics.json` |
| MELD-audio-native | Audio | MELD | WavLMLoRA | native | same | **Complete** | Macro F1 0.342 (test) | `outputs/audio/meld_test_metrics.json` |
| MELD-fusion-native | Text+Audio | MELD | CrossModalAttentionFusion | native | same | **Complete** | Macro F1 0.499 (test) | `outputs/fusion/meld_test_metrics.json` |
| MELD calibration/temp-scaling | all 3 | MELD | all 3 | test | 10-bin ECE, scalar temperature via NLL | **Complete** | see §7 for pre/post-scaling ECE | `outputs/analysis/calibration_plots*.png` |
| MELD significance testing | all 3 | MELD | all 3 | test | McNemar (pairwise) | **Complete** | text≈fusion (not sig. different in the direction of improvement — fusion is numerically *lower*), both sig. > audio | `outputs/analysis/results_analysis.txt` |
| IEMOCAP significance testing | all 3 | IEMOCAP | all 3 | folds 1-5 | McNemar + Wilcoxon (script supports this) | **[ATTEMPTED/ABANDONED — code exists, output was never generated/saved]** | unknown — not computed | none — gap, see §11 |
| Class-conditional ECE (high-risk vs low-risk) | all 3 | MELD | all 3 | test | anger/fear vs joy/neutral/sadness grouping | **Complete** (done in this conversation session) | see §7 | `outputs/analysis/class_conditional_ece.txt`, `.png` |
| Agentic layer — EmotionAgent full eval | Text+Audio (via fusion ckpt) | IEMOCAP | fusion fold-1 checkpoint + kNN | fold 1 only, val + test | k=5 neighbors | **Complete for fold 1 only** | Macro F1 0.748 (test), matches baseline fusion checkpoint | `agentic_ai/outputs/fold1_test_agent_eval.csv`, `fold1_agent_summary.json` |
| Agentic layer — Alert rule (graded_agreement, t=0.6) | — | IEMOCAP | rule-based | fold 1 | threshold tuned on val (session 2) | **Complete for fold 1 only** | test: FPR/TPR/precision documented, ~28% recall over errors | `agentic_ai/outputs/fold1_test_agent_eval.csv`, printed report in `eval_orchestrator.py`/`evaluate_alert.py` |
| Agentic layer — Stage-2 rule (neutral_low_agreement) | — | IEMOCAP | rule-based | fold 1 | t=0.6 (redundant, 0 new alerts) then t=0.8 (59 new alerts) | **Complete for fold 1 only, very recent** | t=0.8: +18 TP/+41 FP, combined recall 0.342 vs primary 0.282 | `agentic_ai/outputs/fold1_test_agent_eval_stage2.csv` |
| Risk-weighted F1 | — | IEMOCAP | rule-based reweighting | fold 1 | anger/fear weight=3, joy/sadness=2, neutral=1 ("under review") | **Complete for fold 1 only** | Standard MacroF1 0.748 → Risk MacroF1 0.647 (mean variant) | `agentic_ai/outputs/fold1_risk_weighted_f1.txt` |
| ROC/PR for alert signal | — | IEMOCAP | rule-based | fold 1 | score = 1 − prediction_agreement | **Complete for fold 1 only** | AUC/PR-AUC reported, operating point at t=0.6 marked | `agentic_ai/outputs/roc_pr_curves.png` |
| HITL pilot utterance selection | — | IEMOCAP | — | fold 1 test | 2 per class, correct/wrong pairs, ≥2 high-confidence wrong | **Selection complete, no human study run yet** | 10 utterances curated | `agentic_ai/outputs/hitl_pilot_utterances.csv` |
| Case-study utterance scenarios | — | IEMOCAP | full SupervisorAgent | fold 1 test, dialog-level | 2 narrative scenarios, live-model-run (not just CSV replay) | **Complete** | 6 utterances w/ full pipeline output | `agentic_ai/outputs/case_study_utterances.csv` |
| Extended orchestrator decision log (CVR placeholder fields) | — | IEMOCAP | — | fold 1 | `utterance_timecode`/`flight_phase`=None, `model_version` populated | **Complete (schema only, no real CVR data)** | 1097-row extended log | `agentic_ai/outputs/fold1_orchestrator_eval_extended.csv` |
| IEMOCAP folds 2-5 through the agentic layer | — | IEMOCAP | — | folds 2-5 | — | **Not started** | — | none exist |
| IEMOCAP↔MELD zero-shot/transfer test | — | both | — | — | — | **Not started / not planned in code** — only parallel independent training exists | — | none |

---

## 7. Verified results

### 7a. IEMOCAP — 5-fold LOSO, TEST set (final, reproducible-in-principle from checkpoints)

| Model | Macro F1 | Weighted F1 | Weighted Acc | Unweighted Acc |
|---|---|---|---|---|
| Text | 0.6291 ± 0.0516 | 0.6681 ± 0.0259 | 66.62% ± 2.56% | 69.12% ± 6.09% |
| Audio | 0.5262 ± 0.0213 | 0.6361 ± 0.0144 | 64.13% ± 1.66% | 55.40% ± 4.86% |
| Fusion | 0.6935 ± 0.0544 | 0.7386 ± 0.0115 | 73.98% ± 1.14% | 74.64% ± 6.22% |

Source: `src/outputs/{text,audio,fusion}/all_folds_summary.json` — mean ± sample std (ddof=1) across the 5 held-out test sessions. **These are TEST results, not validation** — validation was used only for early stopping / checkpoint selection within each fold.

Fusion per-class F1 (mean ± std across folds): anger 0.790±0.037, fear **0.485±0.278**, joy 0.762±0.024, neutral 0.663±0.029, sadness 0.768±0.034. The huge std on fear (0.278, roughly 57% of its own mean) directly reflects its tiny sample count and should always be reported with its std, never as a bare point estimate.

### 7b. MELD — native split, TEST set

| Model | Macro F1 | Weighted F1 | Weighted Acc | Unweighted Acc |
|---|---|---|---|---|
| Text | 0.5000 | 0.6752 | 66.87% | 52.37% |
| Audio | 0.3417 | 0.4752 | 44.80% | 37.28% |
| Fusion | 0.4993 | 0.6603 | 64.26% | 53.61% |

Source: `outputs/{text,audio,fusion}/meld_test_metrics.json` (root-level), cross-checked against `outputs/analysis/results_analysis.txt` and `logs/fusion/meld_train.log`'s printed test line — all three agree exactly.

**Notable finding, stated plainly**: on IEMOCAP, fusion beats text by +0.064 macro F1; on MELD, fusion is *numerically slightly worse* than text (−0.0007). This is a real, measured discrepancy between the two datasets, not noise-level — you should be prepared to discuss *why* fusion might not help on MELD (my best inference, **not verified in the code or any written note**: MELD audio is likely noisier/lower-quality — background laughter, multiple overlapping speakers in a sitcom recording — versus IEMOCAP's controlled elicitation setup, so the audio modality may be contributing less useful signal and more noise to the fusion on MELD).

### 7c. MELD calibration (ECE, 10 bins)

Class-conditional (this session's output): [VERIFIED]

| Model | High-risk (anger, fear) ECE | Low-risk (joy, neutral, sadness) ECE |
|---|---|---|
| Text | 0.0916 | 0.1110 |
| Audio | 0.0956 | 0.1534 |
| Fusion | 0.0937 | 0.1255 |

Overall pattern: **neutral is the worst-calibrated class for every model** (0.25–0.32 ECE) — likely because it's the majority class and the model's confidence there is systematically overconfident. High-risk classes are, somewhat counterintuitively, *better* calibrated in aggregate than low-risk ones, but this is dragged down by `neutral` specifically — `fear` alone (0.146–0.158 ECE) is worse-calibrated than `anger` (0.026–0.045).

Overall (not class-split) ECE before/after temperature scaling: **[NOT independently re-extracted in this audit — the values exist in `outputs/analysis/calibration_plots_after_scaling.png` and were computed by `src/temperature_scaling.py`, but I did not pull the exact before/after numeric ECE values into text form. Read that script's console output or re-derive from the saved probability CSVs if you need the exact numbers for a table.]** [PLANNED BUT NOT FULLY EXTRACTED IN THIS AUDIT]

### 7d. Agentic layer — fold 1 test set only (n=1097, IEMOCAP Session 1)

- Classification via the agent wrapper: Macro F1 = 0.7476 (vs. the raw fusion checkpoint's 0.7483 — the ~0.0007 gap is expected floating-point/AMP nondeterminism between the training-time eval and the agent's separate forward pass, not a bug).
- Retrieval agreement: **overall 0.665**; **when classifier correct, 0.841**; **when classifier wrong, 0.200** — this ~4× gap is the empirical basis for the entire alert-rule design.
- Retrieval "rescue rate": of 301 wrong predictions, 149 (49.5%) had the correct label somewhere in their top-5 retrieved neighbors — i.e., the retrieval evidence *could* have corrected roughly half of the model's errors if used more aggressively than the current simple threshold rule.
- Primary alert rule (t=0.6): 189/1097 alerts fired; 85 true positives, 104 false positives → precision 0.450, recall (over all 301 errors) 0.282, F1 0.347.
- Stage-2 rule at t=0.8 (current default as of this session): 248/1097 alerts fired; 103 TP, 145 FP → precision 0.415, recall 0.342, F1 0.375. Of the 30 anger utterances misclassified as neutral, 19 are now caught (vs 17 under the primary rule alone).

**All of §7d is fold-1-only.** No equivalent numbers exist for folds 2-5 in the agentic layer — this is repeated deliberately across this document because it is the single most important reproducibility caveat for anything you say about the "agent" or "alert system."

### Reproducibility check
- Checkpoints exist for every reported result → in principle every number above is reproducible by re-running inference against the saved checkpoint (I did not personally re-run inference to bit-for-bit confirm the JSON numbers — I read the saved JSON/CSV/log files directly, which is standard practice for an audit but is not the same as an independent re-derivation).
- The one exception: `src/plot_confusion_matrices_meld.py`'s hardcoded arrays are **not** re-derivable from a re-run without manually re-reading a fresh confusion matrix and re-editing the script.

---

## 8. Important decisions

| Decision | Rationale (as found in code/comments) | Alternatives considered | Advantages | Limitations | Defensible? | Revisit? |
|---|---|---|---|---|---|---|
| Drop disgust entirely (both datasets) | Only 2 IEMOCAP samples | Merge into another class; oversample | Avoids an unlearnable class dragging down macro metrics | Loses disgust as a category entirely — can't claim coverage of it | Yes, clearly defensible given n=2 | No |
| Merge happy+excited → joy | [INFERRED — no explicit comment found justifying this merge specifically, only that it was done] | Keep as 2 separate classes | Larger, more learnable positive-affect class | Loses the excited/happy distinction, which might matter for a stress-monitoring use case (e.g., "excited" could look similar to certain stress signatures) | Reasonably defensible, but **you should have your own justification ready since the code doesn't state one explicitly** | **[NEEDS YOUR INPUT]** — decide if you want to defend this as-is or investigate further |
| Keep fear despite ~40 total samples | comment notes low support but doesn't drop it (unlike disgust) | Drop it like disgust; oversample it; report it separately without training on it | Keeps a class that's directly operationally relevant to a CVR/aviation-stress use case | Extremely high variance in reported fear F1 (±0.278 across folds) — any single-fold number for fear is close to meaningless on its own | Defensible **if you present it with its std and are explicit about the small-sample caveat**; not defensible if presented as a clean point estimate | Not necessarily — but your *presentation* of fear results needs revisiting (always show std/CI, never a bare F1) |
| LOSO (session-independent) cross-validation | Speaker/session independence needed for any claim about generalizing to unseen speakers | Random k-fold (would leak speaker identity across train/test); single train/test split | Correct protocol for the "new person" deployment scenario relevant to CVR | Only 5 folds (small — more folds isn't possible since IEMOCAP has exactly 5 sessions); std across folds is large for small classes | Yes, this is the standard, correct choice for IEMOCAP | No |
| RoBERTa-base + WavLM-base-plus as backbones | [INFERRED — no comparison to alternative backbones found anywhere in the repo] | Larger models (RoBERTa-large, WavLM-large), other audio encoders (Wav2Vec2, HuBERT) | Smaller/faster, fits compute budget with LoRA | No evidence a larger/different backbone was tried and rejected — this is an unexplored direction, not a validated choice | Defensible as a reasonable, compute-conscious default; **not** defensible as "we chose this because it's best" — you have no comparison evidence for that claim | **[NEEDS YOUR INPUT]** — decide if backbone ablation is in scope for this project or explicitly out of scope |
| LoRA (rank 8, alpha 16) over full fine-tuning | [INFERRED — no explicit written rationale found, but this is a very standard, defensible choice for compute-constrained fine-tuning] | Full fine-tuning; other PEFT methods (prefix tuning, adapters) | Far fewer trainable params, faster training, smaller checkpoints, less overfitting risk on a small dataset like IEMOCAP | No published ablation in this repo comparing LoRA vs full fine-tuning performance | Defensible as a standard/expected choice; if asked "did you check full fine-tuning would be worse," the honest answer is no, this was not tested here | Optional — nice-to-have if time allows, not essential |
| Cross-attention fusion (text=Query, audio=K/V) + concatenation of 3 pooled vectors | [INFERRED — no explicit comparison to alternative fusion strategies, e.g. simple concatenation without attention, found in the repo] | Late fusion (average logits); simple concatenation of pooled embeddings without cross-attention; gated fusion | Lets text attend to relevant audio frames rather than treating audio as a fixed global vector | More complex, more parameters, and on MELD it did not clearly outperform text-only — so the added complexity's benefit is dataset-dependent | Defensible on IEMOCAP evidence; the MELD result (§7b) is a real limitation you must be ready to discuss, not something to omit | Worth investigating *why* it doesn't help on MELD if you have time (see §13) |
| kNN retrieval (RAG) over training embeddings as a "review evidence" mechanism | Explicitly framed as a prototype substitute for a future CVR-specific knowledge base | A learned confidence/OOD-detection head instead of kNN; no evidence mechanism at all | Simple, interpretable ("here are 5 similar examples"), auditable — good for a decision-support tool aimed at human reviewers | Brute-force cosine similarity over ~3,400 vectors — fine at this scale, would need reengineering (FAISS etc.) at real CVR-corpus scale; explicitly a placeholder, not a finished CVR retrieval system | Defensible as a prototype; **not** defensible if presented as "the CVR retrieval system" — it's explicitly a stand-in | No, as long as you present it as a prototype, which the code itself already does correctly |
| Alert rule threshold t=0.6, chosen over the F1-optimal t=0.9 | Explicitly reasoned in `alert_agent.py`'s docstring: t=0.9 gives higher val F1 but 37.6% test FPR — "alert fatigue unacceptable in a safety-critical context" | t=0.9 (F1-optimal); a learned classifier instead of a threshold rule | Directly prioritizes a safety-relevant constraint (false-alarm rate) over a pure accuracy metric | Still ~17.6% FPR at t=0.6 — explicitly documented as not a standalone safety gate | Yes, this is one of the best-reasoned decisions in the whole codebase — good example to cite when asked about safety-conscious design choices | No |

---

## 9. What you must understand personally

### Essential — you must be able to explain and defend these yourself

**1. Why fear was kept despite having almost no data, and what that means for interpreting any result involving it.**

*Plain language:* Imagine trying to learn what "scared" sounds like from only about 40 example recordings, spread thin across 5 different test rounds. Some rounds might have almost none of those examples in the test set at all — so the "score" for fear in any one round can swing wildly just from bad luck about which few examples happened to land in that round, not because the model got meaningfully better or worse.

*Technical version:* fear's per-fold IEMOCAP F1 has mean 0.485 with standard deviation 0.278 across the 5 LOSO folds — the standard deviation is more than half the mean. Any claim about fear performance must be accompanied by this variance, and ideally by the raw per-fold numbers, not a collapsed average.

*Project example:* `src/outputs/fusion/all_folds_summary.json` — look at the `f1_per_class` std for "fear" versus any other class.

**2. The difference between `retrieval_agreement` (used only in offline evaluation, computed against ground truth) and `prediction_agreement` (used operationally, by the actual alert rule, computed against the prediction).**

*Plain language:* One number asks "if I already knew the right answer, would the retrieved examples have agreed with it?" — useful for research analysis, but you can't compute it in a real deployment because you don't know the right answer yet. The other number asks "do the retrieved examples agree with what the model just guessed?" — this one you *can* compute in real time, which is why it's the one the actual alert system uses.

*Technical version:* `retrieval_agreement` (in `evaluate_agent.py`, stored in `fold1_test_agent_eval.csv`) = fraction of top-5 neighbors sharing the *ground truth* label. `prediction_agreement` (in `alert_agent.py::_prediction_agreement`, `eval_orchestrator.py::compute_pred_agreement`) = fraction of top-5 neighbors sharing the *predicted* label. This distinction was directly confirmed earlier in this conversation using the utterance `Ses01M_script03_1_M003` (predicted=neutral, ground truth=fear, all 5 neighbors labeled neutral → retrieval_agreement=0.0 despite the neighbors unanimously agreeing with the *prediction*).

*Project example:* re-derive this yourself: open `agentic_ai/outputs/fold1_test_agent_eval.csv`, pick any row, and manually recompute both numbers from its `neighbor_labels` column against both `ground_truth` and `predicted` — they will usually differ.

**3. Why the agentic-layer numbers (alert rule performance, retrieval agreement, risk-weighted F1) are fold-1-only, and what that means for any claim you make about them.**

*Plain language:* The core classifier was tested five different ways (once per session, so every session gets a turn as the "unseen" test set) — that part is solid. But the "smart alerting" layer built on top of it was only ever tested using *one* of those five test rounds. So you know the classifier generalizes across different speakers, but you don't yet know if the alerting behavior does.

*Technical version:* only `agentic_ai/cache/knn_fold1.npz` exists; all files in `agentic_ai/outputs/` are `fold1_*`; `orchestrator.py::MODEL_VERSION` is hardcoded to `"...fold1"`. Folds 2-5 have trained fusion checkpoints (`src/checkpoints/fusion/fold{2,3,4,5}_best.pt`) that could be used to build the missing kNN indices and re-run the evaluation, but this has not been done.

**4. The dual-directory-tree structural quirk (`src/checkpoints/` vs `checkpoints/`, etc.) and why it exists.**

*Plain language:* Two completely separate sets of experiment results live in folders with the same names, because the code uses "look in the folder called `outputs` relative to wherever I'm currently running from" rather than a fixed absolute path — so running the IEMOCAP experiments from inside the `src` folder and running the MELD experiments from the main project folder produced two different physical locations that happen to share folder names.

*Technical version:* `src/config.py`'s `output_dir`, `checkpoint_dir`, `log_dir`, `splits_dir` are all relative paths (`"./outputs"` etc.), resolved against the current working directory at script-run time, not against the project root or the `src/` folder specifically.

*Project example:* compare `src/outputs/fusion/fold1_test_metrics.json` (IEMOCAP) against `outputs/fusion/meld_test_metrics.json` (MELD, root-level) — same filename pattern, completely different experiment.

### Important — you should understand the logic and trade-offs, but can consult details as needed

- LoRA mechanics (rank/alpha/target modules) — you should be able to explain *why* PEFT is used (fewer trainable parameters, less overfitting risk on a small dataset) without necessarily reciting every hyperparameter from memory.
- Cross-attention fusion mechanics (text as Query, audio as Key/Value) — understand the high-level idea (text "looks up" relevant moments in the audio) without needing to trace every tensor shape.
- Class weighting formula and cap (`max_class_weight=5.0`) — understand why it's capped (prevents the rare fear class from destabilizing training) without needing the exact formula memorized.
- Temperature scaling for calibration — understand the goal (make confidence scores trustworthy) without needing to explain the NLL-minimization procedure in detail.
- ECE computation and the high-risk/low-risk grouping — understand what it measures and the headline finding (neutral is worst-calibrated) without needing to recompute it by hand.

### Optional — implementation detail, consult when needed

- Exact audio preprocessing constants (sample rate, RMS target, cache file naming convention).
- Exact PyTorch `DataLoader` seeding mechanics.
- The non-vectorized vs vectorized mask-subsampling difference between `models_fusion.py` and `models_audio.py`.
- Exact CLI flag names and their defaults across the various `agentic_ai/` scripts.

---

## 10. Supervisor and examiner questions

For each, I give a short defensible answer, a deeper answer, the evidence, and a weakness you should acknowledge honestly.

### Research motivation
**Q: Why work on IEMOCAP/MELD instead of real CVR data?**
- *Short:* Real CVR data requires aviation-authority (GCAA) access that isn't available yet; IEMOCAP/MELD let the full modeling and evaluation pipeline be built and validated now, ready to be pointed at CVR data later.
- *Deeper:* the project is explicitly architected around this — the retrieval corpus, alert thresholds, and orchestrator all carry comments describing themselves as prototypes/placeholders for the CVR case, and the orchestrator's output schema already has empty `flight_phase`/`utterance_timecode` fields reserved for that future data.
- *Evidence:* `agentic_ai/rag.py` module docstring, `agentic_ai/orchestrator.py`'s CVR-context fields.
- *Weakness to acknowledge:* until real CVR data is used, **no claim in this project transfers to aviation audio with any guarantee** — acted/TV-show emotional speech is acoustically and linguistically different from cockpit communication (different vocabulary, radio/headset audio characteristics, likely much flatter affect even under real stress).

### Novelty
**Q: What's actually new here versus just running an existing IEMOCAP benchmark?**
- *Short:* The combination of (a) properly-validated LOSO multimodal LoRA fusion, (b) a cross-corpus check on MELD, and (c) a retrieval-based review/alerting layer designed with an operational, safety-conscious threshold choice.
- *Deeper:* individually, LoRA fine-tuning and cross-modal attention fusion are established techniques; the project's contribution is the *integration* and the explicit framing toward a decision-support (not just classification) use case, with an alert rule whose threshold was chosen specifically to control false-alarm rate rather than maximize F1.
- *Evidence:* `agentic_ai/alert_agent.py` docstring's explicit rejection of the F1-optimal threshold.
- *Weakness:* the novelty claim rests more on the systems-integration/framing than on a new modeling technique — be ready for "what's the technical contribution beyond engineering," and have your own answer ready since the code doesn't state one explicitly.

### Dataset selection
**Q: Why IEMOCAP and MELD specifically, and not other emotion datasets (CREMA-D, RAVDESS, etc.)?**
- *Short:* IEMOCAP is a standard, well-studied benchmark with session/speaker structure suited to LOSO; MELD adds naturalistic, noisier multi-speaker dialogue as a harder cross-corpus check.
- *Evidence:* `src/config.py`'s comment referencing CREMA-D by name when explaining why `frustration` was dropped ("not present in MELD/CREMA-D") — implying CREMA-D was considered as a *reference* for label-space compatibility even though it wasn't used as training data.
- *Weakness:* no other dataset was actually trained on or compared against — "why not X" only has an answer for CREMA-D's *label space* being referenced, not for why it wasn't used as actual data.

### Fear-class imbalance
**Q: With ~40 fear samples total, is any fear result meaningful?**
- *Short:* Individually, no single fold's fear F1 should be trusted; the 5-fold mean (0.485) with its large std (0.278) is the honest way to report it.
- *Deeper:* the alternative (dropping fear like disgust) was available and not taken — worth being explicit that this was a deliberate choice given fear's operational relevance to a stress/danger-detection use case, at the acknowledged cost of a very noisy metric.
- *Evidence:* `src/outputs/fusion/all_folds_summary.json` per-class std.
- *Weakness:* be ready for "why not oversample or synthesize fear examples" — the honest answer is that this was not attempted in this project.

### Experimental design
**Q: Did you tune hyperparameters on the test set at any point?**
- *Short:* No — checkpoint selection and alert-threshold tuning both used validation splits, evaluated finally on disjoint test splits.
- *Evidence:* `train.py` selects on val Macro F1; `evaluate_alert.py` explicitly labels val-set results "not reportable" and only test results "final".
- *Weakness:* I did not personally verify that no hyperparameters (learning rate, LoRA rank, etc.) were manually adjusted based on test-set feedback during earlier development — this is [INFERRED good practice from the code structure], not something I can prove from a static snapshot of the repo.

### Speaker independence
**Q: Is train/test genuinely speaker-independent?**
- *Short:* Yes for IEMOCAP — LOSO by session, and each session's 2 speakers never appear in any other session.
- *Evidence:* `src/data_preprocessing.py::get_split_dfs` splits strictly by `session` column.
- *Weakness:* for MELD, the native split's speaker independence was not directly verified in this audit — MELD's many recurring TV-show characters mean the *same actors* likely appear across train/dev/test (this is a known property of MELD, not something this project controls), which is a real limitation of the MELD cross-corpus check as a "generalization" claim.

### Cross-corpus generalization
**Q: Does the IEMOCAP-trained model work on MELD?**
- *Short:* This wasn't directly tested — what was tested is whether the *same architecture, retrained on MELD*, performs comparably. That's a different (weaker) claim than transfer/generalization.
- *Evidence:* separate `meld_best.pt` checkpoints per model type, trained from scratch — see §2 caveat.
- *Weakness:* be very careful not to claim "our model generalizes to MELD" in a defense — the honest framing is "our training recipe reproduces reasonable performance on a second, harder dataset," which is meaningfully weaker and you should present it that way.

### Model selection
**Q: Why RoBERTa/WavLM and not larger or different backbones?**
- *Short:* Reasonable, compute-conscious defaults for LoRA fine-tuning; no backbone ablation was run.
- *Weakness:* acknowledge directly that this is unexplored, not a validated-best choice.

### LoRA
**Q: Why parameter-efficient fine-tuning instead of full fine-tuning?**
- *Short:* Fewer trainable parameters, faster training, lower overfitting risk on a comparatively small dataset (IEMOCAP: ~5,571 utterances).
- *Weakness:* no full-fine-tuning comparison exists in this repo to quantify what, if anything, was traded away.

### Fusion
**Q: Does multimodal fusion actually help?**
- *Short:* Yes on IEMOCAP (+6.4 macro-F1 points over text alone), no on MELD (fusion is marginally *below* text-only).
- *Weakness:* this inconsistency is real and must be presented, not hidden — and the significance of the IEMOCAP gap has not been formally tested (see §11 Critical #2), so "yes on IEMOCAP" should currently be phrased as "numerically higher," not "significantly better," until that test is actually run and saved.

### Evaluation metrics
**Q: Why Macro F1 as the primary metric?**
- *Short:* It prevents the majority class (neutral) from dominating the score, which matters given the dataset's real imbalance.
- *Evidence:* class distribution implied by fear's ~40-sample count versus thousands of neutral utterances.

### Limitations
**Q: What's the single biggest limitation of this work?**
- *Short, honestly:* No real CVR/aviation data has been used anywhere — every claim in this project is about acted or TV-show emotional speech, not cockpit speech. This is explicitly acknowledged in the code's own comments (GCAA data access pending), so it is not a hidden flaw — but it is the limitation that most determines what you can and cannot claim.

### Reproducibility
**Q: Can these results be reproduced?**
- *Short:* Checkpoints, logs, and metrics all exist on disk for every reported IEMOCAP and MELD number, so in principle yes — but there is no fixed environment lockfile beyond `requirements.txt`, and that file is itself missing two dependencies (`scipy`, `statsmodels`) that scripts actually require, and lists two more (`matplotlib`, `seaborn`) as merely optional when they're hard requirements for scripts that were clearly run. A fresh environment built strictly from `requirements.txt` would fail on those scripts.
- *Weakness:* acknowledge the requirements.txt gap directly if asked about reproducibility — it's a small, fixable issue, but it's currently real.

### Operational relevance to CVR analysis
**Q: How would this actually be used in a real CVR investigation?**
- *Short:* Not yet — this is a decision-support *prototype architecture* (classify + retrieve evidence + rule-based alert) validated on public speech data, intended to be re-pointed at real CVR audio and FDR-derived context once available.
- *Weakness:* be ready to admit that "how would this integrate into an actual investigation workflow" has not been designed beyond the placeholder `flight_phase`/`utterance_timecode` fields — there's no user interface, no investigator-facing tool, no integration with existing CVR analysis software described anywhere in this repo.

---

## 11. Gaps, risks, and questionable areas — critical audit

### Critical
1. **The entire agentic/RAG/alert layer has only ever been evaluated on IEMOCAP fold 1.** The underlying classifier is fully validated across all 5 LOSO folds, but every retrieval-agreement number, every alert-rule precision/recall number, and the risk-weighted F1 are single-fold. Any claim like "the alert rule catches X% of errors" currently only holds for one specific train/test partition. Fix: build `knn_fold{2,3,4,5}.npz` via `agentic_ai/build_index.py`, then re-run `evaluate_agent.py`/`evaluate_alert.py`/`risk_weighted_f1.py`/`plot_roc_pr.py` for those folds — all the code needed already exists and is parameterized for this (`risk_weighted_f1.py` and `plot_roc_pr.py` would need small edits to accept a `--fold` arg first, since they currently hardcode fold-1 filenames).
2. **IEMOCAP significance testing (McNemar/Wilcoxon) was never actually run and saved, despite the code fully supporting it.** `src/analyse_results.py` is written to handle both datasets, but its one saved output (`outputs/analysis/results_analysis.txt`) is MELD-only, because it was evidently run in a context where its relative `OUTPUTS_ROOT` path resolved to the root `outputs/` tree (MELD) rather than `src/outputs/` (IEMOCAP). This means the headline IEMOCAP fusion-vs-text gap (0.694 vs 0.629 macro F1) has never been formally tested for statistical significance — you currently cannot say "fusion significantly outperforms text on IEMOCAP," only "fusion scores numerically higher."
3. **The README describes a 6-class system (including "disgust" as a kept, mapped class) that directly contradicts the actual running 5-class code** (`src/config.py:125`: `"dis": None  # drop disgust`). Anyone reading only the README — including a supervisor or examiner skimming the repo before a meeting — would form an incorrect understanding of the label space. This must be corrected before the README is shared with anyone external.

### Important
4. **`requirements.txt` is out of date**: missing `scipy` and `statsmodels` (both actually imported and used by working scripts), and mislabels `matplotlib`/`seaborn` as optional when they're hard dependencies of scripts that were run successfully and produced saved output. A fresh clone would fail to run the analysis scripts without manually adding these.
5. **No baseline outside the project's own three models.** There's no majority-class baseline, no published-SOTA comparison number, nothing external to anchor "is 0.694 macro F1 actually good" against. This weakens any performance claim in a write-up.
6. **`plot_confusion_matrices_meld.py` uses hardcoded numeric arrays instead of reading a results file.** If the MELD models are ever retrained, this specific plot will keep showing old numbers until someone manually notices and edits the script. Currently the hardcoded numbers do match the latest saved results (cross-checked against `results_analysis.txt`), so there's no active inconsistency today — but the coupling is fragile.
7. **The "cross-corpus" framing needs clarification with your supervisor** (see §2, §10): what exists is two independently-trained models on two datasets, not a transfer/generalization test of one model across both. If your thesis/paper is expected to make a transfer claim, that experiment doesn't exist yet.
8. **MELD speaker independence was not verified in this audit** — MELD's native split likely has recurring actors across train/dev/test (an inherent property of the dataset, not a bug in this project's code), which weakens how strongly you can frame the MELD result as a "generalization to unseen speakers" check versus IEMOCAP's genuinely clean LOSO protocol.
9. **The stage-2 alert-rule threshold (0.8) is very recently chosen (this session) and validated on fold 1 only** — treat it as provisional, not a finished design decision, until it's checked against other folds.
10. **Risk-weighted F1 weights are explicitly marked "under review" in the code/output itself** — do not present the 0.647 risk-weighted number as a settled, final metric without revisiting the weight scheme (anger=3, fear=3, joy=2, sadness=2, neutral=1) and being ready to justify those specific numbers.
11. **~10.8GB of raw MELD tarballs remain on disk** (`src/MELD.Raw/{train,dev,test}.tar.gz`) unremoved after extraction — not a correctness issue, but worth cleaning up if disk space or repo portability matters.

### Minor
12. **Inconsistent mask-subsampling implementation** between `models_audio.py` (vectorized) and `models_fusion.py` (Python for-loop) — a performance inconsistency, not a correctness bug, but worth unifying for cleanliness.
13. **Stale code comments beyond the README** (e.g., `NUM_CLASSES  # 6` in `src/config.py`, similar 6-class references likely present in `models_text.py`/`models_audio.py`/`models_fusion.py` docstrings — I confirmed the config.py instance directly; the model-file instances are [INFERRED from the audit agent's report, not independently re-verified by me line-by-line] and worth a quick grep pass to confirm and fix).
14. **`test_agent.py`/`test_orchestrator.py` are manual smoke tests, not automated unit tests** (no `assert` statements) — misleading given the `test_` filename convention; fine as-is for a prototype but shouldn't be described as "we have unit tests" without qualification.
15. **No documented seed-verification for full bit-for-bit reproducibility of `train.py`** — the seeding infrastructure exists (`cfg.seed=42`, used in `dataset.py`), but I did not independently confirm every relevant seeding call site inside `train.py` itself.

---

## 12. Reintegration learning plan

Each session ≈30–45 minutes. Do them in order — later sessions assume earlier ones.

### Session 1 — The research question and why these datasets
- **Concept:** what problem this project addresses and why IEMOCAP/MELD stand in for real CVR data.
- **Files to inspect:** `README.md` (read critically — you now know parts of it are stale), `agentic_ai/rag.py` (just the module docstring at the top).
- **Plain explanation:** see §1 of this guide.
- **Code-reading task:** find and read `src/config.py` lines defining `iemocap_root` and `meld_root` — just locate them, don't analyze yet.
- **Questions to answer in your own words:** Why can't this project use real CVR data yet? What would need to change to use it once available?
- **Self-check:** Can you explain, without notes, why the label set has 5 classes and not 6?
- **Practical action:** write a one-paragraph corrected version of the README's "6-class" opening claim (you don't have to edit the file yet, just draft the correction).

### Session 2 — The label space and dataset exclusions
- **Concept:** exactly which emotions were kept, merged, or dropped, and why.
- **Files to inspect:** `src/config.py` lines ~114-152 (the `label_map`, `meld_label_map`, `emotion_classes`).
- **Plain explanation:** see §2 "Emotion classes and why" above.
- **Code-reading task:** trace one raw IEMOCAP label ("exc") through `label_map` to its final class ("joy").
- **Questions to answer:** Why was frustration dropped specifically for cross-corpus reasons, not just rarity? What's the difference between "merged" and "dropped"?
- **Self-check:** Without looking, list the 5 final classes and which raw labels feed into each.
- **Practical action:** open `src/data_splits/iemocap_metadata.csv` and confirm row count / spot-check a few `emotion` values against `raw_label`.

### Session 3 — LOSO splitting and why it matters
- **Concept:** session-independent, speaker-independent cross-validation.
- **Files to inspect:** `src/data_preprocessing.py::get_split_dfs` (find and read this one function only).
- **Plain explanation:** see §2 "Train/val/test protocol — IEMOCAP".
- **Code-reading task:** for fold=3, work out by hand which session is test, which is val, which three are train.
- **Questions to answer:** Why does the validation session "wrap around" (fold%5 + 1) instead of always being session 1?
- **Self-check:** Explain to an imaginary examiner why random k-fold splitting would be wrong for this dataset.
- **Practical action:** open one `src/outputs/fusion/fold{k}_test_predictions.csv` and confirm its utterance IDs all come from session k.

### Session 4 — The dual-directory-tree structural quirk
- **Concept:** why `checkpoints/`, `outputs/`, `logs/` exist twice with different content.
- **Files to inspect:** `src/config.py`'s path fields; directory listings of `src/checkpoints/` vs root `checkpoints/`.
- **Plain explanation:** see §9 Essential concept 4.
- **Code-reading task:** none — this is a filesystem-navigation task. List both `checkpoints/` trees yourself and note the difference.
- **Questions to answer:** If you ran `python train.py` from the project root right now instead of from `src/`, where would the new checkpoint land?
- **Self-check:** Can you locate, without help, the fusion fold-1 IEMOCAP checkpoint versus the fusion MELD checkpoint?
- **Practical action:** write down (on paper or in a notes file, not in the code) the absolute path to each of the four IEMOCAP result trees and each of the four MELD result trees, so you never have to re-derive this under pressure.

### Session 5 — Model architecture, plain-language first
- **Concept:** what RoBERTaLoRA, WavLMLoRA, and CrossModalAttentionFusion each actually do.
- **Files to inspect:** `src/models_text.py`, `src/models_audio.py`, `src/models_fusion.py` — just the class docstrings and `__init__` signatures, not the full forward-pass math yet.
- **Plain explanation:** text model reads words, audio model reads sound, fusion model lets the text model "point at" relevant moments in the audio.
- **Code-reading task:** find where each model's output dimension is defined and confirm it equals `NUM_CLASSES` (5).
- **Questions to answer:** What does "frozen CNN feature extractor" mean and why is it frozen only in the audio model, not elsewhere?
- **Self-check:** In one sentence each, describe what LoRA does and why it's used here instead of fully retraining the whole model.
- **Practical action:** none required beyond reading — this session is about vocabulary you'll need for every later session.

### Session 6 — Reading real results tables without fooling yourself
- **Concept:** distinguishing val from test, single-fold from averaged, and knowing what "reportable" means.
- **Files to inspect:** `src/outputs/fusion/all_folds_summary.json`, `agentic_ai/outputs/fold1_val_agent_eval.csv` vs `fold1_test_agent_eval.csv`.
- **Plain explanation:** see §7 of this guide.
- **Code-reading task:** find the exact line in `evaluate_alert.py` (or its docstring) that labels val results as non-reportable.
- **Questions to answer:** Why would reporting a validation-set number as your final result be misleading, even if the number looks good?
- **Self-check:** State fear's mean IEMOCAP fusion F1 *and* its std, from memory, and explain why the std matters here specifically.
- **Practical action:** build your own one-page results table (by hand, from the JSONs) before you next need to present this work — don't wait until the night before a meeting.

### Session 7 — The agentic layer: what "retrieval agreement" actually measures
- **Concept:** kNN retrieval as evidence, and the retrieval_agreement vs prediction_agreement distinction.
- **Files to inspect:** `agentic_ai/rag.py::KNNRetriever.retrieve`, `agentic_ai/alert_agent.py::_prediction_agreement`.
- **Plain explanation:** see §9 Essential concept 2.
- **Code-reading task:** manually recompute both agreement numbers for one row of `fold1_test_agent_eval.csv` (pick any row with `correct=False`).
- **Questions to answer:** Why is only prediction_agreement usable in a real deployment?
- **Self-check:** explain why the alert rule's threshold (0.6) was chosen over the F1-optimal threshold (0.9) — this is one of the strongest, most citable design decisions in the project.
- **Practical action:** identify, from `agentic_ai/outputs/fold1_test_agent_eval.csv`, one alerted false-positive case (alert fired, prediction was correct) and one missed false-negative case (no alert, prediction was wrong) — understand each concretely.

### Session 8 — What's missing, and owning the gap list
- **Concept:** turning §11 of this guide into your own working checklist.
- **Files to inspect:** none new — this is a synthesis session using this guide and §11 specifically.
- **Plain explanation:** you now have enough context to understand every item in §11 without re-explanation.
- **Code-reading task:** open `agentic_ai/build_index.py` and identify exactly what would need to run (in what order) to produce `knn_fold2.npz`.
- **Questions to answer:** Of the 3 "Critical" gaps in §11, which one would you prioritize first, and why?
- **Self-check:** Without looking, list the 3 critical and at least 3 of the 8 important gaps.
- **Practical action:** decide (see §13 below) which gaps you want to close before your next supervisor meeting, and which you'll present as known, acknowledged limitations instead.

---

## 13. Immediate next steps

### Steps you should understand or decide yourself (no code needed)
- Decide how to frame the MELD experiment: "cross-corpus generalization" (weaker, currently supported claim: same recipe reproduces on a second dataset) vs "transfer" (a claim not currently supported — would require an actual IEMOCAP→MELD zero-shot/fine-tune experiment). **[NEEDS YOUR INPUT]**
- Decide whether fear should remain in the label set as-is, be re-analyzed with its variance foregrounded, or be revisited methodologically (oversampling, synthetic augmentation, or explicit "low-confidence class" framing). **[NEEDS YOUR INPUT]**
- Decide whether backbone/fusion-strategy ablations are in scope for this project or explicitly out of scope given time/compute constraints. **[NEEDS YOUR INPUT]**
- Decide how to correct the README (6-class → 5-class, add the MELD/agentic_ai description, clarify the dual-directory-tree structure) — I have not made this edit; you should review and approve the correction before it's written, given your "no changes without approval" instruction this session.

### Technical checks we can perform together (low cost, no training)
- Grep the whole `src/` and `agentic_ai/` tree for remaining "6-class"/"6 classes" stale references beyond `config.py` and confirm each one is just a comment/docstring issue, not a logic bug.
- Re-derive the overall (non-class-conditional) MELD ECE before/after temperature scaling numbers from the saved probability CSVs, since I did not extract exact figures for §7c.
- Confirm exact seeding call sites inside `train.py` if bit-for-bit reproducibility matters for your write-up.

### Experiments that may need to be run (flagging only — not doing this without your go-ahead)
- Build `knn_fold{2,3,4,5}.npz` and re-run the full agentic evaluation suite for those folds (addresses Critical gap #1).
- Run `analyse_results.py` correctly against `src/outputs/` to get IEMOCAP McNemar/Wilcoxon significance results (addresses Critical gap #2) — this is a fast, non-training script, just needs to be pointed at the right directory.
- If you decide fear needs a different treatment, that would require a retraining run (expensive) — do not schedule this without deciding the exact approach first.

### Writing that must be corrected or updated
- README.md: class count (6→5), disgust status (kept→dropped), missing description of MELD and `agentic_ai/`, missing description of the dual-directory-tree layout.
- `requirements.txt`: add `scipy`, `statsmodels`; promote `matplotlib`/`seaborn` from optional to required.
- Any existing thesis/paper draft (not found in this repo, so not audited here) should be checked against every number in §7 of this guide directly, not against memory of earlier conversations.

### Questions requiring confirmation from your supervisor
- Whether the MELD experiment is expected to demonstrate transfer or merely cross-corpus reproducibility (this changes what experiment, if any, still needs to be run).
- Whether fold-1-only agentic-layer results are acceptable for your current submission deadline, or whether folds 2-5 must be completed first.
- Whether a baseline outside this project's own three models is expected/required.
- Whether the fear class's small-sample issue needs a methodological response (oversampling etc.) or can be presented as an acknowledged limitation.

---

## 14. One-page project briefing (for sharing with another AI assistant or collaborator)

**Research problem:** Build and validate a multimodal (text+audio) speech emotion recognition pipeline as a prototype for future Cockpit Voice Recorder (CVR) stress/emotion analysis. No real aviation/CVR data is available yet (pending GCAA access) — all current work uses public benchmark datasets as stand-ins.

**Research questions:** (1) Can LoRA-fine-tuned multimodal fusion (RoBERTa-base + WavLM-base-plus) outperform single-modality models under proper speaker-independent cross-validation? (2) Does this hold across a second, harder/noisier dataset? (3) Can retrieval-based evidence (kNN over training embeddings) support a rule-based review-alert system with a controlled false-alarm rate?

**Datasets and classes:** IEMOCAP (primary, 5 sessions, 5,571 utterances, 5-fold LOSO — session/speaker-independent) and MELD (secondary, native train/dev/test split, TV dialogue). 5 emotion classes: anger, fear, joy, neutral, sadness. Disgust dropped (2 samples in IEMOCAP); happy+excited merged into joy; frustration/surprise/other dropped mainly for cross-corpus label-space compatibility. **Note: README currently incorrectly describes this as a 6-class system including disgust — that is stale/wrong; the running code is 5-class, disgust fully dropped.**

**Experimental protocol:** IEMOCAP — 5-fold Leave-One-Session-Out, both session- and speaker-independent (verified: each session's 2 speakers appear in no other session). MELD — native train/dev/test, trained as an independent parallel model (not a transfer/zero-shot test of the IEMOCAP model — this is an important framing distinction, not yet resolved with the supervisor). Class imbalance handled via inverse-frequency loss weighting (train-split-only, capped at 5.0) + label smoothing (0.1); no oversampling attempted.

**Models/modalities:** Text-only (RoBERTaLoRA), Audio-only (WavLMLoRA), Fusion (CrossModalAttentionFusion: cross-attention, text=Query/audio=K,V, then concatenated pooled vectors). LoRA rank=8, alpha=16, dropout=0.05.

**Current verified results (test sets):**
- IEMOCAP 5-fold LOSO (mean±std macro F1): text 0.629±0.052, audio 0.526±0.021, **fusion 0.694±0.054** (fusion best; significance not yet formally tested).
- MELD native split: text 0.500, audio 0.342, fusion 0.499 (fusion does *not* beat text on MELD — a real, unresolved cross-dataset discrepancy).
- Fear class: high mean F1 (0.485) but very high variance (std 0.278) due to ~40 total samples — never report without its std.
- Agentic layer (retrieval + rule-based alerting), **fold 1 only**: retrieval agreement 0.84 when correct vs 0.20 when wrong; alert rule (t=0.6) precision 0.45/recall 0.28; extended stage-2 rule (t=0.8, this session) precision 0.42/recall 0.34.

**Completed work:** full IEMOCAP LOSO training (all 3 models × 5 folds), full MELD native-split training (all 3 models), MELD calibration/ECE/significance analysis, the full agentic RAG+alert+orchestrator prototype (fold 1 only), HITL pilot utterance selection, case-study utterance generation.

**Work in progress / not started:** agentic-layer evaluation on folds 2-5 (never run — indexes don't exist), IEMOCAP significance testing (code exists, output never generated/saved due to a relative-path mismatch), any true IEMOCAP→MELD transfer experiment, README/requirements.txt corrections.

**Known issues:** dual directory trees with identical names (`{src/}outputs`, `{src/}checkpoints`, `{src/}logs`, `{src/}data_splits`) caused by relative-path config resolution depending on working directory at run time — IEMOCAP results live under `src/`, MELD results live at repo root; README is stale (says 6-class/keeps disgust — actual code is 5-class/drops disgust); `requirements.txt` missing `scipy`/`statsmodels`, mislabels `matplotlib`/`seaborn` as optional.

**Key file paths:** config `src/config.py`; IEMOCAP metadata `src/data_splits/iemocap_metadata.csv`; IEMOCAP results `src/outputs/{text,audio,fusion}/`; MELD results `outputs/{text,audio,fusion}/` (root); agentic layer `agentic_ai/` (entry points: `agent.py`, `alert_agent.py`, `orchestrator.py`); agentic results `agentic_ai/outputs/` (all fold1-only).

**Next decisions needed:** MELD framing (cross-corpus vs transfer claim), fear-class methodological stance, scope of backbone/fusion ablations, whether folds 2-5 of the agentic layer must be completed before the next deadline.
