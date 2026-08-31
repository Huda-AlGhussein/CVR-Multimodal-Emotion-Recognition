# CVR Multimodal Emotion Recognition — Research Handover

> **Status:** research prototype, handed over on 2026-08-31 (author's final working day).
> **Repository:** private. Contains source code, configs, and small final results only —
> **no raw datasets, no model checkpoints, no pretrained-model caches** (see `.gitignore`).
>
> **Read this file together with [`iemocap_emotion/PROJECT_REINTEGRATION_GUIDE.md`](iemocap_emotion/PROJECT_REINTEGRATION_GUIDE.md)**,
> a 600-line evidence-tagged audit of every decision, result, and gap. Where this README
> summarises, that guide gives the detail and the file-level evidence.

---

## 1. Project objective

Build and validate a **multimodal (text + audio) speech-emotion-recognition pipeline** as a
prototype for future **Cockpit Voice Recorder (CVR)** analysis — automatically flagging
emotionally salient utterances (stress, anger, fear) in cockpit audio to help accident
investigators triage long recordings, and eventually to support real-time crew-state monitoring.

No real CVR/aviation audio is available to the project (pending GCAA — aviation authority —
data access). The strategy is therefore: **prove the full pipeline on public benchmark
speech-emotion datasets (IEMOCAP primary, MELD secondary), quantify its limitations, and keep
the architecture ready to be re-pointed at a real CVR corpus later.** Placeholder fields
(`flight_phase`, `utterance_timecode`) are already wired through the agentic output schema.

**Research questions (as implied by the code — no written thesis statement exists in the repo):**
1. Can LoRA-fine-tuned multimodal fusion (RoBERTa-base + WavLM-base-plus) beat single-modality
   models under proper speaker-independent cross-validation on IEMOCAP?
2. Does that advantage hold on a second, noisier dataset (MELD)?
3. Can retrieval-based evidence (kNN over training embeddings) drive a rule-based
   review-alert system with a controlled false-alarm rate?

---

## 2. Current status of the CVR research

| Component | Status |
|---|---|
| IEMOCAP 5-fold LOSO training — text, audio, fusion | **Complete** (all 15 fold×model runs; checkpoints, logs, metrics on disk) |
| MELD native-split training — text, audio, fusion | **Complete** |
| MELD calibration / ECE / temperature scaling | **Complete** |
| MELD significance testing (McNemar) | **Complete** |
| **IEMOCAP significance testing** | **NOT DONE** — script supports it, output was never generated/saved (relative-path mismatch). See §11. |
| Agentic layer (classifier + kNN retrieval + AlertAgent + orchestrator) | **Built and working, evaluated on IEMOCAP fold 1 ONLY** |
| **Agentic layer on folds 2–5** | **NOT STARTED** — kNN indices for folds 2–5 do not exist |
| Risk-weighted F1 metric | Prototype, **fold 1 only**, weights marked "provisional / under review" |
| HITL pilot / case-study utterance selection | Utterances **curated only** — no human study has been run |
| IEMOCAP↔MELD transfer / zero-shot experiment | **NOT STARTED / not planned in code** |
| Manuscript / thesis / paper draft | **None in this repo** |
| Real CVR / aviation data | **None anywhere** — every result is acted (IEMOCAP) or TV-dialogue (MELD) speech |

**One-line honest summary:** the *classifier* is fully validated with 5-fold LOSO; the
*agentic layer* on top of it is a fold-1-only prototype.

---

## 3. Datasets — how to obtain them (do NOT commit datasets)

### IEMOCAP (primary)
- **Source:** University of Southern California SAIL lab — <https://sail.usc.edu/iemocap/>
- **Access:** requires a signed academic release form; not redistributable. Request via the
  form on that page.
- ~12 h of acted/elicited dyadic dialogue, **5 sessions, 10 unique speakers** (2 per session,
  never repeated across sessions — this is what makes LOSO speaker-independent).
- Download gives `IEMOCAP_full_release/` with `Session{1..5}/{dialog,sentences}/...`.

### MELD (secondary / cross-corpus)
- **Source:** <https://affective-meld.github.io/> / <https://github.com/declare-lab/MELD>
- **Access:** public download (`MELD.Raw.tar.gz`, ~10 GB). Licence is research-only; audio is
  extracted from *Friends* TV episodes.
- Native `train / dev / test` split (dialogue-based, **not** speaker-independent — recurring
  actors appear across splits; a known MELD property).
- `meld_preprocessing.py` extracts the inner tarballs and runs **ffmpeg** to pull 16 kHz
  mono WAVs from the MP4s — **ffmpeg must be on PATH.**

### Emotion classes — FINAL retained set = **5 classes**: `anger, fear, joy, neutral, sadness`
Defined in `src/config.py:150-152`. Label handling:

| Raw IEMOCAP | → | Final | Note |
|---|---|---|---|
| ang | → | anger | kept |
| sad | → | sadness | kept |
| fea | → | fear | kept — **only ~40 samples total**, high-variance, always report with std |
| neu | → | neutral | kept |
| hap + exc | → | joy | **merged** |
| dis | → | *dropped* | only 2 samples |
| fru, sur, oth, xxx | → | *dropped* | cross-corpus incompatibility / annotation artefacts |

MELD keeps the same 5 names, drops its `disgust` and `surprise`.

> ⚠️ **Do not describe this as "6-class" work.** The old inner README and a stale
> `# 6` comment at `src/config.py:318` still say 6 classes / keep disgust — that is
> **wrong**; the running code is 5-class and drops disgust entirely. See §11.

---

## 4. Expected local dataset folder structure

Clone the repo, then place datasets so paths resolve as below (repo root = this folder):

```
CVR_multimodal/
└── iemocap_emotion/
    ├── data_splits/
    │   └── data/
    │       └── IEMOCAP_full_release/        ← place IEMOCAP here (git-ignored)
    │           ├── Session1/ … Session5/
    ├── src/
    │   └── MELD.Raw/                        ← extract MELD.Raw.tar.gz here (git-ignored)
    │       ├── train.tar.gz  dev.tar.gz  test.tar.gz
    │       ├── train_sent_emo.csv  dev_sent_emo.csv  test_sent_emo.csv
    │       └── train_splits/  dev_splits_complete/  output_repeated_splits_test/
    ├── data_splits/meld_audio/{train,dev,test}/*.wav   ← generated by meld_preprocessing.py (git-ignored)
    └── src/data_splits/audio_cache/*.pt               ← generated audio cache (git-ignored)
```

Then edit the two **hard-coded absolute paths** in `src/config.py`:
- `iemocap_root` (line 45) — currently `C:\Users\OpenU\...\IEMOCAP_full_release`
- `meld_root` (line 80) — currently `C:\Users\OpenU\...\src\MELD.Raw`

> ⚠️ **Dual-directory-tree quirk (important).** `src/config.py` uses *relative* output paths
> (`./outputs`, `./checkpoints`, `./logs`, `./data_splits`). They resolve against the current
> working directory. IEMOCAP work was run **from inside `src/`** → artifacts under
> `iemocap_emotion/src/{outputs,checkpoints,logs,data_splits}/`. MELD work was run **from
> `iemocap_emotion/`** → artifacts under `iemocap_emotion/{outputs,checkpoints,logs,data_splits}/`.
> Same folder names, different experiments. Internalise this before touching any path.

---

## 5. Preprocessing pipeline

| Step | Script / function | Output |
|---|---|---|
| IEMOCAP metadata + label mapping + LOSO logic | `src/data_preprocessing.py::build_metadata`, `get_split_dfs` | `src/data_splits/iemocap_metadata.csv` — **NOT committed** (5,571 rows: id, session, raw_label, emotion, text, audio_path, audio_exists); regenerate — see below |
| IEMOCAP audio load/preprocess | `src/data_preprocessing.py::load_and_preprocess_audio` — mono, resample 16 kHz, RMS-normalise −20 dBFS, pad/truncate to 6.0 s (96,000 samples) | `src/data_splits/audio_cache/{utt}_{maxlen}.pt` — not committed |
| Class weights (per fold, train split only, capped 5.0) | `src/data_preprocessing.py::compute_class_weights` | in-memory |
| MELD extract + ffmpeg + metadata | `src/meld_preprocessing.py` | `data_splits/meld/{train,dev,test}.csv` — **NOT committed** (regenerate — see below); `data_splits/meld_audio/**/*.wav` — not committed |
| Text tokenisation | `src/dataset.py::TextDataset` (RoBERTa tokeniser, `max_text_tokens=128`) | in-memory |

`src/data_preprocessing.py::load_metadata()` validates label indices against the current
config and **raises** if stale — regenerate the metadata CSV after any label-map change.

### Regenerating the excluded metadata / split CSVs

These files embed **licence-restricted IEMOCAP / MELD transcript text**, so they are **not in
Git** (even though the repo is private). Recreate them locally from the original corpora with
the existing preprocessing scripts — no code changes needed:

| Excluded file | Regenerate with (run from `iemocap_emotion/src/`) | Needs |
|---|---|---|
| `src/data_splits/iemocap_metadata.csv` | `python data_preprocessing.py` | `IEMOCAP_full_release/` at `cfg.iemocap_root` |
| `data_splits/meld/meld_{train,dev,test}.csv` | `python meld_preprocessing.py` (run from `iemocap_emotion/`) | `MELD.Raw` at `cfg.meld_root`, `ffmpeg` on PATH |
| `agentic_ai/outputs/hitl_pilot_utterances.csv`, `case_study_utterances.csv` | not needed for any completed experiment — unfinished study artefacts; the selection scripts live in `agentic_ai/` (`evaluate_agent.py` + manual curation) if they must be rebuilt | fold-1 agent eval + IEMOCAP audio |

Every **committed** results CSV (`src/outputs/**`, `outputs/**`, `agentic_ai/outputs/fold1_*eval*.csv`)
contains only utterance IDs, labels, predictions and probabilities — **no transcript text and
no local paths** — so those stay in Git.

---

## 6. Models

| Modality | Class | File | Backbone | Adaptation |
|---|---|---|---|---|
| Text | `RoBERTaLoRA` | `src/models_text.py` | `roberta-base` | LoRA r=8, α=16, dropout=0.05, targets `["query","value"]` |
| Audio | `WavLMLoRA` | `src/models_audio.py` | `microsoft/wavlm-base-plus` | LoRA on `["q_proj","v_proj"]`; CNN feature extractor **frozen** |
| Fusion | `CrossModalAttentionFusion` | `src/models_fusion.py` | both of the above | cross-attention (text = Query, audio = Key/Value, 4 heads) → concat `[z_text; z_cross; z_audio]` (768-d) → classifier head; `embed()` exposes the 768-d pre-classifier vector used by the agentic kNN index |

Pretrained backbones download from Hugging Face on first run and are cached to
`~/.cache/huggingface` (git-ignored — never commit).

---

## 7. IEMOCAP LOSO evaluation setup

**Leave-One-Session-Out, 5 folds.** For fold *k*: **test = Session *k***, **val = Session
(*k* mod 5) + 1**, **train = the remaining 3 sessions**. Session- *and* speaker-independent
(each session's 2 speakers appear in no other session). Implemented in
`src/data_preprocessing.py::get_split_dfs`; fold selection via `--fold`.

- Checkpoint selection: best **validation Macro F1**, early-stop patience 5, best checkpoint
  reloaded before the single held-out test-session evaluation. No test-set leakage.
- Primary metric: **Macro F1** (class imbalance: neutral in the thousands vs fear ≈ 40).
  Also weighted F1, weighted/unweighted accuracy, per-class F1, confusion matrix.
- 5-fold aggregation (`src/metrics.py::aggregate_fold_results`) only runs if **all 5 folds
  are executed in one `train.py` invocation**.

## MELD setup

Native `train/dev/test`; `dev` = validation, `test` held out. Separate models trained
**from scratch on MELD** — this is *not* a transfer test of the IEMOCAP model (see §11 #7).
Calibration/ECE (`src/plot_calibration.py`, `src/temperature_scaling.py`), significance
(`src/analyse_results.py`), confusion plots (`src/plot_confusion_matrices_meld.py` — **uses
hardcoded numbers, will not auto-update if MELD is retrained**).

---

## 8. Current completed experiments & results

### 8a. IEMOCAP — 5-fold LOSO, TEST set (mean ± sample std across folds)
Source: `src/outputs/{text,audio,fusion}/all_folds_summary.json`

| Model | Macro F1 | Weighted F1 | Weighted Acc | Unweighted Acc |
|---|---|---|---|---|
| Text | 0.629 ± 0.052 | 0.668 ± 0.026 | 66.6 % ± 2.6 % | 69.1 % ± 6.1 % |
| Audio | 0.526 ± 0.021 | 0.636 ± 0.014 | 64.1 % ± 1.7 % | 55.4 % ± 4.9 % |
| **Fusion** | **0.694 ± 0.054** | 0.739 ± 0.012 | 74.0 % ± 1.1 % | 74.6 % ± 6.2 % |

Fusion per-class F1: anger 0.79 ± 0.04, **fear 0.47 ± 0.25**, joy 0.71 ± 0.02,
neutral 0.61 ± 0.06, sadness 0.64 ± 0.05. Fusion beats text by **+0.064 Macro F1** —
**numerically**; not yet significance-tested (§11 #2).

### 8b. MELD — native split, TEST set
Source: `outputs/{text,audio,fusion}/meld_test_metrics.json`, `outputs/analysis/results_analysis.txt`

| Model | Macro F1 | Weighted F1 | Weighted Acc |
|---|---|---|---|
| Text | 0.500 | 0.675 | 66.9 % |
| Audio | 0.342 | 0.475 | 44.8 % |
| Fusion | 0.499 | 0.660 | 64.3 % |

**Fusion does NOT beat text on MELD** (−0.0007). McNemar text-vs-fusion χ² = 10.04,
p = 0.0015 (fusion significantly *different* but numerically *lower*); both ≫ audio.

### 8c. MELD calibration (class-conditional ECE, 10 bins)
`outputs/analysis/class_conditional_ece.txt`. Headline: **neutral is the worst-calibrated
class for every model** (ECE 0.25–0.32). High-risk (anger/fear) better calibrated in
aggregate than low-risk, but `fear` alone (~0.15) is worse than `anger` (~0.03).
Before/after temperature-scaling overall ECE: in
`outputs/analysis/calibration_plots_after_scaling.png` — exact numeric values were not
extracted to text; re-derive from the saved probability CSVs if needed for a table.

### 8d. Agentic layer — IEMOCAP **fold 1 test set only** (n = 1097)
Source: `agentic_ai/outputs/fold1_agent_summary.json`, `fold1_test_agent_eval*.csv`,
`fold1_risk_weighted_f1.txt`, `roc_pr_curves.png`

- Classification via the agent wrapper: Macro F1 **0.748** (matches the raw fusion checkpoint
  0.748 — the ~0.0007 gap is AMP nondeterminism).
- Retrieval agreement: overall 0.67; **when classifier correct 0.84**, **when wrong 0.20**
  (~4× gap — the empirical basis for the alert rule).
- Retrieval "rescue rate": 149 / 301 wrong predictions (49.5 %) had the true label somewhere
  in their top-5 neighbours.
- Primary alert rule (`graded_agreement`, t = 0.6): 189 alerts, 85 TP / 104 FP →
  precision 0.45, recall over errors 0.28, F1 0.35.
- Stage-2 rule (`neutral_low_agreement`, t = 0.8, added very recently): 248 alerts,
  103 TP / 145 FP → precision 0.42, recall 0.34, F1 0.38.
- Risk-weighted Macro F1 (anger/fear w = 3, joy/sadness w = 2, neutral w = 1 — **provisional**):
  standard 0.748 → risk 0.647.
- The alert threshold t = 0.6 was **deliberately chosen over the F1-optimal t = 0.9**
  (which gave 37.6 % test FPR) — documented in `alert_agent.py` as "alert fatigue
  unacceptable in a safety-critical context". Even at t = 0.6, FPR ≈ 17.6 % — explicitly
  **not sufficient as a standalone safety gate**.

**Everything in §8d is fold-1 only. No folds 2–5 equivalents exist.**

---

## 9. Agentic-AI architecture — where each piece lives

Pipeline: `EmotionAgent` (classify + retrieve) → `AlertAgent` (rule-based flag) →
`SupervisorAgent` (orchestration) → alert evaluation scripts.

| Function | File | Notes |
|---|---|---|
| **Classifier prediction** | `agentic_ai/agent.py::EmotionAgent` (loads fold-1 fusion checkpoint, runs forward pass) | constructor default `fold=1`; folds 2–5 have checkpoints but no kNN index |
| **kNN / retrieval** | `agentic_ai/rag.py::KNNRetriever` (brute-force cosine over training embeddings); index build in `agentic_ai/build_index.py` → `agentic_ai/cache/knn_fold{k}.npz` | **only `knn_fold1.npz` exists** (git-ignored — rebuild with `build_index.py`) |
| **EmotionAgent full-split eval** | `agentic_ai/evaluate_agent.py` (`--fold`, `--split`; asserts index ∩ eval-split = ∅) | only ever run for fold 1 |
| **AlertAgent** (alert logic) | `agentic_ai/alert_agent.py::AlertAgent` — `graded_agreement` primary rule + `neutral_low_agreement` stage-2 | thresholds empirically justified on fold-1 data only; stage-2 t = 0.8 provisional |
| **Orchestration** | `agentic_ai/orchestrator.py::SupervisorAgent` (+ `MODEL_VERSION` constant, hardcodes `"fold1"`); `agentic_ai/eval_orchestrator.py` (batch eval, `compute_pred_agreement`) | CVR placeholder fields `utterance_timecode` / `flight_phase` emitted as `None` |
| **Alert evaluation** | `agentic_ai/evaluate_alert.py` (val vs test contingency tables; labels val results "not reportable") | fold 1 only |
| **Risk-weighted F1** | `agentic_ai/risk_weighted_f1.py` → `fold1_risk_weighted_f1.txt` | **hardcoded to fold 1**, no CLI arg; weights "under review" |
| **ROC/PR for alert signal** | `agentic_ai/plot_roc_pr.py` → `roc_pr_curves.png` | **hardcoded to fold 1** |
| Typed result containers | `agentic_ai/result.py` (`RetrievedExample`, `AgentResult`) | — |
| Smoke tests (NOT unit tests — no asserts) | `agentic_ai/test_agent.py`, `agentic_ai/test_orchestrator.py` | 3 hardcoded example utterances |

**Completed agentic experiments:** EmotionAgent full eval (fold 1 val+test), primary alert
rule (fold 1), stage-2 alert rule (fold 1), risk-weighted F1 (fold 1), ROC/PR (fold 1),
extended orchestrator decision log with CVR placeholder schema (fold 1, 1097 rows),
HITL-pilot utterance selection (10 utts), case-study utterance generation (6 utts, live
pipeline run). The last two produced `hitl_pilot_utterances.csv` / `case_study_utterances.csv`
— **not committed** (transcript text + local paths, unfinished study artefacts).

**Unfinished / not started:** kNN indices + full agentic evaluation for **folds 2–5**;
any actual human-in-the-loop study; any real CVR-context data in the placeholder fields.

---

## 10. Execution order to reproduce the current experiments

All IEMOCAP commands run **from `iemocap_emotion/src/`**; MELD from **`iemocap_emotion/`**.

```bash
# 0. Environment
pip install -r iemocap_emotion/requirements.txt
#    ALSO: pip install scipy statsmodels matplotlib seaborn   (see §12 — missing from the file)

# 1. Edit iemocap_emotion/src/config.py: iemocap_root (L45), meld_root (L80)

# ---- IEMOCAP ----  (cd iemocap_emotion/src)
python data_preprocessing.py                 # -> src/data_splits/iemocap_metadata.csv
python train.py --model text                  # all 5 LOSO folds -> src/{checkpoints,outputs,logs}/text/
python train.py --model audio
python train.py --model fusion
python analyse_results.py                      # IEMOCAP significance  <-- NOT YET RUN; see §11 #2
                                               #   (ensure its OUTPUTS_ROOT resolves to ./src/outputs)

# ---- MELD ----  (cd iemocap_emotion)
python src/meld_preprocessing.py               # needs ffmpeg on PATH -> data_splits/meld_audio/**
python src/train.py --model text  --dataset meld
python src/train.py --model audio --dataset meld
python src/train.py --model fusion --dataset meld
python src/temperature_scaling.py              # -> outputs/{model}/meld_*_probs.csv
python src/plot_calibration.py                 # -> outputs/analysis/calibration_plots*.png
python src/analyse_results.py                  # -> outputs/analysis/results_analysis.txt (MELD)
python src/plot_confusion_matrices_meld.py     # -> outputs/analysis/confusion_matrices_meld.png

# ---- Agentic layer (currently fold 1 only) ----  (cd iemocap_emotion)
python agentic_ai/build_index.py --fold 1                       # -> agentic_ai/cache/knn_fold1.npz
python agentic_ai/evaluate_agent.py --fold 1 --split val
python agentic_ai/evaluate_agent.py --fold 1 --split test      # -> agentic_ai/outputs/fold1_*_agent_eval.csv
python agentic_ai/eval_orchestrator.py --fold 1                 # -> fold1_orchestrator_eval*.csv
python agentic_ai/evaluate_alert.py                             # contingency tables (fold 1)
python agentic_ai/risk_weighted_f1.py                           # -> fold1_risk_weighted_f1.txt
python agentic_ai/plot_roc_pr.py                                # -> roc_pr_curves.png
# To extend to folds 2-5: rerun build_index.py + evaluate_agent.py + eval_orchestrator.py
# with --fold {2..5}; risk_weighted_f1.py and plot_roc_pr.py need a small edit to accept --fold.
```

---

## 11. Known handover limitations & inconsistencies (found during this handover)

**Critical**
1. **Agentic / RAG / alert layer evaluated on IEMOCAP fold 1 only.** The classifier is
   5-fold-validated; every retrieval-agreement, alert-precision/recall and risk-weighted-F1
   number is single-fold. Fix path: build `knn_fold{2..5}.npz`, rerun the agentic suite.
2. **IEMOCAP significance testing never run/saved.** `src/analyse_results.py` supports it,
   but the only saved output (`outputs/analysis/results_analysis.txt`) is MELD-only (it was
   run where the relative `OUTPUTS_ROOT` resolved to the root `outputs/`). So
   "fusion **significantly** beats text on IEMOCAP" **cannot currently be claimed** — only
   "numerically higher". `outputs/analysis/` has **no IEMOCAP files at all**.
3. **Stale "6-class" documentation.** The previous inner README (now replaced with a stub)
   and `src/config.py:318` (`NUM_CLASSES = len(...)  # 6`) describe a 6-class system that
   keeps `disgust`. The running code is **5-class, disgust dropped**. Left in the research
   code deliberately (not fixing research files during handover) — but do not repeat the
   6-class claim to anyone. A quick `grep -rin "6 class\|six class\|disgust" src/` before any
   write-up is recommended; model-file docstrings may also carry stale 6-class text.

**Important**

4. **`requirements.txt` incomplete:** `scipy` and `statsmodels` are imported by
   `src/analyse_results.py` but absent; `matplotlib` / `seaborn` are listed as *optional*
   but are hard requirements of the calibration/plot scripts that were actually run. A fresh
   env built strictly from the file fails on the analysis scripts.
5. **No external baseline.** No majority-class baseline, no published-SOTA comparison —
   the three models only baseline each other.
6. **Hard-coded absolute local paths** (in **research files, not committed**, and in
   **committed code**):
   - Committed code: `src/config.py:45` & `:80` (`C:\Users\OpenU\...` for `iemocap_root` /
     `meld_root`) — must be edited on any new machine. `agentic_ai/test_*.py` use a computed
     `_PROJECT_ROOT` (portable).
   - Not committed (excluded for transcript-licence reasons anyway, but also carry absolute
     paths): `src/data_splits/iemocap_metadata.csv` `audio_path` column (all 5,571 rows),
     `agentic_ai/outputs/hitl_pilot_utterances.csv` `audio_path` column. Regeneration
     produces machine-local paths by design.
7. **"Cross-corpus" ≠ transfer.** What exists is two independently-trained models on two
   datasets, not one model evaluated across both. If a transfer claim is expected, that
   experiment does not exist. Confirm the intended framing with the supervisor.
8. **MELD is not speaker-independent** (recurring actors across splits) — weakens the MELD
   result as a "generalisation to unseen speakers" check versus IEMOCAP's clean LOSO.
9. **Stage-2 alert threshold (0.8)** is very recent and fold-1-only — provisional.
10. **Risk-weighted F1 weights** are marked "provisional / under review" in the code itself —
    not a settled metric.
11. **`plot_confusion_matrices_meld.py` uses hardcoded numeric arrays** (not read from a
    file) — currently correct, but will silently go stale if MELD is retrained.
12. **`src/plot_confusion_matrices_meld.py` / `temperature_scaling.py` / `plot_calibration.py`
    hardcode root-relative paths** — run them from `iemocap_emotion/`, not `src/`.

**Minor**

13. `models_fusion.py::_subsample_mask` is a Python loop vs the vectorised version in
    `models_audio.py` — perf inconsistency, not a bug.
14. `test_agent.py` / `test_orchestrator.py` are smoke tests (no asserts) despite the
    `test_` prefix.
15. No documented bit-for-bit seed verification of `train.py` (seed infra exists,
    `cfg.seed = 42`).
16. ~21 GB of raw dataset archives + extracted corpora + 11 GB of checkpoints remain in the
    working tree — all git-ignored, none committed, but note them if the machine is reused.

---

## 12. Environment / dependencies

- **Python 3.10** (bytecode caches are `cpython-310`). 3.9+ expected to work.
- GPU: trained on an NVIDIA A100-class GPU. CPU works but audio models are ~30 min/epoch.
- **ffmpeg** required on PATH for `meld_preprocessing.py`.
- `pip install -r iemocap_emotion/requirements.txt` — then additionally
  **`pip install scipy statsmodels matplotlib seaborn`** (see §11 #4).
- Core stack: `torch>=2.1`, `torchaudio>=2.1`, `transformers>=4.38`, `peft>=0.10`,
  `accelerate>=0.27`, `pandas>=2.0`, `numpy>=1.26`, `scikit-learn>=1.4`.
- No environment lockfile (no `conda env` / `poetry.lock` / pinned `pip freeze`) — a
  `pip freeze > environment-lock.txt` on the original machine would be a valuable addition.

---

## 13. Important output / result locations (committed)

| Path | Contents |
|---|---|
| `iemocap_emotion/src/outputs/{text,audio,fusion}/` | IEMOCAP per-fold `*_test_metrics.json`, `*_test_predictions.csv`, `*_confusion_matrix.npy`, `all_folds_summary.json` |
| `iemocap_emotion/src/logs/{text,audio,fusion}/fold{1..5}_train.log` | IEMOCAP per-fold training logs (all kept) |
| `iemocap_emotion/outputs/{text,audio,fusion}/` | MELD `meld_test_metrics.json`, prediction + probability CSVs, confusion npy |
| `iemocap_emotion/outputs/analysis/` | **MELD only:** `results_analysis.txt` (per-class F1, confusion, McNemar), `class_conditional_ece.txt`, calibration + confusion + ECE PNGs |
| `iemocap_emotion/logs/{text,audio,fusion}/meld_train.log` | MELD training logs |
| `iemocap_emotion/agentic_ai/outputs/` | **fold 1 only:** agent summary JSON, agent/orchestrator eval CSVs (IDs/labels/probs only), `fold1_risk_weighted_f1.txt`, `roc_pr_curves.png` |

**Git-ignored (regenerate locally — see §5 for the metadata/split CSVs):**
`src/data_splits/iemocap_metadata.csv`, `data_splits/meld/*.csv`,
`agentic_ai/outputs/{hitl_pilot,case_study}_utterances.csv` (transcript text / local paths);
`**/checkpoints/*.pt`, `src/data_splits/audio_cache/`, `data_splits/meld_audio/`,
`agentic_ai/cache/knn_fold1.npz`, all `*.tar.gz`, raw corpora, `__pycache__/`, HF cache.

---

## 14. Manuscript / research status

- **No manuscript, thesis chapter, paper draft, or slide deck is in this repo.**
- The most complete written artefact is
  [`iemocap_emotion/PROJECT_REINTEGRATION_GUIDE.md`](iemocap_emotion/PROJECT_REINTEGRATION_GUIDE.md)
  — treat its §7 (verified results) and §14 (one-page briefing) as the current source of
  truth for numbers.
- Any external draft must be checked against §8 here / §7 of the guide directly — not
  against memory.

---

## 15. Recommended next steps for whoever continues

**Do first (cheap, no training):**
1. Run `src/analyse_results.py` pointed at `src/outputs/` to produce the **IEMOCAP**
   McNemar/Wilcoxon significance results (closes Critical #2). Fast, no training.
2. `pip freeze > environment-lock.txt` on a working machine; fix `requirements.txt` (#4).
3. `grep -rin "disgust\|6 class\|six class"` across `src/` and `agentic_ai/`; confirm each
   hit is a comment/docstring, not logic (Critical #3).
4. Replace absolute paths in `config.py` and the two agentic CSVs with relative / `pathlib`
   paths (#6).

**Do next (cheap compute):**
5. Build `knn_fold{2,3,4,5}.npz` and rerun `evaluate_agent.py` / `eval_orchestrator.py` /
   `evaluate_alert.py` for folds 2–5; add a `--fold` arg to `risk_weighted_f1.py` and
   `plot_roc_pr.py` first (closes Critical #1 — the biggest single gap).

**Decisions needed from the supervisor / project owner:**
6. MELD framing: cross-corpus reproducibility (supported) vs transfer (not done).
7. Whether the `fear` class (~40 samples) stays as-is, is reported with variance
   foregrounded, or gets a methodological response (oversampling / augmentation).
8. Whether backbone / fusion-strategy ablations and an external baseline are in scope.
9. Whether fold-1-only agentic results are acceptable for the next deadline.

**Larger (needs training budget):**
10. Investigate *why* fusion helps on IEMOCAP but not MELD (audio-quality ablation).
11. Any real CVR corpus integration once GCAA access is granted — the placeholder schema
    (`flight_phase`, `utterance_timecode`) is the intended entry point.
