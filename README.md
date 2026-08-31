# CVR Multimodal Emotion Recognition — Research Handover

## 1. Project overview

This project builds a multimodal (text + audio) speech emotion recognition pipeline intended
as groundwork for future Cockpit Voice Recorder (CVR) analysis. Because real CVR audio is not
available, all work uses two public benchmark datasets: **IEMOCAP** (primary) and **MELD**
(secondary). Text is modelled with LoRA-adapted RoBERTa, audio with LoRA-adapted WavLM, and
the two are combined with a cross-modal attention fusion model. A small "agentic" layer sits
on top of the classifier and adds k-NN retrieval evidence plus a rule-based review alert.
**No real CVR or aviation data is used anywhere in this repository.**

## 2. Current status

| Component | Status |
|---|---|
| IEMOCAP Text | Complete – 5-fold LOSO |
| IEMOCAP Audio | Complete – 5-fold LOSO |
| IEMOCAP Fusion | Complete – 5-fold LOSO |
| MELD Text / Audio / Fusion | Complete |
| Agentic AI layer | Implemented and evaluated on Fold 1 only |
| IEMOCAP significance testing | Not completed |
| Real CVR data | Not available |
| Manuscript | In progress outside this repository |

The agentic layer (k-NN index, EmotionAgent, AlertAgent, orchestrator, alert evaluation) has
been run for IEMOCAP Fold 1. Folds 2–5 have trained classifier checkpoints but have not been
evaluated through the agentic layer.

## 3. Quick start / What to run

Working-directory matters: `src/config.py` uses paths relative to where you run the script,
so IEMOCAP steps run from `iemocap_emotion/src/` and MELD steps run from `iemocap_emotion/`.

### Run directly
`src/data_preprocessing.py`, `src/train.py`, `src/analyse_results.py`,
`src/meld_preprocessing.py`, `src/temperature_scaling.py`, `src/plot_calibration.py`,
`src/plot_confusion_matrices_meld.py`, and the `agentic_ai/` scripts listed below.

### Imported only — do NOT run directly
`src/config.py`, `src/utils.py`, `src/dataset.py`, `src/metrics.py`,
`src/models_text.py`, `src/models_audio.py`, `src/models_fusion.py`,
`agentic_ai/agent.py`, `agentic_ai/rag.py`, `agentic_ai/alert_agent.py`,
`agentic_ai/orchestrator.py`, `agentic_ai/result.py`.

### Steps

**1. Setup**
```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r iemocap_emotion/requirements.txt
pip install scipy statsmodels matplotlib seaborn       # not in requirements.txt but required
```

**2. Configure paths** — edit `iemocap_emotion/src/config.py`:
`iemocap_root` (line 45) and `meld_root` (line 80) to your local dataset locations.

**3. IEMOCAP preprocessing** (from `iemocap_emotion/src/`)
```bash
python data_preprocessing.py
# -> src/data_splits/iemocap_metadata.csv
```

**4. IEMOCAP training** (from `iemocap_emotion/src/`) — each command runs all 5 LOSO folds
```bash
python train.py --model text
python train.py --model audio
python train.py --model fusion
# checkpoints -> src/checkpoints/<model>/ , metrics -> src/outputs/<model>/ , logs -> src/logs/<model>/
```

**5. IEMOCAP analysis** (from `iemocap_emotion/src/`)
```bash
python analyse_results.py --dataset iemocap
# -> src/outputs/analysis/results_analysis.txt   (per-class F1, confusion, McNemar + Wilcoxon)
```

**6. MELD preprocessing** (from `iemocap_emotion/`) — needs ffmpeg on PATH
```bash
python src/meld_preprocessing.py --extract_audio
python src/meld_preprocessing.py --build_metadata
```

**7. MELD training and analysis** (from `iemocap_emotion/`)
```bash
python src/train.py --model text  --dataset meld
python src/train.py --model audio --dataset meld
python src/train.py --model fusion --dataset meld
python src/temperature_scaling.py
python src/plot_calibration.py
python src/analyse_results.py --dataset meld
python src/plot_confusion_matrices_meld.py
```

**8. Agentic layer** (from `iemocap_emotion/`) — Fold 1
```bash
python agentic_ai/build_index.py --fold 1
python agentic_ai/evaluate_agent.py --fold 1 --split test
python agentic_ai/evaluate_agent.py --fold 1 --split val
python agentic_ai/evaluate_alert.py --fold 1
python agentic_ai/eval_orchestrator.py      # no arguments
python agentic_ai/risk_weighted_f1.py       # no arguments
python agentic_ai/plot_roc_pr.py            # no arguments
```

## 4. Existing results

| What | Location |
|---|---|
| IEMOCAP results (per-fold + summary) | `iemocap_emotion/src/outputs/{text,audio,fusion}/` |
| IEMOCAP training logs | `iemocap_emotion/src/logs/{text,audio,fusion}/` |
| MELD results | `iemocap_emotion/outputs/{text,audio,fusion}/` |
| MELD analysis / calibration | `iemocap_emotion/outputs/analysis/` |
| Agentic Fold-1 results | `iemocap_emotion/agentic_ai/outputs/` |

Metric tables are not reproduced here. Verified numbers are in the result files above and,
narrated, in `iemocap_emotion/PROJECT_REINTEGRATION_GUIDE.md`.

## 5. Important notes

- The final emotion set is **5 classes** (anger, fear, joy, neutral, sadness), not 6. A stale
  `# 6` comment in `src/config.py` and older text may still say otherwise.
- Agentic-layer results are **Fold 1 only**.
- IEMOCAP significance analysis has **not been run/saved**; `outputs/analysis/` contains MELD
  results only.
- Datasets, model checkpoints, caches, and transcript/metadata CSVs are **excluded from Git**.
- Local dataset paths in `src/config.py` (`iemocap_root`, `meld_root`) **must be updated**
  before running anything.
- MELD models were trained independently on MELD; this is **not** an IEMOCAP→MELD transfer
  experiment.

## 6. Datasets

- **IEMOCAP** — obtain from the original source (USC SAIL, <https://sail.usc.edu/iemocap/>);
  not redistributed here.
- **MELD** — download separately (<https://affective-meld.github.io/>).
- Derived metadata and transcript CSVs are excluded from Git. Regenerate them locally:
  `python src/data_preprocessing.py` (IEMOCAP) and
  `python src/meld_preprocessing.py --extract_audio --build_metadata` (MELD).

## 7. Detailed project documentation

For detailed experiment history, verified results, methodological decisions, and known
research gaps, see `iemocap_emotion/PROJECT_REINTEGRATION_GUIDE.md`.
