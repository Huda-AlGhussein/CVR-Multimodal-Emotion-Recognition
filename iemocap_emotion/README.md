# iemocap_emotion — see the handover README

**This file is superseded.** The authoritative documentation is the repository-root
[`../README.md`](../README.md) (research handover) and
[`PROJECT_REINTEGRATION_GUIDE.md`](PROJECT_REINTEGRATION_GUIDE.md) (full audit).

## Corrections to the previous version of this file (kept here so the record is clear)

The earlier `README.md` in this folder described a **6-class** system that **kept `disgust`**.
That is **wrong**. The running code is **5-class** — `anger, fear, joy, neutral, sadness`
(`src/config.py:150-152`) — and `disgust` is **dropped entirely** (`src/config.py`, only
2 samples in IEMOCAP). `happy` + `excited` are **merged into `joy`**.

Other stale claims in the old file:
- It did not mention MELD or the `agentic_ai/` layer at all.
- Its folder diagram implied a single output tree; there are actually **two** parallel
  trees (`src/{outputs,checkpoints,logs}` for IEMOCAP, `./{outputs,checkpoints,logs}` for
  MELD) because `src/config.py` uses working-directory-relative paths. See root README §4.
- `requirements.txt` is missing `scipy` / `statsmodels` and mislabels
  `matplotlib` / `seaborn` as optional. See root README §11 / §12.

The research code and comments were intentionally **not** edited during handover; the stale
`# 6` comment at `src/config.py:318` and any 6-class docstrings remain and are logged as
known limitations in the root README §11.

## Quick pointers

- Setup, dataset access, folder layout, execution order → root [`../README.md`](../README.md)
- Every decision / result / gap with file-level evidence → [`PROJECT_REINTEGRATION_GUIDE.md`](PROJECT_REINTEGRATION_GUIDE.md)
- Config / paths / hyperparameters → `src/config.py`
- Label map + LOSO split logic → `src/data_preprocessing.py`
- Models → `src/models_{text,audio,fusion}.py`
- Training entry point → `src/train.py --model {text|audio|fusion} [--fold k] [--dataset meld]`
- Agentic layer → `agentic_ai/` (entry points `agent.py`, `alert_agent.py`, `orchestrator.py`)
