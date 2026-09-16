"""
config.py
─────────────────────────────────────────────────────────────────────────────
Central configuration file for the IEMOCAP emotion recognition experiment.

WHY THIS FILE EXISTS
────────────────────
Every number, path, and model choice in the experiment lives here.
Nothing is hard-coded inside other files.
This means:
  - You can change a hyperparameter once and it propagates everywhere.
  - When you write the paper, you know exactly what values you used.
  - Your supervisor can read this file to understand the full experiment setup.

HOW TO USE IT
─────────────
  from config import cfg
  print(cfg.lora_rank)   # → 8

"""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

import os
from dataclasses import dataclass, field
from typing import List


@dataclass
class Config:
    # ─────────────────────────────────────────────────────────────────────────
    # PATHS  ←  MUST EDIT all paths in this block
    # ─────────────────────────────────────────────────────────────────────────

    # Root folder where IEMOCAP is stored on your machine.
    # Example: "/home/huda/data/IEMOCAP_full_release"
    # The expected internal structure is:
    #   IEMOCAP_full_release/
    #     Session1/
    #       dialog/
    #         EmoEvaluation/    ← .txt files with emotion labels
    #         transcriptions/   ← .txt files with text transcripts
    #       sentences/
    #         wav/              ← .wav audio files, one per utterance
    #     Session2/ ...

    iemocap_root: str = str(
    PROJECT_ROOT / "data_splits" / "data" / "IEMOCAP_full_release"
) # ← MUST EDIT

    # Where to save processed outputs: split CSVs, checkpoints, logs.
    output_dir:     str = str(PROJECT_ROOT / "outputs")
    checkpoint_dir: str = str(PROJECT_ROOT / "checkpoints")
    log_dir:        str = str(PROJECT_ROOT / "logs")
    splits_dir:     str = str(PROJECT_ROOT / "data_splits")

    # ─────────────────────────────────────────────────────────────────────────
    # DATASET SELECTION
    # ─────────────────────────────────────────────────────────────────────────
    # Controls which dataset is active for a given training run.
    # Overridden at runtime via  --dataset {iemocap|meld}  in train.py.
    #
    # split_mode is derived automatically in train.py main():
    #   iemocap → "loso"   (Leave-One-Session-Out, 5 folds)
    #   meld    → "native" (official train / dev / test split)
    #
    # METHODOLOGICAL NOTE: the two protocols are deliberately different.
    # IEMOCAP has no standard speaker-independent split, so LOSO is the
    # accepted protocol in the literature. MELD ships an official split
    # tied to TV episode boundaries; using it ensures comparability with
    # published MELD baselines. Both datasets share the same label space
    # (anger=0, fear=1, joy=2, neutral=3, sadness=4) and the same primary
    # metric (Macro F1), enabling direct cross-dataset comparison.
    dataset:    str = "iemocap"   # "iemocap" | "meld"
    split_mode: str = "loso"      # "loso"    | "native"  (set by train.py at startup)

    # ─────────────────────────────────────────────────────────────────────────
    # MELD PATHS  ← MUST EDIT meld_root before running meld_preprocessing.py
    # ─────────────────────────────────────────────────────────────────────────
    # meld_root: top-level folder produced by extracting MELD.Raw.tar.gz.
    # Expected contents:
    #   train.tar.gz, dev.tar.gz, test.tar.gz   ← inner tarballs
    #   train_sent_emo.csv, dev_sent_emo.csv, test_sent_emo.csv
    meld_root: str = str(PROJECT_ROOT / "src" / "MELD.Raw")             
    meld_audio_dir:  str = str(PROJECT_ROOT / "data_splits" / "meld_audio")            # extracted .wav files
    meld_splits_dir: str = str(PROJECT_ROOT / "data_splits" / "meld")                  # meld_{train,dev,test}.csv

    # ─────────────────────────────────────────────────────────────────────────
    # MELD LABEL MAP
    # ─────────────────────────────────────────────────────────────────────────
    # METHODOLOGICAL DECISION: disgust and surprise are dropped from MELD so
    # the label space is identical to IEMOCAP's 5-class set. The remaining
    # five emotions map to the same LABEL_TO_IDX integers as IEMOCAP:
    #   anger=0, fear=1, joy=2, neutral=3, sadness=4
    # This identity of indices means the same model head, loss weights, and
    # metric code work unchanged for both datasets.
    #
    # Justification for drops:
    #   disgust  — low inter-annotator agreement in TV dialogue context
    #   surprise — absent from IEMOCAP, breaking cross-dataset comparability
    meld_label_map: dict = field(default_factory=lambda: {
        "anger":    "anger",
        "fear":     "fear",
        "joy":      "joy",
        "neutral":  "neutral",
        "sadness":  "sadness",
        "disgust":  None,       # dropped
        "surprise": None,       # dropped
    })

    # ─────────────────────────────────────────────────────────────────────────
    # LABEL MAPPING
    # ─────────────────────────────────────────────────────────────────────────
    # IEMOCAP original labels and what we do with each.
    # "keep" → include with the mapped target class name
    # "drop" → remove these utterances from the dataset entirely

    label_map: dict = field(default_factory=lambda: {
        # Keep these — direct mapping to target class
        "ang":        "anger",     # anger
        "sad":        "sadness",   # sadness
        "fea":        "fear",      # fear (n≈40; low support — reported separately)
        "neu":        "neutral",   # neutral

        # Drop: disgust has only 2 samples total across all 5 sessions.
        # Including it would make the class-weight system numerically unstable
        # (weight caps at 10 on a class with 0–1 training samples per fold)
        # and is statistically indefensible in a peer-reviewed paper.
        "dis":        None,        # drop disgust

        # Merge: happiness + excitement → joy
        # Justification: both are high-arousal positive emotions.
        # The affective circumplex (Russell, 1980) places them in the same quadrant.
        "hap":        "joy",       # happiness → joy
        "exc":        "joy",       # excitement → joy

        # Drop: frustration and surprise
        # Justification: frustration is IEMOCAP-specific (not in MELD or CREMA-D),
        # breaking cross-dataset comparability. Surprise is inconsistently annotated.
        "fru":        None,        # drop frustration
        "sur":        None,        # drop surprise

        # Drop: 'other' and 'xxx' which are annotation artefacts
        "oth":        None,
        "xxx":        None,
    })

    # The 5 target class names, in a fixed order.
    # IMPORTANT: this order defines the integer label indices used throughout.
    # anger=0, fear=1, joy=2, neutral=3, sadness=4
    #
    # Disgust was dropped: only 2 samples exist across all 5 IEMOCAP sessions.
    # Fear (n≈40) is retained but flagged as low-support in all reported tables.
    emotion_classes: List[str] = field(default_factory=lambda: [
        "anger", "fear", "joy", "neutral", "sadness"
    ])

    # ─────────────────────────────────────────────────────────────────────────
    # CROSS-VALIDATION PROTOCOL
    # ─────────────────────────────────────────────────────────────────────────
    # IEMOCAP has 5 sessions. We use Leave-One-Session-Out (LOSO).
    # For fold k: test = Session k,
    #             val  = Session (k % 5) + 1  (the next session, wraps around),
    #             train = the remaining 3 sessions.
    # This gives a proper speaker-independent evaluation:
    # no speaker in the test set appears in train or val.

    n_folds: int = 5   # always 5 for IEMOCAP LOSO

    # ─────────────────────────────────────────────────────────────────────────
    # AUDIO PREPROCESSING
    # ─────────────────────────────────────────────────────────────────────────
    sample_rate:     int   = 16000   # Hz — WavLM was pretrained on 16 kHz audio

    # Maximum audio clip length in seconds.
    # WavLM self-attention is O(n²) in frame count (50 frames/sec).
    #   8s → 400 frames → 160,000 attention pairs per head per layer
    #   6s → 300 frames →  90,000 attention pairs  (1.78× faster)
    #   4s → 200 frames →  40,000 attention pairs  (4.0×  faster)
    # IEMOCAP median utterance duration is ~2.5s; 6s covers ~96% of utterances.
    # Run data_preprocessing.py --analyze_lengths to see exact truncation stats.
    # Justified in the paper as: "utterances exceeding 6 s were truncated at
    # the trailing end, as emotion cues are salient in the first few seconds."
    max_audio_secs:  float = 6.0

    # RMS normalization target level in dBFS (decibels relative to full scale).
    # -20 dBFS is the standard telephony level.
    rms_target_dbfs: float = -20.0

    # Directory for pre-processed waveform cache (.pt tensors, one per utterance).
    # Set to None to disable caching and load directly from .wav every epoch.
    # When set, run:  python data_preprocessing.py --cache_audio
    # to build the cache before training. Cache is keyed by max_audio_secs so
    # changing the length automatically invalidates the old cache.
    audio_cache_dir: str = "./data_splits/audio_cache"

    # ─────────────────────────────────────────────────────────────────────────
    # TEXT PREPROCESSING
    # ─────────────────────────────────────────────────────────────────────────
    # RoBERTa tokenizer name — must match the model checkpoint below.
    text_tokenizer:  str = "roberta-base"
    max_text_tokens: int = 128   # covers 95th %ile of IEMOCAP utterance token counts

    # ─────────────────────────────────────────────────────────────────────────
    # MODEL CHECKPOINTS (HuggingFace Hub names)
    # ─────────────────────────────────────────────────────────────────────────
    text_model_name:  str = "roberta-base"
    audio_model_name: str = "microsoft/wavlm-base-plus"

    # ─────────────────────────────────────────────────────────────────────────
    # LORA HYPERPARAMETERS
    # ─────────────────────────────────────────────────────────────────────────
    # These settings apply to BOTH text and audio LoRA adapters.
    # They are the established defaults from Hu et al. (2022).

    lora_rank:    int   = 8     # r: rank of the low-rank decomposition matrices
    lora_alpha:   int   = 16    # scaling factor; effective scale = alpha/rank = 2.0
    lora_dropout: float = 0.05  # dropout inside LoRA layers (light regularisation)

    # Which attention weight matrices to apply LoRA to.
    # "query" and "value" are the standard targets (Hu et al., 2022).
    # Adding "key" gives marginal improvement at extra cost.
    lora_target_modules_text:  List[str] = field(
        default_factory=lambda: ["query", "value"]
    )
    lora_target_modules_audio: List[str] = field(
        default_factory=lambda: ["q_proj", "v_proj"]
        # WavLM uses q_proj/v_proj naming (not query/value like BERT family)
    )

    # ─────────────────────────────────────────────────────────────────────────
    # FUSION ARCHITECTURE
    # ─────────────────────────────────────────────────────────────────────────
    fusion_dim:     int   = 256    # d_fuse: shared projection dimension for both modalities
    fusion_dropout: float = 0.3    # dropout in the classification head

    # ─────────────────────────────────────────────────────────────────────────
    # TRAINING HYPERPARAMETERS
    # ─────────────────────────────────────────────────────────────────────────
    learning_rate:    float = 3e-4  # LoRA-specific; higher than full fine-tuning (1e-5)
    weight_decay:     float = 0.01  # L2 regularisation via AdamW
    batch_size:       int   = 16    # per GPU; reduce to 8 if OOM on audio models
    num_epochs:       int   = 30    # maximum; early stopping will stop sooner
    warmup_steps:     int   = 500   # linear warmup steps for the LR scheduler
    early_stop_patience: int = 5    # stop if val Macro F1 does not improve for 5 epochs
    label_smoothing:  float = 0.1   # reduces overconfidence; helps with class imbalance
    gradient_clip:    float = 1.0   # max gradient norm (prevents exploding gradients)

    # ─────────────────────────────────────────────────────────────────────────
    # CLASS WEIGHTING
    # ─────────────────────────────────────────────────────────────────────────
    # Compute inverse-frequency class weights to handle class imbalance.
    # Fear (n≈40) has very low support; the cap prevents its weight from
    # dominating the loss and destabilising training for majority classes.
    # Set to True to use weighted cross-entropy loss.
    use_class_weights: bool = True

    # Maximum allowed class weight. Formula: total / (n_classes × count_c).
    # Applied identically to IEMOCAP and MELD training splits.
    #   IEMOCAP: fear n≈24/fold  → raw weight ≈ 23 → capped to 5
    #   MELD:    fear n≈268      → raw weight ≈  5.7 → capped to 5 (barely triggered)
    # A single shared cap is methodologically cleaner than per-dataset values;
    # it can be justified in the paper as a fixed regularisation hyperparameter.
    # Values above 10 risk destabilising training on minority classes.
    max_class_weight: float = 5.0

    # ─────────────────────────────────────────────────────────────────────────
    # MIXED PRECISION TRAINING
    # ─────────────────────────────────────────────────────────────────────────
    # torch.cuda.amp.autocast runs the forward pass in fp16 on Tensor Cores
    # and keeps fp32 for numerically sensitive ops (softmax, norms).
    # GradScaler rescales the loss to prevent fp16 underflow during backward.
    # On RTX 4060: ~1.5-2× throughput vs fp32 for transformer workloads.
    # Scientific validity: AMP does not change the training objective; it is
    # a numerically equivalent approximation. Report in the paper as:
    # "Training used automatic mixed precision (PyTorch AMP) on all GPU runs."
    use_amp: bool = True

    # ─────────────────────────────────────────────────────────────────────────
    # REPRODUCIBILITY
    # ─────────────────────────────────────────────────────────────────────────
    seed: int = 42   # used for Python random, NumPy, PyTorch, DataLoader workers

    # ─────────────────────────────────────────────────────────────────────────
    # HARDWARE
    # ─────────────────────────────────────────────────────────────────────────
    # num_workers: number of DataLoader subprocesses for parallel data loading.
    # On Windows, each worker is spawned (not forked), which adds ~1s startup
    # overhead per epoch. 2 workers balances parallelism vs spawn cost.
    # If you see "RuntimeError: DataLoader worker exited unexpectedly", set to 0.
    num_workers:       int  = 2
    pin_memory:        bool = True   # faster CPU→GPU transfer; disable on CPU-only
    # persistent_workers=True keeps worker processes alive between epochs,
    # eliminating the per-epoch spawn cost (~2–4s saved per epoch).
    # Only active when num_workers > 0.
    persistent_workers: bool = True

    # ─────────────────────────────────────────────────────────────────────────
    # DEBUG MODE
    # ─────────────────────────────────────────────────────────────────────────
    # When debug=True (set via --debug flag in train.py):
    #   - Only the first debug_subset_size samples are used per split
    #   - num_epochs is capped at 3
    # Use this to verify the full pipeline runs end-to-end before a full run.
    debug_subset_size: int = 64   # samples per split when --debug is active


# Single global config instance imported everywhere else.
cfg = Config()

# Derived constant: maximum number of audio samples after resampling.
# Placed here so other files can import it directly.
MAX_AUDIO_SAMPLES = int(cfg.sample_rate * cfg.max_audio_secs)  # 128,000

# Integer label index lookup: emotion name → integer 0–5
LABEL_TO_IDX = {name: idx for idx, name in enumerate(cfg.emotion_classes)}
# {"anger": 0, "disgust": 1, "fear": 2, "joy": 3, "neutral": 4, "sadness": 5}

IDX_TO_LABEL = {idx: name for name, idx in LABEL_TO_IDX.items()}
# {0: "anger", 1: "disgust", ...}

NUM_CLASSES = len(cfg.emotion_classes)  # 6
