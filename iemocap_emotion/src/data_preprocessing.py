"""
data_preprocessing.py
─────────────────────────────────────────────────────────────────────────────
IEMOCAP dataset parsing, label mapping, split generation, and audio
preprocessing. This file turns raw IEMOCAP files into clean DataFrames
and preprocessed audio tensors.

WHY THIS FILE EXISTS
────────────────────
The raw IEMOCAP dataset is stored as plain text files with a specific
structure that is not immediately usable by PyTorch. This file:
  1. Reads the emotion label files (.txt) to find each utterance's label.
  2. Reads the transcript files (.txt) to get the spoken text.
  3. Builds the path to each utterance's audio .wav file.
  4. Applies the 6-class label mapping (merges, drops).
  5. Creates the 5 Leave-One-Session-Out fold assignments.
  6. Provides audio preprocessing (resampling, normalisation, padding).

IEMOCAP FILE STRUCTURE (what you will find after downloading)
──────────────────────────────────────────────────────────────
IEMOCAP_full_release/
  Session1/
    dialog/
      EmoEvaluation/
        Ses01F_impro01.txt       ← emotion labels for each utterance
        Ses01F_script01_1.txt    ← scripted dialogue labels
        ...
      transcriptions/
        Ses01F_impro01.txt       ← text transcripts
        ...
    sentences/
      wav/
        Ses01F_impro01/
          Ses01F_impro01_F000.wav  ← one .wav file per utterance
          Ses01F_impro01_F001.wav
          ...
  Session2/ ...
  Session3/ ...
  Session4/ ...
  Session5/ ...

LABEL FILE FORMAT (EmoEvaluation .txt)
────────────────────────────────────────
Each line looks like:
  [6.2901 - 8.2357]	Ses01F_impro01_F000	neu	[2, 2, 2];
  [start - end]         utterance_id          label  annotator votes

We extract: utterance_id and label (the agreed label, not the votes).

TRANSCRIPT FILE FORMAT
──────────────────────
Each line looks like:
  Ses01F_impro01_F000 [6.29-8.24]: I had fun.

We extract: utterance_id and the text after the timestamp.
"""

import os
import re
import pandas as pd
import numpy as np
import torch
import torchaudio
import torchaudio.functional as F

from pathlib import Path
from typing import Tuple, Optional

from config import cfg, LABEL_TO_IDX, MAX_AUDIO_SAMPLES


# ─────────────────────────────────────────────────────────────────────────────
# STEP 1: PARSE EMOTION LABELS FROM IEMOCAP EmoEvaluation FILES
# ─────────────────────────────────────────────────────────────────────────────

def parse_emotion_labels(session_dir: str) -> dict:
    """
    WHAT IT DOES
    ────────────
    Reads all EmoEvaluation .txt files in one IEMOCAP session directory and
    returns a dictionary mapping utterance_id → raw emotion label.

    WHY IT IS NEEDED
    ────────────────
    The emotion labels are stored in plain text files, one per dialogue.
    We need to read all of them to get labels for every utterance in a session.

    WHAT INPUT IT EXPECTS
    ─────────────────────
    session_dir: string path to one session folder, e.g.
                 "/path/to/IEMOCAP_full_release/Session1"

    WHAT OUTPUT IT PRODUCES
    ───────────────────────
    dict: { "Ses01F_impro01_F000": "neu",
            "Ses01F_impro01_F001": "ang", ... }

    COMMON MISTAKES
    ───────────────
    - The label file has comment lines starting with %; skip them.
    - Some lines have extra whitespace or tabs; strip carefully.
    - The utterance_id in the label file must EXACTLY match the audio filename.
    """
    emo_dir = os.path.join(session_dir, "dialog", "EmoEvaluation")
    labels = {}

    # Find all .txt files in the EmoEvaluation folder
    for fname in os.listdir(emo_dir):
        if not fname.endswith(".txt"):
            continue

        filepath = os.path.join(emo_dir, fname)
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()

                # Skip empty lines and comment lines (start with %)
                if not line or line.startswith("%"):
                    continue

                # Each valid label line starts with a time stamp in brackets.
                # Pattern: [start - end]\tutterance_id\tlabel\t[votes];
                if not line.startswith("["):
                    continue

                # Split by tab: fields are timestamp, utterance_id, label, votes
                parts = line.split("\t")
                if len(parts) < 3:
                    continue  # malformed line; skip

                utterance_id = parts[1].strip()
                raw_label    = parts[2].strip()

                # Some label files have a semicolon-separated votes field after
                # the label; we only want the consensus label (first field).
                raw_label = raw_label.split(";")[0].strip()

                labels[utterance_id] = raw_label

    return labels


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2: PARSE TRANSCRIPTS
# ─────────────────────────────────────────────────────────────────────────────

def parse_transcripts(session_dir: str) -> dict:
    """
    WHAT IT DOES
    ────────────
    Reads all transcription .txt files in one IEMOCAP session and returns
    a dictionary mapping utterance_id → text transcript.

    WHY IT IS NEEDED
    ────────────────
    We need the text for the RoBERTa text branch. IEMOCAP provides manual
    transcripts, which are high quality (no ASR errors).

    WHAT INPUT IT EXPECTS
    ─────────────────────
    session_dir: string path to one session folder

    WHAT OUTPUT IT PRODUCES
    ───────────────────────
    dict: { "Ses01F_impro01_F000": "I had fun today.",
            "Ses01F_impro01_F001": "Really? That is great.", ... }

    COMMON MISTAKES
    ───────────────
    - Transcript lines contain timestamps before the text: strip them.
    - Some utterances may have no transcript entry; handle gracefully.
    - Encoding: some IEMOCAP files use latin-1, not UTF-8; use errors='replace'.
    """
    trans_dir = os.path.join(session_dir, "dialog", "transcriptions")
    transcripts = {}

    for fname in os.listdir(trans_dir):
        if not fname.endswith(".txt"):
            continue

        filepath = os.path.join(trans_dir, fname)
        with open(filepath, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                # Each line format:
                # Ses01F_impro01_F000 [6.29-8.24]: I had fun.
                # We split on ": " to separate the header from the text.
                if ": " not in line:
                    continue

                # Take everything before the first ": " as the header,
                # and everything after as the transcript text.
                header, text = line.split(": ", maxsplit=1)

                # The utterance_id is the first token of the header
                # (before the timestamp in brackets)
                utterance_id = header.split()[0].strip()
                text = text.strip()

                transcripts[utterance_id] = text

    return transcripts


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3: BUILD AUDIO FILE PATHS
# ─────────────────────────────────────────────────────────────────────────────

def get_audio_path(iemocap_root: str, utterance_id: str) -> str:
    """
    WHAT IT DOES
    ────────────
    Constructs the full path to an utterance's .wav audio file from the
    utterance_id string alone.

    WHY IT IS NEEDED
    ────────────────
    The audio files are stored in a nested folder structure. The utterance_id
    encodes the session and dialogue name, which we use to navigate to the
    correct folder.

    WHAT INPUT IT EXPECTS
    ─────────────────────
    iemocap_root:  e.g. "/path/to/IEMOCAP_full_release"
    utterance_id:  e.g. "Ses01F_impro01_F000"

    WHAT OUTPUT IT PRODUCES
    ───────────────────────
    Full path to the .wav file:
    "/path/to/IEMOCAP_full_release/Session1/sentences/wav/Ses01F_impro01/Ses01F_impro01_F000.wav"

    HOW THE DECODING WORKS
    ──────────────────────
    "Ses01F_impro01_F000"
      Ses   → session marker
      01    → session number → "Session1"
      F     → speaker gender (F=female, M=male); part of dialogue name
      impro01 → dialogue name
      F000  → utterance within dialogue

    The audio file sits in:
      IEMOCAP_full_release / Session{N} / sentences / wav / {dialogue_id} / {utterance_id}.wav
    where dialogue_id = everything before the last underscore-separated token
    (i.e., everything except "_F000" or "_M001" etc.)

    COMMON MISTAKES
    ───────────────
    - The dialogue folder name does NOT include the speaker token at the end.
      "Ses01F_impro01_F000" → dialogue folder = "Ses01F_impro01" (drop "_F000").
    - Session number is the 4th and 5th characters: "Ses01..." → session_num = "01"
      but the folder is named "Session1" not "Session01". Strip leading zero.
    """
    # Extract session number from utterance_id
    # e.g. "Ses01F_impro01_F000" → session_num_str = "01"
    session_num_str = utterance_id[3:5]            # "01"
    session_num     = str(int(session_num_str))    # "1" (remove leading zero)
    session_folder  = f"Session{session_num}"      # "Session1"

    # Extract dialogue folder: everything up to (not including) the LAST "_token"
    # e.g. "Ses01F_impro01_F000" → parts = ["Ses01F", "impro01", "F000"]
    #      dialogue = "Ses01F_impro01"
    parts        = utterance_id.split("_")
    dialogue_id  = "_".join(parts[:-1])            # join all but the last part

    audio_path = os.path.join(
        iemocap_root,
        session_folder,
        "sentences", "wav",
        dialogue_id,
        f"{utterance_id}.wav"
    )
    return audio_path


# ─────────────────────────────────────────────────────────────────────────────
# STEP 4: BUILD THE FULL METADATA DATAFRAME
# ─────────────────────────────────────────────────────────────────────────────

def build_metadata(iemocap_root: str) -> pd.DataFrame:
    """
    WHAT IT DOES
    ────────────
    Iterates through all 5 IEMOCAP sessions, parses labels and transcripts,
    applies the label mapping, filters dropped classes, and returns a single
    clean pandas DataFrame.

    WHY IT IS NEEDED
    ────────────────
    All downstream code (Dataset class, split generation, class weight
    computation) works with this single DataFrame. Building it once and
    saving it to CSV means you only parse the raw IEMOCAP files once.

    WHAT INPUT IT EXPECTS
    ─────────────────────
    iemocap_root: string path to IEMOCAP_full_release

    WHAT OUTPUT IT PRODUCES
    ───────────────────────
    A pandas DataFrame with columns:
      utterance_id   | str  | e.g. "Ses01F_impro01_F000"
      session        | int  | 1–5
      raw_label      | str  | original IEMOCAP label, e.g. "neu"
      emotion        | str  | mapped class name, e.g. "neutral"
      label_idx      | int  | integer label 0–5
      text           | str  | transcript text
      audio_path     | str  | full path to the .wav file
      audio_exists   | bool | True if the .wav file is present on disk

    COMMON MISTAKES
    ───────────────
    - Some utterance IDs appear in the label file but not in the transcript
      file (or vice versa). The merge below (inner join behaviour via .get())
      handles this gracefully: missing text is set to empty string.
    - Verify that audio_exists=True for all retained rows. If audio files
      are missing, something went wrong with your IEMOCAP download.
    - IEMOCAP has some utterances labelled "xxx" (uncertain) or "oth"
      (other); these are in the label_map as None and are dropped.
    """
    records = []

    for session_num in range(1, 6):   # sessions 1 through 5
        session_dir = os.path.join(iemocap_root, f"Session{session_num}")

        if not os.path.isdir(session_dir):
            print(f"[WARNING] Session directory not found: {session_dir}")
            continue

        print(f"  Parsing Session {session_num}...")

        # Parse labels and transcripts for this session
        labels      = parse_emotion_labels(session_dir)
        transcripts = parse_transcripts(session_dir)

        for utterance_id, raw_label in labels.items():
            # Apply label mapping from config.py
            # cfg.label_map returns:
            #   - a mapped class name (str)  → keep
            #   - None                        → drop this utterance
            #   - KeyError (unknown label)    → skip with warning
            mapped = cfg.label_map.get(raw_label, None)

            # Drop utterances with None mapping (frustration, surprise, xxx, oth)
            if mapped is None:
                continue

            # Get text transcript (empty string if not found)
            text = transcripts.get(utterance_id, "")

            # Build audio file path
            audio_path = get_audio_path(iemocap_root, utterance_id)

            records.append({
                "utterance_id": utterance_id,
                "session":      session_num,
                "raw_label":    raw_label,
                "emotion":      mapped,
                "label_idx":    LABEL_TO_IDX[mapped],
                "text":         text,
                "audio_path":   audio_path,
                "audio_exists": os.path.isfile(audio_path),
            })

    df = pd.DataFrame(records)

    # Report how many rows were built per session and class
    print(f"\n[Metadata] Total utterances retained: {len(df)}")
    print(f"[Metadata] Class distribution:\n{df['emotion'].value_counts()}")
    print(f"[Metadata] Missing audio files: {(~df['audio_exists']).sum()}")

    return df


# ─────────────────────────────────────────────────────────────────────────────
# STEP 5: GENERATE LOSO FOLDS
# ─────────────────────────────────────────────────────────────────────────────

def generate_loso_folds(df: pd.DataFrame) -> pd.DataFrame:
    """
    WHAT IT DOES
    ────────────
    Adds a 'fold' column to the DataFrame that assigns each utterance to
    one of 5 LOSO folds, and a 'split' column that says 'train', 'val',
    or 'test' for a given fold.

    Because LOSO is a cross-validation scheme (not a single split), the
    'split' column changes meaning depending on which fold is active.
    This function writes the permanent assignment into the DataFrame;
    the Dataset class uses the 'session' column to filter appropriately
    at runtime.

    WHY IT IS NEEDED
    ────────────────
    The LOSO protocol ensures that no speaker in the test set appears in
    the training set. IEMOCAP has 10 unique speakers (2 per session).
    LOSO guarantees speaker independence, which is the correct evaluation
    protocol for speech emotion recognition.

    SPLIT ASSIGNMENT PER FOLD:
    ──────────────────────────
    Fold k:
      test  = Session k
      val   = Session (k mod 5) + 1   (next session, wraps: fold 5 → val = Session 1)
      train = all remaining sessions

    Example:
      Fold 1: test=S1, val=S2, train=S3,S4,S5
      Fold 2: test=S2, val=S3, train=S1,S4,S5
      Fold 3: test=S3, val=S4, train=S1,S2,S5
      Fold 4: test=S4, val=S5, train=S1,S2,S3
      Fold 5: test=S5, val=S1, train=S2,S3,S4

    WHAT INPUT IT EXPECTS
    ─────────────────────
    df: the DataFrame from build_metadata(), with a 'session' column (1–5)

    WHAT OUTPUT IT PRODUCES
    ───────────────────────
    The same DataFrame with two additional columns:
      "fold_test_session": which session is the TEST set for each fold (int 1–5)
        → This is just the session number for clarity; not a new concept.
        → An utterance in session 3 is the test set for fold 3.

    IMPORTANT: We do NOT add a "split" column here because every utterance
    belongs to a different split depending on which fold is active.
    The Dataset class takes a fold number as input and derives train/val/test
    membership at runtime using the 'session' column.

    HOW TO USE THE OUTPUT
    ─────────────────────
    To get the train/val/test rows for Fold 2:
      test_fold = 2
      val_session = (2 % 5) + 1 = 3
      train_sessions = [1, 4, 5]

      test_df  = df[df["session"] == 2]
      val_df   = df[df["session"] == 3]
      train_df = df[df["session"].isin([1, 4, 5])]
    """
    # The fold assignment is simply the session number.
    # Utterances in session k are the TEST set for fold k.
    # No new column needed; the 'session' column already encodes this.
    # We just validate the structure and print the fold statistics.

    print("\n[LOSO Folds] Fold statistics:")
    print(f"{'Fold':<8} {'Test Session':<15} {'Val Session':<14} {'Train Sessions':<20} "
          f"{'Test N':<10} {'Val N':<10} {'Train N'}")
    print("-" * 90)

    fold_info = []

    for fold in range(1, 6):
        test_session  = fold
        val_session   = (fold % 5) + 1        # wraps: fold 5 → val session 1
        train_sessions = [s for s in range(1, 6)
                          if s != test_session and s != val_session]

        n_test  = (df["session"] == test_session).sum()
        n_val   = (df["session"] == val_session).sum()
        n_train = df["session"].isin(train_sessions).sum()

        print(f"{fold:<8} {test_session:<15} {val_session:<14} "
              f"{str(train_sessions):<20} {n_test:<10} {n_val:<10} {n_train}")

        fold_info.append({
            "fold":           fold,
            "test_session":   test_session,
            "val_session":    val_session,
            "train_sessions": train_sessions,
        })

    # Per-fold class distribution check.
    # Any class with fewer than this many training samples will get weight=0
    # (because compute_class_weights zeros out absent classes), which means
    # no gradient signal for that class in that fold.
    min_train_samples = 5
    from config import IDX_TO_LABEL, NUM_CLASSES

    print("\n[LOSO Folds] Per-fold class distribution (training split only):")
    any_warning = False
    for fold in range(1, 6):
        test_session   = fold
        val_session    = (fold % 5) + 1
        train_sessions = [s for s in range(1, 6)
                          if s != test_session and s != val_session]
        train_df_fold  = df[df["session"].isin(train_sessions)]
        counts = train_df_fold["label_idx"].value_counts().to_dict()
        for idx in range(NUM_CLASSES):
            n = counts.get(idx, 0)
            if n < min_train_samples:
                print(f"  [WARNING] Fold {fold}: class '{IDX_TO_LABEL[idx]}' "
                      f"has only {n} training samples. "
                      f"Its class weight will be {'0 (absent)' if n == 0 else 'high'}.")
                any_warning = True
    if not any_warning:
        print("  All classes have sufficient training samples in every fold.")

    return fold_info   # list of dicts, one per fold


def get_split_dfs(df: pd.DataFrame, fold: int):
    """
    WHAT IT DOES
    ────────────
    Given the metadata DataFrame and a fold number (1–5), returns three
    DataFrames: train_df, val_df, test_df.

    WHY IT IS NEEDED
    ────────────────
    Convenience function called by the training script. Encapsulates the
    LOSO fold logic in one place so the training script does not need to
    know about session numbers.

    WHAT INPUT IT EXPECTS
    ─────────────────────
    df:   metadata DataFrame from build_metadata()
    fold: int, 1–5

    WHAT OUTPUT IT PRODUCES
    ───────────────────────
    train_df, val_df, test_df: three pandas DataFrames, subsets of df.
    All three are reset_index(drop=True) for clean integer indexing.

    COMMON MISTAKES
    ───────────────
    - Do NOT call this during the HPO phase on test_df. Only inspect test_df
      at the very end, after all hyperparameter decisions are finalised.
    """
    test_session   = fold
    val_session    = (fold % 5) + 1
    train_sessions = [s for s in range(1, 6)
                      if s != test_session and s != val_session]

    test_df  = df[df["session"] == test_session].reset_index(drop=True)
    val_df   = df[df["session"] == val_session].reset_index(drop=True)
    train_df = df[df["session"].isin(train_sessions)].reset_index(drop=True)

    return train_df, val_df, test_df


# ─────────────────────────────────────────────────────────────────────────────
# STEP 6: COMPUTE CLASS WEIGHTS
# ─────────────────────────────────────────────────────────────────────────────

def compute_class_weights(train_df: pd.DataFrame) -> torch.Tensor:
    """
    WHAT IT DOES
    ────────────
    Computes inverse-frequency class weights from the training set class
    distribution. Returns a tensor of shape [num_classes] passed to the
    CrossEntropyLoss weight argument.

    WHY IT IS NEEDED
    ────────────────
    IEMOCAP has severe class imbalance. Fear has ~40 samples in the 5-class
    set (after dropping disgust), while Neutral and Joy have ~1600+ each.
    Without class weighting, the model learns to predict Neutral/Joy most of
    the time and achieves high accuracy but low Macro F1.

    Class weight formula: weight_c = total_samples / (num_classes × count_c)
    This is the sklearn 'balanced' formula.

    WHAT INPUT IT EXPECTS
    ─────────────────────
    train_df: the training fold DataFrame (NEVER the test DataFrame)

    WHAT OUTPUT IT PRODUCES
    ───────────────────────
    torch.Tensor of shape [NUM_CLASSES], dtype float32.
    Minority classes get weight > 1; majority classes get weight < 1.

    COMMON MISTAKES
    ───────────────
    - Compute class weights on the TRAINING SPLIT ONLY, not on the full
      dataset. Computing on the full dataset introduces leakage because
      the test set distribution influences the training loss.
    - If a class is missing from the training fold (possible for fear with
      n≈40), its weight would be infinity. We clamp to cfg.max_class_weight
      (default 5.0) to prevent minority-class gradients from dominating.
    """
    from config import NUM_CLASSES

    counts = np.zeros(NUM_CLASSES, dtype=np.float32)
    for label_idx in train_df["label_idx"]:
        counts[label_idx] += 1

    total = counts.sum()
    max_weight = cfg.max_class_weight   # configurable clamp; default 5.0

    weights = np.zeros(NUM_CLASSES, dtype=np.float32)
    for i in range(NUM_CLASSES):
        if counts[i] > 0:
            weights[i] = min(total / (NUM_CLASSES * counts[i]), max_weight)
        else:
            # Class not present in training fold → set weight 0 (cannot learn it)
            weights[i] = 0.0
            print(f"[WARNING] Class index {i} has zero samples in training fold.")

    print(f"\n[Class Weights] Computed from training fold ({int(total)} samples):")
    from config import IDX_TO_LABEL
    for i, w in enumerate(weights):
        count = int(counts[i])
        print(f"  {IDX_TO_LABEL[i]:>10}: count={count:>5}, weight={w:.4f}")

    return torch.tensor(weights, dtype=torch.float32)


# ─────────────────────────────────────────────────────────────────────────────
# STEP 7: AUDIO PREPROCESSING FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

def load_and_preprocess_audio(
    audio_path: str,
    target_sr:  int   = 16000,
    max_samples: int  = MAX_AUDIO_SAMPLES,
    rms_target_dbfs: float = -20.0,
) -> Tuple[torch.Tensor, bool, int]:
    """
    WHAT IT DOES
    ────────────
    Loads one audio .wav file, resamples it to target_sr, normalises the
    amplitude, pads or truncates to max_samples, and returns a 1D tensor
    of shape [max_samples].

    WHY IT IS NEEDED
    ────────────────
    WavLM was pretrained on 16 kHz audio. Feeding 8 kHz or 44.1 kHz audio
    directly would create a distribution mismatch. Amplitude normalisation
    prevents the model from learning spurious gain-level correlations.
    Fixed-length tensors are needed so we can batch multiple utterances.

    WHAT INPUT IT EXPECTS
    ─────────────────────
    audio_path:       full path to a .wav file
    target_sr:        target sample rate in Hz (default 16000)
    max_samples:      maximum number of samples to keep (default 128,000 = 8 s)
    rms_target_dbfs:  target RMS level in dBFS (default -20.0)

    WHAT OUTPUT IT PRODUCES
    ───────────────────────
    (waveform, success, original_samples):
      waveform:         torch.Tensor of shape [max_samples], dtype float32
      success:          bool, False if the file could not be loaded
      original_samples: int, number of real (non-padding) samples in waveform.
                        Capped at max_samples. Used to build the attention mask
                        without a second disk read.

    PREPROCESSING STEPS IN ORDER
    ──────────────────────────────
    1. Load:     torchaudio loads the .wav file as a float32 tensor
    2. Mono:     convert to mono (average channels) if stereo
    3. Resample: resample from original_sr to target_sr if they differ
    4. Normalise:RMS normalise to rms_target_dbfs
    5. Truncate: if waveform is longer than max_samples, truncate from the right
    6. Pad:      if shorter than max_samples, zero-pad on the right

    COMMON MISTAKES
    ───────────────
    - torchaudio.load() returns (waveform, sample_rate).
      waveform shape is [channels, samples] NOT [samples].
      Always check waveform.shape[0] (channels) before averaging.
    - Resampling a long clip at a high rate is slow. The IEMOCAP clips are
      already at 16 kHz so resampling is usually a no-op here.
    - RMS of silence is 0. Dividing by 0 causes NaN. We add a tiny epsilon
      (1e-8) before dividing.
    """
    try:
        # Step 1: Load audio file
        # torchaudio.load returns:
        #   waveform: Tensor of shape [num_channels, num_samples], dtype float32
        #   sr:       integer sample rate of the file
        waveform, sr = torchaudio.load(audio_path)

        # Step 2: Convert to mono
        # waveform.shape is [channels, samples]
        # For mono, channels=1; for stereo, channels=2.
        # We average across the channel dimension → [1, samples]
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)  # → [1, samples]

        # Step 3: Resample to target_sr if needed
        if sr != target_sr:
            resampler = torchaudio.transforms.Resample(
                orig_freq=sr,
                new_freq=target_sr
            )
            waveform = resampler(waveform)

        # Remove the channel dimension: [1, samples] → [samples]
        waveform = waveform.squeeze(0)  # → 1D tensor of shape [samples]

        # Record actual sample count after resampling, before any padding/truncation.
        # This is the number of real audio samples we will place into the output.
        original_samples = min(waveform.shape[0], max_samples)

        # Step 4: RMS normalisation
        # RMS = root mean square of the waveform values.
        # We want to scale the waveform so its RMS equals rms_target_dbfs.
        # rms_target_dbfs = -20 dBFS → rms_target_linear = 10^(-20/20) = 0.1
        rms_target_linear = 10 ** (rms_target_dbfs / 20.0)   # → 0.1

        # Compute current RMS of the waveform
        rms_current = waveform.pow(2).mean().sqrt().item()

        if rms_current > 1e-8:   # avoid division by near-zero
            scale = rms_target_linear / rms_current
            waveform = waveform * scale

            # Hard clip at ±1.0 to prevent clipping artefacts after scaling
            waveform = waveform.clamp(-1.0, 1.0)

        # Step 5: Truncate or Step 6: Pad to max_samples
        n = waveform.shape[0]

        if n >= max_samples:
            # Truncate: keep only the first max_samples samples
            waveform = waveform[:max_samples]
        else:
            # Pad: append zeros on the right
            # torch.zeros(max_samples - n) creates a zero tensor
            # torch.cat concatenates along dimension 0
            pad = torch.zeros(max_samples - n, dtype=waveform.dtype)
            waveform = torch.cat([waveform, pad])

        # Final check: shape should be exactly [max_samples]
        assert waveform.shape[0] == max_samples, \
            f"Unexpected waveform length: {waveform.shape[0]}"

        return waveform, True, original_samples

    except Exception as e:
        print(f"[WARNING] Could not load audio: {audio_path}\n  Error: {e}")
        # Return a zero tensor so the DataLoader does not crash.
        # original_samples=0 means the attention mask will be all zeros.
        return torch.zeros(max_samples, dtype=torch.float32), False, 0


def create_audio_attention_mask(waveform: torch.Tensor, original_length: int) -> torch.Tensor:
    """
    WHAT IT DOES
    ────────────
    Creates a binary attention mask for a padded waveform.
    1 = real audio, 0 = padding.

    WHY IT IS NEEDED
    ────────────────
    After padding, the waveform has zeros at the end. WavLM should not
    attend to these padding zeros. The attention mask tells the model
    which parts are real audio (1) and which are padding (0).

    WHAT INPUT IT EXPECTS
    ─────────────────────
    waveform:        the padded tensor of shape [max_samples]
    original_length: the number of real (non-padded) samples

    WHAT OUTPUT IT PRODUCES
    ───────────────────────
    torch.Tensor of shape [max_samples], dtype long.
    Values: 1 for real samples, 0 for padding.

    NOTE: WavLM's feature extractor reduces 16,000 samples to ~50 frames
    (1 frame per 320 samples at 16 kHz). The HuggingFace WavLM model
    handles this internal subsampling automatically when you pass
    attention_mask to it. You do NOT need to manually subsample the mask.
    """
    max_samples = waveform.shape[0]
    mask = torch.zeros(max_samples, dtype=torch.long)
    mask[:min(original_length, max_samples)] = 1
    return mask


# ─────────────────────────────────────────────────────────────────────────────
# STEP 8: AUDIO LENGTH ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────

def analyze_audio_lengths(df: pd.DataFrame, check_secs=(4.0, 5.0, 6.0, 7.0, 8.0)):
    """
    WHAT IT DOES
    ────────────
    Reads the actual duration of every audio file in the metadata and reports
    what percentage of utterances would be truncated at each max_audio_secs
    threshold. Helps you choose max_audio_secs with evidence for the paper.

    WHY IT IS NEEDED
    ────────────────
    Truncating audio at 6s instead of 8s cuts WavLM's sequence length from
    400 frames to 300 frames. Since self-attention is O(n²), this gives a
    (300/400)² ≈ 1.78× speedup per layer. But you must verify that 6s covers
    enough of the IEMOCAP utterances to justify it scientifically.

    WHAT INPUT IT EXPECTS
    ─────────────────────
    df:          the metadata DataFrame from build_metadata()
    check_secs:  tuple of thresholds to test (in seconds)

    WHAT OUTPUT IT PRODUCES
    ───────────────────────
    Prints a table like:
      max_secs | covered | truncated | attention_frames | speedup_vs_8s
         4.0s  |  89.4%  |  10.6%   |      200         |    4.00×
         6.0s  |  96.8%  |   3.2%   |      300         |    1.78×
         8.0s  | 100.0%  |   0.0%   |      400         |    1.00×
    """
    sr = cfg.sample_rate
    durations = []

    print(f"\n[Audio Length Analysis] Reading {len(df)} file headers...")
    for _, row in df[df["audio_exists"]].iterrows():
        try:
            info = torchaudio.info(row["audio_path"])
            dur  = info.num_frames / info.sample_rate
            durations.append(dur)
        except Exception:
            pass

    durations = np.array(durations)
    n = len(durations)

    print(f"\n  Duration statistics over {n} utterances:")
    print(f"    min={durations.min():.2f}s  "
          f"median={np.median(durations):.2f}s  "
          f"p95={np.percentile(durations, 95):.2f}s  "
          f"max={durations.max():.2f}s")

    print(f"\n  {'max_secs':>9} | {'covered':>8} | {'truncated':>9} | "
          f"{'WavLM frames':>13} | {'attention speedup vs 8s':>23}")
    print("  " + "-" * 70)
    frames_at_8s = int(8.0 * sr / 320)
    for secs in check_secs:
        covered   = float((durations <= secs).mean()) * 100
        truncated = 100.0 - covered
        n_frames  = int(secs * sr / 320)
        speedup   = (frames_at_8s / n_frames) ** 2  # O(n²) attention
        print(f"  {secs:>8.1f}s | {covered:>7.1f}% | {truncated:>8.1f}% | "
              f"{n_frames:>13d} | {speedup:>23.2f}×")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 9: PRECOMPUTE AUDIO CACHE
# ─────────────────────────────────────────────────────────────────────────────

def precompute_audio_cache(df: pd.DataFrame, cache_dir: str) -> int:
    """
    WHAT IT DOES
    ────────────
    Pre-processes every utterance's audio once and saves the result as a
    .pt (PyTorch tensor) file. During training, the Dataset loads these
    cached tensors instead of reading and processing .wav files from scratch.

    WHY IT MATTERS FOR SPEED
    ────────────────────────
    Without caching: every epoch, for every sample:
      1. Read .wav from disk            (~5–15 ms per file, I/O bound)
      2. Resample if needed             (~1 ms)
      3. RMS normalize                  (~1 ms)
      4. Pad or truncate                (~0.5 ms)
    With 3442 training samples: ~60–90 seconds of pure preprocessing per epoch.

    With caching: every epoch, for every sample:
      1. Load .pt tensor from disk      (~1–3 ms per file, already a tensor)
    Savings: 5–10× faster data loading, which overlaps better with GPU compute.

    CACHE FILE NAMING
    ─────────────────
    Each file is named: {utterance_id}_{max_samples}.pt
    The max_samples suffix ensures the cache is automatically invalidated
    when you change max_audio_secs in config.py (the file simply will not
    be found and fresh preprocessing runs).

    WHAT EACH .pt FILE CONTAINS
    ────────────────────────────
    A dict: {'waveform': FloatTensor[max_samples], 'orig_samples': int}
    Size per file: max_samples × 4 bytes = 96,000 × 4 = ~375 KB at 6s.
    Total for 5573 utterances at 6s: ~2.1 GB. Ensure you have disk space.

    WHAT INPUT IT EXPECTS
    ─────────────────────
    df:        full metadata DataFrame (all sessions)
    cache_dir: directory where .pt files are written

    WHAT OUTPUT IT PRODUCES
    ───────────────────────
    Returns number of successfully cached files.
    Skips files that are already cached (idempotent — safe to re-run).
    """
    from utils import make_dirs

    make_dirs(cache_dir)
    max_samples = MAX_AUDIO_SAMPLES

    rows = df[df["audio_exists"]].to_dict("records")
    n_total   = len(rows)
    n_cached  = 0
    n_skipped = 0
    n_failed  = 0

    print(f"\n[Cache] Pre-computing audio for {n_total} utterances "
          f"→ {cache_dir}")
    print(f"[Cache] max_samples={max_samples} ({cfg.max_audio_secs}s at {cfg.sample_rate}Hz)")
    print(f"[Cache] Estimated disk usage: "
          f"{n_total * max_samples * 4 / 1e9:.2f} GB")

    for i, row in enumerate(rows):
        cache_path = os.path.join(
            cache_dir, f"{row['utterance_id']}_{max_samples}.pt"
        )

        if os.path.exists(cache_path):
            n_skipped += 1
        else:
            waveform, success, orig_samples = load_and_preprocess_audio(
                audio_path=row["audio_path"],
                target_sr=cfg.sample_rate,
                max_samples=max_samples,
                rms_target_dbfs=cfg.rms_target_dbfs,
            )
            if success:
                torch.save({"waveform": waveform, "orig_samples": orig_samples},
                           cache_path)
                n_cached += 1
            else:
                n_failed += 1

        if (i + 1) % 500 == 0 or (i + 1) == n_total:
            print(f"  [{i+1}/{n_total}] cached={n_cached} "
                  f"skipped={n_skipped} failed={n_failed}")

    print(f"[Cache] Done. {n_cached} new files written, "
          f"{n_skipped} already existed, {n_failed} failed.")
    return n_cached + n_skipped


def get_cache_path(utterance_id: str, cache_dir: str) -> str:
    """Returns the expected cache file path for one utterance."""
    return os.path.join(cache_dir, f"{utterance_id}_{MAX_AUDIO_SAMPLES}.pt")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 10: SAVE AND LOAD PROCESSED METADATA
# ─────────────────────────────────────────────────────────────────────────────

def save_metadata(df: pd.DataFrame, path: str):
    """Saves the metadata DataFrame to a CSV file."""
    df.to_csv(path, index=False)
    print(f"[Metadata] Saved to {path}")


def load_metadata(path: str) -> pd.DataFrame:
    """
    Loads a previously saved metadata CSV and validates that the label
    indices are consistent with the current config.py class mapping.

    WHY THIS VALIDATION EXISTS
    ──────────────────────────
    If you change emotion_classes in config.py (e.g., drop disgust to go
    from 6 to 5 classes), the saved CSV has stale label_idx values based
    on the old mapping. Training on those stale indices corrupts the model
    silently — loss looks normal but predictions are systematically wrong.

    If validation fails: delete the CSV and re-run data_preprocessing.py
    to rebuild it with the current mapping.
    """
    from config import IDX_TO_LABEL, LABEL_TO_IDX

    df = pd.read_csv(path)

    # Check every row: does label_idx + emotion agree with the current mapping?
    mismatches = []
    for _, row in df.iterrows():
        idx      = int(row["label_idx"])
        emotion  = row["emotion"]
        expected = IDX_TO_LABEL.get(idx)
        if expected is None or expected != emotion:
            mismatches.append((row["utterance_id"], emotion, idx, expected))
        if len(mismatches) >= 5:   # show at most 5 examples
            break

    if mismatches:
        print("\n[ERROR] Metadata CSV has stale label indices. "
              "This usually means you changed emotion_classes in config.py "
              "without rebuilding the CSV.")
        for uid, emo, idx, expected in mismatches:
            print(f"  {uid}: emotion='{emo}', label_idx={idx}, "
                  f"but current IDX_TO_LABEL[{idx}]='{expected}'")
        print(f"\n  FIX: Delete '{path}' and re-run data_preprocessing.py.\n")
        raise ValueError("Stale metadata CSV. Delete it and rebuild.")

    print(f"[Metadata] Loaded {len(df)} rows from {path} "
          f"(label index validation passed)")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# MAIN: run this script directly to verify IEMOCAP parsing works
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    """
    Usage:
      python data_preprocessing.py                   ← full preprocessing + save CSV
      python data_preprocessing.py --analyze_lengths ← print truncation table, then exit
      python data_preprocessing.py --cache_audio     ← preprocessing + build waveform cache
    """
    import sys
    import argparse
    from utils import make_dirs

    parser = argparse.ArgumentParser()
    parser.add_argument("--analyze_lengths", action="store_true",
                        help="Print audio length truncation table and exit.")
    parser.add_argument("--cache_audio", action="store_true",
                        help="Pre-compute and save waveform .pt cache files.")
    args = parser.parse_args()

    if cfg.iemocap_root == "/path/to/IEMOCAP_full_release":
        print("ERROR: Please edit cfg.iemocap_root in config.py before running.")
        sys.exit(1)

    make_dirs(cfg.splits_dir)

    # ── Load or build metadata ─────────────────────────────────────────────
    save_path = os.path.join(cfg.splits_dir, "iemocap_metadata.csv")
    if os.path.exists(save_path):
        df = load_metadata(save_path)
    else:
        print("=" * 60)
        print("Step 1: Building metadata from IEMOCAP files...")
        print("=" * 60)
        df = build_metadata(cfg.iemocap_root)

        print("\n" + "=" * 60)
        print("Step 2: Generating LOSO folds...")
        print("=" * 60)
        generate_loso_folds(df)

        print("\n" + "=" * 60)
        print("Step 3: Testing one fold split...")
        print("=" * 60)
        train_df, val_df, test_df = get_split_dfs(df, fold=1)
        print(f"Fold 1: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")

        print("\n" + "=" * 60)
        print("Step 4: Testing class weights (training fold)...")
        print("=" * 60)
        compute_class_weights(train_df)

        print("\n" + "=" * 60)
        print("Step 5: Testing audio loading (first utterance)...")
        print("=" * 60)
        row = df[df["audio_exists"]].iloc[0]
        waveform, success, original_samples = load_and_preprocess_audio(row["audio_path"])
        print(f"  Utterance: {row['utterance_id']}")
        print(f"  Emotion:   {row['emotion']}")
        print(f"  Waveform:  shape={waveform.shape}, orig_samples={original_samples}")

        print("\n" + "=" * 60)
        print("Step 6: Saving metadata CSV...")
        print("=" * 60)
        save_metadata(df, save_path)
        print("\n[DONE] Preprocessing verification complete.")

    # ── Optional: truncation analysis ─────────────────────────────────────
    if args.analyze_lengths:
        analyze_audio_lengths(df)
        sys.exit(0)

    # ── Optional: build audio cache ────────────────────────────────────────
    if args.cache_audio:
        cache_dir = cfg.audio_cache_dir
        if cache_dir is None:
            print("ERROR: cfg.audio_cache_dir is None. Set it in config.py first.")
            sys.exit(1)
        precompute_audio_cache(df, cache_dir)
        print(f"\n[DONE] Audio cache written to: {cache_dir}")
        print("       Set cfg.audio_cache_dir in config.py to enable cache loading.")
