"""
meld_preprocessing.py
─────────────────────────────────────────────────────────────────────────────
MELD dataset preparation: tarball extraction, ffmpeg audio extraction,
metadata CSV construction, and waveform cache building.

MELD ARCHIVE STRUCTURE (after extracting MELD.Raw.tar.gz)
──────────────────────────────────────────────────────────
MELD.Raw/
  train.tar.gz          ← inner tarball → train_splits/dia{D}_utt{U}.mp4
  dev.tar.gz            ← inner tarball → dev_splits_complete/dia{D}_utt{U}.mp4
  test.tar.gz           ← inner tarball → output_repeated_splits_test/dia{D}_utt{U}.mp4
                                           (also contains ._dia{D}_utt{U}.mp4 junk files)
  train_sent_emo.csv
  dev_sent_emo.csv
  test_sent_emo.csv

RUN ORDER
─────────
  1. python meld_preprocessing.py --extract_audio     (once, ~30 min)
  2. python meld_preprocessing.py --build_metadata    (once, fast)
  3. python meld_preprocessing.py --cache_audio       (once, ~10 min)  [optional]
  4. python train.py --model audio --dataset meld

LABEL SPACE
───────────
MELD ships 7 emotions. We drop disgust and surprise so the remaining 5
map to the SAME integer indices used for IEMOCAP:
  anger=0, fear=1, joy=2, neutral=3, sadness=4

This identity of label indices means the same model head, loss function,
and metric code work unchanged for both datasets. See config.py
(meld_label_map) for the full justification.

OUTPUT DATAFRAME SCHEMA
────────────────────────
Same columns as iemocap_metadata.csv, minus "session" (not applicable):
  utterance_id  str   "dia3_utt2"
  split         str   "train" | "dev" | "test"
  emotion       str   mapped class name
  label_idx     int   0–4 (same as IEMOCAP)
  text          str   native MELD transcript (Utterance column)
  audio_path    str   absolute path to extracted .wav file
  audio_exists  bool
"""

import os
import re
import sys
import tarfile
import subprocess
import logging
import pandas as pd
import numpy as np
import torch
from pathlib import Path
from typing import Tuple

from config import cfg, LABEL_TO_IDX, IDX_TO_LABEL, MAX_AUDIO_SAMPLES
from data_preprocessing import (
    load_and_preprocess_audio,
    precompute_audio_cache,
    analyze_audio_lengths,
)
from utils import make_dirs


# ─────────────────────────────────────────────────────────────────────────────
# SPLIT CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

# Maps logical split name → (inner tarball filename, folder inside tarball, CSV filename)
# This is the ONLY place that records MELD's irregular folder naming convention.
SPLIT_CONFIG = {
    "train": {
        "tarball":   "train.tar.gz",
        "inner_dir": "train_splits",
        "csv":       "train_sent_emo.csv",
    },
    "dev": {
        "tarball":   "dev.tar.gz",
        "inner_dir": "dev_splits_complete",
        "csv":       "dev_sent_emo.csv",
    },
    "test": {
        "tarball":   "test.tar.gz",
        "inner_dir": "output_repeated_splits_test",
        "csv":       "test_sent_emo.csv",
    },
}


# ─────────────────────────────────────────────────────────────────────────────
# LOGGING SETUP
# ─────────────────────────────────────────────────────────────────────────────

def _get_extraction_logger(log_path: str) -> logging.Logger:
    """
    Returns a logger that writes to both the console and a persistent log file.

    WHY A PERSISTENT LOG FILE
    ──────────────────────────
    ffmpeg failures during audio extraction silently shrink the training set.
    A persistent log file lets you audit exactly which utterances were skipped
    or failed after extraction completes, without re-running the whole process.
    The log file is written to log_path (e.g. meld_audio/extraction.log).
    """
    logger = logging.getLogger("meld_extraction")
    if logger.handlers:
        return logger   # already configured; avoid duplicate handlers

    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s",
                            datefmt="%Y-%m-%d %H:%M:%S")

    # Console handler
    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    # File handler — persistent across runs
    make_dirs(os.path.dirname(log_path))
    fh = logging.FileHandler(log_path, mode="a", encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    return logger


# ─────────────────────────────────────────────────────────────────────────────
# PHASE 1: EXTRACT INNER TARBALLS
# ─────────────────────────────────────────────────────────────────────────────

def extract_inner_tarballs(meld_root: str, video_root: str) -> None:
    """
    WHAT IT DOES
    ────────────
    Extracts train.tar.gz, dev.tar.gz, and test.tar.gz from meld_root
    into video_root. Each inner tarball unpacks into its own subdirectory
    (e.g. train_splits/, dev_splits_complete/, output_repeated_splits_test/).

    WHY SEPARATE FROM PHASE 2
    ──────────────────────────
    Tarball extraction is I/O-bound and takes a few minutes. ffmpeg conversion
    is CPU-bound and takes longer. Separating them means you can inspect the
    extracted .mp4 files before committing to a full ffmpeg run.

    IDEMPOTENT BEHAVIOUR
    ─────────────────────
    If the inner_dir already exists and contains at least one .mp4 file,
    extraction is skipped. Safe to re-run after a partial failure.

    WHAT INPUT IT EXPECTS
    ─────────────────────
    meld_root:  path to the top-level MELD.Raw/ folder
    video_root: path where inner tarballs are extracted (can equal meld_root)
    """
    make_dirs(video_root)

    for split_name, info in SPLIT_CONFIG.items():
        tarball_path = os.path.join(meld_root, info["tarball"])
        inner_dir    = os.path.join(video_root, info["inner_dir"])

        if not os.path.exists(tarball_path):
            print(f"[Extract] WARNING: tarball not found: {tarball_path}")
            continue

        # Check if already extracted
        existing_mp4s = list(Path(inner_dir).glob("*.mp4")) if os.path.isdir(inner_dir) else []
        if existing_mp4s:
            print(f"[Extract] '{info['inner_dir']}' already exists "
                  f"({len(existing_mp4s)} .mp4 files) — skipping extraction.")
            continue

        print(f"[Extract] Extracting {info['tarball']} → {video_root} ...")
        with tarfile.open(tarball_path, "r:gz") as tf:
            tf.extractall(video_root)

        # Count after extraction
        n = len(list(Path(inner_dir).glob("*.mp4")))
        print(f"[Extract] Done. Found {n} .mp4 files in '{info['inner_dir']}'.")


# ─────────────────────────────────────────────────────────────────────────────
# PHASE 2: FFMPEG AUDIO EXTRACTION
# ─────────────────────────────────────────────────────────────────────────────

def extract_audio_ffmpeg(
    video_root: str,
    audio_dir:  str,
    log_path:   str,
) -> dict:
    """
    WHAT IT DOES
    ────────────
    For each valid .mp4 file in all three split directories, runs ffmpeg to
    extract a 16 kHz mono .wav file. Writes a persistent log of all skipped
    and failed files so extraction failures are traceable afterward.

    JUNK FILE FILTERING
    ────────────────────
    The test split's inner tarball contains macOS resource-fork files named
    ._dia{D}_utt{U}.mp4 alongside the real clips. These are not valid video
    files and would cause ffmpeg errors. Any file whose name starts with "._"
    is skipped and logged as JUNK — not as a failure.

    IDEMPOTENT BEHAVIOUR
    ─────────────────────
    If the output .wav already exists, the file is skipped. Re-running after
    a partial failure only processes the missing files.

    FFMPEG FLAGS
    ─────────────
    -vn       : drop the video stream (audio extraction only)
    -ac 1     : force mono (MELD videos may be stereo)
    -ar 16000 : resample to 16 kHz (WavLM's expected input rate)
    -f wav    : output format

    PERSISTENT LOG
    ───────────────
    All events — SKIP (already done), JUNK (._* filtered), FAIL (ffmpeg error),
    and OK — are written to log_path (appended, not overwritten). After a full
    run you can grep the log for FAIL or JUNK to audit the training set.

    WHAT INPUT IT EXPECTS
    ─────────────────────
    video_root: directory containing inner tarball subdirectories
    audio_dir:  root directory for output .wav files; split-specific
                subdirectories (train/, dev/, test/) are created automatically
    log_path:   path to the persistent extraction log file

    WHAT OUTPUT IT PRODUCES
    ───────────────────────
    Returns a summary dict: {split: {ok, skipped, junk, failed}}
    """
    logger = _get_extraction_logger(log_path)
    logger.info("=" * 60)
    logger.info("Starting ffmpeg audio extraction")
    logger.info(f"  video_root : {video_root}")
    logger.info(f"  audio_dir  : {audio_dir}")
    logger.info("=" * 60)

    summary = {}

    for split_name, info in SPLIT_CONFIG.items():
        mp4_dir = Path(video_root) / info["inner_dir"]
        wav_dir = Path(audio_dir) / split_name
        make_dirs(str(wav_dir))

        if not mp4_dir.is_dir():
            logger.warning(f"[{split_name}] Source directory not found: {mp4_dir} "
                           f"— run --extract_audio first.")
            summary[split_name] = {"ok": 0, "skipped": 0, "junk": 0, "failed": 0}
            continue

        # Collect all .mp4 files, sorted for deterministic ordering
        all_mp4s = sorted(mp4_dir.glob("*.mp4"))
        counts   = {"ok": 0, "skipped": 0, "junk": 0, "failed": 0}

        logger.info(f"[{split_name}] {len(all_mp4s)} .mp4 files found in {mp4_dir}")

        for mp4_path in all_mp4s:
            # ── Filter: macOS resource-fork junk files ────────────────────
            # These appear in the test split as ._dia{D}_utt{U}.mp4.
            # They are binary metadata blobs, not video files. ffmpeg would
            # either error or produce a silent/corrupt .wav from them.
            if mp4_path.name.startswith("._"):
                logger.info(f"JUNK     | {mp4_path.name}")
                counts["junk"] += 1
                continue

            wav_path = wav_dir / (mp4_path.stem + ".wav")

            # ── Idempotency check ─────────────────────────────────────────
            if wav_path.exists():
                counts["skipped"] += 1
                continue

            # ── Run ffmpeg ────────────────────────────────────────────────
            cmd = [
                "ffmpeg", "-y",
                "-i",  str(mp4_path),
                "-vn",              # no video
                "-ac", "1",         # mono
                "-ar", "16000",     # 16 kHz
                "-f",  "wav",
                str(wav_path),
            ]
            try:
                result = subprocess.run(
                    cmd,
                    check=True,
                    capture_output=True,   # suppress ffmpeg console spam
                    timeout=60,            # 60-second timeout per clip
                )
                counts["ok"] += 1
                # Log individual successes only every 500 files to keep
                # the log readable; failures always logged individually.
                if counts["ok"] % 500 == 0:
                    logger.info(f"OK       | {split_name} | {counts['ok']} files done so far")

            except subprocess.CalledProcessError as e:
                # ffmpeg returned non-zero exit code
                stderr_tail = e.stderr.decode("utf-8", errors="replace")[-300:]
                logger.error(
                    f"FAIL     | {mp4_path.name} | ffmpeg exit {e.returncode} | "
                    f"stderr: {stderr_tail.strip()}"
                )
                counts["failed"] += 1

            except subprocess.TimeoutExpired:
                logger.error(f"FAIL     | {mp4_path.name} | ffmpeg timeout (>60s)")
                counts["failed"] += 1

            except FileNotFoundError:
                logger.error(
                    "FAIL     | ffmpeg not found on PATH. "
                    "Install ffmpeg and ensure it is accessible from the command line."
                )
                # Hard stop — no point continuing without ffmpeg
                raise

        logger.info(
            f"[{split_name}] DONE — "
            f"ok={counts['ok']}  skipped={counts['skipped']}  "
            f"junk={counts['junk']}  failed={counts['failed']}"
        )
        summary[split_name] = counts

    logger.info("=" * 60)
    logger.info("Extraction complete. Summary:")
    for split_name, c in summary.items():
        logger.info(f"  {split_name:6s}: ok={c['ok']:5d}  skipped={c['skipped']:5d}  "
                    f"junk={c['junk']:3d}  failed={c['failed']:3d}")
    if any(c["failed"] > 0 for c in summary.values()):
        logger.warning(
            f"  {sum(c['failed'] for c in summary.values())} files FAILED. "
            f"Check {log_path} for details. These utterances will be marked "
            f"audio_exists=False in the metadata and excluded from audio training."
        )
    logger.info("=" * 60)

    return summary


# ─────────────────────────────────────────────────────────────────────────────
# PHASE 3: BUILD METADATA DATAFRAMES
# ─────────────────────────────────────────────────────────────────────────────

def build_meld_metadata(
    meld_root: str,
    audio_dir: str,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    WHAT IT DOES
    ────────────
    Reads the three MELD CSV files, applies the label map (dropping disgust
    and surprise), constructs audio paths, and returns three DataFrames with
    the same column schema as iemocap_metadata.csv (minus "session").

    LABEL REMAPPING
    ────────────────
    MELD CSV "Emotion" column values (after .str.lower()):
      anger, disgust, fear, joy, neutral, sadness, surprise
    cfg.meld_label_map maps these to:
      anger→"anger", fear→"fear", joy→"joy", neutral→"neutral", sadness→"sadness"
      disgust→None (dropped), surprise→None (dropped)
    The resulting "emotion" strings are converted to integer label_idx via
    LABEL_TO_IDX — the SAME dictionary used for IEMOCAP.

    UTTERANCE ID FORMAT
    ────────────────────
    Constructed as "dia{Dialogue_ID}_utt{Utterance_ID}" to match the .mp4
    and .wav filenames produced by Phase 2.

    AUDIO PATH CONVENTION
    ──────────────────────
    {audio_dir}/{split}/dia{D}_utt{U}.wav
    e.g. ./data_splits/meld_audio/train/dia3_utt2.wav

    OUTPUT SCHEMA (same as IEMOCAP, "session" column absent)
    ──────────────────────────────────────────────────────────
      utterance_id | split | emotion | label_idx | text | audio_path | audio_exists
    """
    dfs = {}

    for split_name, info in SPLIT_CONFIG.items():
        csv_path = os.path.join(meld_root, info["csv"])
        if not os.path.exists(csv_path):
            raise FileNotFoundError(
                f"MELD CSV not found: {csv_path}\n"
                f"Expected in cfg.meld_root = {meld_root}"
            )

        raw = pd.read_csv(csv_path)

        # ── Construct utterance_id ──────────────────────────────────────────
        # Matches the stem of the .mp4 file:  dia{Dialogue_ID}_utt{Utterance_ID}
        raw["utterance_id"] = (
            "dia" + raw["Dialogue_ID"].astype(str) +
            "_utt" + raw["Utterance_ID"].astype(str)
        )

        # ── Apply label map ─────────────────────────────────────────────────
        # METHODOLOGICAL DECISION: .str.lower() normalises both "Anger" and
        # "anger" variants across MELD release versions.
        raw["emotion"] = raw["Emotion"].str.lower().map(cfg.meld_label_map)

        # Drop rows mapped to None (disgust, surprise)
        n_before = len(raw)
        raw = raw[raw["emotion"].notna()].copy()
        n_dropped = n_before - len(raw)
        if n_dropped:
            print(f"  [{split_name}] Dropped {n_dropped} rows "
                  f"(disgust/surprise) — {len(raw)} retained.")

        # ── Convert emotion name → integer index ────────────────────────────
        # Uses the SAME LABEL_TO_IDX as IEMOCAP (anger=0,fear=1,joy=2,neutral=3,sadness=4)
        raw["label_idx"] = raw["emotion"].map(LABEL_TO_IDX)

        if raw["label_idx"].isna().any():
            bad = raw[raw["label_idx"].isna()]["emotion"].unique().tolist()
            raise ValueError(
                f"Unknown emotion names after mapping in {split_name}: {bad}\n"
                "Check cfg.meld_label_map and LABEL_TO_IDX in config.py."
            )
        raw["label_idx"] = raw["label_idx"].astype(int)

        # ── Text (native MELD transcripts, no ASR) ─────────────────────────
        raw["text"] = raw["Utterance"].fillna("").str.strip()

        # ── Audio path ─────────────────────────────────────────────────────
        wav_dir = Path(audio_dir) / split_name
        raw["audio_path"] = raw["utterance_id"].apply(
            lambda uid: str(wav_dir / (uid + ".wav"))
        )
        raw["audio_exists"] = raw["audio_path"].apply(os.path.isfile)

        # ── Split label ─────────────────────────────────────────────────────
        # Store "dev" (not "val") to match MELD's own naming; internally in
        # train.py this becomes val_df. The file is saved as meld_dev.csv.
        raw["split"] = split_name

        # ── Select final columns (schema matches IEMOCAP minus "session") ──
        df = raw[[
            "utterance_id", "split", "emotion", "label_idx",
            "text", "audio_path", "audio_exists"
        ]].reset_index(drop=True)

        # ── Print class distribution ────────────────────────────────────────
        print(f"\n  [{split_name}] {len(df)} utterances — class distribution:")
        dist = df["emotion"].value_counts()
        for emo, count in dist.items():
            print(f"    {emo:>10}: {count:>5}  ({100*count/len(df):.1f}%)")
        n_missing = (~df["audio_exists"]).sum()
        if n_missing:
            print(f"    WARNING: {n_missing} rows have no .wav file "
                  f"(audio_exists=False). Run --extract_audio first.")

        dfs[split_name] = df

    train_df = dfs["train"]
    val_df   = dfs["dev"]    # "dev" → internal val_df
    test_df  = dfs["test"]

    return train_df, val_df, test_df


def save_meld_metadata(
    train_df: pd.DataFrame,
    val_df:   pd.DataFrame,
    test_df:  pd.DataFrame,
    splits_dir: str,
) -> None:
    """
    Saves the three DataFrames to splits_dir as:
      meld_train.csv, meld_dev.csv, meld_test.csv

    File naming uses "dev" (matching MELD's convention) even though the
    variable is val_df internally.
    """
    make_dirs(splits_dir)
    train_df.to_csv(os.path.join(splits_dir, "meld_train.csv"), index=False)
    val_df.to_csv(  os.path.join(splits_dir, "meld_dev.csv"),   index=False)
    test_df.to_csv( os.path.join(splits_dir, "meld_test.csv"),  index=False)
    print(f"\n[Metadata] Saved meld_train/dev/test.csv to {splits_dir}")


def load_meld_splits(
    splits_dir: str,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    WHAT IT DOES
    ────────────
    Loads the three pre-built MELD metadata CSVs and validates that
    label_idx values are consistent with the current config.py mapping.

    WHY THE VALIDATION
    ──────────────────
    Same reason as load_metadata() in data_preprocessing.py: if
    emotion_classes in config.py changes, stale label indices in the CSV
    would corrupt training silently. The check catches this before
    a single batch is processed.

    WHAT OUTPUT IT PRODUCES
    ───────────────────────
    (train_df, val_df, test_df) — three DataFrames ready for Dataset classes.
    val_df is loaded from meld_dev.csv (renamed to val_df internally).
    """
    paths = {
        "train": os.path.join(splits_dir, "meld_train.csv"),
        "dev":   os.path.join(splits_dir, "meld_dev.csv"),
        "test":  os.path.join(splits_dir, "meld_test.csv"),
    }

    for split_name, path in paths.items():
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"MELD metadata CSV not found: {path}\n"
                f"Run:  python meld_preprocessing.py --build_metadata"
            )

    dfs = {}
    for split_name, path in paths.items():
        df = pd.read_csv(path)

        # Validate label indices match current config
        mismatches = []
        for _, row in df.iterrows():
            idx     = int(row["label_idx"])
            emotion = row["emotion"]
            expected = IDX_TO_LABEL.get(idx)
            if expected is None or expected != emotion:
                mismatches.append((row["utterance_id"], emotion, idx, expected))
            if len(mismatches) >= 5:
                break

        if mismatches:
            print(f"\n[ERROR] meld_{split_name}.csv has stale label indices.")
            for uid, emo, idx, expected in mismatches:
                print(f"  {uid}: emotion='{emo}', label_idx={idx}, "
                      f"but current IDX_TO_LABEL[{idx}]='{expected}'")
            print(f"\n  FIX: Delete files in {splits_dir} and run "
                  f"--build_metadata to rebuild.\n")
            raise ValueError(f"Stale MELD metadata CSV: {split_name}. Delete and rebuild.")

        dfs[split_name] = df

    train_df = dfs["train"]
    val_df   = dfs["dev"]
    test_df  = dfs["test"]

    print(f"[Metadata] Loaded MELD splits from {splits_dir}:")
    print(f"  train={len(train_df)}  val={len(val_df)}  test={len(test_df)}")
    return train_df, val_df, test_df


# ─────────────────────────────────────────────────────────────────────────────
# MAIN: CLI entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    """
    Run order:
      python meld_preprocessing.py --extract_audio    ← Phases 1 + 2 (one-time, slow)
      python meld_preprocessing.py --build_metadata   ← Phase 3 (fast)
      python meld_preprocessing.py --cache_audio      ← optional waveform cache
      python meld_preprocessing.py --analyze_lengths  ← see truncation stats
    """
    import argparse

    parser = argparse.ArgumentParser(
        description="MELD dataset preparation for the SER pipeline."
    )
    parser.add_argument("--extract_audio",   action="store_true",
                        help="Extract inner tarballs and run ffmpeg on all .mp4 files.")
    parser.add_argument("--build_metadata",  action="store_true",
                        help="Parse MELD CSVs and save meld_{train,dev,test}.csv.")
    parser.add_argument("--cache_audio",     action="store_true",
                        help="Pre-compute waveform .pt cache files (optional speedup).")
    parser.add_argument("--analyze_lengths", action="store_true",
                        help="Print audio length truncation table and exit.")
    args = parser.parse_args()

    if not any(vars(args).values()):
        parser.print_help()
        sys.exit(0)

    meld_root  = cfg.meld_root
    audio_dir  = cfg.meld_audio_dir
    splits_dir = cfg.meld_splits_dir

    if meld_root == r"C:\path\to\MELD.Raw":
        print("ERROR: cfg.meld_root is not set. Edit config.py first.")
        sys.exit(1)

    # ── Phase 1 + 2: tarball extraction + ffmpeg ───────────────────────────
    if args.extract_audio:
        # video_root: where inner tarballs are extracted.
        # We extract into meld_root itself (alongside the inner .tar.gz files)
        # to keep the MELD.Raw folder self-contained.
        video_root = meld_root
        log_path   = os.path.join(audio_dir, "extraction.log")

        print("=" * 60)
        print("Phase 1: Extracting inner tarballs...")
        print("=" * 60)
        extract_inner_tarballs(meld_root=meld_root, video_root=video_root)

        print("\n" + "=" * 60)
        print("Phase 2: Extracting audio via ffmpeg...")
        print(f"  Persistent log: {log_path}")
        print("=" * 60)
        summary = extract_audio_ffmpeg(
            video_root=video_root,
            audio_dir=audio_dir,
            log_path=log_path,
        )

        total_failed = sum(c["failed"] for c in summary.values())
        if total_failed:
            print(f"\n  {total_failed} ffmpeg failures logged to: {log_path}")
            print("  These utterances will be excluded from audio training "
                  "(audio_exists=False).")

    # ── Phase 3: build metadata CSVs ──────────────────────────────────────
    if args.build_metadata:
        print("=" * 60)
        print("Phase 3: Building MELD metadata CSVs...")
        print("=" * 60)
        train_df, val_df, test_df = build_meld_metadata(
            meld_root=meld_root,
            audio_dir=audio_dir,
        )
        save_meld_metadata(train_df, val_df, test_df, splits_dir)

    # ── Optional: audio length analysis ───────────────────────────────────
    if args.analyze_lengths:
        print("=" * 60)
        print("Audio length analysis (requires --build_metadata first)...")
        print("=" * 60)
        train_df, val_df, test_df = load_meld_splits(splits_dir)
        all_df = pd.concat([train_df, val_df, test_df], ignore_index=True)
        analyze_audio_lengths(all_df)
        sys.exit(0)

    # ── Optional: waveform cache ───────────────────────────────────────────
    if args.cache_audio:
        # Override audio_cache_dir to MELD-specific path before calling
        # the shared precompute function (which reads cfg.audio_cache_dir
        # indirectly via get_cache_path).
        cfg.audio_cache_dir = "./data_splits/meld_audio_cache"

        # Recompute MAX_AUDIO_SAMPLES in case max_audio_secs was changed
        import config as _cfg_module
        _cfg_module.MAX_AUDIO_SAMPLES = int(cfg.sample_rate * cfg.max_audio_secs)

        print("=" * 60)
        print(f"Building waveform cache → {cfg.audio_cache_dir}")
        print(f"  max_audio_secs = {cfg.max_audio_secs}s  "
              f"({_cfg_module.MAX_AUDIO_SAMPLES} samples)")
        print("=" * 60)

        train_df, val_df, test_df = load_meld_splits(splits_dir)
        all_df = pd.concat([train_df, val_df, test_df], ignore_index=True)
        all_df = all_df[all_df["audio_exists"]].reset_index(drop=True)

        precompute_audio_cache(all_df, cfg.audio_cache_dir)
        print(f"\n[DONE] Cache written to: {cfg.audio_cache_dir}")

    print("\n[DONE] meld_preprocessing.py complete.")
