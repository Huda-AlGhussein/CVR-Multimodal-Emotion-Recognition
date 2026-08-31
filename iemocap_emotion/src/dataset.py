"""
dataset.py
─────────────────────────────────────────────────────────────────────────────
PyTorch Dataset classes for the three model types:
  1. TextDataset       — for RoBERTa text-only model
  2. AudioDataset      — for WavLM audio-only model
  3. MultimodalDataset — for the fusion model (text + audio)

WHY THIS FILE EXISTS
────────────────────
PyTorch's DataLoader needs a Dataset object that knows how to:
  - Return the total number of samples (__len__)
  - Return a single sample given an index (__getitem__)

The Dataset class is the bridge between your pandas DataFrame (metadata)
and the tensors your model receives during training.

HOW A DATASET WORKS IN PYTORCH
────────────────────────────────
1. You create a Dataset object, passing it a DataFrame (subset of the
   metadata for one split: train, val, or test).
2. The DataLoader calls __len__() to know how many samples exist.
3. The DataLoader calls __getitem__(i) for each sample.
4. __getitem__ loads the audio/text, preprocesses it, and returns a
   dict of tensors.
5. The DataLoader collates these dicts into batches automatically.

WHAT IS A BATCH?
────────────────
A batch is a collection of N samples stacked together into tensors.
If one sample has:
  input_ids: tensor of shape [128]
then a batch of 16 samples has:
  input_ids: tensor of shape [16, 128]
PyTorch's default collate_fn handles this stacking automatically.
"""

import os
import torch
import torchaudio  # noqa: F401 — imported here so AudioDataset workers can use it
from torch.utils.data import Dataset
import pandas as pd
from transformers import RobertaTokenizer

from config import cfg, MAX_AUDIO_SAMPLES, NUM_CLASSES
from data_preprocessing import load_and_preprocess_audio, get_cache_path


def _load_audio(audio_path: str, utterance_id: str) -> tuple:
    """
    Loads preprocessed waveform from cache if available, otherwise from
    the raw .wav file.

    WHY THIS EXISTS
    ────────────────
    Both AudioDataset and MultimodalDataset need the same loading logic.
    Centralising it here avoids duplicate code and makes it easy to toggle
    caching on/off.

    Cache check: cfg.audio_cache_dir must be set (not None) and the .pt file
    for this utterance at the current MAX_AUDIO_SAMPLES length must exist.
    The filename includes MAX_AUDIO_SAMPLES so changing max_audio_secs in
    config.py automatically causes a cache miss (no stale audio lengths).

    RETURNS
    ────────
    (waveform, orig_samples):
      waveform:     FloatTensor [MAX_AUDIO_SAMPLES]
      orig_samples: int — real sample count before padding (for attention mask)
    """
    if cfg.audio_cache_dir is not None:
        cache_path = get_cache_path(utterance_id, cfg.audio_cache_dir)
        if os.path.exists(cache_path):
            data = torch.load(cache_path, weights_only=True)
            return data["waveform"], data["orig_samples"]

    # Cache miss or caching disabled → load from raw .wav
    waveform, _, orig_samples = load_and_preprocess_audio(
        audio_path=audio_path,
        target_sr=cfg.sample_rate,
        max_samples=MAX_AUDIO_SAMPLES,
        rms_target_dbfs=cfg.rms_target_dbfs,
    )
    return waveform, orig_samples


class TextDataset(Dataset):
    """
    WHAT IT DOES
    ────────────
    Returns tokenized text and a label for each utterance.
    Used for the RoBERTa text-only model.

    WHAT ONE SAMPLE LOOKS LIKE (__getitem__ return value)
    ──────────────────────────────────────────────────────
    {
      "input_ids":      LongTensor [max_text_tokens],   ← token IDs
      "attention_mask": LongTensor [max_text_tokens],   ← 1=real token, 0=padding
      "label":          LongTensor []  (scalar),        ← integer 0–5
      "utterance_id":   str,                            ← for error analysis
    }

    WHY input_ids AND attention_mask?
    ─────────────────────────────────
    RoBERTa tokenizes text into integer token IDs (input_ids). Shorter
    sequences are padded to max_text_tokens with a padding token ID.
    The attention_mask tells RoBERTa which positions are real tokens (1)
    and which are padding (0), so padding does not influence attention.

    COMMON MISTAKES
    ───────────────
    - Using the wrong tokenizer: RoBERTa requires RobertaTokenizer, not
      BertTokenizer. They use different special tokens and vocabularies.
    - Not setting padding=True and truncation=True in the tokenizer call.
    - Forgetting return_tensors="pt": without this, the tokenizer returns
      Python lists instead of tensors.
    - The tokenizer returns tensors with a batch dimension [1, seq_len].
      We squeeze to [seq_len] since DataLoader adds the batch dim.
    """

    def __init__(self, df: pd.DataFrame):
        """
        INPUT
        ─────
        df: a pandas DataFrame (train, val, or test split) with columns:
              utterance_id, text, label_idx, emotion
        """
        self.df        = df.reset_index(drop=True)
        self.tokenizer = RobertaTokenizer.from_pretrained(cfg.text_tokenizer)
        # The tokenizer is downloaded from HuggingFace Hub on first use.
        # Subsequent calls use the cached version.

    def __len__(self) -> int:
        """Returns the number of utterances in this split."""
        return len(self.df)

    def __getitem__(self, idx: int) -> dict:
        """
        Returns a single sample as a dict of tensors.
        Called by DataLoader for each sample in a batch.
        """
        row = self.df.iloc[idx]

        # Tokenize the text
        # padding="max_length"  → pad all sequences to cfg.max_text_tokens
        # truncation=True       → truncate sequences longer than max_text_tokens
        # return_tensors="pt"   → return PyTorch tensors (not Python lists)
        encoding = self.tokenizer(
            row["text"],
            max_length=cfg.max_text_tokens,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        # encoding["input_ids"] has shape [1, max_text_tokens] (batch dim from tokenizer)
        # squeeze(0) removes the batch dim → [max_text_tokens]
        input_ids      = encoding["input_ids"].squeeze(0)      # [max_text_tokens]
        attention_mask = encoding["attention_mask"].squeeze(0)  # [max_text_tokens]

        # Label: scalar integer tensor
        label = torch.tensor(row["label_idx"], dtype=torch.long)

        return {
            "input_ids":      input_ids,
            "attention_mask": attention_mask,
            "label":          label,
            "utterance_id":   row["utterance_id"],   # string, not a tensor
        }


class AudioDataset(Dataset):
    """
    WHAT IT DOES
    ────────────
    Loads, preprocesses, and returns a fixed-length audio waveform tensor
    and a label for each utterance.
    Used for the WavLM audio-only model.

    WHAT ONE SAMPLE LOOKS LIKE
    ───────────────────────────
    {
      "waveform":      FloatTensor [MAX_AUDIO_SAMPLES],  ← preprocessed audio
      "audio_mask":    LongTensor  [MAX_AUDIO_SAMPLES],  ← 1=real, 0=padding
      "label":         LongTensor  []  (scalar),
      "utterance_id":  str,
    }

    ABOUT AUDIO LOADING
    ────────────────────
    Audio loading (reading .wav from disk) happens inside __getitem__, which
    is called by each DataLoader worker. This means the loading happens in
    parallel across workers, which is efficient.

    If you set num_workers=0, loading is sequential (slow for large datasets).
    If you set num_workers=4, 4 worker processes load data simultaneously.

    COMMON MISTAKES
    ───────────────
    - Large audio files and many workers can cause memory issues. If you
      get memory errors, reduce num_workers or batch_size.
    - The audio_mask must be computed BEFORE padding (based on original length).
      If you compute it after padding, all positions will be marked as real.
    """

    def __init__(self, df: pd.DataFrame):
        self.df = df.reset_index(drop=True)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict:
        row = self.df.iloc[idx]

        waveform, orig_samples = _load_audio(row["audio_path"], row["utterance_id"])

        audio_mask = torch.zeros(MAX_AUDIO_SAMPLES, dtype=torch.long)
        audio_mask[:orig_samples] = 1

        label = torch.tensor(row["label_idx"], dtype=torch.long)

        return {
            "waveform":     waveform,
            "audio_mask":   audio_mask,
            "label":        label,
            "utterance_id": row["utterance_id"],
        }


class MultimodalDataset(Dataset):
    """
    WHAT IT DOES
    ────────────
    Returns both text and audio tensors for each utterance.
    Used for the RoBERTa + WavLM attention fusion model.

    WHAT ONE SAMPLE LOOKS LIKE
    ───────────────────────────
    {
      "input_ids":      LongTensor  [max_text_tokens],
      "attention_mask": LongTensor  [max_text_tokens],
      "waveform":       FloatTensor [MAX_AUDIO_SAMPLES],
      "audio_mask":     LongTensor  [MAX_AUDIO_SAMPLES],
      "label":          LongTensor  []  (scalar),
      "utterance_id":   str,
    }

    WHY NOT INHERIT FROM TextDataset AND AudioDataset?
    ───────────────────────────────────────────────────
    Inheritance would work but would load the audio twice (once in each
    parent's __getitem__). It is cleaner and faster to combine both
    operations in one __getitem__.
    """

    def __init__(self, df: pd.DataFrame):
        self.df        = df.reset_index(drop=True)
        self.tokenizer = RobertaTokenizer.from_pretrained(cfg.text_tokenizer)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict:
        row = self.df.iloc[idx]

        # ── Text ──────────────────────────────────────────────────────────────
        encoding = self.tokenizer(
            row["text"],
            max_length=cfg.max_text_tokens,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        input_ids      = encoding["input_ids"].squeeze(0)
        attention_mask = encoding["attention_mask"].squeeze(0)

        # ── Audio ─────────────────────────────────────────────────────────────
        waveform, orig_samples = _load_audio(row["audio_path"], row["utterance_id"])

        audio_mask = torch.zeros(MAX_AUDIO_SAMPLES, dtype=torch.long)
        audio_mask[:orig_samples] = 1

        # ── Label ─────────────────────────────────────────────────────────────
        label = torch.tensor(row["label_idx"], dtype=torch.long)

        return {
            "input_ids":      input_ids,
            "attention_mask": attention_mask,
            "waveform":       waveform,
            "audio_mask":     audio_mask,
            "label":          label,
            "utterance_id":   row["utterance_id"],
        }


def build_dataloader(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool,
    num_workers: int = 2,
    pin_memory: bool = True,
    persistent_workers: bool = True,
) -> torch.utils.data.DataLoader:
    """
    WHAT IT DOES
    ────────────
    Wraps a Dataset in a DataLoader with appropriate settings.

    WHY IT IS NEEDED
    ────────────────
    The DataLoader handles:
      - Batching: collects __getitem__ outputs into batches
      - Shuffling: randomises sample order every epoch (training only)
      - Parallel loading: multiple worker processes load samples in parallel
      - Pin memory: locks batch tensors in CPU memory for faster GPU transfer

    WHAT INPUT IT EXPECTS
    ─────────────────────
    dataset:     a TextDataset, AudioDataset, or MultimodalDataset object
    batch_size:  number of samples per batch
    shuffle:     True for training, False for val/test
    num_workers: number of parallel data loading workers
    pin_memory:  True if using GPU (speeds up CPU→GPU transfer)

    COMMON MISTAKES
    ───────────────
    - shuffle=True on the test set: this makes it harder to align predictions
      with utterance IDs. Always use shuffle=False for val and test.
    - num_workers > 0 on Windows may cause errors. Use num_workers=0 if so.
    - The DataLoader drops the "utterance_id" string from the batch by default
      (strings cannot be stacked into tensors). This is expected — we handle
      it in the training loop by collecting utterance IDs separately.

    NOTE ON "utterance_id" IN BATCHES
    ───────────────────────────────────
    When the DataLoader collates samples, it stacks tensors but keeps lists
    for strings. So batch["utterance_id"] will be a list of strings like:
      ["Ses01F_impro01_F000", "Ses02M_impro03_M004", ...]
    This is correct and expected.
    """
    from utils import seed_worker
    import torch

    # Generator for reproducible shuffling
    generator = torch.Generator()
    generator.manual_seed(cfg.seed)

    # persistent_workers=True keeps worker processes alive between epochs,
    # eliminating the per-epoch spawn cost (~2–4s on Windows). Only valid
    # when num_workers > 0; force False otherwise to avoid a PyTorch error.
    _persistent = persistent_workers and num_workers > 0

    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        worker_init_fn=seed_worker,
        generator=generator,
        drop_last=False,
        persistent_workers=_persistent,
    )
