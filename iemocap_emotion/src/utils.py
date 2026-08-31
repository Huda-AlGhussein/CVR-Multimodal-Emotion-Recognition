"""
utils.py
─────────────────────────────────────────────────────────────────────────────
Utility functions used across the whole pipeline.

WHY THIS FILE EXISTS
────────────────────
These are small helper functions that don't belong in any single module.
Keeping them here avoids copy-pasting the same code into multiple files.

CONTENTS
────────
  set_seed()          — makes your results reproducible across runs
  get_device()        — detects GPU/CPU automatically
  make_dirs()         — creates output folders if they don't exist
  get_logger()        — returns a configured logger for a given file/experiment
  count_parameters()  — prints how many parameters are trainable (LoRA check)
"""

import os
import random
import logging
import numpy as np
import torch


def set_seed(seed: int = 42):
    """
    WHAT IT DOES
    ────────────
    Sets the random seed for Python's built-in random module, NumPy, and
    PyTorch (both CPU and CUDA). Also makes CUDA operations deterministic.

    WHY IT IS NEEDED
    ────────────────
    Deep learning involves many random operations: weight initialisation,
    dropout, data shuffling. Without fixing the seed, two runs of exactly
    the same code produce slightly different results. You cannot reproduce
    your paper results without this.

    WHAT INPUT IT EXPECTS
    ─────────────────────
    seed: any integer. 42 is the default; use the value in config.py.

    WHAT OUTPUT IT PRODUCES
    ───────────────────────
    No return value. Sets global state in Python, NumPy, and PyTorch.

    COMMON MISTAKES
    ───────────────
    - Setting the seed AFTER creating the model: weight initialisation will
      still be random. Always call set_seed() before any model creation.
    - Setting only torch.manual_seed() and forgetting NumPy or Python random:
      data augmentation and sampling steps use NumPy/random and will vary.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # for multi-GPU setups

    # Makes CUDA convolution operations deterministic.
    # Slight speed cost but required for reproducibility.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False  # set True only if input size is fixed


def get_device() -> torch.device:
    """
    WHAT IT DOES
    ────────────
    Returns a torch.device object pointing to the best available hardware:
    CUDA GPU → Apple MPS → CPU, in that order.

    WHY IT IS NEEDED
    ────────────────
    Your code should work on a GPU server, your laptop (MPS), or CPU
    without changing any code. This function handles the detection.

    WHAT OUTPUT IT PRODUCES
    ───────────────────────
    A torch.device. Use it as: model = model.to(get_device())
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        gpu_name = torch.cuda.get_device_name(0)
        print(f"[Device] Using GPU: {gpu_name}")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("[Device] Using Apple MPS (Metal)")
    else:
        device = torch.device("cpu")
        print("[Device] WARNING: No GPU found. Running on CPU. This will be very slow.")
    return device


def make_dirs(*dirs):
    """
    WHAT IT DOES
    ────────────
    Creates one or more directories if they do not already exist.

    WHY IT IS NEEDED
    ────────────────
    Python raises FileNotFoundError if you try to save a file to a
    directory that does not exist. Creating directories upfront prevents
    this error.

    EXAMPLE
    ───────
    make_dirs("./outputs", "./checkpoints", "./logs")
    """
    for d in dirs:
        os.makedirs(d, exist_ok=True)


def get_logger(name: str, log_file: str = None) -> logging.Logger:
    """
    WHAT IT DOES
    ────────────
    Returns a logger that prints to the console AND optionally writes to
    a log file. Every log message includes a timestamp.

    WHY IT IS NEEDED
    ────────────────
    Using print() everywhere makes it hard to distinguish which part of the
    code produced which message. A logger adds timestamps and can write to
    a file so you can review what happened after the job finishes.

    WHAT INPUT IT EXPECTS
    ─────────────────────
    name:     string identifier for this logger (e.g. "train", "preprocess")
    log_file: optional path to a .log file. If None, only console output.

    EXAMPLE
    ───────
    logger = get_logger("train", "./logs/fold1_train.log")
    logger.info("Epoch 1 complete. Val Mac-F1: 0.612")
    """
    logger = logging.getLogger(name)

    # Avoid adding duplicate handlers if get_logger is called twice.
    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)

    # Format: 2024-01-15 14:32:01 | train | INFO: Epoch 1 complete.
    fmt = logging.Formatter(
        "%(asctime)s | %(name)s | %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # Console handler — always active
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(fmt)
    logger.addHandler(console_handler)

    # File handler — only if log_file is given
    if log_file is not None:
        make_dirs(os.path.dirname(log_file))
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(fmt)
        logger.addHandler(file_handler)

    return logger


def count_parameters(model: torch.nn.Module, verbose: bool = True) -> dict:
    """
    WHAT IT DOES
    ────────────
    Counts total and trainable parameters in a PyTorch model.
    Prints a summary table.

    WHY IT IS NEEDED
    ────────────────
    One of the core claims of this paper is that LoRA reduces trainable
    parameters by >90% vs full fine-tuning. You need to verify this claim
    with actual numbers from your model. This function provides those numbers
    and is the basis for the "Trainable Params" column in Table EX-11.

    WHAT INPUT IT EXPECTS
    ─────────────────────
    model:   any PyTorch nn.Module (after LoRA has been applied)
    verbose: if True, print a formatted summary

    WHAT OUTPUT IT PRODUCES
    ───────────────────────
    dict with keys:
      "total"      → total parameter count
      "trainable"  → trainable parameter count (these are updated by the optimizer)
      "frozen"     → frozen parameter count (LoRA keeps these fixed)
      "pct_trainable" → percentage of parameters that are trainable

    EXAMPLE OUTPUT (RoBERTa-base with LoRA r=8)
    ───────────────────────────────────────────
    ┌─────────────────────────┬──────────────┐
    │ Total parameters        │  125,243,910 │
    │ Trainable parameters    │    2,359,296 │
    │ Frozen parameters       │  122,884,614 │
    │ % Trainable             │        1.88% │
    └─────────────────────────┴──────────────┘
    → If % Trainable is near 100%, LoRA was not applied correctly.
    """
    total_params     = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params    = total_params - trainable_params
    pct_trainable    = 100.0 * trainable_params / total_params if total_params > 0 else 0.0

    if verbose:
        print("\n" + "─" * 45)
        print(f"{'Parameter Count Summary':^45}")
        print("─" * 45)
        print(f"  {'Total parameters':<25} {total_params:>15,}")
        print(f"  {'Trainable parameters':<25} {trainable_params:>15,}")
        print(f"  {'Frozen parameters':<25} {frozen_params:>15,}")
        print(f"  {'% Trainable':<25} {pct_trainable:>14.2f}%")
        print("─" * 45 + "\n")

    return {
        "total":          total_params,
        "trainable":      trainable_params,
        "frozen":         frozen_params,
        "pct_trainable":  pct_trainable,
    }


def seed_worker(worker_id: int):
    """
    WHAT IT DOES
    ────────────
    Worker initialisation function for PyTorch DataLoader workers.

    WHY IT IS NEEDED
    ────────────────
    Each DataLoader worker process has its own random state, which is
    independent of the main process seed. Without this function, data
    loading order can vary between runs even with set_seed() called.
    Pass this to DataLoader(worker_init_fn=seed_worker).
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
