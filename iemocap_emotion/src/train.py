"""
train.py
─────────────────────────────────────────────────────────────────────────────
Main training script. Runs the full LOSO cross-validation experiment for
one model type: text-only, audio-only, or fusion.

HOW TO RUN
──────────
  # Text-only model (all 5 folds):
  python train.py --model text

  # Audio-only model (all 5 folds):
  python train.py --model audio

  # Fusion model (all 5 folds):
  python train.py --model fusion

  # Run only fold 1 (to verify the pipeline works before running all folds):
  python train.py --model text --fold 1

WHAT HAPPENS WHEN YOU RUN THIS SCRIPT
──────────────────────────────────────
1. Parse IEMOCAP or load saved metadata CSV.
2. For each fold (1–5):
   a. Split metadata into train / val / test DataFrames.
   b. Build PyTorch Datasets and DataLoaders.
   c. Instantiate the model and apply LoRA.
   d. Compute class weights from training split.
   e. Run training loop (up to 30 epochs):
      - Forward pass → compute loss → backward → update LoRA params
      - After each epoch: evaluate on validation set
      - Save checkpoint if validation Macro F1 improved
      - Early stopping if validation Macro F1 does not improve for 5 epochs
   f. Load best checkpoint and evaluate on test set.
   g. Save test predictions, metrics, and confusion matrix.
3. After all folds: aggregate results and print the 5-fold summary table.

WHAT FILES ARE WRITTEN
──────────────────────
For each model × fold:
  checkpoints/{model}/fold{k}_best.pt     ← model weights (best val Mac-F1)
  outputs/{model}/fold{k}_test_metrics.json
  outputs/{model}/fold{k}_test_predictions.csv
  outputs/{model}/fold{k}_confusion_matrix.npy
  logs/{model}/fold{k}_train.log

After all folds:
  outputs/{model}/all_folds_summary.json  ← mean ± std across 5 folds
"""

import os
import json
import argparse
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.cuda.amp import GradScaler, autocast
from transformers import get_linear_schedule_with_warmup

from config import cfg, NUM_CLASSES, IDX_TO_LABEL
from utils import set_seed, get_device, make_dirs, get_logger, count_parameters
from data_preprocessing import (
    build_metadata, load_metadata, save_metadata,
    get_split_dfs, compute_class_weights, generate_loso_folds
)
from meld_preprocessing import load_meld_splits
from dataset import TextDataset, AudioDataset, MultimodalDataset, build_dataloader
from models_text import RoBERTaLoRA
from models_audio import WavLMLoRA
from models_fusion import CrossModalAttentionFusion
from metrics import compute_metrics, format_metrics_for_log, aggregate_fold_results, print_fold_summary


# ─────────────────────────────────────────────────────────────────────────────
# MODEL FACTORY
# ─────────────────────────────────────────────────────────────────────────────

def build_model(model_type: str) -> nn.Module:
    """
    WHAT IT DOES
    ────────────
    Creates and returns the correct model based on the --model argument.

    WHY A FACTORY FUNCTION?
    ────────────────────────
    Centralises model creation. The training loop does not need to know
    which model it is training; it just calls build_model() and gets back
    a standard nn.Module.

    WHAT INPUT IT EXPECTS
    ─────────────────────
    model_type: "text", "audio", or "fusion"

    WHAT OUTPUT IT PRODUCES
    ───────────────────────
    An nn.Module with LoRA adapters already applied.
    """
    if model_type == "text":
        return RoBERTaLoRA()
    elif model_type == "audio":
        return WavLMLoRA()
    elif model_type == "fusion":
        return CrossModalAttentionFusion()
    else:
        raise ValueError(f"Unknown model_type: {model_type}. Choose: text, audio, fusion")


def build_dataset(model_type: str, df: pd.DataFrame):
    """
    Returns the correct Dataset class for the given model type.
    text → TextDataset, audio → AudioDataset, fusion → MultimodalDataset.
    """
    if model_type == "text":
        return TextDataset(df)
    elif model_type == "audio":
        return AudioDataset(df)
    elif model_type == "fusion":
        return MultimodalDataset(df)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")


# ─────────────────────────────────────────────────────────────────────────────
# TRAINING LOOP (ONE EPOCH)
# ─────────────────────────────────────────────────────────────────────────────

def train_one_epoch(
    model:      nn.Module,
    loader:     torch.utils.data.DataLoader,
    optimizer:  torch.optim.Optimizer,
    scheduler:  object,
    criterion:  nn.Module,
    device:     torch.device,
    model_type: str,
    logger,
    scaler:     GradScaler = None,
) -> dict:
    """
    WHAT IT DOES
    ────────────
    Runs one full pass over the training DataLoader:
    forward pass → compute loss → backward pass → update parameters.

    WHY IT IS A SEPARATE FUNCTION
    ──────────────────────────────
    Separating train/val/test loops makes the code easier to read, debug,
    and modify. The train loop is different from val/test because:
      - Training requires gradient computation (model.train() mode)
      - Training updates model parameters (optimizer.step())
      - Training has gradient clipping and a scheduler step

    WHAT INPUT IT EXPECTS
    ─────────────────────
    model:      the nn.Module (RoBERTaLoRA, WavLMLoRA, or CrossModalAttentionFusion)
    loader:     training DataLoader
    optimizer:  AdamW optimizer
    scheduler:  linear warmup + decay LR scheduler
    criterion:  CrossEntropyLoss (with class weights)
    device:     torch.device (cuda or cpu)
    model_type: "text", "audio", or "fusion"
    logger:     logging.Logger for progress messages

    WHAT OUTPUT IT PRODUCES
    ───────────────────────
    dict with:
      "loss"        → mean training loss over all batches
      "macro_f1"    → Macro F1 computed over all training predictions
      (plus other metrics from compute_metrics)

    ABOUT model.train() and model.eval()
    ─────────────────────────────────────
    model.train() activates dropout layers (random neurons are zeroed).
      → Used ONLY during the training loop.
    model.eval()  deactivates dropout and batch norm updates.
      → Used during validation and testing.
    NEVER forget to switch between these modes. Forgetting model.eval()
    during validation causes dropout to randomly zero neurons, making val
    loss higher and less stable than it should be.

    ABOUT torch.no_grad()
    ──────────────────────
    Normally PyTorch tracks every tensor operation to enable backpropagation.
    During validation and testing we do not need gradients, so wrapping in
    torch.no_grad() saves memory and speeds up inference significantly.
    We do NOT use torch.no_grad() during training (we need gradients).
    """
    model.train()

    total_loss  = 0.0
    all_preds   = []
    all_labels  = []
    n_batches   = 0

    # Timing accumulators (wall-clock seconds)
    t_data = 0.0   # time waiting for next batch from DataLoader
    t_fwd  = 0.0   # time for forward pass + loss
    t_bwd  = 0.0   # time for backward + optimizer step

    use_amp = (scaler is not None) and device.type == "cuda"

    t_batch_start = time.perf_counter()

    for batch_idx, batch in enumerate(loader):
        t_data += time.perf_counter() - t_batch_start

        labels = batch["label"].to(device, non_blocking=True)

        # ── Forward pass (wrapped in autocast when AMP is active) ──────────
        # autocast casts eligible ops to fp16 automatically, using Tensor Cores.
        # Operations that require fp32 (softmax, layer norm) are kept in fp32.
        # This is numerically equivalent to fp32 within typical tolerances.
        t0 = time.perf_counter()
        with autocast(enabled=use_amp):
            if model_type == "text":
                logits = model(
                    input_ids=batch["input_ids"].to(device, non_blocking=True),
                    attention_mask=batch["attention_mask"].to(device, non_blocking=True),
                )
            elif model_type == "audio":
                logits = model(
                    waveform=batch["waveform"].to(device, non_blocking=True),
                    audio_mask=batch["audio_mask"].to(device, non_blocking=True),
                )
            elif model_type == "fusion":
                logits = model(
                    input_ids=batch["input_ids"].to(device, non_blocking=True),
                    attention_mask=batch["attention_mask"].to(device, non_blocking=True),
                    waveform=batch["waveform"].to(device, non_blocking=True),
                    audio_mask=batch["audio_mask"].to(device, non_blocking=True),
                )
            loss = criterion(logits, labels)
        t_fwd += time.perf_counter() - t0

        # ── Backward + update ──────────────────────────────────────────────
        # With AMP: GradScaler multiplies the loss by a large scale factor to
        # prevent fp16 underflow in the backward pass, then unscales before
        # the optimizer step. clip_grad_norm_ must run on the unscaled gradients.
        t0 = time.perf_counter()
        optimizer.zero_grad()

        if use_amp:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)   # unscale before gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.gradient_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.gradient_clip)
            optimizer.step()

        scheduler.step()
        t_bwd += time.perf_counter() - t0

        preds = logits.detach().argmax(dim=1).cpu().tolist()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().tolist())
        total_loss += loss.item()
        n_batches  += 1

        if (batch_idx + 1) % 50 == 0:
            logger.info(
                f"  Batch {batch_idx+1}/{len(loader)} | loss={loss.item():.4f} | "
                f"data={t_data:.1f}s  fwd={t_fwd:.1f}s  bwd={t_bwd:.1f}s"
            )

        t_batch_start = time.perf_counter()

    mean_loss = total_loss / n_batches
    metrics   = compute_metrics(all_labels, all_preds)
    metrics["loss"]   = mean_loss
    metrics["t_data"] = t_data
    metrics["t_fwd"]  = t_fwd
    metrics["t_bwd"]  = t_bwd
    return metrics


# ─────────────────────────────────────────────────────────────────────────────
# EVALUATION LOOP (validation or test)
# ─────────────────────────────────────────────────────────────────────────────

def evaluate(
    model:      nn.Module,
    loader:     torch.utils.data.DataLoader,
    criterion:  nn.Module,
    device:     torch.device,
    model_type: str,
) -> tuple:
    """
    WHAT IT DOES
    ────────────
    Runs one full pass over a DataLoader WITHOUT computing gradients.
    Used for both validation (during training) and final test evaluation.

    WHY NO GRADIENTS?
    ──────────────────
    Gradient computation uses memory and compute that we do not need
    for evaluation. torch.no_grad() disables it.

    WHAT INPUT IT EXPECTS
    ─────────────────────
    Same as train_one_epoch(), minus optimizer and scheduler.
    The loader can be a val loader or test loader.

    WHAT OUTPUT IT PRODUCES
    ───────────────────────
    (metrics, all_preds, all_labels, all_utt_ids):
      metrics:     dict from compute_metrics()
      all_preds:   list of predicted label integers
      all_labels:  list of true label integers
      all_utt_ids: list of utterance ID strings (for saving predictions)
    """
    model.eval()

    total_loss  = 0.0
    all_preds   = []
    all_labels  = []
    all_utt_ids = []
    n_batches   = 0
    use_amp     = cfg.use_amp and device.type == "cuda"

    t_val_start = time.perf_counter()

    with torch.no_grad():
        for batch in loader:
            labels = batch["label"].to(device, non_blocking=True)

            with autocast(enabled=use_amp):
                if model_type == "text":
                    logits = model(
                        input_ids=batch["input_ids"].to(device, non_blocking=True),
                        attention_mask=batch["attention_mask"].to(device, non_blocking=True),
                    )
                elif model_type == "audio":
                    logits = model(
                        waveform=batch["waveform"].to(device, non_blocking=True),
                        audio_mask=batch["audio_mask"].to(device, non_blocking=True),
                    )
                elif model_type == "fusion":
                    logits = model(
                        input_ids=batch["input_ids"].to(device, non_blocking=True),
                        attention_mask=batch["attention_mask"].to(device, non_blocking=True),
                        waveform=batch["waveform"].to(device, non_blocking=True),
                        audio_mask=batch["audio_mask"].to(device, non_blocking=True),
                    )
                loss = criterion(logits, labels)

            preds = logits.argmax(dim=1).cpu().tolist()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().tolist())
            all_utt_ids.extend(batch["utterance_id"])
            total_loss += loss.item()
            n_batches  += 1

    mean_loss = total_loss / n_batches
    metrics   = compute_metrics(all_labels, all_preds)
    metrics["loss"]   = mean_loss
    metrics["t_eval"] = time.perf_counter() - t_val_start

    return metrics, all_preds, all_labels, all_utt_ids


# ─────────────────────────────────────────────────────────────────────────────
# SAVE AND LOAD CHECKPOINTS
# ─────────────────────────────────────────────────────────────────────────────

def save_checkpoint(model: nn.Module, path: str, metadata: dict = None):
    """
    WHAT IT DOES
    ────────────
    Saves the model's state_dict to a .pt file.
    Optionally saves metadata (epoch, val Mac-F1, etc.) alongside.

    WHY state_dict AND NOT THE WHOLE MODEL?
    ────────────────────────────────────────
    torch.save(model) saves the entire model object, which is fragile:
    it breaks if you rename a class or change the module structure.

    torch.save(model.state_dict()) saves only the weight tensors by name.
    Loading requires you to instantiate the model first, then load weights.
    This is the standard and recommended approach.

    NOTE ON LORA AND state_dict
    ────────────────────────────
    The PEFT library stores LoRA weights inside the wrapped model.
    model.state_dict() includes BOTH frozen base model weights AND
    trainable LoRA weights. For inference/paper results, save the full
    state_dict. For sharing only the LoRA weights (much smaller), use
    model.save_pretrained() from PEFT.
    """
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "metadata":         metadata or {},
    }
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(checkpoint, path)


def load_checkpoint(model: nn.Module, path: str, device: torch.device):
    """
    Loads a checkpoint and updates the model in-place.
    Returns the metadata dict stored alongside the checkpoint.
    """
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    return checkpoint.get("metadata", {})


# ─────────────────────────────────────────────────────────────────────────────
# SAVE PREDICTIONS TO CSV
# ─────────────────────────────────────────────────────────────────────────────

def save_predictions(
    all_utt_ids: list,
    all_labels:  list,
    all_preds:   list,
    path:        str,
):
    """
    WHAT IT DOES
    ────────────
    Saves per-sample predictions to a CSV file for post-hoc analysis.

    WHY IT IS NEEDED
    ────────────────
    Having per-sample predictions allows you to:
      - Compute additional metrics later without rerunning the model.
      - Run McNemar's test (comparing two models' per-sample errors).
      - Build error analysis tables showing which utterances were misclassified.
      - Provide supplementary material for the paper.

    WHAT THE OUTPUT CSV LOOKS LIKE
    ────────────────────────────────
    utterance_id           | true_label_idx | true_label | pred_label_idx | pred_label | correct
    Ses01F_impro01_F000    | 4              | neutral    | 4              | neutral    | True
    Ses01F_impro01_F001    | 0              | anger      | 3              | joy        | False
    ...
    """
    rows = []
    for utt_id, true_idx, pred_idx in zip(all_utt_ids, all_labels, all_preds):
        rows.append({
            "utterance_id":    utt_id,
            "true_label_idx":  true_idx,
            "true_label":      IDX_TO_LABEL[true_idx],
            "pred_label_idx":  pred_idx,
            "pred_label":      IDX_TO_LABEL[pred_idx],
            "correct":         true_idx == pred_idx,
        })
    pd.DataFrame(rows).to_csv(path, index=False)
    print(f"[Predictions] Saved to {path}")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN TRAINING FUNCTION FOR ONE FOLD
# ─────────────────────────────────────────────────────────────────────────────

def train_fold(
    df:         pd.DataFrame,
    fold:       int,
    model_type: str,
    device:     torch.device,
    logger,
    debug:      bool = False,
) -> dict:
    """
    WHAT IT DOES
    ────────────
    Runs the complete training pipeline for one LOSO fold:
    data split → dataset → model → training loop → test evaluation.

    WHAT INPUT IT EXPECTS
    ─────────────────────
    df:         full metadata DataFrame
    fold:       int 1–5
    model_type: "text", "audio", or "fusion"
    device:     torch.device
    logger:     logging.Logger

    WHAT OUTPUT IT PRODUCES
    ───────────────────────
    A dict of test metrics (from compute_metrics()) for this fold.
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"FOLD {fold} | Model: {model_type}")
    logger.info(f"{'='*60}")

    set_seed(cfg.seed + fold)
    # We add fold to the seed so each fold has a slightly different
    # initialisation but is still reproducible.

    # ── Split data ───────────────────────────────────────────────────────────
    train_df, val_df, test_df = get_split_dfs(df, fold)
    logger.info(f"  Split sizes: train={len(train_df)} | val={len(val_df)} | test={len(test_df)}")

    # Filter to only rows where audio file exists
    if model_type in ("audio", "fusion"):
        n_before = len(train_df)
        train_df = train_df[train_df["audio_exists"]].reset_index(drop=True)
        val_df   = val_df[val_df["audio_exists"]].reset_index(drop=True)
        test_df  = test_df[test_df["audio_exists"]].reset_index(drop=True)
        n_after  = len(train_df)
        if n_before != n_after:
            logger.warning(f"  Removed {n_before - n_after} train rows with missing audio.")

    # ── Debug mode: subsample to cfg.debug_subset_size per split ────────────
    if debug:
        n = cfg.debug_subset_size
        train_df = train_df.head(n).reset_index(drop=True)
        val_df   = val_df.head(n).reset_index(drop=True)
        test_df  = test_df.head(n).reset_index(drop=True)
        logger.info(f"  [DEBUG] Subsampled to {n} per split.")

    # ── Compute class weights ─────────────────────────────────────────────────
    class_weights = compute_class_weights(train_df)  # computed on TRAIN ONLY
    logger.info(f"  Class weights: {class_weights.tolist()}")

    # ── Build datasets and dataloaders ────────────────────────────────────────
    train_ds = build_dataset(model_type, train_df)
    val_ds   = build_dataset(model_type, val_df)
    test_ds  = build_dataset(model_type, test_df)

    train_loader = build_dataloader(train_ds, cfg.batch_size, shuffle=True,
                                    num_workers=cfg.num_workers,
                                    pin_memory=cfg.pin_memory,
                                    persistent_workers=cfg.persistent_workers)
    val_loader   = build_dataloader(val_ds,   cfg.batch_size, shuffle=False,
                                    num_workers=cfg.num_workers,
                                    pin_memory=cfg.pin_memory,
                                    persistent_workers=cfg.persistent_workers)
    test_loader  = build_dataloader(test_ds,  cfg.batch_size, shuffle=False,
                                    num_workers=cfg.num_workers,
                                    pin_memory=cfg.pin_memory,
                                    persistent_workers=cfg.persistent_workers)

    logger.info(f"  Batches: train={len(train_loader)} | val={len(val_loader)} | test={len(test_loader)}")

    # ── Build model ───────────────────────────────────────────────────────────
    model = build_model(model_type).to(device)
    param_info = count_parameters(model, verbose=True)
    logger.info(f"  Trainable params: {param_info['trainable']:,} "
                f"({param_info['pct_trainable']:.2f}% of total)")

    # ── Loss function ─────────────────────────────────────────────────────────
    # CrossEntropyLoss with class weights:
    #   - weight: a tensor of shape [num_classes] with inverse-frequency weights
    #   - label_smoothing: 0.1 — replaces hard 1.0/0.0 targets with 0.9/0.1/6
    #     This prevents overconfident predictions on majority classes.
    if cfg.use_class_weights:
        criterion = nn.CrossEntropyLoss(
            weight=class_weights.to(device),
            label_smoothing=cfg.label_smoothing,
        )
    else:
        criterion = nn.CrossEntropyLoss(
            label_smoothing=cfg.label_smoothing,
        )

    # ── Optimizer ─────────────────────────────────────────────────────────────
    # AdamW: Adam with decoupled weight decay.
    # We only pass parameters with requires_grad=True (the LoRA matrices
    # and the classification head). Frozen parameters are automatically excluded
    # by PyTorch's optimizer because they have requires_grad=False.
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW(
        trainable_params,
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
    )

    # ── Learning rate scheduler ────────────────────────────────────────────────
    max_epochs  = 3 if debug else cfg.num_epochs
    total_steps = len(train_loader) * max_epochs
    scheduler   = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=cfg.warmup_steps,
        num_training_steps=total_steps,
    )

    # ── AMP GradScaler ────────────────────────────────────────────────────────
    # GradScaler maintains a dynamic scale factor for the loss to prevent
    # fp16 underflow during the backward pass. It automatically adjusts the
    # scale up or down based on whether inf/NaN gradients are detected.
    # enabled=False makes all scaler operations no-ops on CPU (safe to always create).
    scaler = GradScaler(enabled=(cfg.use_amp and device.type == "cuda"))
    logger.info(f"  AMP: {'enabled' if cfg.use_amp and device.type == 'cuda' else 'disabled'}")

    # ── Training loop ──────────────────────────────────────────────────────────
    best_val_macro_f1   = 0.0
    best_epoch          = 0
    patience_counter    = 0
    checkpoint_path     = os.path.join(
        cfg.checkpoint_dir, model_type, f"fold{fold}_best.pt"
    )

    logger.info(f"\n  Starting training for up to {max_epochs} epochs...")

    for epoch in range(1, max_epochs + 1):
        epoch_start = time.perf_counter()

        # ── Train ──────────────────────────────────────────────────────────
        train_metrics = train_one_epoch(
            model, train_loader, optimizer, scheduler, criterion,
            device, model_type, logger, scaler=scaler
        )

        # ── Validate ───────────────────────────────────────────────────────
        val_metrics, _, _, _ = evaluate(
            model, val_loader, criterion, device, model_type
        )

        epoch_time = time.perf_counter() - epoch_start

        # ── Log epoch summary with timing breakdown ────────────────────────
        logger.info(
            f"  Epoch {epoch:3d}/{max_epochs} | total={epoch_time:.0f}s | "
            f"data={train_metrics['t_data']:.0f}s  "
            f"fwd={train_metrics['t_fwd']:.0f}s  "
            f"bwd={train_metrics['t_bwd']:.0f}s  "
            f"val={val_metrics['t_eval']:.0f}s || "
            f"train_loss={train_metrics['loss']:.4f}  "
            f"train_MacF1={train_metrics['macro_f1']:.4f} | "
            f"{format_metrics_for_log(val_metrics, 'val')}"
        )

        val_macro_f1 = val_metrics["macro_f1"]

        # ── Early stopping and checkpoint saving ───────────────────────────
        if val_macro_f1 > best_val_macro_f1:
            # Improved → save checkpoint and reset patience counter
            best_val_macro_f1 = val_macro_f1
            best_epoch        = epoch
            patience_counter  = 0
            save_checkpoint(
                model, checkpoint_path,
                metadata={"fold": fold, "epoch": epoch, "val_macro_f1": val_macro_f1}
            )
            logger.info(f"  New best checkpoint saved. Val MacF1={val_macro_f1:.4f}")
        else:
            patience_counter += 1
            logger.info(f"  No improvement. Patience: {patience_counter}/{cfg.early_stop_patience}")

            if patience_counter >= cfg.early_stop_patience:
                logger.info(f"  Early stopping triggered at epoch {epoch}.")
                break

    logger.info(f"\n  Best validation MacF1: {best_val_macro_f1:.4f} at epoch {best_epoch}")

    # ── Load best checkpoint and evaluate on test set ─────────────────────────
    logger.info(f"  Loading best checkpoint from: {checkpoint_path}")
    load_checkpoint(model, checkpoint_path, device)

    test_metrics, test_preds, test_labels, test_utt_ids = evaluate(
        model, test_loader, criterion, device, model_type
    )

    logger.info(f"\n  TEST RESULTS: {format_metrics_for_log(test_metrics, 'test')}")
    logger.info("\n" + test_metrics["classification_report"])

    # ── Save test outputs ─────────────────────────────────────────────────────
    out_dir = os.path.join(cfg.output_dir, model_type)
    make_dirs(out_dir)

    # Save metrics JSON
    metrics_to_save = {k: v for k, v in test_metrics.items()
                       if k != "confusion_matrix" and k != "classification_report"}
    with open(os.path.join(out_dir, f"fold{fold}_test_metrics.json"), "w") as f:
        json.dump(metrics_to_save, f, indent=2)

    # Save predictions CSV
    save_predictions(
        test_utt_ids, test_labels, test_preds,
        os.path.join(out_dir, f"fold{fold}_test_predictions.csv")
    )

    # Save confusion matrix as numpy array
    np.save(
        os.path.join(out_dir, f"fold{fold}_confusion_matrix.npy"),
        test_metrics["confusion_matrix"]
    )

    logger.info(f"  Results saved to: {out_dir}")
    return test_metrics


# ─────────────────────────────────────────────────────────────────────────────
# MELD NATIVE-SPLIT TRAINING FUNCTION
# ─────────────────────────────────────────────────────────────────────────────

def train_native_split(
    train_df:   pd.DataFrame,
    val_df:     pd.DataFrame,
    test_df:    pd.DataFrame,
    model_type: str,
    device:     torch.device,
    logger,
    debug:      bool = False,
) -> dict:
    """
    Runs the complete training pipeline on MELD's native train/dev/test split.

    Mirrors train_fold() but takes pre-split DataFrames instead of a fold
    index. No LOSO loop — MELD's official split is used directly.

    Returns a dict of test metrics (same schema as train_fold).
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"MELD NATIVE SPLIT | Model: {model_type}")
    logger.info(f"{'='*60}")

    set_seed(cfg.seed)

    logger.info(f"  Split sizes: train={len(train_df)} | val={len(val_df)} | test={len(test_df)}")

    # Filter to rows where audio file was successfully extracted
    if model_type in ("audio", "fusion"):
        n_before = len(train_df)
        train_df = train_df[train_df["audio_exists"]].reset_index(drop=True)
        val_df   = val_df[val_df["audio_exists"]].reset_index(drop=True)
        test_df  = test_df[test_df["audio_exists"]].reset_index(drop=True)
        n_after  = len(train_df)
        if n_before != n_after:
            logger.warning(f"  Removed {n_before - n_after} train rows with missing audio.")

    if debug:
        n = cfg.debug_subset_size
        train_df = train_df.head(n).reset_index(drop=True)
        val_df   = val_df.head(n).reset_index(drop=True)
        test_df  = test_df.head(n).reset_index(drop=True)
        logger.info(f"  [DEBUG] Subsampled to {n} per split.")

    class_weights = compute_class_weights(train_df)
    logger.info(f"  Class weights: {class_weights.tolist()}")

    train_ds = build_dataset(model_type, train_df)
    val_ds   = build_dataset(model_type, val_df)
    test_ds  = build_dataset(model_type, test_df)

    train_loader = build_dataloader(train_ds, cfg.batch_size, shuffle=True,
                                    num_workers=cfg.num_workers,
                                    pin_memory=cfg.pin_memory,
                                    persistent_workers=cfg.persistent_workers)
    val_loader   = build_dataloader(val_ds,   cfg.batch_size, shuffle=False,
                                    num_workers=cfg.num_workers,
                                    pin_memory=cfg.pin_memory,
                                    persistent_workers=cfg.persistent_workers)
    test_loader  = build_dataloader(test_ds,  cfg.batch_size, shuffle=False,
                                    num_workers=cfg.num_workers,
                                    pin_memory=cfg.pin_memory,
                                    persistent_workers=cfg.persistent_workers)

    logger.info(f"  Batches: train={len(train_loader)} | val={len(val_loader)} | test={len(test_loader)}")

    model = build_model(model_type).to(device)
    param_info = count_parameters(model, verbose=True)
    logger.info(f"  Trainable params: {param_info['trainable']:,} "
                f"({param_info['pct_trainable']:.2f}% of total)")

    if cfg.use_class_weights:
        criterion = nn.CrossEntropyLoss(
            weight=class_weights.to(device),
            label_smoothing=cfg.label_smoothing,
        )
    else:
        criterion = nn.CrossEntropyLoss(label_smoothing=cfg.label_smoothing)

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW(trainable_params, lr=cfg.learning_rate, weight_decay=cfg.weight_decay)

    max_epochs  = 3 if debug else cfg.num_epochs
    total_steps = len(train_loader) * max_epochs
    scheduler   = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=cfg.warmup_steps,
        num_training_steps=total_steps,
    )

    scaler = GradScaler(enabled=(cfg.use_amp and device.type == "cuda"))
    logger.info(f"  AMP: {'enabled' if cfg.use_amp and device.type == 'cuda' else 'disabled'}")

    best_val_macro_f1 = 0.0
    best_epoch        = 0
    patience_counter  = 0
    checkpoint_path   = os.path.join(cfg.checkpoint_dir, model_type, "meld_best.pt")

    logger.info(f"\n  Starting training for up to {max_epochs} epochs...")

    for epoch in range(1, max_epochs + 1):
        epoch_start = time.perf_counter()

        train_metrics = train_one_epoch(
            model, train_loader, optimizer, scheduler, criterion,
            device, model_type, logger, scaler=scaler
        )
        val_metrics, _, _, _ = evaluate(model, val_loader, criterion, device, model_type)

        epoch_time = time.perf_counter() - epoch_start

        logger.info(
            f"  Epoch {epoch:3d}/{max_epochs} | total={epoch_time:.0f}s | "
            f"data={train_metrics['t_data']:.0f}s  "
            f"fwd={train_metrics['t_fwd']:.0f}s  "
            f"bwd={train_metrics['t_bwd']:.0f}s  "
            f"val={val_metrics['t_eval']:.0f}s || "
            f"train_loss={train_metrics['loss']:.4f}  "
            f"train_MacF1={train_metrics['macro_f1']:.4f} | "
            f"{format_metrics_for_log(val_metrics, 'val')}"
        )

        val_macro_f1 = val_metrics["macro_f1"]

        if val_macro_f1 > best_val_macro_f1:
            best_val_macro_f1 = val_macro_f1
            best_epoch        = epoch
            patience_counter  = 0
            save_checkpoint(
                model, checkpoint_path,
                metadata={"split": "meld_native", "epoch": epoch, "val_macro_f1": val_macro_f1}
            )
            logger.info(f"  New best checkpoint saved. Val MacF1={val_macro_f1:.4f}")
        else:
            patience_counter += 1
            logger.info(f"  No improvement. Patience: {patience_counter}/{cfg.early_stop_patience}")
            if patience_counter >= cfg.early_stop_patience:
                logger.info(f"  Early stopping triggered at epoch {epoch}.")
                break

    logger.info(f"\n  Best validation MacF1: {best_val_macro_f1:.4f} at epoch {best_epoch}")

    logger.info(f"  Loading best checkpoint from: {checkpoint_path}")
    load_checkpoint(model, checkpoint_path, device)

    test_metrics, test_preds, test_labels, test_utt_ids = evaluate(
        model, test_loader, criterion, device, model_type
    )

    logger.info(f"\n  TEST RESULTS: {format_metrics_for_log(test_metrics, 'test')}")
    logger.info("\n" + test_metrics["classification_report"])

    out_dir = os.path.join(cfg.output_dir, model_type)
    make_dirs(out_dir)

    metrics_to_save = {k: v for k, v in test_metrics.items()
                       if k != "confusion_matrix" and k != "classification_report"}
    with open(os.path.join(out_dir, "meld_test_metrics.json"), "w") as f:
        json.dump(metrics_to_save, f, indent=2)

    save_predictions(
        test_utt_ids, test_labels, test_preds,
        os.path.join(out_dir, "meld_test_predictions.csv")
    )

    np.save(
        os.path.join(out_dir, "meld_confusion_matrix.npy"),
        test_metrics["confusion_matrix"]
    )

    logger.info(f"  Results saved to: {out_dir}")
    return test_metrics


# ─────────────────────────────────────────────────────────────────────────────
# MAIN ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Train emotion recognition model on IEMOCAP with LOSO cross-validation."
    )
    parser.add_argument(
        "--model", type=str, required=True, choices=["text", "audio", "fusion"],
        help="Which model to train: text (RoBERTa), audio (WavLM), or fusion."
    )
    parser.add_argument(
        "--fold", type=int, default=None,
        help="Run only this fold (1-5). Default: run all 5 folds."
    )
    parser.add_argument(
        "--metadata_csv", type=str, default=None,
        help="Path to pre-built metadata CSV. If not given, rebuilds from IEMOCAP."
    )
    parser.add_argument(
        "--debug", action="store_true",
        help=(
            "Debug mode: use only cfg.debug_subset_size samples per split "
            "and cap at 3 epochs. Verifies the full pipeline runs end-to-end "
            "before committing to a full training run."
        )
    )
    parser.add_argument(
        "--max_audio_secs", type=float, default=None,
        help=(
            "Override cfg.max_audio_secs for this run. E.g. --max_audio_secs 4.0 "
            "uses 4s clips (200 WavLM frames) vs the default 6s (300 frames). "
            "Does NOT affect a pre-built cache — cache is keyed by MAX_AUDIO_SAMPLES."
        )
    )
    parser.add_argument(
        "--dataset", type=str, default="iemocap", choices=["iemocap", "meld"],
        help=(
            "Which dataset to train on. "
            "'iemocap' uses LOSO 5-fold cross-validation. "
            "'meld' uses MELD's native train/dev/test split. "
            "Both share the same 5-class label space (anger, fear, joy, neutral, sadness)."
        )
    )
    args = parser.parse_args()

    # ── Apply CLI overrides to cfg before any module uses it ────────────────
    # Dataset selection: sets split_mode so the rest of main() can branch on it.
    cfg.dataset    = args.dataset
    cfg.split_mode = "loso" if args.dataset == "iemocap" else "native"
    if args.dataset == "meld":
        # Point audio cache at the MELD-specific cache directory so IEMOCAP
        # cache files are not mixed with MELD cache files.
        cfg.audio_cache_dir = cfg.meld_audio_dir + "_cache"

    if args.max_audio_secs is not None:
        cfg.max_audio_secs = args.max_audio_secs
        # MAX_AUDIO_SAMPLES is a module-level constant; we patch it here so
        # the Dataset and model see the correct value for this run.
        import config as _config_module
        _config_module.MAX_AUDIO_SAMPLES = int(cfg.sample_rate * cfg.max_audio_secs)
        print(f"[Main] max_audio_secs overridden to {cfg.max_audio_secs}s "
              f"({_config_module.MAX_AUDIO_SAMPLES} samples)")

    # ── Setup ─────────────────────────────────────────────────────────────────
    set_seed(cfg.seed)
    device = get_device()
    make_dirs(cfg.output_dir, cfg.checkpoint_dir, cfg.log_dir, cfg.splits_dir)

    if args.debug:
        print(f"[Main] DEBUG MODE: {cfg.debug_subset_size} samples/split, max 3 epochs.")

    print(f"[Main] Dataset: {cfg.dataset} | Split mode: {cfg.split_mode} | Model: {args.model}")

    # ── MELD: native train/dev/test split ────────────────────────────────────
    if cfg.split_mode == "native":
        if cfg.meld_root == r"C:\path\to\MELD.Raw":
            print("ERROR: cfg.meld_root not set. Edit config.py (meld_root) first.")
            return

        if not os.path.isdir(cfg.meld_splits_dir):
            print(f"ERROR: MELD splits directory not found: {cfg.meld_splits_dir}")
            print("       Run: python meld_preprocessing.py --build_metadata first.")
            return

        print(f"[Main] Loading MELD splits from: {cfg.meld_splits_dir}")
        train_df, val_df, test_df = load_meld_splits(cfg.meld_splits_dir)
        print(f"[Main] MELD: train={len(train_df)} | val={len(val_df)} | test={len(test_df)}")

        log_path = os.path.join(cfg.log_dir, args.model, "meld_train.log")
        make_dirs(os.path.dirname(log_path))
        logger = get_logger("meld", log_path)

        train_native_split(
            train_df=train_df,
            val_df=val_df,
            test_df=test_df,
            model_type=args.model,
            device=device,
            logger=logger,
            debug=args.debug,
        )
        return

    # ── IEMOCAP: LOSO 5-fold cross-validation ────────────────────────────────
    # Determine which folds to run
    folds_to_run = [args.fold] if args.fold else list(range(1, 6))

    # ── Build or load metadata ────────────────────────────────────────────────
    metadata_path = args.metadata_csv or os.path.join(cfg.splits_dir, "iemocap_metadata.csv")

    if os.path.exists(metadata_path):
        print(f"[Main] Loading existing metadata from: {metadata_path}")
        df = load_metadata(metadata_path)
    else:
        print("[Main] Building metadata from IEMOCAP files...")
        if cfg.iemocap_root == "/path/to/IEMOCAP_full_release":
            print("ERROR: cfg.iemocap_root not set. Edit config.py first.")
            return
        df = build_metadata(cfg.iemocap_root)
        save_metadata(df, metadata_path)

    # ── Run LOSO folds ────────────────────────────────────────────────────────
    all_fold_results = []

    for fold in folds_to_run:
        log_path = os.path.join(cfg.log_dir, args.model, f"fold{fold}_train.log")
        make_dirs(os.path.dirname(log_path))
        logger = get_logger(f"fold{fold}", log_path)

        fold_test_metrics = train_fold(
            df=df,
            fold=fold,
            model_type=args.model,
            device=device,
            logger=logger,
            debug=args.debug,
        )
        all_fold_results.append(fold_test_metrics)

    # ── Aggregate results across folds ────────────────────────────────────────
    if len(all_fold_results) == 5:
        from metrics import aggregate_fold_results, print_fold_summary
        agg = aggregate_fold_results(all_fold_results)
        print_fold_summary(agg)

        # Save aggregate summary
        out_dir = os.path.join(cfg.output_dir, args.model)
        summary_path = os.path.join(out_dir, "all_folds_summary.json")
        summary_to_save = {k: v for k, v in agg.items()
                           if k != "confusion_matrix_normalised"}
        with open(summary_path, "w") as f:
            json.dump(summary_to_save, f, indent=2)
        print(f"\n[Main] Aggregate summary saved to: {summary_path}")
    else:
        print(f"\n[Main] Ran {len(all_fold_results)} fold(s). "
              f"Run all 5 folds to see the aggregate summary.")


if __name__ == "__main__":
    main()
