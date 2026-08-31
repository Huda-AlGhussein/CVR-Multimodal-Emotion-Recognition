"""
build_index.py
─────────────────────────────────────────────────────────────────────────────
Offline script: builds and caches the kNN retrieval index for Agent 3.

Run this once before using EmotionAgent. It encodes all training utterances
for a given LOSO fold through the fusion model's embed() method and saves
the resulting embeddings (L2-normalised) alongside labels and metadata.

USAGE
──────
  # From the project root:
  python agentic_ai/build_index.py --fold 1
  python agentic_ai/build_index.py --fold 1 --batch_size 8

  # Output: agentic_ai/cache/knn_fold1.npz

FOLD CONVENTION (matches src/data_preprocessing.get_split_dfs)
────────────────────────────────────────────────────────────────
  Fold k:  test = Session k,  val = Session (k%5)+1,  train = remaining 3

  Fold 1:  test=S1, val=S2, train=S3,S4,S5   ← index built from S3,S4,S5
  Fold 2:  test=S2, val=S3, train=S1,S4,S5
  ...

The retrieval index is built from the training sessions only — never test or
val — to avoid encoding utterances the fold-1 model was evaluated on.

IMPORTANT: Run from the project root directory so that relative imports and
config paths resolve correctly.
  cd c:\\path\\to\\iemocap_emotion
  python agentic_ai/build_index.py --fold 1
"""

import sys
import argparse
import pathlib
import numpy as np
import torch
from tqdm import tqdm

# ── Add src/ to sys.path before any src/ imports ──────────────────────────
# [AGENT3-SYSPATH] Same pattern as agent.py. Must come before src imports.
_PROJECT_ROOT = pathlib.Path(__file__).parent.parent
_SRC_DIR      = _PROJECT_ROOT / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from config import cfg, IDX_TO_LABEL, MAX_AUDIO_SAMPLES     # noqa: E402
from data_preprocessing import (                             # noqa: E402
    load_metadata,
    get_split_dfs,
    load_and_preprocess_audio,
    create_audio_attention_mask,
)
from models_fusion import CrossModalAttentionFusion          # noqa: E402
from transformers import RobertaTokenizer                    # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description="Build kNN index for Agent 3 RAG")
    p.add_argument("--fold",       type=int, default=1,  help="LOSO fold (1–5)")
    p.add_argument("--batch_size", type=int, default=4,  help="Inference batch size")
    p.add_argument("--device",     type=str, default=None,
                   help="torch device (default: auto)")
    return p.parse_args()


def load_model(fold: int, device: torch.device) -> CrossModalAttentionFusion:
    """
    Loads the fold-k fusion checkpoint from src/checkpoints/fusion/fold{k}_best.pt.

    [AGENT3-MODEL-LOAD] Checkpoint path is src/checkpoints/fusion/ — not the
    project-root checkpoints/ folder, which contains MELD runs only.
    """
    ckpt_path = _SRC_DIR / "checkpoints" / "fusion" / f"fold{fold}_best.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {ckpt_path}\n"
            f"Expected: src/checkpoints/fusion/fold{fold}_best.pt"
        )

    print(f"[build_index] Loading checkpoint: {ckpt_path}")
    model = CrossModalAttentionFusion()
    checkpoint = torch.load(str(ckpt_path), map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    meta = checkpoint.get("metadata", {})
    if meta:
        print(f"[build_index] Checkpoint metadata: {meta}")
    model.to(device)
    model.eval()
    return model


def encode_batch(
    model:      CrossModalAttentionFusion,
    tokenizer:  RobertaTokenizer,
    rows:       list,
    device:     torch.device,
) -> np.ndarray:
    """
    Runs embed() on a list of metadata rows and returns a float32 array [B, 768].
    Returns None for rows where audio loading fails (caller skips them).
    """
    # ── Text ──────────────────────────────────────────────────────────────────
    texts = [r["text"] for r in rows]
    enc = tokenizer(
        texts,
        max_length=cfg.max_text_tokens,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    input_ids      = enc["input_ids"].to(device)       # [B, seq]
    attention_mask = enc["attention_mask"].to(device)  # [B, seq]

    # ── Audio ─────────────────────────────────────────────────────────────────
    # cfg.audio_cache_dir is set to the absolute src/ path below main() so the
    # cache loader finds the pre-built .pt files instead of reading raw .wav.
    waveforms  = []
    audio_masks = []
    valid_mask = []

    for r in rows:
        # Try audio cache first (same logic as dataset._load_audio)
        cache_hit = False
        if cfg.audio_cache_dir is not None:
            from data_preprocessing import get_cache_path
            cp = get_cache_path(r["utterance_id"], cfg.audio_cache_dir)
            if pathlib.Path(cp).exists():
                data = torch.load(cp, weights_only=True)
                wf, orig = data["waveform"], data["orig_samples"]
                cache_hit = True

        if not cache_hit:
            wf, success, orig = load_and_preprocess_audio(
                r["audio_path"],
                target_sr=cfg.sample_rate,
                max_samples=MAX_AUDIO_SAMPLES,
                rms_target_dbfs=cfg.rms_target_dbfs,
            )
            if not success:
                waveforms.append(None)
                audio_masks.append(None)
                valid_mask.append(False)
                continue

        am = create_audio_attention_mask(wf, orig)
        waveforms.append(wf)
        audio_masks.append(am)
        valid_mask.append(True)

    # Filter rows where audio failed
    good_indices = [i for i, v in enumerate(valid_mask) if v]
    if not good_indices:
        return None, []

    waveform_tensor = torch.stack([waveforms[i] for i in good_indices]).to(device)
    audio_mask_tensor = torch.stack([audio_masks[i] for i in good_indices]).to(device)
    input_ids_good     = input_ids[good_indices]
    attention_mask_good = attention_mask[good_indices]

    with torch.no_grad():
        z = model.embed(
            input_ids_good,
            attention_mask_good,
            waveform_tensor,
            audio_mask_tensor,
        )   # [B_good, 768]

    embeddings = z.cpu().numpy().astype(np.float32)
    return embeddings, good_indices


def main():
    args = parse_args()

    device = torch.device(
        args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"[build_index] Device: {device}  |  Fold: {args.fold}")

    # ── Point cfg.audio_cache_dir at the absolute src/ cache path ─────────────
    # config.py sets this as "./data_splits/audio_cache" (relative).
    # When running from project root, that resolves to the wrong location.
    # We override it here with the correct absolute path.
    abs_cache = str(_SRC_DIR / "data_splits" / "audio_cache")
    cfg.audio_cache_dir = abs_cache
    print(f"[build_index] Audio cache: {abs_cache}")

    # ── Load metadata and get fold-1 training split ────────────────────────────
    metadata_path = str(_SRC_DIR / "data_splits" / "iemocap_metadata.csv")
    df = load_metadata(metadata_path)

    # [AGENT3-FOLD-SPLIT] get_split_dfs replicates the exact same fold boundary
    # used during training (src/data_preprocessing.py). This guarantees that no
    # utterance in the retrieval index was part of fold-k's val or test set.
    train_df, val_df, test_df = get_split_dfs(df, fold=args.fold)
    train_df = train_df[train_df["audio_exists"]].reset_index(drop=True)
    print(f"[build_index] Fold {args.fold} training rows with audio: {len(train_df)}")

    # ── Load model and tokenizer ───────────────────────────────────────────────
    model     = load_model(args.fold, device)
    tokenizer = RobertaTokenizer.from_pretrained(cfg.text_model_name)

    # ── Encode all training examples in batches ────────────────────────────────
    all_embeddings    = []
    all_labels        = []
    all_utterance_ids = []
    all_texts         = []

    rows = train_df.to_dict("records")
    B    = args.batch_size
    n_skipped = 0

    for start in tqdm(range(0, len(rows), B), desc="Encoding"):
        batch_rows = rows[start : start + B]
        embeddings, good_indices = encode_batch(model, tokenizer, batch_rows, device)

        if embeddings is None:
            n_skipped += len(batch_rows)
            continue

        for emb, i in zip(embeddings, good_indices):
            r = batch_rows[i]
            all_embeddings.append(emb)
            all_labels.append(r["emotion"])
            all_utterance_ids.append(r["utterance_id"])
            all_texts.append(r["text"])

        n_skipped += len(batch_rows) - len(good_indices)

    print(f"[build_index] Encoded {len(all_embeddings)} examples  "
          f"({n_skipped} skipped due to audio load failures)")

    # ── L2-normalise embeddings for efficient cosine similarity at query time ──
    embeddings_array = np.stack(all_embeddings, axis=0)   # [N, 768]
    norms = np.linalg.norm(embeddings_array, axis=1, keepdims=True) + 1e-8
    embeddings_normed = (embeddings_array / norms).astype(np.float32)

    # ── Save index ─────────────────────────────────────────────────────────────
    cache_dir = _PROJECT_ROOT / "agentic_ai" / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    out_path  = cache_dir / f"knn_fold{args.fold}.npz"

    np.savez(
        str(out_path),
        embeddings=embeddings_normed,
        labels=np.array(all_labels),
        utterance_ids=np.array(all_utterance_ids),
        texts=np.array(all_texts),
    )

    print(f"[build_index] Index saved: {out_path}")
    print(f"  Shape: {embeddings_normed.shape}  |  dtype: {embeddings_normed.dtype}")

    label_counts = {}
    for lbl in all_labels:
        label_counts[lbl] = label_counts.get(lbl, 0) + 1
    print("  Class distribution in index:")
    for lbl, cnt in sorted(label_counts.items()):
        print(f"    {lbl:>10}: {cnt}")


if __name__ == "__main__":
    main()
