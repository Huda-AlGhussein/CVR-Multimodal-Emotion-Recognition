"""
agent.py
─────────────────────────────────────────────────────────────────────────────
Agent 3: Emotion Classification + RAG

Wraps the trained CrossModalAttentionFusion model as a lightweight agent
with a perceive → classify → respond interface. No external agent framework
is used — this is a plain Python class.

AGENT INTERFACE
────────────────
  agent = EmotionAgent(fold=1)
  result = agent(audio_path="path/to/utterance.wav", text="I'm so frustrated")

  result.predicted_emotion     # "anger"
  result.class_probabilities   # {"anger": 0.72, "fear": 0.03, ...}
  result.retrieved_examples    # top-k similar training utterances
  result.explanation           # "Predicted: anger. Similar to 3 training ..."

SCOPE OF THIS PROTOTYPE
────────────────────────
Single-utterance inference only. No batching, async, or orchestration hooks.
Those are deferred to Agent 6. This prototype establishes the perceive →
classify → respond contract and the RAG retrieval mechanism.

FOLD SELECTION
──────────────
The agent loads checkpoint and kNN index for a single LOSO fold (default: 1).
Fold 1 uses sessions 3, 4, 5 for training and sessions 1, 2 for test/val.
In a production system you would either ensemble across folds or pick the
fold whose test set does not overlap with the target deployment domain.
"""

import sys
import pathlib
import torch
import numpy as np
from collections import Counter
from typing import Optional

# ── src/ on sys.path so we can import pipeline modules without duplication ──
# [AGENT3-SYSPATH] This insert makes all src/ modules importable from here.
# It must come before any src/ import statements.
_SRC_DIR = pathlib.Path(__file__).parent.parent / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from transformers import RobertaTokenizer                   # noqa: E402
from config import cfg, IDX_TO_LABEL, NUM_CLASSES           # noqa: E402
from data_preprocessing import (                            # noqa: E402
    load_and_preprocess_audio,
    create_audio_attention_mask,
)
from models_fusion import CrossModalAttentionFusion         # noqa: E402

from .rag import KNNRetriever                               # noqa: E402
from .result import AgentResult, RetrievedExample           # noqa: E402

# Default locations relative to project root
_PROJECT_ROOT = pathlib.Path(__file__).parent.parent
_DEFAULT_CHECKPOINT_TEMPLATE = str(
    _PROJECT_ROOT / "src" / "checkpoints" / "fusion" / "fold{fold}_best.pt"
)
_DEFAULT_INDEX_TEMPLATE = str(
    _PROJECT_ROOT / "agentic_ai" / "cache" / "knn_fold{fold}.npz"
)


class EmotionAgent:
    """
    Lightweight agent wrapping CrossModalAttentionFusion + kNN retrieval.

    Parameters
    ──────────
    fold            : LOSO fold whose checkpoint and index to load (1–5)
    checkpoint_path : override the default checkpoint path
    index_path      : override the default kNN index path
    k               : number of nearest neighbors to retrieve
    device          : torch device string ("cuda", "cpu"); auto-detected if None
    """

    def __init__(
        self,
        fold:             int            = 1,
        checkpoint_path:  Optional[str]  = None,
        index_path:       Optional[str]  = None,
        k:                int            = 5,
        device:           Optional[str]  = None,
    ):
        self.fold = fold
        self.k    = k

        # ── Device ────────────────────────────────────────────────────────────
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        # ── Resolve paths ─────────────────────────────────────────────────────
        ckpt_path = checkpoint_path or _DEFAULT_CHECKPOINT_TEMPLATE.format(fold=fold)
        idx_path  = index_path      or _DEFAULT_INDEX_TEMPLATE.format(fold=fold)

        # ── Load fusion model ─────────────────────────────────────────────────
        # [AGENT3-MODEL-LOAD] Checkpoint path: src/checkpoints/fusion/fold{k}_best.pt
        # Not checkpoints/fusion/ (project root) — that folder holds MELD runs only.
        print(f"[EmotionAgent] Loading fold-{fold} checkpoint: {ckpt_path}")
        self.model = CrossModalAttentionFusion()
        checkpoint = torch.load(ckpt_path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        meta = checkpoint.get("metadata", {})
        if meta:
            print(f"[EmotionAgent] Checkpoint metadata: {meta}")
        self.model.to(self.device)
        self.model.eval()

        # ── Tokenizer ─────────────────────────────────────────────────────────
        self.tokenizer = RobertaTokenizer.from_pretrained(cfg.text_model_name)

        # ── kNN retriever ──────────────────────────────────────────────────────
        self.retriever = KNNRetriever(idx_path)

        print(f"[EmotionAgent] Ready on {self.device}. k={k}")

    # ──────────────────────────────────────────────────────────────────────────
    # PERCEIVE
    # ──────────────────────────────────────────────────────────────────────────

    def perceive(self, audio_path: str, text: str) -> dict:
        """
        Converts raw inputs (wav path + text) into model-ready tensors.

        [AGENT3-PERCEIVE] Entry point for sensory inputs. Audio preprocessing
        reuses src/data_preprocessing.load_and_preprocess_audio exactly as
        during training (same sample rate, RMS normalisation, padding).

        Returns a dict of tensors, all with a batch dimension of 1.
        """
        # ── Text ──────────────────────────────────────────────────────────────
        enc = self.tokenizer(
            text,
            max_length=cfg.max_text_tokens,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        input_ids      = enc["input_ids"].to(self.device)        # [1, seq]
        attention_mask = enc["attention_mask"].to(self.device)   # [1, seq]

        # ── Audio ─────────────────────────────────────────────────────────────
        from config import MAX_AUDIO_SAMPLES
        waveform, success, orig_samples = load_and_preprocess_audio(
            audio_path=audio_path,
            target_sr=cfg.sample_rate,
            max_samples=MAX_AUDIO_SAMPLES,
            rms_target_dbfs=cfg.rms_target_dbfs,
        )
        if not success:
            raise RuntimeError(f"Could not load audio: {audio_path}")

        audio_mask = create_audio_attention_mask(waveform, orig_samples)

        waveform   = waveform.unsqueeze(0).to(self.device)    # [1, max_samples]
        audio_mask = audio_mask.unsqueeze(0).to(self.device)  # [1, max_samples]

        return {
            "input_ids":      input_ids,
            "attention_mask": attention_mask,
            "waveform":       waveform,
            "audio_mask":     audio_mask,
        }

    # ──────────────────────────────────────────────────────────────────────────
    # CLASSIFY
    # ──────────────────────────────────────────────────────────────────────────

    def classify(self, tensors: dict) -> tuple:
        """
        Runs the fusion model to get logits and the penultimate embedding.

        [AGENT3-CLASSIFY] Two forward passes are avoided by calling embed()
        once and then applying the classifier head manually. This is more
        efficient than calling forward() and embed() separately.

        Returns (logits [1, NUM_CLASSES], embedding [768] numpy array).
        """
        with torch.no_grad():
            # embed() returns z [1, 768] — the pre-classifier fusion vector
            z = self.model.embed(
                tensors["input_ids"],
                tensors["attention_mask"],
                tensors["waveform"],
                tensors["audio_mask"],
            )
            # Apply classifier head to get logits — reuse the trained head,
            # don't duplicate it.
            logits = self.model.classifier(z)   # [1, NUM_CLASSES]

        embedding = z.squeeze(0).cpu().numpy()   # [768]
        return logits, embedding

    # ──────────────────────────────────────────────────────────────────────────
    # RESPOND
    # ──────────────────────────────────────────────────────────────────────────

    def respond(self, logits: torch.Tensor, embedding: np.ndarray) -> AgentResult:
        """
        Converts model outputs into a structured AgentResult.

        [AGENT3-RESPOND] Assembles the final agent output: prediction,
        probabilities, retrieved evidence, and a plain-text explanation.
        """
        # ── Probabilities and prediction ──────────────────────────────────────
        probs      = torch.softmax(logits.squeeze(0), dim=0).cpu().numpy()  # [NUM_CLASSES]
        pred_idx   = int(probs.argmax())
        pred_label = IDX_TO_LABEL[pred_idx]

        class_probs = {IDX_TO_LABEL[i]: float(probs[i]) for i in range(NUM_CLASSES)}

        # ── Retrieve nearest neighbors ────────────────────────────────────────
        # [AGENT3-RAG] kNN retrieval using cosine similarity on z embedding.
        # Retrieval corpus: IEMOCAP fold-{fold} training set.
        # PLACEHOLDER: replace corpus with CVR-specific data when available.
        neighbors = self.retriever.retrieve(embedding, k=self.k)

        # ── Build explanation ─────────────────────────────────────────────────
        label_counts = Counter(ex.emotion for ex in neighbors)
        support_parts = ", ".join(
            f"{cnt} labeled {lbl}" for lbl, cnt in label_counts.most_common()
        )
        explanation = (
            f"Predicted: {pred_label} "
            f"(confidence {class_probs[pred_label]:.1%}). "
            f"Similar to {support_parts} among top-{self.k} retrieved examples."
        )

        return AgentResult(
            predicted_emotion=pred_label,
            class_probabilities=class_probs,
            retrieved_examples=neighbors,
            explanation=explanation,
        )

    # ──────────────────────────────────────────────────────────────────────────
    # MAIN ENTRY POINT
    # ──────────────────────────────────────────────────────────────────────────

    def __call__(self, audio_path: str, text: str) -> AgentResult:
        """
        Full perceive → classify → respond pipeline for one utterance.

        Usage:
            result = agent(audio_path="Ses01F_impro01_F000.wav", text="I hate this")
        """
        tensors          = self.perceive(audio_path, text)
        logits, embedding = self.classify(tensors)
        return self.respond(logits, embedding)
