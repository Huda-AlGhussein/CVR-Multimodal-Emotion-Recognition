"""
rag.py
─────────────────────────────────────────────────────────────────────────────
KNNRetriever: nearest-neighbor retrieval over a precomputed embedding index.

DESIGN NOTE — PLACEHOLDER FOR FUTURE CVR-SPECIFIC RETRIEVAL
────────────────────────────────────────────────────────────
This retriever uses IEMOCAP training examples as the knowledge base.
This is a deliberate prototype substitute for a real CVR (cockpit voice
recorder) knowledge base, which does not yet exist pending GCAA data access.

In the final system (Agent 3 v2+), this class should be replaced or extended
with a CVR-specific retrieval corpus: labeled aviation communication excerpts,
incident-linked utterances, or a curated emotional speech database from
domain-relevant sources. The interface (build/load/retrieve) is designed to
make that swap straightforward — only the index file changes, not the agent.

RETRIEVAL MECHANISM
────────────────────
Given a query embedding z (shape [768]), compute cosine similarity against
all N stored training embeddings and return the top-k most similar examples.

Cosine similarity is chosen over Euclidean distance because it is invariant
to embedding magnitude — two utterances at different volume levels (after RMS
normalisation) will still match if their direction in embedding space is close.

At N ≈ 3 000 (IEMOCAP fold-1 training set), brute-force numpy matmul is fast
enough (~1 ms on CPU) without needing FAISS or approximate search.
"""

import numpy as np
from pathlib import Path
from typing import List

from .result import RetrievedExample


class KNNRetriever:
    """
    Loads a precomputed embedding index (built by build_index.py) and
    answers retrieve() queries at inference time.

    The index file is an .npz with four arrays:
      embeddings   : float32 [N, 768]  — L2-normalised fusion embeddings
      labels       : str [N]           — emotion class names
      utterance_ids: str [N]           — IEMOCAP utterance IDs
      texts        : str [N]           — transcript text for each example
    """

    def __init__(self, index_path: str):
        """
        Parameters
        ──────────
        index_path : path to the .npz file written by build_index.py
        """
        path = Path(index_path)
        if not path.exists():
            raise FileNotFoundError(
                f"KNN index not found at {index_path}. "
                f"Run agentic_ai/build_index.py first to build it."
            )

        data = np.load(path, allow_pickle=True)

        # [AGENT3-RAG] Index loaded here. Replace this .npz with a CVR-specific
        # corpus once GCAA data is available.
        self.embeddings    = data["embeddings"].astype(np.float32)   # [N, 768]
        self.labels        = data["labels"]                           # [N] str
        self.utterance_ids = data["utterance_ids"]                   # [N] str
        self.texts         = data["texts"]                           # [N] str

        self._n = len(self.embeddings)
        print(f"[KNNRetriever] Loaded {self._n} examples from {index_path}")

    def retrieve(self, query_embedding: np.ndarray, k: int = 5) -> List[RetrievedExample]:
        """
        Returns the k training examples most similar to query_embedding.

        Parameters
        ──────────
        query_embedding : float32 array of shape [768] (unnormalised is fine)
        k               : number of neighbors to return

        Returns
        ───────
        List of RetrievedExample, sorted by descending similarity.
        """
        # L2-normalise query for cosine similarity
        q = query_embedding.astype(np.float32)
        q_norm = q / (np.linalg.norm(q) + 1e-8)   # shape [768]

        # Cosine similarity = dot product of L2-normalised vectors.
        # self.embeddings is already L2-normalised (stored that way by build_index.py).
        # matmul shape: [N, 768] @ [768] → [N]
        similarities = self.embeddings @ q_norm    # [N]

        # Top-k indices by descending similarity
        k = min(k, self._n)
        top_indices = np.argpartition(similarities, -k)[-k:]
        top_indices = top_indices[np.argsort(similarities[top_indices])[::-1]]

        return [
            RetrievedExample(
                utterance_id=str(self.utterance_ids[i]),
                emotion=str(self.labels[i]),
                text=str(self.texts[i]),
                similarity=float(similarities[i]),
            )
            for i in top_indices
        ]
