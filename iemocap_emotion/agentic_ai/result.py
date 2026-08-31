"""
result.py
─────────────────────────────────────────────────────────────────────────────
Structured output type returned by EmotionAgent.

AgentResult bundles the classification decision, class probabilities,
retrieved nearest-neighbor evidence, and a plain-text explanation in one
place so downstream agents (Agent 4–6) can consume any subset of these
fields without re-running inference.
"""

from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class RetrievedExample:
    """One nearest-neighbor from the kNN retrieval index."""
    utterance_id: str
    emotion: str        # ground-truth label of this training example
    text: str           # transcript of this training example
    similarity: float   # cosine similarity to the query embedding (0–1)


@dataclass
class AgentResult:
    """
    Structured output from EmotionAgent.respond().

    Fields
    ──────
    predicted_emotion   : top-1 predicted class name (e.g. "anger")
    class_probabilities : softmax probability for each of the 5 classes
    retrieved_examples  : top-k nearest neighbors from the kNN index,
                          sorted by descending cosine similarity
    explanation         : human-readable summary referencing retrieved examples
    """
    predicted_emotion:   str
    class_probabilities: Dict[str, float]
    retrieved_examples:  List[RetrievedExample] = field(default_factory=list)
    explanation:         str = ""
