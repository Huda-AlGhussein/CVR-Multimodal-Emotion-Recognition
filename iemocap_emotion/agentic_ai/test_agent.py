"""
test_agent.py
─────────────────────────────────────────────────────────────────────────────
End-to-end test of EmotionAgent: perceive() -> classify() -> respond().

Utterances are drawn from IEMOCAP fold-1 VALIDATION set (session 2).
The kNN retrieval index was built from the fold-1 TRAINING set (sessions 3,4,5),
so none of these utterances appear in the index — this is the realistic
inference scenario, not a sanity check on seen data.

Run from project root:
    python agentic_ai/test_agent.py
"""

import sys
import pathlib

# ── src/ on path for config constants (audio paths use cfg.iemocap_root) ──
_PROJECT_ROOT = pathlib.Path(__file__).parent.parent
sys.path.insert(0, str(_PROJECT_ROOT / "src"))
sys.path.insert(0, str(_PROJECT_ROOT))   # makes `agentic_ai` importable as a package

from agentic_ai.agent import EmotionAgent  # noqa: E402

# ── Fold-1 validation utterances (session 2, never seen by the kNN index) ──
# Ground-truth emotion included for reference; the agent doesn't receive it.
VAL_UTTERANCES = [
    {
        "utterance_id": "Ses02F_impro01_F018",
        "ground_truth": "anger",
        "text": "Well, why didn't the D.M.V. put that you needed your birth certificate on the application then?",
        "audio_path": str(_PROJECT_ROOT / "data_splits/data/IEMOCAP_full_release"
                         / "Session2/sentences/wav/Ses02F_impro01/Ses02F_impro01_F018.wav"),
    },
    {
        "utterance_id": "Ses02F_impro03_F000",
        "ground_truth": "joy",
        "text": "Oh my God.  Guess what, guess what, guess what, guess what, guess what, guess what?",
        "audio_path": str(_PROJECT_ROOT / "data_splits/data/IEMOCAP_full_release"
                         / "Session2/sentences/wav/Ses02F_impro03/Ses02F_impro03_F000.wav"),
    },
    {
        "utterance_id": "Ses02F_impro02_F000",
        "ground_truth": "sadness",
        "text": "It's almost time for me to go.",
        "audio_path": str(_PROJECT_ROOT / "data_splits/data/IEMOCAP_full_release"
                         / "Session2/sentences/wav/Ses02F_impro02/Ses02F_impro02_F000.wav"),
    },
]


def print_result(utt: dict, result) -> None:
    uid   = utt["utterance_id"]
    truth = utt["ground_truth"]
    match = "✓" if result.predicted_emotion == truth else "✗"

    print(f"\n{'='*70}")
    print(f"  {uid}  |  ground truth: {truth}")
    print(f"{'='*70}")
    print(f"  Predicted : {result.predicted_emotion}  {match}")
    print()

    print("  Class probabilities:")
    for emotion, prob in sorted(result.class_probabilities.items(),
                                key=lambda x: -x[1]):
        bar = "█" * int(prob * 30)
        print(f"    {emotion:>10}  {prob:5.1%}  {bar}")

    print()
    print(f"  Retrieved examples (top-{len(result.retrieved_examples)}):")
    for i, ex in enumerate(result.retrieved_examples, 1):
        print(f"    {i}. [{ex.emotion:>10}]  sim={ex.similarity:.3f}  "
              f"{ex.utterance_id}  \"{ex.text[:60]}{'...' if len(ex.text)>60 else ''}\"")

    print()
    print(f"  Explanation: {result.explanation}")


def main():
    print("[test_agent] Initialising EmotionAgent (fold=1, k=5) ...")
    agent = EmotionAgent(fold=1, k=5)

    n_correct = 0
    for utt in VAL_UTTERANCES:
        result = agent(audio_path=utt["audio_path"], text=utt["text"])
        print_result(utt, result)
        if result.predicted_emotion == utt["ground_truth"]:
            n_correct += 1

    print(f"\n{'='*70}")
    print(f"  Result: {n_correct}/{len(VAL_UTTERANCES)} correct on held-out val utterances")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
