"""
test_orchestrator.py
─────────────────────────────────────────────────────────────────────────────
End-to-end smoke test of the full orchestrated pipeline:
  EmotionAgent (Agent 3) -> AlertAgent (Agent 5) -> SupervisorAgent (Agent 6)

Same 3 utterances as test_agent.py — fold-1 VALIDATION set (session 2),
not in the kNN index (built from training sessions 3/4/5).

Run from project root:
    python agentic_ai/test_orchestrator.py
"""

import sys
import pathlib

_PROJECT_ROOT = pathlib.Path(__file__).parent.parent
sys.path.insert(0, str(_PROJECT_ROOT / "src"))
sys.path.insert(0, str(_PROJECT_ROOT))

from agentic_ai.orchestrator import SupervisorAgent  # noqa: E402

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
    pred  = result.emotion.predicted_emotion
    conf  = max(result.emotion.class_probabilities.values())
    match = "correct" if pred == truth else "WRONG"

    print(f"\n{'='*70}")
    print(f"  {uid}  |  ground truth: {truth}")
    print(f"{'='*70}")

    # ── Classification ────────────────────────────────────────────────────
    print(f"  Predicted : {pred} ({match}, conf {conf:.1%})")
    print()
    print("  Class probabilities:")
    for emotion, prob in sorted(result.emotion.class_probabilities.items(),
                                key=lambda x: -x[1]):
        bar = "#" * int(prob * 30)
        print(f"    {emotion:>10}  {prob:5.1%}  {bar}")

    # ── Retrieval ─────────────────────────────────────────────────────────
    print()
    print(f"  Retrieved examples (top-{len(result.emotion.retrieved_examples)}):")
    for i, ex in enumerate(result.emotion.retrieved_examples, 1):
        print(f"    {i}. [{ex.emotion:>10}]  sim={ex.similarity:.3f}  "
              f'"{ex.text[:60]}{"..." if len(ex.text) > 60 else ""}"')

    # ── Alert decision ────────────────────────────────────────────────────
    print()
    alert_tag = "** ALERT **" if result.alert.alert else "no alert"
    print(f"  Alert     : {alert_tag}")
    print(f"  Agreement : {result.alert.prediction_agreement:.0%}  "
          f"({result.alert.agreement_count}/{result.alert.k} neighbors match prediction)")
    if result.alert.majority_neighbor_label:
        print(f"  Majority neighbor label: {result.alert.majority_neighbor_label}")

    # ── Explanations ──────────────────────────────────────────────────────
    print()
    print(f"  Emotion explanation : {result.emotion.explanation}")
    print(f"  Alert explanation   : {result.alert.explanation}")

    # ── One-line summary ──────────────────────────────────────────────────
    print()
    print(f"  Summary: {result.summary()}")


def main():
    print("[test_orchestrator] Initialising SupervisorAgent ...")
    agent = SupervisorAgent(fold=1)

    n_correct = 0
    n_alert   = 0
    for utt in VAL_UTTERANCES:
        result = agent(audio_path=utt["audio_path"], text=utt["text"])
        print_result(utt, result)
        if result.emotion.predicted_emotion == utt["ground_truth"]:
            n_correct += 1
        if result.alert.alert:
            n_alert += 1

    print(f"\n{'='*70}")
    print(f"  {n_correct}/{len(VAL_UTTERANCES)} predictions correct")
    print(f"  {n_alert}/{len(VAL_UTTERANCES)} alerts fired")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
