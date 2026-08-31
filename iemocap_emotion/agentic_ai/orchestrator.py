"""
orchestrator.py
─────────────────────────────────────────────────────────────────────────────
Agent 6: Supervisor / Orchestrator

Sequential coordination layer over:
  Agent 3 — EmotionAgent  (fusion classifier + kNN retrieval)
  Agent 5 — AlertAgent    (rule-based disagreement detector)

SCOPE — INTENTIONALLY MINIMAL
───────────────────────────────
Only two agents currently exist and are wired here. Agent 4 (CVR-specific
reasoning) is blocked pending GCAA data. Agents 1/2 (audio ingestion,
transcription) are not separately built as standalone classes.

No agent registry, no plugin system, no async pipeline — those abstractions
would be premature given the current 2-agent scope. If/when Agent 4 is built,
add it here explicitly rather than building speculative plumbing now.

ERROR HANDLING
───────────────
Errors from either sub-agent propagate to the caller unchanged. Silent failure
modes (catching exceptions into a "status=failed" result object) are avoided
deliberately — visible crashes are easier to diagnose in a prototype.

USAGE
──────
    from agentic_ai.orchestrator import SupervisorAgent

    agent  = SupervisorAgent(fold=1)
    result = agent(audio_path="path/to/utterance.wav", text="I can't do this anymore")

    print(result.emotion.predicted_emotion)   # "sadness"
    print(result.alert.alert)                 # True / False
    print(result.alert.explanation)
    print(result.summary())                   # single-line human-readable summary
"""

from dataclasses import dataclass
from typing import Optional

from .agent import EmotionAgent
from .alert_agent import AlertAgent
from .result import AgentResult
from .alert_agent import AlertResult


# ─────────────────────────────────────────────────────────────────────────────
# MODEL VERSION
# ─────────────────────────────────────────────────────────────────────────────

# [AGENT6-VERSION] Identifier for the fusion checkpoint this orchestrator loads.
# Update this string when the backbone architectures or adapter config change.
MODEL_VERSION = "WavLM-base+LoRA_RoBERTa-base+LoRA_fold1"


# ─────────────────────────────────────────────────────────────────────────────
# COMBINED OUTPUT TYPE
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class SupervisorResult:
    """
    Combined output from the full orchestrated pipeline.

    Fields are composed, not flattened — both sub-results are kept intact
    under their own namespaces so existing code consuming AgentResult or
    AlertResult directly continues to work unchanged.

      result.emotion   -> AgentResult  (prediction, probs, retrieved examples, explanation)
      result.alert     -> AlertResult  (alert bool, agreement score, explanation)

    Access examples:
      result.emotion.predicted_emotion
      result.emotion.class_probabilities["anger"]
      result.emotion.retrieved_examples[0].similarity
      result.alert.alert
      result.alert.prediction_agreement
      result.alert.explanation

    CVR CONTEXT FIELDS — PLACEHOLDER
    ──────────────────────────────────
    utterance_timecode : cockpit voice recorder timestamp for this utterance.
                         None until CVR audio timestamps are wired in (pending
                         GCAA data access — same blocker as Agent 4).
    flight_phase        : flight phase (e.g. "taxi", "climb", "cruise",
                         "approach") at the time of the utterance. None until
                         FDR (flight data recorder) data is available and
                         time-aligned with CVR audio.
    model_version        : identifier for the fusion checkpoint that produced
                         this result. Populated now from MODEL_VERSION.
    """
    emotion: AgentResult
    alert:   AlertResult
    utterance_timecode: Optional[str] = None
    flight_phase:       Optional[str] = None
    model_version:      str           = MODEL_VERSION

    def summary(self) -> str:
        """Single-line human-readable summary of the pipeline output."""
        conf   = max(self.emotion.class_probabilities.values())
        status = "ALERT" if self.alert.alert else "OK"
        return (
            f"[{status}] predicted={self.emotion.predicted_emotion} "
            f"conf={conf:.1%} "
            f"retrieval_agreement={self.alert.prediction_agreement:.0%} "
            f"({self.alert.agreement_count}/{self.alert.k} neighbors agree)"
        )


# ─────────────────────────────────────────────────────────────────────────────
# ORCHESTRATOR
# ─────────────────────────────────────────────────────────────────────────────

class SupervisorAgent:
    """
    Instantiates and sequences Agent 3 (EmotionAgent) and Agent 5 (AlertAgent).

    Both sub-agents are created once at init and reused across calls — the
    fusion model stays loaded in GPU memory, not reloaded per utterance.

    Parameters
    ──────────
    fold            : LOSO fold for EmotionAgent checkpoint and kNN index (1–5)
    alert_rule      : alert rule passed to AlertAgent ("graded_agreement" or "combined")
    alert_threshold : prediction-retrieval agreement threshold for AlertAgent (default 0.6)
    conf_threshold  : confidence threshold for AlertAgent Rule B (default 0.8)
    device          : torch device string; auto-detected if None
    """

    def __init__(
        self,
        fold:            int            = 1,
        alert_rule:      str            = "graded_agreement",
        alert_threshold: float          = 0.6,
        conf_threshold:  float          = 0.8,
        device:          Optional[str]  = None,
    ):
        self.fold = fold

        # [AGENT6-INIT] Sub-agents instantiated once; held for the lifetime of
        # this SupervisorAgent. EmotionAgent loads the fusion model + kNN index.
        print(f"[SupervisorAgent] Initialising pipeline (fold={fold}) ...")
        self._emotion_agent = EmotionAgent(fold=fold, device=device)
        self._alert_agent   = AlertAgent(
            rule=alert_rule,
            agree_threshold=alert_threshold,
            conf_threshold=conf_threshold,
        )
        print(f"[SupervisorAgent] Ready. "
              f"alert_rule={alert_rule}, alert_threshold={alert_threshold}")

    def __call__(self, audio_path: str, text: str) -> SupervisorResult:
        """
        Run the full pipeline on one utterance.

        Sequence:
          1. EmotionAgent.perceive()  — load and preprocess audio + text
          2. EmotionAgent.classify()  — fusion model forward pass + embedding
          3. EmotionAgent.respond()   — retrieve kNN neighbors, build AgentResult
          4. AlertAgent()             — apply alert rule to AgentResult

        Errors from either agent propagate to the caller — no silent catching.

        Parameters
        ──────────
        audio_path : path to a .wav file (16 kHz recommended; resampled if needed)
        text       : transcript of the utterance

        Returns
        ───────
        SupervisorResult with .emotion (AgentResult) and .alert (AlertResult)
        """
        # [AGENT6-CALL-EMOTION] Agent 3: classify and retrieve
        emotion_result = self._emotion_agent(audio_path=audio_path, text=text)

        # [AGENT6-CALL-ALERT] Agent 5: evaluate alert rule against Agent 3's output
        alert_result = self._alert_agent(emotion_result)

        # [AGENT6-CVR-CONTEXT] utterance_timecode and flight_phase remain None
        # until CVR/FDR data is available; model_version is known now.
        return SupervisorResult(
            emotion=emotion_result,
            alert=alert_result,
            utterance_timecode=None,
            flight_phase=None,
            model_version=MODEL_VERSION,
        )
