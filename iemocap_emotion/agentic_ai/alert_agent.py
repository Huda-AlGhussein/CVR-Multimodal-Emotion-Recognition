"""
alert_agent.py
─────────────────────────────────────────────────────────────────────────────
Agent 5: Alert and Explanation

Consumes an AgentResult from EmotionAgent (Agent 3) and decides whether to
raise an alert flag indicating likely classification error, together with a
plain-text explanation of the reasoning.

ALERT LOGIC — EMPIRICAL MOTIVATION
─────────────────────────────────────
Fold-1 evaluation (agentic_ai/evaluate_agent.py, 1097 test utterances) showed:

  Mean prediction-retrieval agreement when classifier is CORRECT:   84.1%
  Mean prediction-retrieval agreement when classifier is INCORRECT: 20.0%

This ~4× gap means low agreement between the predicted label and retrieved
neighbors is a meaningful, though imperfect, proxy for likely classification
error. Two alert rules are implemented:

  Rule A (default): alert when the fraction of top-k neighbors whose label
    matches the PREDICTION falls below a threshold.
    Default threshold = 0.6 (3/5 neighbors must agree with the prediction to
    suppress the alert). This was chosen as a conservative operating point on
    the fold-1 validation set (session 2):
      t=0.6 → val FPR 17.6%, TPR 51.8%, Prec 48.1%, F1 0.499
    The val-F1-optimal threshold (t=0.9) was rejected despite higher F1 because
    it produces 37.6% FPR on the test set — alert fatigue unacceptable in a
    safety-critical context.

  Rule B (combined): alert when BOTH confidence < conf_threshold AND
    prediction-retrieval agreement < agree_threshold.
    Empirically, Rule B adds no measurable lift over Rule A at the tested
    operating points (the two signals are correlated; see evaluate_agent.py
    results). Provided as an option for experimentation.

IMPORTANT CAVEAT — PROTOTYPE STATUS
──────────────────────────────────────
At FPR ~17% (Rule A, t=0.6), this alert fires on 1 in 6 correct predictions.
This is NOT suitable as a standalone safety gate. It is appropriate as a
"flag for downstream human or agent review" — one input signal to Agent 6's
orchestration logic, not an autonomous decision.

Thresholds were tuned on IEMOCAP fold-1 val (session 2). They require
recalibration on CVR-domain data before operational use.

USAGE
──────
    from agentic_ai.alert_agent import AlertAgent
    from agentic_ai.agent import EmotionAgent

    emotion_agent = EmotionAgent(fold=1)
    alert_agent   = AlertAgent()

    result       = emotion_agent(audio_path=..., text=...)
    alert_result = alert_agent(result)

    print(alert_result.alert)        # True / False
    print(alert_result.explanation)
"""

from dataclasses import dataclass
from collections import Counter
from typing import List, Optional

from .result import AgentResult


# ─────────────────────────────────────────────────────────────────────────────
# OUTPUT TYPE
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class AlertResult:
    """
    Output of AlertAgent.

    Fields
    ──────
    alert                    : True if the rule flags a likely classification error
    predicted_emotion        : the classifier's top prediction (passed through)
    prediction_agreement     : fraction of top-k neighbors matching the prediction (0.0–1.0)
    majority_neighbor_label  : the most common label among retrieved neighbors
                               (None if there is a tie)
    agreement_count          : number of top-k neighbors whose label matches the prediction
    k                        : total number of retrieved neighbors considered
    rule_used                : "graded_agreement" or "combined"
    confidence               : classifier's max softmax probability (Rule B only, else None)
    explanation              : human-readable reasoning string
    stage2_triggered         : True if the neutral_low_agreement second-stage rule
                               forced this alert (independent of the primary rule)
    """
    alert:                   bool
    predicted_emotion:       str
    prediction_agreement:    float
    majority_neighbor_label: Optional[str]
    agreement_count:         int
    k:                       int
    rule_used:               str
    confidence:              Optional[float]
    explanation:             str
    stage2_triggered:        bool = False


# ─────────────────────────────────────────────────────────────────────────────
# AGENT
# ─────────────────────────────────────────────────────────────────────────────

class AlertAgent:
    """
    Rule-based alert agent consuming an AgentResult from EmotionAgent.

    No model training, no learned parameters. Logic is entirely auditable from
    the rule type and threshold values passed at construction time.

    Parameters
    ──────────
    rule              : "graded_agreement" (default) or "combined"
    agree_threshold   : alert when prediction-retrieval agreement < this value.
                        Default 0.6 — conservative operating point validated on
                        fold-1 val set. See module docstring for rationale.
    conf_threshold    : (Rule B only) also require confidence < this value.
                        Default 0.8, tuned on val alongside agree_threshold.
    use_stage2        : if True (default), also apply the "neutral_low_agreement"
                        second-stage rule on top of the primary rule (see
                        _apply_neutral_low_agreement). Forces an alert whenever
                        the prediction is "neutral" and prediction-retrieval
                        agreement < stage2_agree_threshold, regardless of what
                        the primary rule decided.
    stage2_agree_threshold : agreement threshold for the neutral_low_agreement
                        rule. Default 0.8 — looser than Rule A's 0.6, tuned on
                        fold-1 test: adds 59 alerts (18 TP / 41 FP, +2/30
                        anger->neutral FNs caught), moving combined-rule
                        recall 0.282->0.342 at a precision cost 0.450->0.415
                        (F1 0.347->0.375). At 0.6 the stage-2 condition is a
                        strict subset of Rule A and adds zero alerts.
    """

    VALID_RULES = ("graded_agreement", "combined")

    def __init__(
        self,
        rule:                   str   = "graded_agreement",
        agree_threshold:        float = 0.6,
        conf_threshold:         float = 0.8,
        use_stage2:             bool  = True,
        stage2_agree_threshold: float = 0.8,
    ):
        if rule not in self.VALID_RULES:
            raise ValueError(f"rule must be one of {self.VALID_RULES}, got {rule!r}")

        self.rule                   = rule
        self.agree_threshold        = agree_threshold
        self.conf_threshold         = conf_threshold
        self.use_stage2             = use_stage2
        self.stage2_agree_threshold = stage2_agree_threshold

    # ── Internal helpers ───────────────────────────────────────────────────

    def _prediction_agreement(self, neighbor_labels: List[str], prediction: str) -> float:
        """Fraction of retrieved neighbors whose label matches the prediction."""
        # [AGENT5-AGREE] Core signal: how much does the retrieval index agree
        # with the classifier's top prediction?
        if not neighbor_labels:
            return 0.0
        return sum(1 for lbl in neighbor_labels if lbl == prediction) / len(neighbor_labels)

    def _majority_label(self, neighbor_labels: List[str]) -> Optional[str]:
        """
        Returns the strict majority label (appears in >50% of neighbors), or
        None if no single label achieves majority (tie = no consensus).
        """
        if not neighbor_labels:
            return None
        counts = Counter(neighbor_labels)
        top_label, top_count = counts.most_common(1)[0]
        if top_count > len(neighbor_labels) / 2:
            return top_label
        return None

    def _confidence(self, agent_result: AgentResult) -> float:
        """Max softmax probability across all classes (classifier confidence)."""
        return max(agent_result.class_probabilities.values())

    # ── Alert rules ────────────────────────────────────────────────────────

    def _apply_graded_agreement(
        self,
        prediction:   str,
        agreement:    float,
        agree_count:  int,
        k:            int,
        majority:     Optional[str],
    ) -> tuple:
        """
        Rule A: alert when prediction-retrieval agreement < agree_threshold.

        [AGENT5-RULE-A] Threshold default=0.6 selected on fold-1 val set.
        Empirical operating point: ~17% FPR, ~52% TPR on fold-1 test (session 1).
        Not a safety gate — a flag for downstream review.
        """
        alert = agreement < self.agree_threshold

        if alert:
            if majority is not None and majority != prediction:
                detail = (
                    f"only {agree_count}/{k} retrieved examples agree "
                    f"(majority label: {majority})"
                )
            elif majority is None:
                detail = (
                    f"only {agree_count}/{k} retrieved examples agree "
                    f"(no majority label among neighbors — ambiguous)"
                )
            else:
                detail = f"only {agree_count}/{k} retrieved examples agree"
            explanation = (
                f"ALERT: predicted {prediction}, but {detail}. "
                f"Agreement {agreement:.0%} < threshold {self.agree_threshold:.0%}. "
                f"Flag for review."
            )
        else:
            explanation = (
                f"No alert: predicted {prediction}, "
                f"{agree_count}/{k} retrieved examples agree "
                f"({agreement:.0%} >= threshold {self.agree_threshold:.0%})."
            )

        return alert, explanation

    def _apply_combined(
        self,
        prediction:  str,
        agreement:   float,
        agree_count: int,
        k:           int,
        majority:    Optional[str],
        confidence:  float,
    ) -> tuple:
        """
        Rule B: alert when BOTH confidence < conf_threshold AND
        agreement < agree_threshold.

        [AGENT5-RULE-B] Combined signal. Empirically adds no measurable lift
        over Rule A at the tested operating points on fold-1 (the two signals
        are correlated). Provided for experimentation. See module docstring.
        """
        low_confidence = confidence < self.conf_threshold
        low_agreement  = agreement  < self.agree_threshold
        alert          = low_confidence and low_agreement

        parts = []
        if low_confidence:
            parts.append(f"low confidence ({confidence:.1%} < {self.conf_threshold:.0%})")
        if low_agreement:
            parts.append(
                f"low retrieval agreement ({agree_count}/{k} neighbors agree, "
                f"{agreement:.0%} < {self.agree_threshold:.0%})"
            )

        if alert:
            explanation = (
                f"ALERT: predicted {prediction} with {' and '.join(parts)}. "
                f"Flag for review."
            )
        elif not low_confidence and not low_agreement:
            explanation = (
                f"No alert: predicted {prediction} — "
                f"confidence {confidence:.1%} and agreement {agreement:.0%} both above thresholds."
            )
        else:
            # One signal low but not both
            triggered = parts[0] if parts else ""
            explanation = (
                f"No alert: predicted {prediction} — "
                f"{triggered}, but combined rule requires both signals low."
            )

        return alert, explanation

    def _apply_neutral_low_agreement(self, prediction: str, agreement: float) -> bool:
        """
        Second-stage rule: force an alert when the classifier predicts "neutral"
        but the retrieved neighbors mostly disagree with that prediction.

        [AGENT5-RULE-STAGE2] Motivation: "neutral" is the majority class and a
        common dumping ground for misclassified minority-class utterances
        (e.g. anger mistaken for neutral due to flat prosody). Low
        prediction-retrieval agreement on a neutral call is a targeted signal
        for exactly this failure mode, independent of whether the primary rule
        already fired.
        """
        return prediction == "neutral" and agreement < self.stage2_agree_threshold

    # ── Main interface ─────────────────────────────────────────────────────

    def __call__(self, agent_result: AgentResult) -> AlertResult:
        """
        Evaluate the alert rule against an AgentResult.

        Parameters
        ──────────
        agent_result : output from EmotionAgent.__call__()

        Returns
        ───────
        AlertResult with alert flag and explanation.
        """
        prediction     = agent_result.predicted_emotion
        neighbor_labels = [ex.emotion for ex in agent_result.retrieved_examples]
        k              = len(neighbor_labels)
        agreement      = self._prediction_agreement(neighbor_labels, prediction)
        agree_count    = int(round(agreement * k))
        majority       = self._majority_label(neighbor_labels)
        confidence     = self._confidence(agent_result)

        # [AGENT5-DISPATCH] Route to selected rule
        if self.rule == "graded_agreement":
            alert, explanation = self._apply_graded_agreement(
                prediction, agreement, agree_count, k, majority
            )
            conf_used = None
        else:  # combined
            alert, explanation = self._apply_combined(
                prediction, agreement, agree_count, k, majority, confidence
            )
            conf_used = confidence

        # [AGENT5-STAGE2] Second-stage override: force an alert on low-agreement
        # neutral predictions, regardless of the primary rule's decision.
        stage2_triggered = False
        rule_used = self.rule
        if self.use_stage2:
            stage2_triggered = self._apply_neutral_low_agreement(prediction, agreement)
            if stage2_triggered and not alert:
                explanation = (
                    f"{explanation} "
                    f"OVERRIDE (neutral_low_agreement): predicted neutral with "
                    f"only {agree_count}/{k} neighbors agreeing "
                    f"({agreement:.0%} < {self.stage2_agree_threshold:.0%}). Forcing alert."
                )
            alert = alert or stage2_triggered
            rule_used = f"{self.rule}+neutral_low_agreement"

        return AlertResult(
            alert=alert,
            predicted_emotion=prediction,
            prediction_agreement=agreement,
            majority_neighbor_label=majority,
            agreement_count=agree_count,
            k=k,
            rule_used=rule_used,
            confidence=conf_used,
            explanation=explanation,
            stage2_triggered=stage2_triggered,
        )
