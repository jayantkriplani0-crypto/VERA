"""Score interpretation module for Spectra-AASIST3 bona fide logits."""
from __future__ import annotations

from dataclasses import dataclass


# Official threshold from SpectraAASIST3.classify in official model.py
OFFICIAL_EER_THRESHOLD: float = -1.0625009
BONAFIDE_CLASS_INDEX: int = 1


@dataclass(frozen=True)
class ScoreInterpretation:
    """Interpreted assessment of the model's raw bona fide logit score."""
    raw_score: float
    threshold: float
    predicted_label: str
    is_bona_fide: bool
    description: str
    metric_type: str = "Raw Bona Fide Logit (Unnormalized)"

    def summary_text(self) -> str:
        """Formatted human-readable summary of the score and interpretation."""
        lines = [
            f"Raw Bona Fide Logit : {self.raw_score:+.6f}",
            f"Operational Cutoff  : {self.threshold:+.6f} (Official EER threshold)",
            f"Decision            : {self.predicted_label}",
            f"Explanation         : {self.description}",
            f"Note                : Score is a raw logit (higher = more genuine). Not a percentage or probability.",
        ]
        return "\n".join(lines)


def interpret_score(
    raw_score: float,
    threshold: float = OFFICIAL_EER_THRESHOLD
) -> ScoreInterpretation:
    """Interpret the raw model logit score without inventing fake probabilities or percentages.

    Args:
        raw_score: The bona-fide class logit (logits[:, 1]) produced by Spectra-AASIST3.
        threshold: The decision threshold (default: official EER threshold -1.0625009).

    Returns:
        ScoreInterpretation dataclass with structured decision and explanation.
    """
    is_bona_fide = bool(raw_score > threshold)
    margin = raw_score - threshold

    if is_bona_fide:
        predicted_label = "BONA FIDE (Genuine Human Speech)"
        if margin > 2.0:
            desc = (
                f"Strong genuine speech signal. The bona fide logit ({raw_score:+.4f}) "
                f"is well above the decision threshold ({threshold:+.4f}) with a margin of {margin:+.4f}."
            )
        else:
            desc = (
                f"Moderate genuine speech signal. The bona fide logit ({raw_score:+.4f}) "
                f"exceeds the decision threshold ({threshold:+.4f}) with a margin of {margin:+.4f}."
            )
    else:
        predicted_label = "SPOOF / DEEPFAKE (Synthetic / Cloned Speech)"
        if margin < -2.0:
            desc = (
                f"Strong spoof / deepfake indicators detected. The bona fide logit ({raw_score:+.4f}) "
                f"is well below the decision threshold ({threshold:+.4f}) with a margin of {margin:+.4f}."
            )
        else:
            desc = (
                f"Borderline / suspected spoof signal. The bona fide logit ({raw_score:+.4f}) "
                f"falls below the decision threshold ({threshold:+.4f}) with a margin of {margin:+.4f}."
            )

    return ScoreInterpretation(
        raw_score=float(raw_score),
        threshold=float(threshold),
        predicted_label=predicted_label,
        is_bona_fide=is_bona_fide,
        description=desc,
    )
