"""
Scorer — compares our system's strategy recommendation against ground truth.

Produces a structured EvalScore with four metrics:

  1. stint_count_match   — did we predict the right number of stops?
  2. compound_accuracy   — how many compounds did we get right?
  3. pit_lap_delta       — how close were our pit laps to actual?
  4. overall_score       — weighted composite 0.0 → 1.0

Special cases handled:
  - DNF: skip position scoring, score strategy quality only
  - DNS: skip entirely (handled upstream in runner)
  - Safety car: flag in score, don't penalize for pit lap delta
  - Rain: flag in score, compound mismatch expected
"""

from dataclasses import dataclass
from evals.dataset import EvalCase, GroundTruthStrategy


# ── Score model ───────────────────────────────────────────────────────────────
@dataclass
class EvalScore:
    # Identifiers
    year:           int
    grand_prix:     str
    driver:         str
    circuit_type:   str

    # Individual metrics
    stint_count_match:  bool        # predicted same number of stints as actual
    compound_accuracy:  float       # 0.0 → 1.0, % of compounds correctly predicted
    pit_lap_delta:      float       # avg absolute difference in pit lap numbers
    pit_lap_score:      float       # 0.0 → 1.0, derived from delta (lower delta = higher score)

    # Composite
    overall_score:      float       # 0.0 → 1.0 weighted composite

    # Flags
    dnf:            bool
    safety_car:     bool
    rain:           bool

    # Human readable
    notes:          str
    recommendation: dict            # what our system said
    ground_truth:   dict            # what actually happened


# ── Scoring helpers ───────────────────────────────────────────────────────────
def score_compounds(predicted: list[str], actual: list[str]) -> float:
    """
    Scores compound accuracy by comparing stint-by-stint.

    Handles different stint counts gracefully:
    - If we predicted 2 stints but actual was 3, we score the overlap
    - Penalizes for wrong stint count via length mismatch

    Returns 0.0 → 1.0
    """
    if not predicted or not actual:
        return 0.0

    # Compare up to the shorter list length
    min_len = min(len(predicted), len(actual))
    matches = sum(
        1 for p, a in zip(predicted[:min_len], actual[:min_len])
        if p.upper() == a.upper()
    )

    # Score = matches / max length (penalizes for wrong stint count)
    return round(matches / max(len(predicted), len(actual)), 3)


def score_pit_laps(predicted: list[int], actual: list[int]) -> tuple[float, float]:
    """
    Scores pit lap accuracy.

    Returns (avg_delta, pit_lap_score):
      - avg_delta: average absolute difference in lap numbers
      - pit_lap_score: 0.0 → 1.0 (0 delta = 1.0, 10+ laps delta = 0.0)

    Handles different pit stop counts by scoring the overlap.
    A 5-lap delta is considered acceptable (pit_lap_score >= 0.5).
    """
    if not predicted and not actual:
        return 0.0, 1.0     # no pit stops predicted and none happened = perfect
    if not predicted or not actual:
        return 10.0, 0.0    # one side has stops and other doesn't = worst score

    min_len = min(len(predicted), len(actual))
    deltas = [
        abs(p - a)
        for p, a in zip(sorted(predicted)[:min_len], sorted(actual)[:min_len])
    ]

    # Penalize for wrong number of pit stops
    stop_count_penalty = abs(len(predicted) - len(actual)) * 5
    avg_delta = round(sum(deltas) / len(deltas) + stop_count_penalty, 2)

    # Convert delta to 0→1 score: 0 laps = 1.0, 10+ laps = 0.0
    pit_lap_score = round(max(0.0, 1.0 - (avg_delta / 10.0)), 3)

    return avg_delta, pit_lap_score


def compute_overall_score(
    compound_accuracy: float,
    pit_lap_score: float,
    stint_count_match: bool,
    safety_car: bool,
    rain: bool,
) -> float:
    """
    Weighted composite score.

    Weights:
      40% compound accuracy   — getting the right tyres matters most
      40% pit lap score       — timing is critical
      20% stint count match   — wrong number of stops is a strategic failure

    Safety car and rain reduce the weight of pit lap score
    since those are inherently unpredictable.
    """
    stint_score = 1.0 if stint_count_match else 0.0

    if safety_car or rain:
        # Pit lap timing is largely luck under SC/rain — reduce its weight
        weights = {"compound": 0.6, "pit_lap": 0.1, "stint": 0.3}
    else:
        weights = {"compound": 0.4, "pit_lap": 0.4, "stint": 0.2}

    overall = (
        compound_accuracy * weights["compound"] +
        pit_lap_score     * weights["pit_lap"] +
        stint_score       * weights["stint"]
    )
    return round(overall, 3)


# ── Main scorer ───────────────────────────────────────────────────────────────
def score(
    case: EvalCase,
    predicted_compounds: list[str],
    predicted_pit_laps: list[int],
) -> EvalScore:
    """
    Scores a single prediction against ground truth.

    Args:
        case: the eval case with ground truth
        predicted_compounds: what our system recommended
        predicted_pit_laps: what pit laps our system recommended

    Returns:
        EvalScore with all metrics populated
    """
    gt = case.ground_truth

    # 1. Stint count
    stint_count_match = len(predicted_compounds) == len(gt.compounds)

    # 2. Compound accuracy
    compound_accuracy = score_compounds(predicted_compounds, gt.compounds)

    # 3. Pit lap delta
    # Skip pit lap scoring for DNF — driver never finished the strategy
    if gt.dnf:
        pit_lap_delta = 0.0
        pit_lap_score = 1.0     # don't penalize for DNF
    else:
        pit_lap_delta, pit_lap_score = score_pit_laps(predicted_pit_laps, gt.pit_laps)

    # 4. Overall
    overall_score = compute_overall_score(
        compound_accuracy=compound_accuracy,
        pit_lap_score=pit_lap_score,
        stint_count_match=stint_count_match,
        safety_car=gt.safety_car,
        rain=gt.rain,
    )

    return EvalScore(
        year=case.year,
        grand_prix=case.grand_prix,
        driver=case.driver,
        circuit_type=case.circuit_type,
        stint_count_match=stint_count_match,
        compound_accuracy=compound_accuracy,
        pit_lap_delta=pit_lap_delta,
        pit_lap_score=pit_lap_score,
        overall_score=overall_score,
        dnf=gt.dnf,
        safety_car=gt.safety_car,
        rain=gt.rain,
        notes=case.notes,
        recommendation={
            "compounds": predicted_compounds,
            "pit_laps":  predicted_pit_laps,
        },
        ground_truth={
            "compounds": gt.compounds,
            "pit_laps":  gt.pit_laps,
        },
    )