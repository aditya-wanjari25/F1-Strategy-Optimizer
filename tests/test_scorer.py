"""Tests for the eval scorer."""

from evals.dataset import get_case
from evals.scorer import score, score_compounds, score_pit_laps


def test_perfect_score():
    """Exact match on compounds and pit laps should score 1.0."""
    case = get_case(2023, "Italian", "VER")
    result = score(case, ["MEDIUM", "HARD"], [20])
    assert result.overall_score == 1.0
    assert result.compound_accuracy == 1.0
    assert result.pit_lap_delta == 0.0
    print(f"\n✅ Perfect score: {result.overall_score}")


def test_wrong_compounds():
    """Wrong compounds should reduce compound accuracy."""
    case = get_case(2023, "Italian", "VER")
    result = score(case, ["SOFT", "HARD"], [20])
    assert result.compound_accuracy < 1.0
    assert result.overall_score < 1.0
    print(f"\n✅ Wrong compound score: {result.overall_score}")


def test_pit_lap_delta():
    """5 lap delta should give partial pit lap score."""
    case = get_case(2023, "Italian", "VER")
    result = score(case, ["MEDIUM", "HARD"], [25])  # actual was 20
    assert result.pit_lap_delta == 5.0
    assert 0.0 < result.pit_lap_score < 1.0
    print(f"\n✅ Pit lap delta: {result.pit_lap_delta}, score: {result.pit_lap_score}")


def test_dnf_not_penalized_on_pit_laps():
    """DNF cases should not be penalized on pit lap timing."""
    case = get_case(2023, "Bahrain", "LEC")
    assert case.ground_truth.dnf is True
    result = score(case, ["SOFT", "HARD"], [13])
    assert result.pit_lap_score == 1.0  # not penalized
    print(f"\n✅ DNF pit lap score not penalized: {result.pit_lap_score}")


def test_safety_car_weights():
    """Safety car races should weight compounds more than pit laps."""
    case = get_case(2023, "British", "VER")
    assert case.ground_truth.safety_car is True
    # Wrong pit lap by 10 laps but right compound
    result = score(case, ["MEDIUM", "SOFT"], [43])
    # Should still score reasonably due to SC weight adjustment
    assert result.overall_score > 0.4
    print(f"\n✅ Safety car adjusted score: {result.overall_score}")


def test_score_compounds_partial():
    assert score_compounds(["SOFT", "HARD"], ["SOFT", "MEDIUM"]) == 0.5
    assert score_compounds(["SOFT", "HARD"], ["SOFT", "HARD"]) == 1.0
    assert score_compounds(["SOFT"], ["SOFT", "HARD"]) == 0.5
    print("\n✅ Compound scoring correct")


def test_score_pit_laps_delta():
    delta, score_val = score_pit_laps([35], [30])
    assert delta == 5.0
    assert score_val == 0.5
    print(f"\n✅ Pit lap delta: {delta}, score: {score_val}")