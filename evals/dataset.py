"""
Eval dataset — ground truth race strategies for scoring.

Each entry contains:
  - Race identifier (year, grand_prix, driver)
  - Actual strategy used (compounds + pit laps)
  - Actual finishing position
  - Circuit type (affects what "good strategy" looks like)
  - Notes on special conditions (safety car, rain, DNF, etc.)

We use 2023 season races across different circuit profiles:
  - High degradation  : Bahrain, Abu Dhabi
  - Low degradation   : Monaco, Monza
  - Mixed             : British
"""

from dataclasses import dataclass, field


@dataclass
class GroundTruthStrategy:
    """The actual strategy run in a race."""
    compounds:      list[str]   # in stint order
    pit_laps:       list[int]   # lap numbers when pitted
    finishing_pos:  int         # actual finishing position
    grid_pos:       int         # starting grid position
    dnf:            bool = False        # retired during race — exclude from position scoring
    dns:            bool = False        # never started — exclude from all scoring
    safety_car:     bool = False        # known safety car influence on strategy
    rain:           bool = False        # wet or mixed conditions


@dataclass
class EvalCase:
    """A single eval case — one driver, one race."""
    year:           int
    grand_prix:     str
    driver:         str
    ground_truth:   GroundTruthStrategy
    circuit_type:   str         # "high_deg", "low_deg", "mixed"
    notes:          str = ""    # special conditions worth knowing


# ── Dataset ───────────────────────────────────────────────────────────────────
EVAL_DATASET: list[EvalCase] = [

    # ── Bahrain 2023 — High degradation ──────────────────────────────────────
    EvalCase(
        year=2023, grand_prix="Bahrain", driver="VER",
        ground_truth=GroundTruthStrategy(
            compounds=["SOFT", "SOFT", "HARD"],
            pit_laps=[14, 36],
            finishing_pos=1,
            grid_pos=1,
        ),
        circuit_type="high_deg",
        notes="Red Bull used a Soft-Soft-Hard strategy to dominate.",
    ),
    EvalCase(
        year=2023, grand_prix="Bahrain", driver="LEC",
        ground_truth=GroundTruthStrategy(
            compounds=["SOFT", "HARD", "HARD"],
            pit_laps=[13, 33],
            finishing_pos=21,
            grid_pos=3,
            dnf=True,
        ),
        circuit_type="high_deg",
        notes="DNF on Lap 39 (Power Unit). Was running P3 before retirement.",
    ),

    # ── British 2023 — Mixed, Safety Car influenced ───────────────────────────
    EvalCase(
        year=2023, grand_prix="British", driver="VER",
        ground_truth=GroundTruthStrategy(
            compounds=["MEDIUM", "SOFT"],
            pit_laps=[33],
            finishing_pos=1,
            grid_pos=1,
            safety_car=True,
        ),
        circuit_type="mixed",
        notes="Safety Car on Lap 33 allowed a 'free' pit stop for Softs.",
    ),
    EvalCase(
        year=2023, grand_prix="British", driver="LEC",
        ground_truth=GroundTruthStrategy(
            compounds=["MEDIUM", "HARD", "SOFT"],
            pit_laps=[18, 33],
            finishing_pos=9,
            grid_pos=4,
            safety_car=True,
        ),
        circuit_type="mixed",
        notes="Early pit for Hard backfired due to Safety Car; pitted again for Softs.",
    ),

    # ── Monaco 2023 — Low degradation, Rain affected ──────────────────────────
    EvalCase(
        year=2023, grand_prix="Monaco", driver="VER",
        ground_truth=GroundTruthStrategy(
            compounds=["MEDIUM", "INTERMEDIATE"],
            pit_laps=[55],
            finishing_pos=1,
            grid_pos=1,
            rain=True,
        ),
        circuit_type="low_deg",
        notes="Stayed out on Mediums until the rain arrived on Lap 55.",
    ),
    EvalCase(
        year=2023, grand_prix="Monaco", driver="ALO",
        ground_truth=GroundTruthStrategy(
            compounds=["HARD", "MEDIUM", "INTERMEDIATE"],
            pit_laps=[54, 55],
            finishing_pos=2,
            grid_pos=2,
            rain=True,
        ),
        circuit_type="low_deg",
        notes="Controversial pit for slicks during rain, forced to pit again for Inters next lap.",
    ),

    # ── Monza 2023 — Low degradation, 1-stop ─────────────────────────────────
    EvalCase(
        year=2023, grand_prix="Italian", driver="SAI",
        ground_truth=GroundTruthStrategy(
            compounds=["MEDIUM", "HARD"],
            pit_laps=[19],
            finishing_pos=3,
            grid_pos=1,
        ),
        circuit_type="low_deg",
        notes="Defended against VER for 14 laps; finished P3 after intense battle with LEC.",
    ),
    EvalCase(
        year=2023, grand_prix="Italian", driver="VER",
        ground_truth=GroundTruthStrategy(
            compounds=["MEDIUM", "HARD"],
            pit_laps=[20],
            finishing_pos=1,
            grid_pos=2,
        ),
        circuit_type="low_deg",
        notes="10th consecutive win. Standard Medium-Hard strategy.",
    ),

    # ── Abu Dhabi 2023 — High degradation, 2-stop ────────────────────────────
    EvalCase(
        year=2023, grand_prix="Abu Dhabi", driver="VER",
        ground_truth=GroundTruthStrategy(
            compounds=["MEDIUM", "HARD", "HARD"],
            pit_laps=[17, 42],
            finishing_pos=1,
            grid_pos=1,
        ),
        circuit_type="high_deg",
        notes="Standard two-stop to manage tire wear under lights.",
    ),
    EvalCase(
        year=2023, grand_prix="Abu Dhabi", driver="LEC",
        ground_truth=GroundTruthStrategy(
            compounds=["MEDIUM", "HARD", "HARD"],
            pit_laps=[17, 35],
            finishing_pos=2,
            grid_pos=2,
        ),
        circuit_type="high_deg",
        notes="P2 finish. Tried to give Perez a slipstream at the end to help Ferrari in WCC.",
    ),
]


# ── Helpers ───────────────────────────────────────────────────────────────────
def get_case(year: int, grand_prix: str, driver: str) -> EvalCase | None:
    """Fetch a specific eval case by identifier."""
    for case in EVAL_DATASET:
        if case.year == year and case.grand_prix == grand_prix and case.driver == driver:
            return case
    return None


def get_cases_by_circuit_type(circuit_type: str) -> list[EvalCase]:
    """Filter dataset by circuit type."""
    return [c for c in EVAL_DATASET if c.circuit_type == circuit_type]


def is_scoreable(case: EvalCase) -> bool:
    """
    Returns True if this case should be included in scoring.
    DNS cases are excluded entirely.
    DNF cases are scored on strategy quality but not finishing position.
    """
    return not case.ground_truth.dns