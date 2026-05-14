"""
Eval runner — runs our agent system against the eval dataset
and collects scores for every case.
"""

from dotenv import load_dotenv
load_dotenv()

import structlog
from evals.dataset import EVAL_DATASET, EvalCase, is_scoreable
from evals.scorer import score, EvalScore
from graph.workflow import run_graph

log = structlog.get_logger()


def run_single(case: EvalCase) -> EvalScore | None:
    """
    Runs the full agent graph for one eval case and scores the result.
    Returns None if the case is not scoreable (DNS).
    """
    if not is_scoreable(case):
        log.info("eval_skipped", driver=case.driver, grand_prix=case.grand_prix, reason="dns")
        return None

    log.info("eval_running", driver=case.driver, grand_prix=case.grand_prix, year=case.year)

    try:
        result = run_graph(case.year, case.grand_prix, case.driver)
        errors = result.get("errors", [])

        if errors:
            log.error("eval_graph_errors", errors=errors)
            return None

        sr = result["strategy_recommendation"]
        eval_score = score(
            case=case,
            predicted_compounds=sr.compounds,
            predicted_pit_laps=sr.pit_laps,
        )

        log.info(
            "eval_complete",
            driver=case.driver,
            grand_prix=case.grand_prix,
            overall_score=eval_score.overall_score,
        )
        return eval_score

    except Exception as e:
        log.error("eval_failed", driver=case.driver, grand_prix=case.grand_prix, error=str(e))
        return None


def run_all(cases: list[EvalCase] | None = None) -> list[EvalScore]:
    """
    Runs evals for all cases in the dataset (or a subset).
    Returns a list of EvalScore results.
    """
    cases = cases or EVAL_DATASET
    scores = []

    for case in cases:
        eval_score = run_single(case)
        if eval_score is not None:
            scores.append(eval_score)

    return scores