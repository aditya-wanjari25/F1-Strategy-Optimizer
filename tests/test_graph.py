"""Test the graph wiring with placeholder nodes."""

from graph.workflow import build_graph, run_graph


def test_graph_compiles():
    graph = build_graph()
    assert graph is not None
    print("\n✅ Graph compiled successfully")


def test_graph_runs_placeholder():
    """
    With all placeholder nodes, the graph should route through
    supervisor → tire → supervisor → weather → supervisor → competitor
    → supervisor → synthesizer → supervisor → END
    without errors.
    """
    result = run_graph(2023, "Bahrain", "VER")
    assert result is not None
    print(f"\n✅ Graph ran to completion")
    print(f"   Final next_agent: {result.get('next_agent')}")

def test_full_graph_end_to_end():
    result = run_graph(2023, "Bahrain", "VER")

    assert result["tire_analysis"] is not None
    assert result["weather_analysis"] is not None
    assert result["competitor_analysis"] is not None
    assert result["strategy_recommendation"] is not None
    assert result["errors"] == []

    sr = result["strategy_recommendation"]
    print(f"\n✅ Full Graph Result:")
    print(f"   Compounds  : {sr.compounds}")
    print(f"   Pit laps   : {sr.pit_laps}")
    print(f"   Confidence : {sr.confidence}")
    print(f"   Rationale  : {sr.rationale}")