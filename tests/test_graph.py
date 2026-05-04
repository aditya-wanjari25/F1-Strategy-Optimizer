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