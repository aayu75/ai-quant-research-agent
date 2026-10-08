from graph.state import AgentState


def test_agent_state():

    state: AgentState = {
        "query": "Analyze NVDA momentum",
        "ticker": "NVDA",
        "plan": [
            "Get market data",
            "Calculate momentum",
            "Analyze risk"
        ],
        "errors": []
    }

    assert state["ticker"] == "NVDA"
    assert len(state["plan"]) == 3
    assert state["errors"] == []