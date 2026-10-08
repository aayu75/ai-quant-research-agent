from graph.workflow import graph


def test_complete_workflow():

    initial_state = {
        "query": "Analyze AAPL momentum and risk"
    }

    result = graph.invoke(
        initial_state
    )

    # Planner
    assert result["ticker"] == "AAPL"

    # Data Agent
    assert "market_data" in result

    # Quant Agent
    assert "quant_analysis" in result

    # Risk Agent
    assert "risk_analysis" in result

    # Critic Agent
    assert "critic_result" in result

    assert result["critic_result"]["status"] in {
        "PASS",
        "FAIL"
    }

    # Report Agent
    assert "final_report" in result

    assert result["final_report"] != ""