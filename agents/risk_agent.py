from graph.state import AgentState

from mcp_server.quant_tools import (
    calculate_volatility,
    calculate_sharpe_ratio,
    calculate_max_drawdown
)


def risk_agent(state: AgentState) -> AgentState:
    """
    Analyze the risk characteristics of the asset.
    """

    market_data = state["market_data"]
    data = market_data["data"]

    try:
        volatility = calculate_volatility(data)

        sharpe_ratio = calculate_sharpe_ratio(data)

        max_drawdown = calculate_max_drawdown(data)

        risk_analysis = {
            "annualized_volatility": volatility,
            "sharpe_ratio": sharpe_ratio,
            "maximum_drawdown": max_drawdown
        }

        return {
            "risk_analysis": risk_analysis,
            "errors": []
        }

    except Exception as e:

        return {
            "risk_analysis": {},
            "errors": [str(e)]
        }