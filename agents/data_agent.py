from graph.state import AgentState

from mcp_server.client import (
    get_market_data
)


def data_agent(
    state: AgentState
) -> AgentState:
    """
    Retrieve historical market data
    through the MCP server.
    """

    ticker = state["ticker"]

    try:

        result = get_market_data(
            ticker=ticker,
            period="2y"
        )

        return {
            "market_data": {
                "ticker": ticker,
                "data": result
            },
            "errors": []
        }

    except Exception as error:

        return {
            "market_data": {},
            "errors": [
                str(error)
            ]
        }