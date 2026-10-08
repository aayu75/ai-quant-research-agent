import re

from graph.state import AgentState


def extract_ticker(query: str) -> str:
    """
    Extract a stock ticker from the user's query.
    """

    # Find words written in uppercase letters.
    matches = re.findall(r"\b[A-Z]{1,5}\b", query)

    # Common words that may accidentally look like tickers.
    ignored_words = {
        "THE",
        "AND",
        "FOR",
        "WITH",
        "FROM",
        "WHAT",
        "HOW",
        "IS",
        "ARE",
        "ON",
        "OF"
    }

    for match in matches:
        if match not in ignored_words:
            return match

    raise ValueError("Could not identify a stock ticker from the query.")


def planner_agent(state: AgentState) -> AgentState:
    """
    Analyze the user query and create an initial research plan.
    """

    query = state["query"]

    ticker = extract_ticker(query)

    plan = [
        "Collect historical market data",
        "Calculate quantitative indicators",
        "Analyze risk",
        "Validate the analysis",
        "Generate final research report"
    ]

    return {
        "query": query,
        "ticker": ticker,
        "plan": plan
    }