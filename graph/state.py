from typing import TypedDict, List, Dict, Any


class AgentState(TypedDict, total=False):
    """
    Shared state passed between agents
    in the AI Quantitative Research workflow.
    """

    # Original user question
    query: str

    # Stock ticker
    ticker: str

    # Tasks decided by Planner
    plan: List[str]

    # Market data collected by Data Agent
    market_data: Dict[str, Any]

    # Quantitative analysis
    quant_analysis: Dict[str, Any]

    # Risk analysis
    risk_analysis: Dict[str, Any]

    # Critic evaluation
    critic_result: Dict[str, Any]

    # Final report
    final_report: str

    # Errors
    errors: List[str]

    # Number of times analysis has been retried
    retry_count: int