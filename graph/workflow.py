from langgraph.graph import (
    StateGraph,
    START,
    END
)

from graph.state import AgentState

from agents.planner import (
    planner_agent
)

from agents.data_agent import (
    data_agent
)

from agents.quant_agent import (
    quant_agent
)

from agents.risk_agent import (
    risk_agent
)

from agents.critic_agent import (
    critic_agent
)

from agents.report_agent import (
    report_agent
)

from monitoring.tracker import (
    run_with_monitoring
)


# ==================================================
# Monitored Agent Wrappers
# ==================================================


def monitored_planner(state):
    return run_with_monitoring(
        planner_agent,
        state,
        "planner_agent"
    )


def monitored_data(state):
    return run_with_monitoring(
        data_agent,
        state,
        "data_agent"
    )


def monitored_quant(state):
    return run_with_monitoring(
        quant_agent,
        state,
        "quant_agent"
    )


def monitored_risk(state):
    return run_with_monitoring(
        risk_agent,
        state,
        "risk_agent"
    )


def monitored_critic(state):
    return run_with_monitoring(
        critic_agent,
        state,
        "critic_agent"
    )


def monitored_report(state):
    return run_with_monitoring(
        report_agent,
        state,
        "report_agent"
    )


# ==================================================
# Routing
# ==================================================


def route_after_critic(
    state: AgentState
):

    critic_result = state.get(
        "critic_result",
        {}
    )

    status = critic_result.get(
        "status"
    )

    retry_count = state.get(
        "retry_count",
        0
    )

    max_retries = 2

    if status == "PASS":
        return "report"

    if retry_count < max_retries:
        return "retry"

    return "report"


def increment_retry(
    state: AgentState
) -> AgentState:

    retry_count = state.get(
        "retry_count",
        0
    )

    return {
        "retry_count":
            retry_count + 1
    }


# ==================================================
# Build Graph
# ==================================================


workflow = StateGraph(
    AgentState
)


workflow.add_node(
    "planner",
    monitored_planner
)

workflow.add_node(
    "data",
    monitored_data
)

workflow.add_node(
    "quant",
    monitored_quant
)

workflow.add_node(
    "risk",
    monitored_risk
)

workflow.add_node(
    "critic",
    monitored_critic
)

workflow.add_node(
    "retry",
    increment_retry
)

workflow.add_node(
    "report",
    monitored_report
)


# ==================================================
# Workflow Edges
# ==================================================


workflow.add_edge(
    START,
    "planner"
)

workflow.add_edge(
    "planner",
    "data"
)

workflow.add_edge(
    "data",
    "quant"
)

workflow.add_edge(
    "quant",
    "risk"
)

workflow.add_edge(
    "risk",
    "critic"
)


workflow.add_conditional_edges(
    "critic",
    route_after_critic,
    {
        "report": "report",
        "retry": "retry"
    }
)


workflow.add_edge(
    "retry",
    "quant"
)

workflow.add_edge(
    "report",
    END
)


graph = workflow.compile()