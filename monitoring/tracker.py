import time

from monitoring.logger import (
    log_agent_start,
    log_agent_end,
    log_agent_error
)


def run_with_monitoring(
    agent_function,
    state,
    agent_name
):
    """
    Execute an agent while recording
    execution metrics and failures.
    """

    start_time = time.perf_counter()

    log_agent_start(
        agent_name,
        {
            "ticker": state.get("ticker"),
            "retry_count": state.get(
                "retry_count",
                0
            )
        }
    )

    try:

        result = agent_function(
            state
        )

        duration = (
            time.perf_counter()
            - start_time
        )

        log_agent_end(
            agent_name,
            duration,
            {
                "error_count": len(
                    result.get(
                        "errors",
                        []
                    )
                )
            }
        )

        return result

    except Exception as error:

        duration = (
            time.perf_counter()
            - start_time
        )

        log_agent_error(
            agent_name,
            error,
            duration
        )

        raise