import json
import time
from pathlib import Path

from graph.workflow import graph

from evaluation.metrics import (
    check_required_sections,
    check_numeric_results,
    check_critic,
    calculate_evaluation_score
)


BASE_DIR = Path(__file__).resolve().parent

TEST_CASE_FILE = (
    BASE_DIR / "test_cases.json"
)


def load_test_cases():

    with open(
        TEST_CASE_FILE,
        "r",
        encoding="utf-8"
    ) as file:

        return json.load(file)


def evaluate_case(
    test_case: dict
) -> dict:

    query = test_case["query"]

    expected_ticker = (
        test_case["ticker"]
    )

    initial_state = {
        "query": query,
        "errors": [],
        "retry_count": 0
    }

    start_time = time.perf_counter()

    try:

        result = graph.invoke(
            initial_state
        )

        duration = (
            time.perf_counter()
            - start_time
        )

        actual_ticker = result.get(
            "ticker"
        )

        quant_analysis = result.get(
            "quant_analysis",
            {}
        )

        risk_analysis = result.get(
            "risk_analysis",
            {}
        )

        critic_result = result.get(
            "critic_result",
            {}
        )

        report = result.get(
            "final_report",
            ""
        )

        # ------------------------------------------
        # Individual checks
        # ------------------------------------------

        ticker_correct = (
            actual_ticker
            == expected_ticker
        )

        numeric_check = (
            check_numeric_results(
                quant_analysis,
                risk_analysis
            )
        )

        report_check = (
            check_required_sections(
                report
            )
        )

        critic_check = (
            check_critic(
                critic_result
            )
        )

        evaluation_results = {

            "ticker_correct":
                ticker_correct,

            "quant_results_valid":
                numeric_check["quant_passed"],

            "risk_results_valid":
                numeric_check["risk_passed"],

            "report_valid":
                report_check["passed"],

            "critic_passed":
                critic_check["passed"]
        }

        score = (
            calculate_evaluation_score(
                evaluation_results
            )
        )

        return {

            "id": test_case["id"],

            "query": query,

            "expected_ticker":
                expected_ticker,

            "actual_ticker":
                actual_ticker,

            "duration_seconds":
                round(
                    duration,
                    3
                ),

            "evaluation":
                evaluation_results,

            "score":
                score,

            "critic_status":
                critic_result.get(
                    "status"
                ),

            "critic_issues":
                critic_result.get(
                    "issues",
                    []
                )
        }

    except Exception as error:

        duration = (
            time.perf_counter()
            - start_time
        )

        return {

            "id": test_case["id"],

            "query": query,

            "expected_ticker":
                expected_ticker,

            "actual_ticker":
                None,

            "duration_seconds":
                round(
                    duration,
                    3
                ),

            "evaluation": {},

            "score": 0.0,

            "error": str(error)
        }


def run_evaluation():

    test_cases = (
        load_test_cases()
    )

    results = []

    for test_case in test_cases:

        print(
            f"\nRunning {test_case['id']}..."
        )

        result = evaluate_case(
            test_case
        )

        results.append(result)

        print(
            f"Score: {result['score']:.2f}"
        )

    average_score = (
        sum(
            result["score"]
            for result in results
        )
        / len(results)
    )

    print("\n" + "=" * 50)

    print(
        "AI QUANT RESEARCH AGENT EVALUATION"
    )

    print("=" * 50)

    print(
        f"Cases: {len(results)}"
    )

    print(
        f"Average Score: "
        f"{average_score:.2f}"
    )

    for result in results:

        print(
            f"{result['id']}: "
            f"{result['score']:.2f} "
            f"({result['duration_seconds']}s)"
        )

    return {
        "results": results,
        "average_score": average_score
    }


if __name__ == "__main__":
    run_evaluation()