def check_required_sections(
    report: str
) -> dict:

    required_sections = [
        "QUANTITATIVE ANALYSIS",
        "MACHINE LEARNING ANALYSIS",
        "HISTORICAL BACKTEST",
        "RISK ANALYSIS",
        "VALIDATION"
    ]

    missing = []

    for section in required_sections:

        if section not in report:
            missing.append(section)

    return {
        "passed": len(missing) == 0,
        "missing_sections": missing
    }


def check_numeric_results(
    quant_analysis: dict,
    risk_analysis: dict
) -> dict:

    required_quant_fields = [
        "latest_price",
        "average_daily_return",
        "momentum_20d",
        "ml_probability"
    ]

    required_risk_fields = [
        "annualized_volatility",
        "sharpe_ratio",
        "maximum_drawdown"
    ]

    missing_quant = [
        field
        for field in required_quant_fields
        if field not in quant_analysis
    ]

    missing_risk = [
        field
        for field in required_risk_fields
        if field not in risk_analysis
    ]

    return {
        "quant_passed": (
            len(missing_quant) == 0
        ),

        "risk_passed": (
            len(missing_risk) == 0
        ),

        "missing_quant_fields":
            missing_quant,

        "missing_risk_fields":
            missing_risk
    }

def check_critic(
    critic_result: dict
) -> dict:

    return {
        "passed": (
            critic_result.get("status")
            == "PASS"
        ),
        "status": critic_result.get(
            "status"
        ),
        "issues": critic_result.get(
            "issues",
            []
        )
    }


def calculate_evaluation_score(
    results: dict
) -> float:

    checks = [
        results["ticker_correct"],
        results["quant_results_valid"],
        results["risk_results_valid"],
        results["report_valid"],
        results["critic_passed"]
    ]

    passed = sum(
        1 for check in checks
        if check
    )

    return passed / len(checks)