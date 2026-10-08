from graph.state import AgentState


def critic_agent(state: AgentState) -> AgentState:

    issues = []
    warnings = []

    quant = state.get(
        "quant_analysis",
        {}
    )

    risk = state.get(
        "risk_analysis",
        {}
    )

    market_data = state.get(
        "market_data",
        {}
    )

    # ==================================================
    # Basic availability checks
    # ==================================================

    if not market_data:
        issues.append(
            "Market data is missing."
        )

    if not quant:
        issues.append(
            "Quantitative analysis is missing."
        )

    if not risk:
        issues.append(
            "Risk analysis is missing."
        )

    # ==================================================
    # Quantitative checks
    # ==================================================

    momentum = quant.get(
        "momentum_20d"
    )

    if momentum is not None:

        if not isinstance(
            momentum,
            (int, float)
        ):
            issues.append(
                "20-day momentum must be numeric."
            )

    latest_price = quant.get(
        "latest_price"
    )

    if latest_price is not None:

        if latest_price <= 0:
            issues.append(
                "Latest price must be greater than zero."
            )

    observations = quant.get(
        "number_of_observations"
    )

    if observations is not None:

        if observations < 30:
            issues.append(
                "Insufficient historical observations."
            )

    # ==================================================
    # Risk checks
    # ==================================================

    volatility = risk.get(
        "annualized_volatility"
    )

    if volatility is not None:

        if volatility < 0:
            issues.append(
                "Annualized volatility cannot be negative."
            )

    sharpe = risk.get(
        "sharpe_ratio"
    )

    if sharpe is not None:

        if not isinstance(
            sharpe,
            (int, float)
        ):
            issues.append(
                "Sharpe ratio must be numeric."
            )

    max_drawdown = risk.get(
        "maximum_drawdown"
    )

    if max_drawdown is not None:

        if max_drawdown > 0:
            issues.append(
                "Maximum drawdown should not be positive."
            )

        if max_drawdown < -1:
            issues.append(
                "Maximum drawdown cannot be below -100%."
            )

    # ==================================================
    # ML checks
    # ==================================================

    best_model = quant.get(
        "best_model"
    )

    valid_models = {
        "logistic_regression",
        "random_forest"
    }

    if best_model is not None:

        if best_model not in valid_models:
            issues.append(
                f"Unknown ML model selected: {best_model}"
            )

    ml_prediction = quant.get(
        "ml_prediction"
    )

    if ml_prediction is not None:

        if ml_prediction not in {0, 1}:
            issues.append(
                "ML prediction must be 0 or 1."
            )

    ml_probability = quant.get(
        "ml_probability"
    )

    if ml_probability is not None:

        if not 0 <= ml_probability <= 1:
            issues.append(
                "ML probability must be between 0 and 1."
            )

    # ==================================================
    # Model comparison
    # ==================================================

    model_comparison = quant.get(
        "model_comparison",
        {}
    )

    if not model_comparison:

        issues.append(
            "Model comparison results are missing."
        )

    else:

        for model_name, metrics in (
            model_comparison.items()
        ):

            # ------------------------------------------
            # ML metrics
            # ------------------------------------------

            for metric_name in [
                "accuracy",
                "precision",
                "recall",
                "f1_score"
            ]:

                value = metrics.get(
                    metric_name
                )

                if value is None:

                    issues.append(
                        f"{model_name}: "
                        f"{metric_name} is missing."
                    )

                elif not 0 <= value <= 1:

                    issues.append(
                        f"{model_name}: "
                        f"{metric_name} must be "
                        f"between 0 and 1."
                    )

            # ------------------------------------------
            # Financial metrics
            # ------------------------------------------

            strategy_return = metrics.get(
                "strategy_return"
            )

            gross_return = metrics.get(
                "gross_strategy_return"
            )

            buy_hold_return = metrics.get(
                "buy_hold_return"
            )

            excess_return = metrics.get(
                "excess_return"
            )

            backtest_sharpe = metrics.get(
                "backtest_sharpe_ratio"
            )

            backtest_drawdown = metrics.get(
                "backtest_max_drawdown"
            )

            trading_costs = metrics.get(
                "total_trading_costs"
            )

            position_changes = metrics.get(
                "number_of_position_changes"
            )

            research_score = metrics.get(
                "research_score"
            )

            # ------------------------------------------
            # Required financial fields
            # ------------------------------------------

            if strategy_return is None:

                issues.append(
                    f"{model_name}: "
                    "strategy return is missing."
                )

            if gross_return is None:

                issues.append(
                    f"{model_name}: "
                    "gross strategy return is missing."
                )

            if buy_hold_return is None:

                issues.append(
                    f"{model_name}: "
                    "buy-and-hold return is missing."
                )

            if excess_return is None:

                issues.append(
                    f"{model_name}: "
                    "excess return is missing."
                )

            if backtest_sharpe is None:

                issues.append(
                    f"{model_name}: "
                    "backtest Sharpe ratio is missing."
                )

            if backtest_drawdown is None:

                issues.append(
                    f"{model_name}: "
                    "backtest drawdown is missing."
                )

            if trading_costs is None:

                issues.append(
                    f"{model_name}: "
                    "trading cost information is missing."
                )

            if position_changes is None:

                issues.append(
                    f"{model_name}: "
                    "position-change information is missing."
                )

            # ------------------------------------------
            # Financial value validation
            # ------------------------------------------

            if backtest_sharpe is not None:

                if not isinstance(
                    backtest_sharpe,
                    (int, float)
                ):

                    issues.append(
                        f"{model_name}: "
                        "backtest Sharpe ratio "
                        "must be numeric."
                    )

            if backtest_drawdown is not None:

                if backtest_drawdown > 0:

                    issues.append(
                        f"{model_name}: "
                        "backtest maximum drawdown "
                        "should not be positive."
                    )

                if backtest_drawdown < -1:

                    issues.append(
                        f"{model_name}: "
                        "backtest drawdown cannot "
                        "be below -100%."
                    )

            if trading_costs is not None:

                if trading_costs < 0:

                    issues.append(
                        f"{model_name}: "
                        "trading costs cannot be negative."
                    )

            if position_changes is not None:

                if position_changes < 0:

                    issues.append(
                        f"{model_name}: "
                        "position changes cannot be negative."
                    )

            # ------------------------------------------
            # Research score
            # ------------------------------------------

            if research_score is not None:

                if not 0 <= research_score <= 1:

                    issues.append(
                        f"{model_name}: "
                        "research score must be "
                        "between 0 and 1."
                    )

    # ==================================================
    # Selected model metrics
    # ==================================================

    selected_metrics = model_comparison.get(
        best_model,
        {}
    )

    selected_f1 = selected_metrics.get(
        "f1_score"
    )

    selected_strategy_return = (
        quant.get(
            "backtest_strategy_return"
        )
    )

    selected_gross_return = (
        quant.get(
            "backtest_gross_strategy_return"
        )
    )

    selected_buy_hold = (
        quant.get(
            "backtest_buy_hold_return"
        )
    )

    selected_excess_return = (
        quant.get(
            "backtest_excess_return"
        )
    )

    selected_costs = (
        quant.get(
            "backtest_total_trading_costs"
        )
    )

    selected_position_changes = (
        quant.get(
            "backtest_position_changes"
        )
    )

    # ==================================================
    # 1. Excess-return consistency
    # ==================================================

    if (
        selected_strategy_return is not None
        and selected_buy_hold is not None
        and selected_excess_return is not None
    ):

        calculated_excess = (
            selected_strategy_return
            - selected_buy_hold
        )

        tolerance = 1e-6

        if abs(
            calculated_excess
            - selected_excess_return
        ) > tolerance:

            issues.append(
                "Backtest excess return does not "
                "match strategy return minus "
                "buy-and-hold return."
            )

    # ==================================================
    # 2. Gross vs net return consistency
    # ==================================================

    if (
        selected_gross_return is not None
        and selected_strategy_return is not None
        and selected_costs is not None
    ):

        # Trading costs are deducted through
        # compounding, so this is not expected
        # to equal gross - costs exactly.
        #
        # We only check the expected direction.

        if (
            selected_strategy_return
            > selected_gross_return
            + 1e-9
        ):

            issues.append(
                "Net strategy return exceeds "
                "gross strategy return."
            )

    # ==================================================
    # 3. Finance viability
    # ==================================================

    finance_viable = True

    finance_failure_reasons = []

    if selected_excess_return is not None:

        if selected_excess_return <= 0:

            finance_viable = False

            finance_failure_reasons.append(
                "Net strategy return does not "
                "outperform buy-and-hold after "
                "execution costs."
            )

    if selected_strategy_return is not None:

        if selected_strategy_return < 0:

            finance_viable = False

            finance_failure_reasons.append(
                "Net strategy return is negative."
            )

    # ==================================================
    # 4. Cost impact
    # ==================================================

    cost_impact = None

    if (
        selected_gross_return is not None
        and selected_strategy_return is not None
    ):

        cost_impact = (
            selected_gross_return
            - selected_strategy_return
        )

        if cost_impact > 0.05:

            warnings.append(
                "Execution costs reduce the "
                "strategy return by more than "
                "5 percentage points."
            )

    # ==================================================
    # 5. Prediction concentration
    # ==================================================

    prediction_bias_detected = False

    for model_name, metrics in (
        model_comparison.items()
    ):

        confusion = metrics.get(
            "confusion_matrix"
        )

        if (
            confusion is not None
            and len(confusion) == 2
            and len(confusion[0]) == 2
            and len(confusion[1]) == 2
        ):

            true_negative = confusion[0][0]
            false_positive = confusion[0][1]
            false_negative = confusion[1][0]
            true_positive = confusion[1][1]

            total_predictions = (
                true_negative
                + false_positive
                + false_negative
                + true_positive
            )

            if total_predictions > 0:

                predicted_positive = (
                    false_positive
                    + true_positive
                )

                positive_rate = (
                    predicted_positive
                    / total_predictions
                )

                # Flag models that predict the positive
                # class more than 90% of the time.

                if positive_rate > 0.90:

                    prediction_bias_detected = True

                    warnings.append(
                        f"{model_name} predicts the "
                        f"positive class in "
                        f"{positive_rate:.1%} of "
                        "test observations."
                    )

    # ==================================================
    # 6. Strong ML metrics but weak finance
    # ==================================================

    if (
        selected_f1 is not None
        and selected_f1 >= 0.60
        and selected_excess_return is not None
        and selected_excess_return <= 0
    ):

        finance_viable = False

        finance_failure_reasons.append(
            "The selected model has relatively "
            "strong F1 performance but does not "
            "generate positive net excess return."
        )

    # ==================================================
    # 7. High prediction probability warning
    # ==================================================

    if ml_probability is not None:

        if ml_probability > 0.90:

            warnings.append(
                "The final model probability is "
                "very high; calibration should be "
                "checked before interpreting it "
                "as a probability of future returns."
            )

    # ==================================================
    # Final critic result
    # ==================================================

    if finance_failure_reasons:

        issues.extend(
            finance_failure_reasons
        )

    # PASS / FAIL is based on actual issues.
    # Warnings alone do not cause failure.

    if issues:

        status = "FAIL"

    else:

        status = "PASS"

    critic_result = {

        "status": status,

        "issues": issues,

        "warnings": warnings,

        "issue_count": len(issues),

        "warning_count": len(warnings),

        "finance_viable": finance_viable,

        "finance_failure_reasons": (
            finance_failure_reasons
        ),

        "cost_impact": cost_impact,

        "prediction_bias_detected": (
            prediction_bias_detected
        )
    }

    return {
        "critic_result": critic_result
    }