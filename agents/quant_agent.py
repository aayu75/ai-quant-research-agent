import pandas as pd

from graph.state import AgentState

from mcp_server.quant_tools import (
    get_close_prices,
    calculate_returns,
    calculate_momentum
)

from mcp_server.backtest_tools import (
    run_backtest
)

from ml.train import (
    train_multiple_models
)

from ml.evaluate import (
    compare_models
)

from ml.predict import (
    predict_test_period,
    predict_next_day,
    retrain_model_on_all_data
)


TRANSACTION_COST = 0.001
SLIPPAGE = 0.0005


def quant_agent(
    state: AgentState
) -> AgentState:

    market_data = state[
        "market_data"
    ]

    data = market_data[
        "data"
    ]

    try:

        # ==================================================
        # 1. Traditional quantitative analysis
        # ==================================================

        close_prices = get_close_prices(
            data
        )

        returns = calculate_returns(
            data
        )

        momentum = calculate_momentum(
            data,
            window=20
        )

        # ==================================================
        # 2. Train candidate models
        # ==================================================

        trained_models = train_multiple_models(
            data
        )

        # ==================================================
        # 3. Identify test period
        # ==================================================

        first_model = next(
            iter(
                trained_models.values()
            )
        )

        common_test_index = (
            first_model[
                "X_test"
            ].index
        )

        test_data = data.loc[
            data.index.isin(
                common_test_index
            )
        ].copy()

        if test_data.empty:

            raise ValueError(
                "Test-period market data is empty."
            )

        # ==================================================
        # 4. Finance-aware model comparison
        # ==================================================

        model_comparison = (
            compare_models(

                trained_models=(
                    trained_models
                ),

                test_data=test_data,

                transaction_cost=(
                    TRANSACTION_COST
                ),

                slippage=(
                    SLIPPAGE
                )
            )
        )

        # ==================================================
        # 5. Select best model
        # ==================================================

        best_model_name = (
            model_comparison[
                "best_model"
            ]
        )

        best_model_data = (
            trained_models[
                best_model_name
            ]
        )

        evaluation_model = (
            best_model_data[
                "model"
            ]
        )

        X_test = (
            best_model_data[
                "X_test"
            ]
        )

        # ==================================================
        # 6. Test-period predictions
        # ==================================================

        test_predictions, _ = (
            predict_test_period(
                evaluation_model,
                X_test
            )
        )

        predictions = pd.Series(
            test_predictions,
            index=X_test.index
        )

        # ==================================================
        # 7. Backtest selected model
        # ==================================================

        selected_test_data = (
            data.loc[
                data.index.isin(
                    X_test.index
                )
            ].copy()
        )

        backtest_result = run_backtest(

            data=selected_test_data,

            predictions=predictions,

            transaction_cost=(
                TRANSACTION_COST
            ),

            slippage=(
                SLIPPAGE
            )
        )

        # ==================================================
        # 8. Research score
        # ==================================================

        selected_model_metrics = (
            model_comparison[
                "models"
            ][
                best_model_name
            ]
        )

        research_score = (
            selected_model_metrics[
                "research_score"
            ]
        )

        # ==================================================
        # 9. Retrain selected model on ALL data
        # ==================================================

        final_model_data = (
            retrain_model_on_all_data(

                data=data,

                model_name=best_model_name
            )
        )

        final_model = (
            final_model_data[
                "model"
            ]
        )

        # ==================================================
        # 10. Next-day prediction
        # ==================================================

        prediction_result = (
            predict_next_day(

                model=final_model,

                data=data
            )
        )

        prediction = (
            prediction_result[
                "prediction"
            ]
        )

        probability = (
            prediction_result[
                "probability"
            ]
        )

        direction = (
            "POSITIVE"
            if prediction == 1
            else "NON-POSITIVE"
        )

        # ==================================================
        # 11. Final quantitative analysis
        # ==================================================

        quant_analysis = {

            # ------------------------------------------
            # Traditional quantitative analysis
            # ------------------------------------------

            "latest_price": float(
                close_prices.iloc[-1]
            ),

            "average_daily_return": float(
                returns.mean()
            ),

            "momentum_20d": float(
                momentum
            ),

            "number_of_observations": len(
                data
            ),

            # ------------------------------------------
            # Model selection
            # ------------------------------------------

            "best_model": (
                best_model_name
            ),

            "research_score": float(
                research_score
            ),

            "model_comparison": (
                model_comparison[
                    "models"
                ]
            ),

            # ------------------------------------------
            # Final prediction
            # ------------------------------------------

            "ml_prediction": int(
                prediction
            ),

            "ml_direction": direction,

            "ml_probability": float(
                probability
            ),

            "final_model_training_observations": (
                final_model_data[
                    "observations"
                ]
            ),

            # ------------------------------------------
            # Backtest
            # ------------------------------------------

            "backtest_strategy_return": (
                backtest_result[
                    "strategy_return"
                ]
            ),

            "backtest_gross_strategy_return": (
                backtest_result[
                    "gross_strategy_return"
                ]
            ),

            "backtest_buy_hold_return": (
                backtest_result[
                    "buy_hold_return"
                ]
            ),

            "backtest_excess_return": (
                backtest_result[
                    "strategy_return"
                ]
                -
                backtest_result[
                    "buy_hold_return"
                ]
            ),

            "backtest_max_drawdown": (
                backtest_result[
                    "maximum_drawdown"
                ]
            ),

            "backtest_sharpe_ratio": (
                backtest_result[
                    "sharpe_ratio"
                ]
            ),

            "backtest_transaction_cost": (
                backtest_result[
                    "transaction_cost"
                ]
            ),

            "backtest_slippage": (
                backtest_result[
                    "slippage"
                ]
            ),

            "backtest_total_trading_costs": (
                backtest_result[
                    "total_trading_costs"
                ]
            ),

            "backtest_position_changes": (
                backtest_result[
                    "number_of_position_changes"
                ]
            ),

            "backtest_observations": (
                backtest_result[
                    "observations"
                ]
            )
        }

        return {

            "quant_analysis":
                quant_analysis,

            "errors": []
        }

    except Exception as error:

        return {

            "quant_analysis": {},

            "errors": [
                str(error)
            ]
        }