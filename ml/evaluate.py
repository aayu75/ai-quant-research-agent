from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

import pandas as pd

from mcp_server.backtest_tools import run_backtest


def evaluate_model(
    model,
    X_test,
    y_test
):
    """
    Evaluate a classification model using
    standard ML metrics.
    """

    predictions = model.predict(
        X_test
    )

    return {

        "accuracy": float(
            accuracy_score(
                y_test,
                predictions
            )
        ),

        "precision": float(
            precision_score(
                y_test,
                predictions,
                zero_division=0
            )
        ),

        "recall": float(
            recall_score(
                y_test,
                predictions,
                zero_division=0
            )
        ),

        "f1_score": float(
            f1_score(
                y_test,
                predictions,
                zero_division=0
            )
        ),

        "confusion_matrix": (
            confusion_matrix(
                y_test,
                predictions
            ).tolist()
        )
    }


def evaluate_model_with_backtest(
    model,
    X_test,
    y_test,
    test_data,
    transaction_cost=0.001,
    slippage=0.0005
):
    """
    Evaluate a model using:

    1. ML metrics
    2. Backtest performance
    3. Transaction costs
    4. Slippage
    """

    ml_metrics = evaluate_model(
        model=model,
        X_test=X_test,
        y_test=y_test
    )

    predictions = model.predict(
        X_test
    )

    predictions_series = pd.Series(
        predictions,
        index=X_test.index
    )

    backtest_result = run_backtest(
        data=test_data,
        predictions=predictions_series,
        transaction_cost=transaction_cost,
        slippage=slippage
    )

    strategy_return = (
        backtest_result[
            "strategy_return"
        ]
    )

    buy_hold_return = (
        backtest_result[
            "buy_hold_return"
        ]
    )

    excess_return = (
        strategy_return
        - buy_hold_return
    )

    return {

        **ml_metrics,

        "strategy_return": float(
            strategy_return
        ),

        "gross_strategy_return": float(
            backtest_result[
                "gross_strategy_return"
            ]
        ),

        "buy_hold_return": float(
            buy_hold_return
        ),

        "excess_return": float(
            excess_return
        ),

        "backtest_sharpe_ratio": float(
            backtest_result[
                "sharpe_ratio"
            ]
        ),

        "backtest_max_drawdown": float(
            backtest_result[
                "maximum_drawdown"
            ]
        ),

        "transaction_cost": float(
            backtest_result[
                "transaction_cost"
            ]
        ),

        "slippage": float(
            backtest_result[
                "slippage"
            ]
        ),

        "total_trading_costs": float(
            backtest_result[
                "total_trading_costs"
            ]
        ),

        "number_of_position_changes": int(
            backtest_result[
                "number_of_position_changes"
            ]
        ),

        "backtest_observations": int(
            backtest_result[
                "observations"
            ]
        )
    }


def calculate_research_score(
    metrics
):
    """
    Calculate finance-aware research score.

    Components:

    - F1
    - Excess return
    - Sharpe ratio
    - Maximum drawdown
    """

    f1 = metrics[
        "f1_score"
    ]

    excess_return = metrics[
        "excess_return"
    ]

    sharpe = metrics[
        "backtest_sharpe_ratio"
    ]

    drawdown = metrics[
        "backtest_max_drawdown"
    ]

    # --------------------------------------------------
    # Excess return
    # --------------------------------------------------

    excess_return_score = max(
        min(excess_return, 1.0),
        -1.0
    )

    excess_return_score = (
        excess_return_score + 1
    ) / 2

    # --------------------------------------------------
    # Sharpe
    # --------------------------------------------------

    sharpe_score = max(
        min(sharpe, 3.0),
        -3.0
    )

    sharpe_score = (
        sharpe_score + 3
    ) / 6

    # --------------------------------------------------
    # Drawdown
    # --------------------------------------------------

    drawdown_score = (
        1 + drawdown
    )

    drawdown_score = max(
        min(drawdown_score, 1.0),
        0.0
    )

    # --------------------------------------------------
    # Final score
    # --------------------------------------------------

    score = (

        0.25 * f1

        + 0.30
        * excess_return_score

        + 0.25
        * sharpe_score

        + 0.20
        * drawdown_score
    )

    return float(
        score
    )


def compare_models(
    trained_models,
    test_data,
    transaction_cost=0.001,
    slippage=0.0005
):
    """
    Compare candidate models using
    ML + realistic backtesting.
    """

    results = {}

    for name, model_data in (
        trained_models.items()
    ):

        metrics = (
            evaluate_model_with_backtest(

                model=model_data[
                    "model"
                ],

                X_test=model_data[
                    "X_test"
                ],

                y_test=model_data[
                    "y_test"
                ],

                test_data=test_data,

                transaction_cost=(
                    transaction_cost
                ),

                slippage=(
                    slippage
                )
            )
        )

        metrics[
            "research_score"
        ] = calculate_research_score(
            metrics
        )

        results[name] = metrics

    best_model = max(
        results,
        key=lambda name:
        results[name][
            "research_score"
        ]
    )

    return {

        "models": results,

        "best_model": best_model
    }