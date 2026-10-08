import pandas as pd
import numpy as np


def calculate_max_drawdown(
    equity_curve: pd.Series
) -> float:
    """
    Calculate maximum drawdown from an equity curve.
    """

    equity_curve = equity_curve.dropna()

    if len(equity_curve) < 2:
        return 0.0

    running_max = equity_curve.cummax()

    drawdown = (
        equity_curve - running_max
    ) / running_max

    return float(drawdown.min())


def calculate_sharpe_ratio(
    returns: pd.Series
) -> float:
    """
    Calculate annualized Sharpe ratio.
    """

    returns = returns.dropna()

    if len(returns) < 2:
        return 0.0

    volatility = returns.std()

    if volatility == 0:
        return 0.0

    return float(
        (
            returns.mean()
            / volatility
        )
        * np.sqrt(252)
    )


def run_backtest(
    data: pd.DataFrame,
    predictions: pd.Series,
    transaction_cost: float = 0.001,
    slippage: float = 0.0005
) -> dict:
    """
    Run a simple long/flat quantitative strategy.

    Parameters
    ----------
    data:
        Historical market data containing Close.

    predictions:
        Model predictions indexed by date.
        1 = long
        0 = flat

    transaction_cost:
        Cost applied whenever the position changes.

        Example:
        0.001 = 0.10%

    slippage:
        Execution slippage applied whenever
        the position changes.

        Example:
        0.0005 = 0.05%

    Returns
    -------
    dict
        Backtest performance metrics.
    """

    # ==================================================
    # 1. Copy and normalize data
    # ==================================================

    df = data.copy()

    if hasattr(
        df.columns,
        "levels"
    ):

        df.columns = [
            column[0]
            if isinstance(column, tuple)
            else column
            for column in df.columns
        ]

    if "Close" not in df.columns:

        raise ValueError(
            "Close column is required."
        )

    if df.empty:

        raise ValueError(
            "Backtest data cannot be empty."
        )

    if transaction_cost < 0:

        raise ValueError(
            "Transaction cost cannot be negative."
        )

    if slippage < 0:

        raise ValueError(
            "Slippage cannot be negative."
        )

    df = df.sort_index()

    # ==================================================
    # 2. Market returns
    # ==================================================

    df["market_return"] = (
        df["Close"].pct_change()
    )

    # ==================================================
    # 3. Align predictions
    # ==================================================

    predictions = predictions.reindex(
        df.index
    )

    # ==================================================
    # 4. Position
    # ==================================================

    # Prediction at time t is used for
    # the return at t+1.
    #
    # This prevents look-ahead bias.

    df["position"] = (
        predictions.shift(1)
    )

    df["position"] = (
        df["position"].fillna(0)
    )

    # Make sure position is either
    # 0 or 1.

    df["position"] = (
        df["position"]
        .clip(0, 1)
    )

    # ==================================================
    # 5. Position changes
    # ==================================================

    df["position_change"] = (
        df["position"]
        .diff()
        .abs()
        .fillna(
            df["position"].abs()
        )
    )

    # ==================================================
    # 6. Gross strategy return
    # ==================================================

    df["gross_strategy_return"] = (
        df["position"]
        * df["market_return"]
    )

    # ==================================================
    # 7. Trading costs
    # ==================================================

    total_cost = (
        transaction_cost
        + slippage
    )

    df["trading_cost"] = (
        df["position_change"]
        * total_cost
    )

    # ==================================================
    # 8. Net strategy return
    # ==================================================

    df["strategy_return"] = (
        df["gross_strategy_return"]
        - df["trading_cost"]
    )

    # ==================================================
    # 9. Equity curves
    # ==================================================

    df["strategy_equity"] = (
        1
        + df["strategy_return"].fillna(0)
    ).cumprod()

    df["buy_hold_equity"] = (
        1
        + df["market_return"].fillna(0)
    ).cumprod()

    # ==================================================
    # 10. Total returns
    # ==================================================

    strategy_return = (
        df["strategy_equity"].iloc[-1]
        - 1
    )

    buy_hold_return = (
        df["buy_hold_equity"].iloc[-1]
        - 1
    )

    # ==================================================
    # 11. Risk metrics
    # ==================================================

    maximum_drawdown = (
        calculate_max_drawdown(
            df["strategy_equity"]
        )
    )

    sharpe_ratio = (
        calculate_sharpe_ratio(
            df["strategy_return"]
        )
    )

    # ==================================================
    # 12. Trading statistics
    # ==================================================

    number_of_position_changes = int(
        (df["position_change"] > 0)
        .sum()
    )

    total_transaction_costs = float(
        df["trading_cost"].sum()
    )

    gross_strategy_return = (
        df["gross_strategy_return"]
        .add(1)
        .prod()
        - 1
    )

    # ==================================================
    # 13. Result
    # ==================================================

    return {

        "strategy_return": float(
            strategy_return
        ),

        "gross_strategy_return": float(
            gross_strategy_return
        ),

        "buy_hold_return": float(
            buy_hold_return
        ),

        "maximum_drawdown": float(
            maximum_drawdown
        ),

        "sharpe_ratio": float(
            sharpe_ratio
        ),

        "transaction_cost": float(
            transaction_cost
        ),

        "slippage": float(
            slippage
        ),

        "total_trading_costs": (
            total_transaction_costs
        ),

        "number_of_position_changes": (
            number_of_position_changes
        ),

        "observations": int(
            len(df)
        )
    }