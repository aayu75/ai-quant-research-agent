import pandas as pd


def get_close_prices(data: pd.DataFrame) -> pd.Series:
    """
    Safely extract the Close price series from market data.

    yfinance can sometimes return Close prices as a DataFrame
    because of its column structure, so we normalize it here.
    """

    close_prices = data["Close"]

    if isinstance(close_prices, pd.DataFrame):
        close_prices = close_prices.iloc[:, 0]

    return close_prices.dropna()


def calculate_returns(data: pd.DataFrame) -> pd.Series:
    """
    Calculate daily percentage returns from closing prices.
    """

    close_prices = get_close_prices(data)

    returns = close_prices.pct_change()

    return returns.dropna()


def calculate_momentum(
    data: pd.DataFrame,
    window: int = 20
) -> float:
    """
    Calculate price momentum over a specified window.

    Formula:

        Momentum = (Current Price - Previous Price)
                   / Previous Price

    Example:
        Current price = 120
        Price 20 days ago = 100

        Momentum = (120 - 100) / 100
                  = 0.20
                  = 20%
    """

    close_prices = get_close_prices(data)

    if len(close_prices) <= window:
        raise ValueError(
            f"Not enough data to calculate {window}-day momentum."
        )

    current_price = float(close_prices.iloc[-1])
    previous_price = float(close_prices.iloc[-window - 1])

    momentum = (
        (current_price - previous_price)
        / previous_price
    )

    return momentum


def calculate_volatility(
    data: pd.DataFrame,
    annualize: bool = True
) -> float:
    """
    Calculate historical volatility from daily returns.

    If annualize=True, daily volatility is multiplied
    by sqrt(252), assuming approximately 252 trading days
    per year.
    """

    returns = calculate_returns(data)

    if len(returns) < 2:
        raise ValueError(
            "Not enough data to calculate volatility."
        )

    volatility = returns.std()

    if annualize:
        volatility = volatility * (252 ** 0.5)

    return float(volatility)


def calculate_sharpe_ratio(
    data: pd.DataFrame,
    risk_free_rate: float = 0.0
) -> float:
    """
    Calculate the annualized Sharpe ratio.

    Sharpe Ratio measures risk-adjusted performance.

    A higher value generally indicates better return
    relative to volatility, but interpretation depends
    on the period and strategy.
    """

    returns = calculate_returns(data)

    if len(returns) < 2:
        raise ValueError(
            "Not enough data to calculate Sharpe ratio."
        )

    # Convert annual risk-free rate to daily rate.
    daily_risk_free_rate = (
        (1 + risk_free_rate) ** (1 / 252)
    ) - 1

    excess_returns = returns - daily_risk_free_rate

    if excess_returns.std() == 0:
        raise ValueError(
            "Cannot calculate Sharpe ratio with zero volatility."
        )

    sharpe = (
        excess_returns.mean()
        / excess_returns.std()
    ) * (252 ** 0.5)

    return float(sharpe)


def calculate_max_drawdown(data: pd.DataFrame) -> float:
    """
    Calculate maximum drawdown from historical prices.

    Returns a negative decimal value.

    Example:
        -0.25 = 25% maximum drawdown.
    """

    close_prices = get_close_prices(data)

    if len(close_prices) < 2:
        raise ValueError(
            "Not enough data to calculate maximum drawdown."
        )

    running_max = close_prices.cummax()

    drawdowns = (
        close_prices - running_max
    ) / running_max

    return float(drawdowns.min())