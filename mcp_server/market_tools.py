import yfinance as yf


def get_historical_prices(
    ticker: str,
    period: str = "1y"
):
    """
    Retrieve historical OHLCV market data for a stock.
    """

    if not ticker:
        raise ValueError("Ticker cannot be empty.")

    data = yf.download(
        ticker,
        period=period,
        auto_adjust=False,
        progress=False
    )

    if data.empty:
        raise ValueError(
            f"No market data found for ticker: {ticker}"
        )

    return data