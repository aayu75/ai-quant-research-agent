import json

from mcp.server.fastmcp import FastMCP

from mcp_server.market_tools import get_historical_prices


# Create MCP server
mcp = FastMCP("AI Quant Research Server")


@mcp.tool()
def get_market_data(
    ticker: str,
    period: str = "1y"
) -> str:
    """
    Retrieve historical market data for a stock.

    Args:
        ticker: Stock ticker such as AAPL, MSFT, or NVDA.
        period: Historical period such as 1mo, 3mo, 6mo, 1y, or 5y.
    """

    data = get_historical_prices(
        ticker=ticker,
        period=period
    )

    # yfinance can return MultiIndex columns such as:
    #
    # ('Close', 'AAPL')
    # ('Open', 'AAPL')
    #
    # Convert them into normal column names:
    #
    # Close
    # Open
    #
    if hasattr(data.columns, "levels"):

        data.columns = [
            column[0]
            if isinstance(column, tuple)
            else column
            for column in data.columns
        ]

    # Keep the MCP response manageable.
    latest_data = data.tail(500)

    # Convert DataFrame into JSON-compatible records.
    records = (
        latest_data
        .reset_index()
        .to_dict(orient="records")
    )

    return json.dumps(
        records,
        default=str
    )


if __name__ == "__main__":
    mcp.run()