from mcp_server.client import get_market_data


def test_mcp_market_data():

    result = get_market_data(
        ticker="AAPL",
        period="5d"
    )

    assert result is not None