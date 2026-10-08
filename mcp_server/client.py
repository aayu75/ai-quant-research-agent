import asyncio
import json
import sys
from pathlib import Path

import pandas as pd

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


PROJECT_ROOT = Path(__file__).resolve().parents[1]


async def call_market_data_tool(
    ticker: str,
    period: str = "1y"
):
    """
    Start the MCP server and call
    the get_market_data tool.
    """

    server_params = StdioServerParameters(
        command=sys.executable,
        args=[
            "-m",
            "mcp_server.server"
        ],
        cwd=str(PROJECT_ROOT)
    )

    async with stdio_client(
        server_params
    ) as (
        read_stream,
        write_stream
    ):

        async with ClientSession(
            read_stream,
            write_stream
        ) as session:

            await session.initialize()

            result = await session.call_tool(
                "get_market_data",
                {
                    "ticker": ticker,
                    "period": period
                }
            )

            return result


def get_market_data(
    ticker: str,
    period: str = "1y"
):
    """
    Synchronous wrapper around
    the asynchronous MCP client.
    """

    result = asyncio.run(
        call_market_data_tool(
            ticker,
            period
        )
    )

    # Check whether MCP reported a tool error.
    if result.isError:
        error_message = result.content[0].text

        raise RuntimeError(
            f"MCP tool failed: {error_message}"
        )

    # Extract text returned by the MCP tool.
    text_content = result.content[0].text

    # Convert JSON string into Python objects.
    records = json.loads(text_content)

    # Convert records into DataFrame.
    data = pd.DataFrame(records)

    # Restore Date as datetime index.
    if "Date" in data.columns:

        data["Date"] = pd.to_datetime(
            data["Date"]
        )

        data = data.set_index("Date")

    return data