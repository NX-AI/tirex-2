# Copyright (c) NXAI GmbH.
# This software may be used and distributed according to the terms of the NXAI Community License Agreement.

import json
import re

import pytest
from fastmcp import Client

from conftest import (
    MEDIAN_QUANTILE_INDEX,
    PREDICTION_LENGTH,
    REFERENCE,
    TARGET,
    assert_forecast_close,
)


@pytest.fixture
async def main_mcp_client(api_server):
    async with Client(f"{api_server}/mcp") as mcp_client:
        yield mcp_client


async def test_list_tools(main_mcp_client: Client):
    list_tools = await main_mcp_client.list_tools()

    assert len(list_tools) == 2


async def test_mcp(main_mcp_client):
    params = {"context": TARGET, "prediction_length": PREDICTION_LENGTH}

    result = await main_mcp_client.call_tool("tirex_model", params)

    assert result.data is not None
    assert "TiRex Forecast Results" in result.data

    result_list = json.loads(re.search(r"Forecasted values:\s*(\[.*?\])", result.data, re.DOTALL).group(1))

    # MCP is single-series and returns the median quantile, matching the univariate reference.
    assert_forecast_close(result_list, REFERENCE.univariate[0, MEDIAN_QUANTILE_INDEX])
