import httpx
from mcp.server.fastmcp import FastMCP

# Initialize the MCP server with metadata, documentation, and prompt messages.
mcp = FastMCP(
    name="CurrencyServer",
    description="A server that provides currency exchange rates using the open.er-api.com API.",
    version="1.0.0"
)

@mcp.prompt("currency_code")
async def get_currency_rates_prompt(currency_code: str) -> str:
    return f"""
    Please get the currency_code exchange rates for the currency code {currency_code} as a natural language string.
    """

@mcp.tool()
async def get_currency_rates(currency_code: str) -> str:
    """
    Returns the currency exchange rates for the given currency code as a natural language string.
    Example usage: rates = get_currency_rates(currency_code="AUD")
    answer(rates)

    :param currency_code: The three-letter currency code to fetch rates for.
    :return: A formatted string with the exchange rates, each value shown with at least 3 decimal places.
    """
    try:
        response = httpx.get(f"https://open.er-api.com/v6/latest/{currency_code}")
        result = response.json()
    except Exception as e:
        return f"Error fetching currency rates: {str(e)}"

    str_result = f"Currency rates for {currency_code}:\n"
    if "rates" in result:
        rates = result["rates"]
        # Format each rate to have at least 3 decimal places.
        formatted_rates = {code: f"{rate:.3f}" for code, rate in rates.items()}
        str_result += "\n".join(f"{code}: {rate}" for code, rate in formatted_rates.items())
    else:
        str_result += str(result)
    return str_result

if __name__ == "__main__":
    # Run the MCP server using STDIO transport for full MCP support.
    mcp.run(transport="stdio")
