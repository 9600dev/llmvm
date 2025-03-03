import httpx
from mcp.server.fastmcp import FastMCP


mcp = FastMCP("currency")

@mcp.tool()
async def get_currency_rates(currency_code: str) -> str:
    """
    Returns the most popular currency rates for a given currency code as a natural language string. Try and use at least 3 decimal places.
    Example: rates = BCL.get_currency_rates(currency_code="AUD")
    answer(rates)

    :param currency_code: The currency code to get the rates for.
    :type currency_code: str
    :return: currency rates as a string
    :rtype: str
    """
    result = httpx.get(f"https://open.er-api.com/v6/latest/{currency_code}").json()
    str_result = f"Currency rates for {currency_code}:\n"
    if 'rates' in result:
        str_result += str(result['rates'])
    else:
        str_result += str(result)
    return str_result


if __name__ == "__main__":
    mcp.run(transport='stdio')


