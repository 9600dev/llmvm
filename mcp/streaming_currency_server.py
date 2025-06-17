import httpx
import click
from typing import Dict, List, Optional, Any
from fastmcp import FastMCP

# Create the FastMCP server instance
mcp = FastMCP("Currency Exchange Rates (Streaming)")

async def _get_exchange_rates(currency_code: str = "USD") -> Dict[str, Any]:
    url = f"https://open.er-api.com/v6/latest/{currency_code.upper()}"

    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(url)
            response.raise_for_status()

            data = response.json()

            if data.get("result") == "success":
                return {
                    "base": data["base_code"],
                    "last_updated": data["time_last_update_utc"],
                    "rates": data["rates"]
                }
            else:
                return {
                    "error": f"Failed to get rates for {currency_code}",
                    "message": data.get("error-type", "Unknown error")
                }

        except httpx.HTTPStatusError as e:
            return {
                "error": f"HTTP error occurred",
                "status_code": e.response.status_code,
                "message": str(e)
            }
        except Exception as e:
            return {
                "error": "Failed to fetch exchange rates",
                "message": str(e)
            }

async def _convert_currency(amount: float, from_currency: str, to_currency: str) -> Dict[str, Any]:
    rates_data = await _get_exchange_rates(from_currency)

    if "error" in rates_data:
        return rates_data

    rates = rates_data["rates"]

    if to_currency.upper() not in rates:
        return {
            "error": f"Currency {to_currency} not found",
            "available_currencies": list(rates.keys())[:10] + ["..."]
        }

    exchange_rate = rates[to_currency.upper()]
    converted_amount = amount * exchange_rate

    return {
        "from_currency": from_currency.upper(),
        "to_currency": to_currency.upper(),
        "amount": amount,
        "converted_amount": round(converted_amount, 2),
        "exchange_rate": exchange_rate,
        "last_updated": rates_data["last_updated"]
    }

@mcp.tool
async def get_exchange_rates(currency_code: str = "USD") -> Dict[str, Any]:
    """
    Get current exchange rates for a given currency code.

    Args:
        currency_code: The base currency code (e.g., "USD", "EUR", "AUD")

    Returns:
        A dictionary of exchange rates with currency codes as keys and rates as values
    """
    return await _get_exchange_rates(currency_code)

@mcp.tool
async def convert_currency(
    amount: float,
    from_currency: str,
    to_currency: str
) -> Dict[str, Any]:
    """
    Convert an amount from one currency to another.

    Args:
        amount: The amount to convert
        from_currency: The source currency code (e.g., "USD")
        to_currency: The target currency code (e.g., "EUR")

    Returns:
        A dictionary containing the conversion result and exchange rate
    """
    return await _convert_currency(amount, from_currency, to_currency)

@mcp.tool
async def get_supported_currencies() -> List[str]:
    """
    Get a list of all supported currency codes.

    Returns:
        A list of supported currency codes
    """
    # Use USD as base to get all available currencies
    rates_data = await _get_exchange_rates("USD")

    if "error" in rates_data:
        return ["Error fetching currencies: " + rates_data.get("message", "Unknown error")]

    return sorted(list(rates_data["rates"].keys()))

@mcp.tool
async def get_currency_comparison(
    base_currency: str,
    target_currencies: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Compare exchange rates between a base currency and multiple target currencies.

    Args:
        base_currency: The base currency code
        target_currencies: List of target currency codes (defaults to major currencies if not provided)

    Returns:
        A dictionary with exchange rate comparisons
    """
    if target_currencies is None:
        # Default to major currencies
        target_currencies = ["USD", "EUR", "GBP", "JPY", "AUD", "CAD", "CHF", "CNY"]

    rates_data = await _get_exchange_rates(base_currency)

    if "error" in rates_data:
        return rates_data

    rates = rates_data["rates"]
    comparison = {}

    for currency in target_currencies:
        if currency.upper() in rates:
            comparison[currency.upper()] = {
                "rate": rates[currency.upper()],
                "inverse_rate": round(1 / rates[currency.upper()], 6) if rates[currency.upper()] != 0 else 0
            }

    return {
        "base_currency": base_currency.upper(),
        "last_updated": rates_data["last_updated"],
        "comparisons": comparison
    }

@mcp.tool
async def stream_rate_updates(
    currency_pairs: List[str],
    interval_seconds: int = 60
) -> Dict[str, Any]:
    """
    Stream exchange rate updates for specified currency pairs.
    Note: This returns current rates - in a real streaming scenario,
    this would be called periodically by the client.

    Args:
        currency_pairs: List of currency pairs like ["USD/EUR", "GBP/USD"]
        interval_seconds: Update interval in seconds (for client reference)

    Returns:
        Current rates for the specified pairs
    """
    results = {}

    for pair in currency_pairs:
        if "/" in pair:
            from_curr, to_curr = pair.split("/")
            conversion = await _convert_currency(1.0, from_curr, to_curr)

            if "error" not in conversion:
                results[pair] = {
                    "rate": conversion["exchange_rate"],
                    "timestamp": conversion["last_updated"]
                }
            else:
                results[pair] = {"error": conversion.get("message", "Unknown error")}
        else:
            results[pair] = {"error": "Invalid pair format. Use FROM/TO (e.g., USD/EUR)"}

    return results

# Add a resource to show server information
@mcp.resource("currency://server/info")
async def get_server_info() -> Dict[str, Any]:
    """Get information about the currency exchange server."""
    return {
        "name": "Currency Exchange Rate Server (Streaming)",
        "version": "1.0.1",
        "transport": "SSE (Server-Sent Events)",
        "description": "Provides real-time currency exchange rates via SSE transport",
        "data_source": "Open Exchange Rates API (https://open.er-api.com)",
        "update_frequency": "Daily",
        "streaming_support": "Yes - use stream_rate_updates tool",
        "available_tools": [
            "get_exchange_rates",
            "convert_currency",
            "get_supported_currencies",
            "get_currency_comparison",
            "stream_rate_updates"
        ]
    }

# Add a resource for current popular exchange rates
@mcp.resource("currency://rates/popular")
async def get_popular_rates() -> Dict[str, Dict[str, float]]:
    """Get current popular exchange rates."""
    popular_pairs = [
        ("USD", ["EUR", "GBP", "JPY", "AUD", "CAD"]),
        ("EUR", ["USD", "GBP", "CHF"]),
        ("GBP", ["USD", "EUR"])
    ]

    results = {}
    for base, targets in popular_pairs:
        rates_data = await _get_exchange_rates(base)
        if "error" not in rates_data:
            results[base] = {
                target: rates_data["rates"][target]
                for target in targets
                if target in rates_data["rates"]
            }

    return results

# Add a prompt for real-time monitoring
@mcp.prompt
def currency_monitoring_prompt(pairs: List[str], threshold: float) -> str:
    """Generate a prompt for monitoring currency pairs."""
    return f"""Monitor the following currency pairs for significant changes: {', '.join(pairs)}.

Alert me when any exchange rate changes by more than {threshold}% from its current value.

Use the stream_rate_updates tool to check rates periodically and track changes.
Provide both the current rate and the percentage change when alerting."""

@click.command()
@click.option(
    "-p", "--port",
    default=8070,
    type=int,
    help="TCP port number to listen on"
)
def main(port):
    print(f"Starting Currency Exchange Rate MCP Server (SSE) on port {port}")
    print(f"Server will be available at: http://localhost:{port}/sse")
    print("Press Ctrl+C to stop\n")

    mcp.run(
        transport="sse",
        host="127.0.0.1",  # Use localhost for security
        port=port,
    )

if __name__ == "__main__":
    main()
