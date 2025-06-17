"""
Currency Exchange Rate MCP Server

This server provides currency exchange rate information using the Open Exchange Rates API.
Built with FastMCP - the fast, Pythonic way to build MCP servers.

Usage:
    python currency_server.py
    or
    fastmcp run currency_server.py
"""

import httpx
from typing import Dict, List, Optional
from fastmcp import FastMCP

# Create the FastMCP server instance
mcp = FastMCP("Currency Exchange Rates")

@mcp.tool
async def get_exchange_rates(currency_code: str = "USD") -> Dict[str, float]:
    """
    Get current exchange rates for a given currency code.
    
    Args:
        currency_code: The base currency code (e.g., "USD", "EUR", "AUD")
        
    Returns:
        A dictionary of exchange rates with currency codes as keys and rates as values
    """
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

@mcp.tool
async def convert_currency(
    amount: float,
    from_currency: str,
    to_currency: str
) -> Dict[str, float | str]:
    """
    Convert an amount from one currency to another.
    
    Args:
        amount: The amount to convert
        from_currency: The source currency code (e.g., "USD")
        to_currency: The target currency code (e.g., "EUR")
        
    Returns:
        A dictionary containing the conversion result and exchange rate
    """
    # Get exchange rates for the source currency
    rates_data = await get_exchange_rates(from_currency)
    
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
async def get_supported_currencies() -> List[str]:
    """
    Get a list of all supported currency codes.
    
    Returns:
        A list of supported currency codes
    """
    # Use USD as base to get all available currencies
    rates_data = await get_exchange_rates("USD")
    
    if "error" in rates_data:
        return ["Error fetching currencies: " + rates_data.get("message", "Unknown error")]
    
    return sorted(list(rates_data["rates"].keys()))

@mcp.tool
async def get_currency_comparison(
    base_currency: str,
    target_currencies: Optional[List[str]] = None
) -> Dict[str, Dict[str, float] | str]:
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
    
    rates_data = await get_exchange_rates(base_currency)
    
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

# Add a resource to show server information
@mcp.resource("currency://server/info")
async def get_server_info() -> Dict[str, str]:
    """Get information about the currency exchange server."""
    return {
        "name": "Currency Exchange Rate Server",
        "version": "1.0.0",
        "description": "Provides real-time currency exchange rates and conversion tools",
        "data_source": "Open Exchange Rates API (https://open.er-api.com)",
        "update_frequency": "Daily",
        "available_tools": [
            "get_exchange_rates",
            "convert_currency", 
            "get_supported_currencies",
            "get_currency_comparison"
        ]
    }

# Add a prompt for common currency queries
@mcp.prompt
def currency_analysis_prompt(currencies: List[str]) -> str:
    """Generate a prompt for analyzing multiple currencies."""
    return f"""Please analyze the following currencies and their recent exchange rates: {', '.join(currencies)}.
    
Consider:
1. Current exchange rates relative to major currencies
2. Which currencies are strongest/weakest
3. Any notable patterns or relationships
4. Practical implications for international transactions

Use the available currency tools to gather the necessary data."""

if __name__ == "__main__":
    # Run the server
    mcp.run()
