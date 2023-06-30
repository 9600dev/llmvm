import yfinance as yf


class MarketHelpers():
    @staticmethod
    def get_stock_price(symbol: str) -> float:
        """
        Get the current or latest price of the specified stock symbol

        :param symbol: The symbol of the stock
        """

        ticker = yf.Ticker(symbol)
        return ticker.history().tail(1)['Close'].iloc[0]

    @staticmethod
    def get_market_capitalization(symbol: str) -> str:
        """
        Get the current market capitalization of the specified stock symbol

        :param symbol: The symbol of the stock
        """
        ticker = yf.Ticker(symbol)
        return str(ticker.info.get('marketCap'))
