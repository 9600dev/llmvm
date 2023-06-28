import yfinance as yf


class MarketHelpers():
    @staticmethod
    def get_latest_stock_price(symbol: str):
        """
        Get the latest price of the symbol

        :param symbol: The symbol of the stock
        """

        ticker = yf.Ticker(symbol)
        return ticker.history().tail(1)['Close'].iloc[0]

    @staticmethod
    def get_last_week_average_price(symbol: str):
        """
        Get the average price of the symbol for the last week
        :param symbol: The symbol of the stock
        """
        ticker = yf.Ticker(symbol)
        return ticker.history(period='1wk')['Close'].mean()
