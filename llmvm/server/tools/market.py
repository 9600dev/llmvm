import datetime as dt


class MarketHelpers():
    @staticmethod
    def get_stock_price(
        symbol: str,
        date: dt.datetime,
    ) -> float:
        """
        Get the closing price of the specified stock symbol at the specified date

        :param symbol: The symbol of the stock
        :param date: The date and time to get the stock price
        """
        import pandas as pd
        import yfinance as yf

        date_only = dt.datetime(date.year, date.month, date.day)

        def adjust_weekend(date):
            if date.weekday() == 5:  # Saturday
                return date - dt.timedelta(days=1)  # Adjust to Friday
            elif date.weekday() == 6:  # Sunday
                return date - dt.timedelta(days=2)  # Adjust to Friday
            else:
                return date

        def subtract_day(date):
            return adjust_weekend(date - dt.timedelta(days=1))

        date_only = adjust_weekend(date_only)
        result = pd.DataFrame()

        result = yf.download(symbol, start=date_only, end=date_only + dt.timedelta(days=5), repair=True)

        if len(result) == 0:
            date_only = adjust_weekend(subtract_day(date_only))
            result = yf.download(symbol, start=date_only, end=date_only + dt.timedelta(days=5), repair=True)

        if len(result) == 0:
            raise ValueError(f'either the symbol {symbol} does not exist, or there is no price data for date {date_only}.')
        close = result.head(1)['Close'].iloc[0]
        return float(close)

    @staticmethod
    def get_current_market_capitalization(symbol: str) -> str:
        """
        Get the current market capitalization of the specified stock symbol

        :param symbol: The symbol of the stock
        """
        import yfinance as yf

        ticker = yf.Ticker(symbol)
        return str(ticker.info.get('marketCap'))
