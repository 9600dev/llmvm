import datetime as dt
from typing import Dict


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

    @staticmethod
    def get_stock_price_history(
        symbol: str,
        start_date: dt.datetime,
        end_date: dt.datetime
    ) -> Dict[dt.datetime, float]:
        """
        Get the closing prices of the specified stock symbol between the specified start and end dates

        :param symbol: The symbol of the stock
        :type symbol: str
        :param start_date: The start date and time to of the price series
        :type start_date: dt.datetime
        :param end_date: The end date and time of the price series
        :type end_date: dt.datetime
        :return: A dictionary mapping dates to closing prices. Keys are datetimes and values are floats.
        """
        import pandas as pd
        import yfinance as yf

        start_date_only = dt.datetime(start_date.year, start_date.month, start_date.day)
        end_date_only = dt.datetime(end_date.year, end_date.month, end_date.day)

        def adjust_weekend(date):
            if date.weekday() == 5:  # Saturday
                return date - dt.timedelta(days=1)  # Adjust to Friday
            elif date.weekday() == 6:  # Sunday
                return date - dt.timedelta(days=2)  # Adjust to Friday
            else:
                return date

        def subtract_day(date):
            return adjust_weekend(date - dt.timedelta(days=1))

        start_date_only = adjust_weekend(start_date_only)
        end_date_only = adjust_weekend(end_date_only)
        result = pd.DataFrame()

        result = yf.download(symbol, start=start_date_only, end=end_date_only + dt.timedelta(days=5), repair=True)

        if len(result) == 0:
            date_only = adjust_weekend(subtract_day(start_date_only))
            result = yf.download(symbol, start=date_only, end=end_date_only + dt.timedelta(days=5), repair=True)

        if len(result) == 0:
            raise ValueError(f'either the symbol {symbol} does not exist, or there is no price data for date {date_only}.')

        result = result.loc[start_date_only:end_date_only]
        # return result as a dictionary
        return result['Close'].to_dict()

    @staticmethod
    def get_stock_volatility(symbol: str, days: int) -> float:
        """
        Calculate the volatility of a stock over a given number of days.

        :param symbol: The stock symbol
        :type symbol: str
        :param days: The number of days to calculate volatility over
        :type days: int
        :return: The annualized volatility as a percentage
        :rtype: float
        """
        import yfinance as yf
        import numpy as np

        # Get the end date (today)
        end_date = dt.datetime.now()

        # Calculate the start date
        start_date = end_date - dt.timedelta(days=days)

        # Download the stock data
        stock_data = yf.download(symbol, start=start_date, end=end_date, repair=True)

        # Calculate daily returns
        daily_returns = stock_data['Close'].pct_change().dropna()

        # Calculate the standard deviation of daily returns
        daily_volatility = np.std(daily_returns)

        # Annualize the volatility (assuming 252 trading days in a year)
        annualized_volatility = daily_volatility * np.sqrt(252)

        # Convert to percentage
        return annualized_volatility * 100