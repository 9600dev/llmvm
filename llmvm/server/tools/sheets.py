import gspread
from gspread import Spreadsheet, Worksheet, Cell
from gspread.utils import ValueInputOption, ValueRenderOption
import pandas as pd
from typing import List, Dict, Any, Optional, Union, Tuple
import gspread_dataframe as gd


class GoogleSheetsManager:
    """
    A unified interface for working with Google Sheets using gspread and gspread-dataframe.

    This class provides a simplified API for common operations on Google Sheets,
    including creating spreadsheets, managing worksheets, and working with data
    using both 2D arrays and pandas DataFrames.
    It uses gspread (https://github.com/burnash/gspread)
    and gspread-dataframe (https://github.com/jonmmease/gspread-dataframe) for the underlying operations.
    """
    def __init__(
        self,
    ):
        oauth_error = None
        service_error = None

        try:
            self.client = gspread.oauth()
            return
        except Exception as e:
            oauth_error = e

        try:
            self.client = gspread.service_account()
            return
        except Exception as e:
            service_error = e

        raise ValueError(
            f"Could not authenticate with Google Sheets:\n"
            f" gspread.oauth() failed with: {oauth_error}\n"
            f" gspread.service_account() failed with: {service_error}"
            f" follow instructions at: https://docs.gspread.org/en/latest/oauth2.html"
        )

    def create_spreadsheet(self, title: str, folder_id: Optional[str] = None) -> Spreadsheet:
        """
        Create a new spreadsheet.

        :param title: Title of the new spreadsheet
        :type title: str
        :param folder_id: ID of the Google Drive folder to create the spreadsheet in
        :type folder_id: str
        :return: The newly created spreadsheet
        :rtype: gspread.Spreadsheet

        Example:
            >>> manager = GoogleSheetsManager()
            >>> spreadsheet = manager.create_spreadsheet("My New Sheet")
        """
        return self.client.create(title, folder_id=folder_id)

    def open_spreadsheet(self, title: Optional[str] = None, key: Optional[str] = None,
                        url: Optional[str] = None) -> Spreadsheet:
        """
        Open an existing spreadsheet by title, key, or URL.

        :param title: Title of the spreadsheet
        :type title: str
        :param key: Spreadsheet key/ID
        :type key: str
        :param url: Full URL of the spreadsheet
        :type url: str
        :return: The opened spreadsheet
        :rtype: gspread.Spreadsheet

        Example:
            >>> manager = GoogleSheetsManager()
            >>> spreadsheet = manager.open_spreadsheet(title="My Sheet")
            >>> # or
            >>> spreadsheet = manager.open_spreadsheet(key="1BxiMVs0XRA5nFMdKvBdBZjgmUUqptlbs74OgvE2upms")
            >>> # or
            >>> spreadsheet = manager.open_spreadsheet(url="https://docs.google.com/spreadsheets/d/1BxiMVs0XRA5nFMdKvBdBZjgmUUqptlbs74OgvE2upms/edit")
        """
        if title:
            return self.client.open(title)
        elif key:
            return self.client.open_by_key(key)
        elif url:
            return self.client.open_by_url(url)
        else:
            raise ValueError("Must provide either title, key, or url")

    def list_all_spreadsheets(self) -> List[Dict[str, Any]]:
        """
        List all spreadsheets accessible by the authenticated user.

        :return: List of spreadsheet metadata dictionaries
        :rtype: List[Dict[str, Any]]

        Example return value:
            [
                {
                    'id': '1BxiMVs0XRA5nFMdKvBdBZjgmUUqptlbs74OgvE2upms',
                    'name': 'My Spreadsheet',
                    'createdTime': '2023-01-01T00:00:00.000Z',
                    'modifiedTime': '2023-01-02T00:00:00.000Z'
                }
            ]
        """
        return self.client.list_spreadsheet_files()

    def list_worksheets(self, spreadsheet: Spreadsheet) -> List[Worksheet]:
        """
        List all worksheets in a spreadsheet.

        :param spreadsheet: The spreadsheet to list worksheets from
        :type spreadsheet: gspread.Spreadsheet
        :return: List of worksheets
        :rtype: List[gspread.Worksheet]
        """
        return spreadsheet.worksheets()

    def get_worksheet(self, spreadsheet: Spreadsheet, title: Optional[str] = None,
                     index: Optional[int] = None) -> Worksheet:
        """
        Get a specific worksheet by title or index.

        :param spreadsheet: The spreadsheet containing the worksheet
        :type spreadsheet: gspread.Spreadsheet
        :param title: Title of the worksheet
        :type title: str
        :param index: Zero-based index of the worksheet
        :type index: int
        :return: The requested worksheet
        :rtype: gspread.Worksheet
        """
        if title:
            return spreadsheet.worksheet(title)
        elif index is not None:
            return spreadsheet.get_worksheet(index)
        else:
            # Return the first worksheet by default
            return spreadsheet.sheet1

    def create_worksheet(self, spreadsheet: Spreadsheet, title: str,
                        rows: int = 1000, cols: int = 26) -> Worksheet:
        """
        Create a new worksheet in a spreadsheet.

        :param spreadsheet: The spreadsheet to add the worksheet to
        :type spreadsheet: gspread.Spreadsheet
        :param title: Title of the new worksheet
        :type title: str
        :param rows: Number of rows
        :type rows: int
        :param cols: Number of columns
        :type cols: int
        :return: The newly created worksheet
        :rtype: gspread.Worksheet
        """
        return spreadsheet.add_worksheet(title=title, rows=rows, cols=cols)

    def get_all_values(self, worksheet: Worksheet) -> List[List[str]]:
        """
        Get all values from a worksheet as a 2D array.

        :param worksheet: The worksheet to get values from
        :type worksheet: gspread.Worksheet
        :return: 2D array of cell values
        :rtype: List[List[str]]
        """
        return worksheet.get_all_values()

    def get_values_as_dataframe(self, worksheet: Worksheet, **kwargs) -> pd.DataFrame:
        """
        Get all values from a worksheet as a pandas DataFrame.

        :param worksheet: The worksheet to get values from
        :type worksheet: gspread.Worksheet
        :param kwargs: Additional arguments passed to gspread_dataframe.get_as_dataframe
        :return: DataFrame containing the worksheet data
        :rtype: pandas.DataFrame

        Example:
            >>> df = manager.get_values_as_dataframe(worksheet, header=0)
        """
        return gd.get_as_dataframe(worksheet, **kwargs)  # type: ignore

    def get_range_values(self, worksheet: Worksheet, range_name: str) -> List[List[str]]:
        """
        Get values from a specific range using A1 notation.

        :param worksheet: The worksheet to get values from
        :type worksheet: gspread.Worksheet
        :param range_name: Range in A1 notation (e.g., 'A1:C10')
        :type range_name: str
        :return: 2D array of cell values
        :rtype: List[List[str]]
        """
        return worksheet.get(range_name)

    def update_cell(self, worksheet: Worksheet, row: int, col: int, value: Any) -> Dict[str, Any]:
        """
        Update a single cell value.

        :param worksheet: The worksheet to update
        :type worksheet: gspread.Worksheet
        :param row: Row number (1-indexed)
        :type row: int
        :param col: Column number (1-indexed)
        :type col: int
        :param value: Value to set
        :type value: Any
        :return: Update response
        :rtype: Dict[str, Any]
        """
        return dict(worksheet.update_cell(row, col, value))

    def update_range(self, worksheet: Worksheet, range_name: str,
                    values: List[List[Any]]) -> Dict[str, Any]:
        """
        Update a range of cells with a 2D array of values.

        :param worksheet: The worksheet to update
        :type worksheet: gspread.Worksheet
        :param range_name: Range in A1 notation (e.g., 'A1:C10')
        :type range_name: str
        :param values: 2D array of values to write
        :type values: List[List[Any]]
        :return: Update response
        :rtype: Dict[str, Any]
        """
        return dict(worksheet.update(values, range_name))

    def update_with_dataframe(self, worksheet: Worksheet, dataframe: pd.DataFrame,
                            row: int = 1, col: int = 1, include_index: bool = False,
                            include_column_header: bool = True, resize: bool = False,
                            **kwargs) -> None:
        """
        Update a worksheet with data from a pandas DataFrame.

        :param worksheet: The worksheet to update
        :type worksheet: gspread.Worksheet
        :param dataframe: DataFrame to write
        :type dataframe: pandas.DataFrame
        :param row: Starting row (1-indexed)
        :type row: int
        :param col: Starting column (1-indexed)
        :type col: int
        :param include_index: Whether to include the DataFrame index
        :type include_index: bool
        :param include_column_header: Whether to include column headers
        :type include_column_header: bool
        :param resize: Whether to resize the worksheet to fit the data
        :type resize: bool
        :param kwargs: Additional arguments passed to gspread_dataframe.set_with_dataframe
        """
        gd.set_with_dataframe(worksheet, dataframe, row=row, col=col,
                             include_index=include_index,
                             include_column_header=include_column_header,
                             resize=resize, **kwargs)

    def batch_update(self, worksheet: Worksheet,
                    updates: List[Dict[str, Union[str, List[List[Any]]]]]) -> Dict[str, Any]:
        """
        Perform multiple updates in a single batch operation.

        :param worksheet: The worksheet to update
        :type worksheet: gspread.Worksheet
        :param updates: List of update dictionaries with 'range' and 'values' keys
        :type updates: List[Dict[str, Union[str, List[List[Any]]]]]
        :return: Batch update response
        :rtype: Dict[str, Any]

        Example:
            >>> updates = [
            ...     {'range': 'A1:B2', 'values': [[1, 2], [3, 4]]},
            ...     {'range': 'D1:E2', 'values': [['a', 'b'], ['c', 'd']]}
            ... ]
            >>> manager.batch_update(worksheet, updates)
        """
        return dict(worksheet.batch_update(updates))

    def clear_worksheet(self, worksheet: Worksheet) -> Dict[str, Any]:
        """
        Clear all values from a worksheet.

        :param worksheet: The worksheet to clear
        :type worksheet: gspread.Worksheet
        :return: Clear response
        :rtype: Dict[str, Any]
        """
        return dict(worksheet.clear())

    def find_cell(self, worksheet: Worksheet, query: str) -> Optional[Cell]:
        """
        Find a cell containing a specific value.

        :param worksheet: The worksheet to search
        :type worksheet: gspread.Worksheet
        :param query: Value to search for
        :type query: str
        :return: Cell object if found, None otherwise
        :rtype: gspread.Cell or None
        """
        return worksheet.find(query)

    def find_all_cells(self, worksheet: Worksheet, query: str) -> List[Cell]:
        """
        Find all cells containing a specific value.

        :param worksheet: The worksheet to search
        :type worksheet: gspread.Worksheet
        :param query: Value to search for
        :type query: str
        :return: List of matching cells
        :rtype: List[gspread.Cell]
        """
        return worksheet.findall(query)

    def get_cell_value(self, worksheet: Worksheet, row: int, col: int) -> Any:
        """
        Get the value of a specific cell.

        :param worksheet: The worksheet to read from
        :type worksheet: gspread.Worksheet
        :param row: Row number (1-indexed)
        :type row: int
        :param col: Column number (1-indexed)
        :type col: int
        :return: Cell value
        :rtype: Any
        """
        return worksheet.cell(row, col).value

    def append_row(self, worksheet: Worksheet, values: List[Any]) -> Dict[str, Any]:
        """
        Append a row to the end of the worksheet.

        :param worksheet: The worksheet to append to
        :type worksheet: gspread.Worksheet
        :param values: List of values for the new row
        :type values: List[Any]
        :return: Append response
        :rtype: Dict[str, Any]
        """
        return dict(worksheet.append_row(values))

    def append_rows(self, worksheet: Worksheet, values: List[List[Any]]) -> Dict[str, Any]:
        """
        Append multiple rows to the end of the worksheet.

        :param worksheet: The worksheet to append to
        :type worksheet: gspread.Worksheet
        :param values: List of rows to append
        :type values: List[List[Any]]
        :return: Append response
        :rtype: Dict[str, Any]
        """
        return dict(worksheet.append_rows(values))
