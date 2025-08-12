import pytest
from unittest.mock import patch, MagicMock, mock_open, PropertyMock
import pandas as pd
import sqlite3
from stock import Stock

@pytest.fixture
def stock():
    return Stock("FAKE")

@pytest.fixture(autouse=True)
def mock_yf_ticker():
    with patch("stock.yf.Ticker") as mock_ticker_class:
        mock_ticker = MagicMock()
        mock_ticker_class.return_value = mock_ticker
        mock_ticker.info = {
            "longName": "Fake Corp",
            "regularMarketPrice": 123.45,
            "fiftyDayAverage": 120,
            "twoHundredDayAverage": 115,
            "forwardPE": 15.2
        }
        mock_ticker.history.return_value = pd.DataFrame({"Open": [1,2], "Close": [3,4]})
        mock_ticker.splits = pd.Series([0.5, 0.25])
        mock_ticker.dividends = pd.Series([1.5, 2.0])
        yield mock_ticker

# --- Fetch methods ---
def test_fetch_analysis_success(stock):
    df = pd.DataFrame({"Rec": ["Buy", "Sell"]})
    with patch.object(stock.stock, "recommendations", df):
        result = stock.fetch_analysis()
        assert isinstance(result, pd.DataFrame)
        assert stock.data["analysis"].equals(df)

def test_fetch_analysis_empty(stock):
    with patch.object(stock.stock, "recommendations", pd.DataFrame()):
        result = stock.fetch_analysis()
        assert result is None

def test_fetch_analysis_failure(stock):
    with patch.object(stock.stock, "recommendations", side_effect=Exception("fail")):
        result = stock.fetch_analysis()
        assert result is None

def test_fetch_balance_sheet_success(stock):
    df = pd.DataFrame({"A": [1]})
    with patch.object(stock.stock, "balance_sheet", df):
        result = stock.fetch_balance_sheet()
        assert isinstance(result, pd.DataFrame)
        assert stock.data["balance_sheet"].equals(df)

def test_fetch_balance_sheet_empty(stock):
    with patch.object(stock.stock, "balance_sheet", pd.DataFrame()):
        result = stock.fetch_balance_sheet()
        assert result is None

def test_fetch_balance_sheet_failure(stock):
    with patch.object(stock.stock, "balance_sheet", side_effect=Exception("fail")):
        result = stock.fetch_balance_sheet()
        assert result is None

def test_fetch_cash_flows_success(stock):
    df = pd.DataFrame({"A": [1]})
    with patch.object(stock.stock, "cashflow", df):
        result = stock.fetch_cash_flows()
        assert isinstance(result, pd.DataFrame)
        assert stock.data["cashflow"].equals(df)

def test_fetch_cash_flows_empty(stock):
    with patch.object(stock.stock, "cashflow", pd.DataFrame()):
        result = stock.fetch_cash_flows()
        assert result is None

def test_fetch_cash_flows_failure(stock):
    with patch.object(stock.stock, "cashflow", side_effect=Exception("fail")):
        result = stock.fetch_cash_flows()
        assert result is None

def test_fetch_data_success(stock):
    data = stock.fetch_data()
    assert data is not None
    assert "longName" in data
    assert stock.data["info"] == data

def test_fetch_data_failure(stock):
    prop = PropertyMock(side_effect=Exception("fail"))
    type(stock.stock).info = prop
    result = stock.fetch_data()
    assert result is None

def test_fetch_data_empty(stock):
    with patch.object(stock.stock, "info", {}):
        result = stock.fetch_data()
        assert result is None

def test_fetch_dividends_success(stock):
    s = pd.Series([1.5, 2.0])
    with patch.object(stock.stock, "dividends", s):
        result = stock.fetch_dividends()
        assert isinstance(result, pd.Series)
        assert result.equals(s)
        assert stock.data["dividends"].equals(s)

def test_fetch_dividends_empty(stock):
    with patch.object(stock.stock, "dividends", pd.Series()):
        result = stock.fetch_dividends()
        assert result is None

def test_fetch_dividends_failure(stock):
    with patch.object(stock.stock, "dividends", side_effect=Exception("fail")):
        result = stock.fetch_dividends()
        assert result is None

def test_fetch_earnings_success(stock):
    df = pd.DataFrame({"Date": ["2023-01-01"]})
    with patch.object(stock.stock, "earnings_dates", df):
        result = stock.fetch_earnings()
        assert isinstance(result, pd.DataFrame)
        assert stock.data["earnings"].equals(df)

def test_fetch_earnings_empty(stock):
    with patch.object(stock.stock, "earnings_dates", pd.DataFrame()):
        result = stock.fetch_earnings()
        assert result is None

def test_fetch_earnings_failure(stock):
    with patch.object(stock.stock, "earnings_dates", side_effect=Exception("fail")):
        result = stock.fetch_earnings()
        assert result is None

def test_fetch_financials_success(stock):
    df = pd.DataFrame({"A": [1]})
    with patch.object(stock.stock, "financials", df):
        result = stock.fetch_financials()
        assert isinstance(result, pd.DataFrame)
        assert stock.data["financials"].equals(df)

def test_fetch_financials_empty(stock):
    with patch.object(stock.stock, "financials", pd.DataFrame()):
        result = stock.fetch_financials()
        assert result is None

def test_fetch_financials_failure(stock):
    with patch.object(stock.stock, "financials", side_effect=Exception("fail")):
        result = stock.fetch_financials()
        assert result is None

def test_fetch_history_success(stock):
    df = pd.DataFrame({"Open": [1,2], "Close": [3,4]})
    with patch.object(stock.stock, "history", return_value=df):
        result = stock.fetch_history()
        assert isinstance(result, pd.DataFrame)
        assert result.equals(df)
        assert stock.data["history"].equals(df)

def test_fetch_history_empty(stock):
    with patch.object(stock.stock, "history", return_value=pd.DataFrame()):
        result = stock.fetch_history()
        assert result is None

def test_fetch_history_failure(stock):
    with patch.object(stock.stock, "history", side_effect=Exception("fail")):
        result = stock.fetch_history()
        assert result is None

def test_fetch_institutional_holders_success(stock):
    df = pd.DataFrame({"Holder": ["A", "B"]})
    with patch.object(stock.stock, "institutional_holders", df):
        result = stock.fetch_institutional_holders()
        assert isinstance(result, pd.DataFrame)
        assert stock.data["institutional_holders"].equals(df)

def test_fetch_institutional_holders_empty(stock):
    with patch.object(stock.stock, "institutional_holders", pd.DataFrame()):
        result = stock.fetch_institutional_holders()
        assert result is None

def test_fetch_institutional_holders_failure(stock):
    with patch.object(stock.stock, "institutional_holders", side_effect=Exception("fail")):
        result = stock.fetch_institutional_holders()
        assert result is None

def test_fetch_major_holders_success(stock):
    df = pd.DataFrame({"Holder": ["X", "Y"]})
    with patch.object(stock.stock, "major_holders", df):
        result = stock.fetch_major_holders()
        assert isinstance(result, pd.DataFrame)
        assert stock.data["major_holders"].equals(df)

def test_fetch_major_holders_empty(stock):
    with patch.object(stock.stock, "major_holders", pd.DataFrame()):
        result = stock.fetch_major_holders()
        assert result is None

def test_fetch_major_holders_failure(stock):
    with patch.object(stock.stock, "major_holders", side_effect=Exception("fail")):
        result = stock.fetch_major_holders()
        assert result is None

def test_fetch_options_success(stock):
    with patch.object(stock.stock, "options", ["2024-07-01"]), \
         patch.object(stock.stock, "option_chain") as mock_chain:
        calls = pd.DataFrame([{"strike": 100}])
        puts = pd.DataFrame([{"strike": 90}])
        mock_chain.return_value = MagicMock(calls=calls, puts=puts)
        result = stock.fetch_options()
        assert isinstance(result, dict)
        assert "calls" in result and "puts" in result and "expiration_date" in result
        assert stock.data["options"] == result

def test_fetch_options_empty(stock):
    with patch.object(stock.stock, "options", []):
        result = stock.fetch_options()
        assert result is None

def test_fetch_options_failure(stock):
    type(stock.stock).options = PropertyMock(side_effect=Exception("fail"))
    result = stock.fetch_options()
    assert result is None

def test_fetch_splits_success(stock):
    s = pd.Series([0.5, 0.25])
    with patch.object(stock.stock, "splits", s):
        result = stock.fetch_splits()
        assert isinstance(result, pd.Series)
        assert result.equals(s)
        assert stock.data["splits"].equals(s)

def test_fetch_splits_empty(stock):
    with patch.object(stock.stock, "splits", pd.Series()):
        result = stock.fetch_splits()
        assert result is None

def test_fetch_splits_failure(stock):
    with patch.object(stock.stock, "splits", side_effect=Exception("fail")):
        result = stock.fetch_splits()
        assert result is None

def test_fetch_sustainability_success(stock):
    df = pd.DataFrame({"ESG": [1, 2]})
    with patch.object(stock.stock, "sustainability", df):
        result = stock.fetch_sustainability()
        assert isinstance(result, pd.DataFrame)
        assert stock.data["sustainability"].equals(df)

def test_fetch_sustainability_empty(stock):
    with patch.object(stock.stock, "sustainability", pd.DataFrame()):
        result = stock.fetch_sustainability()
        assert result is None

def test_fetch_sustainability_failure(stock):
    with patch.object(stock.stock, "sustainability", side_effect=Exception("fail")):
        result = stock.fetch_sustainability()
        assert result is None

# --- Save to file methods ---
@patch.object(Stock, "_save_json")
def test_save_info_to_file_success(mock_save_json, stock):
    stock.data["info"] = {"key": "value"}
    stock.save_info_to_file()
    mock_save_json.assert_called_once_with({"key": "value"}, "info.json")

def test_save_info_to_file_no_data(stock):
    result = stock.save_info_to_file()
    assert result is None

@patch.object(Stock, "_save_df")
def test_save_history_to_file_success(mock_save_df, stock):
    stock.data["history"] = pd.DataFrame({"Open": [1]})
    stock.save_history_to_file()
    mock_save_df.assert_called_once()

def test_save_history_to_file_no_data(stock):
    result = stock.save_history_to_file()
    assert result is None

@patch.object(Stock, "_save_df")
def test_save_splits_to_file_success(mock_save_df, stock):
    stock.data["splits"] = pd.Series([1])
    stock.save_splits_to_file()
    mock_save_df.assert_called_once()

def test_save_splits_to_file_no_data(stock):
    result = stock.save_splits_to_file()
    assert result is None

@patch.object(Stock, "_save_df")
def test_save_dividends_to_file_success(mock_save_df, stock):
    stock.data["dividends"] = pd.Series([1])
    stock.save_dividends_to_file()
    mock_save_df.assert_called_once()

def test_save_dividends_to_file_no_data(stock):
    result = stock.save_dividends_to_file()
    assert result is None

@patch.object(Stock, "_save_df")
def test_save_analysis_to_file_success(mock_save_df, stock):
    stock.data["analysis"] = pd.DataFrame({"Rec": ["Buy"]})
    stock.save_analysis_to_file()
    mock_save_df.assert_called_once()

def test_save_analysis_to_file_no_data(stock):
    result = stock.save_analysis_to_file()
    assert result is None

@patch.object(Stock, "_save_df")
def test_save_earnings_to_file_success(mock_save_df, stock):
    stock.data["earnings"] = pd.DataFrame({"Date": ["2023-01-01"]})
    stock.save_earnings_to_file()
    mock_save_df.assert_called_once()

def test_save_earnings_to_file_no_data(stock):
    result = stock.save_earnings_to_file()
    assert result is None

@patch.object(Stock, "_save_df")
def test_save_balance_sheet_to_file_success(mock_save_df, stock):
    stock.data["balance_sheet"] = pd.DataFrame({"A": [1]})
    stock.save_balance_sheet_to_file()
    mock_save_df.assert_called_once()

def test_save_balance_sheet_to_file_no_data(stock):
    result = stock.save_balance_sheet_to_file()
    assert result is None

@patch.object(Stock, "_save_df")
def test_save_cash_flows_to_file_success(mock_save_df, stock):
    stock.data["cashflow"] = pd.DataFrame({"A": [1]})
    stock.save_cash_flows_to_file()
    mock_save_df.assert_called_once()

def test_save_cash_flows_to_file_no_data(stock):
    result = stock.save_cash_flows_to_file()
    assert result is None

@patch.object(Stock, "_save_df")
def test_save_financials_to_file_success(mock_save_df, stock):
    stock.data["financials"] = pd.DataFrame({"A": [1]})
    stock.save_financials_to_file()
    mock_save_df.assert_called_once()

def test_save_financials_to_file_no_data(stock):
    result = stock.save_financials_to_file()
    assert result is None

@patch.object(Stock, "_save_df")
def test_save_institutional_holders_to_file_success(mock_save_df, stock):
    stock.data["institutional_holders"] = pd.DataFrame({"Holder": ["A"]})
    stock.save_institutional_holders_to_file()
    mock_save_df.assert_called_once()

def test_save_institutional_holders_to_file_no_data(stock):
    result = stock.save_institutional_holders_to_file()
    assert result is None

@patch.object(Stock, "_save_df")
def test_save_major_holders_to_file_success(mock_save_df, stock):
    stock.data["major_holders"] = pd.DataFrame({"Holder": ["X"]})
    stock.save_major_holders_to_file()
    mock_save_df.assert_called_once()

def test_save_major_holders_to_file_no_data(stock):
    result = stock.save_major_holders_to_file()
    assert result is None

@patch.object(Stock, "_save_json")
def test_save_options_to_file_success(mock_save_json, stock):
    stock.data["options"] = {"calls": [], "puts": [], "expiration_date": "2024-07-01"}
    stock.save_options_to_file()
    mock_save_json.assert_called_once_with(stock.data["options"], "options.json")

def test_save_options_to_file_no_data(stock):
    result = stock.save_options_to_file()
    assert result is None

@patch.object(Stock, "_save_df")
def test_save_sustainability_to_file_success(mock_save_df, stock):
    stock.data["sustainability"] = pd.DataFrame({"ESG": [1]})
    stock.save_sustainability_to_file()
    mock_save_df.assert_called_once()

def test_save_sustainability_to_file_no_data(stock):
    result = stock.save_sustainability_to_file()
    assert result is None

# --- Load from file methods ---
@patch.object(Stock, "_load_json", return_value={"a": 1})
def test_load_info_from_file_success(mock_load_json, stock):
    result = stock.load_info_from_file()
    assert result == {"a": 1}
    assert stock.data["info"] == {"a": 1}

@patch.object(Stock, "_load_json", return_value=None)
def test_load_info_from_file_none(mock_load_json, stock):
    result = stock.load_info_from_file()
    assert result is None

@patch.object(Stock, "_load_df", return_value=pd.DataFrame({"A": [1]}))
def test_load_history_from_file_success(mock_load_df, stock):
    result = stock.load_history_from_file()
    assert isinstance(result, pd.DataFrame)
    assert stock.data["history"].equals(result)

@patch.object(Stock, "_load_df", return_value=None)
def test_load_history_from_file_empty(mock_load_df, stock):
    result = stock.load_history_from_file()
    assert result is None

@patch.object(Stock, "_load_df", return_value=pd.DataFrame({"Rec": ["Buy"]}))
def test_load_analysis_from_file_success(mock_load_df, stock):
    result = stock.load_analysis_from_file()
    assert isinstance(result, pd.DataFrame)
    assert stock.data["analysis"].equals(result)

@patch.object(Stock, "_load_df", return_value=None)
def test_load_analysis_from_file_empty(mock_load_df, stock):
    result = stock.load_analysis_from_file()
    assert result is None

@patch.object(Stock, "_load_df", return_value=pd.DataFrame({"Date": ["2023-01-01"]}))
def test_load_earnings_from_file_success(mock_load_df, stock):
    result = stock.load_earnings_from_file()
    assert isinstance(result, pd.DataFrame)
    assert stock.data["earnings"].equals(result)

@patch.object(Stock, "_load_df", return_value=None)
def test_load_earnings_from_file_empty(mock_load_df, stock):
    result = stock.load_earnings_from_file()
    assert result is None

@patch.object(Stock, "_load_df", return_value=pd.DataFrame({"A": [1]}))
def test_load_balance_sheet_from_file_success(mock_load_df, stock):
    result = stock.load_balance_sheet_from_file()
    assert isinstance(result, pd.DataFrame)
    assert stock.data["balance_sheet"].equals(result)

@patch.object(Stock, "_load_df", return_value=None)
def test_load_balance_sheet_from_file_empty(mock_load_df, stock):
    result = stock.load_balance_sheet_from_file()
    assert result is None

@patch.object(Stock, "_load_df", return_value=pd.DataFrame({"A": [1]}))
def test_load_cash_flows_from_file_success(mock_load_df, stock):
    result = stock.load_cash_flows_from_file()
    assert isinstance(result, pd.DataFrame)
    assert stock.data["cashflow"].equals(result)

@patch.object(Stock, "_load_df", return_value=None)
def test_load_cash_flows_from_file_empty(mock_load_df, stock):
    result = stock.load_cash_flows_from_file()
    assert result is None

@patch.object(Stock, "_load_df", return_value=pd.DataFrame({"A": [1]}))
def test_load_financials_from_file_success(mock_load_df, stock):
    result = stock.load_financials_from_file()
    assert isinstance(result, pd.DataFrame)
    assert stock.data["financials"].equals(result)

@patch.object(Stock, "_load_df", return_value=None)
def test_load_financials_from_file_empty(mock_load_df, stock):
    result = stock.load_financials_from_file()
    assert result is None

@patch.object(Stock, "_load_df", return_value=pd.DataFrame({"Holder": ["A"]}))
def test_load_institutional_holders_from_file_success(mock_load_df, stock):
    result = stock.load_institutional_holders_from_file()
    assert isinstance(result, pd.DataFrame)
    assert stock.data["institutional_holders"].equals(result)

@patch.object(Stock, "_load_df", return_value=None)
def test_load_institutional_holders_from_file_empty(mock_load_df, stock):
    result = stock.load_institutional_holders_from_file()
    assert result is None

@patch.object(Stock, "_load_df", return_value=pd.DataFrame({"Holder": ["X"]}))
def test_load_major_holders_from_file_success(mock_load_df, stock):
    result = stock.load_major_holders_from_file()
    assert isinstance(result, pd.DataFrame)
    assert stock.data["major_holders"].equals(result)

@patch.object(Stock, "_load_df", return_value=None)
def test_load_major_holders_from_file_empty(mock_load_df, stock):
    result = stock.load_major_holders_from_file()
    assert result is None

@patch.object(Stock, "_load_json", return_value={"calls": [], "puts": [], "expiration_date": "2024-07-01"})
def test_load_options_from_file_success(mock_load_json, stock):
    result = stock.load_options_from_file()
    assert isinstance(result, dict)
    assert stock.data["options"] == result

@patch.object(Stock, "_load_json", return_value=None)
def test_load_options_from_file_empty(mock_load_json, stock):
    result = stock.load_options_from_file()
    assert result is None

@patch.object(Stock, "_load_df", return_value=pd.Series([1]))
def test_load_splits_from_file_success(mock_load_df, stock):
    result = stock.load_splits_from_file()
    assert isinstance(result, pd.Series) or isinstance(result, pd.DataFrame)
    assert stock.data["splits"].equals(result)

@patch.object(Stock, "_load_df", return_value=None)
def test_load_splits_from_file_empty(mock_load_df, stock):
    result = stock.load_splits_from_file()
    assert result is None

@patch.object(Stock, "_load_df", return_value=pd.DataFrame({"ESG": [1]}))
def test_load_sustainability_from_file_success(mock_load_df, stock):
    result = stock.load_sustainability_from_file()
    assert isinstance(result, pd.DataFrame)
    assert stock.data["sustainability"].equals(result)

@patch.object(Stock, "_load_df", return_value=None)
def test_load_sustainability_from_file_empty(mock_load_df, stock):
    result = stock.load_sustainability_from_file()
    assert result is None

# --- Database connection ---
def test_connect_db_success(stock):
    conn = stock._connect_db()
    assert isinstance(conn, sqlite3.Connection)
    conn.close()

@patch("sqlite3.connect", side_effect=Exception("fail"))
def test_connect_db_failure(mock_connect, stock):
    conn = stock._connect_db()
    assert conn is None

# --- update_database_info ---
@patch("sqlite3.connect")
def test_update_database_info_success(mock_connect, stock):
    stock.data["info"] = {
        "longName": "Test Company",
        "regularMarketPrice": 100,
        "fiftyDayAverage": 90,
        "twoHundredDayAverage": 80,
        "forwardPE": 10
    }
    mock_conn = MagicMock()
    mock_connect.return_value = mock_conn
    mock_cursor = MagicMock()
    mock_conn.cursor.return_value = mock_cursor

    stock.update_database_info()

    mock_cursor.execute.assert_called()
    mock_conn.commit.assert_called()
    mock_conn.close.assert_called()

@patch("sqlite3.connect", side_effect=Exception("fail"))
def test_update_database_info_db_fail(mock_connect, stock):
    stock.data["info"] = {"longName": "Test"}
    result = stock.update_database_info()
    assert result is None

def test_update_database_info_no_data(stock):
    result = stock.update_database_info()
    assert result is None