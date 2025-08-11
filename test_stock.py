import pytest
from unittest.mock import patch, MagicMock, mock_open
import json
import os
import pandas as pd
import sqlite3
from stock import Stock
from unittest.mock import PropertyMock

@pytest.fixture
def stock():
    return Stock("FAKE")

# --- Mocka yf.Ticker ---
@pytest.fixture(autouse=True)
def mock_yf_ticker():
    with patch("stock.yf.Ticker") as mock_ticker_class:
        mock_ticker = MagicMock()
        mock_ticker_class.return_value = mock_ticker

        # Default mock data
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

# --- fetch_data ---

def test_fetch_data_success(stock):
    data = stock.fetch_data()
    assert data is not None
    assert "longName" in data
    assert stock.data == data

def test_fetch_data_failure(stock):
    prop = PropertyMock(side_effect=Exception("fail"))
    type(stock.stock).info = prop  # sätter property på mock-klassen av stock.stock

    result = stock.fetch_data()
    assert result is None

def test_fetch_data_empty(stock):
    with patch.object(stock.stock, "info", {}):
        result = stock.fetch_data()
        assert result is None

# --- fetch_history ---

def test_fetch_history_success(stock):
    df = pd.DataFrame({"Open": [1,2], "Close": [3,4]})
    with patch.object(stock.stock, "history", return_value=df):
        result = stock.fetch_history()
        assert isinstance(result, pd.DataFrame)
        assert (result == df).all().all()
        assert stock.history.equals(df)

def test_fetch_history_empty(stock):
    with patch.object(stock.stock, "history", return_value=pd.DataFrame()):
        result = stock.fetch_history()
        assert result is None

def test_fetch_history_failure(stock):
    with patch.object(stock.stock, "history", side_effect=Exception("fail")):
        result = stock.fetch_history()
        assert result is None

# --- fetch_splits ---

def test_fetch_splits_success(stock):
    s = pd.Series([0.5, 0.25])
    with patch.object(stock.stock, "splits", s):
        result = stock.fetch_splits()
        assert isinstance(result, pd.Series)
        assert stock.splits.equals(s)

def test_fetch_splits_empty(stock):
    with patch.object(stock.stock, "splits", pd.Series()):
        result = stock.fetch_splits()
        assert result is None

def test_fetch_splits_failure(stock):
    with patch.object(stock.stock, "splits", side_effect=Exception("fail")):
        result = stock.fetch_splits()
        assert result is None

# --- fetch_dividends ---

def test_fetch_dividends_success(stock):
    s = pd.Series([1.5, 2.0])
    with patch.object(stock.stock, "dividends", s):
        result = stock.fetch_dividends()
        assert isinstance(result, pd.Series)
        assert stock.dividends.equals(s)

def test_fetch_dividends_empty(stock):
    with patch.object(stock.stock, "dividends", pd.Series()):
        result = stock.fetch_dividends()
        assert result is None

def test_fetch_dividends_failure(stock):
    with patch.object(stock.stock, "dividends", side_effect=Exception("fail")):
        result = stock.fetch_dividends()
        assert result is None

# --- Save to file mocks ---

@patch("builtins.open", new_callable=mock_open)
@patch("os.makedirs")
def test_save_data_to_success(mock_makedirs, mock_file, stock):
    stock.data = {"key": "value"}
    stock.save_data_to(base_folder="test_folder")
    mock_makedirs.assert_called_once()
    mock_file.assert_called_once()
    handle = mock_file()
    handle.write.assert_called()  # check json dump wrote data

@patch("pandas.DataFrame.to_csv")
@patch("os.makedirs")
def test_save_history_to_success(mock_makedirs, mock_to_csv, stock):
    stock.history = pd.DataFrame({"Open": [1]})
    stock.save_history_to(base_folder="test_folder")
    mock_makedirs.assert_called_once()
    mock_to_csv.assert_called_once()

@patch("pandas.Series.to_csv")
@patch("os.makedirs")
def test_save_splits_to_success(mock_makedirs, mock_to_csv, stock):
    stock.splits = pd.Series([1])
    stock.save_splits_to(base_folder="test_folder")
    mock_makedirs.assert_called_once()
    mock_to_csv.assert_called_once()

@patch("pandas.Series.to_csv")
@patch("os.makedirs")
def test_save_dividends_to_success(mock_makedirs, mock_to_csv, stock):
    stock.dividends = pd.Series([1])
    stock.save_dividends_to(base_folder="test_folder")
    mock_makedirs.assert_called_once()
    mock_to_csv.assert_called_once()

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
    stock.data = {
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
def test_update_database_info_db_fail(stock):
    stock.data = {
        "longName": "Test Company",
        "regularMarketPrice": 100,
        "fiftyDayAverage": 90,
        "twoHundredDayAverage": 80,
        "forwardPE": 10
    }
    stock.db_path = "invalid_path"
    stock.update_database_info()  # Should handle exception gracefully

@patch("sqlite3.connect")
def test_update_database_info_no_data(mock_connect, stock):
    stock.data = None
    stock.update_database_info()
    mock_connect.assert_not_called()

# --- Real API call test (can be slow and flaky, so marked separately) ---

@pytest.mark.real_api
def test_real_api_call():
    stock = Stock("AAPL")
    data = stock.fetch_data()
    assert data is not None
    history = stock.fetch_history()
    assert isinstance(history, pd.DataFrame)
    splits = stock.fetch_splits()
    dividends = stock.fetch_dividends()

    # Just check types, not values
    assert isinstance(splits, (pd.Series, type(None)))  # can be None if no splits
    assert isinstance(dividends, (pd.Series, type(None)))