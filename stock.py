# Stock Data Management System
# This module provides a comprehensive interface for fetching, storing, and managing stock market data
# using the Yahoo Finance API (yfinance) with local file storage and database integration

import yfinance as yf  # Yahoo Finance API for stock data
import json           # JSON handling for data serialization
import os            # Operating system interface for file operations
import pandas as pd  # Data manipulation and analysis library
import sqlite3      # SQLite database interface
from datetime import datetime  # Date and time utilities

class Stock:
    """
    A comprehensive stock data management class that provides:
    - Data fetching from Yahoo Finance API
    - Local file storage in CSV and JSON formats
    - Database integration for metadata storage
    - Support for multiple data types (financials, market data, analysis, etc.)
    """
    
    def __init__(self, ticker, db_path="magic_beans.db", base_folder="stocks"):
        """
        Initialize Stock object with ticker symbol and storage configuration.
        
        Args:
            ticker (str): Stock ticker symbol (e.g., 'AAPL', 'MSFT')
            db_path (str): Path to SQLite database file
            base_folder (str): Base directory for storing stock data files
        """
        self.ticker = ticker.upper()  # Standardize ticker to uppercase
        self.stock = yf.Ticker(ticker)  # Create yfinance Ticker object
        self.db_path = db_path
        self.base_folder = base_folder
        self.data = {}  # Cache for loaded data

    # --- Helper methods for database and file operations ---

    def _connect_db(self):
        """
        Establish connection to SQLite database.
        
        Returns:
            sqlite3.Connection: Database connection object or None if failed
        """
        try:
            return sqlite3.connect(self.db_path)
        except Exception as e:
            print(f"Error connecting to DB: {e}")
            return None

    def _load_df(self, filename):
        """
        Load DataFrame from CSV file with proper date parsing and indexing.
        
        Args:
            filename (str): Name of the CSV file to load
            
        Returns:
            pandas.DataFrame: Loaded DataFrame or None if failed
        """
        path = os.path.join(self.base_folder, self.ticker, filename)
        if not os.path.exists(path):
            print(f"File not found: {path}")
            return None
        try:
            df = pd.read_csv(path)
            # Parse Date column and set as index if present
            if "Date" in df.columns:
                df["Date"] = pd.to_datetime(df["Date"], errors="coerce", utc=True)
                df = df.set_index("Date", drop=True)
            return df
        except Exception as e:
            print(f"Error loading {filename} from file: {e}")
            return None

    def _load_json(self, filename):
        """
        Load JSON data from file.
        
        Args:
            filename (str): Name of the JSON file to load
            
        Returns:
            dict: Loaded JSON data or None if failed
        """
        path = os.path.join(self.base_folder, self.ticker, filename)
        if not os.path.exists(path):
            print(f"File not found: {path}")
            return None
        try:
            with open(path, "r") as f:
                obj = json.load(f)
            return obj
        except Exception as e:
            print(f"Error loading {filename} from file: {e}")
            return None

    def _prepare_for_json(self, obj):
        """
        Recursively prepare data structures for JSON serialization.
        Handles pandas objects, timestamps, and NaN values.
        
        Args:
            obj: Object to prepare for JSON serialization
            
        Returns:
            JSON-serializable object
        """
        if isinstance(obj, dict):
            return {key: self._prepare_for_json(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._prepare_for_json(item) for item in obj]
        elif isinstance(obj, pd.Timestamp):
            return obj.strftime('%Y-%m-%d %H:%M:%S') if pd.notna(obj) else None
        elif isinstance(obj, datetime):
            return obj.strftime('%Y-%m-%d %H:%M:%S')
        elif pd.isna(obj):
            return None
        elif isinstance(obj, (pd.Series, pd.DataFrame)):
            return self._prepare_for_json(obj.to_dict())
        else:
            return obj

    def _save_df(self, df, filename):
        """
        Save DataFrame to CSV file with proper formatting and directory creation.
        
        Args:
            df (pandas.DataFrame or pandas.Series): Data to save
            filename (str): Name of the CSV file to create
        """
        folder = os.path.join(self.base_folder, self.ticker)
        os.makedirs(folder, exist_ok=True)  # Create directory if it doesn't exist
        path = os.path.join(folder, filename)
        
        try:
            # Handle different pandas data types
            if isinstance(df, pd.Series):
                # Convert Series to DataFrame with proper column naming
                df_to_save = df.to_frame().reset_index()
                df_to_save.columns = ['Date', filename.split('.')[0].capitalize()]
            elif isinstance(df.index, pd.DatetimeIndex) and "Date" not in df.columns:
                # Reset DatetimeIndex to column for CSV storage
                df_to_save = df.reset_index()
            else:
                df_to_save = df.copy()
                
            df_to_save.to_csv(path, index=False)
            print(f"Saved {filename} to {path}")
        except Exception as e:
            print(f"Error saving {filename} to file: {e}")
            raise

    def _save_json(self, obj, filename):
        """
        Save object to JSON file with proper formatting and directory creation.
        
        Args:
            obj: Object to save as JSON
            filename (str): Name of the JSON file to create
        """
        folder = os.path.join(self.base_folder, self.ticker)
        os.makedirs(folder, exist_ok=True)  # Create directory if it doesn't exist
        path = os.path.join(folder, filename)
        
        try:
            # Prepare object for JSON serialization
            json_ready_obj = self._prepare_for_json(obj)
            with open(path, "w") as f:
                json.dump(json_ready_obj, f, indent=2, default=str)
            print(f"Saved {filename} to {path}")
        except Exception as e:
            print(f"Error saving {filename} to file: {e}")
            raise

    # --- Data Fetching Methods from Yahoo Finance API ---

    def fetch_analysis(self):
        """
        Fetch analyst recommendations and analysis data for the stock.
        
        Returns:
            pandas.DataFrame: Analyst recommendations data or None if failed
        """
        try:
            data = self.stock.recommendations
            if data is None or data.empty:
                print(f"No recommendations data found for {self.ticker}")
                return None
            self.data["analysis"] = data
            return data
        except Exception as e:
            print(f"Error fetching recommendations for {self.ticker}: {e}")
            return None

    def fetch_balance_sheet(self):
        """
        Fetch balance sheet financial data for the stock.
        
        Returns:
            pandas.DataFrame: Balance sheet data or None if failed
        """
        try:
            data = self.stock.balance_sheet
            if data is None or data.empty:
                print(f"No balance sheet data found for {self.ticker}")
                return None
            self.data["balance_sheet"] = data
            return data
        except Exception as e:
            print(f"Error fetching balance sheet for {self.ticker}: {e}")
            return None

    def fetch_cash_flows(self):
        """
        Fetch cash flow statement data for the stock.
        
        Returns:
            pandas.DataFrame: Cash flow data or None if failed
        """
        try:
            data = self.stock.cashflow
            if data is None or data.empty:
                print(f"No cash flow data found for {self.ticker}")
                return None
            self.data["cashflow"] = data
            return data
        except Exception as e:
            print(f"Error fetching cash flows for {self.ticker}: {e}")
            return None

    def fetch_data(self):
        """
        Fetch general company information and key statistics.
        
        Returns:
            dict: Company info dictionary or None if failed
        """
        try:
            data = self.stock.info
            if not data or len(data) == 0:
                print(f"No info found for {self.ticker}")
                return None
            self.data["info"] = data
            return data
        except Exception as e:
            print(f"Error fetching info for {self.ticker}: {e}")
            return None

    def fetch_dividends(self):
        """
        Fetch historical dividend payment data for the stock.
        
        Returns:
            pandas.Series: Dividend payment history or None if failed
        """
        try:
            dividends = self.stock.dividends
            if dividends.empty:
                print(f"No dividends data found for {self.ticker}")
                return None
            self.data["dividends"] = dividends
            return dividends
        except Exception as e:
            print(f"Error fetching dividends for {self.ticker}: {e}")
            return None

    def fetch_earnings(self):
        try:
            data = self.stock.earnings_dates
            if data is None or data.empty:
                print(f"No earnings dates found for {self.ticker}")
                return None
            self.data["earnings"] = data
            return data
        except Exception as e:
            print(f"Error fetching earnings for {self.ticker}: {e}")
            return None

    def fetch_financials(self):
        try:
            data = self.stock.financials
            if data is None or data.empty:
                print(f"No financials data found for {self.ticker}")
                return None
            self.data["financials"] = data
            return data
        except Exception as e:
            print(f"Error fetching financials for {self.ticker}: {e}")
            return None

    def fetch_history(self):
        try:
            history = self.stock.history(period="max")
            if history.empty:
                print(f"No historical data found for {self.ticker}")
                return None
            self.data["history"] = history
            return history
        except Exception as e:
            print(f"Error fetching historical data for {self.ticker}: {e}")
            return None

    def fetch_institutional_holders(self):
        try:
            holders = self.stock.institutional_holders
            if holders is None or holders.empty:
                print(f"No institutional holders data found for {self.ticker}")
                return None
            self.data["institutional_holders"] = holders
            return holders
        except Exception as e:
            print(f"Error fetching institutional holders for {self.ticker}: {e}")
            return None

    def fetch_major_holders(self):
        try:
            holders = self.stock.major_holders
            if holders is None or holders.empty:
                print(f"No major holders data found for {self.ticker}")
                return None
            self.data["major_holders"] = holders
            return holders
        except Exception as e:
            print(f"Error fetching major holders for {self.ticker}: {e}")
            return None

    def fetch_options(self):
        try:
            options = self.stock.options
            if not options:
                print(f"No options data found for {self.ticker}")
                return None
            opt = self.stock.option_chain(options[0])
            calls_data = opt.calls.to_dict(orient="records")
            puts_data = opt.puts.to_dict(orient="records")
            result = {
                "calls": calls_data,
                "puts": puts_data,
                "expiration_date": str(options[0])
            }
            self.data["options"] = result
            return result
        except Exception as e:
            print(f"Error fetching options for {self.ticker}: {e}")
            return None

    def fetch_splits(self):
        try:
            splits = self.stock.splits
            if splits.empty:
                print(f"No splits data found for {self.ticker}")
                return None
            self.data["splits"] = splits
            return splits
        except Exception as e:
            print(f"Error fetching splits for {self.ticker}: {e}")
            return None

    def fetch_sustainability(self):
        try:
            data = self.stock.sustainability
            if data is None or data.empty:
                print(f"No sustainability data found for {self.ticker}")
                return None
            self.data["sustainability"] = data
            return data
        except Exception as e:
            print(f"Error fetching sustainability for {self.ticker}: {e}")
            return None

    # --- Data Saving Methods to Local Files ---

    def save_analysis_to_file(self):
        """Save analyst recommendations data to CSV file."""
        if "analysis" not in self.data or self.data["analysis"] is None or self.data["analysis"].empty:
            print("No analysis data to save.")
            return
        self._save_df(self.data["analysis"], "analysis.csv")

    def save_balance_sheet_to_file(self):
        if "balance_sheet" not in self.data or self.data["balance_sheet"] is None or self.data["balance_sheet"].empty:
            print("No balance sheet data to save.")
            return
        self._save_df(self.data["balance_sheet"], "balance_sheet.csv")

    def save_cash_flows_to_file(self):
        if "cashflow" not in self.data or self.data["cashflow"] is None or self.data["cashflow"].empty:
            print("No cash flow data to save.")
            return
        self._save_df(self.data["cashflow"], "cashflow.csv")

    def save_dividends_to_file(self):
        if "dividends" not in self.data or self.data["dividends"] is None or self.data["dividends"].empty:
            print("No dividends data to save.")
            return
        self._save_df(self.data["dividends"], "dividends.csv")

    def save_earnings_to_file(self):
        if "earnings" not in self.data or self.data["earnings"] is None or self.data["earnings"].empty:
            print("No earnings data to save.")
            return
        self._save_df(self.data["earnings"], "earnings.csv")

    def save_financials_to_file(self):
        if "financials" not in self.data or self.data["financials"] is None or self.data["financials"].empty:
            print("No financials data to save.")
            return
        self._save_df(self.data["financials"], "financials.csv")

    def save_history_to_file(self):
        if "history" not in self.data or self.data["history"] is None or self.data["history"].empty:
            print("No history data to save.")
            return
        self._save_df(self.data["history"], "history.csv")

    def save_info_to_file(self):
        if "info" not in self.data or not self.data["info"]:
            print("No info data to save.")
            return
        self._save_json(self.data["info"], "info.json")

    def save_institutional_holders_to_file(self):
        if "institutional_holders" not in self.data or self.data["institutional_holders"] is None or self.data["institutional_holders"].empty:
            print("No institutional holders data to save.")
            return
        self._save_df(self.data["institutional_holders"], "institutional_holders.csv")

    def save_major_holders_to_file(self):
        if "major_holders" not in self.data or self.data["major_holders"] is None or self.data["major_holders"].empty:
            print("No major holders data to save.")
            return
        self._save_df(self.data["major_holders"], "major_holders.csv")

    def save_options_to_file(self):
        if "options" not in self.data or not self.data["options"]:
            print("No options data to save.")
            return
        self._save_json(self.data["options"], "options.json")

    def save_splits_to_file(self):
        if "splits" not in self.data or self.data["splits"] is None or self.data["splits"].empty:
            print("No splits data to save.")
            return
        self._save_df(self.data["splits"], "splits.csv")

    def save_sustainability_to_file(self):
        if "sustainability" not in self.data or self.data["sustainability"] is None or self.data["sustainability"].empty:
            print("No sustainability data to save.")
            return
        self._save_df(self.data["sustainability"], "sustainability.csv")

    # --- Data Loading Methods from Local Files ---

    def load_analysis_from_file(self):
        """Load analyst recommendations data from CSV file."""
        df = self._load_df("analysis.csv")
        if df is not None and not df.empty:
            self.data["analysis"] = df
        return df

    def load_balance_sheet_from_file(self):
        df = self._load_df("balance_sheet.csv")
        if df is not None and not df.empty:
            self.data["balance_sheet"] = df
        return df

    def load_cash_flows_from_file(self):
        df = self._load_df("cashflow.csv")
        if df is not None and not df.empty:
            self.data["cashflow"] = df
        return df

    def load_dividends_from_file(self):
        df = self._load_df("dividends.csv")
        if df is not None and not df.empty:
            self.data["dividends"] = df
        return df

    def load_earnings_from_file(self):
        df = self._load_df("earnings.csv")
        if df is not None and not df.empty:
            self.data["earnings"] = df
        return df

    def load_financials_from_file(self):
        df = self._load_df("financials.csv")
        if df is not None and not df.empty:
            self.data["financials"] = df
        return df

    def load_history_from_file(self):
        df = self._load_df("history.csv")
        if df is not None and not df.empty:
            self.data["history"] = df
        return df

    def load_info_from_file(self):
        info = self._load_json("info.json")
        if info is not None:
            self.data["info"] = info
        return info

    def load_institutional_holders_from_file(self):
        df = self._load_df("institutional_holders.csv")
        if df is not None and not df.empty:
            self.data["institutional_holders"] = df
        return df

    def load_major_holders_from_file(self):
        df = self._load_df("major_holders.csv")
        if df is not None and not df.empty:
            self.data["major_holders"] = df
        return df

    def load_options_from_file(self):
        options = self._load_json("options.json")
        if options is not None:
            self.data["options"] = options
        return options

    def load_splits_from_file(self):
        df = self._load_df("splits.csv")
        if df is not None and not df.empty:
            self.data["splits"] = df
        return df

    def load_sustainability_from_file(self):
        df = self._load_df("sustainability.csv")
        if df is not None and not df.empty:
            self.data["sustainability"] = df
        return df

    # --- Database update ---

    def update_database_info(self):
        if "info" not in self.data or not self.data["info"]:
            print("No info data to update database with.")
            return
        conn = self._connect_db()
        if not conn:
            return
        try:
            cur = conn.cursor()
            columns = ["full_name", "price", "fifty_day_avg", "two_hundred_day_avg", "forward_pe"]
            values = [
                self.data["info"].get("longName", "Unknown"),
                self.data["info"].get("regularMarketPrice", 0.0),
                self.data["info"].get("fiftyDayAverage", 0.0),
                self.data["info"].get("twoHundredDayAverage", 0.0),
                self.data["info"].get("forwardPE", 0.0),
            ]
            sql = f"""
                INSERT INTO stock_data (ticker, {', '.join(columns)})
                VALUES (?, {', '.join(['?']*len(columns))})
                ON CONFLICT(ticker) DO UPDATE SET
                {', '.join([f"{col}=excluded.{col}" for col in columns])}
            """
            cur.execute(sql, [self.ticker] + values)
            conn.commit()
            print(f"Updated database with info for {self.ticker}")
        except Exception as e:
            print(f"Error updating database: {e}")
        finally:
            conn.close()

if __name__ == "__main__":
    ticker = "AAPL"
    stock = Stock(ticker)
    stock.fetch_data()
    stock.fetch_history()
    stock.fetch_splits()
    stock.fetch_dividends()
    stock.fetch_analysis()
    stock.fetch_earnings()
    stock.fetch_balance_sheet()
    stock.fetch_cash_flows()
    stock.fetch_financials()
    stock.fetch_institutional_holders()
    stock.fetch_major_holders()
    stock.fetch_sustainability()
    stock.fetch_options()
    stock.save_info_to_file()
    stock.save_history_to_file()
    stock.save_splits_to_file()
    stock.save_dividends_to_file()
    stock.save_analysis_to_file()
    stock.save_earnings_to_file()
    stock.save_balance_sheet_to_file()
    stock.save_cash_flows_to_file()
    stock.save_financials_to_file()
    stock.save_institutional_holders_to_file()
    stock.save_major_holders_to_file()
    stock.save_sustainability_to_file()
    stock.save_options_to_file()
    stock.load_info_from_file()
    stock.load_history_from_file()
    stock.load_splits_from_file()
    stock.load_dividends_from_file()
    stock.load_analysis_from_file()
    stock.load_earnings_from_file()
    stock.load_balance_sheet_from_file()
    stock.load_cash_flows_from_file()
    stock.load_financials_from_file()
    stock.load_institutional_holders_from_file()
    stock.load_major_holders_from_file()
    stock.load_sustainability_from_file()
    stock.load_options_from_file()
    stock.update_database_info()
    print("Info Data:", stock.data.get("info"))
    print("History Data:", stock.data.get("history"))
    print("Splits Data:", stock.data.get("splits"))
    print("Dividends Data:", stock.data.get("dividends"))
    print("Analysis Data:", stock.data.get("analysis"))
    print("Earnings Data:", stock.data.get("earnings"))
    print("Balance Sheet Data:", stock.data.get("balance_sheet"))
    print("Cash Flows Data:", stock.data.get("cashflow"))
    print("Financials Data:", stock.data.get("financials"))
    print("Institutional Holders Data:", stock.data.get("institutional_holders"))
    print("Major Holders Data:", stock.data.get("major_holders"))
    print("Sustainability Data:", stock.data.get("sustainability"))
    print("Options Data:", stock.data.get("options"))
    print("Database updated successfully.")
