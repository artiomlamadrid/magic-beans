# stock.py

import yfinance as yf
import json
import os
import pandas as pd
import sqlite3

class Stock():
    def __init__(self, ticker, db_path="magic_beans.db"):
        self.ticker = ticker
        self.stock = yf.Ticker(ticker)
        self.db_path = db_path

    def fetch_data(self):
        try:
            self.data = self.stock.info
            if not self.data:
                print(f"No data found for {self.ticker}")
                return None
            return self.data
        except Exception as e:
            print(f"Error fetching data for {self.ticker}: {e}")
            return None
    
    def fetch_history(self):
        try:
            history = self.stock.history(period="max")
            if history.empty:
                print(f"No historical data found for {self.ticker}")
                return None
            self.history = history
            return history
        except Exception as e:
            print(f"Error fetching historical data for {self.ticker}: {e}")
            return None

    def fetch_splits(self):
        try:
            splits = self.stock.splits
            if splits.empty:
                print(f"No splits data found for {self.ticker}")
                return None
            self.splits = splits
            return splits
        except Exception as e:
            print(f"Error fetching splits data for {self.ticker}: {e}")
            return None
    
    def fetch_dividends(self):
        try:
            dividends = self.stock.dividends
            if dividends.empty:
                print(f"No dividends data found for {self.ticker}")
                return None
            self.dividends = dividends
            return dividends
        except Exception as e:
            print(f"Error fetching dividends data for {self.ticker}: {e}")
            return None

    # ---- Spara till fil ----

    def save_data_to(self, base_folder="stocks"):
        if not hasattr(self, "data") or not self.data:
            print("No general info data to save.")
            return
        folder = os.path.join(base_folder, self.ticker)
        os.makedirs(folder, exist_ok=True)
        filepath = os.path.join(folder, "info.json")
        try:
            with open(filepath, "w") as f:
                json.dump(self.data, f, indent=2)
            print(f"Saved info data to {filepath}")
        except Exception as e:
            print(f"Error saving info data to file: {e}")
    
    def save_history_to(self, base_folder="stocks"):
        if not hasattr(self, "history") or self.history is None:
            print("No history data to save.")
            return
        folder = os.path.join(base_folder, self.ticker)
        os.makedirs(folder, exist_ok=True)
        filepath = os.path.join(folder, "history.csv")
        try:
            self.history.to_csv(filepath)  # Här ska det vara self.history, inte self.splits
            print(f"Saved history data to {filepath}")
        except Exception as e:
            print(f"Error saving history data to file: {e}")


    def save_splits_to(self, base_folder="stocks"):
        if not hasattr(self, "splits") or self.splits is None:
            print("No split data to save.")
            return
        folder = os.path.join(base_folder, self.ticker)
        os.makedirs(folder, exist_ok=True)
        filepath = os.path.join(folder, "splits.csv")
        try:
            self.splits.to_csv(filepath)
            print(f"Saved split data to {filepath}")
        except Exception as e:
            print(f"Error saving split data to file: {e}")

    def save_dividends_to(self, base_folder="stocks"):
        if not hasattr(self, "dividends") or self.dividends is None:
            print("No dividends data to save.")
            return
        folder = os.path.join(base_folder, self.ticker)
        os.makedirs(folder, exist_ok=True)
        filepath = os.path.join(folder, "dividends.csv")
        try:
            self.dividends.to_csv(filepath)
            print(f"Saved dividends data to {filepath}")
        except Exception as e:
            print(f"Error saving dividends data to file: {e}")

    # ---- Uppdatera SQL-databas ----
   
    def _connect_db(self):
        try:
            conn = sqlite3.connect(self.db_path)
            return conn
        except Exception as e:
            print(f"Error connecting to database: {e}")
            return None

    def update_database_info(self):
        """Update the stock_data table with the latest info data."""
        if not hasattr(self, "data") or not self.data:
            print("No info data to update database with.")
            return
        conn = self._connect_db()
        if not conn:
            return

        try:
            cur = conn.cursor()
            columns = ["full_name", "price", "fifty_day_avg", "two_hundred_day_avg", "forward_pe"]
            
            # Bygg värden från self.data med fallback
            values = [
                self.data.get("longName", None),
                self.data.get("regularMarketPrice", None),
                self.data.get("fiftyDayAverage", None),
                self.data.get("twoHundredDayAverage", None),
                self.data.get("forwardPE", None)
            ]

            sql = f"""
            INSERT INTO stock_data (ticker, {", ".join(columns)})
            VALUES (?, {", ".join(["?"]*len(columns))})
            ON CONFLICT(ticker) DO UPDATE SET
            {", ".join([f"{col}=excluded.{col}" for col in columns])}
            """

            print("SQL to execute:", sql)
            print("Values:", [self.ticker] + values)

            cur.execute(sql, [self.ticker] + values)
            conn.commit()
            print(f"Updated stock_data table with info for {self.ticker}")
        except Exception as e:
            print(f"Error updating stock_data table: {e}")
        finally:
            conn.close()

    def dividends_as_dicts(self):
        if not hasattr(self, "dividends") or self.dividends is None:
            return []
        return [{"Date": str(date), "Dividends": float(value)} for date, value in self.dividends.items()]

    def splits_as_dicts(self):
        if not hasattr(self, "splits") or self.splits is None:
            return []
        return [{"Date": str(date), "Stock Splits": float(value)} for date, value in self.splits.items()]

def main():
    symbol = "AAPL"
    stock = Stock(symbol)

    print("Basic info:")
    stock.fetch_data()

    print("\nHistorical data:")
    print(stock.fetch_history())

    print("\nSplits:")
    print(stock.fetch_splits())

    print("\nDividends:")
    print(stock.fetch_dividends())

    print("\nAnalysis:")
    print(stock.fetch_analysis())

    print("\nFinancials:")
    print(stock.fetch_financials())

    print("\nBalance Sheet:")
    print(stock.fetch_balance_sheet())

    print("\nCashflow:")
    print(stock.fetch_cashflow())

    print("\nEarnings:")
    print(stock.fetch_earnings())

    print("\nInstitutional Holders:")
    print(stock.fetch_institutional_holders())

    print("\nMajor Holders:")
    print(stock.fetch_major_holders())

    print("\nOptions:")
    opts = stock.fetch_options()
    print(opts)

    if opts:
        print("\nOption Chain for first expiry:")
        print(stock.fetch_option_chain(opts[0]))

    print("\nSustainability:")
    print(stock.fetch_sustainability())

if __name__ == "__main__":
    main()