import yfinance as yf
import json
import os
import pandas as pd
import sqlite3

class Stock:
    def __init__(self, ticker, db_path="magic_beans.db", base_folder="stocks"):
        self.ticker = ticker.upper()
        self.stock = yf.Ticker(ticker)
        self.db_path = db_path
        self.base_folder = base_folder
        self.data = {}

    # --- Fetch methods ---

    def fetch_data(self):
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

    def fetch_dividends(self):
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

    def fetch_analysis(self):
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
            # Fetch option chain for the first expiration date
            opt = self.stock.option_chain(options[0])
            result = {
                "calls": opt.calls.to_dict(orient="records"),
                "puts": opt.puts.to_dict(orient="records"),
                "expiration_date": options[0]
            }
            self.data["options"] = result
            return result
        except Exception as e:
            print(f"Error fetching options for {self.ticker}: {e}")
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

    # --- Save helpers ---

    def _save_df(self, df, filename):
        folder = os.path.join(self.base_folder, self.ticker)
        os.makedirs(folder, exist_ok=True)
        path = os.path.join(folder, filename)
        try:
            # Handle Series by converting to DataFrame
            if isinstance(df, pd.Series):
                df_to_save = df.to_frame().reset_index()
                # Rename the column to something meaningful
                df_to_save.columns = ['Date', filename.split('.')[0].capitalize()]
            elif isinstance(df.index, pd.DatetimeIndex) and "Date" not in df.columns:
                df_to_save = df.reset_index()
            else:
                df_to_save = df.copy()
            df_to_save.to_csv(path, index=False)
            print(f"Saved {filename} to {path}")
        except Exception as e:
            print(f"Error saving {filename} to file: {e}")
            raise

    def _save_json(self, obj, filename):
        folder = os.path.join(self.base_folder, self.ticker)
        os.makedirs(folder, exist_ok=True)
        path = os.path.join(folder, filename)
        try:
            with open(path, "w") as f:
                json.dump(obj, f, indent=2)
            print(f"Saved {filename} to {path}")
        except Exception as e:
            print(f"Error saving {filename} to file: {e}")
            raise

    def save_info_to_file(self):
        if "info" not in self.data or not self.data["info"]:
            print("No info data to save.")
            return
        self._save_json(self.data["info"], "info.json")

    def save_history_to_file(self):
        if "history" not in self.data or self.data["history"] is None or self.data["history"].empty:
            print("No history data to save.")
            return
        self._save_df(self.data["history"], "history.csv")

    def save_splits_to_file(self):
        if "splits" not in self.data or self.data["splits"] is None or self.data["splits"].empty:
            print("No splits data to save.")
            return
        self._save_df(self.data["splits"], "splits.csv")

    def save_dividends_to_file(self):
        if "dividends" not in self.data or self.data["dividends"] is None or self.data["dividends"].empty:
            print("No dividends data to save.")
            return
        self._save_df(self.data["dividends"], "dividends.csv")

    def save_analysis_to_file(self):
        if "analysis" not in self.data or self.data["analysis"] is None or self.data["analysis"].empty:
            print("No analysis data to save.")
            return
        self._save_df(self.data["analysis"], "analysis.csv")

    def save_earnings_to_file(self):
        if "earnings" not in self.data or self.data["earnings"] is None or self.data["earnings"].empty:
            print("No earnings data to save.")
            return
        self._save_df(self.data["earnings"], "earnings.csv")

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

    def save_sustainability_to_file(self):
        if "sustainability" not in self.data or self.data["sustainability"] is None or self.data["sustainability"].empty:
            print("No sustainability data to save.")
            return
        self._save_df(self.data["sustainability"], "sustainability.csv")

    # --- Load helpers ---

    def _load_df(self, filename):
        path = os.path.join(self.base_folder, self.ticker, filename)
        if not os.path.exists(path):
            print(f"File not found: {path}")
            return None
        try:
            df = pd.read_csv(path)
            if "Date" in df.columns:
                df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
                df = df.set_index("Date", drop=True)
            return df
        except Exception as e:
            print(f"Error loading {filename} from file: {e}")
            return None

    def _load_json(self, filename):
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

    def load_info_from_file(self):
        info = self._load_json("info.json")
        if info is not None:
            self.data["info"] = info
        return info

    def load_history_from_file(self):
        df = self._load_df("history.csv")
        if df is not None and not df.empty:
            self.data["history"] = df
        return df

    def save_cash_flow(self):
        if "cashflow" not in self.data or self.data["cashflow"] is None or self.data["cashflow"].empty:
            print("No cashflow data to save.")
            return
        self._save_df(self.data["cashflow"], "cashflow.csv")

    def load_cash_flow(self):
        df = self._load_df("cashflow.csv")
        if df is not None and not df.empty:
            self.data["cashflow"] = df
        return df

    def save_income_statement(self):
        if "income_statement" not in self.data or self.data["income_statement"] is None or self.data["income_statement"].empty:
            print("No income statement data to save.")
            return
        self._save_df(self.data["income_statement"], "income_statement.csv")

    def load_income_statement(self):
        df = self._load_df("income_statement.csv")
        if df is not None and not df.empty:
            self.data["income_statement"] = df
        return df

    def save_balance_sheet(self):
        if "balance_sheet" not in self.data or self.data["balance_sheet"] is None or self.data["balance_sheet"].empty:
            print("No balance sheet data to save.")
            return
        self._save_df(self.data["balance_sheet"], "balance_sheet.csv")

    def load_balance_sheet(self):
        df = self._load_df("balance_sheet.csv")
        if df is not None and not df.empty:
            self.data["balance_sheet"] = df
        return df

    def load_splits_from_file(self):
        df = self._load_df("splits.csv")
        if df is not None and not df.empty:
            self.data["splits"] = df
        return df

    def load_dividends_from_file(self):
        df = self._load_df("dividends.csv")
        if df is not None and not df.empty:
            self.data["dividends"] = df
        return df

    def load_analysis_from_file(self):
        df = self._load_df("analysis.csv")
        if df is not None and not df.empty:
            self.data["analysis"] = df
        return df

    def load_earnings_from_file(self):
        df = self._load_df("earnings.csv")
        if df is not None and not df.empty:
            self.data["earnings"] = df
        return df

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

    def _connect_db(self):
        try:
            return sqlite3.connect(self.db_path)
        except Exception as e:
            print(f"Error connecting to DB: {e}")
            return None
# debug
# if __name__ == "__main__":
#     ticker = "AAPL"
#     stock = Stock(ticker)
#     stock.fetch_data()
#     stock.fetch_history()
#     stock.fetch_splits()
#     stock.fetch_dividends()
#     stock.save_info_to_file()
#     stock.save_history_to_file()
#     stock.save_splits_to_file()
#     stock.save_dividends_to_file()
#     stock.update_database_info()
#     print("Info Data:", stock.data.get("info"))
#     print("History Data:", stock.data.get("history"))
#     print("Splits Data:", stock.data.get("splits"))
#     print("Dividends Data:", stock.data.get("dividends"))
#     print("Database updated successfully.")