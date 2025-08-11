# stock_analysis.py

from stock import Stock
import numpy as np

from stock import Stock

class StockAnalysis(Stock):
    def __init__(self, ticker, db_path="magic_beans.db", discount_rate=0.1):
        super().__init__(ticker, db_path)
        self.discount_rate = discount_rate

    def discounted_cash_flow(self, forecast_years=5, growth_rate=0.02, terminal_growth=0.02):
        if not hasattr(self, "data") or not self.data:
            raise ValueError("No info data loaded. Call load_data_from_file() first.")

        fcf = self.data.get("freeCashflow")
        if fcf is None:
            raise ValueError("freeCashflow not available in loaded data.")

        discounted_fcf = 0
        for year in range(1, forecast_years + 1):
            projected_fcf = fcf * ((1 + growth_rate) ** year)
            discounted_fcf += projected_fcf / ((1 + self.discount_rate) ** year)

        terminal_value = (fcf * (1 + growth_rate) ** forecast_years) * (1 + terminal_growth) / (self.discount_rate - terminal_growth)
        discounted_terminal = terminal_value / ((1 + self.discount_rate) ** forecast_years)

        return discounted_fcf + discounted_terminal

    def dividend_discount_model(self, forecast_years=5, dividend_growth=0.03):
        if not hasattr(self, "dividends") or self.dividends is None:
            raise ValueError("No dividends data loaded. Call load_dividends_from_file() first.")

        if self.dividends.empty:
            raise ValueError("Loaded dividends data is empty.")

        last_dividend = self.dividends.iloc[-1]

        discounted_divs = 0
        for year in range(1, forecast_years + 1):
            projected_div = last_dividend * ((1 + dividend_growth) ** year)
            discounted_divs += projected_div / ((1 + self.discount_rate) ** year)

        terminal_value = (last_dividend * ((1 + dividend_growth) ** forecast_years)) * (1 + dividend_growth) / (self.discount_rate - dividend_growth)
        discounted_terminal = terminal_value / ((1 + self.discount_rate) ** forecast_years)

        return discounted_divs + discounted_terminal

    def price_earnings_valuation(self, peer_pe=15):
        if not hasattr(self, "data") or not self.data:
            raise ValueError("No info data loaded. Call load_data_from_file() first.")

        eps = self.data.get("trailingEps")
        if eps is None:
            raise ValueError("trailingEps not available in loaded data.")

        return eps * peer_pe
    
if __name__ == "__main__":
    ticker = "AAPL"
    analysis = StockAnalysis(ticker)
    analysis.fetch_data()
    analysis.fetch_history()
    analysis.fetch_splits()
    analysis.fetch_dividends()

    analysis.discounted_cash_flow()
    analysis.dividend_discount_model()
    analysis.price_earnings_valuation()

    print("Discounted Cash Flow Valuation:", analysis.discounted_cash_flow())
    print("Dividend Discount Model Valuation:", analysis.dividend_discount_model())
    print("Price/Earnings Valuation:", analysis.price_earnings_valuation())