# helpers.py

import yfinance as yf
import json

def usd(value):
    """Format value as USD."""
    return f"${value:,.2f}"


def lookup(symbol):
    """Look up quote for symbol."""
    try:
        stock = yf.Ticker(symbol)
        info = stock.info

        price = info.get("regularMarketPrice")
        if price is None:
            return None

        return {
            "symbol": symbol.upper(),
            "price": float(price)
        }
    except Exception:
        return None