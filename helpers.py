# helpers.py

from flask import redirect, render_template, session
from functools import wraps

import yfinance as yf
import json

def apology(message, code=400):
    """Render message as an apology to user."""

    def escape(s):
        """
        Escape special characters.

        https://github.com/jacebrowning/memegen#special-characters
        """
        for old, new in [
            ("-", "--"),
            (" ", "-"),
            ("_", "__"),
            ("?", "~q"),
            ("%", "~p"),
            ("#", "~h"),
            ("/", "~s"),
            ('"', "''"),
        ]:
            s = s.replace(old, new)
        return s

    return render_template("apology.html", top=code, bottom=escape(message)), code


def login_required(f):
    """
    Decorate routes to require login.

    https://flask.palletsprojects.com/en/latest/patterns/viewdecorators/
    """

    @wraps(f)
    def decorated_function(*args, **kwargs):
        if session.get("user_id") is None:
            return redirect("/login")
        return f(*args, **kwargs)

    return decorated_function
import requests

# def lookup(symbol): # Removed temporarily
#     """Look up quote for symbol from Alpha Vantage."""

#     try:
#         api_key = "6HXQBD2PPOJMSQQZ"
#         url = f"https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol={symbol}&apikey={api_key}"
#         response = requests.get(url)
#         response.raise_for_status()
#     except requests.RequestException:
#         return None

#     try:
#         quote = response.json().get("Global Quote", {})
#         price = float(quote.get("05. price"))
#         name = symbol.upper()  # Alpha Vantage ger inte fullständigt namn här
#         return {
#             "symbol": symbol.upper(),
#             "name": name,
#             "price": price
#         }
#     except (KeyError, TypeError, ValueError):
#         return None


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


def usd(value):
    """Format value as USD."""
    return f"${value:,.2f}"