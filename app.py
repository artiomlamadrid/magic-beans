#app.py


from cs50 import SQL
from flask import Flask, flash, redirect, render_template, request, session
from flask_session import Session
from werkzeug.security import check_password_hash, generate_password_hash
from stock import Stock
from helpers import apology, login_required, usd
import sqlite3, os, json

# Configure application
app = Flask(__name__)
app.secret_key = "d7e9a6f8c4b12e03a59f7d8c6e4b1a0f3c29d5b7e8f1a6c4d7b9e3f0a2c1d8e7"

# Custom filter
app.jinja_env.filters["usd"] = usd

# Configure session to use filesystem (instead of signed cookies)
app.config["SESSION_PERMANENT"] = False
app.config["SESSION_TYPE"] = "filesystem"
Session(app)

# Configure CS50 Library to use SQLite database
db = SQL("sqlite:///magic_beans.db")


@app.after_request
def after_request(response):
    """Ensure responses aren't cached"""
    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    response.headers["Expires"] = 0
    response.headers["Pragma"] = "no-cache"
    return response


@app.route("/")
@login_required
def index():
    return render_template("index.html")


@app.route("/login", methods=["GET", "POST"])
def login():
    """
    Log in a user using their email and password.
    """
    # Forget any existing session
    session.clear()

    if request.method == "POST":
        email = request.form.get("email")
        password = request.form.get("password")

        # Ensure email was submitted
        if not email:
            flash("Must provide email")
            return render_template("login.html")

        # Ensure password was submitted
        if not password:
            flash("Must provide password")
            return render_template("login.html")

        # Query database for the user by email
        rows = db.execute(
            "SELECT * FROM users WHERE email = ?",
            email
        )

        # Check if user exists and password is correct
        if len(rows) != 1 or not check_password_hash(rows[0]["hash"], password):
            flash("Invalid email or password")
            return render_template("login.html")

        # Remember the user_id in session
        session["user_id"] = rows[0]["user_id"]

        # Redirect to home page
        return redirect("/")

    # GET request – show login form
    return render_template("login.html")

@app.route("/logout")
def logout():
    """
    Log out the current user by clearing the session.
    """
    session.clear()
    flash("You have been logged out.")
    return redirect("/login")

@app.route("/quote", methods=["GET", "POST"])
def quote():
    stock = None
    data_type = "info"  # default data type
    fetched_data = None

    if request.method == "POST":
        ticker = request.form.get("ticker", "").upper().strip()
        data_type = request.form.get("data_type", "info")
        action = request.form.get("action")

        if not ticker:
            flash("Please enter a ticker symbol.", "warning")
            return redirect(url_for("quote"))

        stock = Stock(ticker)

        # Fetcha data beroende på valt data_type
        if action == "fetch_data":
            if data_type == "info":
                data = stock.fetch_data()
                stock.data = data  # spara i objektet
                fetched_data = data

            elif data_type == "history":
                data = stock.fetch_history()
                stock.history = data
                fetched_data = data.to_dict(orient="records") if data is not None else None

            elif data_type == "dividends":
                data = stock.fetch_dividends()
                stock.dividends = data
                fetched_data = data.to_dict() if data is not None else None

            elif data_type == "splits":
                data = stock.fetch_splits()
                stock.splits = data
                fetched_data = data.to_dict() if data is not None else None

            if fetched_data:
                flash(f"Successfully fetched {data_type} data for {ticker}.", "success")
                # Spara i session för save/load knappar
                if "fetched" not in session:
                    session["fetched"] = {}
                session["fetched"][data_type] = True
                session.modified = True
            else:
                flash(f"No {data_type} data found for {ticker}.", "warning")

            # Spara basic info oavsett (för visning i tabell)
            if data_type == "info" and data:
                stock.data = data

        elif action == "save_data":
            # Spara datan till filer och/eller DB beroende på data_type
            if data_type == "info":
                stock.data = stock.fetch_data()
                if stock.data:
                    stock.save_data_to()
                    stock.update_database_info()
                    flash("Info data saved to file and database.", "success")
                else:
                    flash("No info data to save.", "warning")

            elif data_type == "history":
                stock.history = stock.fetch_history()
                if stock.history is not None:
                    stock.save_history_to()
                    flash("History data saved to file.", "success")
                else:
                    flash("No history data to save.", "warning")

            elif data_type == "dividends":
                stock.dividends = stock.fetch_dividends()
                if stock.dividends is not None:
                    stock.save_dividends_to()
                    flash("Dividends data saved to file.", "success")
                else:
                    flash("No dividends data to save.", "warning")

            elif data_type == "splits":
                stock.splits = stock.fetch_splits()
                if stock.splits is not None:
                    stock.save_splits_to()
                    flash("Splits data saved to file.", "success")
                else:
                    flash("No splits data to save.", "warning")

        elif action == "load_data":
            # Ladda data från filer (för demo: läs fil från stocks/<ticker>/<fil>)
            base_folder = "stocks"
            folder = os.path.join(base_folder, ticker)

            try:
                if data_type == "info":
                    filepath = os.path.join(folder, "info.json")
                    with open(filepath, "r") as f:
                        loaded = json.load(f)
                    stock.data = loaded
                    fetched_data = loaded
                    flash("Loaded info data from file.", "success")

                elif data_type == "history":
                    filepath = os.path.join(folder, "history.csv")
                    df = pd.read_csv(filepath)
                    stock.history = df
                    fetched_data = df.to_dict(orient="records")
                    flash("Loaded history data from file.", "success")

                elif data_type == "dividends":
                    filepath = os.path.join(folder, "dividends.csv")
                    df = pd.read_csv(filepath)
                    stock.dividends = df
                    fetched_data = df.to_dict()
                    flash("Loaded dividends data from file.", "success")

                elif data_type == "splits":
                    filepath = os.path.join(folder, "splits.csv")
                    df = pd.read_csv(filepath)
                    stock.splits = df
                    fetched_data = df.to_dict()
                    flash("Loaded splits data from file.", "success")

            except Exception as e:
                flash(f"Failed to load {data_type} data from file: {e}", "danger")

        # Skicka med data för visning i template
        stock_info = {
            "ticker": ticker,
            "message": f"Showing {data_type} data for {ticker}",
            "data_type": data_type,
            "data": fetched_data
        }
        return render_template("quote.html", stock=stock_info, selected_data_type=data_type)

    # GET request
    return render_template("quote.html", stock=None, selected_data_type=data_type)


@app.route("/register", methods=["GET", "POST"])
def register():
    """Register user"""

    if request.method == "POST":
        email = request.form.get("email")
        password = request.form.get("password")
        confirmation = request.form.get("confirmation")

        if not email:
            return apology("must provide email", 400)
        if not password:
            return apology("must provide password", 400)
        if not confirmation:
            return apology("must confirm password", 400)
        if password != confirmation:
            return apology("password and confirmation don't match", 400)

        rows = db.execute("SELECT * FROM users WHERE email = ?", email)
        if len(rows) != 0:
            return apology("email already registered", 400)

        hashed = generate_password_hash(password, method='pbkdf2')
        db.execute("INSERT INTO users (email, hash) VALUES (?, ?)", email, hashed)

        user_id = db.execute("SELECT user_id FROM users WHERE email = ?", email)[0]["user_id"]
        session["user_id"] = user_id

        return redirect("/")

    else:
        return render_template("register.html")


if __name__ == "__main__":
    app.run(debug=True)