from cs50 import SQL
from flask import Flask, flash, redirect, render_template, request, session, get_flashed_messages
from flask_session import Session
from werkzeug.security import check_password_hash, generate_password_hash
from stock import Stock
from helpers import apology, login_required, usd
import os
import pandas as pd

app = Flask(__name__)
app.secret_key = "d7e9a6f8c4b12e03a59f7d8c6e4b1a0f3c29d5b7e8f1a6c4d7b9e3f0a2c1d8e7"
app.jinja_env.filters["usd"] = usd
app.config["SESSION_PERMANENT"] = False
app.config["SESSION_TYPE"] = "filesystem"
Session(app)
db = SQL("sqlite:///magic_beans.db")

@app.after_request
def after_request(response):
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
    session.clear()
    if request.method == "POST":
        email = request.form.get("email")
        password = request.form.get("password")
        if not email:
            flash("Must provide email", "warning")
            return render_template("login.html")
        if not password:
            flash("Must provide password", "warning")
            return render_template("login.html")
        rows = db.execute("SELECT * FROM users WHERE email = ?", email)
        if len(rows) != 1 or not check_password_hash(rows[0]["hash"], password):
            flash("Invalid email or password", "danger")
            return render_template("login.html")
        session["user_id"] = rows[0]["user_id"]
        return redirect("/")
    return render_template("login.html")

@app.route("/logout")
def logout():
    session.clear()
    flash("You have been logged out.", "info")
    return redirect("/login")

@app.route("/quote", methods=["GET", "POST"])
@login_required
def quote():
    stock = None
    data_type = "info"
    fetched_data = None

    supported_data_types = [
        "info", "history", "dividends", "splits",
        "analysis", "earnings", "institutional_holders",
        "major_holders", "options", "sustainability"
    ]

    if request.method == "POST":
        ticker = request.form.get("ticker", "").upper().strip()
        data_type = request.form.get("data_type", "info")
        action = request.form.get("action")

        if not ticker:
            flash("Please enter a ticker symbol.", "warning")
            return render_template("quote.html", stock=None, selected_data_type=data_type, data=None)

        if data_type not in supported_data_types:
            flash(f"Unsupported data type: {data_type}", "danger")
            return render_template("quote.html", stock=None, selected_data_type=data_type, data=None)

        if "fetched" not in session or session["fetched"].get("ticker") != ticker:
            session["fetched"] = {"ticker": ticker}
        session.modified = True

        stock = Stock(ticker)

        fetch_map = {
            "info": stock.fetch_data,
            "history": stock.fetch_history,
            "dividends": stock.fetch_dividends,
            "splits": stock.fetch_splits,
            "analysis": stock.fetch_analysis,
            "earnings": stock.fetch_earnings,
            "institutional_holders": stock.fetch_institutional_holders,
            "major_holders": stock.fetch_major_holders,
            "options": stock.fetch_options,
            "sustainability": stock.fetch_sustainability,
        }

        save_map = {
            "info": stock.save_info_to_file,
            "history": stock.save_history_to_file,
            "dividends": stock.save_dividends_to_file,
            "splits": stock.save_splits_to_file,
            "analysis": stock.save_analysis_to_file,
            "earnings": stock.save_earnings_to_file,
            "institutional_holders": stock.save_institutional_holders_to_file,
            "major_holders": stock.save_major_holders_to_file,
            "options": stock.save_options_to_file,
            "sustainability": stock.save_sustainability_to_file,
        }

        load_map = {
            "info": stock.load_info_from_file,
            "history": stock.load_history_from_file,
            "dividends": stock.load_dividends_from_file,
            "splits": stock.load_splits_from_file,
            "analysis": stock.load_analysis_from_file,
            "earnings": stock.load_earnings_from_file,
            "institutional_holders": stock.load_institutional_holders_from_file,
            "major_holders": stock.load_major_holders_from_file,
            "options": stock.load_options_from_file,
            "sustainability": stock.load_sustainability_from_file,
        }

        def to_records(data):
            if data is None:
                return None
            if isinstance(data, pd.DataFrame):
                df = data.copy()
                if df.index.name == "Date" and "Date" not in df.columns:
                    df = df.reset_index()
                return df.to_dict(orient="records")
            elif isinstance(data, pd.Series):
                return [{"Date": str(idx), data.name: val} for idx, val in data.items()]
            elif isinstance(data, dict):
                return data
            elif isinstance(data, (list, tuple)) and len(data) > 0 and isinstance(data[0], dict):
                return data
            return data

        if action == "fetch_data":
            fetch_func = fetch_map.get(data_type)
            if fetch_func:
                try:
                    data = fetch_func()
                    stock.data[data_type] = data
                    if data is None or (hasattr(data, 'empty') and data.empty):
                        flash(f"No {data_type.replace('_', ' ')} data found for {ticker}.", "warning")
                        fetched_data = None
                    else:
                        fetched_data = to_records(data)
                        flash(f"Successfully fetched {data_type.replace('_', ' ')} data for {ticker}.", "success")
                        session["fetched"][data_type] = True
                        session.modified = True
                except Exception as e:
                    flash(f"Error fetching {data_type.replace('_', ' ')} data for {ticker}: {str(e)}", "danger")
                    fetched_data = None
            else:
                flash(f"Fetching {data_type.replace('_', ' ')} not supported.", "danger")

        elif action == "save_data":
            save_func = save_map.get(data_type)
            data_to_save = stock.data.get(data_type)
            if data_to_save is None:
                fetch_func = fetch_map.get(data_type)
                if fetch_func:
                    try:
                        data_to_save = fetch_func()
                        stock.data[data_type] = data_to_save
                    except Exception as e:
                        flash(f"Error fetching {data_type.replace('_', ' ')} for saving: {str(e)}", "danger")
                        return render_template("quote.html", stock={"ticker": ticker}, selected_data_type=data_type, data=None)

            if data_to_save is None or (hasattr(data_to_save, 'empty') and data_to_save.empty):
                flash(f"No {data_type.replace('_', ' ')} data to save for {ticker}.", "warning")
                fetched_data = None
            else:
                if save_func:
                    try:
                        save_func()
                        if data_type == "info":
                            stock.update_database_info()
                        flash(f"{data_type.replace('_', ' ').capitalize()} data saved for {ticker}.", "success")
                        fetched_data = to_records(data_to_save)  # Preserve the displayed data
                    except Exception as e:
                        flash(f"Failed to save {data_type.replace('_', ' ')} data for {ticker}: {str(e)}", "danger")
                        fetched_data = None
                else:
                    flash(f"Saving {data_type.replace('_', ' ')} data not implemented.", "danger")
                    fetched_data = None

        elif action == "load_data":
            load_func = load_map.get(data_type)
            if not load_func:
                flash(f"Loading {data_type.replace('_', ' ')} data not supported.", "danger")
            else:
                try:
                    loaded = load_func()
                    if loaded is None or (hasattr(loaded, 'empty') and loaded.empty):
                        flash(f"No saved {data_type.replace('_', ' ')} data found for {ticker}.", "warning")
                        fetched_data = None
                    else:
                        stock.data[data_type] = loaded
                        fetched_data = to_records(loaded)
                        flash(f"Loaded {data_type.replace('_', ' ')} data for {ticker}.", "success")
                except Exception as e:
                    flash(f"Error loading {data_type.replace('_', ' ')} data for {ticker}: {str(e)}", "danger")
                    fetched_data = None

        else:
            flash("Invalid action.", "danger")

        response = render_template(
            "quote.html",
            stock={"ticker": ticker},
            selected_data_type=data_type,
            data=fetched_data
        )
        get_flashed_messages()
        return response

    return render_template("quote.html", stock=None, selected_data_type=data_type)

@app.route("/register", methods=["GET", "POST"])
def register():
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