# Flask web application for stock data analysis and management
# This application provides a web interface for fetching, storing, and analyzing stock market data

# Import required libraries and modules
from cs50 import SQL
from flask import Flask, flash, redirect, render_template, request, session, get_flashed_messages
from flask_session import Session
from werkzeug.security import check_password_hash, generate_password_hash
from stock import Stock  # Custom Stock class for data fetching and management
from helpers import apology, login_required, usd  # Helper functions for the web app
import os
import pandas as pd

# Initialize Flask application
app = Flask(__name__)

# Configure application settings
app.secret_key = "d7e9a6f8c4b12e03a59f7d8c6e4b1a0f3c29d5b7e8f1a6c4d7b9e3f0a2c1d8e7"  # Secret key for session management
app.jinja_env.filters["usd"] = usd  # Add USD formatting filter to Jinja templates
app.config["SESSION_PERMANENT"] = False  # Sessions expire when browser is closed
app.config["SESSION_TYPE"] = "filesystem"  # Store session data in filesystem
Session(app)  # Initialize session management

# Initialize database connection
db = SQL("sqlite:///magic_beans.db")

# After request handler to prevent caching of sensitive data
@app.after_request
def after_request(response):
    """Ensure responses aren't cached for security purposes"""
    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    response.headers["Expires"] = 0
    response.headers["Pragma"] = "no-cache"
    return response

# Home page route - requires user authentication
@app.route("/")
@login_required
def index():
    """Display the main dashboard page"""
    return render_template("index.html")

# User authentication routes
@app.route("/login", methods=["GET", "POST"])
def login():
    """Handle user login with email and password authentication"""
    session.clear()  # Clear any existing session data
    
    if request.method == "POST":
        # Get form data
        email = request.form.get("email")
        password = request.form.get("password")
        
        # Validate input fields
        if not email:
            flash("Must provide email", "warning")
            return render_template("login.html")
        if not password:
            flash("Must provide password", "warning")
            return render_template("login.html")
            
        # Query database for user
        rows = db.execute("SELECT * FROM users WHERE email = ?", email)
        
        # Verify user exists and password is correct
        if len(rows) != 1 or not check_password_hash(rows[0]["hash"], password):
            flash("Invalid email or password", "danger")
            return render_template("login.html")
            
        # Login successful - create session
        session["user_id"] = rows[0]["user_id"]
        return redirect("/")
        
    # GET request - show login form
    return render_template("login.html")

@app.route("/logout")
def logout():
    """Handle user logout by clearing session data"""
    session.clear()
    flash("You have been logged out.", "info")
    return redirect("/login")

# Stock data management route
@app.route("/quote", methods=["GET", "POST"])
@login_required
def quote():
    """
    Handle stock data fetching, saving, and loading operations.
    Supports multiple data types: financial statements, market data, analysis data, etc.
    """
    # Initialize variables
    stock = None
    data_type = "info"  # Default data type
    fetched_data = None

    # Define supported data types for stock information
    supported_data_types = [
        "analysis", "balance_sheet", "cashflow", "dividends",
        "earnings", "financials", "history", "info",
        "institutional_holders", "major_holders", "options",
        "splits", "sustainability"
    ]

    if request.method == "POST":
        # Get form parameters
        ticker = request.form.get("ticker", "").upper().strip()  # Stock ticker symbol
        data_type = request.form.get("data_type", "info")        # Type of data to fetch
        action = request.form.get("action")                      # Action to perform (fetch/save/load)

        # Validate ticker symbol
        if not ticker:
            flash("Please enter a ticker symbol.", "warning")
            return render_template("quote.html", stock=None, selected_data_type=data_type, data=None)

        # Validate data type
        if data_type not in supported_data_types:
            flash(f"Unsupported data type: {data_type}", "danger")
            return render_template("quote.html", stock=None, selected_data_type=data_type, data=None)

        # Manage session data for the current ticker
        if "fetched" not in session or session["fetched"].get("ticker") != ticker:
            session["fetched"] = {"ticker": ticker}
        session.modified = True

        # Create Stock instance for the requested ticker
        stock = Stock(ticker)

        # Define mapping of data types to their corresponding fetch methods
        fetch_map = {
            "analysis": stock.fetch_analysis,
            "balance_sheet": stock.fetch_balance_sheet,
            "cashflow": stock.fetch_cash_flows,
            "dividends": stock.fetch_dividends,
            "earnings": stock.fetch_earnings,
            "financials": stock.fetch_financials,
            "history": stock.fetch_history,
            "info": stock.fetch_data,
            "institutional_holders": stock.fetch_institutional_holders,
            "major_holders": stock.fetch_major_holders,
            "options": stock.fetch_options,
            "splits": stock.fetch_splits,
            "sustainability": stock.fetch_sustainability,
        }

        # Define mapping of data types to their corresponding save methods
        save_map = {
            "analysis": stock.save_analysis_to_file,
            "balance_sheet": stock.save_balance_sheet_to_file,
            "cashflow": stock.save_cash_flows_to_file,
            "dividends": stock.save_dividends_to_file,
            "earnings": stock.save_earnings_to_file,
            "financials": stock.save_financials_to_file,
            "history": stock.save_history_to_file,
            "info": stock.save_info_to_file,
            "institutional_holders": stock.save_institutional_holders_to_file,
            "major_holders": stock.save_major_holders_to_file,
            "options": stock.save_options_to_file,
            "splits": stock.save_splits_to_file,
            "sustainability": stock.save_sustainability_to_file,
        }

        # Define mapping of data types to their corresponding load methods
        load_map = {
            "analysis": stock.load_analysis_from_file,
            "balance_sheet": stock.load_balance_sheet_from_file,
            "cashflow": stock.load_cash_flows_from_file,
            "dividends": stock.load_dividends_from_file,
            "earnings": stock.load_earnings_from_file,
            "financials": stock.load_financials_from_file,
            "history": stock.load_history_from_file,
            "info": stock.load_info_from_file,
            "institutional_holders": stock.load_institutional_holders_from_file,
            "major_holders": stock.load_major_holders_from_file,
            "options": stock.load_options_from_file,
            "splits": stock.load_splits_from_file,
            "sustainability": stock.load_sustainability_from_file,
        }

        def to_records(data):
            """
            Convert pandas DataFrames and Series to dictionary format for web display.
            Handles different data types and ensures proper JSON serialization.
            """
            if data is None:
                return None
            if isinstance(data, pd.DataFrame):
                df = data.copy()
                # Reset index if it's a DatetimeIndex without Date column
                if df.index.name == "Date" and "Date" not in df.columns:
                    df = df.reset_index()
                return df.to_dict(orient="records")
            elif isinstance(data, pd.Series):
                # Convert Series to list of dictionaries with Date and value
                return [{"Date": str(idx), data.name: val} for idx, val in data.items()]
            elif isinstance(data, dict):
                return data
            elif isinstance(data, (list, tuple)) and len(data) > 0 and isinstance(data[0], dict):
                return data
            return data

        # Handle different actions based on user selection
        if action == "fetch_data":
            """Fetch stock data from external API (Yahoo Finance)"""
            fetch_func = fetch_map.get(data_type)
            if fetch_func:
                try:
                    data = fetch_func()
                    stock.data[data_type] = data
                    
                    # Check if data was successfully retrieved
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
            """Save fetched stock data to local files"""
            save_func = save_map.get(data_type)
            data_to_save = stock.data.get(data_type)
            
            # If data not in memory, try to fetch it first
            if data_to_save is None:
                fetch_func = fetch_map.get(data_type)
                if fetch_func:
                    try:
                        data_to_save = fetch_func()
                        stock.data[data_type] = data_to_save
                    except Exception as e:
                        flash(f"Error fetching {data_type.replace('_', ' ')} for saving: {str(e)}", "danger")
                        return render_template("quote.html", stock={"ticker": ticker}, selected_data_type=data_type, data=None)

            # Check if there's data to save
            if data_to_save is None or (hasattr(data_to_save, 'empty') and data_to_save.empty):
                flash(f"No {data_type.replace('_', ' ')} data to save for {ticker}.", "warning")
                fetched_data = None
            else:
                if save_func:
                    try:
                        save_func()
                        # Update database if saving info data
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
            """Load previously saved stock data from local files"""
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
            # Invalid action provided
            flash("Invalid action.", "danger")

        # Render the quote template with processed data
        response = render_template(
            "quote.html",
            stock={"ticker": ticker},
            selected_data_type=data_type,
            data=fetched_data
        )
        get_flashed_messages()  # Clear flash messages from session
        return response

    # GET request - show the quote form
    return render_template("quote.html", stock=None, selected_data_type=data_type)

# User registration route
@app.route("/register", methods=["GET", "POST"])
def register():
    """Handle new user registration with email and password validation"""
    if request.method == "POST":
        # Get form data
        email = request.form.get("email")
        password = request.form.get("password")
        confirmation = request.form.get("confirmation")
        
        # Validate all required fields
        if not email:
            return apology("must provide email", 400)
        if not password:
            return apology("must provide password", 400)
        if not confirmation:
            return apology("must confirm password", 400)
        if password != confirmation:
            return apology("password and confirmation don't match", 400)
            
        # Check if email already exists
        rows = db.execute("SELECT * FROM users WHERE email = ?", email)
        if len(rows) != 0:
            return apology("email already registered", 400)
            
        # Create new user account
        hashed = generate_password_hash(password, method='pbkdf2')  # Hash password for security
        db.execute("INSERT INTO users (email, hash) VALUES (?, ?)", email, hashed)
        
        # Get the new user's ID and create session
        user_id = db.execute("SELECT user_id FROM users WHERE email = ?", email)[0]["user_id"]
        session["user_id"] = user_id
        return redirect("/")
    else:
        # GET request - show registration form
        return render_template("register.html")

# Application entry point
if __name__ == "__main__":
    # Run the Flask application in debug mode for development
    app.run(debug=True)