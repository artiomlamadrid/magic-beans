# Flask web application for stock data analysis and management
# This application provides a web interface for fetching, storing, and analyzing stock market data

# Import required libraries and modules
from cs50 import SQL
from flask import Flask, flash, redirect, render_template, request, session, get_flashed_messages
from flask_session import Session
from werkzeug.security import check_password_hash, generate_password_hash
from helpers import apology, login_required, usd  # Helper functions for the web app
from analysis_service import StockDataService, ComprehensiveAnalysisService
import os

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
            return render_template("quote.html", stock=None, selected_data_type=data_type, data=None, analysis=None)

        # Validate data type
        if data_type not in supported_data_types:
            flash(f"Unsupported data type: {data_type}", "danger")
            return render_template("quote.html", stock=None, selected_data_type=data_type, data=None, analysis=None)

        # Manage session data for the current ticker
        if "fetched" not in session or session["fetched"].get("ticker") != ticker:
            session["fetched"] = {"ticker": ticker}
        session.modified = True

        # Create data service for the requested ticker
        try:
            data_service = StockDataService(ticker)
        except Exception as e:
            flash(f"Error initializing data service for {ticker}: {str(e)}", "danger")
            return render_template("quote.html", stock=None, selected_data_type=data_type, data=None, analysis=None)

        # Handle different actions based on user selection
        if action == "fetch_data":
            """Fetch stock data from external API (Yahoo Finance)"""
            try:
                fetched_data, message = data_service.fetch_data(data_type)
                if fetched_data:
                    flash(message, "success")
                    session["fetched"][data_type] = True
                    session.modified = True
                else:
                    flash(message, "warning")
            except Exception as e:
                flash(f"Error fetching {data_type.replace('_', ' ')} data for {ticker}: {str(e)}", "danger")

        elif action == "save_data":
            """Save fetched stock data to local files"""
            try:
                fetched_data, message = data_service.save_data(data_type)
                if fetched_data:
                    flash(message, "success")
                else:
                    flash(message, "warning")
            except Exception as e:
                flash(f"Failed to save {data_type.replace('_', ' ')} data for {ticker}: {str(e)}", "danger")

        elif action == "load_data":
            """Load previously saved stock data from local files"""
            try:
                fetched_data, message = data_service.load_data(data_type)
                if fetched_data:
                    flash(message, "success")
                else:
                    flash(message, "warning")
            except Exception as e:
                flash(f"Error loading {data_type.replace('_', ' ')} data for {ticker}: {str(e)}", "danger")

        elif action == "analyze_stock":
            """Perform comprehensive stock analysis using ComprehensiveAnalysisService"""
            try:
                analysis_service = ComprehensiveAnalysisService(ticker)
                analysis_results = analysis_service.analyze_stock()
                
                flash(f"Analysis completed for {ticker}.", "success")
                
                # Render template with analysis results
                return render_template(
                    "quote.html",
                    stock={"ticker": ticker},
                    selected_data_type=data_type,
                    data=fetched_data,
                    analysis=analysis_results
                )
                
            except Exception as e:
                flash(f"Error performing analysis for {ticker}: {str(e)}", "danger")
                import traceback
                print(traceback.format_exc())  # For debugging
                return render_template("quote.html", stock={"ticker": ticker}, selected_data_type=data_type, data=None, analysis=None)
        else:
            # Invalid action provided
            flash("Invalid action.", "danger")

        # Render the quote template with processed data
        response = render_template(
            "quote.html",
            stock={"ticker": ticker},
            selected_data_type=data_type,
            data=fetched_data,
            analysis=None
        )
        get_flashed_messages()  # Clear flash messages from session
        return response

    # GET request - show the quote form
    return render_template("quote.html", stock=None, selected_data_type=data_type, analysis=None)

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