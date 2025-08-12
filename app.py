# Flask web application for stock data analysis and management
# This application provides a web interface for fetching, storing, and analyzing stock market data

# Import required libraries and modules
from cs50 import SQL
from flask import Flask, flash, redirect, render_template, request, session, get_flashed_messages
from flask_session import Session
from werkzeug.security import check_password_hash, generate_password_hash
from stock import Stock  # Custom Stock class for data fetching and management
from stock_analysis import StockAnalysis
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
            return render_template("quote.html", stock=None, selected_data_type=data_type, data=None, analysis=None)

        # Validate data type
        if data_type not in supported_data_types:
            flash(f"Unsupported data type: {data_type}", "danger")
            return render_template("quote.html", stock=None, selected_data_type=data_type, data=None, analysis=None)

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
                        return render_template("quote.html", stock={"ticker": ticker}, selected_data_type=data_type, data=None, analysis=None)

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

        elif action == "analyze_stock":
            """Perform comprehensive stock analysis using StockAnalysis class"""
            try:
                # Create StockAnalysis instance
                stock_analyzer = StockAnalysis(ticker)
                
                # Try to load existing data first, then fetch if needed
                data_methods = [
                    'load_info_from_file', 'load_analysis_from_file', 'load_balance_sheet_from_file',
                    'load_cash_flows_from_file', 'load_dividends_from_file', 'load_earnings_from_file',
                    'load_financials_from_file', 'load_history_from_file'
                ]
                
                data_loaded = False
                for method in data_methods:
                    try:
                        result = getattr(stock_analyzer, method)()
                        if result is not None:
                            data_loaded = True
                    except Exception:
                        continue
                
                # If no data loaded from files, fetch from API
                if not data_loaded:
                    try:
                        stock_analyzer.fetch_data()
                        stock_analyzer.fetch_history()
                        stock_analyzer.fetch_cash_flows()
                        stock_analyzer.fetch_dividends()
                        stock_analyzer.fetch_analysis()
                        data_loaded = True
                    except Exception as e:
                        flash(f"Error fetching data for analysis: {str(e)}", "danger")
                        return render_template("quote.html", stock={"ticker": ticker}, selected_data_type=data_type, data=None, analysis=None)
                
                # Verify we have minimum required data
                info = stock_analyzer.data.get('info', {})
                current_price = info.get('currentPrice', 0)
                
                if not current_price:
                    flash(f"Unable to get current price for {ticker}. Please try fetching basic info first.", "warning")
                    return render_template("quote.html", stock={"ticker": ticker}, selected_data_type=data_type, data=None, analysis=None)
                
                # Perform comprehensive analysis
                analysis_results = {}
                
                # Get basic info and technical data
                ma_50 = info.get('fiftyDayAverage', 0)
                ma_200 = info.get('twoHundredDayAverage', 0)
                
                # Helper function to format currency
                def format_currency(value):
                    if value and value > 0 and isinstance(value, (int, float)) and value == value and abs(value) != float('inf'):
                        return f"${value:,.2f}"
                    return "N/A"
                
                # Helper function to format large numbers
                def format_large_number(value):
                    if value and value > 0 and isinstance(value, (int, float)) and value == value and abs(value) != float('inf'):
                        return f"${value:,.0f}"
                    return "N/A"
                
                # Helper function to format percentage
                def format_percentage(value):
                    if value is not None and isinstance(value, (int, float)) and value == value and abs(value) != float('inf'):
                        return f"{value:+.1f}%"
                    return "N/A"
                
                analysis_results['basic_info'] = {
                    'ticker': ticker,
                    'current_price': current_price,
                    'current_price_formatted': format_currency(current_price),
                    'ma_50': ma_50,
                    'ma_50_formatted': format_currency(ma_50),
                    'ma_200': ma_200,
                    'ma_200_formatted': format_currency(ma_200),
                    'sector': info.get('sector', 'N/A'),
                    'industry': info.get('industry', 'N/A'),
                    'market_cap': info.get('marketCap', 0),
                    'market_cap_formatted': format_large_number(info.get('marketCap', 0))
                }
                
                # Determine if hypergrowth and run appropriate analysis
                is_hypergrowth = stock_analyzer._identify_hypergrowth_company()
                
                # Run valuations with error handling
                valuations = []
                dcf_value = None
                analysis_type = 'Traditional DCF'
                
                def calculate_safe_upside(value, current_price):
                    """Calculate upside percentage with proper error handling"""
                    if not value or not current_price or current_price <= 0:
                        return None
                    try:
                        upside = ((value - current_price) / current_price) * 100
                        # Check for infinite or NaN values
                        if not (isinstance(upside, (int, float)) and upside == upside and abs(upside) != float('inf')):
                            return None
                        # Also check for extremely large values that might cause formatting issues
                        if abs(upside) > 1000000:  # More than 1 million percent seems unreasonable
                            return None
                        return round(upside, 2)  # Round to 2 decimal places
                    except Exception as e:
                        print(f"Error calculating upside: {e}")
                        return None
                
                try:
                    if is_hypergrowth:
                        stock_analyzer.evaluate_hypergrowth_stock()
                        dcf_value = stock_analyzer._last_results.get('hypergrowth_valuation')
                        analysis_type = 'Hypergrowth'
                    else:
                        dcf_value = stock_analyzer.evaluate_DCF()
                        analysis_type = 'Traditional DCF'
                    
                    if dcf_value:
                        upside = calculate_safe_upside(dcf_value, current_price)
                        print(f"DCF: value={dcf_value}, current_price={current_price}, upside={upside}")
                        valuations.append({
                            'method': analysis_type, 
                            'value': dcf_value,
                            'value_formatted': format_currency(dcf_value),
                            'upside': upside,
                            'upside_formatted': format_percentage(upside),
                            'upside_class': 'text-success' if upside and upside > 0 else 'text-danger' if upside and upside < 0 else 'text-muted'
                        })
                except Exception as e:
                    print(f"DCF analysis error: {e}")
                
                # Run P/E analysis
                try:
                    pe_result = stock_analyzer.evaluate_PE()
                    if pe_result and pe_result.get('fair_value_justified'):
                        pe_value = pe_result['fair_value_justified']
                        upside = calculate_safe_upside(pe_value, current_price)
                        print(f"P/E: value={pe_value}, current_price={current_price}, upside={upside}")
                        valuations.append({
                            'method': 'P/E Analysis', 
                            'value': pe_value,
                            'value_formatted': format_currency(pe_value),
                            'upside': upside,
                            'upside_formatted': format_percentage(upside),
                            'upside_class': 'text-success' if upside and upside > 0 else 'text-danger' if upside and upside < 0 else 'text-muted'
                        })
                except Exception as e:
                    print(f"P/E analysis error: {e}")
                
                # Run DDM analysis
                try:
                    ddm_value = stock_analyzer.evaluate_DDM()
                    if ddm_value:
                        upside = calculate_safe_upside(ddm_value, current_price)
                        print(f"DDM: value={ddm_value}, current_price={current_price}, upside={upside}")
                        valuations.append({
                            'method': 'Dividend Model', 
                            'value': ddm_value,
                            'value_formatted': format_currency(ddm_value),
                            'upside': upside,
                            'upside_formatted': format_percentage(upside),
                            'upside_class': 'text-success' if upside and upside > 0 else 'text-danger' if upside and upside < 0 else 'text-muted'
                        })
                except Exception as e:
                    print(f"DDM analysis error: {e}")
                
                # Get analyst recommendations
                analyst_consensus = None
                try:
                    analyst_consensus = stock_analyzer.parse_analyst_recommendations()
                except Exception as e:
                    print(f"Analyst analysis error: {e}")
                
                # Calculate moving averages and trends
                ma_result = None
                try:
                    stock_analyzer.calculate_moving_averages()
                    ma_result = stock_analyzer.get_last_moving_averages()
                except Exception as e:
                    print(f"Technical analysis error: {e}")
                
                # Calculate average valuation and recommendation
                if valuations and current_price and current_price > 0:
                    avg_valuation = sum(v['value'] for v in valuations) / len(valuations)
                    upside_pct = calculate_safe_upside(avg_valuation, current_price)
                    
                    if upside_pct and upside_pct > 15:
                        our_recommendation = 'STRONG BUY'
                        rec_class = 'success'
                    elif upside_pct and upside_pct > 5:
                        our_recommendation = 'BUY'
                        rec_class = 'success'
                    elif upside_pct and upside_pct > -5:
                        our_recommendation = 'HOLD'
                        rec_class = 'warning'
                    elif upside_pct and upside_pct > -15:
                        our_recommendation = 'SELL'
                        rec_class = 'danger'
                    else:
                        our_recommendation = 'STRONG SELL'
                        rec_class = 'danger'
                else:
                    avg_valuation = sum(v['value'] for v in valuations) / len(valuations) if valuations else 0
                    upside_pct = None  # Set to None when we can't calculate
                    our_recommendation = 'INSUFFICIENT DATA'
                    rec_class = 'secondary'
                
                analysis_results['valuations'] = valuations
                analysis_results['average_valuation'] = avg_valuation
                analysis_results['average_valuation_formatted'] = format_currency(avg_valuation)
                analysis_results['upside_percent'] = upside_pct
                analysis_results['upside_percent_formatted'] = format_percentage(upside_pct)
                analysis_results['upside_class'] = 'text-success' if upside_pct and upside_pct > 0 else 'text-danger' if upside_pct and upside_pct < 0 else 'text-muted'
                analysis_results['our_recommendation'] = our_recommendation
                analysis_results['recommendation_class'] = rec_class
                
                # Process analyst consensus with formatting
                if analyst_consensus:
                    analysis_results['analyst_consensus'] = {
                        'consensus': analyst_consensus.get('consensus', 'N/A'),
                        'bullish_pct': analyst_consensus.get('bullish_pct', 0),
                        'neutral_pct': analyst_consensus.get('neutral_pct', 0),
                        'bearish_pct': analyst_consensus.get('bearish_pct', 0),
                        'sentiment_score': analyst_consensus.get('sentiment_score', 0),
                        'bullish_formatted': f"{analyst_consensus.get('bullish_pct', 0):.1f}%",
                        'neutral_formatted': f"{analyst_consensus.get('neutral_pct', 0):.1f}%",
                        'bearish_formatted': f"{analyst_consensus.get('bearish_pct', 0):.1f}%",
                        'sentiment_formatted': f"{analyst_consensus.get('sentiment_score', 0):+.2f}"
                    }
                else:
                    analysis_results['analyst_consensus'] = None
                
                # Process technical analysis with formatting
                if ma_result:
                    analysis_results['technical_analysis'] = {
                        'trend_50': ma_result.get('trend_50', 'N/A'),
                        'trend_200': ma_result.get('trend_200', 'N/A'),
                        'overall_trend': ma_result.get('overall_trend', 'N/A'),
                        'ma_50_pct': ma_result.get('ma_50_pct', 0),
                        'ma_200_pct': ma_result.get('ma_200_pct', 0),
                        'ma_50_formatted': format_percentage(ma_result.get('ma_50_pct', 0)),
                        'ma_200_formatted': format_percentage(ma_result.get('ma_200_pct', 0)),
                        'ma_50_class': 'text-success' if ma_result.get('trend_50') == 'Above' else 'text-danger',
                        'ma_200_class': 'text-success' if ma_result.get('trend_200') == 'Above' else 'text-danger',
                        'overall_class': 'text-success' if 'Uptrend' in str(ma_result.get('overall_trend', '')) else 'text-danger' if 'Downtrend' in str(ma_result.get('overall_trend', '')) else 'text-warning'
                    }
                else:
                    analysis_results['technical_analysis'] = None
                
                analysis_results['is_hypergrowth'] = is_hypergrowth
                
                # Debug print to see what we're sending
                print(f"Analysis results for {ticker}:")
                print(f"  Basic info: {analysis_results['basic_info']}")
                print(f"  Valuations: {len(valuations)} methods")
                for i, val in enumerate(valuations):
                    print(f"    {i+1}. {val['method']}: value={val['value']}, upside={val['upside']}")
                print(f"  Recommendation: {our_recommendation}")
                print(f"  Analyst consensus: {analyst_consensus is not None}")
                print(f"  Technical analysis: {ma_result is not None}")
                print(f"  About to render template with analysis data")
                
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