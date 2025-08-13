# Flask web application for stock data analysis and management
# This application provides a web interface for fetching, storing, and analyzing stock market data

# Import required libraries and modules
from flask import Flask, flash, redirect, render_template, request, get_flashed_messages
from helpers import usd  # Helper functions for the web app
from analysis_service import StockDataService, ComprehensiveAnalysisService
import os

# Initialize Flask application
app = Flask(__name__)

# Configure application settings
app.secret_key = "d7e9a6f8c4b12e03a59f7d8c6e4b1a0f3c29d5b7e8f1a6c4d7b9e3f0a2c1d8e7"  # Secret key for flash messages
app.jinja_env.filters["usd"] = usd  # Add USD formatting filter to Jinja templates

# After request handler to prevent caching of sensitive data
@app.after_request
def after_request(response):
    """Ensure responses aren't cached for security purposes"""
    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    response.headers["Expires"] = 0
    response.headers["Pragma"] = "no-cache"
    return response

# Home page route - redirects to stock analysis
@app.route("/")
def index():
    """Display the main dashboard page"""
    return redirect("/quote")

# Stock data management route
@app.route("/quote", methods=["GET", "POST"])
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
                # Extract user-defined parameters from form
                user_params = {
                    'dcf_params': {
                        'years': int(request.form.get('dcf_years', 12)),
                        'fade_years': int(request.form.get('dcf_fade_years', 8)),
                        'terminal_g': float(request.form.get('dcf_terminal_g', 4.0)) / 100,
                        'forward_uplift': float(request.form.get('dcf_forward_uplift', 10.0)) / 100
                    },
                    'ddm_params': {
                        'growth_rate': float(request.form.get('ddm_growth_rate', 5.0)) / 100,
                        'discount_rate': float(request.form.get('ddm_discount_rate', 7.0)) / 100,
                        'projection_years': int(request.form.get('ddm_projection_years', 5))
                    },
                    'pe_params': {
                        'use_forward': 'pe_use_forward' in request.form,
                        'use_sector_premium': 'pe_use_sector_premium' in request.form
                    },
                    'general_params': {
                        'discount_rate': float(request.form.get('general_discount_rate', 8.0)) / 100,
                        'terminal_growth': float(request.form.get('general_terminal_growth', 2.0)) / 100,
                        'use_enhanced_models': 'use_enhanced_models' in request.form
                    }
                }
                
                analysis_service = ComprehensiveAnalysisService(ticker)
                analysis_results = analysis_service.analyze_stock(user_params)
                
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
        get_flashed_messages()  # Clear flash messages
        return response

    # GET request - show the quote form
    return render_template("quote.html", stock=None, selected_data_type=data_type, analysis=None)

# About page route - provides information about the project
@app.route("/about")
def about():
    """Display the about page with project information"""
    return render_template("about.html")

# Application entry point
if __name__ == "__main__":
    # Run the Flask application in debug mode for development
    app.run(debug=True)