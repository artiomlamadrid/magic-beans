# Analysis Service Classes for Stock Analysis Web Application
# This module contains business logic classes that can be easily tested

import pandas as pd
from stock import Stock
from stock_analysis import StockAnalysis


class DataFormatter:
    """Utility class for formatting financial data consistently"""
    
    @staticmethod
    def format_currency(value):
        """Format value as currency with proper validation"""
        if not isinstance(value, (int, float)) or not value or value <= 0 or value != value or abs(value) == float('inf'):
            return "N/A"
        return f"${value:,.2f}"
    
    @staticmethod
    def format_large_number(value):
        """Format large numbers (like market cap) with proper validation"""
        if not isinstance(value, (int, float)) or not value or value <= 0 or value != value or abs(value) == float('inf'):
            return "N/A"
        return f"${value:,.0f}"
    
    @staticmethod
    def format_percentage(value):
        """Format percentage values with proper validation"""
        if value is not None and isinstance(value, (int, float)) and value == value and abs(value) != float('inf'):
            return f"{value:+.1f}%"
        return "N/A"
    
    @staticmethod
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


class DataConverter:
    """Class for converting pandas data structures to web-friendly formats"""
    
    @staticmethod
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


class StockDataService:
    """Service class for handling stock data operations (fetch, save, load)"""
    
    def __init__(self, ticker):
        self.ticker = ticker
        self.stock = Stock(ticker)
        self.converter = DataConverter()
        
        # Define mappings for data operations
        self.fetch_map = {
            "analysis": self.stock.fetch_analysis,
            "balance_sheet": self.stock.fetch_balance_sheet,
            "cashflow": self.stock.fetch_cash_flows,
            "dividends": self.stock.fetch_dividends,
            "earnings": self.stock.fetch_earnings,
            "financials": self.stock.fetch_financials,
            "history": self.stock.fetch_history,
            "info": self.stock.fetch_data,
            "institutional_holders": self.stock.fetch_institutional_holders,
            "major_holders": self.stock.fetch_major_holders,
            "options": self.stock.fetch_options,
            "splits": self.stock.fetch_splits,
            "sustainability": self.stock.fetch_sustainability,
        }
        
        self.save_map = {
            "analysis": self.stock.save_analysis_to_file,
            "balance_sheet": self.stock.save_balance_sheet_to_file,
            "cashflow": self.stock.save_cash_flows_to_file,
            "dividends": self.stock.save_dividends_to_file,
            "earnings": self.stock.save_earnings_to_file,
            "financials": self.stock.save_financials_to_file,
            "history": self.stock.save_history_to_file,
            "info": self.stock.save_info_to_file,
            "institutional_holders": self.stock.save_institutional_holders_to_file,
            "major_holders": self.stock.save_major_holders_to_file,
            "options": self.stock.save_options_to_file,
            "splits": self.stock.save_splits_to_file,
            "sustainability": self.stock.save_sustainability_to_file,
        }
        
        self.load_map = {
            "analysis": self.stock.load_analysis_from_file,
            "balance_sheet": self.stock.load_balance_sheet_from_file,
            "cashflow": self.stock.load_cash_flows_from_file,
            "dividends": self.stock.load_dividends_from_file,
            "earnings": self.stock.load_earnings_from_file,
            "financials": self.stock.load_financials_from_file,
            "history": self.stock.load_history_from_file,
            "info": self.stock.load_info_from_file,
            "institutional_holders": self.stock.load_institutional_holders_from_file,
            "major_holders": self.stock.load_major_holders_from_file,
            "options": self.stock.load_options_from_file,
            "splits": self.stock.load_splits_from_file,
            "sustainability": self.stock.load_sustainability_from_file,
        }
    
    def fetch_data(self, data_type):
        """Fetch stock data from external API"""
        fetch_func = self.fetch_map.get(data_type)
        if not fetch_func:
            raise ValueError(f"Fetching {data_type.replace('_', ' ')} not supported.")
        
        data = fetch_func()
        self.stock.data[data_type] = data
        
        if data is None or (hasattr(data, 'empty') and data.empty):
            return None, f"No {data_type.replace('_', ' ')} data found for {self.ticker}."
        
        fetched_data = self.converter.to_records(data)
        return fetched_data, f"Successfully fetched {data_type.replace('_', ' ')} data for {self.ticker}."
    
    def save_data(self, data_type):
        """Save fetched stock data to local files"""
        save_func = self.save_map.get(data_type)
        data_to_save = self.stock.data.get(data_type)
        
        # If data not in memory, try to fetch it first
        if data_to_save is None:
            fetch_func = self.fetch_map.get(data_type)
            if fetch_func:
                data_to_save = fetch_func()
                self.stock.data[data_type] = data_to_save
        
        if data_to_save is None or (hasattr(data_to_save, 'empty') and data_to_save.empty):
            return None, f"No {data_type.replace('_', ' ')} data to save for {self.ticker}."
        
        if not save_func:
            raise ValueError(f"Saving {data_type.replace('_', ' ')} data not implemented.")
        
        save_func()
        # Update database if saving info data
        if data_type == "info":
            self.stock.update_database_info()
        
        fetched_data = self.converter.to_records(data_to_save)
        return fetched_data, f"{data_type.replace('_', ' ').capitalize()} data saved for {self.ticker}."
    
    def load_data(self, data_type):
        """Load previously saved stock data from local files"""
        load_func = self.load_map.get(data_type)
        if not load_func:
            raise ValueError(f"Loading {data_type.replace('_', ' ')} data not supported.")
        
        loaded = load_func()
        if loaded is None or (hasattr(loaded, 'empty') and loaded.empty):
            return None, f"No saved {data_type.replace('_', ' ')} data found for {self.ticker}."
        
        self.stock.data[data_type] = loaded
        fetched_data = self.converter.to_records(loaded)
        return fetched_data, f"Loaded {data_type.replace('_', ' ')} data for {self.ticker}."


class ComprehensiveAnalysisService:
    """Service class for performing comprehensive stock analysis"""
    
    def __init__(self, ticker):
        self.ticker = ticker
        self.formatter = DataFormatter()
        self.stock_analyzer = None
        self.analysis_results = {}
    
    def analyze_stock(self):
        """Perform comprehensive stock analysis"""
        # Create StockAnalysis instance
        self.stock_analyzer = StockAnalysis(self.ticker)
        
        # Try to load existing data first, then fetch if needed
        if not self._load_or_fetch_data():
            raise ValueError("Unable to load or fetch required data for analysis")
        
        # Verify we have minimum required data
        info = self.stock_analyzer.data.get('info', {})
        current_price = info.get('currentPrice', 0)
        
        if not current_price:
            raise ValueError(f"Unable to get current price for {self.ticker}")
        
        # Perform comprehensive analysis
        self._analyze_basic_info(info, current_price)
        self._analyze_valuations(current_price)
        self._analyze_recommendations()
        self._analyze_consensus()
        self._analyze_technical()
        self._finalize_analysis()
        
        return self.analysis_results
    
    def _load_or_fetch_data(self):
        """Try to load existing data first, then fetch if needed"""
        data_methods = [
            'load_info_from_file', 'load_analysis_from_file', 'load_balance_sheet_from_file',
            'load_cash_flows_from_file', 'load_dividends_from_file', 'load_earnings_from_file',
            'load_financials_from_file', 'load_history_from_file'
        ]
        
        data_loaded = False
        for method in data_methods:
            try:
                result = getattr(self.stock_analyzer, method)()
                if result is not None:
                    data_loaded = True
            except Exception:
                continue
        
        # If no data loaded from files, fetch from API
        if not data_loaded:
            try:
                self.stock_analyzer.fetch_data()
                self.stock_analyzer.fetch_history()
                self.stock_analyzer.fetch_cash_flows()
                self.stock_analyzer.fetch_dividends()
                self.stock_analyzer.fetch_analysis()
                data_loaded = True
            except Exception as e:
                raise ValueError(f"Error fetching data for analysis: {str(e)}")
        
        return data_loaded
    
    def _analyze_basic_info(self, info, current_price):
        """Analyze and format basic stock information"""
        ma_50 = info.get('fiftyDayAverage', 0)
        ma_200 = info.get('twoHundredDayAverage', 0)
        
        self.analysis_results['basic_info'] = {
            'ticker': self.ticker,
            'current_price': current_price,
            'current_price_formatted': self.formatter.format_currency(current_price),
            'ma_50': ma_50,
            'ma_50_formatted': self.formatter.format_currency(ma_50),
            'ma_200': ma_200,
            'ma_200_formatted': self.formatter.format_currency(ma_200),
            'sector': info.get('sector', 'N/A'),
            'industry': info.get('industry', 'N/A'),
            'market_cap': info.get('marketCap', 0),
            'market_cap_formatted': self.formatter.format_large_number(info.get('marketCap', 0))
        }
    
    def _analyze_valuations(self, current_price):
        """Perform valuation analysis (DCF, P/E, DDM)"""
        valuations = []
        
        # Determine if hypergrowth and run appropriate analysis
        is_hypergrowth = self.stock_analyzer._identify_hypergrowth_company()
        self.analysis_results['is_hypergrowth'] = is_hypergrowth
        
        # DCF Analysis
        try:
            if is_hypergrowth:
                self.stock_analyzer.evaluate_hypergrowth_stock()
                dcf_value = self.stock_analyzer._last_results.get('hypergrowth_valuation')
                analysis_type = 'Hypergrowth'
            else:
                dcf_value = self.stock_analyzer.evaluate_DCF()
                analysis_type = 'Traditional DCF'
            
            if dcf_value:
                upside = self.formatter.calculate_safe_upside(dcf_value, current_price)
                print(f"DCF: value={dcf_value}, current_price={current_price}, upside={upside}")
                valuations.append({
                    'method': analysis_type,
                    'value': dcf_value,
                    'value_formatted': self.formatter.format_currency(dcf_value),
                    'upside': upside,
                    'upside_formatted': self.formatter.format_percentage(upside),
                    'upside_class': 'text-success' if upside and upside > 0 else 'text-danger' if upside and upside < 0 else 'text-muted'
                })
        except Exception as e:
            print(f"DCF analysis error: {e}")
        
        # P/E Analysis
        try:
            pe_result = self.stock_analyzer.evaluate_PE()
            if pe_result and pe_result.get('fair_value_justified'):
                pe_value = pe_result['fair_value_justified']
                upside = self.formatter.calculate_safe_upside(pe_value, current_price)
                print(f"P/E: value={pe_value}, current_price={current_price}, upside={upside}")
                valuations.append({
                    'method': 'P/E Analysis',
                    'value': pe_value,
                    'value_formatted': self.formatter.format_currency(pe_value),
                    'upside': upside,
                    'upside_formatted': self.formatter.format_percentage(upside),
                    'upside_class': 'text-success' if upside and upside > 0 else 'text-danger' if upside and upside < 0 else 'text-muted'
                })
        except Exception as e:
            print(f"P/E analysis error: {e}")
        
        # DDM Analysis
        try:
            ddm_value = self.stock_analyzer.evaluate_DDM()
            if ddm_value:
                upside = self.formatter.calculate_safe_upside(ddm_value, current_price)
                print(f"DDM: value={ddm_value}, current_price={current_price}, upside={upside}")
                valuations.append({
                    'method': 'Dividend Model',
                    'value': ddm_value,
                    'value_formatted': self.formatter.format_currency(ddm_value),
                    'upside': upside,
                    'upside_formatted': self.formatter.format_percentage(upside),
                    'upside_class': 'text-success' if upside and upside > 0 else 'text-danger' if upside and upside < 0 else 'text-muted'
                })
        except Exception as e:
            print(f"DDM analysis error: {e}")
        
        self.analysis_results['valuations'] = valuations
    
    def _analyze_recommendations(self):
        """Calculate average valuation and generate recommendation"""
        valuations = self.analysis_results['valuations']
        current_price = self.analysis_results['basic_info']['current_price']
        
        if valuations and current_price and current_price > 0:
            avg_valuation = sum(v['value'] for v in valuations) / len(valuations)
            upside_pct = self.formatter.calculate_safe_upside(avg_valuation, current_price)
            
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
            upside_pct = None
            our_recommendation = 'INSUFFICIENT DATA'
            rec_class = 'secondary'
        
        self.analysis_results.update({
            'average_valuation': avg_valuation,
            'average_valuation_formatted': self.formatter.format_currency(avg_valuation),
            'upside_percent': upside_pct,
            'upside_percent_formatted': self.formatter.format_percentage(upside_pct),
            'upside_class': 'text-success' if upside_pct and upside_pct > 0 else 'text-danger' if upside_pct and upside_pct < 0 else 'text-muted',
            'our_recommendation': our_recommendation,
            'recommendation_class': rec_class
        })
    
    def _analyze_consensus(self):
        """Analyze analyst consensus"""
        try:
            analyst_consensus = self.stock_analyzer.parse_analyst_recommendations()
            if analyst_consensus:
                self.analysis_results['analyst_consensus'] = {
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
                self.analysis_results['analyst_consensus'] = None
        except Exception as e:
            print(f"Analyst analysis error: {e}")
            self.analysis_results['analyst_consensus'] = None
    
    def _analyze_technical(self):
        """Analyze technical indicators"""
        try:
            self.stock_analyzer.calculate_moving_averages()
            ma_result = self.stock_analyzer.get_last_moving_averages()
            if ma_result:
                self.analysis_results['technical_analysis'] = {
                    'trend_50': ma_result.get('trend_50', 'N/A'),
                    'trend_200': ma_result.get('trend_200', 'N/A'),
                    'overall_trend': ma_result.get('overall_trend', 'N/A'),
                    'ma_50_pct': ma_result.get('ma_50_pct', 0),
                    'ma_200_pct': ma_result.get('ma_200_pct', 0),
                    'ma_50_formatted': self.formatter.format_percentage(ma_result.get('ma_50_pct', 0)),
                    'ma_200_formatted': self.formatter.format_percentage(ma_result.get('ma_200_pct', 0)),
                    'ma_50_class': 'text-success' if ma_result.get('trend_50') == 'Above' else 'text-danger',
                    'ma_200_class': 'text-success' if ma_result.get('trend_200') == 'Above' else 'text-danger',
                    'overall_class': 'text-success' if 'Uptrend' in str(ma_result.get('overall_trend', '')) else 'text-danger' if 'Downtrend' in str(ma_result.get('overall_trend', '')) else 'text-warning'
                }
            else:
                self.analysis_results['technical_analysis'] = None
        except Exception as e:
            print(f"Technical analysis error: {e}")
            self.analysis_results['technical_analysis'] = None
    
    def _finalize_analysis(self):
        """Add final debug information and summary"""
        valuations = self.analysis_results['valuations']
        our_recommendation = self.analysis_results['our_recommendation']
        analyst_consensus = self.analysis_results['analyst_consensus']
        ma_result = self.analysis_results['technical_analysis']
        
        # Debug print to see what we're sending
        print(f"Analysis results for {self.ticker}:")
        print(f"  Basic info: {self.analysis_results['basic_info']}")
        print(f"  Valuations: {len(valuations)} methods")
        for i, val in enumerate(valuations):
            print(f"    {i+1}. {val['method']}: value={val['value']}, upside={val['upside']}")
        print(f"  Recommendation: {our_recommendation}")
        print(f"  Analyst consensus: {analyst_consensus is not None}")
        print(f"  Technical analysis: {ma_result is not None}")
        print(f"  Analysis completed successfully")
