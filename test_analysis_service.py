# Unit tests for Stock Analysis Web Application Services
# Tests the refactored business logic classes for comprehensive coverage

import pytest
import sys
import os
from unittest.mock import Mock, patch, MagicMock

# Add the project directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from analysis_service import (
    DataFormatter, 
    DataConverter, 
    StockDataService, 
    ComprehensiveAnalysisService
)


class TestDataFormatter:
    """Test suite for DataFormatter utility class"""
    
    def test_format_currency_valid_values(self):
        """Test currency formatting with valid values"""
        assert DataFormatter.format_currency(100.50) == "$100.50"
        assert DataFormatter.format_currency(1000) == "$1,000.00"
        assert DataFormatter.format_currency(1234567.89) == "$1,234,567.89"
    
    def test_format_currency_invalid_values(self):
        """Test currency formatting with invalid values"""
        assert DataFormatter.format_currency(None) == "N/A"
        assert DataFormatter.format_currency(0) == "N/A"
        assert DataFormatter.format_currency(-100) == "N/A"
        assert DataFormatter.format_currency(float('inf')) == "N/A"
        assert DataFormatter.format_currency(float('nan')) == "N/A"
        assert DataFormatter.format_currency("invalid") == "N/A"
    
    def test_format_large_number_valid_values(self):
        """Test large number formatting with valid values"""
        assert DataFormatter.format_large_number(1000000) == "$1,000,000"
        assert DataFormatter.format_large_number(2500000000) == "$2,500,000,000"
    
    def test_format_large_number_invalid_values(self):
        """Test large number formatting with invalid values"""
        assert DataFormatter.format_large_number(None) == "N/A"
        assert DataFormatter.format_large_number(0) == "N/A"
        assert DataFormatter.format_large_number(-1000000) == "N/A"
    
    def test_format_percentage_valid_values(self):
        """Test percentage formatting with valid values"""
        assert DataFormatter.format_percentage(15.5) == "+15.5%"
        assert DataFormatter.format_percentage(-10.2) == "-10.2%"
        assert DataFormatter.format_percentage(0) == "+0.0%"
    
    def test_format_percentage_invalid_values(self):
        """Test percentage formatting with invalid values"""
        assert DataFormatter.format_percentage(None) == "N/A"
        assert DataFormatter.format_percentage(float('inf')) == "N/A"
        assert DataFormatter.format_percentage(float('nan')) == "N/A"
    
    def test_calculate_safe_upside_valid_values(self):
        """Test upside calculation with valid values"""
        # 50% upside
        result = DataFormatter.calculate_safe_upside(150, 100)
        assert result == 50.0
        
        # -20% downside
        result = DataFormatter.calculate_safe_upside(80, 100)
        assert result == -20.0
        
        # 0% (equal values)
        result = DataFormatter.calculate_safe_upside(100, 100)
        assert result == 0.0
    
    def test_calculate_safe_upside_invalid_values(self):
        """Test upside calculation with invalid values"""
        assert DataFormatter.calculate_safe_upside(None, 100) is None
        assert DataFormatter.calculate_safe_upside(100, None) is None
        assert DataFormatter.calculate_safe_upside(100, 0) is None
        assert DataFormatter.calculate_safe_upside(100, -50) is None
        
        # Test division by very small number (could cause overflow)
        result = DataFormatter.calculate_safe_upside(100, 0.0001)
        # Should return None for extremely large values
        assert result is None


class TestDataConverter:
    """Test suite for DataConverter class"""
    
    def test_to_records_with_none(self):
        """Test data conversion with None input"""
        assert DataConverter.to_records(None) is None
    
    def test_to_records_with_dict(self):
        """Test data conversion with dictionary input"""
        test_dict = {"key1": "value1", "key2": "value2"}
        result = DataConverter.to_records(test_dict)
        assert result == test_dict
    
    def test_to_records_with_list_of_dicts(self):
        """Test data conversion with list of dictionaries"""
        test_list = [{"date": "2023-01-01", "price": 100}, {"date": "2023-01-02", "price": 105}]
        result = DataConverter.to_records(test_list)
        assert result == test_list
    
    @patch('analysis_service.pd.DataFrame')
    def test_to_records_with_dataframe(self, mock_df_class):
        """Test data conversion with pandas DataFrame"""
        # Create a mock DataFrame instance
        mock_df_instance = Mock()
        mock_df_instance.copy.return_value = mock_df_instance
        mock_df_instance.index.name = "Date"
        mock_df_instance.columns = ["Price", "Volume"]
        mock_df_instance.reset_index.return_value = mock_df_instance
        mock_df_instance.to_dict.return_value = [{"Date": "2023-01-01", "Price": 100}]
        
        # Test with the mock instance (simulating a real DataFrame)
        with patch('analysis_service.pd') as mock_pd:
            mock_pd.DataFrame = type(mock_df_instance)  # Set the type correctly
            result = DataConverter.to_records(mock_df_instance)
            mock_df_instance.copy.assert_called_once()
            mock_df_instance.to_dict.assert_called_once_with(orient="records")


class TestStockDataService:
    """Test suite for StockDataService class"""
    
    @patch('analysis_service.Stock')
    def test_init(self, mock_stock):
        """Test StockDataService initialization"""
        service = StockDataService("AAPL")
        assert service.ticker == "AAPL"
        mock_stock.assert_called_once_with("AAPL")
    
    @patch('analysis_service.Stock')
    def test_fetch_data_success(self, mock_stock):
        """Test successful data fetching"""
        # Mock the Stock instance and its methods
        mock_stock_instance = Mock()
        mock_stock_instance.fetch_data.return_value = {"currentPrice": 150}
        mock_stock_instance.data = {}
        mock_stock.return_value = mock_stock_instance
        
        service = StockDataService("AAPL")
        
        # Mock the converter
        service.converter.to_records = Mock(return_value={"currentPrice": 150})
        
        data, message = service.fetch_data("info")
        
        assert data == {"currentPrice": 150}
        assert "Successfully fetched" in message
        mock_stock_instance.fetch_data.assert_called_once()
    
    @patch('analysis_service.Stock')
    def test_fetch_data_no_data(self, mock_stock):
        """Test data fetching when no data is available"""
        # Mock the Stock instance returning None
        mock_stock_instance = Mock()
        mock_stock_instance.fetch_data.return_value = None
        mock_stock_instance.data = {}
        mock_stock.return_value = mock_stock_instance
        
        service = StockDataService("AAPL")
        
        data, message = service.fetch_data("info")
        
        assert data is None
        assert "No info data found" in message
    
    @patch('analysis_service.Stock')
    def test_fetch_data_invalid_type(self, mock_stock):
        """Test data fetching with invalid data type"""
        service = StockDataService("AAPL")
        
        with pytest.raises(ValueError, match="not supported"):
            service.fetch_data("invalid_type")


class TestComprehensiveAnalysisService:
    """Test suite for ComprehensiveAnalysisService class"""
    
    @patch('analysis_service.StockAnalysis')
    def test_init(self, mock_stock_analysis):
        """Test ComprehensiveAnalysisService initialization"""
        service = ComprehensiveAnalysisService("AAPL")
        assert service.ticker == "AAPL"
        assert service.analysis_results == {}
    
    @patch('analysis_service.StockAnalysis')
    def test_analyze_stock_success(self, mock_stock_analysis):
        """Test successful stock analysis"""
        # Mock StockAnalysis instance
        mock_analyzer = Mock()
        mock_analyzer.data = {
            'info': {
                'currentPrice': 150.0,
                'fiftyDayAverage': 145.0,
                'twoHundredDayAverage': 140.0,
                'sector': 'Technology',
                'industry': 'Consumer Electronics',
                'marketCap': 2500000000000
            }
        }
        mock_analyzer._identify_hypergrowth_company.return_value = False
        mock_analyzer.evaluate_DCF.return_value = 160.0
        mock_analyzer.evaluate_DDM.return_value = 155.0
        mock_analyzer.evaluate_PE.return_value = 145.0
        mock_analyzer._last_results = {}
        mock_analyzer.parse_analyst_recommendations.return_value = None
        mock_analyzer.calculate_moving_averages.return_value = None
        mock_analyzer.get_last_moving_averages.return_value = None
        
        mock_stock_analysis.return_value = mock_analyzer
        
        service = ComprehensiveAnalysisService("AAPL")
        
        # Mock the data loading method
        service._load_or_fetch_data = Mock(return_value=True)
        
        result = service.analyze_stock()
        
        # Verify basic structure
        assert 'basic_info' in result
        assert 'valuations' in result
        assert 'our_recommendation' in result
        assert result['basic_info']['ticker'] == "AAPL"
        assert result['basic_info']['current_price'] == 150.0
    
    @patch('analysis_service.StockAnalysis')
    def test_analyze_stock_no_current_price(self, mock_stock_analysis):
        """Test analysis when no current price is available"""
        # Mock StockAnalysis instance with no current price
        mock_analyzer = Mock()
        mock_analyzer.data = {'info': {}}
        mock_stock_analysis.return_value = mock_analyzer
        
        service = ComprehensiveAnalysisService("AAPL")
        service._load_or_fetch_data = Mock(return_value=True)
        
        with pytest.raises(ValueError, match="Unable to get current price"):
            service.analyze_stock()
    
    @patch('analysis_service.StockAnalysis')
    def test_analyze_stock_data_load_failure(self, mock_stock_analysis):
        """Test analysis when data loading fails"""
        mock_analyzer = Mock()
        mock_stock_analysis.return_value = mock_analyzer
        
        service = ComprehensiveAnalysisService("AAPL")
        service._load_or_fetch_data = Mock(return_value=False)
        
        with pytest.raises(ValueError, match="Unable to load or fetch required data"):
            service.analyze_stock()
    
    def test_analyze_basic_info(self):
        """Test basic info analysis"""
        service = ComprehensiveAnalysisService("AAPL")
        
        info = {
            'currentPrice': 150.0,
            'fiftyDayAverage': 145.0,
            'twoHundredDayAverage': 140.0,
            'sector': 'Technology',
            'industry': 'Consumer Electronics',
            'marketCap': 2500000000000
        }
        
        service._analyze_basic_info(info, 150.0)
        
        basic_info = service.analysis_results['basic_info']
        assert basic_info['ticker'] == "AAPL"
        assert basic_info['current_price'] == 150.0
        assert basic_info['current_price_formatted'] == "$150.00"
        assert basic_info['sector'] == 'Technology'
        assert basic_info['industry'] == 'Consumer Electronics'
    
    def test_analyze_recommendations_strong_buy(self):
        """Test recommendation analysis for strong buy scenario"""
        service = ComprehensiveAnalysisService("AAPL")
        
        # Set up analysis results with valuations showing >15% upside
        service.analysis_results = {
            'valuations': [
                {'value': 175.0},  # 16.67% upside from 150
            ],
            'basic_info': {'current_price': 150.0}
        }
        
        service._analyze_recommendations()
        
        assert service.analysis_results['our_recommendation'] == 'STRONG BUY'
        assert service.analysis_results['recommendation_class'] == 'success'
    
    def test_analyze_recommendations_insufficient_data(self):
        """Test recommendation analysis with insufficient data"""
        service = ComprehensiveAnalysisService("AAPL")
        
        # Set up analysis results with no valuations
        service.analysis_results = {
            'valuations': [],
            'basic_info': {'current_price': 150.0}
        }
        
        service._analyze_recommendations()
        
        assert service.analysis_results['our_recommendation'] == 'INSUFFICIENT DATA'
        assert service.analysis_results['recommendation_class'] == 'secondary'


# Integration test examples
class TestIntegration:
    """Integration tests for the service classes working together"""
    
    @patch('analysis_service.StockAnalysis')
    @patch('analysis_service.Stock')
    def test_end_to_end_analysis_flow(self, mock_stock, mock_stock_analysis):
        """Test complete analysis flow from data service to analysis service"""
        # This would test the full workflow but requires more complex mocking
        # For now, we'll keep it as a placeholder for future integration tests
        pass


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v"])
