# Magic Beans - Stock Analysis Platform

A comprehensive web-based stock analysis platform built with Flask for educational and research purposes. This project was developed as a final project for CS50's Introduction to Computer Science course.

## Project Overview

Magic Beans provides sophisticated financial analysis tools with real-time data integration, multiple valuation models, and an intuitive web interface. The application demonstrates advanced programming concepts including object-oriented design, web development, API integration, and comprehensive testing methodologies.

## Features

### Core Functionality
- **Real-Time Data Fetching**: Direct integration with Yahoo Finance API
- **Multiple Valuation Models**: DCF, DDM, and P/E analysis with customizable parameters
- **Data Management**: Save and load functionality for offline analysis
- **Professional UI**: Bootstrap-based responsive design
- **Comprehensive Testing**: 153 unit tests covering all major functionality

### Analysis Capabilities
- **Discounted Cash Flow (DCF)**: Projects future cash flows with configurable parameters
- **Dividend Discount Model (DDM)**: Values companies based on dividend payments
- **Price-to-Earnings (P/E)**: Comparative analysis using earnings multiples
- **Hypergrowth Detection**: Algorithmic identification of high-growth companies
- **Risk Assessment**: Industry-specific discount rate calculations

## Getting Started

### Prerequisites
- Python 3.9 or higher
- pip (Python package installer)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/artiomlamadrid/magic-beans.git
   cd magic-beans
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the application**
   ```bash
   python app.py
   ```

5. **Access the application**
   Open your browser and navigate to `http://localhost:5000`

### Running Tests

```bash
# Install test dependencies
pip install -r test-requirements.txt

# Run all tests
pytest

# Run tests with coverage
pytest --cov=.

# Run specific test file
pytest test_stock.py
```

## Usage

1. **Enter Stock Ticker**: Input any valid stock ticker symbol (e.g., AAPL, MSFT, TSLA)
2. **Configure Analysis**: Adjust valuation parameters or use intelligent defaults
3. **Review Results**: Analyze comprehensive valuation results and investment insights

### Example Analysis Parameters

**DCF Model**:
- Projection Period: 5-20 years (default: 12)
- Fade Period: 3-15 years (default: 8)
- Terminal Growth: 0-10% (default: 4%)
- Discount Rate: Industry-specific calculation

**DDM Model**:
- Dividend Growth Rate: 0-20% (default: 4%)
- Required Return: 3-15% (default: 8%)
- Projection Period: 3-10 years (default: 6)

**P/E Analysis**:
- Forward vs. Trailing P/E selection
- Sector premium adjustments
- Peer comparison analysis

## Technical Architecture

### Backend Components
- **Flask Application** (`app.py`): Web framework and routing
- **Stock Class** (`stock.py`): Data fetching and management
- **StockAnalysis Class** (`stock_analysis.py`): Core valuation algorithms
- **Analysis Services** (`analysis_service.py`): Business logic layer
- **Helper Functions** (`helpers.py`): Utility functions

### Frontend Components
- **Bootstrap 5**: Responsive CSS framework
- **Jinja2 Templates**: Server-side rendering
- **Interactive Forms**: Parameter configuration
- **Font Awesome**: Icons and visual elements

### Data Management
- **Save Function**: JSON storage for offline access
- **Load Function**: Retrieve previously saved data
- **Export Function**: CSV format for external analysis

## Testing

The project includes comprehensive testing with 153 unit tests covering:

- **Data Fetching**: API integration and error handling
- **Valuation Models**: Mathematical calculations and edge cases
- **Data Management**: Save/load functionality
- **Error Scenarios**: Graceful failure handling
- **Mock Testing**: Isolated component testing

**Test Coverage**:
- `test_stock.py`: Stock class functionality (103 tests)
- `test_stock_analysis.py`: Valuation algorithms (25 tests)
- `test_analysis_service.py`: Service layer logic (25 tests)

## Project Structure

```
magic-beans/
├── app.py                      # Flask application entry point
├── stock.py                    # Stock data management class
├── stock_analysis.py           # Financial analysis algorithms
├── analysis_service.py         # Business logic services
├── helpers.py                  # Utility functions
├── requirements.txt            # Production dependencies
├── test-requirements.txt       # Testing dependencies
├── pytest.ini                 # Test configuration
├── templates/                  # HTML templates
│   ├── layout.html            # Base template
│   ├── quote.html             # Main analysis page
│   └── about.html             # Project documentation
├── static/                     # Static assets
│   ├── styles.css             # Custom styles
│   └── favicon.ico            # Site icon
├── test_stock.py              # Stock class tests
├── test_stock_analysis.py     # Analysis algorithm tests
└── test_analysis_service.py   # Service layer tests
```

## Demonstration Value

This project demonstrates:

### Programming Concepts
- Object-oriented design patterns
- Web application development with Flask
- API integration and data processing
- Test-driven development methodology
- Error handling and validation
- Code organization and modularity

### Financial Concepts
- Fundamental analysis techniques
- Valuation model implementation
- Risk assessment methodologies
- Market data interpretation
- Investment decision frameworks

## Configuration

### Environment Variables
- `FLASK_ENV`: Set to 'development' for debug mode
- `SECRET_KEY`: Flask secret key (set in app.py)

### Default Parameters
- DCF projection period: 12 years
- Terminal growth rate: 4%
- DDM required return: 8%
- Risk-free rate: 3%

## Contributing

This is an educational project developed for CS50. While contributions are welcome, please note this is primarily for learning and demonstrational purposes.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Disclaimer

Magic Beans is an educational project and should not be used as the sole basis for investment decisions. The analysis provided is for educational and research purposes only. Always consult with qualified financial professionals before making investment choices.

## Acknowledgments

- CS50 course staff for excellent educational resources
- Yahoo Finance for providing free financial data API
- Flask and Python communities for excellent documentation
- Bootstrap team for responsive design framework

## Support

For questions about this CS50 project, please refer to the comprehensive documentation in the `/about` page of the application or review the inline code comments.

---
