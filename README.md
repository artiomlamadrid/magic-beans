# Magic Beans - Professional Stock Analysis Platform

A sophisticated web-based financial analysis platform built with Flask, designed for investors, financial professionals, and trading firms. This production-ready application provides institutional-grade valuation models and real-time market data integration for comprehensive equity research and investment decision-making.

## Project Overview

Magic Beans is a professional-grade financial analysis platform that provides sophisticated tools for equity research and investment analysis. Built with modern web technologies, it offers real-time data integration, multiple industry-standard valuation models, and an intuitive interface designed for financial professionals. The application demonstrates enterprise-level programming practices including object-oriented design, comprehensive testing, and scalable architecture.

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

This is a professional portfolio project demonstrating full-stack development capabilities. Contributions and collaboration opportunities are welcome for enhancement and scaling.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Disclaimer

Magic Beans is a demonstration platform showcasing advanced financial modeling capabilities. While built with production-quality code and methodologies, it should be used in conjunction with professional financial advice for actual investment decisions. The analysis provided demonstrates technical proficiency in financial modeling and web application development.

## Acknowledgments

- Yahoo Finance for providing comprehensive financial data API
- Flask and Python communities for excellent documentation and frameworks
- Bootstrap team for responsive design components
- Open source financial modeling libraries and methodologies

## Support

For questions about implementation details, customization, or potential collaboration opportunities, please reach out through GitHub or review the comprehensive inline documentation throughout the codebase.

---
