# LUBRA Trading Analysis Dashboard

## Overview

LUBRA Trading Analysis Dashboard is a Streamlit-based web application for financial market analysis. The system provides technical analysis capabilities for various financial instruments including cryptocurrencies, stocks, and forex pairs. The application utilizes the yfinance library for data retrieval and Plotly for interactive visualizations, implementing the LUBRA (Levy Unified Binary Risk Assessment) trading methodology.

## User Preferences

Preferred communication style: Simple, everyday language.

## System Architecture

The application follows a single-file Streamlit architecture pattern, implementing a web-based dashboard for financial analysis. The system is designed as a monolithic application that combines data fetching, processing, visualization, and user interface components in a single Python file.

**Key Architectural Decisions:**
- **Problem**: Need for an accessible trading analysis tool
- **Solution**: Streamlit web framework for rapid prototyping and deployment
- **Rationale**: Streamlit provides immediate web interface generation from Python code, making it ideal for data analysis dashboards

## Key Components

### Data Layer
- **yfinance Integration**: Primary data source for financial market data
- **Pandas DataFrames**: Core data structure for time-series manipulation
- **NumPy Arrays**: Numerical computations for technical indicators

### Analysis Engine
- **Technical Indicators**: 
  - Simple Moving Averages (SMA_60, SMA_70, SMA_20)
  - Relative Strength Index (RSI_14)
  - Relative Strength Levy (RSL_20)
- **Signal Generation**: LUBRA methodology implementation for buy/sell signals

### Visualization Layer
- **Plotly Graphics**: Interactive charting capabilities
- **Subplots Support**: Multi-panel chart layouts for comprehensive analysis

### User Interface
- **Streamlit Framework**: Web-based dashboard with sidebar controls
- **Interactive Widgets**: Date pickers, text inputs, and parameter controls
- **Responsive Layout**: Wide layout configuration for optimal chart display

## Data Flow

1. **User Input Collection**: Asset symbol, date range, and analysis parameters through sidebar
2. **Data Retrieval**: yfinance API calls to fetch historical price data
3. **Data Processing**: 
   - DataFrame normalization and column standardization
   - Technical indicator calculations
   - Signal generation using LUBRA methodology
4. **Visualization**: Plotly chart generation with interactive features
5. **Display**: Streamlit rendering of analysis results and charts

## External Dependencies

### Core Libraries
- **streamlit**: Web application framework
- **yfinance**: Yahoo Finance API wrapper for market data
- **pandas**: Data manipulation and analysis
- **numpy**: Numerical computing
- **plotly**: Interactive visualization library

### Data Sources
- **Yahoo Finance API**: Primary source for financial market data
- **Support for**: Cryptocurrencies (BTC-USD), stocks (AAPL, PETR4.SA), forex pairs (EURUSD=X)

## Deployment Strategy

The application is designed for simple deployment scenarios:

### Local Development
- Direct Python execution with `streamlit run app.py`
- No database requirements
- Real-time data fetching from external APIs

### Cloud Deployment Options
- **Streamlit Cloud**: Native hosting platform
- **Heroku**: Container-based deployment
- **Replit**: In-browser development and hosting

### Configuration Requirements
- Python 3.7+ environment
- Internet connectivity for data fetching
- No persistent storage requirements (stateless design)

### Scaling Considerations
- Current architecture is suitable for single-user or low-traffic scenarios
- Future enhancements may require:
  - Caching mechanisms for API responses
  - Database integration for historical analysis
  - Session state management for multi-user scenarios