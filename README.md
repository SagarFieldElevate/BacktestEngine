# Field Elevate Bitcoin Backtester

A comprehensive backtesting framework for Bitcoin trading strategies, integrated with Airtable data infrastructure and Pinecone strategy storage.

## Features

- **Data Integration**: Seamless fetching of market data from Airtable
- **Strategy Management**: Load and manage strategies from Pinecone vector database
- **Comprehensive Metrics**: Calculate Sharpe ratio, Sortino ratio, drawdowns, and more
- **Visualization**: Generate equity curves, trade analysis, and comparative reports
- **Modular Architecture**: Easy to extend with new strategies and data sources

## Setup

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Configure Environment**:
   Create a `.env` file in the project root:
   ```
   AIRTABLE_API_KEY=your_airtable_api_key
   AIRTABLE_BASE_ID=your_base_id
   AIRTABLE_TABLE_NAME=Bitcoin_Market_Data
   
   PINECONE_API_KEY=your_pinecone_api_key
   ```

3. **Prepare Data Directory**:
   ```bash
   mkdir -p data/raw outputs/reports outputs/charts
   ```

## Usage

### Run Single Strategy Backtest
```bash
python main.py --strategy strategy_id_123 --start-date 2015-01-01 --end-date 2024-12-31
```

### Run All Strategies
```bash
python main.py --all-strategies --initial-capital 100000
```

### Refresh Data from Airtable
```bash
python main.py --refresh-data --strategy strategy_id_123
```

## Project Structure

- `main.py` - Main orchestration script
- `data/` - Data fetching and caching modules
- `strategy/` - Strategy loading, backtesting engine, and metrics
- `utils/` - Helper utilities for Airtable and Pinecone
- `outputs/` - Generated reports and visualizations

## Strategy Format

Strategies stored in Pinecone should have the following metadata structure:
```json
{
  "name": "RSI Mean Reversion",
  "description": "Buy oversold, sell overbought",
  "parameters": {
    "rsi_period": 14,
    "oversold_threshold": 30,
    "overbought_threshold": 70
  },
  "entry_conditions": [
    {
      "indicator": "rsi_14",
      "operator": "<",
      "value": 30
    }
  ],
  "exit_conditions": [
    {
      "indicator": "rsi_14",
      "operator": ">",
      "value": 70
    }
  ],
  "risk_management": {
    "position_size": 0.95,
    "stop_loss": 0.05
  }
}
```

## Output Reports

The backtester generates:
- **Individual Strategy Reports**: Metrics, trades, equity curves
- **Comparative Analysis**: Side-by-side strategy comparison
- **Trade Analysis**: Win/loss distribution, trade duration
- **Risk Metrics**: Drawdown analysis, volatility measures

## Extending the Framework

### Adding New Data Sources
1. Extend `AirtableDataFetcher` in `data/fetch_from_airtable.py`
2. Add new fetch methods for your data tables
3. Integrate into `_merge_data_sources()`

### Creating Custom Strategies
1. Inherit from `StrategyTemplate` in `strategy/templates/`
2. Implement `generate_signal()` method
3. Define required indicators in `get_required_indicators()`

### Adding New Metrics
1. Extend `MetricsCalculator` in `strategy/metrics.py`
2. Add calculation method
3. Include in `calculate_all_metrics()`

## Performance Optimization

- Data is cached locally to minimize Airtable API calls
- Strategies are vectorized in Pinecone for fast retrieval
- Pandas operations are optimized for large datasets
- Batch processing for multiple strategy backtests

## License

Proprietary - Field Elevate