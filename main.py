"""
Field Elevate Bitcoin Backtester
Main orchestration script for running backtests
"""

import argparse
import json
import logging
from datetime import datetime
from pathlib import Path

from data.fetch_from_airtable import AirtableDataFetcher
from strategy.load_strategies import StrategyLoader
from strategy.backtest_engine import BacktestEngine
from utils.airtable_helper import AirtableHelper

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description='Field Elevate Bitcoin Backtester')
    parser.add_argument('--strategy', type=str, help='Specific strategy ID to test')
    parser.add_argument('--all-strategies', action='store_true', help='Test all strategies')
    parser.add_argument('--start-date', type=str, help='Backtest start date (YYYY-MM-DD)')
    parser.add_argument('--end-date', type=str, help='Backtest end date (YYYY-MM-DD)')
    parser.add_argument('--initial-capital', type=float, default=100000, help='Initial capital')
    parser.add_argument('--refresh-data', action='store_true', help='Force refresh data from Airtable')
    
    args = parser.parse_args()
    
    # Step 1: Fetch or load market data
    logger.info("Loading market data...")
    data_fetcher = AirtableDataFetcher()
    
    if args.refresh_data or not data_fetcher.has_cached_data():
        try:
            logger.info("Fetching fresh data from Airtable...")
            market_data = data_fetcher.fetch_all_data()
        except Exception as e:
            logger.warning(f"Failed to fetch from Airtable: {e}")
            
            # Check if we have cached data to fall back on
            if data_fetcher.has_cached_data():
                logger.info("Falling back to cached data...")
                market_data = data_fetcher.load_cached_data()
            else:
                # Generate mock data as last resort
                logger.info("No cached data available. Generating mock data for testing...")
                from data.mock_data_generator import create_mock_market_data
                market_data = create_mock_market_data()
                
                # Save the mock data as cache
                data_fetcher._save_to_cache(market_data)
    else:
        logger.info("Loading cached data...")
        market_data = data_fetcher.load_cached_data()
    
    # Step 2: Load strategies from Pinecone
    logger.info("Loading strategies from Pinecone...")
    strategy_loader = StrategyLoader()
    
    if args.strategy:
        strategies = [strategy_loader.load_strategy(args.strategy)]
    elif args.all_strategies:
        strategies = strategy_loader.load_all_strategies()
    else:
        strategies = strategy_loader.load_top_strategies(n=5)
    
    logger.info(f"Loaded {len(strategies)} strategies")
    
    # Step 3: Initialize backtest engine
    engine = BacktestEngine(
        market_data=market_data,
        initial_capital=args.initial_capital,
        start_date=args.start_date,
        end_date=args.end_date
    )
    
    # Step 4: Run backtests
    results = {}
    for strategy in strategies:
        logger.info(f"Running backtest for strategy: {strategy.name}")
        result = engine.run_backtest(strategy)
        results[strategy.id] = result
        
        # Generate individual report
        engine.generate_report(strategy, result)
    
    # Step 5: Generate comparative analysis
    if len(strategies) > 1:
        logger.info("Generating comparative analysis...")
        engine.generate_comparative_report(results)
    
    logger.info("Backtest complete! Results saved to outputs/")


if __name__ == "__main__":
    main()