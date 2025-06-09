"""
Airtable data fetcher for Bitcoin market data
"""

import json
import logging
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from airtable import Airtable
import config
from utils.airtable_helper import AirtableHelper

logger = logging.getLogger(__name__)


class AirtableDataFetcher:
    def __init__(self):
        self.helper = AirtableHelper()
        self.cache_dir = Path(config.DATA_CACHE_DIR)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
    def fetch_all_data(self) -> pd.DataFrame:
        """Fetch all Bitcoin market data from Airtable"""
        try:
            # Fetch Bitcoin price data
            price_data = self._fetch_bitcoin_prices()
            
            # Fetch technical indicators
            technical_data = self._fetch_technical_indicators()
            
            # Fetch macro data
            macro_data = self._fetch_macro_indicators()
            
            # Merge all data
            market_data = self._merge_data_sources(price_data, technical_data, macro_data)
            
            # Cache the data
            self._save_to_cache(market_data)
            
            return market_data
            
        except Exception as e:
            logger.error(f"Error fetching data from Airtable: {e}")
            raise
    
    def _fetch_bitcoin_prices(self) -> pd.DataFrame:
        """Fetch Bitcoin OHLCV data"""
        logger.info("Fetching Bitcoin price data...")
        
        records = self.helper.fetch_records(
            table_name="BTCPriceDaily",
            fields=["Date", "Open", "High", "Low", "Close", "Volume"]
        )
        
        df = pd.DataFrame(records)
        df['Date'] = pd.to_datetime(df['Date'])
        df.set_index('Date', inplace=True)
        
        return df
    
    def _fetch_technical_indicators(self) -> pd.DataFrame:
        """Fetch technical indicators"""
        logger.info("Fetching technical indicators...")
        
        # RSI, MACD, Bollinger Bands, etc.
        indicators = {}
        
        # Fetch RSI
        rsi_records = self.helper.fetch_records(
            table_name="Technicals",
            fields=["Date", "RSI_14", "RSI_30"]
        )
        indicators['RSI'] = pd.DataFrame(rsi_records)
        
        # Fetch Moving Averages
        ma_records = self.helper.fetch_records(
            table_name="Technicals",
            fields=["Date", "SMA_20", "SMA_50", "SMA_200", "EMA_12", "EMA_26"]
        )
        indicators['MA'] = pd.DataFrame(ma_records)
        
        # Merge indicators
        technical_df = indicators['RSI']
        for name, df in indicators.items():
            if name != 'RSI':
                technical_df = technical_df.merge(df, on='Date', how='outer')
        
        technical_df['Date'] = pd.to_datetime(technical_df['Date'])
        technical_df.set_index('Date', inplace=True)
        
        return technical_df
    
    def _fetch_macro_indicators(self) -> pd.DataFrame:
        """Fetch macro economic indicators"""
        logger.info("Fetching macro indicators...")
        
        macro_data = {}
        
        # DXY
        dxy_records = self.helper.fetch_records(
            table_name="AlphaVantage_DXY",
            fields=["Date", "Close"]
        )
        macro_data['DXY'] = pd.DataFrame(dxy_records).rename(columns={'Close': 'DXY'})
        
        # Fear & Greed Index
        fng_records = self.helper.fetch_records(
            table_name="Alternative_Fear_Greed",
            fields=["Date", "Value", "Classification"]
        )
        macro_data['FearGreed'] = pd.DataFrame(fng_records).rename(columns={'Value': 'FearGreed'})
        
        # Merge macro data
        macro_df = macro_data['DXY']
        for name, df in macro_data.items():
            if name != 'DXY':
                macro_df = macro_df.merge(df[['Date', df.columns[-1]]], on='Date', how='outer')
        
        macro_df['Date'] = pd.to_datetime(macro_df['Date'])
        macro_df.set_index('Date', inplace=True)
        
        return macro_df
    
    def _merge_data_sources(self, price_df: pd.DataFrame, 
                           technical_df: pd.DataFrame, 
                           macro_df: pd.DataFrame) -> pd.DataFrame:
        """Merge all data sources into single DataFrame"""
        logger.info("Merging data sources...")
        
        # Start with price data as base
        market_data = price_df.copy()
        
        # Merge technical indicators
        market_data = market_data.join(technical_df, how='left')
        
        # Merge macro indicators
        market_data = market_data.join(macro_df, how='left')
        
        # Forward fill missing values
        market_data.ffill(inplace=True)
        
        # Add derived features
        market_data['Returns'] = market_data['Close'].pct_change()
        market_data['LogReturns'] = np.log(market_data['Close'] / market_data['Close'].shift(1))
        market_data['Volatility'] = market_data['Returns'].rolling(window=20).std()
        
        return market_data
    
    def _save_to_cache(self, data: pd.DataFrame):
        """Save data to cache"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        cache_file = self.cache_dir / f"market_data_{timestamp}.parquet"
        data.to_parquet(cache_file)
        
        # Also save as latest
        latest_file = self.cache_dir / "market_data_latest.parquet"
        data.to_parquet(latest_file)
        
        logger.info(f"Data cached to {cache_file}")
    
    def has_cached_data(self) -> bool:
        """Check if cached data exists"""
        latest_file = self.cache_dir / "market_data_latest.parquet"
        return latest_file.exists()
    
    def load_cached_data(self) -> pd.DataFrame:
        """Load cached data"""
        latest_file = self.cache_dir / "market_data_latest.parquet"
        if not latest_file.exists():
            raise FileNotFoundError("No cached data found")
        
        return pd.read_parquet(latest_file)