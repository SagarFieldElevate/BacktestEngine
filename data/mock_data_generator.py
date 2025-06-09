"""
Generate mock Bitcoin market data for testing
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
from pathlib import Path

def generate_bitcoin_price_data(start_date="2020-01-01", end_date="2024-12-31"):
    """Generate realistic Bitcoin OHLCV data"""
    
    dates = pd.date_range(start=start_date, end=end_date, freq='D')
    n_days = len(dates)
    
    # Generate price series with realistic volatility
    initial_price = 10000
    returns = np.random.normal(0.002, 0.04, n_days)  # 0.2% daily return, 4% volatility
    prices = initial_price * np.exp(np.cumsum(returns))
    
    # Generate OHLCV data
    data = []
    for i, (date, close_price) in enumerate(zip(dates, prices)):
        # Generate realistic OHLC from close price
        daily_volatility = np.random.uniform(0.01, 0.05)
        
        high = close_price * (1 + daily_volatility * np.random.uniform(0.5, 1.5))
        low = close_price * (1 - daily_volatility * np.random.uniform(0.5, 1.5))
        open_price = close_price * (1 + np.random.uniform(-0.02, 0.02))
        
        # Ensure OHLC constraints
        high = max(high, open_price, close_price)
        low = min(low, open_price, close_price)
        
        volume = np.random.lognormal(20, 1) * 1e6  # Log-normal volume
        
        data.append({
            'Date': date,
            'Open': round(open_price, 2),
            'High': round(high, 2),
            'Low': round(low, 2),
            'Close': round(close_price, 2),
            'Volume': round(volume, 2)
        })
    
    return pd.DataFrame(data)

def generate_technical_indicators(price_df):
    """Generate technical indicators based on price data"""
    
    df = price_df.copy()
    
    # Simple Moving Averages
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    df['SMA_200'] = df['Close'].rolling(window=200).mean()
    
    # Exponential Moving Averages
    df['EMA_12'] = df['Close'].ewm(span=12).mean()
    df['EMA_26'] = df['Close'].ewm(span=26).mean()
    
    # RSI
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI_14'] = 100 - (100 / (1 + rs))
    
    # RSI 30
    gain30 = (delta.where(delta > 0, 0)).rolling(window=30).mean()
    loss30 = (-delta.where(delta < 0, 0)).rolling(window=30).mean()
    rs30 = gain30 / loss30
    df['RSI_30'] = 100 - (100 / (1 + rs30))
    
    return df

def generate_macro_indicators(dates):
    """Generate macro economic indicators"""
    
    n_days = len(dates)
    
    # DXY (Dollar Index) - mean reverting around 95
    dxy_base = 95
    dxy_returns = np.random.normal(0, 0.002, n_days)
    dxy_values = dxy_base + np.cumsum(dxy_returns)
    dxy_values = dxy_base + 0.8 * (dxy_values - dxy_base)  # Mean reversion
    
    # Fear & Greed Index (0-100)
    fear_greed = 50 + 30 * np.sin(np.arange(n_days) * 2 * np.pi / 365) + np.random.normal(0, 10, n_days)
    fear_greed = np.clip(fear_greed, 0, 100)
    
    return pd.DataFrame({
        'Date': dates,
        'DXY': dxy_values,
        'FearGreed': fear_greed
    })

def create_mock_market_data():
    """Create complete mock market data"""
    
    # Generate price data
    price_df = generate_bitcoin_price_data()
    
    # Generate technical indicators
    technical_df = generate_technical_indicators(price_df)
    
    # Generate macro indicators
    macro_df = generate_macro_indicators(price_df['Date'])
    
    # Merge all data
    market_data = technical_df.copy()
    market_data = market_data.merge(macro_df, on='Date', how='left')
    
    # Set date as index
    market_data['Date'] = pd.to_datetime(market_data['Date'])
    market_data.set_index('Date', inplace=True)
    
    # Add derived features
    market_data['Returns'] = market_data['Close'].pct_change()
    market_data['LogReturns'] = np.log(market_data['Close'] / market_data['Close'].shift(1))
    market_data['Volatility'] = market_data['Returns'].rolling(window=20).std()
    
    # Forward fill any NaN values
    market_data.ffill(inplace=True)
    market_data.dropna(inplace=True)
    
    return market_data

if __name__ == "__main__":
    # Generate and save mock data
    data_dir = Path(__file__).parent / "raw"
    data_dir.mkdir(exist_ok=True)
    
    print("Generating mock Bitcoin market data...")
    market_data = create_mock_market_data()
    
    # Save as parquet
    output_file = data_dir / "market_data_latest.parquet"
    market_data.to_parquet(output_file)
    
    print(f"Mock data saved to {output_file}")
    print(f"Data shape: {market_data.shape}")
    print(f"Date range: {market_data.index.min()} to {market_data.index.max()}")
    print(f"Columns: {list(market_data.columns)}")