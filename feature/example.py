"""
Example script demonstrating how to use the FeatureEngineer class.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

# Add parent directory to path so we can import our module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from feature.feature import FeatureEngineer, create_features

def load_sample_data():
    """
    Load sample data or create synthetic data if real data is not available.
    """
    try:
        # Try to load real data from parquet file
        df = pd.read_parquet('../data.parquet')
        print(f"Loaded data with shape: {df.shape}")
        return df
    except Exception as e:
        print(f"Could not load real data: {e}")
        print("Creating synthetic data instead")
        
        # Create synthetic data
        np.random.seed(42)
        dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
        symbols = ['BTC', 'ETH', 'XRP', 'SOL']
        
        data = []
        for symbol in symbols:
            base_price = 100 if symbol != 'BTC' else 30000
            volatility = 0.02 if symbol != 'BTC' else 0.015
            
            closes = [base_price]
            for _ in range(len(dates) - 1):
                closes.append(closes[-1] * (1 + np.random.normal(0, volatility)))
            
            for i, date in enumerate(dates):
                close = closes[i]
                data.append({
                    'timestamp': date,
                    'symbol': symbol,
                    'open': close * (1 + np.random.normal(0, 0.005)),
                    'high': close * (1 + abs(np.random.normal(0, 0.01))),
                    'low': close * (1 - abs(np.random.normal(0, 0.01))),
                    'close': close,
                    'volume': np.random.randint(1000, 10000)
                })
        
        df = pd.DataFrame(data)
        df.set_index('timestamp', inplace=True)
        return df

def plot_features(df, symbol, features, figsize=(15, 10)):
    """
    Plot selected features for a given symbol.
    
    Args:
        df: DataFrame with features
        symbol: Symbol to plot
        features: List of features to plot
        figsize: Figure size
    """
    symbol_df = df[df['symbol'] == symbol].copy()
    
    fig, axes = plt.subplots(len(features), 1, figsize=figsize, sharex=True)
    
    if len(features) == 1:
        axes = [axes]
    
    for i, feature in enumerate(features):
        if feature in symbol_df.columns:
            axes[i].plot(symbol_df.index, symbol_df[feature])
            axes[i].set_title(f"{symbol} - {feature}")
            axes[i].grid(True)
    
    plt.tight_layout()
    plt.show()

def main():
    # Load sample data
    df = load_sample_data()
    
    # Print DataFrame info
    print("\nDataFrame Summary:")
    print(f"Symbols: {df['symbol'].unique()}")
    print(f"Date Range: {df.index.min()} to {df.index.max()}")
    print(f"Columns: {df.columns.tolist()}")
    
    # Create features
    print("\nGenerating features...")
    features_df = create_features(df)
    
    # Print new features
    print("\nNew columns added:")
    new_columns = [col for col in features_df.columns if col not in df.columns]
    print(new_columns)
    
    # Print sample of the data with features for one symbol
    symbol = 'BTC' if 'BTC' in features_df['symbol'].unique() else features_df['symbol'].unique()[0]
    print(f"\nSample data for {symbol}:")
    sample = features_df[features_df['symbol'] == symbol].head()
    print(sample)
    
    # Prompt for plotting
    print("\nWould you like to plot some features? Run this in a notebook for visualization.")

if __name__ == "__main__":
    main() 