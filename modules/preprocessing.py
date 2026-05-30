"""
Data preprocessing module for trader sentiment analysis.
Handles loading, cleaning, and merging trade and sentiment data.
"""

import pandas as pd
import numpy as np
from typing import Tuple


def load_data(trades_path: str, sentiment_path: str) -> pd.DataFrame:
    """
    Load and merge trade data with sentiment data.
    
    Args:
        trades_path: Path to historical trades CSV
        sentiment_path: Path to Fear & Greed Index CSV
        
    Returns:
        Merged dataframe with trades and sentiment
    """
    trades_df = pd.read_csv(trades_path)
    sentiment_df = pd.read_csv(sentiment_path)
    
    # Clean both datasets
    trades_df = clean_trades(trades_df)
    sentiment_df = clean_sentiment(sentiment_df)
    
    # Merge on date
    merged_df = merge_sentiment(trades_df, sentiment_df)
    
    return merged_df
    
    return merged_df


def clean_trades(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and prepare trade data.
    
    Args:
        df: Raw trades dataframe
        
    Returns:
        Cleaned trades dataframe
    """
    df = df.copy()
    
    # Normalize column names (handle both formats)
    column_mapping = {
        'Account': 'account',
        'Coin': 'coin',
        'Timestamp': 'timestamp',
        'Execution Price': 'px',
        'Size Tokens': 'sz',
        'Closed PnL': 'closedPnl',
        'Fee': 'fee',
        'Direction': 'direction',
        'Side': 'side'
    }
    
    # Rename columns if they exist
    df = df.rename(columns={k: v for k, v in column_mapping.items() if k in df.columns})
    
    # Parse timestamp - handle scientific notation and convert from milliseconds
    if 'timestamp' in df.columns:
        # Convert to numeric first (handles scientific notation like 1.73E+12)
        df['timestamp'] = pd.to_numeric(df['timestamp'], errors='coerce')
        
        # Check if timestamps are in milliseconds (> 1e12) or seconds
        sample_ts = df['timestamp'].iloc[0]
        if sample_ts > 1e12:
            # Milliseconds - convert to seconds
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', errors='coerce')
        else:
            # Already in seconds
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s', errors='coerce')
    
    # Extract date for merging
    df['date'] = df['timestamp'].dt.date
    
    # Handle missing values
    df['closedPnl'] = pd.to_numeric(df['closedPnl'], errors='coerce').fillna(0)
    df['fee'] = pd.to_numeric(df['fee'], errors='coerce').fillna(0)
    
    # Calculate net PnL (after fees)
    df['net_pnl'] = df['closedPnl'] - df['fee']
    
    # Ensure direction column exists
    if 'direction' not in df.columns and 'side' in df.columns:
        df['direction'] = df['side']
    
    # Remove any rows with missing critical data
    df = df.dropna(subset=['account', 'timestamp', 'coin'])
    
    return df


def clean_sentiment(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and prepare sentiment data.
    
    Args:
        df: Raw sentiment dataframe
        
    Returns:
        Cleaned sentiment dataframe
    """
    df = df.copy()
    
    # Normalize column names
    column_mapping = {
        'classification': 'value_classification'
    }
    df = df.rename(columns={k: v for k, v in column_mapping.items() if k in df.columns})
    
    # Parse date
    df['date'] = pd.to_datetime(df['date']).dt.date
    
    # Ensure value is numeric
    df['value'] = pd.to_numeric(df['value'], errors='coerce')
    
    # Remove duplicates, keep most recent
    df = df.drop_duplicates(subset=['date'], keep='last')
    
    # Sort by date
    df = df.sort_values('date')
    
    return df


def merge_sentiment(trades_df: pd.DataFrame, sentiment_df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge trade data with sentiment data on date.
    
    Args:
        trades_df: Cleaned trades dataframe
        sentiment_df: Cleaned sentiment dataframe
        
    Returns:
        Merged dataframe
    """
    # Merge on date
    merged = trades_df.merge(
        sentiment_df[['date', 'value', 'value_classification']], 
        on='date', 
        how='left'
    )
    
    # Rename columns for clarity
    merged = merged.rename(columns={
        'value': 'fg_value',
        'value_classification': 'fg_label'
    })
    
    # Forward fill missing sentiment values (for dates without sentiment data)
    merged['fg_value'] = merged['fg_value'].ffill()
    merged['fg_label'] = merged['fg_label'].ffill()
    
    return merged


def compute_daily_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute daily metrics per account.
    
    Args:
        df: Merged dataframe with trades and sentiment
        
    Returns:
        Dataframe with daily metrics per account
    """
    daily_metrics = df.groupby(['account', 'date', 'fg_label', 'fg_value']).agg({
        'net_pnl': ['sum', 'mean', 'count'],
        'sz': 'mean'
    }).reset_index()
    
    # Flatten column names
    daily_metrics.columns = ['account', 'date', 'fg_label', 'fg_value', 
                             'daily_pnl', 'avg_pnl_per_trade', 'trade_count', 'avg_position_size']
    
    # Calculate win rate per day
    wins_per_day = df[df['net_pnl'] > 0].groupby(['account', 'date']).size().reset_index(name='wins')
    daily_metrics = daily_metrics.merge(wins_per_day, on=['account', 'date'], how='left')
    daily_metrics['wins'] = daily_metrics['wins'].fillna(0)
    daily_metrics['win_rate'] = (daily_metrics['wins'] / daily_metrics['trade_count'] * 100).round(2)
    
    return daily_metrics


def classify_sentiment(value: float) -> str:
    """
    Classify Fear & Greed value into category.
    
    Args:
        value: Fear & Greed Index value (0-100)
        
    Returns:
        Sentiment classification string
    """
    if value < 25:
        return "Extreme Fear"
    elif value < 45:
        return "Fear"
    elif value < 55:
        return "Neutral"
    elif value < 75:
        return "Greed"
    else:
        return "Extreme Greed"
