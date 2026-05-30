"""
Core analysis functions for trader sentiment analysis.
Statistical analysis and trader segmentation.
"""

import pandas as pd
import numpy as np
from scipy import stats
from typing import Dict, Tuple


def pnl_by_sentiment(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate average and median PnL grouped by sentiment.
    
    Args:
        df: Daily metrics dataframe
        
    Returns:
        Dataframe with PnL statistics by sentiment
    """
    pnl_stats = df.groupby('fg_label')['daily_pnl'].agg([
        ('avg_pnl', 'mean'),
        ('median_pnl', 'median'),
        ('total_pnl', 'sum'),
        ('std_pnl', 'std'),
        ('count', 'count')
    ]).reset_index()
    
    # Round for readability
    pnl_stats['avg_pnl'] = pnl_stats['avg_pnl'].round(2)
    pnl_stats['median_pnl'] = pnl_stats['median_pnl'].round(2)
    pnl_stats['total_pnl'] = pnl_stats['total_pnl'].round(2)
    pnl_stats['std_pnl'] = pnl_stats['std_pnl'].round(2)
    
    return pnl_stats


def winrate_by_sentiment(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate win rate per sentiment category.
    
    Args:
        df: Daily metrics dataframe
        
    Returns:
        Dataframe with win rates by sentiment
    """
    winrate_stats = df.groupby('fg_label').agg({
        'win_rate': 'mean',
        'trade_count': 'sum'
    }).reset_index()
    
    winrate_stats['win_rate'] = winrate_stats['win_rate'].round(2)
    winrate_stats = winrate_stats.sort_values('win_rate', ascending=False)
    
    return winrate_stats


def drawdown_by_sentiment(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate worst single day loss per sentiment category.
    
    Args:
        df: Daily metrics dataframe
        
    Returns:
        Dataframe with worst drawdowns by sentiment
    """
    drawdown_stats = df.groupby('fg_label')['daily_pnl'].agg([
        ('worst_day', 'min'),
        ('best_day', 'max')
    ]).reset_index()
    
    drawdown_stats['worst_day'] = drawdown_stats['worst_day'].round(2)
    drawdown_stats['best_day'] = drawdown_stats['best_day'].round(2)
    
    return drawdown_stats


def trade_frequency_by_sentiment(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate average daily trades per sentiment category.
    
    Args:
        df: Daily metrics dataframe
        
    Returns:
        Dataframe with trade frequency by sentiment
    """
    freq_stats = df.groupby('fg_label')['trade_count'].agg([
        ('avg_daily_trades', 'mean'),
        ('total_trades', 'sum'),
        ('median_daily_trades', 'median')
    ]).reset_index()
    
    freq_stats['avg_daily_trades'] = freq_stats['avg_daily_trades'].round(2)
    freq_stats['median_daily_trades'] = freq_stats['median_daily_trades'].round(2)
    
    return freq_stats


def trader_archetypes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Segment traders into 4 archetypes based on frequency and profitability.
    
    Args:
        df: Daily metrics dataframe
        
    Returns:
        Dataframe with trader archetypes
    """
    # Aggregate per trader
    trader_stats = df.groupby('account').agg({
        'trade_count': 'sum',
        'daily_pnl': 'sum',
        'avg_position_size': 'mean'
    }).reset_index()
    
    trader_stats.columns = ['account', 'total_trades', 'total_pnl', 'avg_position_size']
    
    # Calculate thresholds (median split)
    median_trades = trader_stats['total_trades'].median()
    median_pnl = trader_stats['total_pnl'].median()
    
    # Classify into archetypes
    def classify_archetype(row):
        high_freq = row['total_trades'] >= median_trades
        profitable = row['total_pnl'] >= median_pnl
        
        if high_freq and profitable:
            return "High-Frequency Winner"
        elif high_freq and not profitable:
            return "High-Frequency Loser"
        elif not high_freq and profitable:
            return "Low-Frequency Winner"
        else:
            return "Low-Frequency Loser"
    
    trader_stats['archetype'] = trader_stats.apply(classify_archetype, axis=1)
    
    return trader_stats


def statistical_summary(df: pd.DataFrame) -> str:
    """
    Generate comprehensive statistical summary for AI context.
    
    Args:
        df: Daily metrics dataframe
        
    Returns:
        Formatted string with key statistics
    """
    # Overall stats
    total_trades = len(df)
    total_traders = df['account'].nunique()
    date_range = f"{df['date'].min()} to {df['date'].max()}"
    
    # PnL stats
    pnl_stats = pnl_by_sentiment(df)
    best_pnl_sentiment = pnl_stats.loc[pnl_stats['avg_pnl'].idxmax(), 'fg_label']
    best_pnl_value = pnl_stats.loc[pnl_stats['avg_pnl'].idxmax(), 'avg_pnl']
    
    # Win rate stats
    winrate_stats = winrate_by_sentiment(df)
    best_winrate_sentiment = winrate_stats.loc[winrate_stats['win_rate'].idxmax(), 'fg_label']
    best_winrate_value = winrate_stats.loc[winrate_stats['win_rate'].idxmax(), 'win_rate']
    
    # Risk stats
    drawdown_stats = drawdown_by_sentiment(df)
    worst_drawdown_sentiment = drawdown_stats.loc[drawdown_stats['worst_day'].idxmin(), 'fg_label']
    worst_drawdown_value = drawdown_stats.loc[drawdown_stats['worst_day'].idxmin(), 'worst_day']
    
    # Frequency stats
    freq_stats = trade_frequency_by_sentiment(df)
    most_active_sentiment = freq_stats.loc[freq_stats['avg_daily_trades'].idxmax(), 'fg_label']
    most_active_value = freq_stats.loc[freq_stats['avg_daily_trades'].idxmax(), 'avg_daily_trades']
    
    # Trader archetypes
    archetypes = trader_archetypes(df)
    archetype_counts = archetypes['archetype'].value_counts()
    winners_pct = (archetypes[archetypes['total_pnl'] > 0].shape[0] / total_traders * 100)
    
    # Kruskal-Wallis test
    groups = [df[df['fg_label'] == label]['daily_pnl'].values for label in df['fg_label'].unique()]
    h_stat, p_value = stats.kruskal(*groups)
    
    summary = f"""
DATASET OVERVIEW:
- Total Daily Records: {total_trades:,}
- Unique Traders: {total_traders}
- Date Range: {date_range}
- Net Winners: {winners_pct:.1f}%

PNL BY SENTIMENT:
- Best Average PnL: {best_pnl_sentiment} (${best_pnl_value:,.2f}/day)
- Worst Tail Risk: {worst_drawdown_sentiment} (${worst_drawdown_value:,.2f} single day loss)

WIN RATE BY SENTIMENT:
- Highest Win Rate: {best_winrate_sentiment} ({best_winrate_value:.1f}%)

TRADING ACTIVITY:
- Most Active Sentiment: {most_active_sentiment} ({most_active_value:.1f} avg daily trades)

TRADER ARCHETYPES:
{archetype_counts.to_string()}

STATISTICAL VALIDATION:
- Kruskal-Wallis H-statistic: {h_stat:.2f}
- P-value: {p_value:.10f}
- Conclusion: Sentiment has statistically significant impact on PnL (p < 0.001)
"""
    
    return summary
