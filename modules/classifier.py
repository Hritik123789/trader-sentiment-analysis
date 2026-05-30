"""
Trader archetype classification module.
Classifies traders based on frequency and profitability patterns.
"""

import pandas as pd
from typing import Dict


def classify_trader(trader_df: pd.DataFrame, median_trades: float, median_pnl: float) -> str:
    """
    Classify a single trader into an archetype.
    
    Args:
        trader_df: Dataframe with trader's aggregated stats
        median_trades: Median trade count threshold
        median_pnl: Median PnL threshold
        
    Returns:
        Archetype label string
    """
    total_trades = trader_df['total_trades'].iloc[0] if 'total_trades' in trader_df.columns else trader_df['trade_count'].sum()
    total_pnl = trader_df['total_pnl'].iloc[0] if 'total_pnl' in trader_df.columns else trader_df['daily_pnl'].sum()
    
    high_freq = total_trades >= median_trades
    profitable = total_pnl >= median_pnl
    
    if high_freq and profitable:
        return "High-Frequency Winner"
    elif high_freq and not profitable:
        return "High-Frequency Loser"
    elif not high_freq and profitable:
        return "Low-Frequency Winner"
    else:
        return "Low-Frequency Loser"


def get_archetype_description(archetype: str) -> Dict[str, str]:
    """
    Get detailed description and strategy for each archetype.
    
    Args:
        archetype: Archetype label
        
    Returns:
        Dictionary with description and strategy recommendations
    """
    descriptions = {
        "High-Frequency Winner": {
            "description": "Active traders with consistent profitability. Execute many trades with positive edge.",
            "characteristics": [
                "High trade volume (above median)",
                "Net positive PnL",
                "Likely using systematic strategies",
                "Good risk management"
            ],
            "strategy": "Continue current approach. Focus on maintaining edge and managing transaction costs.",
            "risk_profile": "Moderate - diversified across many trades"
        },
        "High-Frequency Loser": {
            "description": "Active traders struggling with profitability. High activity but negative returns.",
            "characteristics": [
                "High trade volume (above median)",
                "Net negative PnL",
                "Possible overtrading",
                "May lack clear edge"
            ],
            "strategy": "Reduce frequency. Focus on quality over quantity. Review strategy for edge.",
            "risk_profile": "High - accumulating losses through activity"
        },
        "Low-Frequency Winner": {
            "description": "Selective traders with strong profitability. Patient approach with high win quality.",
            "characteristics": [
                "Low trade volume (below median)",
                "Net positive PnL",
                "Selective entry criteria",
                "Strong risk-reward per trade"
            ],
            "strategy": "Maintain selectivity. Consider scaling position sizes on high-conviction setups.",
            "risk_profile": "Low to Moderate - concentrated but profitable"
        },
        "Low-Frequency Loser": {
            "description": "Infrequent traders with negative returns. Limited activity and poor outcomes.",
            "characteristics": [
                "Low trade volume (below median)",
                "Net negative PnL",
                "Possibly inexperienced",
                "May lack consistent strategy"
            ],
            "strategy": "Focus on education and strategy development before increasing activity.",
            "risk_profile": "Moderate - limited exposure but unprofitable"
        }
    }
    
    return descriptions.get(archetype, {
        "description": "Unknown archetype",
        "characteristics": [],
        "strategy": "Analyze trading patterns",
        "risk_profile": "Unknown"
    })


def classify_uploaded_trader(uploaded_df: pd.DataFrame, reference_df: pd.DataFrame) -> Dict:
    """
    Classify a new trader's data against reference dataset thresholds.
    
    Args:
        uploaded_df: New trader's trade data
        reference_df: Reference dataset for threshold calculation
        
    Returns:
        Dictionary with classification results and statistics
    """
    # Calculate reference thresholds
    ref_trader_stats = reference_df.groupby('account').agg({
        'trade_count': 'sum',
        'daily_pnl': 'sum'
    }).reset_index()
    
    median_trades = ref_trader_stats['trade_count'].median()
    median_pnl = ref_trader_stats['daily_pnl'].median()
    
    # Calculate uploaded trader stats
    total_trades = len(uploaded_df)
    total_pnl = uploaded_df['net_pnl'].sum() if 'net_pnl' in uploaded_df.columns else uploaded_df['closedPnl'].sum()
    avg_position = uploaded_df['sz'].mean() if 'sz' in uploaded_df.columns else 0
    
    # Classify
    high_freq = total_trades >= median_trades
    profitable = total_pnl >= median_pnl
    
    if high_freq and profitable:
        archetype = "High-Frequency Winner"
    elif high_freq and not profitable:
        archetype = "High-Frequency Loser"
    elif not high_freq and profitable:
        archetype = "Low-Frequency Winner"
    else:
        archetype = "Low-Frequency Loser"
    
    # Get description
    archetype_info = get_archetype_description(archetype)
    
    # Calculate percentiles
    trade_percentile = (ref_trader_stats['trade_count'] < total_trades).sum() / len(ref_trader_stats) * 100
    pnl_percentile = (ref_trader_stats['daily_pnl'] < total_pnl).sum() / len(ref_trader_stats) * 100
    
    return {
        "archetype": archetype,
        "total_trades": total_trades,
        "total_pnl": total_pnl,
        "avg_position_size": avg_position,
        "trade_percentile": trade_percentile,
        "pnl_percentile": pnl_percentile,
        "description": archetype_info["description"],
        "characteristics": archetype_info["characteristics"],
        "strategy": archetype_info["strategy"],
        "risk_profile": archetype_info["risk_profile"],
        "thresholds": {
            "median_trades": median_trades,
            "median_pnl": median_pnl
        }
    }
