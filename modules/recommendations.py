"""
Strategy recommendation module based on Fear & Greed Index.
Provides actionable trading signals and risk management guidance.
"""

import pandas as pd
from typing import Dict, List


def get_strategy_recommendation(fg_value: float, fg_label: str) -> Dict[str, str]:
    """
    Generate strategy recommendation based on current Fear & Greed value.
    
    Args:
        fg_value: Fear & Greed Index value (0-100)
        fg_label: Fear & Greed classification label
        
    Returns:
        Dictionary with signal, action, risk level, and reasoning
    """
    if fg_value < 25:  # Extreme Fear
        return {
            "signal": "🟢 STRONG BUY",
            "action": "Increase position sizes by 25-50%",
            "risk_level": "LOW",
            "reasoning": "Historical data shows Extreme Fear offers best risk-reward. "
                        "Trading activity spikes 4x (panic creates opportunities). "
                        "While not highest win rate, excellent average PnL per day.",
            "position_sizing": "Aggressive - scale into positions",
            "stop_loss": "Standard stops - volatility creates opportunities",
            "profit_target": "Extended targets - expect mean reversion",
            "max_daily_loss": "Standard risk limits"
        }
    
    elif fg_value < 45:  # Fear
        return {
            "signal": "🟢 BUY",
            "action": "Increase position sizes by 10-25%",
            "risk_level": "LOW-MODERATE",
            "reasoning": "Fear days produce highest average daily PnL ($5,328). "
                        "Good balance of opportunity and manageable volatility. "
                        "Market pessimism creates entry opportunities.",
            "position_sizing": "Moderately aggressive",
            "stop_loss": "Standard stops",
            "profit_target": "Standard to extended targets",
            "max_daily_loss": "Standard risk limits"
        }
    
    elif fg_value < 55:  # Neutral
        return {
            "signal": "⚪ NEUTRAL",
            "action": "Maintain standard position sizes",
            "risk_level": "MODERATE",
            "reasoning": "Neutral sentiment shows balanced market conditions. "
                        "No strong directional bias from sentiment. "
                        "Trade based on technical setups, not sentiment edge.",
            "position_sizing": "Standard sizing",
            "stop_loss": "Standard stops",
            "profit_target": "Standard targets",
            "max_daily_loss": "Standard risk limits"
        }
    
    elif fg_value < 75:  # Greed
        return {
            "signal": "🟡 CAUTION",
            "action": "Reduce position sizes by 25-40%",
            "risk_level": "MODERATE-HIGH",
            "reasoning": "Greed days show elevated risk. Historical data reveals "
                        "worst single-day loss (-$358,963) occurred during Greed. "
                        "Reduce exposure and tighten risk management.",
            "position_sizing": "Conservative - reduce size",
            "stop_loss": "Tighter stops - protect capital",
            "profit_target": "Take profits earlier",
            "max_daily_loss": "Reduce to 50-75% of standard limit"
        }
    
    else:  # Extreme Greed (>= 75)
        return {
            "signal": "🔴 HIGH RISK",
            "action": "Reduce position sizes by 50%+ or stay flat",
            "risk_level": "HIGH",
            "reasoning": "Extreme Greed produces highest win rate (~89%) BUT "
                        "comes with severe tail risk. Market euphoria precedes corrections. "
                        "Protect capital - wins are smaller, losses can be catastrophic.",
            "position_sizing": "Minimal - preserve capital",
            "stop_loss": "Very tight stops - exit quickly",
            "profit_target": "Quick profits - don't overstay",
            "max_daily_loss": "Reduce to 25-50% of standard limit"
        }


def get_historical_strategy_performance(df: pd.DataFrame) -> pd.DataFrame:
    """
    Backtest strategy recommendations on historical data.
    
    Args:
        df: Daily metrics dataframe with sentiment
        
    Returns:
        Dataframe with strategy performance by sentiment
    """
    # Define position size multipliers based on strategy
    def get_position_multiplier(fg_value: float) -> float:
        if fg_value < 25:  # Extreme Fear
            return 1.375  # Average of 25-50% increase
        elif fg_value < 45:  # Fear
            return 1.175  # Average of 10-25% increase
        elif fg_value < 55:  # Neutral
            return 1.0  # Standard
        elif fg_value < 75:  # Greed
            return 0.675  # Average of 25-40% reduction
        else:  # Extreme Greed
            return 0.5  # 50% reduction
    
    # Define max daily loss limits
    def get_max_loss_limit(fg_value: float, standard_limit: float = -10000) -> float:
        if fg_value < 55:  # Fear/Neutral
            return standard_limit
        elif fg_value < 75:  # Greed
            return standard_limit * 0.625  # 50-75% average
        else:  # Extreme Greed
            return standard_limit * 0.375  # 25-50% average
    
    # Apply strategy
    df = df.copy()
    df['position_multiplier'] = df['fg_value'].apply(get_position_multiplier)
    df['strategy_pnl'] = df['daily_pnl'] * df['position_multiplier']
    
    # Apply loss limits
    standard_limit = df['daily_pnl'].quantile(0.05)  # 5th percentile as standard limit
    df['max_loss_limit'] = df['fg_value'].apply(lambda x: get_max_loss_limit(x, standard_limit))
    
    # Cap losses at limit
    df['strategy_pnl_capped'] = df.apply(
        lambda row: max(row['strategy_pnl'], row['max_loss_limit']), 
        axis=1
    )
    
    # Calculate performance by sentiment
    performance = df.groupby('fg_label').agg({
        'daily_pnl': ['sum', 'mean', 'std'],
        'strategy_pnl_capped': ['sum', 'mean', 'std']
    }).reset_index()
    
    performance.columns = ['sentiment', 
                          'baseline_total', 'baseline_avg', 'baseline_std',
                          'strategy_total', 'strategy_avg', 'strategy_std']
    
    # Calculate improvement
    performance['improvement_pct'] = (
        (performance['strategy_total'] - performance['baseline_total']) / 
        performance['baseline_total'].abs() * 100
    ).round(2)
    
    # Round values
    for col in ['baseline_total', 'baseline_avg', 'baseline_std', 
                'strategy_total', 'strategy_avg', 'strategy_std']:
        performance[col] = performance[col].round(2)
    
    return performance


def get_risk_alerts(fg_value: float, fg_label: str) -> List[str]:
    """
    Generate risk alerts based on current sentiment.
    
    Args:
        fg_value: Fear & Greed Index value
        fg_label: Fear & Greed classification label
        
    Returns:
        List of risk alert strings
    """
    alerts = []
    
    if fg_value >= 75:
        alerts.append("⚠️ EXTREME GREED ALERT: Highest tail risk environment. Historical worst loss occurred in this regime.")
        alerts.append("🛡️ Implement strict daily loss limits (50% of normal)")
        alerts.append("⏱️ Reduce holding periods - take profits quickly")
        
    elif fg_value >= 55:
        alerts.append("⚠️ GREED WARNING: Elevated risk environment")
        alerts.append("🛡️ Tighten stop losses and reduce position sizes")
        
    elif fg_value < 25:
        alerts.append("✅ EXTREME FEAR: Historically best risk-reward environment")
        alerts.append("📈 Consider scaling into positions on further weakness")
        alerts.append("⚡ Expect 4x normal trading activity - liquidity may be volatile")
        
    elif fg_value < 45:
        alerts.append("✅ FEAR: Historically highest average daily PnL")
        alerts.append("📈 Good environment for position building")
    
    return alerts


def get_sentiment_summary() -> Dict[str, Dict]:
    """
    Get comprehensive summary of all sentiment regimes.
    
    Returns:
        Dictionary with summary for each sentiment category
    """
    return {
        "Extreme Fear": {
            "range": "0-24",
            "win_rate": "~85%",
            "avg_pnl": "Good",
            "tail_risk": "Low",
            "activity": "Very High (4x normal)",
            "recommendation": "Aggressive buying opportunity"
        },
        "Fear": {
            "range": "25-44",
            "win_rate": "~86%",
            "avg_pnl": "Best ($5,328/day)",
            "tail_risk": "Low",
            "activity": "Above Average",
            "recommendation": "Strong buying opportunity"
        },
        "Neutral": {
            "range": "45-54",
            "win_rate": "~85%",
            "avg_pnl": "Moderate",
            "tail_risk": "Moderate",
            "activity": "Normal",
            "recommendation": "Trade technical setups"
        },
        "Greed": {
            "range": "55-74",
            "win_rate": "~87%",
            "avg_pnl": "Below Average",
            "tail_risk": "High (worst loss: -$358K)",
            "activity": "Below Average",
            "recommendation": "Reduce exposure, tight stops"
        },
        "Extreme Greed": {
            "range": "75-100",
            "win_rate": "Highest (~89%)",
            "avg_pnl": "Low",
            "tail_risk": "Very High",
            "activity": "Low",
            "recommendation": "Minimal exposure, capital preservation"
        }
    }
