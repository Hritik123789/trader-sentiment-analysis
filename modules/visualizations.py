"""
Visualization module for trader sentiment analysis.
All charts use Plotly for interactivity.
"""

import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from typing import Dict


# Color scheme for sentiment categories
SENTIMENT_COLORS = {
    "Extreme Fear": "#8B0000",
    "Fear": "#DC143C",
    "Neutral": "#FFD700",
    "Greed": "#90EE90",
    "Extreme Greed": "#006400"
}


def pnl_bar_chart(analysis_df: pd.DataFrame) -> go.Figure:
    """
    Create grouped bar chart showing average vs median PnL by sentiment.
    
    Args:
        analysis_df: Dataframe with PnL statistics by sentiment
        
    Returns:
        Plotly figure object
    """
    # Sort by sentiment order
    sentiment_order = ["Extreme Fear", "Fear", "Neutral", "Greed", "Extreme Greed"]
    analysis_df['fg_label'] = pd.Categorical(analysis_df['fg_label'], categories=sentiment_order, ordered=True)
    analysis_df = analysis_df.sort_values('fg_label')
    
    fig = go.Figure()
    
    # Add average PnL bars
    fig.add_trace(go.Bar(
        name='Average PnL',
        x=analysis_df['fg_label'],
        y=analysis_df['avg_pnl'],
        marker_color='#1f77b4',
        text=analysis_df['avg_pnl'].apply(lambda x: f'${x:,.0f}'),
        textposition='outside'
    ))
    
    # Add median PnL bars
    fig.add_trace(go.Bar(
        name='Median PnL',
        x=analysis_df['fg_label'],
        y=analysis_df['median_pnl'],
        marker_color='#ff7f0e',
        text=analysis_df['median_pnl'].apply(lambda x: f'${x:,.0f}'),
        textposition='outside'
    ))
    
    fig.update_layout(
        title='Daily PnL by Sentiment Category',
        xaxis_title='Sentiment',
        yaxis_title='PnL ($)',
        barmode='group',
        hovermode='x unified',
        template='plotly_white',
        height=500
    )
    
    return fig


def winrate_chart(analysis_df: pd.DataFrame) -> go.Figure:
    """
    Create horizontal bar chart of win rate by sentiment.
    
    Args:
        analysis_df: Dataframe with win rate statistics by sentiment
        
    Returns:
        Plotly figure object
    """
    # Sort by win rate
    analysis_df = analysis_df.sort_values('win_rate', ascending=True)
    
    # Assign colors
    colors = [SENTIMENT_COLORS.get(label, '#808080') for label in analysis_df['fg_label']]
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        y=analysis_df['fg_label'],
        x=analysis_df['win_rate'],
        orientation='h',
        marker_color=colors,
        text=analysis_df['win_rate'].apply(lambda x: f'{x:.1f}%'),
        textposition='outside'
    ))
    
    fig.update_layout(
        title='Win Rate by Sentiment Category',
        xaxis_title='Win Rate (%)',
        yaxis_title='Sentiment',
        template='plotly_white',
        height=400
    )
    
    return fig


def drawdown_chart(analysis_df: pd.DataFrame) -> go.Figure:
    """
    Create bar chart of worst daily loss by sentiment (shown as positive numbers).
    
    Args:
        analysis_df: Dataframe with drawdown statistics by sentiment
        
    Returns:
        Plotly figure object
    """
    # Sort by sentiment order
    sentiment_order = ["Extreme Fear", "Fear", "Neutral", "Greed", "Extreme Greed"]
    analysis_df['fg_label'] = pd.Categorical(analysis_df['fg_label'], categories=sentiment_order, ordered=True)
    analysis_df = analysis_df.sort_values('fg_label')
    
    # Convert to positive for display
    analysis_df['worst_day_abs'] = analysis_df['worst_day'].abs()
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=analysis_df['fg_label'],
        y=analysis_df['worst_day_abs'],
        marker_color='#DC143C',
        text=analysis_df['worst_day_abs'].apply(lambda x: f'${x:,.0f}'),
        textposition='outside'
    ))
    
    fig.update_layout(
        title='Worst Single Day Loss by Sentiment',
        xaxis_title='Sentiment',
        yaxis_title='Maximum Loss ($)',
        template='plotly_white',
        height=500
    )
    
    return fig


def frequency_chart(analysis_df: pd.DataFrame) -> go.Figure:
    """
    Create bar chart of average daily trades by sentiment.
    
    Args:
        analysis_df: Dataframe with trade frequency statistics by sentiment
        
    Returns:
        Plotly figure object
    """
    # Sort by sentiment order
    sentiment_order = ["Extreme Fear", "Fear", "Neutral", "Greed", "Extreme Greed"]
    analysis_df['fg_label'] = pd.Categorical(analysis_df['fg_label'], categories=sentiment_order, ordered=True)
    analysis_df = analysis_df.sort_values('fg_label')
    
    # Assign colors
    colors = [SENTIMENT_COLORS.get(label, '#808080') for label in analysis_df['fg_label']]
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=analysis_df['fg_label'],
        y=analysis_df['avg_daily_trades'],
        marker_color=colors,
        text=analysis_df['avg_daily_trades'].apply(lambda x: f'{x:.1f}'),
        textposition='outside'
    ))
    
    fig.update_layout(
        title='Average Daily Trades by Sentiment',
        xaxis_title='Sentiment',
        yaxis_title='Avg Daily Trades',
        template='plotly_white',
        height=500
    )
    
    return fig


def archetype_scatter(df: pd.DataFrame) -> go.Figure:
    """
    Create scatter plot of traders by trade count and total PnL.
    
    Args:
        df: Dataframe with trader archetypes
        
    Returns:
        Plotly figure object
    """
    # Color mapping for archetypes
    archetype_colors = {
        "High-Frequency Winner": "#006400",
        "High-Frequency Loser": "#DC143C",
        "Low-Frequency Winner": "#90EE90",
        "Low-Frequency Loser": "#FFA07A"
    }
    
    fig = px.scatter(
        df,
        x='total_trades',
        y='total_pnl',
        color='archetype',
        size='avg_position_size',
        hover_data=['account'],
        color_discrete_map=archetype_colors,
        title='Trader Archetypes: Frequency vs Profitability'
    )
    
    fig.update_layout(
        xaxis_title='Total Trades',
        yaxis_title='Total PnL ($)',
        template='plotly_white',
        height=600
    )
    
    # Add quadrant lines
    median_trades = df['total_trades'].median()
    median_pnl = df['total_pnl'].median()
    
    fig.add_hline(y=median_pnl, line_dash="dash", line_color="gray", opacity=0.5)
    fig.add_vline(x=median_trades, line_dash="dash", line_color="gray", opacity=0.5)
    
    return fig


def correlation_heatmap(df: pd.DataFrame) -> go.Figure:
    """
    Create correlation heatmap of key trading metrics.
    
    Args:
        df: Daily metrics dataframe
        
    Returns:
        Plotly figure object
    """
    # Select numeric columns for correlation
    corr_cols = ['daily_pnl', 'win_rate', 'trade_count', 'avg_position_size', 'fg_value']
    corr_matrix = df[corr_cols].corr()
    
    # Create heatmap
    fig = go.Figure(data=go.Heatmap(
        z=corr_matrix.values,
        x=corr_matrix.columns,
        y=corr_matrix.columns,
        colorscale='RdBu',
        zmid=0,
        text=corr_matrix.values.round(2),
        texttemplate='%{text}',
        textfont={"size": 12},
        colorbar=dict(title="Correlation")
    ))
    
    fig.update_layout(
        title='Correlation Matrix: Trading Metrics',
        template='plotly_white',
        height=500,
        xaxis={'side': 'bottom'}
    )
    
    return fig
