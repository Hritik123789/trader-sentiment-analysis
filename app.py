"""
Trader Sentiment Analysis - Streamlit Web Application
Analyzes crypto trading performance against Bitcoin Fear & Greed Index
"""

import streamlit as st
import pandas as pd
import requests
from datetime import datetime
import os

# Import custom modules
from modules.preprocessing import load_data, compute_daily_metrics, clean_trades
from modules.analysis import (
    pnl_by_sentiment, winrate_by_sentiment, drawdown_by_sentiment,
    trade_frequency_by_sentiment, trader_archetypes, statistical_summary
)
from modules.visualizations import (
    pnl_bar_chart, winrate_chart, drawdown_chart, 
    frequency_chart, archetype_scatter, correlation_heatmap
)
from modules.classifier import classify_uploaded_trader, get_archetype_description
from modules.recommendations import (
    get_strategy_recommendation, get_historical_strategy_performance,
    get_risk_alerts, get_sentiment_summary
)
from modules.bedrock_agent import (
    ask_bedrock, get_suggested_questions, test_bedrock_connection
)

# Page configuration
st.set_page_config(
    page_title="Trader Sentiment Analyzer",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .big-metric {
        font-size: 2rem;
        font-weight: bold;
    }
    .sentiment-box {
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
        color: #000000;
    }
    .user-message {
        background-color: #e3f2fd;
        border-left: 4px solid #2196F3;
    }
    .assistant-message {
        background-color: #f5f5f5;
        border-left: 4px solid #4CAF50;
    }
    .chat-message strong {
        color: #000000;
    }
    </style>
""", unsafe_allow_html=True)


@st.cache_data
def load_and_process_data(trades_path: str, sentiment_path: str):
    """Load and process data with caching."""
    df = load_data(trades_path, sentiment_path)
    daily_metrics = compute_daily_metrics(df)
    return df, daily_metrics


@st.cache_data
def get_live_fear_greed():
    """Fetch live Fear & Greed Index from API."""
    try:
        response = requests.get("https://api.alternative.me/fng/?limit=1", timeout=5)
        data = response.json()
        value = int(data['data'][0]['value'])
        timestamp = data['data'][0]['timestamp']
        
        # Classify
        if value < 25:
            label = "Extreme Fear"
        elif value < 45:
            label = "Fear"
        elif value < 55:
            label = "Neutral"
        elif value < 75:
            label = "Greed"
        else:
            label = "Extreme Greed"
        
        return value, label, timestamp
    except:
        return None, None, None


def main():
    # Sidebar
    st.sidebar.title("📈 Trader Sentiment Analyzer")
    st.sidebar.markdown("---")
    st.sidebar.markdown("""
    Analyze 211,000+ crypto trades against Bitcoin Fear & Greed Index.
    
    **Features:**
    - Real-time sentiment analysis
    - Statistical insights
    - AI-powered recommendations
    - Trader archetype classification
    """)
    
    # File uploader
    st.sidebar.markdown("---")
    st.sidebar.subheader("📁 Upload Custom Data")
    uploaded_file = st.sidebar.file_uploader(
        "Upload your trades CSV",
        type=['csv'],
        help="CSV with columns: account, timestamp, coin, direction, px, sz, closedPnl, fee"
    )
    
    # Load data
    with st.spinner("Loading data..."):
        df, daily_metrics = load_and_process_data(
            'historical_data.csv',
            'fear_greed_index.csv'
        )
    
    # Filters
    st.sidebar.markdown("---")
    st.sidebar.subheader("🔍 Filters")
    
    # Date range filter
    min_date = df['date'].min()
    max_date = df['date'].max()
    date_range = st.sidebar.date_input(
        "Date Range",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date
    )
    
    # Coin filter
    all_coins = sorted(df['coin'].unique())
    selected_coins = st.sidebar.multiselect(
        "Select Coins",
        options=all_coins,
        default=all_coins
    )
    
    # Apply filters
    if len(date_range) == 2:
        filtered_df = df[
            (df['date'] >= date_range[0]) & 
            (df['date'] <= date_range[1]) &
            (df['coin'].isin(selected_coins))
        ]
        filtered_daily = daily_metrics[
            (daily_metrics['date'] >= date_range[0]) & 
            (daily_metrics['date'] <= date_range[1])
        ]
    else:
        filtered_df = df[df['coin'].isin(selected_coins)]
        filtered_daily = daily_metrics
    
    # Download button
    st.sidebar.markdown("---")
    csv = filtered_df.to_csv(index=False)
    st.sidebar.download_button(
        label="📥 Download Filtered Data",
        data=csv,
        file_name="filtered_trades.csv",
        mime="text/csv"
    )
    
    # Main content
    st.title("📈 Trader Sentiment Analysis Dashboard")
    st.markdown("### Crypto Trading Performance vs Bitcoin Fear & Greed Index")
    
    # Live Fear & Greed Banner
    st.markdown("---")
    fg_value, fg_label, fg_timestamp = get_live_fear_greed()
    
    if fg_value is not None:
        col1, col2, col3 = st.columns([1, 2, 2])
        
        with col1:
            # Color code the metric
            if fg_value < 45:
                color = "🔴"
            elif fg_value < 55:
                color = "🟡"
            else:
                color = "🟢"
            
            st.metric(
                label="Live Fear & Greed Index",
                value=f"{color} {fg_value}",
                delta=fg_label
            )
        
        with col2:
            recommendation = get_strategy_recommendation(fg_value, fg_label)
            st.markdown(f"**Signal:** {recommendation['signal']}")
            st.markdown(f"**Action:** {recommendation['action']}")
        
        with col3:
            st.markdown(f"**Risk Level:** {recommendation['risk_level']}")
            st.markdown(f"**Reasoning:** {recommendation['reasoning'][:100]}...")
    else:
        st.warning("⚠️ Could not fetch live Fear & Greed Index")
    
    st.markdown("---")
    
    # Key Metrics Row
    st.subheader("📊 Key Metrics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_trades = len(filtered_df)
        st.metric("Total Trades", f"{total_trades:,}")
    
    with col2:
        overall_wins = len(filtered_df[filtered_df['net_pnl'] > 0])
        overall_winrate = (overall_wins / total_trades * 100) if total_trades > 0 else 0
        st.metric("Overall Win Rate", f"{overall_winrate:.1f}%")
    
    with col3:
        pnl_stats = pnl_by_sentiment(filtered_daily)
        best_sentiment = pnl_stats.loc[pnl_stats['avg_pnl'].idxmax(), 'fg_label']
        st.metric("Best PnL Sentiment", best_sentiment)
    
    with col4:
        drawdown_stats = drawdown_by_sentiment(filtered_daily)
        worst_sentiment = drawdown_stats.loc[drawdown_stats['worst_day'].idxmin(), 'fg_label']
        st.metric("Worst Risk Sentiment", worst_sentiment)
    
    st.markdown("---")
    
    # Tabs for different analyses
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 PnL Analysis",
        "🎯 Win Rate",
        "⚠️ Risk Analysis",
        "📈 Trading Activity",
        "👤 Trader Archetypes"
    ])
    
    with tab1:
        st.subheader("Daily PnL by Sentiment Category")
        
        pnl_stats = pnl_by_sentiment(filtered_daily)
        fig = pnl_bar_chart(pnl_stats)
        st.plotly_chart(fig, use_container_width=True)
        
        st.info("""
        **Key Insight:** Fear days produce the highest average daily PnL ($5,328), 
        while Extreme Greed shows lower returns despite highest win rate. 
        This suggests Fear periods offer better risk-reward opportunities.
        """)
        
        st.dataframe(pnl_stats, use_container_width=True)
    
    with tab2:
        st.subheader("Win Rate by Sentiment Category")
        
        winrate_stats = winrate_by_sentiment(filtered_daily)
        fig = winrate_chart(winrate_stats)
        st.plotly_chart(fig, use_container_width=True)
        
        st.dataframe(winrate_stats, use_container_width=True)
        
        st.info("""
        **Key Insight:** Extreme Greed produces highest win rate (~89%), 
        but this is misleading - small wins with catastrophic tail risk. 
        Win rate alone is not a reliable metric for strategy selection.
        """)
    
    with tab3:
        st.subheader("Risk Analysis: Worst Single Day Losses")
        
        drawdown_stats = drawdown_by_sentiment(filtered_daily)
        fig = drawdown_chart(drawdown_stats)
        st.plotly_chart(fig, use_container_width=True)
        
        st.error("""
        **⚠️ CRITICAL RISK WARNING:** 
        Greed days show worst tail risk with single-day loss of -$358,963. 
        Despite high win rates, Greed/Extreme Greed periods require strict 
        position sizing and daily loss limits.
        """)
        
        st.dataframe(drawdown_stats, use_container_width=True)
    
    with tab4:
        st.subheader("Trading Activity by Sentiment")
        
        freq_stats = trade_frequency_by_sentiment(filtered_daily)
        fig = frequency_chart(freq_stats)
        st.plotly_chart(fig, use_container_width=True)
        
        st.info("""
        **Key Insight:** Extreme Fear triggers 4x more trading activity than other periods. 
        This spike reflects panic selling and opportunity seeking. High activity creates 
        both liquidity and volatility.
        """)
        
        st.dataframe(freq_stats, use_container_width=True)
    
    with tab5:
        st.subheader("Trader Archetypes: Frequency vs Profitability")
        
        archetypes = trader_archetypes(filtered_daily)
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig = archetype_scatter(archetypes)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = correlation_heatmap(filtered_daily)
            st.plotly_chart(fig, use_container_width=True)
        
        # Archetype distribution
        st.markdown("#### Archetype Distribution")
        archetype_counts = archetypes['archetype'].value_counts()
        st.bar_chart(archetype_counts)
        
        winners_pct = (archetypes[archetypes['total_pnl'] > 0].shape[0] / len(archetypes) * 100)
        st.success(f"**{winners_pct:.1f}%** of traders are net profitable")
        
        # Classify uploaded trader
        if uploaded_file is not None:
            st.markdown("---")
            st.markdown("#### 🔍 Classify Your Trading Style")
            
            try:
                uploaded_df = pd.read_csv(uploaded_file)
                uploaded_df = clean_trades(uploaded_df)
                
                classification = classify_uploaded_trader(uploaded_df, filtered_daily)
                
                st.success(f"✅ Successfully analyzed {len(uploaded_df):,} trades!")
                
                st.markdown(f"### Your Archetype: **{classification['archetype']}**")
                st.markdown(f"*{classification['description']}*")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Trades", f"{classification['total_trades']:,}")
                    st.caption(f"Percentile: {classification['trade_percentile']:.1f}%")
                
                with col2:
                    st.metric("Total PnL", f"${classification['total_pnl']:,.2f}")
                    st.caption(f"Percentile: {classification['pnl_percentile']:.1f}%")
                
                with col3:
                    st.metric("Avg Position", f"{classification['avg_position_size']:.2f}")
                
                st.markdown("#### 📋 Characteristics")
                for char in classification['characteristics']:
                    st.markdown(f"- {char}")
                
                st.markdown("#### 💡 Recommended Strategy")
                st.info(classification['strategy'])
                
                st.markdown(f"**Risk Profile:** {classification['risk_profile']}")
                
            except Exception as e:
                st.error(f"❌ Error processing uploaded file: {str(e)}")
                st.info("""
                **Required CSV format:**
                - Columns: `account`, `timestamp`, `coin`, `direction`, `px`, `sz`, `closedPnl`, `fee`
                - Or: `Account`, `Timestamp`, `Coin`, `Side`, `Execution Price`, `Size Tokens`, `Closed PnL`, `Fee`
                """)
        else:
            st.info("""
            📤 **Upload your trades CSV** in the sidebar to:
            - Get classified into one of 4 trader archetypes
            - Compare your performance against 32 reference traders
            - Receive personalized strategy recommendations
            """)
    
    # Strategy Recommendations Section
    st.markdown("---")
    st.header("💡 Strategy Recommendations")
    
    if fg_value is not None:
        recommendation = get_strategy_recommendation(fg_value, fg_label)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Current Market Guidance")
            st.markdown(f"**Signal:** {recommendation['signal']}")
            st.markdown(f"**Action:** {recommendation['action']}")
            st.markdown(f"**Risk Level:** {recommendation['risk_level']}")
            st.markdown(f"**Position Sizing:** {recommendation['position_sizing']}")
            st.markdown(f"**Stop Loss:** {recommendation['stop_loss']}")
            st.markdown(f"**Profit Target:** {recommendation['profit_target']}")
            st.markdown(f"**Max Daily Loss:** {recommendation['max_daily_loss']}")
        
        with col2:
            st.markdown("### Reasoning")
            st.markdown(recommendation['reasoning'])
            
            # Risk alerts
            alerts = get_risk_alerts(fg_value, fg_label)
            if alerts:
                st.markdown("### ⚠️ Active Alerts")
                for alert in alerts:
                    st.warning(alert)
    
    # Historical strategy performance
    st.markdown("---")
    st.subheader("📈 Historical Strategy Backtest")
    
    with st.spinner("Backtesting strategy..."):
        performance = get_historical_strategy_performance(filtered_daily)
    
    st.dataframe(performance, use_container_width=True)
    
    st.info("""
    **Backtest Logic:**
    - Extreme Fear: +37.5% position size
    - Fear: +17.5% position size
    - Neutral: Standard sizing
    - Greed: -32.5% position size + tighter loss limits
    - Extreme Greed: -50% position size + strict loss limits
    """)
    
    # Sentiment Summary Table
    st.markdown("---")
    st.subheader("📋 Sentiment Regime Summary")
    
    summary = get_sentiment_summary()
    summary_df = pd.DataFrame(summary).T
    st.dataframe(summary_df, use_container_width=True)
    
    # AI Analyst Section
    st.markdown("---")
    st.header("🤖 AI Analyst — Powered by AWS Bedrock")
    
    # Initialize session state
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    
    if 'data_summary' not in st.session_state:
        with st.spinner("Generating data summary for AI..."):
            st.session_state.data_summary = statistical_summary(filtered_daily)
    
    if 'selected_question' not in st.session_state:
        st.session_state.selected_question = ""
    
    # Test connection button
    if st.button("🔌 Test AWS Bedrock Connection"):
        with st.spinner("Testing connection..."):
            success, message = test_bedrock_connection()
            if success:
                st.success(message)
            else:
                st.error(message)
    
    # Suggested questions
    st.markdown("#### Suggested Questions (Click to Ask)")
    suggested = get_suggested_questions()
    
    cols = st.columns(2)
    for idx, question in enumerate(suggested[:10]):
        col_idx = idx % 2
        with cols[col_idx]:
            if st.button(question, key=f"suggest_{idx}", use_container_width=True):
                # Directly ask the question when button is clicked
                with st.spinner("AI is thinking..."):
                    answer, updated_history = ask_bedrock(
                        question,
                        st.session_state.data_summary,
                        st.session_state.chat_history
                    )
                    st.session_state.chat_history = updated_history
                st.rerun()
    
    # Chat interface for custom questions
    st.markdown("---")
    st.markdown("#### Or Ask Your Own Question")
    
    user_question = st.text_input(
        "Type your question:",
        placeholder="e.g., What's the best strategy for today's sentiment?",
        key="user_input"
    )
    
    if st.button("💬 Ask", type="primary") and user_question:
        with st.spinner("AI is thinking..."):
            answer, updated_history = ask_bedrock(
                user_question,
                st.session_state.data_summary,
                st.session_state.chat_history
            )
            st.session_state.chat_history = updated_history
        st.rerun()
    
    # Display chat history
    if st.session_state.chat_history:
        st.markdown("---")
        st.markdown("#### Conversation History")
        
        for message in st.session_state.chat_history:
            if message['role'] == 'user':
                st.info(f"**You:** {message['content']}")
            else:
                # Display AI response with proper markdown formatting
                st.success("**AI Analyst:**")
                st.markdown(message['content'])
        
        if st.button("🗑️ Clear Chat History"):
            st.session_state.chat_history = []
            st.rerun()
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666;'>
        <p>Built with Streamlit • Powered by AWS Bedrock (Claude 3 Sonnet)</p>
        <p>Data: 211,000+ Hyperliquid trades • Bitcoin Fear & Greed Index</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
