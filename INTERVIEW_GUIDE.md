# 📋 Interview Guide: Trader Sentiment Analysis Project

## Project Overview

**Project Name:** Trader Sentiment Analysis Dashboard  
**Tech Stack:** Python, Streamlit, AWS Bedrock, Pandas, Plotly, SciPy  
**Dataset:** 211,000+ cryptocurrency trades from Hyperliquid + Bitcoin Fear & Greed Index  
**Purpose:** Analyze trading performance against market sentiment to generate actionable insights

---

## 🎯 What Does This Project Do?

This is a **full-stack web application** that:
1. Analyzes 211K+ crypto trades against Bitcoin Fear & Greed Index
2. Identifies which market sentiment conditions produce best/worst results
3. Classifies traders into 4 archetypes based on behavior
4. Provides AI-powered trading recommendations using AWS Bedrock
5. Visualizes patterns through 6 interactive Plotly charts

**Key Finding:** Extreme Greed has 89% win rate BUT worst tail risk (-$358K loss). Fear days offer best risk-reward.

---

## 🔧 Technical Architecture

### Frontend
- **Streamlit** - Web framework for rapid dashboard development
- **Plotly** - Interactive JavaScript-based visualizations
- **Custom CSS** - Styled chat interface and responsive layout

### Backend
- **Pandas** - Data processing and analysis (211K+ rows)
- **NumPy** - Numerical computations
- **SciPy** - Statistical tests (Kruskal-Wallis)
- **Scikit-learn** - Trader classification algorithms

### AI Integration
- **AWS Bedrock** - Serverless AI service
- **Amazon Nova Pro** - LLM for conversational insights
- **Boto3** - AWS SDK for Python

### Data Pipeline
```
CSV Files → Pandas → Clean/Merge → Daily Metrics → Analysis → Visualization → Streamlit
```

---

## 💡 How to Use the Application

### 1. Installation
```bash
pip install -r requirements.txt
```

### 2. Run the App
```bash
streamlit run app.py
```

### 3. Features to Explore

**Live Sentiment Banner**
- Shows current Bitcoin Fear & Greed Index
- Displays today's trading signal (BUY/CAUTION/HIGH RISK)

**5 Analysis Tabs**
- 📊 PnL Analysis - Average vs median profits by sentiment
- 🎯 Win Rate - Success rates across sentiment categories
- ⚠️ Risk Analysis - Worst single-day losses
- 📈 Trading Activity - Trade frequency patterns
- 👤 Trader Archetypes - 4 trader types classification

**Strategy Recommendations**
- Position sizing guidance based on current sentiment
- Historical backtest performance
- Risk alerts for dangerous conditions

**AI Analyst (AWS Bedrock)**
- Ask questions about the data
- Get insights on trading strategies
- Conversational interface with history

**File Upload**
- Upload your own trades CSV
- Get classified into one of 4 archetypes
- Compare against 32 reference traders

---

## 📚 Common Interview Questions & Answers

### General Questions

**Q1: What is this project about?**

**A:** This is a full-stack data analytics web application that analyzes cryptocurrency trading performance against market sentiment. It processes 211,000+ trades to identify which Fear & Greed Index conditions produce the best risk-adjusted returns, and provides AI-powered recommendations for position sizing and risk management.

---

**Q2: Why did you build this?**

**A:** To solve a real trading problem: traders often don't know when to increase or decrease position sizes based on market sentiment. This app quantifies the relationship between sentiment and performance, revealing that Fear days offer the best risk-reward despite lower win rates, while Extreme Greed has high win rates but catastrophic tail risk.

---

**Q3: What was the biggest challenge?**

**A:** Handling the timestamp parsing from scientific notation (1.73E+12) and ensuring the data pipeline could process 211K+ rows efficiently. I implemented smart caching with `@st.cache_data` and optimized pandas operations to keep the app responsive.

---

### Technical Questions

**Q4: Explain your data pipeline.**

**A:** 
1. **Load:** Read CSV files with pandas (historical_data.csv + fear_greed_index.csv)
2. **Clean:** Normalize column names, parse timestamps (handle milliseconds), convert scientific notation
3. **Merge:** Join trades with sentiment data on date
4. **Transform:** Calculate daily metrics (PnL, win rate, trade count) per trader
5. **Analyze:** Group by sentiment, run statistical tests (Kruskal-Wallis)
6. **Visualize:** Generate Plotly charts
7. **Cache:** Store processed data in Streamlit session state

---

**Q5: How did you integrate AWS Bedrock?**

**A:**
1. **Authentication:** Used boto3 with credentials from `.env` file
2. **Model Selection:** Amazon Nova Pro (no approval form needed)
3. **Message Formatting:** Structured messages as arrays with text objects per Nova API spec
4. **Context Injection:** Built system prompt with dataset statistics
5. **Conversation Management:** Maintained chat history in `st.session_state`
6. **Error Handling:** Graceful degradation - app works without AWS

```python
client = boto3.client('bedrock-runtime', region_name='us-east-1')
response = client.invoke_model(
    modelId="us.amazon.nova-pro-v1:0",
    body=json.dumps({
        "messages": formatted_messages,
        "system": [{"text": system_context}],
        "inferenceConfig": {"maxTokens": 2000}
    })
)
```

---

**Q6: How do you classify traders into archetypes?**

**A:** Using a **2x2 matrix** based on median thresholds:
- **X-axis:** Trade frequency (high vs low)
- **Y-axis:** Profitability (winner vs loser)

```python
median_trades = df['trade_count'].median()
median_pnl = df['total_pnl'].median()

if trades >= median_trades and pnl >= median_pnl:
    return "High-Frequency Winner"
# ... 3 other archetypes
```

This creates 4 quadrants visible in the scatter plot.

---

**Q7: What statistical tests did you use and why?**

**A:** **Kruskal-Wallis H-test** because:
- Non-parametric (doesn't assume normal distribution)
- Compares 3+ groups (5 sentiment categories)
- Robust to outliers (important for financial data)

Result: H=730, p≈0.000 → Sentiment has statistically significant impact on PnL.

---

**Q8: How did you optimize performance for 211K+ rows?**

**A:**
1. **Caching:** `@st.cache_data` on data loading (loads once, reuses)
2. **Vectorization:** Used pandas vectorized operations instead of loops
3. **Lazy Loading:** Charts render only when tabs are clicked
4. **Efficient Groupby:** Pre-computed daily metrics instead of real-time aggregation
5. **Filtered Processing:** Applied filters before heavy computations

---

**Q9: Explain your visualization choices.**

**A:**
- **Plotly over Matplotlib:** Interactive (zoom, pan, hover), better UX
- **Grouped Bar Chart:** Compare avg vs median PnL (shows distribution skew)
- **Horizontal Bar:** Win rates easier to compare horizontally
- **Scatter Plot:** Shows 2D relationship (frequency vs profitability)
- **Heatmap:** Correlation matrix reveals hidden relationships
- **Color Coding:** Red=Fear, Yellow=Neutral, Green=Greed (intuitive)

---

**Q10: How do you handle missing data?**

**A:**
```python
# Numeric columns - fill with 0
df['closedPnl'] = pd.to_numeric(df['closedPnl'], errors='coerce').fillna(0)

# Sentiment data - forward fill (use previous day's sentiment)
merged['fg_value'] = merged['fg_value'].ffill()

# Critical columns - drop rows
df = df.dropna(subset=['account', 'timestamp', 'coin'])
```

---

### AWS & Cloud Questions

**Q11: Why AWS Bedrock instead of OpenAI API?**

**A:**
- **Serverless:** No infrastructure management
- **AWS Integration:** Works with existing AWS services
- **Multiple Models:** Can switch between Claude, Nova, Llama
- **Enterprise Ready:** Built-in security, compliance, IAM
- **Cost Effective:** Pay per token, no monthly fees

---

**Q12: How do you secure AWS credentials?**

**A:**
1. **Never commit:** `.env` in `.gitignore`
2. **Environment variables:** Load with `python-dotenv`
3. **Least privilege:** IAM user with only Bedrock access
4. **Rotation:** Can create new keys anytime
5. **Deployment:** Use Streamlit secrets or IAM roles (EC2)

---

**Q13: What if AWS Bedrock fails?**

**A:** **Graceful degradation:**
```python
try:
    answer = ask_bedrock(question)
except Exception as e:
    return "AWS Bedrock unavailable. App continues without AI features."
```
All other features (charts, analysis, recommendations) work independently.

---

### Data Science Questions

**Q14: What insights did you discover?**

**A:** **Key Findings:**
1. **Paradox:** Extreme Greed = 89% win rate BUT -$358K worst loss
2. **Best Risk-Reward:** Fear days = $5,328 avg daily PnL
3. **Activity Spike:** Extreme Fear = 4x more trades (panic creates opportunity)
4. **Trader Success:** 90.6% are net profitable
5. **Archetypes:** High-frequency winners dominate in Fear conditions

**Actionable Insight:** Increase positions during Fear, reduce during Greed (counterintuitive but data-proven).

---

**Q15: How would you improve this project?**

**A:**
1. **Real-time Data:** WebSocket connection to live exchange data
2. **Backtesting Engine:** Simulate strategy performance with historical data
3. **Email Alerts:** Notify when sentiment reaches extreme levels
4. **Multi-Exchange:** Aggregate data from Binance, Coinbase, etc.
5. **Machine Learning:** Predict next-day PnL using sentiment + technical indicators
6. **User Authentication:** Save personal trades and track performance over time
7. **PDF Reports:** Export analysis as downloadable reports

---

### Behavioral Questions

**Q16: Describe a technical challenge you faced.**

**A:** **Challenge:** AWS Bedrock message format kept failing with "invalid request" errors.

**Solution:**
1. Read AWS Nova API documentation carefully
2. Discovered messages need `content: [{"text": "..."}]` format (array of objects)
3. Refactored message formatting function
4. Added proper error handling with helpful messages
5. Tested with simple requests first, then complex conversations

**Learning:** Always read API docs thoroughly and test incrementally.

---

**Q17: How do you ensure code quality?**

**A:**
- **Modular Design:** 7 separate modules (preprocessing, analysis, visualizations, etc.)
- **Type Hints:** All functions have type annotations
- **Docstrings:** Every function documented
- **Error Handling:** Try-catch blocks with user-friendly messages
- **DRY Principle:** No code duplication
- **Testing:** `test_setup.py` validates all components

---

**Q18: How would you deploy this to production?**

**A:** **3 Options:**

**Option 1: Streamlit Cloud (Easiest)**
- Push to GitHub
- Connect at share.streamlit.io
- Add AWS secrets
- Auto-deploys on push

**Option 2: AWS EC2**
- Launch t2.micro (free tier)
- Install dependencies
- Run as systemd service
- Use Nginx reverse proxy
- SSL with Let's Encrypt

**Option 3: Docker + ECS**
- Containerize with Dockerfile
- Push to ECR
- Deploy on ECS Fargate
- Auto-scaling enabled

---

### Scenario Questions

**Q19: A user reports the app is slow. How do you debug?**

**A:**
1. **Check Data Size:** 211K rows should be fine, but verify with `len(df)`
2. **Profile Code:** Use `@st.cache_data` decorator, check cache hits
3. **Monitor Network:** Check API calls (Fear & Greed, AWS Bedrock)
4. **Browser DevTools:** Check for JavaScript errors, large DOM
5. **Streamlit Profiler:** Use `streamlit run app.py --logger.level=debug`
6. **Optimize:** Add pagination, lazy loading, reduce chart complexity

---

**Q20: How would you add user authentication?**

**A:**
```python
# Option 1: Simple password
import streamlit_authenticator as stauth

authenticator = stauth.Authenticate(
    credentials, cookie_name, key, cookie_expiry_days
)
name, authentication_status, username = authenticator.login()

# Option 2: OAuth (Google, GitHub)
# Use streamlit-oauth library

# Option 3: AWS Cognito
# Integrate with boto3 for enterprise auth
```

Store user-specific data in database (PostgreSQL/DynamoDB) keyed by username.

---

## 🎓 Key Takeaways for Interviews

### What You Built
✅ Full-stack data analytics web application  
✅ Real-time API integration (Fear & Greed Index)  
✅ AI-powered insights (AWS Bedrock)  
✅ Interactive visualizations (Plotly)  
✅ Statistical analysis (SciPy)  
✅ Cloud deployment ready  

### Skills Demonstrated
✅ **Python:** Pandas, NumPy, SciPy, Boto3  
✅ **Web Development:** Streamlit, HTML/CSS  
✅ **Cloud:** AWS Bedrock, IAM, API integration  
✅ **Data Science:** Statistical testing, classification, visualization  
✅ **Software Engineering:** Modular design, error handling, documentation  
✅ **DevOps:** Environment management, deployment strategies  

### Business Impact
✅ **Quantified Risk:** Identified $358K tail risk in Greed conditions  
✅ **Actionable Insights:** Clear position sizing recommendations  
✅ **Data-Driven:** Replaced gut feeling with statistical evidence  
✅ **Scalable:** Can handle millions of trades with minor optimizations  

---

## 📊 Project Metrics

- **Lines of Code:** ~2,500+
- **Data Points:** 211,224 trades
- **Traders Analyzed:** 32
- **Visualizations:** 6 interactive charts
- **AI Model:** Amazon Nova Pro (200K context)
- **Statistical Tests:** Kruskal-Wallis (H=730, p<0.001)
- **Modules:** 7 Python files
- **Dependencies:** 9 packages

---

## 🚀 Demo Script (2 minutes)

**Opening:**
"I built a full-stack trading analytics dashboard that analyzes 211,000 cryptocurrency trades against market sentiment to generate AI-powered recommendations."

**Live Demo:**
1. **Show Live Sentiment:** "Here's today's Fear & Greed Index with real-time signal"
2. **Navigate Tabs:** "5 analysis views - PnL, win rate, risk, activity, archetypes"
3. **Highlight Insight:** "Notice: Extreme Greed has 89% win rate but -$358K worst loss"
4. **Show AI:** "Ask the AI analyst powered by AWS Bedrock"
5. **Upload Feature:** "Upload your trades, get classified into archetype"

**Technical Highlight:**
"Built with Streamlit for rapid development, Plotly for interactive charts, AWS Bedrock for AI, and statistical validation using Kruskal-Wallis test."

**Business Value:**
"This helps traders make data-driven decisions on position sizing, potentially avoiding catastrophic losses during Greed periods."

---

## 🔗 Resources

- **GitHub:** [Your repo URL]
- **Live Demo:** [Streamlit Cloud URL]
- **Documentation:** See README.md
- **Setup Guide:** See SETUP.md

---

## 💬 Talking Points

**When asked "Tell me about a project":**
- Start with the problem (traders don't know when to size positions)
- Explain the solution (data-driven sentiment analysis)
- Highlight the tech (Streamlit, AWS Bedrock, Plotly)
- Share the impact (identified $358K tail risk)

**When asked "What did you learn":**
- AWS Bedrock API integration and message formatting
- Handling large datasets efficiently with pandas
- Statistical testing for financial data
- Building production-ready dashboards with Streamlit

**When asked "What would you do differently":**
- Add real-time data streaming
- Implement machine learning predictions
- Build backtesting engine
- Add user authentication and personalization

---

**Good luck with your interviews! 🎉**
