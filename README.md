# 📈 Trader Sentiment Analysis Dashboard

A full-stack interactive web application that analyzes cryptocurrency trading performance against the Bitcoin Fear & Greed Index, powered by AWS Bedrock AI.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.32+-red.svg)
![AWS](https://img.shields.io/badge/AWS-Bedrock-orange.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

## 🎯 Overview

This application analyzes **211,000+ cryptocurrency trades** from Hyperliquid against Bitcoin Fear & Greed Index sentiment data to uncover actionable trading insights. It features real-time sentiment analysis, statistical validation, and an AI-powered analyst using Claude 3 Sonnet on AWS Bedrock.

## 🔑 Key Findings

- **Extreme Greed** produces highest win rate (~89%) but worst tail risk (-$358,963 single day loss)
- **Fear days** offer best average daily PnL ($5,328)
- **Extreme Fear** triggers 4x more trading activity
- **90.6%** of traders are net profitable
- **Statistically validated** via Kruskal-Wallis test (H=730, p≈0.000)

## ✨ Features

### 📊 Interactive Analytics
- Real-time Fear & Greed Index integration
- Dynamic PnL analysis by sentiment category
- Win rate and risk analysis visualizations
- Trading activity patterns
- Correlation heatmaps

### 👤 Trader Classification
- Automatic archetype detection (4 categories)
- Upload your own trades for personalized analysis
- Performance benchmarking against 32 traders

### 💡 Strategy Recommendations
- Sentiment-based position sizing guidance
- Risk management alerts
- Historical strategy backtesting
- Daily loss limit recommendations

### 🤖 AI Analyst (AWS Bedrock)
- Conversational insights powered by Claude 3 Sonnet
- Multi-turn chat with context retention
- Pre-loaded with comprehensive dataset statistics
- Suggested questions for quick insights

## 🛠️ Tech Stack

- **Frontend:** Streamlit
- **Data Processing:** Pandas, NumPy
- **Visualization:** Plotly
- **Statistics:** SciPy, Scikit-learn
- **AI:** AWS Bedrock (Claude 3 Sonnet)
- **API:** Fear & Greed Index API

## 📁 Project Structure

```
trader-sentiment-analysis/
├── app.py                        # Main Streamlit application
├── modules/
│   ├── __init__.py
│   ├── preprocessing.py          # Data loading and cleaning
│   ├── analysis.py               # Statistical analysis
│   ├── visualizations.py         # Plotly charts
│   ├── classifier.py             # Trader archetype classification
│   ├── recommendations.py        # Strategy recommendations
│   └── bedrock_agent.py          # AWS Bedrock integration
├── historical_data.csv           # 211K+ trades
├── fear_greed_index.csv          # Daily sentiment data
├── images/                       # Screenshots (add your own)
├── .env                          # AWS credentials (create from .env.example)
├── .env.example                  # Template for environment variables
├── .gitignore
├── requirements.txt
├── test_setup.py                 # Setup verification script
├── SETUP.md                      # Quick start guide
└── README.md
```

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- AWS Account with Bedrock access
- pip or conda package manager

### Installation

1. **Clone the repository**
```bash
git clone <your-repo-url>
cd trader-sentiment-analysis
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Set up AWS Bedrock** (see detailed instructions below)

4. **Configure environment variables**
```bash
cp .env.example .env
# Edit .env with your AWS credentials
```

5. **Run the application**
```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

## 🔐 AWS Bedrock Setup

### Step 1: Enable Model Access

1. Log in to [AWS Console](https://console.aws.amazon.com/)
2. Navigate to **Amazon Bedrock**
3. Go to **Model access** in the left sidebar
4. Click **Request model access**
5. Enable **Claude 3 Sonnet** (anthropic.claude-3-sonnet-20240229-v1:0)
6. Wait for approval (usually instant)

### Step 2: Create IAM User

1. Navigate to **IAM** in AWS Console
2. Click **Users** → **Add users**
3. Enter username (e.g., `bedrock-app-user`)
4. Select **Access key - Programmatic access**
5. Click **Next: Permissions**

### Step 3: Attach Permissions

1. Click **Attach existing policies directly**
2. Search for and select **AmazonBedrockFullAccess**
3. Click **Next** through to **Create user**

### Step 4: Get Access Keys

1. After user creation, click **Create access key**
2. Select **Application running outside AWS**
3. Copy the **Access Key ID** and **Secret Access Key**
4. **Important:** Save these securely - you won't see the secret again!

### Step 5: Configure Application

Create a `.env` file in the project root:

```bash
AWS_ACCESS_KEY_ID=AKIAIOSFODNN7EXAMPLE
AWS_SECRET_ACCESS_KEY=wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY
AWS_DEFAULT_REGION=us-east-1
```

**Security Note:** Never commit `.env` to version control!

## 📊 Data Format

### Historical Trades CSV
Required columns:
- `account`: Trader identifier
- `timestamp`: Trade timestamp (ISO format)
- `coin`: Cryptocurrency symbol
- `direction`: Trade direction (long/short)
- `px`: Price
- `sz`: Size/quantity
- `closedPnl`: Closed profit/loss
- `fee`: Transaction fee

### Fear & Greed Index CSV
Required columns:
- `date`: Date (YYYY-MM-DD)
- `value`: Index value (0-100)
- `value_classification`: Category (Extreme Fear, Fear, Neutral, Greed, Extreme Greed)

## 🎨 Screenshots

*Add screenshots of your application here*

### Dashboard Overview
![Dashboard](images/dashboard.png)

### AI Analyst
![AI Analyst](images/ai_analyst.png)

### Trader Archetypes
![Archetypes](images/archetypes.png)

## 🚀 Deployment

### Option 1: Streamlit Cloud (Recommended)

1. Push your code to GitHub (exclude `.env`)
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your repository
4. Add AWS credentials in **Secrets**:
```toml
AWS_ACCESS_KEY_ID = "your_key"
AWS_SECRET_ACCESS_KEY = "your_secret"
AWS_DEFAULT_REGION = "us-east-1"
```
5. Deploy!

### Option 2: AWS EC2 (Free Tier)

1. **Launch EC2 Instance**
```bash
# Choose Ubuntu Server 22.04 LTS (Free tier eligible)
# t2.micro instance type
```

2. **Connect and Install Dependencies**
```bash
ssh -i your-key.pem ubuntu@your-ec2-ip

# Update system
sudo apt update && sudo apt upgrade -y

# Install Python and pip
sudo apt install python3-pip python3-venv -y

# Clone repository
git clone <your-repo-url>
cd trader-sentiment-analysis

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

3. **Configure Environment**
```bash
# Create .env file
nano .env
# Add your AWS credentials

# Test the app
streamlit run app.py --server.port 8501
```

4. **Set Up as Service (Optional)**
```bash
# Create systemd service
sudo nano /etc/systemd/system/streamlit.service
```

Add:
```ini
[Unit]
Description=Streamlit Trader Sentiment App
After=network.target

[Service]
User=ubuntu
WorkingDirectory=/home/ubuntu/trader-sentiment-analysis
Environment="PATH=/home/ubuntu/trader-sentiment-analysis/venv/bin"
ExecStart=/home/ubuntu/trader-sentiment-analysis/venv/bin/streamlit run app.py --server.port 8501

[Install]
WantedBy=multi-user.target
```

```bash
# Enable and start service
sudo systemctl enable streamlit
sudo systemctl start streamlit

# Check status
sudo systemctl status streamlit
```

5. **Configure Security Group**
- Add inbound rule: Custom TCP, Port 8501, Source: 0.0.0.0/0
- Access at: `http://your-ec2-ip:8501`

### Option 3: Docker

```dockerfile
# Dockerfile
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

```bash
# Build and run
docker build -t trader-sentiment-app .
docker run -p 8501:8501 --env-file .env trader-sentiment-app
```

## 🧪 Testing

Test AWS Bedrock connection:
```python
from modules.bedrock_agent import test_bedrock_connection

success, message = test_bedrock_connection()
print(message)
```

## 📈 Usage Examples

### Analyze Your Own Trades

1. Prepare CSV with required columns
2. Click "Upload Custom Data" in sidebar
3. View your trader archetype classification
4. Get personalized strategy recommendations

### Ask the AI Analyst

Example questions:
- "When is the best sentiment to increase position size?"
- "What is the risk profile of Greed days?"
- "How should I adjust strategy given today's sentiment?"
- "Why does Extreme Greed have high win rate but worst tail risk?"

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License - see below for details:

```
MIT License

Copyright (c) 2026

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## 🐛 Troubleshooting

### AWS Bedrock Connection Issues

**Error: "Could not connect to Bedrock"**
- Verify AWS credentials in `.env`
- Check IAM user has `AmazonBedrockFullAccess` policy
- Ensure Claude 3 Sonnet is enabled in Bedrock console
- Verify region is set to `us-east-1` (or your enabled region)

**Error: "Model not found"**
- Request model access in AWS Bedrock console
- Wait for approval (usually instant for Claude models)

### Data Loading Issues

**Error: "File not found"**
- Ensure CSV files are in `data/` directory
- Check file names match exactly: `historical_data.csv` and `fear_greed_index.csv`

**Error: "Missing columns"**
- Verify CSV has all required columns
- Check column names match exactly (case-sensitive)

### Performance Issues

- Use date range filters to reduce data size
- Clear browser cache if visualizations lag
- For large datasets (>500K trades), consider data sampling

## 📞 Support

For issues and questions:
- Open an issue on GitHub
- Check existing issues for solutions
- Review AWS Bedrock documentation

## 🙏 Acknowledgments

- Data source: Hyperliquid trading platform
- Sentiment data: Alternative.me Fear & Greed Index API
- AI: AWS Bedrock (Claude 3 Sonnet by Anthropic)
- Framework: Streamlit

---

**Built with ❤️ for traders who want data-driven insights**
