# Quick Setup

## 1. Install Dependencies
```bash
pip install -r requirements.txt
```

## 2. Run the App
```bash
streamlit run app.py
```

App opens at http://localhost:8501

## Optional: AWS Bedrock (for AI Analyst)
Create `.env` file:
```
AWS_ACCESS_KEY_ID=your_key
AWS_SECRET_ACCESS_KEY=your_secret
AWS_DEFAULT_REGION=us-east-1
```

See README.md for AWS setup details.

## Test Setup (Optional)
```bash
python test_setup.py
```
