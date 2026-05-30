"""
Test script to verify the application setup.
Run this before starting the Streamlit app to catch any issues.
"""

import sys
import os

def test_imports():
    """Test that all required packages are installed."""
    print("Testing imports...")
    try:
        import streamlit
        import pandas
        import numpy
        import plotly
        import requests
        import boto3
        import dotenv
        import sklearn
        import scipy
        print("✅ All required packages installed")
        return True
    except ImportError as e:
        print(f"❌ Missing package: {e}")
        print("Run: pip install -r requirements.txt")
        return False


def test_data_files():
    """Test that data files exist."""
    print("\nTesting data files...")
    files = [
        'historical_data.csv',
        'fear_greed_index.csv'
    ]
    
    all_exist = True
    for file in files:
        if os.path.exists(file):
            print(f"✅ Found: {file}")
        else:
            print(f"❌ Missing: {file}")
            all_exist = False
    
    return all_exist


def test_modules():
    """Test that custom modules can be imported."""
    print("\nTesting custom modules...")
    try:
        from modules import preprocessing
        from modules import analysis
        from modules import visualizations
        from modules import classifier
        from modules import recommendations
        from modules import bedrock_agent
        print("✅ All custom modules imported successfully")
        return True
    except ImportError as e:
        print(f"❌ Module import error: {e}")
        return False


def test_data_loading():
    """Test that data can be loaded."""
    print("\nTesting data loading...")
    try:
        from modules.preprocessing import load_data, compute_daily_metrics
        df = load_data(
            'historical_data.csv',
            'fear_greed_index.csv'
        )
        daily_metrics = compute_daily_metrics(df)
        print(f"✅ Loaded {len(df):,} trades")
        print(f"✅ Computed {len(daily_metrics):,} daily metrics")
        print(f"✅ Date range: {df['date'].min()} to {df['date'].max()}")
        print(f"✅ Unique traders: {df['account'].nunique()}")
        return True
    except Exception as e:
        print(f"❌ Data loading error: {e}")
        return False


def test_aws_config():
    """Test AWS configuration (optional)."""
    print("\nTesting AWS configuration...")
    
    if not os.path.exists('.env'):
        print("⚠️  No .env file found (AWS Bedrock features will be unavailable)")
        print("   Create .env from .env.example to enable AI Analyst")
        return None
    
    from dotenv import load_dotenv
    load_dotenv()
    
    aws_key = os.getenv('AWS_ACCESS_KEY_ID')
    aws_secret = os.getenv('AWS_SECRET_ACCESS_KEY')
    aws_region = os.getenv('AWS_DEFAULT_REGION')
    
    if aws_key and aws_secret:
        print(f"✅ AWS credentials found")
        print(f"✅ Region: {aws_region}")
        
        # Test connection
        try:
            from modules.bedrock_agent import test_bedrock_connection
            success, message = test_bedrock_connection()
            if success:
                print(f"✅ {message}")
                return True
            else:
                print(f"⚠️  {message}")
                return False
        except Exception as e:
            print(f"⚠️  Could not test Bedrock connection: {e}")
            return False
    else:
        print("⚠️  AWS credentials not configured in .env")
        print("   AI Analyst features will be unavailable")
        return None


def test_api_access():
    """Test Fear & Greed API access."""
    print("\nTesting Fear & Greed API...")
    try:
        import requests
        response = requests.get("https://api.alternative.me/fng/?limit=1", timeout=5)
        data = response.json()
        value = data['data'][0]['value']
        print(f"✅ API accessible - Current F&G Index: {value}")
        return True
    except Exception as e:
        print(f"⚠️  API not accessible: {e}")
        print("   Live sentiment features will be unavailable")
        return False


def main():
    """Run all tests."""
    print("=" * 60)
    print("TRADER SENTIMENT ANALYSIS - SETUP TEST")
    print("=" * 60)
    
    results = {
        "Imports": test_imports(),
        "Data Files": test_data_files(),
        "Modules": test_modules(),
        "Data Loading": test_data_loading(),
        "API Access": test_api_access(),
        "AWS Config": test_aws_config()
    }
    
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    for test_name, result in results.items():
        if result is True:
            status = "✅ PASS"
        elif result is False:
            status = "❌ FAIL"
        else:
            status = "⚠️  OPTIONAL"
        print(f"{test_name:.<40} {status}")
    
    # Overall status
    critical_tests = ["Imports", "Data Files", "Modules", "Data Loading"]
    critical_passed = all(results[test] for test in critical_tests)
    
    print("\n" + "=" * 60)
    if critical_passed:
        print("✅ ALL CRITICAL TESTS PASSED")
        print("\nYou can now run the app:")
        print("  streamlit run app.py")
        
        if results["AWS Config"] is not True:
            print("\nNote: AWS Bedrock not configured (AI Analyst unavailable)")
            print("      App will work without it!")
    else:
        print("❌ SOME CRITICAL TESTS FAILED")
        print("\nPlease fix the issues above before running the app.")
        sys.exit(1)
    
    print("=" * 60)


if __name__ == "__main__":
    main()
