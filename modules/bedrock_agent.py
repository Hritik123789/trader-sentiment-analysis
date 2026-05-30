"""
AWS Bedrock integration for AI-powered analysis.
Uses Claude 3 Sonnet on AWS Bedrock for conversational insights.

SETUP INSTRUCTIONS:
1. AWS Console → Bedrock → Model access → Enable Claude 3 Sonnet
2. IAM → Create user → Attach AmazonBedrockFullAccess policy
3. Create access key → Add to .env file:
   AWS_ACCESS_KEY_ID=your_key_here
   AWS_SECRET_ACCESS_KEY=your_secret_here
   AWS_DEFAULT_REGION=us-east-1
"""

import boto3
import json
import os
from typing import List, Dict, Tuple
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


def get_bedrock_client():
    """
    Initialize and return AWS Bedrock client.
    
    Returns:
        boto3 Bedrock Runtime client
        
    Raises:
        Exception if AWS credentials not configured
    """
    try:
        # Try Streamlit secrets first (for deployment)
        try:
            import streamlit as st
            if hasattr(st, 'secrets') and 'AWS_ACCESS_KEY_ID' in st.secrets:
                aws_access_key = st.secrets['AWS_ACCESS_KEY_ID']
                aws_secret_key = st.secrets['AWS_SECRET_ACCESS_KEY']
                aws_region = st.secrets.get('AWS_DEFAULT_REGION', 'us-east-1')
            else:
                raise KeyError("Not in Streamlit Cloud")
        except (ImportError, KeyError):
            # Fall back to environment variables (for local development)
            aws_access_key = os.getenv('AWS_ACCESS_KEY_ID')
            aws_secret_key = os.getenv('AWS_SECRET_ACCESS_KEY')
            aws_region = os.getenv('AWS_DEFAULT_REGION', 'us-east-1')
        
        if aws_access_key and aws_secret_key:
            client = boto3.client(
                'bedrock-runtime',
                region_name=aws_region,
                aws_access_key_id=aws_access_key,
                aws_secret_access_key=aws_secret_key
            )
        else:
            # Fall back to default credentials (IAM role, AWS CLI config, etc.)
            client = boto3.client('bedrock-runtime', region_name=aws_region)
        
        return client
    
    except Exception as e:
        raise Exception(f"Failed to initialize AWS Bedrock client: {str(e)}")


def build_context(data_summary: str) -> str:
    """
    Build system context prompt with data summary.
    
    Args:
        data_summary: Statistical summary of the dataset
        
    Returns:
        Formatted system prompt string
    """
    system_prompt = f"""You are an expert quantitative trading analyst specializing in cryptocurrency markets and behavioral finance. You have deep expertise in:

- Statistical analysis of trading data
- Market sentiment analysis (Fear & Greed Index)
- Risk management and position sizing
- Trader psychology and behavioral patterns
- Quantitative strategy development

You are analyzing a comprehensive dataset of 211,000+ cryptocurrency trades from Hyperliquid, merged with Bitcoin Fear & Greed Index data. Your role is to provide actionable insights, identify patterns, and recommend strategies based on this data.

KEY FINDINGS FROM THE DATASET:
{data_summary}

GUIDELINES FOR YOUR RESPONSES:
- Be direct and actionable - traders need clear guidance
- Support claims with specific statistics from the data
- Consider both profitability AND risk when making recommendations
- Acknowledge the paradox: Extreme Greed has highest win rate but worst tail risk
- Explain the "why" behind patterns (market psychology, liquidity, volatility)
- When discussing risk, be specific about downside scenarios
- Tailor advice to different trader archetypes when relevant

RESPONSE STYLE:
- Professional but conversational
- Use bullet points for clarity
- Include specific numbers and percentages
- Highlight key takeaways
- Warn about risks explicitly

Remember: You're advising real traders with real capital at risk. Be thorough, honest, and practical."""

    return system_prompt


def ask_bedrock(
    question: str, 
    data_summary: str, 
    conversation_history: List[Dict] = None
) -> Tuple[str, List[Dict]]:
    """
    Send question to AWS Bedrock and get response.
    
    Args:
        question: User's question
        data_summary: Statistical summary for context
        conversation_history: Previous conversation turns
        
    Returns:
        Tuple of (answer text, updated conversation history)
    """
    if conversation_history is None:
        conversation_history = []
    
    try:
        client = get_bedrock_client()
        
        # Build system context
        system_context = build_context(data_summary)
        
        # Format messages for Amazon Nova (needs specific structure)
        formatted_messages = []
        for msg in conversation_history:
            formatted_messages.append({
                "role": msg["role"],
                "content": [{"text": msg["content"]}]
            })
        
        # Add current question
        formatted_messages.append({
            "role": "user",
            "content": [{"text": question}]
        })
        
        # Prepare request body for Amazon Nova Pro
        request_body = {
            "messages": formatted_messages,
            "system": [{"text": system_context}],
            "inferenceConfig": {
                "maxTokens": 2000,
                "temperature": 0.7
            }
        }
        
        # Call Bedrock - using Amazon Nova Pro (no approval needed)
        response = client.invoke_model(
            modelId="us.amazon.nova-pro-v1:0",
            body=json.dumps(request_body)
        )
        
        # Parse response from Amazon Nova
        response_body = json.loads(response['body'].read())
        answer = response_body['output']['message']['content'][0]['text']
        
        # Add to conversation history (in simple format for storage)
        conversation_history.append({
            "role": "user",
            "content": question
        })
        conversation_history.append({
            "role": "assistant",
            "content": answer
        })
        
        return answer, conversation_history
    
    except Exception as e:
        error_message = f"Error communicating with AWS Bedrock: {str(e)}"
        
        # Provide helpful error messages
        if "credentials" in str(e).lower():
            error_message += "\n\n💡 **Setup Required:**\n"
            error_message += "1. Add AWS credentials to `.env` file:\n"
            error_message += "   ```\n"
            error_message += "   AWS_ACCESS_KEY_ID=your_key\n"
            error_message += "   AWS_SECRET_ACCESS_KEY=your_secret\n"
            error_message += "   AWS_DEFAULT_REGION=us-east-1\n"
            error_message += "   ```\n"
            error_message += "2. Ensure Claude 3 Sonnet is enabled in AWS Bedrock console"
        
        elif "model" in str(e).lower():
            error_message += "\n\n💡 **Model Access Required:**\n"
            error_message += "Enable Claude 3 Sonnet in AWS Console:\n"
            error_message += "AWS Console → Bedrock → Model access → Request access"
        
        return error_message, conversation_history


def get_suggested_questions() -> List[str]:
    """
    Get list of suggested questions users can ask.
    
    Returns:
        List of suggested question strings
    """
    return [
        "When is the best sentiment to increase position size?",
        "What is the risk profile of Greed days?",
        "Which trader archetype performs best in Fear conditions?",
        "How should I adjust strategy given today's sentiment?",
        "What are the key differences between winning and losing traders?",
        "Why does Extreme Greed have the highest win rate but worst tail risk?",
        "How much should I scale positions during Extreme Fear?",
        "What daily loss limits should I set during Greed periods?",
        "How do high-frequency and low-frequency winners differ?",
        "What causes the 4x spike in trading activity during Extreme Fear?"
    ]


def test_bedrock_connection() -> Tuple[bool, str]:
    """
    Test AWS Bedrock connection and model access.
    
    Returns:
        Tuple of (success boolean, message string)
    """
    try:
        client = get_bedrock_client()
        
        # Try a simple test request with Amazon Nova Pro
        test_body = {
            "messages": [
                {
                    "role": "user",
                    "content": [{"text": "Respond with 'Connection successful' if you receive this."}]
                }
            ],
            "inferenceConfig": {
                "maxTokens": 100
            }
        }
        
        response = client.invoke_model(
            modelId="us.amazon.nova-pro-v1:0",
            body=json.dumps(test_body)
        )
        
        return True, "✅ AWS Bedrock connection successful!"
    
    except Exception as e:
        return False, f"❌ Connection failed: {str(e)}"
