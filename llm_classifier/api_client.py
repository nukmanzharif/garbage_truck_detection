# llm_classifier/api_client.py
import os
import requests
import json
import base64
from PIL import Image
import io
import yaml
import time
from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()

class LLMApiClient:
    """Client for sending requests to OpenAI GPT-4o vision API."""
    
    def __init__(self, config_path=None):
        """
        Initialize API client with configuration.
        
        Args:
            config_path: Path to the LLM configuration YAML file
        """
        # Use the hardcoded API key for now
        api_key = os.getenv('OPENAI_API_KEY')
        
        # Default config with the key already set
        self.config = {
            'api_key': api_key,
            'model': 'gpt-4o',  # Using GPT-4o model
            'max_tokens': 100,
            'temperature': 0.2,
            'retry_limit': 3,
            'retry_delay': 2  # seconds
        }
        
        # Initialize OpenAI client immediately with the API key
        self.client = OpenAI(api_key=api_key)
    
    def _prepare_image(self, image_path):
        """
        Prepare image for the API request.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Base64 encoded image or file path depending on usage
        """
        # Return file path directly for file-based inputs with the OpenAI client
        if os.path.exists(image_path):
            return image_path
        else:
            raise FileNotFoundError(f"Image file not found: {image_path}")
    
    def classify_image(self, image_path, prompt):
        """
        Send image to GPT-4o API for classification.
        
        Args:
            image_path: Path to the image file
            prompt: Text prompt to send with the image
            
        Returns:
            The API response
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")
        
        # Try with retries
        for attempt in range(self.config['retry_limit']):
            try:
                # Open the image file
                with open(image_path, "rb") as image_file:
                    # Create a chat completion with the OpenAI client
                    response = self.client.chat.completions.create(
                        model=self.config['model'],
                        messages=[
                            {
                                "role": "user",
                                "content": [
                                    {"type": "text", "text": prompt},
                                    {
                                        "type": "image_url", 
                                        "image_url": {"url": f"data:image/jpeg;base64,{base64.b64encode(image_file.read()).decode('utf-8')}"}
                                    }
                                ]
                            }
                        ],
                        max_tokens=self.config['max_tokens'],
                        temperature=self.config['temperature']
                    )
                
                return response
                
            except Exception as e:
                print(f"Request failed (attempt {attempt+1}/{self.config['retry_limit']}): {str(e)}")
                if attempt < self.config['retry_limit'] - 1:
                    # Wait before retrying
                    time.sleep(self.config['retry_delay'])
                else:
                    # Last attempt failed
                    raise