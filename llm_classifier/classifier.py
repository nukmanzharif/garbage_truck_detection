# llm_classifier/classifier.py
import os
import pandas as pd
import yaml
import re
import json
from .api_client import LLMApiClient

class GarbageTruckClassifier:
    """
    Classifier for identifying garbage trucks in images using GPT-4o vision API.
    """
    
    def __init__(self, config_path=None):
        """
        Initialize garbage truck classifier.
        
        Args:
            config_path: Path to the LLM configuration YAML file
        """
        self.config_path = config_path
        self.api_client = LLMApiClient(config_path)
        
        # Load config
        self.config = {
            'prompt_template': "You are a waste management vehicle identification expert. Look at this image and determine if this is a garbage truck (waste collection vehicle). Answer with just 'Yes' or 'No', followed by a brief explanation of why you made that determination.",
            'confidence_threshold': 0.8,
            'result_dir': 'outputs/results'
        }
        
        # Load custom config if available
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r') as f:
                custom_config = yaml.safe_load(f)
                if custom_config and 'classifier' in custom_config:
                    self.config.update(custom_config['classifier'])
        
        # Create results directory
        os.makedirs(self.config['result_dir'], exist_ok=True)
    
    def _parse_response(self, response):
        """
        Parse the GPT-4o response to determine if the image shows a garbage truck.
        
        Args:
            response: The API response from OpenAI
            
        Returns:
            is_garbage_truck (bool): True if classified as garbage truck, False otherwise
            confidence (float): Confidence score between 0-1
            raw_text (str): The raw response text
        """
        # Extract response content
        raw_text = response.choices[0].message.content.strip()
        
        # Look for yes/no in the response
        is_garbage_truck = False
        confidence = 0.9  # Default high confidence for direct yes/no answers
        
        # Simple pattern matching
        if re.match(r'^yes', raw_text.lower()):
            is_garbage_truck = True
        elif re.match(r'^no', raw_text.lower()):
            is_garbage_truck = False
        else:
            # If not a clear yes/no, do more detailed analysis
            yes_patterns = ['garbage truck', 'waste collection', 'trash', 'waste management']
            no_patterns = ['not a garbage truck', 'not waste collection', 'regular truck']
            
            text_lower = raw_text.lower()
            
            # Check for positive indicators
            yes_matches = sum(1 for pattern in yes_patterns if pattern in text_lower)
            no_matches = sum(1 for pattern in no_patterns if pattern in text_lower)
            
            if yes_matches > no_matches:
                is_garbage_truck = True
                confidence = 0.7  # Lower confidence for inferred results
            else:
                is_garbage_truck = False
                confidence = 0.7
        
        return is_garbage_truck, confidence, raw_text
    
    def classify_truck_image(self, image_path):
        """
        Classify a single truck image.

        Args:
            image_path: Path to the truck image
            
        Returns:
            Dictionary with classification results
        """
        if not os.path.exists(image_path):
            return {"error": f"Image file not found: {image_path}"}

        try:
            # Send to GPT-4o API
            response = self.api_client.classify_image(
                image_path, 
                self.config['prompt_template']
            )
            
            # Parse response
            is_garbage_truck, confidence, raw_text = self._parse_response(response)
            
            # Calculate a unique identifier from the image path
            image_id = os.path.basename(image_path).split('.')[0]
            
            # Save detailed results
            result_path = os.path.join(self.config['result_dir'], f"{image_id}_result.json")
            with open(result_path, 'w') as f:
                json.dump({
                    "image_path": image_path,
                    "is_garbage_truck": is_garbage_truck,
                    "confidence": confidence,
                    "raw_response": raw_text,
                    "prompt": self.config['prompt_template']
                }, f, indent=2)
            
            # Use the already annotated image that was created during video processing
            # The annotated image path is included in the extracted_frames.csv
            annotated_image_path = image_path.replace('/trucks/', '/annotated_trucks/').replace('truck_', 'annotated_truck_')
            
            return {
                "image_path": image_path,
                "is_garbage_truck": is_garbage_truck,
                "confidence": confidence,
                "raw_response": raw_text,
                "result_file": result_path,
                "annotated_image": annotated_image_path if os.path.exists(annotated_image_path) else None
            }
            
        except Exception as e:
            return {"error": str(e), "image_path": image_path}
    
    def classify_truck_images(self, images_df):
        """
        Classify multiple truck images from a DataFrame.
        
        Args:
            images_df: DataFrame with truck_id and image_path columns
            
        Returns:
            DataFrame with original data plus classification results
        """
        if 'image_path' not in images_df.columns:
            raise ValueError("DataFrame must contain 'image_path' column")
        
        # Create copy to avoid modifying the original
        result_df = images_df.copy()
        
        # Add classification columns
        result_df['is_garbage_truck'] = False
        result_df['classification_confidence'] = 0.0
        result_df['llm_response'] = ''
        result_df['result_file'] = ''
        
        # Process each image
        for idx, row in result_df.iterrows():
            try:
                # Classify image
                result = self.classify_truck_image(row['image_path'])
                
                # Update dataframe with results
                if 'error' not in result:
                    result_df.at[idx, 'is_garbage_truck'] = result['is_garbage_truck']
                    result_df.at[idx, 'classification_confidence'] = result['confidence']
                    result_df.at[idx, 'llm_response'] = result['raw_response']
                    result_df.at[idx, 'result_file'] = result.get('result_file', '')
                    
                else:
                    result_df.at[idx, 'llm_response'] = f"Error: {result['error']}"
                    print(f"  Error: {result['error']}")
                    
            except Exception as e:
                result_df.at[idx, 'llm_response'] = f"Exception: {str(e)}"
                print(f"  Exception: {str(e)}")
        
        # Save the complete results
        results_path = os.path.join(self.config['result_dir'], 'classification_results.csv')
        result_df.to_csv(results_path, index=False)
        print(f"Classification results saved to {results_path}")
        
        return result_df