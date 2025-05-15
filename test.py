import os
import base64
from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()

api_key = os.environ.get("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)

image_path = "outputs/visualizations/images/trucks/truck_1_frame_210_00-00-07.00.jpg"
with open(image_path, "rb") as image_file:
    base64_image = base64.b64encode(image_file.read()).decode('utf-8')

try:
    response = client.chat.completions.create(
        model="gpt-4.1",  # or "gpt-4.1" or "gpt-4o"
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Is this a garbage truck? Answer Yes or No."},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}
                    }
                ]
            }
        ],
        max_tokens=50,
    )
    print(response.choices[0].message.content)
except Exception as e:
    print("Error:", e)