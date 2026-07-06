import os
from google import genai

# We check both GEMINI_API_KEY and GOOGLE_API_KEY
api_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")

client = genai.Client(api_key=api_key)

try:
    for m in client.models.list():
        # Print model name and supported methods
        methods = getattr(m, 'supported_methods', [])
        # We check if embedContent is in supported methods
        if any('embed' in method.lower() for method in methods):
            print(f"Embedding Model: {m.name} | Methods: {methods}")
        else:
            print(f"Other Model: {m.name}")
except Exception as e:
    print(f"Error: {e}")
