import os
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings

load_dotenv()

models_to_try = [
    "models/gemini-embedding-2",
    "models/gemini-embedding-001",
    "gemini-embedding-2"
]

print("Starting Embeddings Test...")
for m in models_to_try:
    try:
        print(f"Trying model: '{m}' ...")
        embeddings = GoogleGenerativeAIEmbeddings(model=m)
        res = embeddings.embed_query("Hello world")
        print(f"  SUCCESS! Vector length: {len(res)}")
        break
    except Exception as e:
        print(f"  FAILED: {e}\n")
