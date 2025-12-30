import os
import google.generativeai as genai
from dotenv import load_dotenv

# 1. Load API Key
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

if not api_key:
    print("❌ Error: GOOGLE_API_KEY not found in environment variables.")
    exit()

# 2. Configure the library
genai.configure(api_key=api_key)

print(f"Checking models for key: {api_key[:5]}...*****")
print("-" * 40)

try:
    # 3. List all models
    count = 0
    for m in genai.list_models():
        # Filter: Only show models that can generate content (Chat/Text)
        if 'generateContent' in m.supported_generation_methods:
            print(f"✅ Available: {m.name}")
            print(f"   Name:      {m.display_name}")
            print(f"   Desc:      {m.description[:100]}...")
            print("-" * 40)
            count += 1
            
    if count == 0:
        print("⚠️ No models found. Your API key might be invalid or has no access.")
    else:
        print(f"\n🎉 Found {count} usable models.")

except Exception as e:
    print(f"\n❌ API Error: {str(e)}")
    print("Tip: Check if your API Key is correct and has 'Generative Language API' enabled in Google Cloud Console.")