import sys, os
sys.path.insert(0, '/workspaces/trading-rag')
os.chdir('/workspaces/trading-rag')

from dotenv import load_dotenv
load_dotenv('/workspaces/trading-rag/.env')

from app.core.config import get_settings
settings = get_settings()
print("Provider:", settings.llm_provider)
print("Model:", settings.llm_model)
print("Key starts with:", settings.anthropic_api_key[:15])

import anthropic
client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
try:
    resp = client.messages.create(
        model=settings.llm_model,
        max_tokens=100,
        messages=[{"role": "user", "content": "Say hello."}]
    )
    print("SUCCESS:", resp.content[0].text)
except Exception as e:
    print("FULL ERROR:", str(e))