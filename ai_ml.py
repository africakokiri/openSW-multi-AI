import requests
import json
from dotenv import load_dotenv
import os

load_dotenv()


def ai_api(model, prompt):
    response = requests.post(
        url="https://api.aimlapi.com/chat/completions",
        headers={
            "Authorization": "Bearer " + os.environ.get("AIML_API_KEY"),
            "Content-Type": "application/json",
        },
        data=json.dumps(
            {
                "model": model,
                "messages": [
                    {
                        "role": "user",
                        "content": prompt,
                    },
                ],
                "max_tokens": 512,
                "stream": False,
            }
        ),
    )


response.raise_for_status()
print(response.json())
