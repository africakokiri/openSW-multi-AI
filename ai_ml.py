import requests
import json

response = requests.post(
    url="https://api.aimlapi.com/chat/completions",
    headers={
        "Authorization": "Bearer 9e844f4c03124e4d86ab0d08be65de92",
        "Content-Type": "application/json",
    },
    data=json.dumps(
        {
            "model": "gpt-4o",
            "messages": [
                {
                    "role": "user",
                    "content": "What kind of model are you?",
                },
            ],
            "max_tokens": 512,
            "stream": False,
        }
    ),
)

response.raise_for_status()
print(response.json())
