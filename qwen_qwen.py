from dotenv import load_dotenv
import requests
import os
import json

load_dotenv()


def qwen_prompt(prompt):
    response = requests.post(
        url="https://api.aimlapi.com/chat/completions",
        headers={
            "Authorization": "Bearer " + os.environ.get("AIML_API_KEY"),
            "Content-Type": "application/json",
        },
        data=json.dumps(
            {
                "model": "Qwen/Qwen2.5-72B-Instruct-Turbo",
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

    res = json.dumps(response.json(), indent=2)
    res_to_json = json.loads(res)
    content = res_to_json["choices"][0]["message"].get("content")

    return content
