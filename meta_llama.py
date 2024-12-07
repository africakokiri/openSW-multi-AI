from llamaapi import LlamaAPI
from dotenv import load_dotenv
import os
import json

load_dotenv()


def llama_prompt(prompt):
    api_token = os.environ.get("LLAMA_API_KEY")

    llama = LlamaAPI(api_token)

    api_request_json = {
        "model": "llama3.2-90b-vision",
        "messages": [
            {"role": "user", "content": prompt},
        ],
    }

    # Make your request and handle the response
    response = llama.run(api_request_json)
    res = json.dumps(response.json(), indent=2)
    res_to_json = json.loads(res)
    content = res_to_json["choices"][0]["message"].get("content")

    return content
