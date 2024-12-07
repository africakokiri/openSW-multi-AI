import anthropic
from dotenv import load_dotenv
import os

load_dotenv()


def claude_prompt(prompt):
    client = anthropic.Anthropic(api_key=os.environ.get("CLAUDE_API_KEY"))
    message = client.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=1024,
        messages=[{"role": "user", "content": prompt}],
    )

    if isinstance(message.content, list):
        for item in message.content:
            if hasattr(item, "text"):
                return item.text
    else:
        print(message.content)
