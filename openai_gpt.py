import streamlit as st
from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()


# OpenAI GPT 모델 설정
def gpt_prompt(input):
    client = OpenAI(api_key=os.environ.get("GPT_API_KEY"))
    response = client.chat.completions.create(
        model="gpt-4o-mini", messages=[{"role": "user", "content": input}]
    )
    return response.choices[0].message.content
