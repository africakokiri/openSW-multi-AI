import asyncio
import streamlit as st
from openai_gpt import gpt_prompt
from google_gemini import gemini_prompt
from anthropic_claude import claude_prompt
from meta_llama import llama_prompt

# 페이지 설정
st.set_page_config(layout="wide")

# 유저의 prompt를 state에 저장
prompt = st.chat_input("프롬프트를 입력하세요.")
st.session_state["prompt"] = prompt


# 비동기 API 호출 함수 정의
async def fetch_gpt_response(prompt):
    return gpt_prompt(prompt) if prompt else "GPT 응답을 기다리는 중..."


async def fetch_gemini_response(prompt):
    return gemini_prompt(prompt) if prompt else "Gemini 응답을 기다리는 중..."


async def fetch_claude_response(prompt):
    return claude_prompt(prompt) if prompt else "Claude 응답을 기다리는 중..."


async def fetch_llama_response(prompt):
    return llama_prompt(prompt) if prompt else "LLaMA 응답을 기다리는 중..."


# 비동기 처리 함수
async def fetch_all_responses(prompt):
    responses = await asyncio.gather(
        fetch_gpt_response(prompt),
        fetch_gemini_response(prompt),
        fetch_claude_response(prompt),
        fetch_llama_response(prompt),
    )
    return responses


# 비동기 응답 받아오기
responses = asyncio.run(fetch_all_responses(prompt))

# 탭 구성
All, gpt_as_tab, gemini_as_tab, claude_as_tab, llama_as_tab = st.tabs(
    ["전체", "chatGPT", "Gemini", "Claude", "llama"]
)

# 탭: 전체
with All:
    gpt_as_col, gemini_as_col, claude_as_col, llama_as_col = st.columns(4)

    with gpt_as_col:
        st.write(responses[0])  # GPT 응답

    with gemini_as_col:
        st.write(responses[1])  # Gemini 응답

    with claude_as_col:
        st.write(responses[2])  # Claude 응답

    with llama_as_col:
        st.write(responses[3])  # LLaMA 응답

# 탭: chatGPT
with gpt_as_tab:
    st.title("💬 openAI: GPT-4o-mini")
    st.caption("🚀 A Streamlit chatbot powered by openAI ChatGPT")
    st.write(responses[0])

# 탭: Gemini
with gemini_as_tab:
    st.title("💬 Google: Gemini-1.5-flash")
    st.caption("🚀 A Streamlit chatbot powered by Google Gemini")
    st.write(responses[1])

# 탭: Claude
with claude_as_tab:
    st.title("💬 Anthropic: claude-3-5-sonnet")
    st.caption("🚀 A Streamlit chatbot powered by Anthropic Claude")
    st.write(responses[2])

# 탭: llama
with llama_as_tab:
    st.title("💬 Meta: llama3.2-90b-vision")
    st.caption("🚀 A Streamlit chatbot powered by Meta LLaMA")
    st.write(responses[3])
