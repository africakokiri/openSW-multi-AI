import asyncio
import streamlit as st
import streamlit.components.v1 as components
from openai_gpt import gpt_prompt
from google_gemini import gemini_prompt
from anthropic_claude import claude_prompt
from meta_llama import llama_prompt

# 페이지 설정
st.set_page_config(layout="wide")

# 유저의 prompt를 state에 저장
prompt = st.chat_input("프롬프트를 입력하세요.")
st.session_state["prompt"] = prompt

st.markdown(
    """
        <style>
            .stColumn {
                background-color: #F1F2F6;
                border-radius: 8px;
                height: 510px;
                padding: 16px;
                max-width: 280px;
            }
            
            .stChatMessage {
                padding: 0px;
                width: 100%;
                display: flex;
                justify-content: center;
            }
        </style>
    """,
    unsafe_allow_html=True,
)


# 비동기 API 호출 함수 정의
async def fetch_gpt_response(prompt):
    return gpt_prompt(prompt) if prompt else ""


async def fetch_gemini_response(prompt):
    return gemini_prompt(prompt) if prompt else ""


async def fetch_claude_response(prompt):
    return claude_prompt(prompt) if prompt else ""


async def fetch_llama_response(prompt):
    return llama_prompt(prompt) if prompt else ""


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
All, gpt_as_tab, gemini_as_tab, claude_as_tab, llama_as_tab, settings = st.tabs(
    ["전체", "chatGPT", "Gemini", "Claude", "llama", "설정"]
)

# 탭: 전체
with All:
    gpt_as_col, gemini_as_col, claude_as_col, llama_as_col = st.columns(4)

    with gpt_as_col:
        with st.chat_message("ai", avatar="./assets/gpt.svg"):
            st.markdown("**openAI: gpt-4o-mini**")

        st.write(responses[0])  # GPT 응답

    with gemini_as_col:
        with st.chat_message("ai", avatar="./assets/gemini.svg"):
            st.markdown("**Google: Gemini-1.5-flash**")

        st.write(responses[1])  # Gemini 응답

    with claude_as_col:
        with st.chat_message("ai", avatar="./assets/claude.svg"):
            st.markdown("**Anthropic: Claude-3-5-sonnet**")

        st.write(responses[2])  # Claude 응답

    with llama_as_col:
        with st.chat_message("ai", avatar="./assets/meta.png"):
            st.markdown("**Meta: Llama3.2-90b-vision**")

        st.write(responses[3])  # LLaMA 응답


# 탭: chatGPT
with gpt_as_tab:
    st.title("💬 openAI: gpt-4o-mini")
    st.caption("🚀 A Streamlit chatbot powered by openAI ChatGPT")
    st.write(responses[0])

# 탭: Gemini
with gemini_as_tab:
    st.title("💬 Google: Gemini-1.5-flash")
    st.caption("🚀 A Streamlit chatbot powered by Google Gemini")
    st.write(responses[1])

# 탭: Claude
with claude_as_tab:
    st.title("💬 Anthropic: Claude-3-5-sonnet")
    st.caption("🚀 A Streamlit chatbot powered by Anthropic Claude")
    st.write(responses[2])

# 탭: llama
with llama_as_tab:
    st.title("💬 Meta: Llama3.2-90b-vision")
    st.caption("🚀 A Streamlit chatbot powered by Meta LLaMA")
    st.write(responses[3])

# 탭: 설정
with settings:
    st.title("💬 Meta: Llama3.2-90b-vision")
