import asyncio
import streamlit as st
from openai_gpt import gpt_prompt
from google_gemini import gemini_prompt
from anthropic_claude import claude_prompt
from meta_llama import llama_prompt

# 페이지 설정
st.set_page_config(layout="wide")

st.markdown(
    """
        <style>
            div.st-emotion-cache-keje6w:nth-child(1) {
                max-width: 320px;
            }
            
            .st-emotion-cache-14hz42n > div:nth-child(1) {
                display: flex;
                flex-wrap: wrap;
            }
            
            div.st-emotion-cache-12w0qpk:nth-child(1), div.st-emotion-cache-12w0qpk:nth-child(2), div.st-emotion-cache-12w0qpk:nth-child(3), div.st-emotion-cache-12w0qpk:nth-child(4){
                min-width: 320px;
                max-width: 320px;
            }
            
        </style>
    """,
    unsafe_allow_html=True,
)

# 유저의 prompt를 session_state에 저장
if "prompt_history" not in st.session_state:
    st.session_state["prompt_history"] = []

if "gpt_responses" not in st.session_state:
    st.session_state["gpt_responses"] = []

if "gemini_responses" not in st.session_state:
    st.session_state["gemini_responses"] = []

if "claude_responses" not in st.session_state:
    st.session_state["claude_responses"] = []

if "llama_responses" not in st.session_state:
    st.session_state["llama_responses"] = []


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


# 유저의 새로운 prompt 입력
prompt = st.chat_input("프롬프트를 입력하세요.")
if prompt:
    # 기존 prompt 기록에 새로운 prompt 추가
    st.session_state["prompt_history"].append(prompt)

    # 각 AI의 응답을 비동기적으로 받아오기
    responses = asyncio.run(fetch_all_responses(prompt))

    # 각 AI의 응답을 session_state에 기록
    st.session_state["gpt_responses"].append(responses[0])
    st.session_state["gemini_responses"].append(responses[1])
    st.session_state["claude_responses"].append(responses[2])
    st.session_state["llama_responses"].append(responses[3])


# 탭 구성
All, gpt_as_tab, gemini_as_tab, claude_as_tab, llama_as_tab, settings = st.tabs(
    ["전체", "chatGPT", "Gemini", "Claude", "llama", "설정"]
)

# 탭: 전체
with All:
    input_as_col, output_as_col = st.columns(2)

    with input_as_col:
        with st.chat_message("user"):
            st.markdown("**USER**")

        # 이전 입력도 포함하여 보여주기
        for prompt_text in st.session_state["prompt_history"]:
            st.write(prompt_text)

    with output_as_col:
        gpt_as_col, gemini_as_col, claude_as_col, llama_as_col = st.columns(4)

        with gpt_as_col:
            with st.chat_message("ai", avatar="./assets/gpt.svg"):
                st.markdown("**openAI: gpt-4o-mini**")

            # 이전 응답도 포함하여 보여주기
            for response in st.session_state["gpt_responses"]:
                st.write(response)

        with gemini_as_col:
            with st.chat_message("ai", avatar="./assets/gemini.svg"):
                st.markdown("**Google: Gemini-1.5-flash**")

            for response in st.session_state["gemini_responses"]:
                st.write(response)

        with claude_as_col:
            with st.chat_message("ai", avatar="./assets/claude.svg"):
                st.markdown("**Anthropic: Claude-3-5-sonnet**")

            for response in st.session_state["claude_responses"]:
                st.write(response)

        with llama_as_col:
            with st.chat_message("ai", avatar="./assets/meta.png"):
                st.markdown("**Meta: Llama3.2-90b-vision**")

            for response in st.session_state["llama_responses"]:
                st.write(response)


# 탭: chatGPT
with gpt_as_tab:
    st.title("💬 openAI: gpt-4o-mini")
    st.caption("🚀 A Streamlit chatbot powered by openAI ChatGPT")
    for response in st.session_state["gpt_responses"]:
        st.write(response)

# 탭: Gemini
with gemini_as_tab:
    st.title("💬 Google: Gemini-1.5-flash")
    st.caption("🚀 A Streamlit chatbot powered by Google Gemini")
    for response in st.session_state["gemini_responses"]:
        st.write(response)

# 탭: Claude
with claude_as_tab:
    st.title("💬 Anthropic: Claude-3-5-sonnet")
    st.caption("🚀 A Streamlit chatbot powered by Anthropic Claude")
    for response in st.session_state["claude_responses"]:
        st.write(response)

# 탭: llama
with llama_as_tab:
    st.title("💬 Meta: Llama3.2-90b-vision")
    st.caption("🚀 A Streamlit chatbot powered by Meta LLaMA")
    for response in st.session_state["llama_responses"]:
        st.write(response)

# 탭: 설정
with settings:
    st.title("💬 Meta: Llama3.2-90b-vision")
