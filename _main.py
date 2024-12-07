import asyncio
import time
import streamlit as st
from openai_gpt import gpt_prompt
from google_gemini import gemini_prompt
from anthropic_claude import claude_prompt
from meta_llama import llama_prompt
from qwen_qwen import qwen_prompt

# 페이지 설정
st.set_page_config(layout="wide")

st.markdown(
    """
    <style>
            div.st-emotion-cache-keje6w:nth-child(1) {
                max-width: 360px;
            }

            div.stColumn:nth-child(2) > div:nth-child(1) > div:nth-child(1) > div:nth-child(1) {
                display: flex;
                flex-wrap: wrap;
                flex-direction: row;
            }
            
            div.stColumn:nth-child(2) > div:nth-child(1) > div:nth-child(1) > div:nth-child(1) > .stChatMessage {
                background-color: #F1F2F6;
                border-radius: 8px;
                padding-right: 24px;
                min-width: 400px;
                max-width: 400px;
            }
            
               div.stColumn:nth-child(2) > div:nth-child(1) > div:nth-child(1) > div:nth-child(1) > .stElementContainer {
                max-width: 120px;   
            }

            .st-emotion-cache-1khdzpl > div:nth-child(1) .stColumn {
                background-color: black;
            }
        </style>
    """,
    unsafe_allow_html=True,
)

# 세션 상태 초기화
if "response_times" not in st.session_state:
    st.session_state["response_times"] = {
        "gpt": [],
        "gemini": [],
        "claude": [],
        "llama": [],
        "qwen": [],
    }

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

if "qwen_responses" not in st.session_state:
    st.session_state["qwen_responses"] = []

if "ai_display_selection" not in st.session_state:
    st.session_state["ai_display_selection"] = ["ChatGPT", "Gemini", "Claude", "Llama"]

if "disable_ai_in_tabs" not in st.session_state:
    st.session_state["disable_ai_in_tabs"] = True  # Set to True by default


# 비동기 API 호출 함수 정의
async def fetch_gpt_response(prompt):
    if (
        st.session_state["disable_ai_in_tabs"]
        and "ChatGPT" not in st.session_state["ai_display_selection"]
    ):
        return ""
    start_time = time.time()
    response = gpt_prompt(prompt) if prompt else ""
    end_time = time.time()
    st.session_state["response_times"]["gpt"].append(end_time - start_time)
    return response


async def fetch_gemini_response(prompt):
    if (
        st.session_state["disable_ai_in_tabs"]
        and "Gemini" not in st.session_state["ai_display_selection"]
    ):
        return ""
    start_time = time.time()
    response = gemini_prompt(prompt) if prompt else ""
    end_time = time.time()
    st.session_state["response_times"]["gemini"].append(end_time - start_time)
    return response


async def fetch_claude_response(prompt):
    if (
        st.session_state["disable_ai_in_tabs"]
        and "Claude" not in st.session_state["ai_display_selection"]
    ):
        return ""
    start_time = time.time()
    response = claude_prompt(prompt) if prompt else ""
    end_time = time.time()
    st.session_state["response_times"]["claude"].append(end_time - start_time)
    return response


async def fetch_llama_response(prompt):
    if (
        st.session_state["disable_ai_in_tabs"]
        and "Llama" not in st.session_state["ai_display_selection"]
    ):
        return ""
    start_time = time.time()
    response = llama_prompt(prompt) if prompt else ""
    end_time = time.time()
    st.session_state["response_times"]["llama"].append(end_time - start_time)
    return response


async def fetch_qwen_response(prompt):
    if (
        st.session_state["disable_ai_in_tabs"]
        and "Qwen" not in st.session_state["ai_display_selection"]
    ):
        return ""
    start_time = time.time()
    response = qwen_prompt(prompt) if prompt else ""
    end_time = time.time()
    st.session_state["response_times"]["qwen"].append(end_time - start_time)
    return response


# 비동기 처리 함수
async def fetch_all_responses(prompt):
    responses = await asyncio.gather(
        fetch_gpt_response(prompt),
        fetch_gemini_response(prompt),
        fetch_claude_response(prompt),
        fetch_llama_response(prompt),
        fetch_qwen_response(prompt),
    )
    return responses


# 초기 선택 옵션 설정
options = ["ChatGPT", "Gemini", "Claude", "Llama", "Qwen"]
default_selection = ["ChatGPT", "Gemini", "Claude", "Llama"]

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
    st.session_state["qwen_responses"].append(responses[4])

# 탭 구성
(
    All,
    gpt_as_tab,
    gemini_as_tab,
    claude_as_tab,
    llama_as_tab,
    qwen_as_tab,
    settings_as_tab,
) = st.tabs(["전체", "ChatGPT", "Gemini", "Claude", "Llama", "Qwen", "설정"])

# 탭: Settings
with settings_as_tab:
    st.title("Settings")

    # Pre-select all options by default
    selection = st.pills(
        "전체 탭에서 표시할 AI를 선택하세요.",
        options,
        default=default_selection,
        selection_mode="multi",
    )

    # Store the selection in session state for persistence
    st.session_state["ai_display_selection"] = selection

    # Add the option to disable AI in individual tabs
    st.session_state["disable_ai_in_tabs"] = st.checkbox(
        "비활성화한 AI 모델을 프로그램 전체에서 비활성화하여 응답 속도를 높입니다.",
        value=True,
    )

# 탭: 전체
with All:
    input_as_col, output_as_col = st.columns(2)

    with input_as_col:
        if not prompt:
            with st.chat_message("user"):
                st.markdown("**USER**")

        # 이전 입력도 포함하여 보여주기
        for prompt_text in st.session_state["prompt_history"]:
            with st.chat_message("user"):
                st.write(prompt_text)

    with output_as_col:
        # Retrieve the selection from session state, defaulting to all options if not set
        display_selection = st.session_state.get(
            "ai_display_selection", default_selection
        )

        ai_configs = [
            {
                "name": "ChatGPT",
                "responses": st.session_state["gpt_responses"],
                "times": st.session_state["response_times"]["gpt"],
                "avatar": "./assets/gpt.svg",
                "model": "openAI: gpt-4o-mini",
            },
            {
                "name": "Gemini",
                "responses": st.session_state["gemini_responses"],
                "times": st.session_state["response_times"]["gemini"],
                "avatar": "./assets/gemini.svg",
                "model": "Google: Gemini-1.5-flash",
            },
            {
                "name": "Claude",
                "responses": st.session_state["claude_responses"],
                "times": st.session_state["response_times"]["claude"],
                "avatar": "./assets/claude.svg",
                "model": "Anthropic: Claude-3-5-sonnet",
            },
            {
                "name": "Llama",
                "responses": st.session_state["llama_responses"],
                "times": st.session_state["response_times"]["llama"],
                "avatar": "./assets/meta.png",
                "model": "Meta: Llama-3.2-90B-Vision-Instruct-Turbo",
            },
            {
                "name": "Qwen",
                "responses": st.session_state["qwen_responses"],
                "times": st.session_state["response_times"]["qwen"],
                "avatar": "./assets/qwen.png",
                "model": "Qwen: Qwen2.5-72B-Instruct-Turbo",
            },
        ]

        for config in ai_configs:
            if config["name"] in display_selection:
                with st.chat_message("ai", avatar=config["avatar"]):
                    if not prompt:
                        st.markdown(f"**{config['model']}**")
                    else:
                        # 응답과 해당 응답의 시간을 함께 결합
                        responses_with_times = []
                        for response, time in zip(config["responses"], config["times"]):
                            responses_with_times.append(
                                f"{response}\n\n응답 시간: {time:.2f} 초\n\n---\n\n"
                            )

                        # 결합된 응답 표시 (응답 사이 간격 넓힘)
                        st.markdown("\n\n".join(responses_with_times))

# 탭: GPT
with gpt_as_tab:
    if "ChatGPT" in st.session_state["ai_display_selection"]:
        st.title("💬 openAI: gpt-4o-mini")
        for response in st.session_state["gpt_responses"]:
            with st.chat_message("user"):
                st.write(prompt)
            with st.chat_message("ai", avatar="./assets/gpt.svg"):
                st.write(response)
        if st.session_state["response_times"]["gpt"]:
            st.write(
                f"응답 시간: {sum(st.session_state['response_times']['gpt']):.2f} 초"
            )
    else:
        st.markdown("# ~~💬 openAI: gpt-4o-mini~~")
        st.write("해당 AI 모델은 비활성화되었습니다. 활성화는 설정 탭에서 가능합니다.")

# 탭: Gemini
with gemini_as_tab:
    if "Gemini" in st.session_state["ai_display_selection"]:
        st.title("💬 Google: Gemini-1.5-flash")
        for response in st.session_state["gemini_responses"]:
            with st.chat_message("user"):
                st.write(prompt)
            with st.chat_message("ai", avatar="./assets/gemini.svg"):
                st.write(response)
        if st.session_state["response_times"]["gemini"]:
            st.write(
                f"응답 시간: {sum(st.session_state['response_times']['gemini']):.2f} 초"
            )
    else:
        st.markdown("# ~~💬 Google: Gemini-1.5-flash~~")
        st.write("해당 AI 모델은 비활성화되었습니다. 활성화는 설정 탭에서 가능합니다.")

# 탭: Claude
with claude_as_tab:
    if "Claude" in st.session_state["ai_display_selection"]:
        st.title("💬 Anthropic: Claude-3-5-sonnet")
        for response in st.session_state["claude_responses"]:
            with st.chat_message("user"):
                st.write(prompt)
            with st.chat_message("ai", avatar="./assets/claude.svg"):
                st.write(response)
        if st.session_state["response_times"]["claude"]:
            st.write(
                f"응답 시간: {sum(st.session_state['response_times']['claude']):.2f} 초"
            )
    else:
        st.markdown("# ~~💬 Anthropic: Claude-3-5-sonnet~~")
        st.write("해당 AI 모델은 비활성화되었습니다. 활성화는 설정 탭에서 가능합니다.")

# 탭: Llama
with llama_as_tab:
    if "Llama" in st.session_state["ai_display_selection"]:
        st.title("💬 Meta: Llama-3.2-90B-Vision-Instruct-Turbo")
        for response in st.session_state["llama_responses"]:
            with st.chat_message("user"):
                st.write(prompt)
            with st.chat_message("ai", avatar="./assets/meta.png"):
                st.write(response)
        if st.session_state["response_times"]["llama"]:
            st.write(
                f"응답 시간: {sum(st.session_state['response_times']['llama']):.2f} 초"
            )
    else:
        st.markdown("# ~~💬 Meta: Llama-3.2-90B-Vision-Instruct-Turbo~~")
        st.write("해당 AI 모델은 비활성화되었습니다. 활성화는 설정 탭에서 가능합니다.")


# 탭: Qwen
with qwen_as_tab:
    if "Qwen" in st.session_state["ai_display_selection"]:
        st.title("💬 Qwen: Qwen2.5-72B-Instruct-Turbo")
        for response in st.session_state["qwen_responses"]:
            with st.chat_message("user"):
                st.write(prompt)
            with st.chat_message("ai", avatar="./assets/qwen.png"):
                st.write(response)
        if st.session_state["response_times"]["qwen"]:
            st.write(
                f"응답 시간: {sum(st.session_state['response_times']['qwen']):.2f} 초"
            )
    else:
        st.markdown("# ~~💬 Qwen: Qwen2.5-72B-Instruct-Turbo~~")
        st.write("해당 AI 모델은 비활성화되었습니다. 활성화는 설정 탭에서 가능합니다.")
