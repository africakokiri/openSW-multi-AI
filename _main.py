import asyncio
import time
import json
import streamlit as st

from streamlit_local_storage import LocalStorage

from openai_gpt import gpt_prompt
from google_gemini import gemini_prompt
from anthropic_claude import claude_prompt

# 응답요약기능 관련 라이브러리 및 WordCloud 생성 함수 - 종현 추가
from transformers import pipeline
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from io import BytesIO
import base64

# 요약 모델 초기화
summarizer = pipeline("summarization")


def summarize_text(text, max_length=60, min_length=25):
    """텍스트를 요약하는 함수"""
    if not text or len(text.strip()) == 0:
        return "요약할 내용이 없습니다."
    summary = summarizer(
        text, max_length=max_length, min_length=min_length, do_sample=False
    )
    return summary[0]["summary_text"]


def generate_wordcloud(text):
    """텍스트 기반으로 WordCloud를 생성하는 함수"""
    if not text or len(text.strip()) == 0:
        return None

    wordcloud = WordCloud(
        width=800,
        height=400,
        background_color="white",
        colormap="viridis",
        max_words=100,
    ).generate(text)

    image_stream = BytesIO()
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.tight_layout(pad=0)
    plt.savefig(image_stream, format="png")
    plt.close()
    image_stream.seek(0)
    return base64.b64encode(image_stream.getvalue()).decode("utf-8")


# 페이지 설정
st.set_page_config(layout="wide")


# 로컬 스토리지 초기화
localS = LocalStorage()


# localStorage에 저장된 prompt를 가져오는 함수
def get_local_storage():
    prompts = localS.getItem("prompt_history")

    if prompts is None:
        prompts = []
    return prompts


# localStorage에 prompt를 저장하는 함수
def set_local_storage(key, value):
    if value:
        prompts = get_local_storage()
        prompts.append(value)
        localS.setItem(key, prompts)


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
                align-items: flex-start;
            }

            div.stColumn:nth-child(2) > div:nth-child(1) > div:nth-child(1) > div:nth-child(1) > .stChatMessage {
                background-color: #F1F2F6;
                border-radius: 8px;
                padding-right: 24px;
                min-width: 400px;
                resize: horizontal;
                overflow: auto;
            }

               div.stColumn:nth-child(2) > div:nth-child(1) > div:nth-child(1) > div:nth-child(1) > .stElementContainer {
                max-width: 120px;
            }

            .st-emotion-cache-1khdzpl > div:nth-child(1) .stColumn {
                background-color: black;
            }
            
            div[data-testid=stToast] {
                background-color: #000000;
                color: #FFFFFF;
                position: absolute;
                top: 0px;
                left: 0px;
            }
             
            [data-testid=toastContainer] [data-testid=stMarkdownContainer] > p {
                foreground-color: #FFFFFF;
            }
            
            body > div.stToastContainer.st-et.st-eu.st-ev.st-ew.st-ag.st-ex.st-ey.st-ez.st-f0.st-f1.st-f2.st-f3.st-f4.st-f5 > div > svg {
                color: white;
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
    }

if "prompt_history" not in st.session_state:
    st.session_state["prompt_history"] = []

if "gpt_responses" not in st.session_state:
    st.session_state["gpt_responses"] = []

if "gemini_responses" not in st.session_state:
    st.session_state["gemini_responses"] = []

if "claude_responses" not in st.session_state:
    st.session_state["claude_responses"] = []

if "ai_display_selection" not in st.session_state:
    st.session_state["ai_display_selection"] = ["ChatGPT", "Gemini", "Claude"]

if "disable_ai_in_tabs" not in st.session_state:
    st.session_state["disable_ai_in_tabs"] = True


# 비동기 API 호출 함수 정의
async def fetch_gpt_response(prompt):
    if (
        st.session_state["disable_ai_in_tabs"]
        and "ChatGPT" not in st.session_state["ai_display_selection"]
    ):
        return ""
    start_time = time.time()
    response = await asyncio.to_thread(gpt_prompt, prompt) if prompt else ""
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
    response = await asyncio.to_thread(gemini_prompt, prompt) if prompt else ""
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
    response = await asyncio.to_thread(claude_prompt, prompt) if prompt else ""
    end_time = time.time()
    st.session_state["response_times"]["claude"].append(end_time - start_time)
    return response


# 비동기 처리 함수: 각 AI의 응답을 반환할 때 시간을 포함한 튜플로 반환
async def fetch_all_responses_with_time(prompt):
    responses = await asyncio.gather(
        fetch_gpt_response(prompt),
        fetch_gemini_response(prompt),
        fetch_claude_response(prompt),
    )

    # AI 이름 리스트
    ai_names = ["GPT", "Gemini", "Claude"]

    # 응답과 시간을 묶어 리스트로 반환
    return [
        {
            "name": ai_names[i],
            "response": responses[i],
            "time": st.session_state["response_times"][ai_names[i].lower()][-1],
        }
        for i in range(len(ai_names))
    ]


# 초기 선택 옵션 설정
options = ["ChatGPT", "Gemini", "Claude"]
default_selection = ["ChatGPT", "Gemini", "Claude"]

# 유저의 새로운 prompt 입력
prompt = st.chat_input("프롬프트를 입력하세요.")

# 새로운 Prompt가 입력되었을 때 가장 짧은 응답 출력
if prompt:
    # 기존 prompt 기록에 새로운 prompt 추가
    st.session_state["prompt_history"].append(prompt)

    # 각 AI의 응답과 시간을 받아오기
    ai_responses_with_time = asyncio.run(fetch_all_responses_with_time(prompt))

    # 가장 빠른 응답 선택
    shortest_response = min(ai_responses_with_time, key=lambda x: x["time"])

    if prompt and shortest_response:
        st.toast(
            f"가장 응답이 빠른 AI: {shortest_response["name"]}, 응답 시간: {shortest_response['time']:.2f} 초"
        )

    # 각 AI의 응답을 session_state에 기록
    for ai_response in ai_responses_with_time:
        if ai_response["name"] == "GPT":
            st.session_state["gpt_responses"].append(ai_response["response"])
        elif ai_response["name"] == "Gemini":
            st.session_state["gemini_responses"].append(ai_response["response"])
        elif ai_response["name"] == "Claude":
            st.session_state["claude_responses"].append(ai_response["response"])

# 탭 구성 / 응답 요약, Wordcloud 탭 추가 - 종현 추가
(
    All,
    records_as_tab,
    summarization_tab,
    wordcloud_tab,
    settings_as_tab,
) = st.tabs(["메인 페이지", "로그", "응답 요약", "WordCloud", "설정"])


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


# 탭: 기록
with records_as_tab:
    stored_prompts = localS.getItem("prompt_history")
    st.markdown("#### 로그를 확인하시려면 새로고침 해 주세요!")

    def delete_prompt_history():
        localS.deleteItem("prompt_history")

    if stored_prompts:
        st.button("로그 전체 삭제", type="primary", on_click=delete_prompt_history)

    if stored_prompts:

        # 데이터 처리 함수
        def clean_response(response):
            # 문자열로 인코딩된 리스트를 파싱하고 첫 번째 요소를 반환
            parsed = eval(response)  # ['응답 문자열'] -> 리스트로 변환
            return parsed[0] if parsed else ""  # 첫 번째 요소 반환

        for result in stored_prompts:
            with st.chat_message("user"):
                if result[1]["prompt"]:
                    st.write(result[1]["prompt"])
            with st.chat_message("ai", avatar="./assets/gpt.svg"):
                st.markdown(f"{clean_response(result[1]['gpt_response'])}")
                st.divider()
            with st.chat_message("ai", avatar="./assets/gemini.svg"):
                st.markdown(f"{clean_response(result[1]['gemini_response'])}")
                st.divider()
            with st.chat_message("ai", avatar="./assets/claude.svg"):
                st.markdown(f"{clean_response(result[1]['claude_response'])}")
                st.divider()


# 탭: 전체
with All:
    input_as_col, output_as_col = st.columns(2)

    with input_as_col:
        if not prompt:
            with st.chat_message("user"):
                st.markdown("**USER**")

        # 이전 입력도 포함하여 보여주기
        for prompt_text in st.session_state["prompt_history"]:
            if not isinstance(prompt_text, dict):
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
                "model": "OpenAI: gpt-4o-mini",
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
                "model": "Anthropic: Claude-3.5-Sonnet",
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


# 'prompt_history'가 세션에 없으면 초기화
if "prompt_history" not in st.session_state:
    st.session_state["prompt_history"] = []

if prompt:
    st.session_state["prompt_history"].append(
        {
            "prompt": prompt,
            "gpt_response": str(st.session_state.get("gpt_responses", "")),
            "gemini_response": str(st.session_state.get("gemini_responses", "")),
            "claude_response": str(st.session_state.get("claude_responses", "")),
        }
    )

# 로컬 스토리지에 저장
set_local_storage("prompt_history", st.session_state["prompt_history"])


###############종현기능추가##

# 탭: 응답 요약
with summarization_tab:
    st.title("📄 응답 요약")
    if prompt:
        st.write("**입력된 프롬프트:**")
        st.info(prompt)

        # 각 AI 응답 요약
        summaries = [
            summarize_text(response)
            for response in [
                (
                    st.session_state["gpt_responses"][-1]
                    if st.session_state["gpt_responses"]
                    else ""
                ),
                (
                    st.session_state["gemini_responses"][-1]
                    if st.session_state["gemini_responses"]
                    else ""
                ),
                (
                    st.session_state["claude_responses"][-1]
                    if st.session_state["claude_responses"]
                    else ""
                ),
            ]
        ]

        ai_models = ["ChatGPT", "Gemini", "Claude"]
        for model, summary in zip(ai_models, summaries):
            st.subheader(f"{model} 요약:")
            st.write(summary)

    else:
        st.write("프롬프트를 입력한 후 요약 결과를 확인하세요.")

# 탭: WordCloud
with wordcloud_tab:
    st.title("☁️ WordCloud")
    if prompt:
        st.write("**입력된 프롬프트:**")
        st.info(prompt)

        wordcloud_image = generate_wordcloud(prompt)

        if wordcloud_image:
            st.image(f"data:image/png;base64,{wordcloud_image}", use_column_width=True)
        else:
            st.write("WordCloud를 생성할 내용이 없습니다.")
    else:
        st.write("프롬프트를 입력한 후 WordCloud를 확인하세요.")
