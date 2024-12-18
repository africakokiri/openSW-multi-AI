import asyncio
import time
import streamlit as st

from streamlit_local_storage import LocalStorage

from openai_gpt import gpt_prompt
from google_gemini import gemini_prompt
from anthropic_claude import claude_prompt
from meta_llama import llama_prompt
from qwen_qwen import qwen_prompt

#응답요약기능 관련 라이브러리 및 WordCloud 생성 함수 - 종현 추가
from transformers import pipeline
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from io import BytesIO
import base64

# 요약 모델 초기화
def load_summarizer():
    """요약 모델을 불러오는 함수"""
    try:
        return pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")
    except Exception as e:
        print(f"모델 로드 오류: {e}")
        return None

async def summarize_text_async(summarizer, text):
    """비동기식 요약 함수"""
    if not summarizer or not text:
        return "요약할 내용이 없습니다."
    return await asyncio.to_thread(
        lambda: summarizer(text, max_length=60, min_length=25, do_sample=False)[0]["summary_text"]
    )

# 비동기 WordCloud 생성 함수
async def generate_wordcloud_async(text):
    """비동기식으로 WordCloud 생성"""
    if not text or len(text.strip()) == 0:
        return None
    try:
        return await asyncio.to_thread(lambda: _generate_wordcloud_image(text))
    except Exception as e:
        return None

def _generate_wordcloud_image(text):
    """WordCloud 이미지 생성 로직 (비동기 호출을 위해 별도 함수로 분리)"""
    wordcloud = WordCloud(
        width=800,
        height=400,
        background_color="white",
        colormap="viridis",
        max_words=100
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


async def fetch_llama_response(prompt):
    if (
        st.session_state["disable_ai_in_tabs"]
        and "Llama" not in st.session_state["ai_display_selection"]
    ):
        return ""
    start_time = time.time()
    response = await asyncio.to_thread(llama_prompt, prompt) if prompt else ""
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
    response = await asyncio.to_thread(qwen_prompt, prompt) if prompt else ""
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
default_selection = ["ChatGPT", "Gemini", "Claude"]

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

# 탭 구성 / 응답 요약, Wordcloud 탭 추가 - 종현 추가
(
    All,
    gpt_as_tab,
    gemini_as_tab,
    claude_as_tab,
    llama_as_tab,
    qwen_as_tab,
    records_as_tab,
    settings_as_tab,
    summarization_tab,
    wordcloud_tab,
) = st.tabs(["전체", "ChatGPT", "Gemini", "Claude", "Llama", "Qwen", "로그", "설정", "응답 요약", "WordCloud"])



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
        for result in stored_prompts:
            with st.chat_message("user"):
                if result[1]["prompt"]:
                    st.write(result[1]["prompt"])
            with st.chat_message("ai", avatar="./assets/gpt.svg"):
                st.markdown(
                    f"""
             <p>
                 {result[1]["gpt_response"].replace("['", "").replace("']", "").replace("\\n", "<br />").replace("`", "")}
             </p>
             """,
                    unsafe_allow_html=True,
                )
            with st.chat_message("ai", avatar="./assets/gemini.svg"):
                st.markdown(
                    f"""
             <p>
                 {result[1]["gemini_response"].replace("['", "").replace("']", "").replace("\\n", "<br />").replace("`", "")}
             </p>
             """,
                    unsafe_allow_html=True,
                )
            with st.chat_message("ai", avatar="./assets/claude.svg"):
                st.markdown(
                    f"""
             <p>
                 {result[1]["claude_response"].replace("['", "").replace("']", "").replace("\\n", "<br />").strip("[]").strip('""').replace("`", "")}
             </p>
             """,
                    unsafe_allow_html=True,
                )
            with st.chat_message("ai", avatar="./assets/meta.png"):
                st.markdown(
                    f"""
             <p>
                 {result[1]["llama_response"].replace("['", "").replace("']", "").replace("\\n", "<br />").replace("`", "")}
             </p>
             """,
                    unsafe_allow_html=True,
                )
            with st.chat_message("ai", avatar="./assets/qwen.png"):
                st.markdown(
                    f"""
             <p>
                 {result[1]["qwen_response"].replace("['", "").replace("']", "").replace("\\n", "<br />").replace("`", "")}
             </p>
             """,
                    unsafe_allow_html=True,
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
        st.title("💬 OpenAI: gpt-4o-mini")

        # 'prompt_history'와 'gpt_responses'가 존재하는지 확인
        if "prompt_history" in st.session_state and "gpt_responses" in st.session_state:
            prompt_history = st.session_state["prompt_history"]
            gpt_responses = st.session_state["gpt_responses"]

            # 대화 기록을 순차적으로 표시
            for i in range(len(prompt_history)):
                # 유저의 프롬프트 표시 (유저 메시지가 먼저)
                prompt_text = prompt_history[i]
                if prompt_text and not isinstance(prompt_text, dict):
                    with st.chat_message("user"):
                        st.write(prompt_text)

                # AI의 응답 표시 (AI 응답이 그 뒤에)
                if i < len(gpt_responses):
                    response = gpt_responses[i]
                    with st.chat_message("ai", avatar="./assets/gpt.svg"):
                        st.write(response)

        else:
            st.write("대화 내역이 없습니다.")

        # 응답 시간 출력
        if st.session_state["response_times"]["gpt"]:
            st.write(
                f"응답 시간: {sum(st.session_state['response_times']['gpt']):.2f} 초"
            )
    else:
        st.markdown("# ~~💬 OpenAI: gpt-4o-mini~~")
        st.write(
            "해당 AI 모델은 비활성화되었습니다. 설정 탭에서 활성화 할 수 있습니다."
        )


# 탭: Gemini
with gemini_as_tab:
    if "Gemini" in st.session_state["ai_display_selection"]:
        st.title("💬 Google: Gemini-1.5-flash")

        # 'prompt_history'와 'gemini_responses'가 존재하는지 확인
        if (
            "prompt_history" in st.session_state
            and "gemini_responses" in st.session_state
        ):
            prompt_history = st.session_state["prompt_history"]
            gemini_responses = st.session_state["gemini_responses"]

            # 대화 기록을 순차적으로 표시
            for i in range(len(prompt_history)):
                # 유저의 프롬프트 표시 (유저 메시지가 먼저)
                prompt_text = prompt_history[i]
                if prompt_text and not isinstance(prompt_text, dict):
                    with st.chat_message("user"):
                        st.write(prompt_text)

                # AI의 응답 표시 (AI 응답이 그 뒤에)
                if i < len(gemini_responses):
                    response = gemini_responses[i]
                    with st.chat_message("ai", avatar="./assets/gemini.svg"):
                        st.write(response)

        else:
            st.write("대화 내역이 없습니다.")

        # 응답 시간 출력
        if st.session_state["response_times"]["gemini"]:
            st.write(
                f"응답 시간: {sum(st.session_state['response_times']['gemini']):.2f} 초"
            )
    else:
        st.markdown("# ~~💬 Google: Gemini-1.5-flash~~")
        st.write(
            "해당 AI 모델은 비활성화되었습니다. 설정 탭에서 활성화 할 수 있습니다."
        )


# 탭: Claude
with claude_as_tab:
    if "Claude" in st.session_state["ai_display_selection"]:
        st.title("💬 Anthropic: Claude-3.5-Sonnet")

        # 'prompt_history'와 'claude_responses'가 존재하는지 확인
        if (
            "prompt_history" in st.session_state
            and "claude_responses" in st.session_state
        ):
            prompt_history = st.session_state["prompt_history"]
            claude_responses = st.session_state["claude_responses"]

            # 대화 기록을 순차적으로 표시
            for i in range(len(prompt_history)):
                # 유저의 프롬프트 표시 (유저 메시지가 먼저)
                prompt_text = prompt_history[i]
                if prompt_text and not isinstance(prompt_text, dict):
                    with st.chat_message("user"):
                        st.write(prompt_text)

                # AI의 응답 표시 (AI 응답이 그 뒤에)
                if i < len(claude_responses):
                    response = claude_responses[i]
                    with st.chat_message("ai", avatar="./assets/claude.svg"):
                        st.write(response)

        else:
            st.write("대화 내역이 없습니다.")

        # 응답 시간 출력
        if st.session_state["response_times"]["claude"]:
            st.write(
                f"응답 시간: {sum(st.session_state['response_times']['claude']):.2f} 초"
            )
    else:
        st.markdown("# ~~💬 Anthropic: Claude-3.5-Sonnet~~")
        st.write(
            "해당 AI 모델은 비활성화되었습니다. 설정 탭에서 활성화 할 수 있습니다."
        )


# 탭: Llama
with llama_as_tab:
    if "Llama" in st.session_state["ai_display_selection"]:
        st.title("💬 Meta: Llama-3.2-90B-Vision-Instruct-Turbo")

        # 'prompt_history'와 'llama_responses'가 존재하는지 확인
        if (
            "prompt_history" in st.session_state
            and "llama_responses" in st.session_state
        ):
            prompt_history = st.session_state["prompt_history"]
            llama_responses = st.session_state["llama_responses"]

            # 대화 기록을 순차적으로 표시
            for i in range(len(prompt_history)):
                # 유저의 프롬프트 표시 (유저 메시지가 먼저)
                prompt_text = prompt_history[i]
                if prompt_text and not isinstance(prompt_text, dict):
                    with st.chat_message("user"):
                        st.write(prompt_text)

                # AI의 응답 표시 (AI 응답이 그 뒤에)
                if i < len(llama_responses):
                    response = llama_responses[i]
                    with st.chat_message("ai", avatar="./assets/meta.png"):
                        st.write(response)

        else:
            st.write("대화 내역이 없습니다.")

        # 응답 시간 출력
        if st.session_state["response_times"]["llama"]:
            st.write(
                f"응답 시간: {sum(st.session_state['response_times']['llama']):.2f} 초"
            )
    else:
        st.markdown("# ~~💬 Meta: Llama-3.2-90B-Vision-Instruct-Turbo~~")
        st.write(
            "해당 AI 모델은 비활성화되었습니다. 설정 탭에서 활성화 할 수 있습니다."
        )


# 탭: Qwen
with qwen_as_tab:
    if "Qwen" in st.session_state["ai_display_selection"]:
        st.title("💬 Qwen: Qwen2.5-72B-Instruct-Turbo")

        # 'prompt_history'와 'qwen_responses'가 존재하는지 확인
        if (
            "prompt_history" in st.session_state
            and "qwen_responses" in st.session_state
        ):
            prompt_history = st.session_state["prompt_history"]
            qwen_responses = st.session_state["qwen_responses"]

            # 대화 기록을 순차적으로 표시
            for i in range(len(prompt_history)):
                # 유저의 프롬프트 표시 (유저 메시지가 먼저)
                prompt_text = prompt_history[i]
                if prompt_text and not isinstance(prompt_text, dict):
                    with st.chat_message("user"):
                        st.write(prompt_text)

                # AI의 응답 표시 (AI 응답이 그 뒤에)
                if i < len(qwen_responses):
                    response = qwen_responses[i]
                    with st.chat_message("ai", avatar="./assets/qwen.png"):
                        st.write(response)

        else:
            st.write("대화 내역이 없습니다.")

        # 응답 시간 출력
        if st.session_state["response_times"]["qwen"]:
            st.write(
                f"응답 시간: {sum(st.session_state['response_times']['qwen']):.2f} 초"
            )
    else:
        st.markdown("# ~~💬 Qwen: Qwen2.5-72B-Instruct-Turbo~~")
        st.write(
            "해당 AI 모델은 비활성화되었습니다. 설정 탭에서 활성화 할 수 있습니다."
        )


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
            "llama_response": str(st.session_state.get("llama_responses", "")),
            "qwen_response": str(st.session_state.get("qwen_responses", "")),
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

        # 비동기 요약 실행
        summaries = asyncio.run(asyncio.gather(
            summarize_text_async(prompt),
            summarize_text_async(st.session_state["gpt_responses"][-1] if st.session_state["gpt_responses"] else ""),
            summarize_text_async(st.session_state["gemini_responses"][-1] if st.session_state["gemini_responses"] else "")
        ))

        # 요약 결과 출력
        ai_models = ["입력 프롬프트", "ChatGPT 응답", "Gemini 응답"]
        for model, summary in zip(ai_models, summaries):
            st.subheader(f"{model} 요약:")
            st.write(summary)

# 탭: WordCloud
with wordcloud_tab:
    st.title("☁️ WordCloud")
    if prompt:
        st.write("**입력된 프롬프트:**")
        st.info(prompt)

        # 비동기 WordCloud 실행
        wordcloud_image = asyncio.run(generate_wordcloud_async(prompt))
        if wordcloud_image:
            st.image(f"data:image/png;base64,{wordcloud_image}", use_column_width=True)
        else:
            st.write("WordCloud 생성 중 오류가 발생했습니다.")
    else:
        st.write("프롬프트를 입력한 후 WordCloud를 확인하세요.")
