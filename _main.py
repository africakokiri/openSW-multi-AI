import streamlit as st


# 페이지 설정
st.set_page_config(layout="wide")


# 탭 구성
All, gpt_as_tab, gemini_as_tab, claude_as_tab, llama_as_tab = st.tabs(
    ["전체", "chatGPT", "Gemini", "Claude", "llama"]
)

# 탭: 전체
with All:
    gpt_as_col, gemini_as_col, claude_as_col, llama_as_col = st.columns(4)

    with gpt_as_col:
        st.write("GPT")

    with gemini_as_col:
        st.write("Gemini")

    with claude_as_col:
        st.write("Claude")

    with llama_as_col:
        st.write("LlaMA")


# 탭: chatGPT
with gpt_as_tab:
    st.title("💬 openAI: GPT-4o-mini")
    st.caption("🚀 A Streamlit chatbot powered by openAI ChatGPT")


# 탭: Gemini
with gemini_as_tab:
    st.title("💬 Google: Gemini-1.5-flash")
    st.caption("🚀 A Streamlit chatbot powered by Google Gemini")


# 탭: Claude
with claude_as_tab:
    st.title("💬 Anthropic: claude-3-5-sonnet")
    st.caption("🚀 A Streamlit chatbot powered by Anthropic Claude")


# 탭: llama
with llama_as_tab:
    st.title("💬 Meta: llama3.2-90b-vision")
    st.caption("🚀 A Streamlit chatbot powered by Meta LlaMA")


# 유저의 prompt를 state에 저장
prompt = st.chat_input("프롬프트를 입력하세요.")
st.session_state["prompt"] = prompt
