import asyncio
import time
from textblob import TextBlob
import streamlit as st

# Mock AI 응답 함수 (실제 AI API로 대체 가능)
async def fetch_gpt_response(prompt):
    await asyncio.sleep(1.5)  # 응답 지연 시뮬레이션
    return f"ChatGPT의 답변: {prompt}에 대해 긍정적인 답을 제안합니다."

async def fetch_gemini_response(prompt):
    await asyncio.sleep(2.0)  # 응답 지연 시뮬레이션
    return f"Gemini의 답변: {prompt}에 대한 논리적인 접근을 제공합니다."

async def fetch_claude_response(prompt):
    await asyncio.sleep(1.0)  # 응답 지연 시뮬레이션
    return f"Claude의 답변: {prompt}에 대해 깊은 공감을 표현합니다."

async def fetch_llama_response(prompt):
    await asyncio.sleep(1.2)  # 응답 지연 시뮬레이션
    return f"Llama의 답변: {prompt}에 대해 창의적인 아이디어를 제공합니다."

async def fetch_qwen_response(prompt):
    await asyncio.sleep(1.8)  # 응답 지연 시뮬레이션
    return f"Qwen의 답변: {prompt}에 대한 중립적 정보를 제공합니다."

# 감정 분석 함수
def analyze_emotion(text):
    analysis = TextBlob(text)
    polarity = analysis.sentiment.polarity

    if polarity > 0:
        return "positive"
    elif polarity < 0:
        return "negative"
    else:
        return "neutral"

# 비동기 처리 함수: 각 AI의 응답과 응답 시간을 반환
async def fetch_all_responses_with_time(prompt):
    start_times = {}
    responses = []

    # AI 응답 호출
    ai_responses = await asyncio.gather(
        fetch_gpt_response(prompt),
        fetch_gemini_response(prompt),
        fetch_claude_response(prompt),
        fetch_llama_response(prompt),
        fetch_qwen_response(prompt),
    )

    # 응답 시간을 측정
    for i, response in enumerate(ai_responses):
        start_times[i] = time.time()  # 시작 시간 저장
        responses.append(response)

    ai_names = ["ChatGPT", "Gemini", "Claude", "Llama", "Qwen"]
    response_times = [time.time() - start_times.get(i, 0) for i in range(len(ai_names))]

    # 응답과 시간 묶어 반환
    return [
        {"name": ai_names[i], "response": responses[i], "time": response_times[i]}
        for i in range(len(ai_names))
    ]

# Streamlit 앱
def main():
    st.title("🎭 감정 분석 기반 AI 응답 최적화")
    st.write("질문을 입력하면 감정을 분석한 뒤, 각 AI의 응답을 받아 최적의 답변을 제공합니다.")

    # 사용자 입력
    prompt = st.text_input("질문을 입력하세요:", "")

    if prompt:
        # 감정 분석
        emotion = analyze_emotion(prompt)

        # 감정 분석 결과 출력
        st.subheader("감정 분석 결과:")
        if emotion == "positive":
            st.success("긍정적인 감정으로 분석되었습니다! 😊")
        elif emotion == "negative":
            st.error("부정적인 감정으로 분석되었습니다. 😢")
        else:
            st.info("중립적인 감정으로 분석되었습니다. 😐")

        # AI 응답 가져오기
        ai_responses_with_time = asyncio.run(fetch_all_responses_with_time(prompt))

        # 감정에 따라 AI 추천 (예: 부정적이면 공감형 AI 우선)
        if emotion == "positive":
            selected_response = next((resp for resp in ai_responses_with_time if resp["name"] == "ChatGPT"), None)
        elif emotion == "negative":
            selected_response = next((resp for resp in ai_responses_with_time if resp["name"] == "Claude"), None)
        else:
            selected_response = next((resp for resp in ai_responses_with_time if resp["name"] == "Qwen"), None)

        # 최적의 응답 출력
        st.subheader("최적의 AI 응답:")
        if selected_response:
            st.markdown(f"**{selected_response['name']}의 답변**")
            st.markdown(selected_response["response"])
            st.markdown(f"응답 시간: {selected_response['time']:.2f} 초")

        # 모든 AI의 응답 출력
        st.subheader("모든 AI 응답:")
        for ai_response in ai_responses_with_time:
            st.markdown(f"**{ai_response['name']}**: {ai_response['response']} (응답 시간: {ai_response['time']:.2f} 초)")

if __name__ == "__main__":
    main()
