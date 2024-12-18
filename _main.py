import streamlit as st
from textblob import TextBlob

# 감정 분석을 기반으로 AI 선택
def analyze_emotion(text):
    analysis = TextBlob(text)
    polarity = analysis.sentiment.polarity
    
    if polarity > 0:
        return "positive"
    elif polarity < 0:
        return "negative"
    else:
        return "neutral"

# 각 감정별 AI 답변
def get_ai_response(emotion, user_input):
    if emotion == "positive":
        return f"🤖 [AI 1 - 긍정형 AI]: '{user_input}'에 대한 긍정적인 답변을 준비했습니다! 항상 긍정적인 마음이 중요하죠!"
    elif emotion == "negative":
        return f"🤖 [AI 2 - 공감형 AI]: '{user_input}'에 대해 공감합니다. 무언가 힘들거나 해결이 필요하다면 제가 도와드릴게요."
    else:
        return f"🤖 [AI 3 - 중립형 AI]: '{user_input}'에 대한 중립적이고 정확한 정보를 제공하겠습니다."

# Streamlit 웹앱 UI 구성
def main():
    st.title("🎭 감정 분석 기반 AI 활성화 시스템")
    st.write("사용자의 입력된 질문을 감정 분석하여, 상황에 맞는 AI가 답변을 제공합니다.")
    
    # 사용자 입력 받기
    user_input = st.text_input("질문을 입력해주세요:", "")
    
    if user_input:
        # 감정 분석
        emotion = analyze_emotion(user_input)
        
        # 감정 분석 결과 표시
        st.subheader("감정 분석 결과:")
        if emotion == "positive":
            st.success("긍정적인 감정으로 분석되었습니다! 😊")
        elif emotion == "negative":
            st.error("부정적인 감정으로 분석되었습니다. 😢")
        else:
            st.info("중립적인 감정으로 분석되었습니다. 😐")
        
        # 상황에 맞는 AI 답변 제공
        st.subheader("AI 답변:")
        response = get_ai_response(emotion, user_input)
        st.write(response)

if __name__ == "__main__":
    main()
