# 비동기 처리 함수: 각 AI의 응답을 반환할 때 시간을 포함한 튜플로 반환
async def fetch_all_responses_with_time(prompt):
    responses = await asyncio.gather(
        fetch_gpt_response(prompt),
        fetch_gemini_response(prompt),
        fetch_claude_response(prompt),
        fetch_llama_response(prompt),
        fetch_qwen_response(prompt),
    )

    # AI 이름 리스트
    ai_names = ["ChatGPT", "Gemini", "Claude", "Llama", "Qwen"]

    # 응답과 시간을 묶어 리스트로 반환
    return [
        {
            "name": ai_names[i],
            "response": responses[i],
            "time": st.session_state["response_times"][ai_names[i].lower()][-1],
        }
        for i in range(len(ai_names))
    ]


# 새로운 Prompt가 입력되었을 때 가장 짧은 응답 출력
if prompt:
    # 기존 prompt 기록에 새로운 prompt 추가
    st.session_state["prompt_history"].append(prompt)

    # 각 AI의 응답과 시간을 받아오기
    ai_responses_with_time = asyncio.run(fetch_all_responses_with_time(prompt))

    # 가장 빠른 응답 선택
    shortest_response = min(ai_responses_with_time, key=lambda x: x["time"])

    # 선택된 응답 출력
    st.markdown(f"### 가장 빠른 응답: {shortest_response['name']}")
    st.markdown(f"응답 내용:\n{shortest_response['response']}")
    st.markdown(f"응답 시간: {shortest_response['time']:.2f} 초")

    # 각 AI의 응답을 session_state에 기록
    for ai_response in ai_responses_with_time:
        if ai_response["name"] == "ChatGPT":
            st.session_state["gpt_responses"].append(ai_response["response"])
        elif ai_response["name"] == "Gemini":
            st.session_state["gemini_responses"].append(ai_response["response"])
        elif ai_response["name"] == "Claude":
            st.session_state["claude_responses"].append(ai_response["response"])
        elif ai_response["name"] == "Llama":
            st.session_state["llama_responses"].append(ai_response["response"])
        elif ai_response["name"] == "Qwen":
            st.session_state["qwen_responses"].append(ai_response["response"])
