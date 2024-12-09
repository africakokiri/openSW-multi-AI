# openSW - multi AI platform

2024 - 2학기 오픈소스 SW의 이해 7조 기말 프로젝트입니다.

- 20233245 장세인
- 20233251 정종현
- 20233253 조은서
- 20225271 최찬환

&nbsp;

## 주요 기능

- 하나의 프롬프트로 여러 AI들의 답변을 얻을 수 있습니다.
- AI별로 성능과 응답 시간이 상이하기 때문에, 원하는 AI만 사용할 수 있습니다.
- 사용자가 입력한 프롬프트와 그에 대한 AI의 답변을 기록합니다.

&nbsp;

## 설치 방법

> [!IMPORTANT]
>
> 본 레포지토리를 직접 설치하시려면, API KEY가 필요합니다.

1. 본 레포지토리를 clone 합니다.
   - `git clone https://github.com/africakokiri/openSW-multi-AI.git <경로>`
2. 가상환경을 사용하고 다음 명령어를 입력해 필요한 패키지를 설치합니다.
   - `pip(또는 pip3) install -r requirements.txt` 
3. Streamlit으로 프로그램을 실행합니다.
   - `streamlit run _main.py`

&nbsp;

## 사용 가능한 AI 모델

- OpenAI: **gpt**-4o-mini
- Google: **Gemini**-1.5-flash
- Anthropic: **Claude**-3.5-Sonnet
- Meta: **Llama**-3.2-90B-Vision-Instruct-Turbo
- Qwen: **Qwen**2.5-72B-Instruct-Turbo

&nbsp;

## 라이선스

- MIT 라이선스로 배포했습니다.
- 누구나 자유롭게 상업적 이용, 수정, 배포 가능합니다.

