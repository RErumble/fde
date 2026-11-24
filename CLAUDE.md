# CLAUDE.md

## LLM API 사용 규칙

- 개발 시 Claude API 호출은 전부 **Claude Haiku 5.5** (`claude-haiku-5-5`) 모델만 사용한다.
- 다른 모델(Opus, Sonnet, Fable 등)은 사용자가 명시적으로 요청하지 않는 한 코드에 넣지 않는다.

## 환경

- Python 가상환경은 `venv`를 사용한다. 실행: `venv\Scripts\python.exe <파일>`
- API 키는 `C:\Users\hi\Documents\fde\.env`의 `CLAUDE_API_KEY`에서 읽는다. 키 값은 출력하거나 커밋하지 않는다.
