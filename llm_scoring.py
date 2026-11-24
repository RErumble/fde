import os
import random
from pathlib import Path

import anthropic
from dotenv import load_dotenv

# C:\Users\hi\Documents\fde\.env 에서 CLAUDE_API_KEY를 읽음
load_dotenv(Path(r"C:\Users\hi\Documents\fde\.env"))

client = anthropic.Anthropic(api_key=os.environ["CLAUDE_API_KEY"])

SYSTEM_PROMPT = "AI 문의의 문의 정보 완성도를 0~2점으로 판정하세요."

SAMPLE_INQUIRIES = [
    "AI가 이상해요.",
    "챗봇 답변이 틀려요.",
    "사내 AI 챗봇에 업로드한 PDF(계약서_v3.pdf, 12MB)를 요약해 달라고 했더니 어제 오후 3시부터 '처리 중 오류가 발생했습니다'라는 메시지만 나옵니다. 크롬 최신 버전이고, 같은 파일을 어제 오전에는 정상 요약했습니다. 스크린샷 첨부합니다.",
    "AI 코드 어시스턴트를 쓰고 있는데 가끔 느려요.",
    "GPT 기반 번역 도구에서 영문 이메일을 한국어로 번역하면 존댓말이 아니라 반말로 나옵니다. 입력 예시: 'Please send me the report by Friday.' 출력: '금요일까지 보고서 보내.' 설정에서 어조를 바꾸는 옵션이 있는지 궁금합니다.",
    "AI 사용 권한 신청하고 싶어요.",
    "이미지 생성 AI에서 한국어 프롬프트를 넣으면 결과가 이상하게 나와요. 영어로 하면 괜찮은데 왜 그런가요?",
    "Claude를 API로 연동하려는데 키는 어디서 발급받나요?",
]

inquiry = random.choice(SAMPLE_INQUIRIES)

response = client.messages.create(
    model="claude-haiku-5-5",
    max_tokens=1024,
    system=SYSTEM_PROMPT,
    messages=[{"role": "user", "content": inquiry}],
)

print(f"[문의] {inquiry}")
for block in response.content:
    if block.type == "text":
        print(f"[응답] {block.text}")
