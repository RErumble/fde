import os
import random
from pathlib import Path

import anthropic
from dotenv import load_dotenv

# C:\Users\hi\Documents\fde\.env 에서 CLAUDE_API_KEY를 읽음
load_dotenv(Path(r"C:\Users\hi\Documents\fde\.env"))

client = anthropic.Anthropic(api_key=os.environ["CLAUDE_API_KEY"])

SYSTEM_PROMPT = (
    "IT 문의에서 asset_id, os, 증상을 추출해서 자연스러운 한국어 문장으로 알려주세요. "
    "문의에 없는 정보는 없다고 말하세요."
)

SAMPLE_INQUIRIES = [
    "자산번호 NB-2024-0153 노트북인데 윈도우 11에서 부팅하다가 파란 화면 뜨면서 계속 재시작돼요.",
    "맥북(asset: MB-0872, macOS Sonoma 14.5)에서 Zoom 켜면 화면 공유가 검은 화면으로만 나옵니다.",
    "제 PC 인터넷이 갑자기 안 돼요. 어제까지는 됐어요.",
    "DT-0431 데스크탑, Ubuntu 22.04 쓰는데 업데이트 이후에 소리가 안 나와요.",
    "윈도우 10 쓰는 노트북인데 엑셀이 열 때마다 멈추고 응답 없음이 떠요. 자산번호는 기억이 안 나요.",
    "NB-2023-0907 노트북 배터리가 완충해도 30분 만에 꺼집니다.",
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
