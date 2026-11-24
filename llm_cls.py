import os
import random
from pathlib import Path

import anthropic
from dotenv import load_dotenv

# C:\Users\hi\Documents\fde\.env 에서 CLAUDE_API_KEY를 읽음
load_dotenv(Path(r"C:\Users\hi\Documents\fde\.env"))

client = anthropic.Anthropic(api_key=os.environ["CLAUDE_API_KEY"])

SYSTEM_PROMPT = (
    "당신은 사내 IT 헬프데스크 담당자입니다. "
    "사용자가 IT 문의를 보내면 그 문의가 '장애', '권한', '사용법', '기타' 네 가지 중 "
    "어디에 해당하는지 판별하세요. "
    "장애는 시스템이나 장비가 고장났거나 동작하지 않는 경우, "
    "권한은 계정이나 접근 권한, 승인과 관련된 경우, "
    "사용법은 도구나 기능을 어떻게 쓰는지 묻는 경우, "
    "기타는 위 셋에 해당하지 않는 경우입니다. "
    "응답은 마크다운이나 JSON 같은 형식 없이, 분류 결과와 그렇게 판단한 이유를 "
    "자연스러운 한국어 문장으로 간단히 말해 주세요."
)

SAMPLE_INQUIRIES = [
    "오늘 아침부터 사내 VPN에 접속이 안 돼요. 계속 타임아웃이 납니다.",
    "신규 입사자인데 공유 드라이브 접근 권한을 받으려면 어떻게 해야 하나요?",
    "회의실 화상회의 장비에서 Zoom 화면 공유는 어떻게 하나요?",
    "노트북이 부팅 중에 파란 화면이 뜨면서 계속 재시작돼요.",
    "슬랙 관리자 권한을 팀장님께 부여해 주실 수 있나요?",
    "엑셀에서 피벗 테이블 만드는 방법을 알려주세요.",
    "다음 달 사무실 이전 때 내 자리 모니터는 누가 옮겨주나요?",
    "프린터에서 출력 버튼을 눌렀는데 아무 반응이 없어요.",
    "퇴사한 팀원의 메일함을 제가 열람할 수 있게 해주세요.",
    "사내 메신저에서 예약 메시지 보내는 기능은 어디 있나요?",
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
