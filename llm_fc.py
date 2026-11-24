import json
import os
import random
import re
import sys
from pathlib import Path

import anthropic
from dotenv import load_dotenv

from llm_cls_v2 import classify  # 1단계: 분류
from llm_extract_v2 import extract, parse_category  # 2단계: 정보 추출
from llm_scoring_v2 import result_section, score  # 3단계: 채점

# C:\Users\hi\Documents\fde\.env 에서 CLAUDE_API_KEY를 읽음
load_dotenv(Path(r"C:\Users\hi\Documents\fde\.env"))

client = anthropic.Anthropic(api_key=os.environ["CLAUDE_API_KEY"])

# ---------------------------------------------------------------------------
# 로컬 JSON DB (더미 자산 데이터)
# ---------------------------------------------------------------------------
DB_PATH = Path(__file__).parent / "asset_db.json"


def load_db() -> dict:
    return json.loads(DB_PATH.read_text(encoding="utf-8"))


def get_asset_status(asset_id: str) -> dict:
    asset = load_db().get(asset_id)
    if asset is None:
        return {"found": False, "message": f"자산 DB에 {asset_id} 이(가) 없습니다."}
    info = {k: v for k, v in asset.items() if k != "known_issues"}
    return {"found": True, **info}


def get_known_issues(asset_id: str) -> dict:
    asset = load_db().get(asset_id)
    if asset is None:
        return {"found": False, "message": f"자산 DB에 {asset_id} 이(가) 없습니다."}
    return {"found": True, "asset_id": asset_id, "known_issues": asset["known_issues"]}


TOOL_FUNCTIONS = {
    "get_asset_status": get_asset_status,
    "get_known_issues": get_known_issues,
}

# ---------------------------------------------------------------------------
# Function calling 도구 정의 (Anthropic tools 형식)
# ---------------------------------------------------------------------------
TOOLS = [
    {
        "name": "get_asset_status",
        "description": (
            "자산 ID로 사내 자산 DB에서 장비의 현재 상태를 조회합니다. "
            "장비 종류, 모델, OS, 소속 부서, 상태(정상/점검필요/수리중), 보증 만료일, 마지막 점검일을 반환합니다. "
            "자산 ID가 DB에 없으면 found가 false입니다."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "asset_id": {
                    "type": "string",
                    "description": "조회할 자산 ID. 예: NB-2024-0153, DT-0431",
                }
            },
            "required": ["asset_id"],
            "additionalProperties": False,
        },
        "strict": True,
    },
    {
        "name": "get_known_issues",
        "description": (
            "자산 ID로 해당 장비에 등록된 알려진 장애 목록을 조회합니다. "
            "각 장애의 번호, 제목, 심각도, 처리 상태(진행중/해결됨), 최초 접수일, 임시 해결 방법을 반환합니다. "
            "자산 ID가 DB에 없으면 found가 false이고, 등록된 장애가 없으면 known_issues가 빈 목록입니다."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "asset_id": {
                    "type": "string",
                    "description": "조회할 자산 ID. 예: NB-2024-0153, DT-0431",
                }
            },
            "required": ["asset_id"],
            "additionalProperties": False,
        },
        "strict": True,
    },
]

FC_SYSTEM_PROMPT = """# 역할
당신은 사내 IT 헬프데스크 상담 담당자입니다. 앞선 분석 결과와 자산 DB 조회 결과를 바탕으로 사용자 문의에 답합니다.

# 입력
사용자 메시지에는 아래 내용이 들어옵니다.
- **문의**: 사용자가 보낸 원문
- **1단계 분류 결과**: 장애, 권한, 사용법, 기타 중 하나이거나 빈값
- **2단계 추출 결과**: asset_id, os, 증상 ("없음"은 문의에 해당 정보가 없다는 뜻)
- **3단계 채점 결과**: 문의 정보 완성도 종합 점수와 보완 요청 사항

# 함수 사용 규칙
- 추출 결과의 asset_id가 있으면 `get_asset_status`와 `get_known_issues`를 모두 호출해 자산 상태와 알려진 장애를 확인합니다.
- asset_id가 "없음"이면 함수를 호출하지 않습니다. 자산 ID를 추측해서 호출하지 않습니다.
- 함수 결과의 found가 false이면 DB에 해당 자산이 없다고 알립니다. 자산 정보를 지어내지 않습니다.

# 응답 규칙
- 조회 결과가 있으면 그 내용을 근거로 답합니다. 자산의 상태, 보증 정보, 알려진 장애를 문의 증상과 연결해서 설명합니다.
- 문의 증상이 알려진 장애와 일치하면 장애 번호, 처리 상태, 임시 해결 방법을 안내합니다.
- 알려진 장애가 없거나 증상과 맞지 않으면 신규 장애로 접수가 필요하다고 안내합니다.
- 해결됨 상태의 장애는 해결 방법을 안내하되, 같은 증상이 계속되면 재접수가 필요하다고 덧붙입니다.
- 3단계 채점 결과에 보완 요청 사항이 있으면 응답 끝에 필요한 정보를 간단히 요청합니다.
- 문의에 없거나 조회되지 않은 내용은 추측하지 않습니다.

# 응답 형식
마크다운 없이, 상담 담당자가 말하듯 자연스러운 한국어 문장으로 간결하게 답합니다.
"""

# 자산 DB에 맞춘 샘플 문의 (DB에 없는 자산, 자산 ID가 없는 문의도 포함)
SAMPLE_INQUIRIES = [
    "자산번호 NB-2024-0153 노트북인데 윈도우 11에서 부팅하다가 파란 화면 뜨면서 계속 재시작돼요.",
    "맥북(asset: MB-0872, macOS Sonoma 14.5)에서 Zoom 켜면 화면 공유가 검은 화면으로만 나옵니다.",
    "DT-0431 데스크탑, Ubuntu 22.04 쓰는데 업데이트 이후에 소리가 안 나와요.",
    "NB-2023-0907 노트북 배터리가 완충해도 30분 만에 꺼집니다. 배터리 교체 요청은 아직 안 했어요.",
    "자산번호 NB-2022-0310 노트북이고 윈도우 10인데 와이파이가 오늘 또 끊겼어요. 어제도 그랬어요.",
    "PR-0058 프린터에서 출력 버튼을 눌렀는데 아무 반응이 없어요. 어제부터 그래요.",
    "자산번호 NB-2024-0220, Windows 11 노트북에서 사내 AI 코드 어시스턴트 자동완성이 오늘 오전부터 몇 초씩 늦게 떠요.",
    "자산번호 NB-2025-9999 노트북에서 사내 AI 번역 도구에 파일을 올리면 어제 오후부터 오류가 납니다.",
    "사내 AI 챗봇 답변이 틀려요.",
]


def run_function_calling(inquiry: str, category: str, extracted: str, scored: str) -> str:
    """도구를 붙여 호출하고, tool_use가 나오면 실행해 결과를 돌려주는 루프."""
    category_text = category if category else "(빈값 - 분류 불명확)"
    user_message = (
        f"**문의**: {inquiry}\n"
        f"**1단계 분류 결과**: {category_text}\n"
        f"**2단계 추출 결과**:\n{extracted}\n"
        f"**3단계 채점 결과**:\n{scored}"
    )
    messages = [{"role": "user", "content": user_message}]

    for _ in range(5):  # 무한 루프 방지
        response = client.messages.create(
            model="claude-haiku-5-5",
            max_tokens=4096,
            system=FC_SYSTEM_PROMPT,
            tools=TOOLS,
            messages=messages,
        )

        if response.stop_reason != "tool_use":
            return "".join(b.text for b in response.content if b.type == "text")

        # assistant 응답(tool_use 블록 포함)을 그대로 대화에 추가
        messages.append({"role": "assistant", "content": response.content})

        # 모든 tool_use를 실행하고 결과를 한 번에 user 메시지로 돌려줌
        tool_results = []
        for block in response.content:
            if block.type != "tool_use":
                continue
            result = TOOL_FUNCTIONS[block.name](**block.input)
            print(f"[Function Calling] {block.name}({json.dumps(block.input, ensure_ascii=False)})")
            print(f"  -> {json.dumps(result, ensure_ascii=False)}")
            tool_results.append(
                {
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": json.dumps(result, ensure_ascii=False),
                }
            )
        messages.append({"role": "user", "content": tool_results})

    return "(도구 호출 반복 횟수를 초과했습니다.)"


if __name__ == "__main__":
    # 인자로 샘플 번호(0부터)를 주면 해당 문의를 사용, 없으면 랜덤
    if len(sys.argv) > 1:
        inquiry = SAMPLE_INQUIRIES[int(sys.argv[1])]
    else:
        inquiry = random.choice(SAMPLE_INQUIRIES)
    print(f"[문의] {inquiry}\n")

    # 1단계: 분류
    category = parse_category(classify(inquiry))
    print(f"[1단계 분류 결과] {category if category else '(빈값)'}\n")

    # 2단계: 정보 추출
    extracted = result_section(extract(inquiry, category))
    print(f"[2단계 추출 결과]\n{extracted}\n")

    # 3단계: 채점
    scored = result_section(score(inquiry, category, extracted))
    match = re.search(r"종합 점수:\s*(\d)", scored)
    print(f"[3단계 채점 결과] 종합 점수: {match.group(1) if match else '?'}\n")

    # 4단계: Function calling으로 자산 상태와 알려진 장애 조회 후 응답
    print("[4단계 Function Calling]")
    answer = run_function_calling(inquiry, category, extracted, scored)
    print(f"\n[최종 응답]\n{answer}")
