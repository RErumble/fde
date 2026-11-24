"""한 턴의 처리 흐름: 분류 -> 추출 -> 채점 -> function calling 응답 (SSE 이벤트를 yield).

컨텍스트 엔지니어링
- 고정 prefix(tools + system)는 llm.py에서 캐시되고, 이 파일은 매 턴 달라지는 동적 suffix(user 메시지)만 만든다.
- 모델에는 현재 요청, 확정 상태, 최근 대화 윈도우, 분석 결과(= 관련 사실)만 보낸다.
  전체 원본 이력은 서버(requests.jsonl)에 따로 보관하고 모델에는 보내지 않는다.
"""
import json
import re
import time
import uuid

import anthropic

import llm
import store
import tools

# '자산번호'라는 말이 없어도 NB-2024-0153, DT-0431 같은 형식이면 자산 ID로 본다
ASSET_RE = re.compile(r"(?<![A-Za-z0-9-])[A-Z]{2,3}-\d{2,4}(?:-\d{2,4})?(?![A-Za-z0-9-])")
RECENT_MAX = 8  # 단기 메모리: 최근 메시지 8개(4턴)만 모델에 전달
PENDING_MAX = 3
NONE = "없음"


def parse_category(text: str) -> str:
    m = re.search(r"최종 분류:[ \t]*([^\n]*)", text)
    return m.group(1).strip() if m else ""


def result_section(text: str) -> str:
    return text.split("## 결과")[-1].strip()


def field(extracted: str, name: str) -> str:
    m = re.search(rf"-\s*{name}:\s*(.+)", extracted)
    return m.group(1).strip() if m else NONE


def normalize_asset(value: str) -> str:
    m = ASSET_RE.search(value.upper())
    return m.group(0) if m else NONE


def clip(text: str, n: int) -> str:
    return text if len(text) <= n else text[:n] + "…"


def stage_done(stage: str, summary: str, usage: dict, ms: int, text_in: str, text_out: str) -> dict:
    return {"type": "stage", "stage": stage, "status": "done", "summary": summary,
            "usage": usage, "ms": ms, "input": text_in, "output": text_out}


def build_context(session: store.Session, message: str, prev_pending: list[str],
                  category: str, extracted: str, scored: str) -> str:
    """매 턴 달라지는 동적 suffix. 고정 지침은 chat.md(system)에 있고 여기에는 넣지 않는다."""
    user = session.user
    conf = session.memory["confirmed"]
    lines = [
        "## 사용자",
        f"- 이름: {user['name']} ({user['dept']})",
        "",
        "## 확정 상태",
        f"- 자산 ID: {conf['asset_id'] or NONE}",
        f"- OS: {conf['os'] or NONE}",
        "",
        "## 최근 대화",
    ]
    recent = session.memory["recent"]
    if recent:
        for r in recent:
            who = "사용자" if r["role"] == "user" else "챗봇"
            lines.append(f"- {who}: {clip(r['text'], 300)}")
    else:
        lines.append("(없음)")
    lines += ["", "## 현재 요청"]
    if prev_pending:
        lines.append("보류 중이던 이전 문의: " + " / ".join(prev_pending))
        lines.append("방금 보낸 메시지: " + message)
    else:
        lines.append(message)
    lines += ["", "## 분석 결과", f"- 1단계 분류: {category or '(빈값)'}",
              "- 2단계 추출:", extracted, "- 3단계 채점:", scored]
    return "\n".join(lines)


async def run_turn(session: store.Session, message: str):
    rid = uuid.uuid4().hex[:8]
    mem = session.memory
    conf = mem["confirmed"]
    prev_pending = list(mem["pending"])
    total_usage: dict = {}
    t_start = time.perf_counter()
    yield {"type": "start", "request_id": rid}

    try:
        # 분석 입력: 보류 중이던 문의 + 이번 메시지 (+ 확정 상태를 한 줄로)
        case = "\n".join(prev_pending + [message])
        if conf["asset_id"] and not ASSET_RE.search(case):
            known = f"자산번호 {conf['asset_id']}" + (f", OS {conf['os']}" if conf["os"] else "")
            case += f"\n(이전 대화에서 확정된 정보: {known})"

        # ---------- 1단계 분류 ----------
        yield {"type": "stage", "stage": "classify", "status": "running"}
        text, usage, ms = await llm.call_stage("classify", case, 2048)
        llm.add_usage(total_usage, usage)
        category = parse_category(text)
        yield stage_done("classify", category or "(빈값)", usage, ms, case, text)

        # ---------- 2단계 정보 추출 (분류 결과를 입력으로 사용) ----------
        yield {"type": "stage", "stage": "extract", "status": "running"}
        ext_in = f"**문의**: {case}\n**1단계 분류 결과**: {category if category else '(빈값 - 분류 불명확)'}"
        text, usage, ms = await llm.call_stage("extract", ext_in, 2048)
        llm.add_usage(total_usage, usage)
        extracted = result_section(text)
        asset_id = normalize_asset(field(extracted, "asset_id"))
        os_name = field(extracted, "os")
        yield stage_done("extract", f"{asset_id} · {os_name}", usage, ms, ext_in, text)

        # 단기 메모리(확정 상태) 갱신: 이 사용자의 DB에 실제로 있는 자산만 '확정'한다
        if asset_id != NONE and asset_id in store.load_assets(session.user_id):
            if conf["asset_id"] != asset_id:
                conf["os"] = None  # 장비가 바뀌면 이전 OS는 버린다
            conf["asset_id"] = asset_id
            if os_name != NONE:
                conf["os"] = os_name

        # ---------- 3단계 채점 (분류 + 추출 결과를 입력으로 사용) ----------
        yield {"type": "stage", "stage": "score", "status": "running"}
        sc_in = (f"**문의**: {case}\n**1단계 분류 결과**: {category if category else '(빈값 - 분류 불명확)'}\n"
                 f"**2단계 추출 결과**:\n{extracted}")
        text, usage, ms = await llm.call_stage("scoring", sc_in, 4096)
        llm.add_usage(total_usage, usage)
        scored = result_section(text)
        m = re.search(r"종합 점수:\s*(\d)", scored)
        yield stage_done("score", f"종합 {m.group(1)}점" if m else "종합 ?점", usage, ms, sc_in, text)

        # ---------- 4단계 function calling 응답 (스트리밍) ----------
        yield {"type": "stage", "stage": "chat", "status": "running"}
        context = build_context(session, message, prev_pending, category, extracted, scored)
        messages = [{"role": "user", "content": context}]
        reply_parts: list[str] = []
        chat_usage: dict = {}
        tool_log: list[dict] = []
        t_chat = time.perf_counter()
        rounds = 0
        for _ in range(5):  # 무한 루프 방지
            rounds += 1
            async with llm.client.messages.stream(
                model=llm.MODEL,
                max_tokens=4096,
                system=llm.system_blocks("chat"),
                tools=tools.TOOLS,
                messages=llm.with_rolling_cache(messages),
            ) as stream:
                round_text = ""
                async for delta in stream.text_stream:
                    if not round_text and reply_parts:
                        yield {"type": "delta", "text": "\n\n"}
                    round_text += delta
                    yield {"type": "delta", "text": delta}
                final = await stream.get_final_message()
            if round_text:
                reply_parts.append(round_text)
            llm.add_usage(chat_usage, llm.usage_dict(final.usage))
            if final.stop_reason != "tool_use":
                break
            messages.append({"role": "assistant", "content": final.content})
            results = []
            for block in final.content:
                if block.type != "tool_use":
                    continue
                out = tools.run_tool(session.user_id, block.name, block.input)  # user_id는 세션에서만
                tool_log.append({"name": block.name, "input": block.input, "result": out})
                yield {"type": "tool", "name": block.name, "input": block.input, "result": out}
                results.append({"type": "tool_result", "tool_use_id": block.id,
                                "content": json.dumps(out, ensure_ascii=False)})
            messages.append({"role": "user", "content": results})
        llm.add_usage(total_usage, chat_usage)
        reply = "\n\n".join(reply_parts).strip() or "죄송합니다. 응답을 만들지 못했습니다. 다시 시도해 주세요."
        calls = len(tool_log)
        yield stage_done("chat", f"함수 호출 {calls}회" if calls else "함수 호출 없음", chat_usage,
                         int((time.perf_counter() - t_chat) * 1000), context, reply) | {"rounds": rounds}

    except anthropic.APIError as e:
        yield {"type": "error", "message": f"모델 호출 중 오류가 났습니다 ({type(e).__name__}). 잠시 뒤 다시 시도해 주세요."}
        return

    # ---------- 턴 종료: 메모리와 서버 이력 갱신 ----------
    mem["recent"] = (mem["recent"] + [{"role": "user", "text": message},
                                      {"role": "assistant", "text": reply}])[-RECENT_MAX:]
    # 자산 ID가 아예 없을 때만 문의를 보류한다 (다음 메시지에 ID가 오면 합쳐서 다시 분석).
    # ID를 적었지만 DB에 없는 경우에는 보류하지 않는다: 잘못된 ID가 다음 분석에 섞이지 않게 한다.
    mem["pending"] = (prev_pending + [message])[-PENDING_MAX:] if (asset_id == NONE and not conf["asset_id"]) else []
    session.display += [{"role": "user", "text": message, "request_id": rid},
                        {"role": "assistant", "text": reply, "request_id": rid}]
    store.append_request_log(session.user_id, {
        "request_id": rid, "message": message, "analysis_input": case, "category": category,
        "asset_id": asset_id, "analysis_score": scored, "tool_calls": tool_log, "reply": reply,
        "usage": total_usage,
    })
    yield {"type": "memory", "memory": mem}
    yield {"type": "done", "request_id": rid, "usage": total_usage,
           "ms": int((time.perf_counter() - t_start) * 1000)}
