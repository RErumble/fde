"""Anthropic API 호출 계층 (프레임워크 없이 SDK만 사용).

프롬프트 캐싱 설계
- 렌더 순서는 tools -> system -> messages 이고, 캐시는 앞부분(prefix)이 바이트 단위로 같아야 적중한다.
- 그래서 tools와 system(프롬프트 .md 파일)은 요청마다 절대 바뀌지 않게 고정하고, system 끝에 cache_control을 둔다.
- 사용자 ID, 시각, 대화 내용처럼 요청마다 달라지는 값은 전부 messages(동적 suffix)에만 넣는다.
"""
import os
import time
from pathlib import Path

import anthropic
from dotenv import load_dotenv

load_dotenv(Path(os.environ.get("ENV_PATH", r"C:\Users\hi\Documents\fde\.env")))

MODEL = "claude-haiku-5-5"  # CLAUDE.md 규칙: 개발 시 API는 Haiku 5.5만 사용
client = anthropic.AsyncAnthropic(api_key=os.environ["CLAUDE_API_KEY"])

PROMPT_DIR = Path(__file__).parent / "prompts"
# 시작 시 한 번만 읽어 고정한다 (요청마다 같은 바이트를 보내야 캐시가 유지됨)
PROMPTS = {
    name: (PROMPT_DIR / f"{name}.md").read_text(encoding="utf-8")
    for name in ("classify", "extract", "scoring", "chat")
}

EPHEMERAL = {"type": "ephemeral"}


def system_blocks(name: str) -> list[dict]:
    return [{"type": "text", "text": PROMPTS[name], "cache_control": EPHEMERAL}]


def usage_dict(u) -> dict:
    return {
        "in": u.input_tokens or 0,
        "out": u.output_tokens or 0,
        "cache_write": getattr(u, "cache_creation_input_tokens", 0) or 0,
        "cache_read": getattr(u, "cache_read_input_tokens", 0) or 0,
    }


def add_usage(total: dict, u: dict) -> dict:
    for k in ("in", "out", "cache_write", "cache_read"):
        total[k] = total.get(k, 0) + u[k]
    return total


def with_rolling_cache(messages: list[dict]) -> list[dict]:
    """마지막 user 메시지의 마지막 블록에 cache_control을 붙인 사본을 돌려준다.

    함수 호출 루프에서 2번째 호출부터는 앞 호출에서 쓴 prefix(tools + system + 첫 user 메시지)를
    그대로 읽을 수 있다. 원본 messages는 건드리지 않는다.
    """
    out = list(messages)
    last = out[-1]
    content = last["content"]
    if isinstance(content, str):
        blocks = [{"type": "text", "text": content, "cache_control": EPHEMERAL}]
    else:
        blocks = list(content)
        blocks[-1] = {**blocks[-1], "cache_control": EPHEMERAL}
    out[-1] = {**last, "content": blocks}
    return out


async def call_stage(name: str, user_text: str, max_tokens: int) -> tuple[str, dict, int]:
    """1~3단계: 고정 system(캐시) + 동적 user 메시지 한 건."""
    t = time.perf_counter()
    resp = await client.messages.create(
        model=MODEL,
        max_tokens=max_tokens,
        system=system_blocks(name),
        messages=[{"role": "user", "content": user_text}],
    )
    text = "".join(b.text for b in resp.content if b.type == "text")
    return text, usage_dict(resp.usage), int((time.perf_counter() - t) * 1000)
