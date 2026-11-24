"""로컬 JSON 저장소: 사용자, 사용자별 자산 DB, 세션(서버 쪽 상태), 요청 로그."""
import asyncio
import json
import re
import secrets
import time
from pathlib import Path

DATA_DIR = Path(__file__).parent / "data"
USER_ID_RE = re.compile(r"^[0-9]{1,6}$")  # 경로 조작 방지: 숫자만 허용


def load_users() -> dict:
    return json.loads((DATA_DIR / "users.json").read_text(encoding="utf-8"))


def get_user(user_id: str) -> dict | None:
    if not USER_ID_RE.match(user_id):
        return None
    return load_users().get(user_id)


def load_assets(user_id: str) -> dict:
    """해당 사용자의 자산 DB만 읽는다. 다른 사용자의 파일에는 접근하지 않는다."""
    if not USER_ID_RE.match(user_id):
        return {}
    path = DATA_DIR / "users" / user_id / "assets.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def append_request_log(user_id: str, record: dict) -> None:
    """원본 이력은 서버에 따로 보관한다. request_id로 결과와 로그를 연결한다."""
    if not USER_ID_RE.match(user_id):
        return
    path = DATA_DIR / "users" / user_id / "requests.jsonl"
    record = {"ts": time.strftime("%Y-%m-%d %H:%M:%S"), **record}
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def new_memory() -> dict:
    """단기 메모리: 확정 상태 + 미해결 문의 + 최근 대화 윈도우."""
    return {"confirmed": {"asset_id": None, "os": None}, "pending": [], "recent": []}


class Session:
    def __init__(self, user_id: str, user: dict):
        self.user_id = user_id  # 이 값이 곧 상태값: 모든 DB 조회는 이 ID로만 이뤄진다
        self.user = user
        self.memory = new_memory()
        self.display: list[dict] = []  # 화면 복원용 전체 메시지
        self.lock = asyncio.Lock()

    def reset(self) -> None:
        self.memory = new_memory()
        self.display = []


SESSIONS: dict[str, Session] = {}


def create_session(user_id: str, user: dict) -> str:
    token = secrets.token_urlsafe(24)
    SESSIONS[token] = Session(user_id, user)
    return token


def get_session(token: str | None) -> Session | None:
    return SESSIONS.get(token) if token else None


def drop_session(token: str | None) -> None:
    if token:
        SESSIONS.pop(token, None)
