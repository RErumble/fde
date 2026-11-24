"""IT 요청 챗봇 웹 서비스 (FastAPI 백엔드).

실행: cd webapp && ..\\venv\\Scripts\\python.exe -m uvicorn app:app --port 8000
"""
import json
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

import pipeline
import store

BASE = Path(__file__).parent
app = FastAPI(title="IT 요청 챗봇")
app.mount("/static", StaticFiles(directory=BASE / "static"), name="static")


class LoginIn(BaseModel):
    user_id: str = Field(min_length=1, max_length=16)


class ChatIn(BaseModel):
    message: str = Field(min_length=1, max_length=2000)


def require_session(request: Request) -> store.Session:
    session = store.get_session(request.cookies.get("sid"))
    if session is None:
        raise HTTPException(status_code=401, detail="로그인이 필요합니다.")
    return session


# ---------- 페이지 ----------
@app.get("/")
def login_page(request: Request):
    if store.get_session(request.cookies.get("sid")):
        return RedirectResponse("/chat")
    return FileResponse(BASE / "static" / "login.html")


@app.get("/chat")
def chat_page(request: Request):
    if store.get_session(request.cookies.get("sid")) is None:
        return RedirectResponse("/")
    return FileResponse(BASE / "static" / "chat.html")


@app.get("/favicon.ico", include_in_schema=False)
def favicon():
    return Response(status_code=204)


# ---------- 로그인 (ID만 입력받아 상태값으로 사용) ----------
@app.post("/api/login")
def login(body: LoginIn):
    user_id = body.user_id.strip()
    user = store.get_user(user_id)
    if user is None:
        raise HTTPException(status_code=404, detail="등록되지 않은 사용자 ID입니다.")
    token = store.create_session(user_id, user)
    resp = JSONResponse({"user_id": user_id, **user})
    resp.set_cookie("sid", token, httponly=True, samesite="lax", max_age=60 * 60 * 8)
    return resp


@app.post("/api/logout")
def logout(request: Request):
    store.drop_session(request.cookies.get("sid"))
    resp = JSONResponse({"ok": True})
    resp.delete_cookie("sid")
    return resp


@app.get("/api/me")
def me(request: Request):
    s = require_session(request)
    return {"user_id": s.user_id, **s.user, "memory": s.memory, "display": s.display}


@app.get("/api/assets")
def assets(request: Request):
    """로그인한 사용자 본인의 자산 목록만 돌려준다."""
    s = require_session(request)
    return [
        {"asset_id": a["asset_id"], "type": a["type"], "model": a["model"], "status": a["status"]}
        for a in store.load_assets(s.user_id).values()
    ]


@app.post("/api/reset")
def reset(request: Request):
    """새 대화: 단기 메모리를 비운다. 서버에 쌓인 요청 이력은 그대로 남는다."""
    s = require_session(request)
    s.reset()
    return {"ok": True, "memory": s.memory}


# ---------- 채팅 (SSE) ----------
@app.post("/api/chat")
async def chat(body: ChatIn, request: Request):
    s = require_session(request)
    message = body.message.strip()
    if not message:
        raise HTTPException(status_code=422, detail="메시지가 비어 있습니다.")

    async def events():
        if s.lock.locked():
            yield f"data: {json.dumps({'type': 'error', 'message': '이전 요청을 처리 중입니다.'}, ensure_ascii=False)}\n\n"
            return
        async with s.lock:
            async for ev in pipeline.run_turn(s, message):
                yield f"data: {json.dumps(ev, ensure_ascii=False)}\n\n"

    return StreamingResponse(
        events(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
