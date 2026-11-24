(function () {
  "use strict";
  const $ = (s) => document.querySelector(s);
  const STAGES = [
    { key: "classify", n: "①", name: "분류" },
    { key: "extract", n: "②", name: "정보 추출" },
    { key: "score", n: "③", name: "채점" },
    { key: "chat", n: "④", name: "응답 생성 (함수 호출)" },
  ];
  const STARTERS = [
    "프린터 출력 버튼을 눌렀는데 아무 반응이 없어요.",
    "노트북이 부팅하다가 파란 화면이 떠요.",
    "AI 사용 권한 신청하고 싶어요.",
  ];
  const fmt = (n) => Number(n || 0).toLocaleString("ko-KR");
  const el = (tag, cls, text) => {
    const e = document.createElement(tag);
    if (cls) e.className = cls;
    if (text !== undefined) e.textContent = text;
    return e;
  };

  const messagesEl = $("#messages");
  const inputEl = $("#input");
  const sendBtn = $("#send");
  let busy = false;
  let sessionUsage = { in: 0, out: 0, cache_write: 0, cache_read: 0 };

  /* ---------- 메시지 ---------- */
  function clearEmpty() {
    const e = messagesEl.querySelector(".empty");
    if (e) e.remove();
  }
  function showEmpty() {
    messagesEl.textContent = "";
    const box = el("div", "empty");
    box.append(el("h2", "", "무엇을 도와드릴까요?"));
    box.append(el("p", "", "장비 문제, 권한 신청, 사용법 등 IT 문의를 남겨 주세요. 자산번호를 알려 주시면 내 자산의 상태와 알려진 장애를 함께 확인합니다."));
    const row = el("div", "starters");
    STARTERS.forEach((t) => {
      const b = el("button", "starter", t);
      b.type = "button";
      b.addEventListener("click", () => { inputEl.value = t; inputEl.focus(); });
      row.append(b);
    });
    box.append(row);
    messagesEl.append(box);
  }
  function addMessage(role, text, requestId) {
    clearEmpty();
    const wrap = el("div", "msg " + (role === "user" ? "user" : "bot"));
    const bubble = el("div", "bubble", text || "");
    wrap.append(bubble);
    if (requestId) wrap.append(el("div", "meta", "request_id " + requestId));
    messagesEl.append(wrap);
    messagesEl.scrollTop = messagesEl.scrollHeight;
    return { wrap, bubble };
  }
  function typingBubble() {
    const t = el("span", "typing");
    t.append(el("i"), el("i"), el("i"), document.createTextNode(" 분석 중"));
    return t;
  }

  /* ---------- 처리 과정 패널 ---------- */
  function buildStages() {
    const box = $("#stages");
    box.textContent = "";
    STAGES.forEach((s) => {
      const row = el("div", "stage");
      row.id = "stage-" + s.key;
      const btn = el("button");
      btn.type = "button";
      btn.append(el("span", "dot"), el("span", "nm", s.n + " " + s.name), el("span", "ms", "대기"));
      btn.append(el("span", "sm", ""), el("span", "cache", ""));
      btn.addEventListener("click", () => row.classList.toggle("open"));
      const body = el("div", "body");
      row.append(btn, body);
      box.append(row);
    });
  }
  function setStage(key, patch) {
    const row = $("#stage-" + key);
    if (!row) return;
    const btn = row.querySelector("button");
    if (patch.status) {
      row.classList.remove("running", "done");
      row.classList.add(patch.status);
      btn.querySelector(".ms").textContent = patch.status === "running" ? "실행 중" : (patch.ms ? fmt(patch.ms) + " ms" : "완료");
    }
    if (patch.summary !== undefined) btn.querySelector(".sm").textContent = patch.summary;
    if (patch.usage) {
      const u = patch.usage;
      const c = btn.querySelector(".cache");
      c.textContent = "";
      if (u.cache_read) c.append(el("span", "tag-read", "캐시 읽기 " + fmt(u.cache_read)));
      if (u.cache_write) c.append(el("span", "tag-write", "캐시 쓰기 " + fmt(u.cache_write)));
      c.append(el("span", "", "입력 " + fmt(u.in) + " · 출력 " + fmt(u.out)));
    }
    if (patch.input !== undefined) {
      const body = row.querySelector(".body");
      body.textContent = "";
      body.append(el("div", "lbl", "입력 (동적 suffix)"), el("pre", "", patch.input));
      body.append(el("div", "lbl", "출력"), el("pre", "", patch.output));
    }
  }
  function addToolCall(ev) {
    const row = $("#stage-chat");
    let list = row.querySelector(".tools");
    if (!list) {
      list = el("div", "tools");
      row.append(list);
      list.style.padding = "0 10px 10px";
    }
    const found = ev.result && ev.result.found;
    const t = el("div", "tool" + (found ? "" : " miss"), ev.name + "(" + (ev.input.asset_id || "") + ") → found: " + found);
    list.append(t);
  }
  function resetStages() {
    buildStages();
    $("#req-id").textContent = "";
  }

  /* ---------- 캐시 / 메모리 / 자산 ---------- */
  function renderCache(u) {
    const total = u.in + u.cache_write + u.cache_read;
    const bar = $("#cache-bar");
    bar.textContent = "";
    const pct = (n) => (total ? (n / total) * 100 : 0);
    [["r", u.cache_read], ["w", u.cache_write], ["n", u.in]].forEach(([c, n]) => {
      const s = el("span", c); s.style.width = pct(n) + "%"; bar.append(s);
    });
    const lg = $("#cache-legend");
    lg.textContent = "";
    [["r", "캐시 읽기 (0.1배)", u.cache_read], ["w", "캐시 쓰기 (1.25배)", u.cache_write], ["n", "일반 입력", u.in]].forEach(([c, k, v]) => {
      lg.append(el("span", "k " + c, k), el("span", "v", fmt(v)));
    });
    const ratio = total ? Math.round((u.cache_read / total) * 100) : 0;
    const st = sessionUsage;
    const stTotal = st.in + st.cache_write + st.cache_read;
    const stRatio = stTotal ? Math.round((st.cache_read / stTotal) * 100) : 0;
    $("#cache-total").textContent = "이번 요청 적중률 " + ratio + "% · 이 대화 누적 " + stRatio + "%";
  }
  function renderMemory(m) {
    const dl = $("#memory");
    dl.textContent = "";
    const rows = [
      ["확정 자산 ID", m.confirmed.asset_id || "없음"],
      ["확정 OS", m.confirmed.os || "없음"],
      ["보류 중인 문의", m.pending.length + "건"],
      ["최근 대화 윈도우", Math.floor(m.recent.length / 2) + "턴 (최대 4턴)"],
    ];
    rows.forEach(([k, v]) => { dl.append(el("dt", "", k), el("dd", "", v)); });
  }
  async function loadAssets() {
    const res = await fetch("/api/assets");
    if (!res.ok) return;
    const list = await res.json();
    const box = $("#assets");
    box.textContent = "";
    list.forEach((a) => {
      const b = el("button", "asset");
      b.type = "button";
      b.append(el("b", "", a.asset_id), el("span", "", a.type + " · " + a.model));
      b.append(el("span", "st" + (a.status === "정상" ? "" : " warn"), a.status));
      b.addEventListener("click", () => {
        inputEl.value = (inputEl.value ? inputEl.value + " " : "") + "자산번호 " + a.asset_id;
        inputEl.focus();
      });
      box.append(b);
    });
  }

  /* ---------- 전송 / SSE ---------- */
  function setBusy(b) {
    busy = b;
    sendBtn.disabled = b;
    inputEl.disabled = b;
    $("#btn-new").disabled = b;
  }

  function handleEvent(ev, ctx) {
    switch (ev.type) {
      case "start":
        $("#req-id").textContent = "request_id " + ev.request_id;
        ctx.requestId = ev.request_id;
        break;
      case "stage":
        if (ev.status === "running") setStage(ev.stage, { status: "running" });
        else setStage(ev.stage, { status: "done", summary: ev.summary, ms: ev.ms, usage: ev.usage, input: ev.input, output: ev.output });
        break;
      case "tool":
        addToolCall(ev);
        break;
      case "delta":
        if (!ctx.started) { ctx.started = true; ctx.bubble.textContent = ""; }
        ctx.text += ev.text;
        ctx.bubble.textContent = ctx.text;
        messagesEl.scrollTop = messagesEl.scrollHeight;
        break;
      case "memory":
        renderMemory(ev.memory);
        break;
      case "done":
        ["in", "out", "cache_write", "cache_read"].forEach((k) => { sessionUsage[k] += ev.usage[k] || 0; });
        renderCache(ev.usage);
        if (ctx.requestId) ctx.wrap.append(el("div", "meta", "request_id " + ctx.requestId));
        break;
      case "error":
        ctx.bubble.textContent = ev.message;
        ctx.failed = true;
        break;
    }
  }

  async function send(text) {
    if (busy || !text.trim()) return;
    setBusy(true);
    addMessage("user", text);
    const { wrap, bubble } = addMessage("bot", "");
    bubble.append(typingBubble());
    resetStages();
    const ctx = { wrap, bubble, text: "", started: false, requestId: "", failed: false };
    try {
      const res = await fetch("/api/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message: text }),
      });
      if (res.status === 401) { location.href = "/"; return; }
      if (!res.ok) { bubble.textContent = "요청을 처리하지 못했습니다. (" + res.status + ")"; return; }
      const reader = res.body.getReader();
      const dec = new TextDecoder();
      let buf = "";
      for (;;) {
        const { done, value } = await reader.read();
        if (done) break;
        buf += dec.decode(value, { stream: true });
        let idx;
        while ((idx = buf.indexOf("\n\n")) >= 0) {
          const chunk = buf.slice(0, idx);
          buf = buf.slice(idx + 2);
          const line = chunk.split("\n").find((l) => l.startsWith("data: "));
          if (line) handleEvent(JSON.parse(line.slice(6)), ctx);
        }
      }
      if (!ctx.started && !ctx.failed) bubble.textContent = "응답이 비어 있습니다. 다시 시도해 주세요.";
    } catch (e) {
      bubble.textContent = "서버와 연결이 끊겼습니다. 잠시 뒤 다시 시도해 주세요.";
    } finally {
      setBusy(false);
      inputEl.focus();
    }
  }

  /* ---------- 초기화 ---------- */
  $("#composer").addEventListener("submit", (e) => {
    e.preventDefault();
    const t = inputEl.value;
    inputEl.value = "";
    inputEl.style.height = "";
    send(t);
  });
  inputEl.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey && !e.isComposing) {
      e.preventDefault();
      $("#composer").requestSubmit();
    }
  });
  inputEl.addEventListener("input", () => {
    inputEl.style.height = "auto";
    inputEl.style.height = Math.min(inputEl.scrollHeight, 140) + "px";
  });
  $("#btn-logout").addEventListener("click", async () => {
    await fetch("/api/logout", { method: "POST" });
    location.href = "/";
  });
  $("#btn-new").addEventListener("click", async () => {
    const res = await fetch("/api/reset", { method: "POST" });
    if (!res.ok) return;
    sessionUsage = { in: 0, out: 0, cache_write: 0, cache_read: 0 };
    showEmpty();
    resetStages();
    renderMemory((await res.json()).memory);
  });

  (async function init() {
    buildStages();
    const res = await fetch("/api/me");
    if (!res.ok) { location.href = "/"; return; }
    const me = await res.json();
    $("#who").textContent = me.name + " · " + me.dept;
    const chip = el("span", "chip-id", "ID " + me.user_id);
    $("#who").append(" ", chip);
    renderMemory(me.memory);
    renderCache({ in: 0, out: 0, cache_write: 0, cache_read: 0 });
    if (me.display.length) {
      messagesEl.textContent = "";
      me.display.forEach((m) => addMessage(m.role, m.text, m.role === "assistant" ? m.request_id : ""));
    } else {
      showEmpty();
    }
    loadAssets();
    inputEl.focus();
  })();
})();
