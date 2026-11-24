"""정보 추출(extract)과 스코어링(score) 평가 태스크 정의. run_eval.py가 불러와 TASKS에 등록한다.

- extract: asset_id, os는 규칙 기반으로 비교하고, 증상은 정답 필수 요소를 기준으로 LLM 판정(Haiku 5.5)으로 비교한다.
- score: 종합 점수(0~2)의 accuracy와 MAE. v1과 v2 모두 종합 점수를 내므로 이 항목이 공통 비교 기준이다.
- variant v1은 기존 단일 프롬프트 한 번 호출, v2는 llm_*_v2.py와 같은 체인(분류 -> 추출 -> 채점)을 그대로 호출한다.
"""
import json
import re
import time
from pathlib import Path

NONE = "없음"
ASSET_RE = re.compile(r"(?<![A-Za-z0-9-])[A-Z]{2,3}-\d{2,4}(?:-\d{2,4})?(?![A-Za-z0-9-])")

READ_SYSTEM = """아래는 IT 문의에서 정보를 추출한 어떤 모델의 응답입니다. 이 응답이 최종적으로 밝힌 값만 그대로 옮겨 주세요. 새로 추측하거나 보충하지 마세요.

- asset_id: 응답이 이 문의의 자산번호(장비 식별 코드)라고 밝힌 값. 없다고 했거나 언급이 없으면 없음
- os: 응답이 사용자의 운영체제라고 밝힌 값(예: Windows 11, macOS Sonoma 14.5). 없다고 했거나 언급이 없으면 없음
- "예: Windows 11처럼" 같은 예시, 사용자에게 알려 달라고 요청하는 문장, 일반 안내문에 나온 값은 값으로 보지 않습니다.
- 응답이 장비 이름(맥북 등)만 말하고 운영체제를 밝히지 않았으면 os는 없음입니다. 응답이 운영체제를 직접 적었다면 그대로 옮깁니다.

출력은 아래 두 줄만 씁니다.
asset_id: <값 또는 없음>
os: <값 또는 없음>
"""

JUDGE_SYSTEM = """당신은 정보 추출 결과를 채점하는 평가자입니다. 문의에서 '증상'(사용자가 겪는 문제나 요청 내용)을 모델이 제대로 추출했는지 판정합니다.

# 입력
- 문의: 사용자가 보낸 원문
- 정답 증상: 정답으로 정한 요약
- 필수 요소: 모델 응답이 반드시 담아야 하는 핵심 내용 (없으면 "(없음)")
- 모델 응답: 평가 대상. 자산번호나 OS 같은 다른 항목이 함께 적혀 있어도 증상이나 요청 내용만 보고 판단합니다.

# 판정 기준
1. 필수 요소가 있으면, 모델 응답이 모든 필수 요소를 의미상 담고 있어야 정답입니다. 표현이 달라도 됩니다.
2. 모델 응답이 문의에 없는 내용을 지어내거나 정답과 모순되면 오답입니다. 문의에 있는 부가 정보를 덧붙인 것은 허용합니다.
3. 필수 요소가 "(없음)"이면, 모델 응답이 증상이나 요청이 없다는 뜻이거나 구체적인 내용을 적지 않았을 때만 정답입니다. 문제나 요청을 지어내면 오답입니다.
4. 되묻는 문장이나 안내 문구는 무시합니다.

# 출력 형식
판정: 정답 또는 오답
이유: 한 문장
"""


def build_tasks(RE):
    ROOT, EVALS, MODEL = RE.ROOT, RE.EVALS, RE.MODEL
    DIFFS = RE.DIFFICULTIES
    fmt, print_table = RE.fmt, RE.print_table

    # ------------------------------------------------------------ 공통 호출
    def zero_usage():
        # ms: API 호출 시간의 합, n: 호출 횟수, stages: 체인 단계별 사용량
        return {"in": 0, "out": 0, "cache_read": 0, "cache_write": 0, "ms": 0, "n": 0, "stages": {}}

    async def call(client, system, user, max_tokens, usage, stage=None):
        t = time.perf_counter()
        resp = await client.messages.create(
            model=MODEL, max_tokens=max_tokens,
            system=[{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
            messages=[{"role": "user", "content": user}])
        ms = int((time.perf_counter() - t) * 1000)
        u = resp.usage
        d = {"in": u.input_tokens or 0, "out": u.output_tokens or 0,
             "cache_read": getattr(u, "cache_read_input_tokens", 0) or 0,
             "cache_write": getattr(u, "cache_creation_input_tokens", 0) or 0, "ms": ms}
        for k, v in d.items():
            usage[k] += v
        usage["n"] += 1
        if stage:  # 체인의 단계별 기록
            s = usage["stages"].setdefault(stage, {"in": 0, "out": 0, "cache_read": 0, "cache_write": 0, "ms": 0})
            for k, v in d.items():
                s[k] += v
        return "".join(b.text for b in resp.content if b.type == "text")

    def parse_category(text):  # webapp/pipeline.py 와 같은 규칙
        m = re.search(r"최종 분류:[ \t]*([^\n]*)", text)
        return m.group(1).strip() if m else ""

    def result_section(text):
        return text.split("## 결과")[-1].strip()

    def field(extracted, name):
        m = re.search(rf"-\s*{name}:\s*(.+)", extracted)
        return m.group(1).strip() if m else None

    def mean(xs):
        xs = list(xs)
        return sum(xs) / len(xs) if xs else None

    def per_difficulty(rows, key):
        out = {}
        for d in DIFFS:
            sub = [r for r in rows if r["difficulty"] == d]
            out[f"acc_{d}"] = mean(r[key] for r in sub)
        return out

    async def run_classify_extract(client, inquiry, prompts, usage):
        """llm_extract_v2.py 체인: 분류(llm_cls_v2) -> 추출. 반환: (분류 응답, 분류값, 추출 응답, 추출 결과 섹션)."""
        cls_text = await call(client, prompts["classify"], inquiry, 2048, usage, "classify")
        category = parse_category(cls_text)
        cat_text = category if category else "(빈값 - 분류 불명확)"
        ext_user = f"**문의**: {inquiry}\n**1단계 분류 결과**: {cat_text}"
        ext_text = await call(client, prompts["extract"], ext_user, 2048, usage, "extract")
        return cls_text, category, cat_text, ext_text, result_section(ext_text)

    # ------------------------------------------------------------ 값 정규화 (extract)
    def canon_os(text):
        """텍스트에서 가장 먼저 언급된 OS를 정규 표기로 바꾼다. 없으면 None."""
        t = text.lower()
        found = []  # (위치, 계열, 정규 표기)

        def add(m, family, canon):
            found.append((m.start(), family, canon))

        for m in re.finditer(r"(?<![a-z])(?:windows|win)(?![a-z])\s*(\d{1,2})?|윈도우즈?\s*(\d{1,2})?", t):
            ver = m.group(1) or m.group(2)
            add(m, "windows", "Windows" + (f" {ver}" if ver else ""))
        for m in re.finditer(r"(?:(?<![a-z])mac\s?os(?![a-z])|맥\s?os|맥오에스)\s*(sonoma|ventura|monterey|sequoia|big\s?sur)?\s*(\d{1,2}(?:\.\d+)?)?", t):
            name = (m.group(1) or "").replace("  ", " ").title()
            ver = m.group(2)
            add(m, "mac", "macOS" + (f" {name}" if name else "") + (f" {ver}" if ver else ""))
        for m in re.finditer(r"(?:(?<![a-z])ubuntu(?![a-z])|우분투)\)?\s*(\d{2}\.\d{2})?", t):
            add(m, "ubuntu", "Ubuntu" + (f" {m.group(1)}" if m.group(1) else ""))
        for m in re.finditer(r"(?:(?<![a-z])android(?![a-z])|안드로이드)\s*(\d{1,2})?", t):
            add(m, "android", "Android" + (f" {m.group(1)}" if m.group(1) else ""))
        for m in re.finditer(r"(?<![a-z])ios(?![a-z])\s*(\d{1,2})?", t):
            add(m, "ios", "iOS" + (f" {m.group(1)}" if m.group(1) else ""))
        for m in re.finditer(r"(?<![a-z])linux(?![a-z])|리눅스", t):
            add(m, "linux", "Linux")
        if any(f[1] == "ubuntu" for f in found):  # 리눅스(우분투)처럼 같이 나오면 구체적인 쪽을 쓴다
            found = [f for f in found if f[1] != "linux"]
        if not found:
            return None
        found.sort()
        return found[0][2]

    def norm_asset_value(v):
        """v2 응답의 asset_id 값 -> 자산번호 또는 '없음' (값이 자산번호 형식이 아니면 원문 그대로)."""
        if v is None:
            return NONE
        v = v.replace("*", "").replace("`", "").strip()
        if v.startswith(NONE) or v in ("-", "N/A", "해당 없음", "해당없음"):
            return NONE
        m = ASSET_RE.search(v.upper())
        return m.group(0) if m else v

    def norm_os_value(v):
        if v is None:
            return NONE
        v = v.replace("*", "").replace("`", "").strip()
        if v.startswith(NONE) or v in ("-", "N/A", "해당 없음", "해당없음"):
            return NONE
        return canon_os(v) or v

    async def v1_read_fields(client, text, usage):
        """v1은 정해진 형식 없이 자유 서술로 답하므로, 응답이 밝힌 asset_id와 os를 읽어 오는 별도 호출을 쓴다(평가 보조 호출).

        응답에 적힌 것만 옮기게 하고, 예시로 든 값이나 사용자에게 요청하는 문장은 값으로 치지 않는다."""
        for _ in range(2):
            out = await call(client, READ_SYSTEM, text, 1024, usage)
            ma = re.search(r"^\s*-?\s*\**asset_id\**\s*:\s*(.+)", out, re.M | re.I)
            mo = re.search(r"^\s*-?\s*\**os\**\s*:\s*(.+)", out, re.M | re.I)
            a, o = (ma.group(1) if ma else None), (mo.group(1) if mo else None)
            if a is not None and o is not None:
                return norm_asset_value(a), norm_os_value(o)
        return NONE, NONE

    async def judge_symptom(client, item, text, usage):
        g = item["gold"]
        pts = "\n".join(f"- {p}" for p in g["symptom_points"]) or "(없음)"
        user = (f"**문의**: {item['inquiry']}\n**정답 증상**: {g['symptom']}\n**필수 요소**:\n{pts}\n"
                f"**모델 응답**:\n{text}")
        for _ in range(2):  # 판정 형식이 비어 오면 한 번 다시 묻는다 (thinking에 토큰을 다 쓴 경우 대비)
            out = await call(client, JUDGE_SYSTEM, user, 1024, usage)
            m = re.search(r"판정:\s*(정답|오답)", out)
            if m:
                break
        reason = (re.search(r"이유:\s*(.+)", out) or [None, ""])[1].strip() if m else "판정 형식 오류: " + out[:80]
        return (m.group(1) if m else "오답"), reason

    # ------------------------------------------------------------ extract 태스크
    def extract_load(path: Path):
        data = json.loads(path.read_text(encoding="utf-8"))
        items, seen = data["items"], set()
        for it in items:
            probs = []
            if not it.get("id") or it["id"] in seen:
                probs.append("id 누락/중복")
            seen.add(it.get("id"))
            g = it.get("gold") or {}
            for k in ("asset_id", "os", "symptom"):
                if not g.get(k):
                    probs.append(f"gold.{k} 없음")
            if g.get("symptom") == NONE and g.get("symptom_points"):
                probs.append("증상 없음인데 필수 요소가 있음")
            if g.get("symptom") not in (None, NONE) and not g.get("symptom_points"):
                probs.append("증상 있음인데 필수 요소가 없음")
            if g.get("os") not in (None, NONE) and canon_os(g["os"]) != g["os"]:
                probs.append(f"os 정답이 정규 표기가 아님: {g['os']!r}")
            if it.get("difficulty") not in DIFFS:
                probs.append("difficulty 이상")
            if not (it.get("inquiry") or "").strip():
                probs.append("inquiry 비어 있음")
            if probs:
                raise SystemExit(f"데이터 오류 ({it.get('id')}): {', '.join(probs)}")
            it["showcase"] = False
        return items

    async def extract_run_item(client, item, prompts):
        # usage는 평가 대상 모델의 호출만 센다. 판정과 읽기 보조 호출은 ev_usage에 따로 모아 버린다.
        usage, ev_usage, g, inquiry = zero_usage(), zero_usage(), item["gold"], item["inquiry"]
        if "classify" in prompts:  # v2 체인
            cls_text, category, _cat, ext_text, extracted = await run_classify_extract(client, inquiry, prompts, usage)
            asset = norm_asset_value(field(extracted, "asset_id"))
            os_ = norm_os_value(field(extracted, "os"))
            sym_text = field(extracted, "증상") or extracted
            raw = f"[분류] {category or '(빈값)'}\n{ext_text}"
        else:  # v1: 단일 호출, 자유 서술 응답 (필드는 평가 보조 호출로 읽는다)
            raw = await call(client, prompts["extract"], inquiry, 1024, usage, "extract")
            asset, os_ = await v1_read_fields(client, raw, ev_usage)
            sym_text = raw
        verdict, reason = await judge_symptom(client, item, sym_text, ev_usage)
        a_ok, o_ok, s_ok = int(asset == g["asset_id"]), int(os_ == g["os"]), int(verdict == "정답")
        mark = lambda b: "o" if b else "x"
        return {"raw": raw, "usage": usage, "latency_ms": usage["ms"], "pred": f"A{mark(a_ok)}O{mark(o_ok)}S{mark(s_ok)}",
                "correct": int(a_ok and o_ok and s_ok), "asset_ok": a_ok, "os_ok": o_ok, "symptom_ok": s_ok,
                "pred_asset": asset, "pred_os": os_, "judge_reason": reason}

    FIELDS = [("asset_id", "asset_ok", "pred_asset"), ("os", "os_ok", "pred_os"), ("symptom", "symptom_ok", None)]

    def extract_metrics(rows):
        m = {"accuracy": mean(r["correct"] for r in rows)}
        accs = []
        for f, k, pk in FIELDS:
            absent = [r for r in rows if r["gold"][f] == NONE]
            present = [r for r in rows if r["gold"][f] != NONE]
            m[f"acc_{f}"] = mean(r[k] for r in rows)
            m[f"acc_{f}_present"] = mean(r[k] for r in present)
            m[f"acc_{f}_absent"] = mean(r[k] for r in absent)
            m[f"_n_{f}_absent"], m[f"_n_{f}_present"] = len(absent), len(present)
            accs.append(m[f"acc_{f}"])
        m["acc_field_mean"] = mean(a for a in accs if a is not None)
        for f, k, pk in FIELDS:  # 환각률: 정답이 '없음'인데 값을 지어낸 비율
            absent = [r for r in rows if r["gold"][f] == NONE]
            m[f"halluc_{f}"] = mean(1 - r[k] for r in absent)
        m.update(per_difficulty(rows, "correct"))
        return m

    EXTRACT_COLS = (["accuracy", "acc_field_mean", "acc_asset_id", "acc_os", "acc_symptom"]
                    + [f"acc_{f}_{s}" for f in ("asset_id", "os", "symptom") for s in ("present", "absent")]
                    + ["halluc_asset_id", "halluc_os", "halluc_symptom", "acc_easy", "acc_medium", "acc_hard"])

    def extract_report(rows, m):
        print(f"\n전체 정확도(세 항목 모두 정답): {fmt(m['accuracy'])}   항목별 정확도 평균: {fmt(m['acc_field_mean'])}")
        print("\n[항목별 정확도]  (있는 경우=정답이 값, 없는 경우=정답이 '없음')")
        body = []
        for f, k, pk in FIELDS:
            body.append([f, fmt(m[f"acc_{f}"]), f"{fmt(m[f'acc_{f}_present'])} (n={m[f'_n_{f}_present']})",
                         f"{fmt(m[f'acc_{f}_absent'])} (n={m[f'_n_{f}_absent']})", fmt(m[f"halluc_{f}"])])
        print_table(["항목", "전체", "있는 경우", "없는 경우", "환각률"], body)
        print("\n[난이도별 정확도 (세 항목 모두)]")
        for d in DIFFS:
            print(f"  {d:9s} {fmt(m['acc_' + d])}  (n={sum(1 for r in rows if r['difficulty'] == d)})")
        print("\n[오류 유형] 항목별 개수 (증상의 '값 틀림'에는 누락 포함)")
        print_table(["항목", "정답(값)", "정답(없음)", "값 틀림", "누락", "환각"],
                    [[row[0]] + [str(x) for x in row[1:6]] for row in extract_confusion(rows)[1:]])
        wrong = [r for r in rows if not r["correct"]]
        print(f"\n[오답 목록] {len(wrong)}건")
        body = []
        for r in wrong:
            g, bad = r["gold"], []
            if not r["asset_ok"]:
                bad.append(f"asset {g['asset_id']}→{r['pred_asset']}")
            if not r["os_ok"]:
                bad.append(f"os {g['os']}→{r['pred_os']}")
            if not r["symptom_ok"]:
                bad.append("증상")
            body.append([r["id"], r["difficulty"], ", ".join(bad), r["inquiry"].replace("\n", " ")[:30]])
        if body:
            print_table(["id", "난이도", "틀린 항목", "문의(앞 30자)"], body)

    def extract_confusion(rows):
        out = [["항목", "정답(값 일치)", "정답(없음 일치)", "값 틀림", "누락", "환각", "합계"]]
        for f, k, pk in FIELDS:
            c = [0, 0, 0, 0, 0]
            for r in rows:
                absent = r["gold"][f] == NONE
                if r[k]:
                    c[1 if absent else 0] += 1
                elif absent:
                    c[4] += 1  # 환각
                elif pk is None:
                    c[2] += 1  # 증상은 판정 결과만 알 수 있어 틀림과 누락을 나누지 않음
                elif r[pk] == NONE:
                    c[3] += 1  # 누락
                else:
                    c[2] += 1  # 값 틀림
            out.append([f] + c + [sum(c)])
        return out

    EXTRACT_DETAIL = ["id", "difficulty", "correct", "asset_ok", "os_ok", "symptom_ok", "gold_asset_id", "pred_asset_id",
                      "gold_os", "pred_os", "gold_symptom", "judge_reason", "latency_ms", "tags", "inquiry", "raw_output"]

    def extract_detail_row(r):
        g = r["gold"]
        return [r["id"], r["difficulty"], "" if r["correct"] is None else r["correct"], r.get("asset_ok", ""),
                r.get("os_ok", ""), r.get("symptom_ok", ""), g["asset_id"], r.get("pred_asset", ""), g["os"],
                r.get("pred_os", ""), g["symptom"], r.get("judge_reason", ""),
                "" if r["latency_ms"] is None else r["latency_ms"], r["tags"], r["inquiry"], r["raw"][:RE.RAW_MAX]]

    def extract_dry(items):
        from collections import Counter
        print("정보 없는 문항 수:", {f: sum(1 for i in items if i["gold"][f] == NONE) for f in ("asset_id", "os", "symptom")})
        print("난이도 분포:", dict(Counter(i["difficulty"] for i in items)))

    # ------------------------------------------------------------ score 태스크
    def score_load(path: Path):
        data = json.loads(path.read_text(encoding="utf-8"))
        items, seen = data["items"], set()
        for it in items:
            probs = []
            if not it.get("id") or it["id"] in seen:
                probs.append("id 누락/중복")
            seen.add(it.get("id"))
            if it.get("gold") not in (0, 1, 2):
                probs.append(f"gold 이상: {it.get('gold')!r}")
            gm = it.get("gold_metrics")
            if not (isinstance(gm, list) and len(gm) == 7 and all(x in (0, 1, 2) for x in gm)):
                probs.append("gold_metrics 이상")
            if it.get("difficulty") not in DIFFS:
                probs.append("difficulty 이상")
            if not (it.get("inquiry") or "").strip():
                probs.append("inquiry 비어 있음")
            if probs:
                raise SystemExit(f"데이터 오류 ({it.get('id')}): {', '.join(probs)}")
            it["showcase"] = False
        return items

    def parse_v1_score(text):
        m = (re.search(r"판정[^0-9\n]{0,12}([0-9])\s*점", text) or re.search(r"([0-9])\s*점", text)
             or re.search(r"점수[^0-9\n]{0,8}([0-9])", text))
        v = int(m.group(1)) if m else None
        return v if v in (0, 1, 2) else None

    def parse_v2_score(text):
        ms = re.findall(r"종합 점수:\s*(\d)", text)
        v = int(ms[-1]) if ms else None
        return v if v in (0, 1, 2) else None

    def parse_v2_metrics(text):
        vals = re.findall(r"\|\s*[1-7]\.\s*[^|\n]+\|\s*([0-2])\s*\|", text)
        return [int(v) for v in vals] if len(vals) == 7 else None

    async def score_run_item(client, item, prompts):
        usage, inquiry = zero_usage(), item["inquiry"]
        if "classify" in prompts:  # v2 체인: 분류 -> 추출 -> 채점
            cls_text, category, cat_text, ext_text, extracted = await run_classify_extract(client, inquiry, prompts, usage)
            sc_user = f"**문의**: {inquiry}\n**1단계 분류 결과**: {cat_text}\n**2단계 추출 결과**:\n{extracted}"
            text = await call(client, prompts["scoring"], sc_user, 4096, usage, "scoring")
            pred, pm = parse_v2_score(result_section(text) if "## 결과" in text else text), parse_v2_metrics(text)
            raw = f"[분류] {category or '(빈값)'}\n[추출]\n{extracted}\n---\n{text}"
        else:  # v1: 단일 호출
            text = await call(client, prompts["scoring"], inquiry, 1024, usage, "scoring")
            pred, pm, raw = parse_v1_score(text), None, text
        return {"raw": raw, "usage": usage, "latency_ms": usage["ms"], "pred": pred if pred is not None else "INVALID",
                "correct": int(pred == item["gold"]), "pred_metrics": pm}

    def score_metrics(rows):
        valid = [r for r in rows if isinstance(r["pred"], int)]
        m = {"accuracy": mean(r["correct"] for r in rows),
             "mae": mean(abs(r["pred"] - r["gold"]) for r in valid),
             "bias": mean(r["pred"] - r["gold"] for r in valid),
             "n_invalid": len(rows) - len(valid)}
        for g in (0, 1, 2):
            m[f"acc_gold{g}"] = mean(r["correct"] for r in rows if r["gold"] == g)
        m.update(per_difficulty(rows, "correct"))
        pm = [r for r in valid if r.get("pred_metrics")]
        m["mae_metrics"] = mean(mean(abs(a - b) for a, b in zip(r["pred_metrics"], r["gold_metrics"])) for r in pm) if pm else None
        return m

    SCORE_COLS = ["accuracy", "mae", "bias", "acc_gold0", "acc_gold1", "acc_gold2",
                  "acc_easy", "acc_medium", "acc_hard", "n_invalid", "mae_metrics"]

    def score_report(rows, m):
        print(f"\n정확도 accuracy: {fmt(m['accuracy'])}   MAE: {fmt(m['mae'])}   편향(예측-정답 평균): {fmt(m['bias'])}"
              f"   INVALID: {m['n_invalid']}건")
        if m["mae_metrics"] is not None:
            print(f"(참고, v2만) 지표 7개 점수의 MAE: {fmt(m['mae_metrics'])}")
        print("\n[정답 점수별 정확도]")
        for g in (0, 1, 2):
            print(f"  정답 {g}점  {fmt(m[f'acc_gold{g}'])}  (n={sum(1 for r in rows if r['gold'] == g)})")
        print("\n[난이도별 정확도]")
        for d in DIFFS:
            print(f"  {d:9s} {fmt(m['acc_' + d])}  (n={sum(1 for r in rows if r['difficulty'] == d)})")
        cols = ["0", "1", "2", "INVALID"]
        print("\n[혼동 행렬] 행=정답, 열=예측")
        print_table(["정답\\예측"] + cols,
                    [[str(g)] + [str(sum(1 for r in rows if r["gold"] == g and str(r["pred"]) == c)) for c in cols]
                     for g in (0, 1, 2)])
        wrong = [r for r in rows if not r["correct"]]
        print(f"\n[오답 목록] {len(wrong)}건")
        if wrong:
            print_table(["id", "난이도", "정답", "예측", "문의(앞 36자)"],
                        [[r["id"], r["difficulty"], str(r["gold"]), str(r["pred"]), r["inquiry"].replace("\n", " ")[:36]]
                         for r in wrong])

    def score_confusion(rows):
        cols = ["0", "1", "2", "INVALID"]
        out = [["정답\\예측"] + cols + ["합계", "recall"]]
        for g in (0, 1, 2):
            cnt = [sum(1 for r in rows if r["gold"] == g and str(r["pred"]) == c) for c in cols]
            tot = sum(cnt)
            out.append([str(g)] + cnt + [tot, f"{cnt[g] / tot:.4f}" if tot else "0.0000"])
        return out

    SCORE_DETAIL = ["id", "difficulty", "gold", "pred", "correct", "signed_error", "gold_metrics", "pred_metrics",
                    "latency_ms", "tags", "inquiry", "raw_output"]

    def score_detail_row(r):
        pe = r["pred"] - r["gold"] if isinstance(r["pred"], int) else ""
        return [r["id"], r["difficulty"], r["gold"], r["pred"], "" if r["correct"] is None else r["correct"], pe,
                "".join(map(str, r.get("gold_metrics") or [])), "".join(map(str, r.get("pred_metrics") or [])),
                "" if r["latency_ms"] is None else r["latency_ms"], r["tags"], r["inquiry"], r["raw"][:RE.RAW_MAX]]

    def score_dry(items):
        from collections import Counter
        print("정답 점수 분포:", dict(sorted(Counter(i["gold"] for i in items).items())))
        print("난이도 분포:", dict(Counter(i["difficulty"] for i in items)))

    # score 평가는 res에 gold_metrics가 필요하므로 run_item 결과에 붙여 준다
    async def score_run_item_with_gold(client, item, prompts):
        out = await score_run_item(client, item, prompts)
        out["gold_metrics"] = item["gold_metrics"]
        return out

    P = "evals/prompts/"
    return {
        "extract": {
            "dataset": EVALS / "extract_eval.json",
            "csv": EVALS / "results" / "extract_runs.csv",
            "detail_dir": EVALS / "results" / "details" / "extract",
            "confusion_dir": EVALS / "results" / "confusion" / "extract",
            "variants": {
                "v1": {"prompts": {"extract": P + "extract_v1.txt"}},
                "v2": {"prompts": {"classify": P + "classify_v2.txt", "extract": P + "extract_v2.txt"}},
            },
            "default_variant": "v2",
            "load": extract_load, "run_item": extract_run_item, "metrics": extract_metrics,
            "metric_cols": EXTRACT_COLS, "report": extract_report, "confusion": extract_confusion,
            "detail_cols": EXTRACT_DETAIL, "detail_row": extract_detail_row,
            "compare_keys": ("accuracy", "acc_field_mean"), "dry_summary": extract_dry,
        },
        "score": {
            "dataset": EVALS / "score_eval.json",
            "csv": EVALS / "results" / "score_runs.csv",
            "detail_dir": EVALS / "results" / "details" / "score",
            "confusion_dir": EVALS / "results" / "confusion" / "score",
            "variants": {
                "v1": {"prompts": {"scoring": P + "scoring_v1.txt"}},
                "v2": {"prompts": {"classify": P + "classify_v2.txt", "extract": P + "extract_v2.txt",
                                   "scoring": P + "scoring_v2.txt"}},
            },
            "default_variant": "v2",
            "load": score_load, "run_item": score_run_item_with_gold, "metrics": score_metrics,
            "metric_cols": SCORE_COLS, "report": score_report, "confusion": score_confusion,
            "detail_cols": SCORE_DETAIL, "detail_row": score_detail_row,
            "compare_keys": ("accuracy", "mae"), "dry_summary": score_dry,
        },
    }
