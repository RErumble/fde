"""평가 스크립트: 태스크별 데이터셋을 돌려 지표를 계산하고 누적 CSV에 한 행씩 기록한다.

사용: venv\\Scripts\\python.exe evals\\run_eval.py --task classify --change "변경사항 설명"
태스크별 차이(데이터 로드, 요청 만들기, 예측 파싱, 채점)는 TASKS 레지스트리에 모여 있다.
"""
import argparse
import asyncio
import csv
import hashlib
import json
import math
import os
import re
import sys
import time
import unicodedata
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
EVALS = ROOT / "evals"
MODEL = "claude-haiku-5-5"  # CLAUDE.md 규칙: 개발 시 API는 Haiku 5.5만 사용 (변경 옵션 없음)
MAX_TOKENS = 2048
RAW_MAX = 3000

# ---------------------------------------------------------------- classify 태스크
LABELS = ["장애", "권한", "사용법", "기타", "빈값"]
INVALID = "INVALID"
DIFFICULTIES = ["easy", "medium", "hard"]


def norm_gold(g: str) -> str:
    g = (g or "").strip()
    return g if g else "빈값"


def classify_load(path: Path) -> list[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    items = data["items"]
    seen = set()
    for it in items:
        problems = []
        if not it.get("id") or it["id"] in seen:
            problems.append("id 누락/중복")
        seen.add(it.get("id"))
        if norm_gold(it.get("gold")) not in LABELS:
            problems.append(f"gold 이상: {it.get('gold')!r}")
        if it.get("difficulty") not in DIFFICULTIES:
            problems.append(f"difficulty 이상: {it.get('difficulty')!r}")
        if not (it.get("inquiry") or "").strip():
            problems.append("inquiry 비어 있음")
        if problems:
            raise SystemExit(f"데이터 오류 ({it.get('id')}): {', '.join(problems)}")
        it["gold"] = norm_gold(it["gold"])
        it["showcase"] = bool(it.get("showcase"))
    return items


def classify_parse(text: str) -> str:
    """응답 텍스트 -> 5개 라벨 또는 INVALID (webapp/pipeline.py parse_category와 같은 정규식)."""
    m = re.search(r"최종 분류:[ \t]*([^\n]*)", text)
    val = m.group(1).strip() if m else ""
    if not val:
        return "빈값"
    if val in LABELS:
        return val
    found = [lb for lb in LABELS if lb in val]
    return found[0] if len(found) == 1 else INVALID  # 라벨이 여럿 섞이면 모호하므로 INVALID


_MENTION_LABELS = ["장애", "권한", "사용법", "기타"]


def classify_parse_mention(text: str) -> str:
    """자유 서술 응답용 파서(--parser mention). llm_cls.py(v1)처럼 '최종 분류:' 줄이 없는 응답에 쓴다.

    따옴표로 묶인 라벨('장애')이 있으면 그중 첫 번째, 없으면 본문에서 가장 먼저 언급된 라벨을 예측으로 본다.
    v1에는 빈값(분류 보류)이라는 개념이 없으므로 이 파서는 빈값을 내지 않는다. 라벨 언급이 없으면 INVALID.
    """
    m = re.search(r"['\"‘’“”]\s*(장애|권한|사용법|기타)\s*['\"‘’“”]", text)
    if m:
        return m.group(1)
    hits = [(text.find(lb), lb) for lb in _MENTION_LABELS if lb in text]
    return min(hits)[1] if hits else INVALID


def classify_metrics(rows: list[dict]) -> dict:
    """rows: 오류 제외 결과. precision/recall은 분모가 0이면 0.0으로 처리한다(0/0 -> 0)."""
    def div(a, b):
        return a / b if b else 0.0

    m = {"accuracy": div(sum(r["correct"] for r in rows), len(rows)) if rows else None}
    f1s = []
    for lb in LABELS:
        tp = sum(1 for r in rows if r["gold"] == lb and r["pred"] == lb)
        n_pred = sum(1 for r in rows if r["pred"] == lb)
        n_gold = sum(1 for r in rows if r["gold"] == lb)
        p, rc = div(tp, n_pred), div(tp, n_gold)
        f1 = div(2 * p * rc, p + rc)
        m[f"precision_{lb}"], m[f"recall_{lb}"], m[f"f1_{lb}"] = p, rc, f1
        m[f"_support_{lb}"] = n_gold
        f1s.append(f1)
    m["macro_f1"] = sum(f1s) / len(f1s) if rows else None  # 지원 수 0인 라벨도 F1=0으로 평균에 포함
    for d in DIFFICULTIES:
        sub = [r for r in rows if r["difficulty"] == d]
        m[f"acc_{d}"] = div(sum(r["correct"] for r in sub), len(sub)) if sub else None
    sc = [r for r in rows if r["showcase"]]
    m["acc_showcase"] = div(sum(r["correct"] for r in sc), len(sc)) if sc else None
    return m


CLASSIFY_METRIC_COLS = (
    ["accuracy", "macro_f1", "acc_easy", "acc_medium", "acc_hard", "acc_showcase"]
    + [f"f1_{lb}" for lb in LABELS]  # precision/recall은 콘솔 출력과 F1 계산용 내부 값
)


def dispw(s: str) -> int:
    return sum(2 if unicodedata.east_asian_width(c) in "WF" else 1 for c in s)


def pad(s: str, w: int, right: bool = False) -> str:
    sp = " " * (w - dispw(s))
    return sp + s if right else s + sp


def print_table(header: list[str], body: list[list[str]]):
    ws = [max(dispw(r[i]) for r in [header] + body) for i in range(len(header))]
    print("  ".join(pad(h, w, i > 0) for i, (h, w) in enumerate(zip(header, ws))))
    for r in body:
        print("  ".join(pad(c, w, i > 0) for i, (c, w) in enumerate(zip(r, ws))))


def fmt(v) -> str:
    return "-" if v is None else f"{v:.4f}"


def classify_report(rows: list[dict], m: dict):
    print(f"\n정확도 accuracy: {fmt(m['accuracy'])}   macro-F1: {fmt(m['macro_f1'])}")
    print("\n[라벨별 지표]  (precision/recall 분모가 0이면 0.0)")
    print_table(["라벨", "precision", "recall", "F1", "지원수"],
                [[lb, fmt(m[f"precision_{lb}"]), fmt(m[f"recall_{lb}"]), fmt(m[f"f1_{lb}"]),
                  str(m[f"_support_{lb}"])] for lb in LABELS])
    print("\n[난이도별 정확도]")
    for k in DIFFICULTIES + ["showcase"]:
        sub = [r for r in rows if (r["showcase"] if k == "showcase" else r["difficulty"] == k)]
        print(f"  {k:9s} {fmt(m['acc_' + k])}  (n={len(sub)})")
    cols = LABELS + [INVALID]
    print("\n[혼동 행렬] 행=정답, 열=예측")
    body = []
    for g in LABELS:
        body.append([g] + [str(sum(1 for r in rows if r["gold"] == g and r["pred"] == c)) for c in cols])
    print_table(["정답\\예측"] + cols, body)
    wrong = [r for r in rows if not r["correct"]]
    print(f"\n[오답 목록] {len(wrong)}건")
    if wrong:
        print_table(["id", "난이도", "정답", "예측", "문의(앞 40자)"],
                    [[r["id"], r["difficulty"], r["gold"], r["pred"],
                      r["inquiry"].replace("\n", " ")[:40]] for r in wrong])


def classify_confusion(rows: list[dict]) -> list[list]:
    """혼동 행렬 파일용 행: 헤더 + 정답 라벨별 행(예측 개수, 합계, recall)."""
    cols = LABELS + [INVALID]
    out = [["정답\\예측"] + cols + ["합계", "recall"]]
    for g in LABELS:
        cnt = [sum(1 for r in rows if r["gold"] == g and r["pred"] == c) for c in cols]
        tot = sum(cnt)
        out.append([g] + cnt + [tot, f"{cnt[LABELS.index(g)] / tot:.4f}" if tot else "0.0000"])
    return out


def classify_request(item: dict, prompt_text: str) -> dict:
    return {
        "system": [{"type": "text", "text": prompt_text, "cache_control": {"type": "ephemeral"}}],
        "messages": [{"role": "user", "content": item["inquiry"]}],
    }


TASKS = {
    "classify": {
        "dataset": EVALS / "classify_eval.json",
        "prompt": ROOT / "webapp" / "prompts" / "classify.md",
        "csv": EVALS / "results" / "classify_runs.csv",
        "detail_dir": EVALS / "results" / "details",
        "load": classify_load,
        "request": classify_request,
        "parse": classify_parse,
        "metrics": classify_metrics,
        "metric_cols": CLASSIFY_METRIC_COLS,
        "report": classify_report,
        "confusion": classify_confusion,
        "confusion_dir": EVALS / "results" / "confusion",
    },
}

# ---------------------------------------------------------------- 공통
BASE_COLS_HEAD = ["run_id", "date", "task", "variant", "change", "dataset", "n_items", "n_scored", "n_errors",
                  "model", "prompt_file", "prompt_sha"]
# 속도: 요청(문의 1건)당 응답 시간 = 그 요청이 일으킨 평가 대상 API 호출의 시간 합(체인은 순차 호출이라 합이 곧 대기 시간)
# 토큰, 비용: 평가 대상 API 호출 전체의 합. 판정과 필드 읽기 같은 평가 보조 호출은 제외한다.
BASE_COLS_TAIL = ["avg_latency_ms", "p50_latency_ms", "p95_latency_ms", "max_latency_ms", "calls_per_item",
                  "total_input_tokens", "total_output_tokens", "cache_read_tokens", "cache_write_tokens",
                  "prompt_tokens_per_item", "output_tokens_per_item",
                  "total_cost_usd", "cost_per_item_usd", "cost_per_1k_usd", "cost_warm_per_1k_usd", "cost_nocache_per_1k_usd",
                  "stage_json", "concurrency", "price_ref", "duration_s", "subset", "confusion_file"]
DETAIL_COLS = ["id", "difficulty", "gold", "pred", "correct", "latency_ms", "tags", "inquiry", "raw_output"]
# 모든 태스크의 문항별 상세 파일 끝에 붙는 속도, 토큰, 비용 열
EXTRA_DETAIL_COLS = ["n_calls", "input_tokens", "output_tokens", "cache_read_tokens", "cache_write_tokens", "cost_usd"]

# 공식 요금표(platform.claude.com/docs/en/about-claude/pricing, 2026-11-21 조회): Claude Haiku 5.5, 프롬프트 100K 토큰 이하, USD per MTok
PRICES = {"claude-haiku-5-5": {"in": 0.10, "cache_write": 0.125, "cache_read": 0.01, "out": 0.50}}
PRICE_REF = "claude-haiku-5-5 per MTok: in 0.10, cache_write(5m) 0.125, cache_read 0.01, out 0.50 (<=100K prompt)"


def usage_cost(u: dict, mode: str = "actual") -> float:
    """API가 돌려준 토큰 사용량으로 비용(USD)을 계산한다.

    actual: 실제 사용량 그대로(캐시 쓰기와 읽기 구분). 첫 호출의 캐시 쓰기 같은 콜드 스타트 비용이 포함된다.
    nocache: 캐시를 안 썼다고 가정(캐시 대상 토큰도 모두 입력 단가).
    warm: 캐시가 이미 데워진 정상 상태 가정(캐시 쓰기 토큰도 읽기 단가). 운영 중 반복 트래픽의 비용에 가깝다.
    """
    p = PRICES[MODEL]
    if mode == "nocache":
        return ((u["in"] + u["cache_write"] + u["cache_read"]) * p["in"] + u["out"] * p["out"]) / 1e6
    if mode == "warm":
        return (u["in"] * p["in"] + (u["cache_write"] + u["cache_read"]) * p["cache_read"] + u["out"] * p["out"]) / 1e6
    return (u["in"] * p["in"] + u["cache_write"] * p["cache_write"]
            + u["cache_read"] * p["cache_read"] + u["out"] * p["out"]) / 1e6


def pct(xs, q: float):
    """nearest-rank 백분위수."""
    xs = sorted(xs)
    if not xs:
        return None
    return xs[max(0, math.ceil(q * len(xs)) - 1)]


def rel(p: Path) -> str:
    try:
        return Path(os.path.relpath(p, ROOT)).as_posix()
    except ValueError:
        return str(p)


def load_api_key() -> str:
    from dotenv import load_dotenv
    load_dotenv(Path(os.environ.get("ENV_PATH", r"C:\Users\hi\Documents\fde\.env")))
    key = os.environ.get("CLAUDE_API_KEY")
    if not key:
        raise SystemExit("CLAUDE_API_KEY를 .env에서 찾지 못했습니다.")
    return key


async def run_items(task: dict, items: list[dict], prompt_text, concurrency: int) -> list[dict]:
    """prompt_text: classify는 프롬프트 문자열, variant 태스크(extract, score)는 {이름: 프롬프트} 딕셔너리."""
    import anthropic
    client = anthropic.AsyncAnthropic(api_key=load_api_key())
    sem = asyncio.Semaphore(concurrency)
    done = 0

    async def one(item: dict) -> dict:
        nonlocal done
        res = {"id": item["id"], "difficulty": item["difficulty"], "gold": item["gold"],
               "showcase": item["showcase"], "tags": ",".join(item.get("tags") or []),
               "inquiry": item["inquiry"], "pred": "ERROR", "correct": None, "raw": "",
               "latency_ms": None, "usage": {"in": 0, "out": 0, "cache_read": 0, "cache_write": 0},
               "error": ""}
        async with sem:
            t = time.perf_counter()
            try:
                if task.get("run_item"):  # 여러 번 호출하는 체인이나 판정 호출이 필요한 태스크
                    out = await task["run_item"](client, item, prompt_text)
                    res.update(out)
                    # 평가 보조 호출(판정, 필드 읽기)을 뺀, 평가 대상 호출 시간의 합
                    res["latency_ms"] = out.get("latency_ms", int((time.perf_counter() - t) * 1000))
                else:
                    resp = await client.messages.create(model=MODEL, max_tokens=MAX_TOKENS,
                                                        **task["request"](item, prompt_text))
                    res["latency_ms"] = int((time.perf_counter() - t) * 1000)
                    text = "".join(b.text for b in resp.content if b.type == "text")
                    u = resp.usage
                    res["usage"] = {"in": u.input_tokens or 0, "out": u.output_tokens or 0,
                                    "cache_read": getattr(u, "cache_read_input_tokens", 0) or 0,
                                    "cache_write": getattr(u, "cache_creation_input_tokens", 0) or 0,
                                    "ms": res["latency_ms"], "n": 1, "stages": {}}
                    res["raw"] = text
                    res["pred"] = task["parse"](text)
                    res["correct"] = int(res["pred"] == res["gold"])
            except anthropic.APIError as e:
                res["error"] = f"{type(e).__name__}: {e}"
                res["raw"] = res["error"]
        if res["pred"] != "ERROR":
            res["cost_usd"] = usage_cost(res["usage"])
            res["cost_nocache_usd"] = usage_cost(res["usage"], "nocache")
            res["cost_warm_usd"] = usage_cost(res["usage"], "warm")
        done += 1
        print(f"  [{done}/{len(items)}] {item['id']} -> {res['pred']}", flush=True)
        return res

    return await asyncio.gather(*(one(i) for i in items))


def append_csv_row(path: Path, columns: list[str], row: dict):
    """한 행 append. 헤더가 다르면 기존 행을 보존한 채 임시 파일에 새로 써서 교체한다."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.stat().st_size > 0:
        with path.open(newline="", encoding="utf-8-sig") as f:
            rd = csv.DictReader(f)
            old_cols = list(rd.fieldnames or [])
            old_rows = list(rd) if old_cols != columns else []
        # 파일 핸들을 닫은 뒤에 교체해야 Windows에서 os.replace가 실패하지 않는다
        if old_cols != columns:
            new_cols = columns + [c for c in old_cols if c not in columns]
            tmp = path.with_name(path.name + ".tmp")
            with tmp.open("w", newline="", encoding="utf-8-sig") as g:
                w = csv.DictWriter(g, fieldnames=new_cols, restval="")
                w.writeheader()
                w.writerows(old_rows)
                w.writerow(row)
            os.replace(tmp, path)
            print(f"(CSV 헤더가 달라 새 열 집합으로 다시 썼습니다: {rel(path)})")
            return
        with path.open("a", newline="", encoding="utf-8") as f:  # BOM 중복 방지
            csv.DictWriter(f, fieldnames=columns, restval="").writerow(row)
    else:
        with path.open("w", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=columns, restval="")
            w.writeheader()
            w.writerow(row)


def last_comparable(path: Path, task: str, dataset: str, n_items: int) -> dict | None:
    if not path.exists():
        return None
    with path.open(newline="", encoding="utf-8-sig") as f:
        rows = [r for r in csv.DictReader(f)
                if r.get("task") == task and r.get("dataset") == dataset
                and r.get("n_items") == str(n_items) and not (r.get("subset") or "").strip()]
    return rows[-1] if rows else None


def read_details(path: Path) -> dict:
    if not path.exists():
        return {}
    with path.open(newline="", encoding="utf-8-sig") as f:
        return {r["id"]: r for r in csv.DictReader(f)}


def write_details(path: Path, results: list[dict], task: dict | None = None):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        extra = lambda r: [r["usage"].get("n", ""), r["usage"]["in"], r["usage"]["out"], r["usage"]["cache_read"],
                           r["usage"]["cache_write"], f"{r['cost_usd']:.8f}" if "cost_usd" in r else ""]
        if task and task.get("detail_cols"):  # 태스크별 상세 열 (extract, score)
            w.writerow(task["detail_cols"] + EXTRA_DETAIL_COLS)
            for r in results:
                w.writerow(task["detail_row"](r) + extra(r))
            return
        w.writerow(DETAIL_COLS + EXTRA_DETAIL_COLS)
        for r in results:
            c = "" if r["correct"] is None else r["correct"]
            w.writerow([r["id"], r["difficulty"], r["gold"], r["pred"], c,
                        "" if r["latency_ms"] is None else r["latency_ms"], r["tags"],
                        r["inquiry"], r["raw"][:RAW_MAX]] + extra(r))


def r4(v):
    if v is None:
        return ""
    return str(v) if isinstance(v, int) else f"{v:.4f}"


def select_items(items, args) -> tuple[list[dict], str]:
    parts = []
    if args.ids:
        want = [s.strip() for s in args.ids.split(",") if s.strip()]
        missing = [i for i in want if i not in {it["id"] for it in items}]
        if missing:
            raise SystemExit(f"--ids에 없는 id: {', '.join(missing)}")
        items = [it for it in items if it["id"] in want]
        parts.append("ids=" + ",".join(want))
    if args.difficulty:
        want = [s.strip() for s in args.difficulty.split(",") if s.strip()]
        bad = [d for d in want if d not in DIFFICULTIES]
        if bad:
            raise SystemExit(f"--difficulty 값 오류: {', '.join(bad)} (easy/medium/hard)")
        items = [it for it in items if it["difficulty"] in want]
        parts.append("difficulty=" + ",".join(want))
    if args.limit is not None:
        items = items[:args.limit]
        parts.append(f"limit={args.limit}")
    return items, ";".join(parts)


import tasks_extra  # noqa: E402  (extract, score 태스크를 TASKS에 등록)

TASKS.update(tasks_extra.build_tasks(sys.modules[__name__]))


def main():
    ap = argparse.ArgumentParser(description="태스크 평가 실행 (결과는 누적 CSV에 한 행씩 기록)")
    ap.add_argument("--task", default="classify", choices=list(TASKS))
    ap.add_argument("--change", help="이번 평가의 변경사항 설명 (필수, 첫 실행이면 '초기 베이스라인')")
    ap.add_argument("--dataset")
    ap.add_argument("--prompt")
    ap.add_argument("--parser", choices=["final-line", "mention"], default="final-line",
                    help="예측 파서: final-line(기본, '최종 분류:' 줄) 또는 mention(자유 서술 응답에서 라벨 언급 추출, v1용)")
    ap.add_argument("--variant", help="extract, score 태스크의 평가 대상(v1: 기존 단일 프롬프트, v2: 고도화 체인). 기본 v2")
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--ids")
    ap.add_argument("--difficulty")
    ap.add_argument("--csv", help="누적 결과 CSV 경로")
    ap.add_argument("--detail-dir", help="문항별 상세 저장 디렉터리")
    ap.add_argument("--confusion-dir", help="혼동 행렬 파일 저장 디렉터리")
    ap.add_argument("--dry-run", action="store_true", help="검증과 실행 계획만 출력 (API 호출 없음)")
    args = ap.parse_args()
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8")
        except Exception:
            pass

    task = TASKS[args.task]
    if args.parser == "mention":
        if args.task != "classify":
            raise SystemExit("--parser mention은 classify 태스크에서만 쓸 수 있습니다.")
        task = {**task, "parse": classify_parse_mention}
    change = (args.change or "").strip()
    if not change and not args.dry_run:
        if sys.stdin.isatty():
            print("변경사항을 입력하세요 (첫 실행이면 '초기 베이스라인').")
            try:
                change = input("변경사항> ").strip()
            except EOFError:
                change = ""
        else:
            print("오류: --change(변경사항)가 필요합니다. 예: --change \"초기 베이스라인\"", file=sys.stderr)
            sys.exit(2)
        if not change:
            print("오류: 변경사항은 비워 둘 수 없습니다.", file=sys.stderr)
            sys.exit(2)
    if args.concurrency < 1:
        raise SystemExit("--concurrency는 1 이상이어야 합니다.")

    dataset_path = Path(args.dataset).resolve() if args.dataset else task["dataset"]
    csv_path = Path(args.csv).resolve() if args.csv else task["csv"]
    detail_dir = Path(args.detail_dir).resolve() if args.detail_dir else task["detail_dir"]
    confusion_dir = Path(args.confusion_dir).resolve() if args.confusion_dir else task["confusion_dir"]

    variants = task.get("variants")
    if variants:  # extract, score: 변형(v1, v2)마다 쓰는 프롬프트 파일 묶음이 정해져 있다
        vname = args.variant or task["default_variant"]
        if vname not in variants:
            raise SystemExit(f"--variant는 {', '.join(variants)} 중 하나여야 합니다.")
        if args.prompt:
            print("경고: --prompt는 variant 태스크에서 무시됩니다.", file=sys.stderr)
        vpaths = {k: ROOT / p for k, p in variants[vname]["prompts"].items()}
        prompt_path = next(iter(vpaths.values()))
        check_paths = [dataset_path, *vpaths.values()]
    else:
        vname, vpaths = (args.variant or ""), None  # classify는 프롬프트를 --prompt로 고르고 --variant는 기록용 이름
        prompt_path = Path(args.prompt).resolve() if args.prompt else task["prompt"]
        check_paths = [dataset_path, prompt_path]
    for p in check_paths:
        if not p.exists():
            raise SystemExit(f"파일이 없습니다: {p}")

    all_items = task["load"](dataset_path)
    items, subset = select_items(all_items, args)
    if not items:
        raise SystemExit("선택된 문항이 없습니다.")
    if vpaths:
        prompt_text = {k: p.read_text(encoding="utf-8") for k, p in vpaths.items()}
        prompt_sha = hashlib.sha256(b"".join(p.read_bytes() for p in vpaths.values())).hexdigest()[:8]
        prompt_label = "+".join(rel(p) for p in vpaths.values())
    else:
        prompt_text = prompt_path.read_text(encoding="utf-8")
        prompt_sha = hashlib.sha256(prompt_path.read_bytes()).hexdigest()[:8]
        prompt_label = rel(prompt_path)

    print(f"task={args.task}  dataset={rel(dataset_path)} (전체 {len(all_items)}문항)")
    print(f"prompt={prompt_label} sha={prompt_sha}  "
          + (f"variant={vname}  " if vname else f"parser={args.parser}  ")
          + f"model={MODEL}  concurrency={args.concurrency}")
    print(f"실행 문항 {len(items)}개  subset={subset or '(전체)'}")
    print(f"CSV={csv_path}\n상세={detail_dir}")
    if args.dry_run:
        from collections import Counter
        if task.get("dry_summary"):
            task["dry_summary"](items)
        else:
            print("정답 분포:", dict(Counter(i["gold"] for i in items)))
            print("난이도 분포:", dict(Counter(i["difficulty"] for i in items)),
                  " showcase:", sum(i["showcase"] for i in items))
        print(f"[dry-run] 데이터 검증 통과 ({len(all_items)}문항). API 호출/파일 쓰기 없음.")
        return

    now = datetime.now()
    run_id = now.strftime("%Y%m%d-%H%M%S")
    t0 = time.perf_counter()
    results = asyncio.run(run_items(task, items, prompt_text, args.concurrency))
    duration = time.perf_counter() - t0

    scored = [r for r in results if r["pred"] != "ERROR"]
    errors = [r for r in results if r["pred"] == "ERROR"]
    m = task["metrics"](scored)
    task["report"](scored, m)

    if errors:
        print(f"\n경고: API 오류 {len(errors)}건 (지표에서 제외): "
              + ", ".join(r["id"] for r in errors), file=sys.stderr)
        print("  첫 오류:", errors[0]["error"][:200], file=sys.stderr)

    lat = [r["latency_ms"] for r in scored if r["latency_ms"] is not None]
    tot = lambda k: sum(r["usage"][k] for r in results)
    n_ok = len(scored) or 1
    total_cost = sum(r.get("cost_usd", 0.0) for r in scored)
    total_nocache = sum(r.get("cost_nocache_usd", 0.0) for r in scored)
    total_warm = sum(r.get("cost_warm_usd", 0.0) for r in scored)
    prompt_tok = sum(r["usage"]["in"] + r["usage"]["cache_read"] + r["usage"]["cache_write"] for r in scored)
    # 단계별(체인) 평균: 지연, 프롬프트 토큰, 출력 토큰, 비용
    stage_names = []
    for r in scored:
        for s in r["usage"].get("stages", {}):
            if s not in stage_names:
                stage_names.append(s)
    stages = {}
    for s in stage_names:
        sus = [r["usage"]["stages"][s] for r in scored if s in r["usage"].get("stages", {})]
        stages[s] = {"avg_ms": round(sum(u["ms"] for u in sus) / len(sus)),
                     "avg_prompt_tok": round(sum(u["in"] + u["cache_read"] + u["cache_write"] for u in sus) / len(sus)),
                     "avg_out_tok": round(sum(u["out"] for u in sus) / len(sus), 1),
                     "avg_cost_usd": round(sum(usage_cost(u) for u in sus) / len(sus), 8)}
    row = {"run_id": run_id, "date": now.strftime("%Y-%m-%d %H:%M:%S"), "task": args.task, "variant": vname,
           "change": change, "dataset": rel(dataset_path), "n_items": len(items),
           "n_scored": len(scored), "n_errors": len(errors), "model": MODEL,
           "prompt_file": prompt_label, "prompt_sha": prompt_sha,
           "avg_latency_ms": f"{sum(lat) / len(lat):.0f}" if lat else "",
           "p50_latency_ms": pct(lat, 0.50) if lat else "", "p95_latency_ms": pct(lat, 0.95) if lat else "",
           "max_latency_ms": max(lat) if lat else "",
           "calls_per_item": f"{sum(r['usage'].get('n', 1) for r in scored) / n_ok:.2f}",
           "total_input_tokens": tot("in"), "total_output_tokens": tot("out"),
           "cache_read_tokens": tot("cache_read"), "cache_write_tokens": tot("cache_write"),
           "prompt_tokens_per_item": f"{prompt_tok / n_ok:.1f}",
           "output_tokens_per_item": f"{sum(r['usage']['out'] for r in scored) / n_ok:.1f}",
           "total_cost_usd": f"{total_cost:.6f}", "cost_per_item_usd": f"{total_cost / n_ok:.8f}",
           "cost_per_1k_usd": f"{total_cost / n_ok * 1000:.4f}",
           "cost_warm_per_1k_usd": f"{total_warm / n_ok * 1000:.4f}",
           "cost_nocache_per_1k_usd": f"{total_nocache / n_ok * 1000:.4f}",
           "stage_json": json.dumps(stages, ensure_ascii=False) if stages else "",
           "concurrency": args.concurrency, "price_ref": PRICE_REF,
           "duration_s": f"{duration:.1f}", "subset": subset}

    print("\n[속도와 비용]  (평가 대상 모델 호출만, 판정 같은 평가 보조 호출 제외)")
    if lat:
        print(f"  요청당 응답 시간(ms): 평균 {sum(lat) / len(lat):.0f}  P50 {pct(lat, 0.5)}  P95 {pct(lat, 0.95)}  최대 {max(lat)}"
              f"   (요청당 API 호출 {row['calls_per_item']}회)")
    print(f"  토큰: 입력 {tot('in')}  출력 {tot('out')}  캐시 읽기 {tot('cache_read')}  캐시 쓰기 {tot('cache_write')}")
    print(f"  비용(실측, 콜드 스타트 포함): 총 ${total_cost:.5f}  요청당 ${total_cost / n_ok:.6f}  1천 건당 ${total_cost / n_ok * 1000:.4f}")
    print(f"  1천 건당 비용 가정별: 캐시 데워진 상태 ${total_warm / n_ok * 1000:.4f}  /  캐시 없음 ${total_nocache / n_ok * 1000:.4f}")
    for s, v in stages.items():
        print(f"    단계 {s}: 평균 {v['avg_ms']}ms, 프롬프트 {v['avg_prompt_tok']}토큰, 출력 {v['avg_out_tok']}토큰, 비용 ${v['avg_cost_usd']:.6f}")
    for c in task["metric_cols"]:
        row[c] = r4(m[c])
    columns = BASE_COLS_HEAD + list(task["metric_cols"]) + BASE_COLS_TAIL

    # 직전(전체 실행, 같은 task/dataset/문항 수) 비교 -- 현재 행을 쓰기 전에 조회
    prev = None if subset else last_comparable(csv_path, args.task, rel(dataset_path), len(items))
    prev_details = read_details(detail_dir / f"{prev['run_id']}.csv") if prev else {}

    conf_path = confusion_dir / f"{run_id}.csv"
    conf_path.parent.mkdir(parents=True, exist_ok=True)
    with conf_path.open("w", newline="", encoding="utf-8-sig") as f:
        csv.writer(f).writerows(task["confusion"](scored))
    row["confusion_file"] = rel(conf_path)
    append_csv_row(csv_path, columns, row)
    write_details(detail_dir / f"{run_id}.csv", results, task)

    print("\n[직전 실행과 비교]")
    if subset:
        print("  일부 문항 실행(subset)이라 비교하지 않습니다.")
    elif not prev:
        print("  비교할 직전 전체 실행이 없습니다.")
    else:
        print(f"  직전: {prev['run_id']} ({prev['change']})")
        for k in task.get("compare_keys", ("accuracy", "macro_f1")):
            try:
                pv = float(prev[k])
                cur = m[k]
                print(f"  {k}: {pv:.4f} -> {fmt(cur)}  ({cur - pv:+.4f})")
            except (ValueError, TypeError, KeyError):
                print(f"  {k}: 비교 불가")
        if prev_details:
            cur = {r["id"]: r["correct"] for r in results}
            reg = [i for i, c in cur.items() if c == 0 and prev_details.get(i, {}).get("correct") == "1"]
            fix = [i for i, c in cur.items() if c == 1 and prev_details.get(i, {}).get("correct") == "0"]
            print(f"  새로 틀린 문항(회귀) {len(reg)}건: {', '.join(reg) or '-'}")
            print(f"  새로 맞춘 문항 {len(fix)}건: {', '.join(fix) or '-'}")
        else:
            print("  직전 실행의 문항별 상세 파일이 없어 문항 단위 비교는 생략합니다.")

    print(f"\n토큰 in={tot('in')} out={tot('out')} cache_read={tot('cache_read')} "
          f"cache_write={tot('cache_write')}  평균지연 {row['avg_latency_ms']}ms  소요 {duration:.1f}s")
    print(f"기록: {csv_path} (run_id={run_id}, 오류 {len(errors)}건)")


if __name__ == "__main__":
    main()
