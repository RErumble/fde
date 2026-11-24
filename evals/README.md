# 평가 스크립트 (evals/run_eval.py)

IT 문의 분류(`classify`) 평가. 모델은 `claude-haiku-5-5` 고정(변경 옵션 없음). 지표는 accuracy와 F1만 기록합니다.

## 실행 (프로젝트 루트에서)

```powershell
$env:PYTHONIOENCODING='utf-8'
# 전체 실행 (--change 필수, 첫 실행이면 "초기 베이스라인")
venv\Scripts\python.exe evals\run_eval.py --task classify --change "초기 베이스라인"
# 일부만 (기록은 되지만 subset 열로 구분, 직전 비교 대상 아님)
venv\Scripts\python.exe evals\run_eval.py --change "스모크" --limit 6
venv\Scripts\python.exe evals\run_eval.py --change "스모크" --ids C001,C002 --difficulty hard
# 프롬프트를 바꿔서
venv\Scripts\python.exe evals\run_eval.py --change "프롬프트 v2" --prompt path\to\classify_v2.md
# 검증만 (API 호출 없음)
venv\Scripts\python.exe evals\run_eval.py --dry-run
```

옵션: `--dataset`, `--prompt`, `--concurrency`(기본 8), `--limit`, `--ids`, `--difficulty`, `--csv`, `--detail-dir`, `--confusion-dir`, `--dry-run`.
`--change`가 없으면 대화형에서는 묻고, 비대화형이면 종료 코드 2로 끝납니다.

## 결과 파일 (evals/results/)

- `classify_runs.csv`: 실행마다 한 행 누적(utf-8-sig). 열: run_id, date, task, change, dataset, n_items, n_scored, n_errors, model, prompt_file, prompt_sha, accuracy, macro_f1, acc_easy, acc_medium, acc_hard, acc_showcase, f1_장애, f1_권한, f1_사용법, f1_기타, f1_빈값, avg_latency_ms, total_input_tokens, total_output_tokens, cache_read_tokens, cache_write_tokens, duration_s, subset, confusion_file
- `confusion/<run_id>.csv`: 혼동 행렬(행=정답, 열=예측 장애/권한/사용법/기타/빈값/INVALID, 끝에 합계와 행별 recall)
- `details/<run_id>.csv`: 문항별 상세(id, difficulty, gold, pred, correct, latency_ms, tags, inquiry, raw_output)

## 참고

- API 오류 문항은 ERROR로 기록하고 지표에서 제외하며 경고를 출력합니다.
- precision/recall 분모가 0이면 0으로 처리하고, macro-F1은 5개 라벨 F1의 단순 평균입니다(지원 수 0인 라벨도 0으로 포함). 일부 문항만 돌리면 macro-F1이 낮게 보일 수 있습니다.
- "직전 실행과 비교"는 같은 task/dataset/문항 수의 전체 실행(subset 빈 행)끼리만 합니다.
