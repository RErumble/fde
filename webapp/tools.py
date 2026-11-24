"""Function calling 도구 정의와 실행. 사용자 ID는 서버 세션에서만 가져온다.

모델이 넘기는 인자는 asset_id 하나뿐이다. user_id를 인자로 받지 않으므로
모델이 다른 사용자의 DB를 조회하도록 유도되어도 접근할 수 없다.
"""
import store

# 정의는 요청마다 같은 순서, 같은 내용으로 보내야 prefix 캐시가 유지된다.
TOOLS = [
    {
        "name": "get_asset_status",
        "description": (
            "자산 ID로 로그인한 사용자 본인의 자산 DB에서 장비의 현재 상태를 조회합니다. "
            "장비 종류, 모델, OS, 소속 부서, 상태(정상/점검필요/수리중), 보증 만료일, 마지막 점검일을 반환합니다. "
            "자산 ID가 본인 자산 DB에 없으면 found가 false입니다."
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
            "자산 ID로 로그인한 사용자 본인 장비에 등록된 알려진 장애 목록을 조회합니다. "
            "각 장애의 번호, 제목, 심각도, 처리 상태(진행중/해결됨), 최초 접수일, 임시 해결 방법을 반환합니다. "
            "자산 ID가 본인 자산 DB에 없으면 found가 false이고, 등록된 장애가 없으면 known_issues가 빈 목록입니다."
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


def run_tool(user_id: str, name: str, args: dict) -> dict:
    asset_id = str(args.get("asset_id", "")).strip().upper()
    asset = store.load_assets(user_id).get(asset_id)
    if asset is None:
        return {"found": False, "message": f"자산 DB에 {asset_id} 이(가) 없습니다."}
    if name == "get_asset_status":
        info = {k: v for k, v in asset.items() if k != "known_issues"}
        return {"found": True, **info}
    if name == "get_known_issues":
        return {"found": True, "asset_id": asset_id, "known_issues": asset["known_issues"]}
    return {"found": False, "message": f"알 수 없는 함수: {name}"}
