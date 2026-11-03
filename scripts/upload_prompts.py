"""코드의 도구 선택 지시문(app/router.py 의 ROUTER_TEMPLATE)을 Langfuse 프롬프트 router 의 버전 1 로 올린다.

실행:
  python scripts/upload_prompts.py --dry-run   # 보내지 않고 템플릿의 변수와 길이만 출력
  python scripts/upload_prompts.py             # router 가 이미 있으면 건너뛴다. 없으면 버전 1 을 만들고 production, staging 라벨을 붙인다

키는 .env 의 LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY, LANGFUSE_BASE_URL 에서 읽고 출력하지 않는다.
새 버전은 Langfuse 화면에서 만든다. 이 스크립트는 버전 1(= 코드의 지시문)을 만드는 데만 쓴다.
"""
import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.config import load_dotenv  # noqa: E402
from app.router import ROUTER_TEMPLATE  # noqa: E402

NAME = "router"


def main():
    if "--dry-run" in sys.argv:
        print(f"{NAME}: {len(ROUTER_TEMPLATE)}자, 변수 {sorted(set(re.findall(r'{{(\w+)}}', ROUTER_TEMPLATE)))}")
        return

    load_dotenv(ROOT / ".env")
    if not (os.environ.get("LANGFUSE_PUBLIC_KEY") and os.environ.get("LANGFUSE_SECRET_KEY")):
        sys.exit("LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY 가 .env 에 없습니다.")
    from langfuse import Langfuse

    client = Langfuse()
    try:  # ponytail: 가져오기가 실패하면(없음, 네트워크) 새로 만든다. 만들기도 실패하면 오류가 그대로 나온다.
        existing = client.get_prompt(NAME, label="production", type="text", cache_ttl_seconds=0)
        print(f"{NAME} 가 이미 있어 건너뜁니다(버전 {existing.version}, 라벨 {existing.labels}).")
        return
    except Exception:
        pass
    created = client.create_prompt(name=NAME, prompt=ROUTER_TEMPLATE, labels=["production", "staging"], type="text",
                                   commit_message="코드의 도구 선택 지시문(app/router.py)을 버전 1로 올림")
    print(f"{NAME} 버전 {created.version} 을 만들었습니다(라벨 {created.labels}).")


if __name__ == "__main__":
    main()
