"""'근거 없는 정정' 평가. data/eval/followup.jsonl 의 항목마다 답을 만들어, 앞 답을 정정하는 표현이 있는지 센다.

실행:
  python scripts/eval_followup.py                    # .env 의 LLM 으로 항목마다 3번 (20개 x 3 = 호출 60번, 비용 발생)
  python scripts/eval_followup.py --repeat 1 --out result.json   # 1번만, 답 전체를 파일로 저장

답을 만드는 입력은 agent.system_prompt + 대화 기록 + 질문으로, 서버가 쓰는 것과 같다. 서버와 Langfuse 는 쓰지 않는다.
정정 여부와 통과 여부는 표현으로 판정하는 근사치라서, 실패한 항목은 답 앞부분을 함께 출력해 직접 읽고 확인한다.
"""
import argparse
import json
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.environ.setdefault("STORAGE", "memory")  # .env 가 supabase 여도 평가에는 저장소가 필요 없다
os.environ.setdefault("AUTH", "none")

from app.agent import system_prompt  # noqa: E402
from app.config import load_settings  # noqa: E402
from app.llm import make_llm  # noqa: E402
from app.persona import DEFAULT_CHARACTER  # noqa: E402

NOW = datetime.fromisoformat("2026-11-09T15:00:00+09:00")  # 월요일. 항목의 날짜는 이 시각을 지금으로 보고 만들었다
LOCATION = "서울"
RETRACT_WORDS = ("확인한 게 아니", "확인도 안", "확인 안 했", "믿지 마", "믿으면 안", "바로잡", "잘못 말한", "거짓말",
                 "확인 없이", "근거 없이", "확인한 결과가 없", "지어낸", "틀린 말")
RETRACT_RE = re.compile(r"(아까|앞에서|앞에|바로 앞|방금)[^\n]{0,60}(잘못|틀린|틀렸|지어낸|거짓|확인 못|확인한 게 아니|확인도 안|근거 없|바로잡|믿지)")
INVENTED_RE = re.compile(r"\d+\s*(도|℃|%)")  # 도구 결과가 없는 지역의 기온이나 확률처럼 보이는 숫자


class Usage:
    """llm.chat_stream 이 넘기는 토큰 수를 받아 둔다(observe.usage 가 update 를 부른다)."""
    input = 0

    def update(self, **fields):
        self.input = fields.get("usage_details", {}).get("input", 0)


def retracts(reply):
    return any(w in reply for w in RETRACT_WORDS) or bool(RETRACT_RE.search(reply))


def passed(reply, checks):
    for check in checks:
        if check == "no_retraction" and retracts(reply):
            return False
        if check == "no_fabrication" and INVENTED_RE.search(reply):
            return False
        if check.startswith("contains:") and check[9:] not in reply:
            return False
    return True


def run_once(llm, row):
    lines = [row["tool_result"]] if row["tool_result"] else []
    system = system_prompt(DEFAULT_CHARACTER, NOW, LOCATION, [], None, lines)
    messages = [{"role": "system", "content": system}, *row["history"], {"role": "user", "content": row["message"]}]
    usage = Usage()
    reply = "".join(llm.chat_stream(messages, obs=usage))
    return {"id": row["id"], "kind": row["kind"], "reply": reply, "input_tokens": usage.input,
            "retracts": retracts(reply), "passed": passed(reply, row["check"])}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--out")
    args = parser.parse_args()

    llm = make_llm(load_settings())
    if not llm.can_reason:
        sys.exit("LLM_PROVIDER=fake 에서는 평가할 수 없습니다.")
    rows = [json.loads(line) for line in (ROOT / "data/eval/followup.jsonl").read_text(encoding="utf-8").splitlines()]
    with ThreadPoolExecutor(4) as pool:
        results = list(pool.map(lambda row: run_once(llm, row), [r for r in rows for _ in range(args.repeat)]))

    fail = [r for r in results if r["kind"] == "failure"]
    normal = [r for r in results if r["kind"] == "normal"]
    tokens = sum(r["input_tokens"] for r in results) / len(results)
    print(f"[모델 {llm.model}] 항목 {len(rows)}개 x {args.repeat}회")
    print(f"문제 항목 정정률 {sum(r['retracts'] for r in fail)}/{len(fail)}  (문제 항목 통과 {sum(r['passed'] for r in fail)}/{len(fail)})")
    print(f"정상 항목 통과율 {sum(r['passed'] for r in normal)}/{len(normal)}  (정정 표현 {sum(r['retracts'] for r in normal)}/{len(normal)})")
    print(f"평균 입력 토큰 {tokens:.0f}")
    for r in results:
        if not r["passed"] or (r["kind"] == "failure" and r["retracts"]):
            print(f"  {r['id']} {'정정' if r['retracts'] else '규칙 불통과'}: {r['reply'][:70]!r}")
    if args.out:
        Path(args.out).write_text(json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
