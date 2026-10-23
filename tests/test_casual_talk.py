import json

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_casual_talk_uses_no_tool():
    # 예시 E06: "오늘 회사에서 힘들었어" → 도구를 쓰지 않는다
    r = client.post("/api/chat", json={"message": "오늘 회사에서 힘들었어"})
    turn = next(e for e in map(json.loads, r.text.splitlines()) if e["type"] == "turn")
    assert turn["tools"] == []
