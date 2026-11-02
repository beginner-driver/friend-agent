"""Langfuse 계측: 가리기, 키가 없을 때 꺼짐, 한 턴의 관찰 단위 구조, 트레이스 속성. 실제 Langfuse 는 부르지 않는다."""
from contextlib import contextmanager

import langfuse

from app import agent, observe
from app.llm import FakeLLM
from app.main import store

CURRENT = {}  # 지금 propagate_attributes 안에서 걸려 있는 속성
SEEN = {}     # 단위 이름 → 그 단위가 만들어질 때 걸려 있던 속성
FLUSHES = []  # flush() 가 불린 횟수


@contextmanager
def fake_propagate(**attrs):
    CURRENT.clear()
    CURRENT.update(attrs)
    try:
        yield
    finally:
        CURRENT.clear()


class FakeLangfuse:
    """start_observation 호출을 (부모 이름, 이름, 종류)로 기록하고, update 로 들어온 값을 이름별로 모은다."""

    def __init__(self, calls, updates, name=None):
        self.calls, self.updates, self.name = calls, updates, name

    def start_observation(self, *, name, as_type="span", **fields):
        self.calls.append((self.name, name, as_type))
        self.updates.setdefault(name, {}).update(fields)
        SEEN[name] = dict(CURRENT)
        return FakeLangfuse(self.calls, self.updates, name)

    def update(self, **fields):
        self.updates.setdefault(self.name, {}).update(fields)

    def end(self):
        pass

    def flush(self):
        FLUSHES.append(1)


def run(message):
    return list(agent.run_turn(message, "langfuse-test", FakeLLM(delay=0), store, True))


def test_mask_hides_phone_email_card_but_keeps_other_text():
    data = {"a": "전화 010-1234-5678, 메일 kim.sh@example.com", "b": ["카드 1234 5678 9012 3456", 7], "c": "2026-11-02 오후 3시"}
    masked = observe.mask(data=data)
    assert masked == {"a": "전화 [전화번호], 메일 [이메일]", "b": ["카드 [카드번호]", 7], "c": "2026-11-02 오후 3시"}


def test_mask_never_raises():
    class Bad(dict):
        def items(self):
            raise RuntimeError("boom")

    assert observe.mask(data=Bad()) == "[가리지 못해 뺌]"


def test_off_without_keys_and_turn_still_works():
    assert observe._client() is None
    with observe.observation("x", input="a") as obs:
        obs.update(output="b")
    assert any(e["type"] == "done" for e in run("안녕"))


def test_turn_records_named_observations(monkeypatch):
    calls, updates = [], {}
    monkeypatch.setattr(observe, "_client", lambda: FakeLangfuse(calls, updates))
    run("안녕, 오늘 기분 좋아")
    assert calls == [
        (None, "chat-turn", "span"),
        ("chat-turn", "safety", "span"),
        ("chat-turn", "choose-tools", "generation"),
        ("chat-turn", "reply", "generation"),
        ("chat-turn", "memory-update", "generation"),
    ]
    assert updates["safety"]["output"] == {"crisis": False}
    assert updates["choose-tools"]["metadata"]["decided_by"] == "rule"
    assert updates["reply"]["model"] == "fake" and "completion_start_time" in updates["reply"]
    assert updates["chat-turn"]["output"].startswith("(가짜 응답)")


def test_every_observation_carries_trace_attributes(monkeypatch):
    """사용자별 비용 같은 집계는 속성이 붙은 단위만 센다. 루트뿐 아니라 모든 단위에 붙어야 한다."""
    monkeypatch.setattr(langfuse, "propagate_attributes", fake_propagate)
    monkeypatch.setattr(observe, "_client", lambda: FakeLangfuse([], {}))
    SEEN.clear()
    run("안녕, 오늘 기분 좋아")
    assert set(SEEN) == {"chat-turn", "safety", "choose-tools", "reply", "memory-update"}
    for name, attrs in SEEN.items():
        assert attrs["user_id"] == "langfuse-test", name
        assert attrs["session_id"].startswith("langfuse-test-20"), name
        assert attrs["version"] == "local" and attrs["metadata"]["commit"] == "local", name
    assert SEEN["chat-turn"]["metadata"]["safety"] == "false" and "tags" not in SEEN["chat-turn"]


def test_chosen_tools_are_tagged_on_later_steps(monkeypatch):
    monkeypatch.setattr(langfuse, "propagate_attributes", fake_propagate)
    monkeypatch.setattr(observe, "_client", lambda: FakeLangfuse([], {}))
    SEEN.clear()
    run("내일 서울 날씨 어때?")  # 도구 호출은 실패해도 기록은 남는다
    assert "get_weather" in SEEN["reply"]["tags"] and SEEN["reply"]["metadata"]["tools"] == "get_weather"
    assert "tags" not in SEEN["choose-tools"]  # 고르기 전에 만들어진 단위에는 아직 모른다


def test_crisis_turn_is_tagged_from_the_start(monkeypatch):
    monkeypatch.setattr(langfuse, "propagate_attributes", fake_propagate)
    monkeypatch.setattr(observe, "_client", lambda: FakeLangfuse([], {}))
    SEEN.clear()
    run("요즘 너무 죽고 싶어")
    assert SEEN["chat-turn"]["tags"] == ["safety"] and SEEN["safety"]["metadata"]["safety"] == "true"


def test_greeting_carries_user_and_session(monkeypatch):
    monkeypatch.setattr(langfuse, "propagate_attributes", fake_propagate)
    monkeypatch.setattr(observe, "_client", lambda: FakeLangfuse([], {}))
    SEEN.clear()
    assert agent.greet("greeting-test", FakeLLM(delay=0), store) is not None
    assert SEEN["greeting"]["user_id"] == "greeting-test" and SEEN["greeting"]["session_id"].startswith("greeting-test-20")


def test_records_are_flushed_before_the_response_closes(monkeypatch):
    """Cloud Run 은 응답이 끝나면 CPU 를 불규칙하게 줘서, 백그라운드 전송에 맡기면 기록이 빠진다."""
    monkeypatch.setattr(observe, "_client", lambda: FakeLangfuse([], {}))
    FLUSHES.clear()
    run("안녕")
    assert len(FLUSHES) == 1
    assert agent.greet("flush-test", FakeLLM(delay=0), store) is not None
    assert len(FLUSHES) == 2
    stream = agent.run_turn("안녕", "flush-test", FakeLLM(delay=0), store, True)
    next(stream)
    stream.close()  # 사용자가 중간에 화면을 닫은 경우에도 보낸다
    assert len(FLUSHES) == 3


def test_flush_does_nothing_without_keys():
    observe.flush()  # 키가 없으면 오류 없이 지나간다


def test_crisis_turn_records_safety_only(monkeypatch):
    calls, updates = [], {}
    monkeypatch.setattr(observe, "_client", lambda: FakeLangfuse(calls, updates))
    run("요즘 너무 죽고 싶어")
    assert [c[1] for c in calls] == ["chat-turn", "safety"]
    assert updates["safety"]["output"] == {"crisis": True}


def test_client_exports_only_our_spans(monkeypatch):
    """Anthropic SDK 가 만드는 span 까지 보내면 같은 모델 호출이 두 번 기록된다."""
    import langfuse
    from langfuse.span_filter import is_langfuse_span

    made = {}
    monkeypatch.setattr(langfuse, "Langfuse", lambda **kwargs: made.update(kwargs) or "client")
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-lf-test")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-lf-test")
    observe._client.cache_clear()
    try:
        assert observe._client() == "client"
    finally:
        observe._client.cache_clear()  # 다음 테스트가 키 없는 상태로 다시 만들게 한다
    assert made["should_export_span"] is is_langfuse_span and made["mask"] is observe.mask


def test_instrumentation_failure_does_not_break_the_turn(monkeypatch):
    class Broken:
        def start_observation(self, **fields):
            raise RuntimeError("boom")

    monkeypatch.setattr(observe, "_client", lambda: Broken())
    assert any(e["type"] == "done" for e in run("안녕"))
