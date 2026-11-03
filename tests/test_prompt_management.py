"""Langfuse 프롬프트 관리: 키가 없으면 코드 문장, 가져오면 그 문장과 변수 채움, 실패하면 코드 문장, 라벨 선택, 트레이스에 프롬프트 연결.
실제 Langfuse 나 LLM 은 부르지 않는다."""
from datetime import datetime

import pytest

from app import observe, router

NOW = datetime.fromisoformat("2026-11-02T14:00:00+09:00")
FROM_LANGFUSE = "Langfuse 버전 3 / 도구:\n{{tools}}\n최대 {{max_tools}}개"


class FakePrompt:
    version = 3

    def __init__(self, text, is_fallback=False):
        self.text, self.is_fallback = text, is_fallback

    def compile(self, **variables):
        return observe.fill(self.text, **variables)


class FakeClient:
    """get_prompt 호출을 기록하고, 정해 둔 프롬프트를 주거나 오류를 낸다."""

    def __init__(self, prompt=None, error=None):
        self.prompt, self.error, self.calls = prompt, error, []

    def get_prompt(self, name, **kwargs):
        self.calls.append((name, kwargs))
        if self.error:
            raise self.error
        return self.prompt


@pytest.fixture(autouse=True)
def reset_retry_clock(monkeypatch):
    """앞 테스트의 가져오기 실패가 다음 테스트를 막지 않게 한다."""
    monkeypatch.setattr(observe, "_prompt_retry_at", 0.0)


def use(monkeypatch, client):
    monkeypatch.setattr(observe, "_client", lambda: client)


def managed(label=None):
    return observe.managed_prompt("router", router.ROUTER_TEMPLATE, label, **router.router_variables())


def test_without_keys_uses_code_prompt():
    text, prompt = managed()  # conftest 가 키를 비워 두어 계측이 꺼져 있다
    assert text == router.router_prompt() and prompt is None


def test_langfuse_prompt_is_used_with_variables(monkeypatch):
    client = FakeClient(FakePrompt(FROM_LANGFUSE))
    use(monkeypatch, client)
    text, prompt = managed()
    assert text == observe.fill(FROM_LANGFUSE, **router.router_variables())
    assert "{{" not in text and prompt.version == 3
    name, kwargs = client.calls[0]
    assert name == "router" and kwargs["type"] == "text" and kwargs["fallback"] == router.ROUTER_TEMPLATE


def test_fetch_error_falls_back_to_code_prompt(monkeypatch):
    use(monkeypatch, FakeClient(error=RuntimeError("연결 안 됨")))
    assert managed() == (router.router_prompt(), None)


def test_sdk_fallback_is_not_linked(monkeypatch):
    use(monkeypatch, FakeClient(FakePrompt(router.ROUTER_TEMPLATE, is_fallback=True)))
    assert managed() == (router.router_prompt(), None)


def test_after_a_failure_it_does_not_retry_for_a_while(monkeypatch):
    client = FakeClient(error=RuntimeError("연결 안 됨"))
    use(monkeypatch, client)
    managed()
    managed()  # 두 번째는 기다리지 않고 바로 코드 문장을 쓴다
    assert len(client.calls) == 1
    monkeypatch.setattr(observe, "_prompt_retry_at", 0.0)  # 시간이 지나면 다시 시도한다
    client.error, client.prompt = None, FakePrompt(FROM_LANGFUSE)
    assert managed()[1].version == 3 and len(client.calls) == 2


def test_label_follows_env(monkeypatch):
    client = FakeClient(FakePrompt(FROM_LANGFUSE))
    use(monkeypatch, client)
    monkeypatch.setenv("ENV", "production")
    managed()
    monkeypatch.setenv("ENV", "staging")
    managed()
    managed("custom")
    assert [kwargs["label"] for _, kwargs in client.calls] == ["production", "staging", "custom"]
    managed(observe.CODE)  # 코드 문장만 쓰는 모드는 Langfuse 를 부르지 않는다
    assert len(client.calls) == 3


def test_choose_tools_links_prompt_to_trace(monkeypatch):
    use(monkeypatch, FakeClient(FakePrompt(FROM_LANGFUSE)))
    seen = {}

    class Llm:
        def complete_json(self, system, user, obs=None):
            seen["system"] = system
            return {"tools": [], "confidence": 0.9, "reason": "일상 대화"}

    class Step:
        def __init__(self):
            self.updates = {}

        def update(self, **fields):
            self.updates.update(fields)

    step = Step()
    router.model_choose("안녕", [], NOW, "서울", Llm(), step)
    assert seen["system"].startswith("Langfuse 버전 3") and step.updates["prompt"].version == 3
