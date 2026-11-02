"""Langfuse 계측을 한곳에 모은다.

LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY 가 없으면 아무것도 하지 않는다(CI, 테스트, 키 없는 컴퓨터).
계측이 실패해도 대화는 그대로 이어지도록, 이 파일 밖에서는 Langfuse 를 직접 부르지 않는다.
입력과 출력 원문은 남기되, 보내기 전에 이메일, 카드 번호, 전화번호 패턴은 가린다(mask).
서버 로그(log)에는 대화 원문을 넣지 않는다. 경고에는 오류 종류만 남긴다.

관찰 단위는 현재 컨텍스트에 걸지 않고 부모에서 직접 만든다. run_turn 은 yield 하는 제너레이터라서
next() 마다 다른 스레드에서 실행될 수 있고, 그러면 컨텍스트에 건 부모-자식 관계가 끊어진다.
같은 이유로 사용자, 세션, 버전 같은 속성(propagate_attributes)도 블록 전체에 걸지 않고,
단위를 만드는 그 한 줄만 감싼다. 속성은 부모의 attrs 를 자식이 물려받는다.
"""
import functools
import os
import re
from contextlib import contextmanager

from app import when
from app.logs import log

# 순서가 중요하다: 16자리 카드 번호가 전화번호 패턴에 일부 걸리지 않게 카드 번호를 먼저 가린다.
MASKS = [
    (re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)+"), "[이메일]"),
    (re.compile(r"(?<!\d)\d{4}[- ]?\d{4}[- ]?\d{4}[- ]?\d{4}(?!\d)"), "[카드번호]"),
    (re.compile(r"(?<!\d)0(?:1[016-9]|2|[3-6]\d)[-.\s]?\d{3,4}[-.\s]?\d{4}(?!\d)"), "[전화번호]"),
]


def mask(*, data, **kwargs):
    """Langfuse 가 값을 만들 때마다 부른다. 여기서 예외가 나면 그 묶음 전체가 버려지므로 절대 예외를 내지 않는다."""
    try:
        if isinstance(data, str):
            for pattern, label in MASKS:
                data = pattern.sub(label, data)
            return data
        if isinstance(data, dict):
            return {key: mask(data=value) for key, value in data.items()}
        if isinstance(data, (list, tuple)):
            return [mask(data=value) for value in data]
        return data
    except Exception:
        return "[가리지 못해 뺌]"


def _warn(error):
    log("WARNING", "Langfuse 계측 실패, 대화는 계속합니다", error=type(error).__name__)


@functools.cache
def _client():
    """키가 있을 때만 Langfuse 클라이언트를 만든다. .env 를 읽은 뒤에 불려야 해서 처음 쓸 때 만든다."""
    if not (os.environ.get("LANGFUSE_PUBLIC_KEY") and os.environ.get("LANGFUSE_SECRET_KEY")):
        return None
    try:
        from langfuse import Langfuse
        from langfuse.span_filter import is_langfuse_span

        # 기본 필터는 Anthropic SDK 가 스스로 만드는 gen_ai span(anthropic.messages.create)도 보내서,
        # 같은 모델 호출이 두 번 기록되고 비용이 두 배로 보인다. 이 파일이 만든 관찰 단위만 보낸다.
        # 접속 주소는 LANGFUSE_BASE_URL 을 SDK 가 읽는다. environment 는 모든 단위에 한 번에 붙는다(local, staging, production).
        return Langfuse(mask=mask, should_export_span=is_langfuse_span,
                        environment=os.environ.get("ENV", "local").strip().lower() or "local")
    except Exception as error:
        _warn(error)
        return None


def trace_attributes(user_id, tags=(), **metadata):
    """트레이스에 붙일 속성. user_id 는 서버가 로그인으로 확인한 ID(이메일 아님).

    session_id 는 사용자 ID + 한국 날짜라서 하루치 대화가 한 세션으로 묶인다. Langfuse 는 ID 를 영문, 숫자 200자까지만 받는다.
    metadata 값은 인코딩 후 200자 이하여야 해서 짧은 문자열만 넣는다.
    """
    attrs = {"user_id": user_id,
             "session_id": f"{user_id}-{when.now_in('Asia/Seoul'):%Y%m%d}",  # ponytail: 한국 시간만 지원하는 서비스라 고정
             "version": os.environ.get("K_REVISION", "local"),  # Cloud Run 이 리비전 이름을 넣어 준다
             "metadata": {"commit": os.environ.get("COMMIT_SHA", "local"), **metadata}}
    if tags:
        attrs["tags"] = list(tags)
    return attrs


def tag(obs, *tags, **metadata):
    """이후에 만드는 단계에 붙일 속성을 더한다. 도구 선택처럼 턴 도중에 알게 되는 값에 쓴다."""
    if tags:
        obs.attrs["tags"] = [*obs.attrs.get("tags", []), *tags]
    obs.attrs["metadata"] = {**obs.attrs.get("metadata", {}), **metadata}


class Obs:
    """관찰 단위 하나. 계측이 꺼져 있으면 raw 가 None 이고 아무것도 하지 않는다."""

    def __init__(self, raw=None, attrs=None):
        self.raw = raw
        self.attrs = attrs if attrs is not None else {}  # 자식 단위가 만들어질 때 물려받는 트레이스 속성

    @property
    def trace_id(self):
        """이 단위가 속한 트레이스 ID. 계측이 꺼져 있으면 None. 사용자 반응(점수)을 이 트레이스에 붙일 때 쓴다."""
        return getattr(self.raw, "trace_id", None)

    def update(self, **fields):
        try:
            if self.raw is not None:
                self.raw.update(**fields)
        except Exception as error:
            _warn(error)


def _start(source, name, as_type, attrs, fields, new_trace):
    from langfuse import propagate_attributes
    from opentelemetry import context as otel_context

    # 새 트레이스의 첫 단위는 들어온 요청의 추적 상태를 이어받지 않게 빈 컨텍스트에서 만든다.
    # Cloud Run 은 요청에 traceparent 를 붙여 일부만 sampled=1 로 표시하는데, 그 요청의 하위로 만들어지면
    # sampled=0 인 요청의 기록이 오류 없이 사라진다(span 이 기록되지 않음). 대화 기록은 요청 표본과 상관없이 항상 남겨야 한다.
    # 같은 스레드에서 짧게 걸었다 바로 풀어서 yield 하는 제너레이터에서도 안전하다.
    token = otel_context.attach(otel_context.Context()) if new_trace else None
    try:
        if not attrs:
            return source.start_observation(name=name, as_type=as_type, **fields)
        with propagate_attributes(**attrs):  # 만드는 순간의 컨텍스트에서 속성을 읽는다
            return source.start_observation(name=name, as_type=as_type, **fields)
    finally:
        if token is not None:
            otel_context.detach(token)


@contextmanager
def observation(name, as_type="span", parent=None, attrs=None, **fields):
    """with 블록 동안 관찰 단위 하나를 연다. parent 가 없으면 새 트레이스의 첫 단위가 된다.

    as_type 은 모델을 부르지 않으면 span, 부르면 generation. 블록에서 오류가 나면 ERROR 로 표시하고 다시 던진다.
    attrs 는 트레이스 속성(trace_attributes). 첫 단위에서 정하면 자식 단위가 물려받는다.
    """
    source = parent.raw if parent else _client()
    attrs = attrs if attrs is not None else (parent.attrs if parent else {})
    obs = Obs(attrs=attrs)
    if source is not None:
        try:
            obs = Obs(_start(source, name, as_type, attrs, fields, new_trace=parent is None), attrs)
        except Exception as error:
            _warn(error)
    try:
        yield obs
    except Exception as error:
        obs.update(level="ERROR", status_message=type(error).__name__)
        raise
    finally:
        try:
            if obs.raw is not None:
                obs.raw.end()
        except Exception as error:
            _warn(error)


def flush():
    """모아 둔 기록을 지금 보낸다. 키가 없으면 아무것도 하지 않는다.

    Cloud Run 은 응답이 끝나면 CPU 를 불규칙하게 줘서, SDK 의 백그라운드 전송(5초마다)에 맡기면 기록이 빠지거나 늦는다.
    그래서 응답을 닫기 전에 한 턴이 끝날 때 직접 보낸다. 대가는 응답 종료가 전송 시간만큼 늦어지는 것이다.
    """
    client = _client()
    if client is None:
        return
    try:
        client.flush()
    except Exception as error:
        _warn(error)


def score(trace_id, name, value, comment=None):
    """사용자 반응을 트레이스의 점수(BOOLEAN)로 남긴다. 보냈으면 True, 키나 트레이스 ID 가 없어 건너뛰면 False.

    같은 트레이스에 같은 이름으로 다시 보내면 덮어쓴다(score_id 가 같아서). 마음을 바꿔 눌러도 점수가 겹치지 않는다.
    comment 는 사용자가 쓴 글이라 mask 를 거치고, 점수는 응답을 닫기 전에 보낸다(Cloud Run 은 응답 뒤 CPU 를 불규칙하게 준다).
    """
    client = _client()
    if client is None or not trace_id:
        return False
    try:
        client.create_score(name=name, value=bool(value), trace_id=trace_id, data_type="BOOLEAN",
                            score_id=f"{trace_id}-{name}", comment=mask(data=comment) if comment else None)
    except Exception as error:
        _warn(error)
        return False
    flush()
    return True


def usage(obs, tokens):
    """모델 응답의 토큰 수를 generation 에 남긴다. 토큰 수가 없는 응답(가짜 LLM, 테스트)은 건너뛴다.

    Claude 는 input_tokens/output_tokens, OpenAI 호환은 prompt_tokens/completion_tokens 로 준다.
    """
    if obs is None or tokens is None:
        return
    obs.update(usage_details={
        "input": getattr(tokens, "input_tokens", None) or getattr(tokens, "prompt_tokens", None) or 0,
        "output": getattr(tokens, "output_tokens", None) or getattr(tokens, "completion_tokens", None) or 0,
    })
