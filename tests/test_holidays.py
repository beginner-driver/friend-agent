"""공휴일 도구의 기간 계산. 외부 공휴일 API 는 부르지 않고 가짜 데이터를 쓴다."""
from datetime import date, timedelta
from types import SimpleNamespace

import pytest

from app import tools, when

CTX = SimpleNamespace(timezone="Asia/Seoul")
FAKE = {2030: [(date(2030, 2, 2), "설날"), (date(2030, 3, 1), "3·1절")], 2031: [(date(2031, 1, 1), "새해")]}


@pytest.fixture(autouse=True)
def fake_holidays(monkeypatch):
    monkeypatch.setattr(tools, "holidays_of", lambda year: FAKE.get(year, []))


def dates(result):
    return [h["date"] for h in result.data["holidays"]]


def test_without_to_the_year_is_counted_from_the_start_date():
    # to 를 비우면 '오늘부터'가 아니라 from 부터 1년이다. 먼 연도를 from 만 주고 물어도 그 해 공휴일이 나와야 한다.
    r = tools.get_holidays({"from": "2030-01-02"}, CTX)
    assert r.data["to"] == "2031-01-02"
    assert dates(r) == ["2030-02-02", "2030-03-01", "2031-01-01"]


def test_without_arguments_it_is_one_year_from_today():
    today = when.now_in("Asia/Seoul").date()
    r = tools.get_holidays({}, CTX)
    assert r.data["from"] == today.isoformat() and r.data["to"] == (today + timedelta(days=365)).isoformat()


def test_given_range_is_kept():
    r = tools.get_holidays({"from": "2030-02-01", "to": "2030-02-28"}, CTX)
    assert (r.data["from"], r.data["to"]) == ("2030-02-01", "2030-02-28") and dates(r) == ["2030-02-02"]


def test_bad_dates_fall_back_to_one_year_from_today():
    today = when.now_in("Asia/Seoul").date()
    r = tools.get_holidays({"from": "내일", "to": "모레"}, CTX)
    assert r.data["from"] == today.isoformat() and r.data["to"] == (today + timedelta(days=365)).isoformat()
