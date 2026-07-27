"""Tests for original-post creation on Moltbotden.

The property under test is not "Langford can post". It is that **posting is
held to a stricter standard than replying**, because the guard that does most
of the work on replies has no corpus on a blank page.

Structured must-refuse / must-allow like `test_grounding.py`, for the same
reason: a rule that refuses everything satisfies a deny-only suite, and the
strict-original rule is exactly the kind that would.
"""

from __future__ import annotations

import asyncio

import pytest

from langford.grounding import refusal_reason, refusal_reason_for_original
from langford.moltbotden import (
    POST_CHAR_CAP,
    POST_URL_CAP,
    MoltbotdenPlatform,
    make_ref,
)
from langford.participation import (
    COULD_NOT_REACH,
    DECLINED_CADENCE,
    DECLINED_EMPTY_COMPOSE,
    POSTED,
    REFUSED_UNGROUNDED,
    CadenceGate,
    Refusal,
    originate_once,
)


def _run(coro):
    return asyncio.run(coro)


def gate(tmp_path, **kw):
    from datetime import datetime, timezone
    return CadenceGate(
        ledger_path=tmp_path / "originate.jsonl",
        _clock=lambda: datetime(2026, 7, 27, 12, 0, tzinfo=timezone.utc),
        **kw,
    )


def rows(g):
    import json
    return [json.loads(l) for l in g.ledger_path.read_text().splitlines() if l.strip()]


# --- the asymmetry that justifies a separate rule ---------------------------


def test_a_figure_allowed_in_a_reply_is_refused_in_a_post():
    """The whole reason `refusal_reason_for_original` exists.

    In a reply, "40%" is fine when the thread being replied to contains it —
    the corpus grounds it. On a blank page there is no corpus, so the identical
    sentence is unsourced by construction.
    """
    sentence = "The 40% figure is doing a lot of work here."
    source_with_it = "Someone measured 40% false positives behind the NAT pool."

    assert refusal_reason(sentence, source=source_with_it) is None
    assert refusal_reason_for_original(sentence) is not None, (
        "an original post has no corpus; every figure in it is invented by "
        "construction and must be refused")


ORIGINAL_MUST_REFUSE = [
    ("Rate limiters fail at 40% of deployments.", "bare unsourced percentage"),
    ("I benchmarked three approaches last week.", "first-person measurement"),
    ("Our gateway drops 12ms per hop.", "owned infrastructure + figure"),
    ("p95 latency under 12ms is achievable here.", "unit-bearing figure"),
]

ORIGINAL_MUST_ALLOW = [
    ("Two things this den keeps conflating: the limiter and the identity it "
     "keys on.", "small integers are rhetorical counting"),
    ("If the bucket state lives in process memory, a restart voids the "
     "limiter. Is that a bug or an accepted cost?",
     "conditional reasoning with no figures"),
    ("I have no telemetry, so treat this as a question rather than a finding.",
     "explicit disclaimer must not be eaten"),
    ("Does anyone key rate limits on identity rather than address?",
     "a plain question"),
]


@pytest.mark.parametrize("text,why", ORIGINAL_MUST_REFUSE,
                         ids=[w for _, w in ORIGINAL_MUST_REFUSE])
def test_original_must_refuse(text, why):
    assert refusal_reason_for_original(text) is not None, (
        f"strict original rule PERMITTED {why}: {text!r}")


@pytest.mark.parametrize("text,why", ORIGINAL_MUST_ALLOW,
                         ids=[w for _, w in ORIGINAL_MUST_ALLOW])
def test_original_must_allow(text, why):
    reason = refusal_reason_for_original(text)
    assert reason is None, (
        f"strict original rule REFUSED ordinary writing ({why}): {text!r} -> {reason}\n"
        "A rule that refuses everything passes every deny-only test and makes "
        "the feature useless without failing anything.")


def test_strict_original_rule_is_load_bearing():
    """Mutation: if it delegates to the permissive path, the suite must notice."""
    import langford.grounding as g

    probe = "Rate limiters fail at 40% of deployments."
    assert refusal_reason_for_original(probe) is not None

    original = g.refusal_reason
    try:
        g.refusal_reason = lambda text, *, source: None
        assert g.refusal_reason_for_original(probe) is None, (
            "sabotage changed nothing — the original-post rule is not actually "
            "delegating to the guard, so its green means nothing")
    finally:
        g.refusal_reason = original
    assert refusal_reason_for_original(probe) is not None


# --- adapter caps ------------------------------------------------------------


class FakeAdapter(MoltbotdenPlatform):
    def __init__(self, ok=True):
        super().__init__(api_key="k", agent_id="langford")
        self.calls = []
        self._ok = ok

    def _request(self, method, path, body=None):
        self.calls.append((method, path, body))
        if not self._ok:
            from langford.moltbotden import MoltbotdenError
            raise MoltbotdenError("boom")
        return {"post": {"id": "p-new"}}


def test_create_post_refuses_over_cap_without_calling_the_server():
    a = FakeAdapter()
    assert _run(a.create_post("technical", "t", "z" * (POST_CHAR_CAP + 1))) is None
    assert a.calls == [], "an over-cap post must not reach the server at all"


def test_create_post_refuses_too_many_urls_rather_than_stripping():
    a = FakeAdapter()
    body = " ".join(f"see https://example{i}.com" for i in range(POST_URL_CAP + 2))
    assert _run(a.create_post("technical", "t", body)) is None
    assert a.calls == [], "link-capped post must not be silently stripped and sent"


def test_create_post_allows_one_link():
    """Must-allow control for the URL cap — one link is the measured-safe case."""
    a = FakeAdapter()
    ref = _run(a.create_post("technical", "t", "read https://example.com for context"))
    assert ref == make_ref("technical", "p-new")


def test_create_post_returns_a_ref_not_a_bare_id():
    a = FakeAdapter()
    assert _run(a.create_post("philosophy", "title", "body")) == "philosophy/p-new"


def test_create_post_returns_none_on_transport_failure():
    assert _run(FakeAdapter(ok=False).create_post("technical", "t", "b")) is None


# --- the originate flow records exactly one typed row ------------------------


class FakePlatform:
    name = "moltbotden"
    supports_threading = True
    max_reply_chars = 500

    def __init__(self, ref="technical/p1"):
        self.agent_id = "langford"
        self._ref = ref
        self.created = []

    async def create_post(self, den, title, content):
        self.created.append((den, title, content))
        return self._ref


@pytest.mark.parametrize("compose,expected", [
    (lambda den: ("t", "a real post"), POSTED),
    (lambda den: None, DECLINED_EMPTY_COMPOSE),
    (lambda den: ("", ""), DECLINED_EMPTY_COMPOSE),
    (lambda den: Refusal(REFUSED_UNGROUNDED, {"why": "invented a figure"}),
     REFUSED_UNGROUNDED),
], ids=["posted", "compose-none", "compose-empty", "ungrounded"])
def test_originate_records_one_row(tmp_path, compose, expected):
    p = FakePlatform()
    g = gate(tmp_path)
    row = _run(originate_once(p, g, compose, dens=["technical"]))
    assert row["decision"] == expected
    assert row["kind"] == "post", "post rows must be distinguishable from replies"
    assert len(rows(g)) == 1


def test_create_post_failure_is_not_filed_as_a_choice(tmp_path):
    """`create_post` returns None for a refused body AND an unreachable server.

    From the caller those are the same observation, so it must record
    COULD_NOT_REACH rather than implying Langford decided anything.
    """
    class Failing(FakePlatform):
        async def create_post(self, den, title, content):
            return None

    g = gate(tmp_path)
    row = _run(originate_once(Failing(), g, lambda den: ("t", "b"), dens=["technical"]))
    assert row["decision"] == COULD_NOT_REACH
    assert row["stage"] == "create_post"


def test_post_cadence_is_independent_of_the_reply_cadence(tmp_path):
    """Separate ledger, separate gate — replies must not buy posting budget."""
    g = gate(tmp_path, min_interval_hours=999)
    g.record(POSTED, kind="post", ref="technical/earlier")
    row = _run(originate_once(FakePlatform(), g, lambda den: ("t", "b"),
                              dens=["technical"]))
    assert row["decision"] in (DECLINED_CADENCE, "declined_daily_cap")
    assert FakePlatform().created == [], "gated run must not have created anything"
