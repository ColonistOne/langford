"""Tests for the for-you discovery supplement in the engage loop (v0.16).

Covers ``_pull_for_you_posts`` (extract POST items, non-fatal) and the
``_engage_tick`` wiring: when ``for_you`` is set the personalised feed is
consulted BEFORE the per-colony round-robin, but it's additive — an empty or
failed for-you pull falls through to the per-colony source unchanged, and with
``for_you=False`` the endpoint is never touched.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import langford.__main__ as m
from langford.__main__ import _engage_tick, _pull_for_you_posts


def _run(coro):
    return asyncio.run(coro)


def _post(pid: str, author_id: str = "other-1") -> dict:
    return {
        "id": pid,
        "author": {"id": author_id, "username": "other"},
        "title": f"post {pid}",
        "body": "context",
        "comment_count": 0,
    }


class _FakeClient:
    """Stub client with a controllable for-you feed + per-colony posts."""

    def __init__(self, for_you=None, posts_by_colony=None, raise_for_you=False):
        self._for_you = for_you if for_you is not None else {"items": []}
        self._posts_by_colony = posts_by_colony or {}
        self._raise_for_you = raise_for_you
        self.raw_calls: list[tuple] = []
        self.get_posts_calls: list[dict] = []

    def _raw_request(self, method, path, **kwargs):
        self.raw_calls.append((method, path))
        if self._raise_for_you:
            raise RuntimeError("boom")
        return self._for_you

    def get_posts(self, **kwargs):
        self.get_posts_calls.append(kwargs)
        return {"items": self._posts_by_colony.get(kwargs.get("colony"), [])}

    def get_comments(self, pid):
        return []


def _toolkit(client: _FakeClient) -> SimpleNamespace:
    return SimpleNamespace(client=client)


def _patch_dispatch(monkeypatch):
    """Record what the agent would have been handed, without running LangGraph."""
    dispatched: list[str] = []

    async def _fake_invoke(agent, payload):
        # payload messages[0] is the HumanMessage built from the candidate; we
        # only need to know a dispatch happened + which post via the recorder.
        return {"messages": [SimpleNamespace(content="ok")]}

    monkeypatch.setattr(m, "_invoke_agent_with_retry", _fake_invoke)

    # Wrap _build_engage_message to capture the candidate id under test.
    real_build = m._build_engage_message

    def _spy_build(post, comments):
        dispatched.append(post["id"])
        return real_build(post, comments)

    monkeypatch.setattr(m, "_build_engage_message", _spy_build)
    return dispatched


# ---- _pull_for_you_posts -------------------------------------------------

def test_pull_for_you_extracts_only_post_items():
    feed = {
        "items": [
            {"kind": "post", "post": _post("p1")},
            {"kind": "comment", "comment": {"id": "c1"}},  # dropped
            {"kind": "post", "post": _post("p2")},
            {"kind": "post"},  # no post payload → dropped
        ]
    }
    posts = _run(_pull_for_you_posts(_toolkit(_FakeClient(for_you=feed)), 10))
    assert [p["id"] for p in posts] == ["p1", "p2"]


def test_pull_for_you_is_non_fatal_on_error():
    posts = _run(_pull_for_you_posts(_toolkit(_FakeClient(raise_for_you=True)), 10))
    assert posts == []


# ---- _engage_tick for-you wiring ----------------------------------------

def test_for_you_candidate_engaged_before_colony(monkeypatch):
    dispatched = _patch_dispatch(monkeypatch)
    client = _FakeClient(
        for_you={"items": [{"kind": "post", "post": _post("fy-1")}]},
        posts_by_colony={"findings": [_post("col-1")]},
    )
    _run(_engage_tick(
        agent=object(), toolkit=_toolkit(client), colonies=["findings"],
        my_id="me", seen_ids=set(), rr_index=[0], candidate_limit=10,
        seen_file=None, for_you=True,
    ))
    # For-you post engaged; per-colony source never consulted this tick.
    assert dispatched == ["fy-1"]
    assert client.raw_calls == [("GET", "/feed/for-you?limit=10")]
    assert client.get_posts_calls == []


def test_for_you_empty_falls_through_to_colony(monkeypatch):
    dispatched = _patch_dispatch(monkeypatch)
    client = _FakeClient(
        for_you={"items": []},
        posts_by_colony={"findings": [_post("col-1")]},
    )
    _run(_engage_tick(
        agent=object(), toolkit=_toolkit(client), colonies=["findings"],
        my_id="me", seen_ids=set(), rr_index=[0], candidate_limit=10,
        seen_file=None, for_you=True,
    ))
    assert dispatched == ["col-1"]  # fell through
    assert client.raw_calls == [("GET", "/feed/for-you?limit=10")]
    assert client.get_posts_calls  # colony source was used


def test_for_you_disabled_never_hits_endpoint(monkeypatch):
    dispatched = _patch_dispatch(monkeypatch)
    client = _FakeClient(
        for_you={"items": [{"kind": "post", "post": _post("fy-1")}]},
        posts_by_colony={"findings": [_post("col-1")]},
    )
    _run(_engage_tick(
        agent=object(), toolkit=_toolkit(client), colonies=["findings"],
        my_id="me", seen_ids=set(), rr_index=[0], candidate_limit=10,
        seen_file=None, for_you=False,
    ))
    assert dispatched == ["col-1"]
    assert client.raw_calls == []  # for-you endpoint untouched
    assert client.get_posts_calls


def test_for_you_skips_self_and_seen_then_falls_through(monkeypatch):
    dispatched = _patch_dispatch(monkeypatch)
    client = _FakeClient(
        for_you={"items": [
            {"kind": "post", "post": _post("mine", author_id="me")},  # self → skip
            {"kind": "post", "post": _post("seen-1")},                # already seen → skip
        ]},
        posts_by_colony={"findings": [_post("col-1")]},
    )
    _run(_engage_tick(
        agent=object(), toolkit=_toolkit(client), colonies=["findings"],
        my_id="me", seen_ids={"seen-1"}, rr_index=[0], candidate_limit=10,
        seen_file=None, for_you=True, for_you_limit=5,
    ))
    assert dispatched == ["col-1"]  # both for-you items filtered → fell through
    assert client.raw_calls == [("GET", "/feed/for-you?limit=5")]  # honors for_you_limit
