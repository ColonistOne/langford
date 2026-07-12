"""Tests for the one-shot suggestions consumer (v0.17).

Langford treats the Colony's /suggestions feed as ADVISORY INPUT: the
executable candidates (follow_user, join_colony by default) are handed to its
agent, which DECIDES which, if any, to act on. Only agent-approved actions are
executed (via the suggestion's own api_method/api_path). Budget/dedup/cap
safety is enforced around the decision; kinds it can't do are never offered;
everything is non-fatal.

Mirrors dantic's test_consume_suggestions (cross-stack equivalence is the
point) — adapted to langford's toolkit.client + LangChain message shape.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import langford.__main__ as m
from langford.__main__ import _consume_suggestions, _parse_suggestion_choices


def _run(coro):
    return asyncio.run(coro)


def _sug(kind, api_method, api_path, handle="alice", api_body=None, rationale="reason"):
    return {
        "kind": kind,
        "target": {"handle": handle},
        "rationale": rationale,
        "action": {"api_method": api_method, "api_path": api_path, "api_body": api_body},
    }


class _FakeClient:
    def __init__(self, suggestions, raise_fetch=False):
        self._suggestions = suggestions
        self._raise_fetch = raise_fetch
        self.calls: list[tuple] = []  # executed actions: (method, path, body)
        self.fetched = False

    def _raw_request(self, method, path, body=None, **kwargs):
        if method == "GET" and path.startswith("/suggestions"):
            self.fetched = True
            if self._raise_fetch:
                raise RuntimeError("boom")
            return {"suggestions": self._suggestions}
        self.calls.append((method, path, body))
        return {}


def _toolkit(client):
    return SimpleNamespace(client=client)


def _patch_agent(monkeypatch, decision, prompts=None):
    async def fake_invoke(agent, payload, **kwargs):
        if prompts is not None:
            # payload is {"messages": [HumanMessage(content=...)]}
            prompts.append(str(payload["messages"][0].content))
        return {"messages": [SimpleNamespace(content=decision)]}

    monkeypatch.setattr(m, "_invoke_agent_with_retry", fake_invoke)


def _consume(client, tmp_path, monkeypatch, decision, prompts=None, **overrides):
    _patch_agent(monkeypatch, decision, prompts)
    kwargs = dict(
        limit=20,
        kinds_allowed={"follow_user", "join_colony"},
        max_actions=3,
        followed_file=tmp_path / "followed.txt",
        follows_log_file=tmp_path / "log.txt",
        follow_daily_limit=2,
    )
    kwargs.update(overrides)
    return _run(_consume_suggestions(object(), _toolkit(client), **kwargs))


# --- the parser ---


def test_parse_choices():
    assert _parse_suggestion_choices("thinking...\n1, 3", 5) == [0, 2]
    assert _parse_suggestion_choices("I'll pass.\nNONE", 5) == []
    assert _parse_suggestion_choices("2", 5) == [1]
    assert _parse_suggestion_choices("7, 2", 3) == [1]  # out-of-range dropped
    assert _parse_suggestion_choices("2, 2, 3", 5) == [1, 2]  # de-duped
    assert _parse_suggestion_choices("", 3) == []


# --- the agent decides ---


def test_only_agent_approved_actions_execute(tmp_path, monkeypatch):
    client = _FakeClient([
        _sug("follow_user", "POST", "/api/v1/users/u1/follow", handle="alice"),
        _sug("join_colony", "POST", "/api/v1/colonies/c1/join", handle="bip"),
    ])
    _consume(client, tmp_path, monkeypatch, decision="2")  # agent picks join only
    assert [p for _, p, _ in client.calls] == ["/colonies/c1/join"]
    assert not (tmp_path / "followed.txt").exists()  # follow not taken


def test_agent_none_executes_nothing(tmp_path, monkeypatch):
    client = _FakeClient([_sug("follow_user", "POST", "/api/v1/users/u1/follow")])
    _consume(client, tmp_path, monkeypatch, decision="NONE")
    assert client.calls == []


def test_candidates_and_rationales_reach_the_agent(tmp_path, monkeypatch):
    prompts: list[str] = []
    client = _FakeClient([
        _sug("follow_user", "POST", "/api/v1/users/u1/follow", handle="alice",
             rationale="you replied to alice 15 times"),
    ])
    _consume(client, tmp_path, monkeypatch, decision="NONE", prompts=prompts)
    assert prompts and "alice" in prompts[0]
    assert "you replied to alice 15 times" in prompts[0]


def test_unsupported_kinds_not_offered_agent_not_called(tmp_path, monkeypatch):
    prompts: list[str] = []
    client = _FakeClient([
        _sug("reply_intro", "POST", "/api/v1/posts/p1/comments"),
        _sug("respond_to_dm", "POST", "/api/v1/messages/m1"),
    ])
    _consume(client, tmp_path, monkeypatch, decision="1", prompts=prompts)
    assert client.calls == []
    assert prompts == []  # no candidates → agent never consulted


def test_path_guard_excludes_candidate(tmp_path, monkeypatch):
    prompts: list[str] = []
    client = _FakeClient([_sug("follow_user", "POST", "/api/v1/posts/p1/delete", handle="x")])
    _consume(client, tmp_path, monkeypatch, decision="1", prompts=prompts)
    assert client.calls == []
    assert prompts == []


def test_already_followed_not_offered(tmp_path, monkeypatch):
    (tmp_path / "followed.txt").write_text("alice\n")
    prompts: list[str] = []
    client = _FakeClient([_sug("follow_user", "POST", "/api/v1/users/u1/follow", handle="alice")])
    _consume(client, tmp_path, monkeypatch, decision="1", prompts=prompts)
    assert client.calls == []
    assert prompts == []


def test_follow_budget_enforced_after_choice(tmp_path, monkeypatch):
    log = tmp_path / "log.txt"
    today = m._today_iso_utc()
    log.write_text(f"{today}T00:00:00+00:00 a\n{today}T00:01:00+00:00 b\n")
    client = _FakeClient([
        _sug("follow_user", "POST", "/api/v1/users/u1/follow", handle="alice"),
        _sug("join_colony", "POST", "/api/v1/colonies/c1/join", handle="bip"),
    ])
    _consume(client, tmp_path, monkeypatch, decision="1, 2", follow_daily_limit=2)
    paths = [p for _, p, _ in client.calls]
    assert "/users/u1/follow" not in paths  # budget spent
    assert "/colonies/c1/join" in paths


def test_max_actions_caps_executions(tmp_path, monkeypatch):
    client = _FakeClient([
        _sug("join_colony", "POST", "/api/v1/colonies/c1/join", handle="a"),
        _sug("join_colony", "POST", "/api/v1/colonies/c2/join", handle="b"),
        _sug("join_colony", "POST", "/api/v1/colonies/c3/join", handle="c"),
    ])
    _consume(client, tmp_path, monkeypatch, decision="1, 2, 3", max_actions=2)
    assert len(client.calls) == 2


def test_approved_follow_is_recorded(tmp_path, monkeypatch):
    client = _FakeClient([_sug("follow_user", "POST", "/api/v1/users/u1/follow", handle="alice")])
    _consume(client, tmp_path, monkeypatch, decision="1")
    assert ("POST", "/users/u1/follow", None) in client.calls
    assert "alice" in (tmp_path / "followed.txt").read_text()


def test_fetch_failure_is_non_fatal(tmp_path, monkeypatch):
    prompts: list[str] = []
    client = _FakeClient([], raise_fetch=True)
    _consume(client, tmp_path, monkeypatch, decision="1", prompts=prompts)
    assert client.calls == []
    assert prompts == []


def test_disabled_short_circuits_before_fetch(tmp_path, monkeypatch):
    client = _FakeClient([_sug("follow_user", "POST", "/api/v1/users/u1/follow")])
    _consume(client, tmp_path, monkeypatch, decision="1", kinds_allowed=set())
    _consume(client, tmp_path, monkeypatch, decision="1", max_actions=0)
    assert client.calls == []
    assert client.fetched is False
