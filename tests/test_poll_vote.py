"""Tests for the poll-vote loop helpers (v0.14).

Covers:
- Ledger I/O (load + record, including the ``_skip`` sentinel).
- Prompt-builder shape for a synthetic poll dict.
- Tool-call extractor for both ``option_id`` (scalar) and
  ``option_ids`` (list) arg shapes the LLM may emit.
- ``_pull_poll_snapshot`` two-stage fetch (v0.14.1): list endpoint
  strips poll metadata, so the detail endpoint must be consulted for
  each candidate to discover options + closure + user_voted state.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace

from langford.__main__ import (
    _build_poll_message,
    _extract_voted_option,
    _load_voted_polls,
    _pull_poll_snapshot,
    _record_voted_poll,
)


def _poll(
    *,
    pid: str = "poll-uuid-1",
    title: str = "What's the most useful colony for technical findings?",
    body: str = "Trying to decide where to crosspost a write-up.",
    options: list | None = None,
) -> dict:
    return {
        "id": pid,
        "colony": "meta",
        "title": title,
        "body": body,
        "author": "someone-else",
        "options": options
        or [
            {"id": "opt_a", "text": "c/findings"},
            {"id": "opt_b", "text": "c/general"},
        ],
        "multiple_choice": False,
        "closes_at": None,
    }


# ── Ledger ──────────────────────────────────────────────────────────


def test_load_voted_polls_missing_file_returns_empty_set(tmp_path: Path) -> None:
    assert _load_voted_polls(tmp_path / "absent.txt") == set()


def test_record_and_load_round_trip(tmp_path: Path) -> None:
    f = tmp_path / "voted.txt"
    _record_voted_poll(f, "post-1", "opt_a")
    _record_voted_poll(f, "post-2", "opt_b")
    assert _load_voted_polls(f) == {"post-1", "post-2"}


def test_skip_sentinel_is_treated_as_voted(tmp_path: Path) -> None:
    f = tmp_path / "voted.txt"
    _record_voted_poll(f, "post-skipped", "_skip")
    assert "post-skipped" in _load_voted_polls(f)


def test_blank_lines_in_ledger_are_ignored(tmp_path: Path) -> None:
    f = tmp_path / "voted.txt"
    f.write_text("\npost-1 opt_a 2026-01-01T00:00:00+00:00\n\n")
    assert _load_voted_polls(f) == {"post-1"}


# ── Prompt builder ──────────────────────────────────────────────────


def test_build_poll_message_includes_title_options_and_post_id() -> None:
    poll = _poll()
    msg = _build_poll_message(poll)
    content = msg.content
    assert poll["title"] in content
    assert 'option_id="opt_a"' in content
    assert 'option_id="opt_b"' in content
    assert 'c/findings' in content
    assert poll["id"] in content
    # One-shot guard — agent must be told to skip OR vote, nothing else.
    assert "skip" in content.lower()
    assert "colony_vote_poll" in content


def test_build_poll_message_handles_missing_body() -> None:
    poll = _poll(body="")
    msg = _build_poll_message(poll)
    # No "Context:" line when body is empty.
    assert "Context:" not in msg.content


# ── Tool-call extractor ────────────────────────────────────────────


def _tool_msg(tool_calls: list[dict]) -> SimpleNamespace:
    return SimpleNamespace(tool_calls=tool_calls, content="")


def test_extract_voted_option_scalar_option_id() -> None:
    result = {
        "messages": [
            _tool_msg(
                [
                    {
                        "name": "colony_vote_poll",
                        "args": {"post_id": "p1", "option_id": "opt_a"},
                    }
                ]
            )
        ]
    }
    assert _extract_voted_option(result) == "opt_a"


def test_extract_voted_option_list_option_ids() -> None:
    result = {
        "messages": [
            _tool_msg(
                [
                    {
                        "name": "colony_vote_poll",
                        "args": {"post_id": "p1", "option_ids": ["opt_b", "opt_c"]},
                    }
                ]
            )
        ]
    }
    assert _extract_voted_option(result) == "opt_b"


def test_extract_voted_option_ignores_unrelated_tool_calls() -> None:
    result = {
        "messages": [
            _tool_msg(
                [
                    {"name": "colony_search_posts", "args": {"query": "foo"}},
                    {"name": "colony_get_post", "args": {"post_id": "p1"}},
                ]
            )
        ]
    }
    assert _extract_voted_option(result) is None


def test_extract_voted_option_picks_most_recent_call() -> None:
    # Two calls — extractor walks newest-first and returns the first match.
    result = {
        "messages": [
            _tool_msg(
                [{"name": "colony_vote_poll", "args": {"option_id": "old"}}]
            ),
            _tool_msg(
                [{"name": "colony_vote_poll", "args": {"option_id": "new"}}]
            ),
        ]
    }
    assert _extract_voted_option(result) == "new"


def test_extract_voted_option_returns_none_on_empty_result() -> None:
    assert _extract_voted_option({}) is None
    assert _extract_voted_option({"messages": []}) is None


# ── _pull_poll_snapshot (two-stage fetch, v0.14.1) ─────────────────


class _FakeClient:
    """Stub for ColonyToolkit.client with controllable list+detail responses.

    Tracks call counts to confirm the snapshot logic only hits get_poll
    for candidates that survive the cheap envelope filters.
    """

    def __init__(self, posts_by_colony: dict, polls_by_id: dict) -> None:
        self._posts_by_colony = posts_by_colony
        self._polls_by_id = polls_by_id
        self.get_posts_calls: list[dict] = []
        self.get_poll_calls: list[str] = []

    def get_posts(self, **kwargs):
        self.get_posts_calls.append(kwargs)
        return {"items": self._posts_by_colony.get(kwargs.get("colony"), [])}

    def get_poll(self, post_id: str):
        self.get_poll_calls.append(post_id)
        return self._polls_by_id[post_id]


def _toolkit_with(client: _FakeClient) -> SimpleNamespace:
    return SimpleNamespace(client=client)


def _envelope(pid: str, *, author_id: str = "other-1", **overrides) -> dict:
    base = {
        "id": pid,
        "author": {"id": author_id, "username": "other"},
        "title": f"poll {pid}",
        "body": "context",
        # Crucially: the real list endpoint returns metadata={} for polls.
        "metadata": {},
    }
    base.update(overrides)
    return base


def _poll_detail(*, options=None, is_closed=False, user_voted=False, multiple_choice=False) -> dict:
    # NB: ``options is None`` (not truthy check) so callers can pass [] deliberately.
    if options is None:
        options = [{"id": "opt_a", "text": "A"}, {"id": "opt_b", "text": "B"}]
    return {
        "options": options,
        "is_closed": is_closed,
        "user_voted": user_voted,
        "multiple_choice": multiple_choice,
    }


def _run(coro):
    return asyncio.run(coro)


def test_pull_snapshot_two_stage_fetches_options_from_detail() -> None:
    client = _FakeClient(
        posts_by_colony={"meta": [_envelope("p1")]},
        polls_by_id={"p1": _poll_detail()},
    )
    snap = _run(
        _pull_poll_snapshot(
            _toolkit_with(client),
            ["meta"],
            per_colony=10,
            my_id="me",
            voted=set(),
        )
    )
    assert len(snap) == 1
    assert snap[0]["id"] == "p1"
    assert len(snap[0]["options"]) == 2
    # Detail-fetch was called exactly once per candidate.
    assert client.get_poll_calls == ["p1"]


def test_pull_snapshot_skips_already_voted_via_ledger() -> None:
    client = _FakeClient(
        posts_by_colony={"meta": [_envelope("p1")]},
        polls_by_id={"p1": _poll_detail()},
    )
    snap = _run(
        _pull_poll_snapshot(
            _toolkit_with(client),
            ["meta"],
            per_colony=10,
            my_id="me",
            voted={"p1"},
        )
    )
    assert snap == []
    # Detail-fetch must NOT be called for ledger-filtered candidates —
    # that's the whole point of the cheap envelope filter.
    assert client.get_poll_calls == []


def test_pull_snapshot_skips_own_polls() -> None:
    client = _FakeClient(
        posts_by_colony={"meta": [_envelope("p1", author_id="me")]},
        polls_by_id={"p1": _poll_detail()},
    )
    snap = _run(
        _pull_poll_snapshot(
            _toolkit_with(client),
            ["meta"],
            per_colony=10,
            my_id="me",
            voted=set(),
        )
    )
    assert snap == []
    assert client.get_poll_calls == []


def test_pull_snapshot_skips_closed_polls() -> None:
    client = _FakeClient(
        posts_by_colony={"meta": [_envelope("p1")]},
        polls_by_id={"p1": _poll_detail(is_closed=True)},
    )
    snap = _run(
        _pull_poll_snapshot(
            _toolkit_with(client),
            ["meta"],
            per_colony=10,
            my_id="me",
            voted=set(),
        )
    )
    assert snap == []


def test_pull_snapshot_skips_user_voted_per_server_state() -> None:
    # Server says we've voted — even though our local ledger is empty.
    client = _FakeClient(
        posts_by_colony={"meta": [_envelope("p1")]},
        polls_by_id={"p1": _poll_detail(user_voted=True)},
    )
    snap = _run(
        _pull_poll_snapshot(
            _toolkit_with(client),
            ["meta"],
            per_colony=10,
            my_id="me",
            voted=set(),
        )
    )
    assert snap == []


def test_pull_snapshot_skips_polls_with_no_options() -> None:
    client = _FakeClient(
        posts_by_colony={"meta": [_envelope("p1")]},
        polls_by_id={"p1": _poll_detail(options=[])},
    )
    snap = _run(
        _pull_poll_snapshot(
            _toolkit_with(client),
            ["meta"],
            per_colony=10,
            my_id="me",
            voted=set(),
        )
    )
    assert snap == []


def test_pull_snapshot_handles_get_poll_exception_gracefully() -> None:
    class _ExplodingDetail(_FakeClient):
        def get_poll(self, post_id: str):
            self.get_poll_calls.append(post_id)
            raise RuntimeError("upstream 500")

    client = _ExplodingDetail(
        posts_by_colony={"meta": [_envelope("p1"), _envelope("p2")]},
        polls_by_id={},
    )
    # Per-candidate failures must not abort the whole tick.
    snap = _run(
        _pull_poll_snapshot(
            _toolkit_with(client),
            ["meta"],
            per_colony=10,
            my_id="me",
            voted=set(),
        )
    )
    assert snap == []
    assert client.get_poll_calls == ["p1", "p2"]


def test_pull_snapshot_aggregates_across_multiple_colonies() -> None:
    client = _FakeClient(
        posts_by_colony={
            "meta": [_envelope("p1")],
            "findings": [_envelope("p2")],
        },
        polls_by_id={
            "p1": _poll_detail(),
            "p2": _poll_detail(options=[{"id": "x", "text": "X"}]),
        },
    )
    snap = _run(
        _pull_poll_snapshot(
            _toolkit_with(client),
            ["meta", "findings"],
            per_colony=10,
            my_id="me",
            voted=set(),
        )
    )
    assert {s["id"] for s in snap} == {"p1", "p2"}
    assert {s["colony"] for s in snap} == {"meta", "findings"}
