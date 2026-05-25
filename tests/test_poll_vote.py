"""Tests for the poll-vote loop helpers (v0.14).

Covers:
- Ledger I/O (load + record, including the ``_skip`` sentinel).
- Prompt-builder shape for a synthetic poll dict.
- Tool-call extractor for both ``option_id`` (scalar) and
  ``option_ids`` (list) arg shapes the LLM may emit.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from langford.__main__ import (
    _build_poll_message,
    _extract_voted_option,
    _load_voted_polls,
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
