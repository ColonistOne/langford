"""End-to-end check of the duplicate guard through the REAL langchain tool.

`test_dedupe.py` exercises the guard against a fake client, which proves the
policy but not the thing that actually decides whether this is safe to ship:
**what the agent framework does when the guard raises.**

If a raised refusal escaped the tool boundary it would abort the agent run, and
a crashed run is a worse outcome than the duplicate it prevents. So this builds a
real `ColonyToolkit` over a stub client, installs the guard, and invokes the real
`create_comment` tool — asserting the run survives, the model is told why, and
nothing was sent.

Kept separate from the unit tests because it depends on langchain-colony's
internals (`_api`'s catch-all at the tool boundary). If that contract ever
changes, this file should fail loudly rather than the guard quietly starting to
crash agent runs in production.
"""

from __future__ import annotations

import pytest

from langchain_colony import ColonyToolkit

from langford.dedupe import install_duplicate_guard

ME = "langford"


class StubClient:
    """Enough of ColonyClient for the comment tool to run."""

    def __init__(self, comments):
        self._comments = comments
        self.create_calls: list[dict] = []

    def get_comments(self, post_id, **kw):
        return self._comments

    def create_comment(self, **kwargs):
        self.create_calls.append(kwargs)
        return {"id": "should-not-exist", "body": kwargs.get("body")}


def _mine(cid):
    return {"id": cid, "parent_id": None, "author": {"username": ME}}


#: The tool is named `colony_comment_on_post`, not `create_comment` — the SDK
#: method and the exposed tool have different names. The first version of this
#: file looked for the method name, found nothing, and SKIPPED both tests: two
#: green skips that asserted nothing about the guard. A skip is not a pass, and
#: a lookup that silently degrades to "not applicable" is the same shape as the
#: bug this whole module exists to prevent.
COMMENT_TOOL = "colony_comment_on_post"


def _comment_tool(toolkit):
    names = [t.name for t in toolkit.get_tools()]
    for t in toolkit.get_tools():
        if t.name == COMMENT_TOOL:
            return t
    raise AssertionError(
        f"{COMMENT_TOOL!r} not exposed by this langchain-colony version; "
        f"available: {sorted(names)}"
    )


def test_refusal_reaches_the_model_as_text_and_does_not_crash_the_run():
    client = StubClient(comments=[_mine("already-here")])
    toolkit = ColonyToolkit(client=client)
    install_duplicate_guard(toolkit, ME, enabled=True)

    out = _comment_tool(toolkit).invoke({"post_id": "p1", "body": "duplicate"})

    # 1. The run survived — we got a value back rather than an exception.
    assert isinstance(out, str)
    # 2. The model is told what happened and what to do instead.
    low = out.lower()
    assert "refusing" in low or "error" in low
    assert "parent_id" in low
    # 3. And the whole point: nothing was sent.
    assert client.create_calls == []


def test_legitimate_comment_still_goes_through_the_real_tool():
    """CONTROL. Without this, a guard that broke create_comment entirely would
    pass the test above."""
    client = StubClient(comments=[])
    toolkit = ColonyToolkit(client=client)
    install_duplicate_guard(toolkit, ME, enabled=True)

    _comment_tool(toolkit).invoke({"post_id": "p1", "body": "first post here"})

    assert len(client.create_calls) == 1
    assert client.create_calls[0]["body"] == "first post here"
