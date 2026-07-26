"""Tests for the platform seam (``langford.platform``) and the two call sites
in ``__main__`` that were moved onto it.

Written because the refactor's 86-green baseline proved nothing about it: no
existing test touches ``_self_comments_on_post`` or ``_delete_comment_via_api``,
so the suite could not have told a working relocation from a broken one. A
passing suite over code it never reaches is the same instrument-with-no-needle
problem the seam's docstring is about.

Every behaviour-preservation assertion below is paired with a control that must
fail if the delegation is vestigial — see ``test_delegation_is_load_bearing``.
"""

from __future__ import annotations

import asyncio

import pytest

from langford.__main__ import _delete_comment_via_api, _self_comments_on_post
from langford.platform import Comment, ColonyPlatform, Platform, Thread

ME = "langford"


def _c(cid, author, parent=None, body="x"):
    return {"id": cid, "parent_id": parent, "author": {"username": author}, "body": body}


class FakeClient:
    """Minimal stand-in for ``ColonySDK`` — records calls, replays canned data."""

    def __init__(self, comments=None, raise_on_get=False, create_result=None,
                 raise_on_create=False):
        self._comments = comments
        self._raise_on_get = raise_on_get
        self._create_result = create_result
        self._raise_on_create = raise_on_create
        self.create_calls: list[dict] = []

    def get_comments(self, post_id):
        if self._raise_on_get:
            raise RuntimeError("transport blew up")
        return self._comments

    def create_comment(self, **kwargs):
        self.create_calls.append(kwargs)
        if self._raise_on_create:
            raise RuntimeError("nope")
        return self._create_result

    def get_me(self):
        return {"username": ME}


class FakeToolkit:
    def __init__(self, client):
        self.client = client


def _run(coro):
    return asyncio.run(coro)


# --- envelope handling -------------------------------------------------------
# The Colony API has shipped a bare list AND {items:[...]} AND {comments:[...]}.
# Guessing one and getting [] back is indistinguishable from an empty thread.


@pytest.mark.parametrize(
    "payload",
    [
        [_c("a", ME)],
        {"items": [_c("a", ME)]},
        {"comments": [_c("a", ME)]},
    ],
    ids=["bare-list", "items-key", "comments-key"],
)
def test_raw_comments_handles_every_envelope(payload):
    p = ColonyPlatform(FakeToolkit(FakeClient(comments=payload)))
    items, ok = _run(p.raw_comments("p1"))
    assert ok is True
    assert len(items) == 1


def test_raw_comments_distinguishes_empty_from_unreachable():
    """The distinction the old helper could not express."""
    empty = ColonyPlatform(FakeToolkit(FakeClient(comments=[])))
    assert _run(empty.raw_comments("p1")) == ([], True)

    broken = ColonyPlatform(FakeToolkit(FakeClient(raise_on_get=True)))
    assert _run(broken.raw_comments("p1")) == ([], False)


# --- behaviour preservation at the moved call sites --------------------------


def test_self_comments_counts_only_own_top_level():
    payload = [
        _c("a", ME),                      # counts
        _c("b", ME, parent="a"),          # nested — must not count
        _c("c", "someone-else"),          # not mine — must not count
        _c("d", ME),                      # counts
    ]
    tk = FakeToolkit(FakeClient(comments=payload))
    items, n = _run(_self_comments_on_post(tk, "p1", ME))
    assert len(items) == 4
    assert n == 2


def test_self_comments_reports_zero_on_fetch_failure():
    """Original semantics kept ON PURPOSE, including their flaw.

    A failed fetch reports ``([], 0)`` — indistinguishable from an empty thread,
    which will read as "no duplicate here, go ahead and post". The adapter can
    now tell the two apart, but consuming that distinction changes dedupe
    behaviour, so it is a separate change and not smuggled into a refactor.
    """
    tk = FakeToolkit(FakeClient(raise_on_get=True))
    assert _run(_self_comments_on_post(tk, "p1", ME)) == ([], 0)


def test_delete_delegates_and_survives_a_tokenless_client():
    """No token ⇒ False, without raising. Exercises the delegation path."""
    class NoToken:
        def _ensure_token(self):
            return None
        _token = None

    assert _run(_delete_comment_via_api(FakeToolkit(NoToken()), "c1")) is False


# --- the seam itself ---------------------------------------------------------


def test_colony_platform_satisfies_the_protocol():
    assert isinstance(ColonyPlatform(FakeToolkit(FakeClient())), Platform)


def test_colony_declares_its_capabilities():
    """These are what a second platform must contradict to be worth a seam."""
    p = ColonyPlatform(FakeToolkit(FakeClient()))
    assert p.name == "colony"
    assert p.supports_threading is True
    assert p.max_reply_chars is None


def test_fetch_thread_drops_idless_comments_but_keeps_the_rest():
    payload = [_c("a", ME), {"author": {"username": ME}, "body": "no id"}, _c("b", ME)]
    p = ColonyPlatform(FakeToolkit(FakeClient(comments=payload)))
    thread = _run(p.fetch_thread("p1"))
    assert thread is not None
    assert [c.id for c in thread.comments] == ["a", "b"]
    # CONTROL: the dropped row must not have quietly become a counted comment.
    assert thread.self_top_level_count(ME) == 2


def test_fetch_thread_returns_none_when_unreachable():
    p = ColonyPlatform(FakeToolkit(FakeClient(raise_on_get=True)))
    assert _run(p.fetch_thread("p1")) is None


def test_thread_self_top_level_count():
    t = Thread(
        ref="p1", title="", body="", author="",
        comments=(
            Comment("a", ME, "x"),
            Comment("b", ME, "x", parent_id="a"),
            Comment("c", "other", "x"),
        ),
    )
    assert t.self_top_level_count(ME) == 1
    assert t.self_top_level_count("other") == 1
    assert t.self_top_level_count("nobody") == 0


@pytest.mark.parametrize(
    "result,expected",
    [
        ({"id": "c9"}, "c9"),
        ({"comment": {"id": "c9"}}, "c9"),
        ({}, None),
        (None, None),
    ],
    ids=["flat", "nested", "empty-dict", "none"],
)
def test_reply_extracts_the_id_from_either_envelope(result, expected):
    p = ColonyPlatform(FakeToolkit(FakeClient(create_result=result)))
    assert _run(p.reply("p1", "hello")) == expected


def test_reply_omits_parent_id_when_not_threading():
    """``parent_id=None`` must not be sent — a flat platform 422s on the key."""
    client = FakeClient(create_result={"id": "c1"})
    p = ColonyPlatform(FakeToolkit(client))
    _run(p.reply("p1", "hi"))
    assert client.create_calls == [{"post_id": "p1", "body": "hi"}]

    _run(p.reply("p1", "hi", parent_id="c0"))
    assert client.create_calls[-1]["parent_id"] == "c0"


def test_reply_returns_none_rather_than_raising():
    p = ColonyPlatform(FakeToolkit(FakeClient(raise_on_create=True)))
    assert _run(p.reply("p1", "hi")) is None


# --- control: is any of this actually wired in? ------------------------------


def test_delegation_is_load_bearing(monkeypatch):
    """Break the adapter; the call site must break with it.

    Without this, every assertion above could pass while ``_self_comments_on_post``
    quietly kept its own inline copy of the transport — the refactor would be
    decorative and the tests would still be green. This is the mutation check:
    it asserts the moved code is the code that runs.
    """
    import langford.__main__ as main

    async def _sabotaged(self, ref):
        return [_c("z", ME), _c("y", ME), _c("x", ME)], True

    monkeypatch.setattr(ColonyPlatform, "raw_comments", _sabotaged)
    tk = FakeToolkit(FakeClient(comments=[]))  # client says empty…
    items, n = _run(main._self_comments_on_post(tk, "p1", ME))
    # …but the sabotaged ADAPTER says three, so the adapter is what runs.
    assert (len(items), n) == (3, 3)
