"""Tests for the duplicate top-level guard.

The bug these cover has survived three mitigations since 2026-05-02 — a prompt
directive, a post-dispatch validator, and deletion — because all three ran
*after* generation. So the assertions here are mostly about the create never
happening: ``orig_called is False`` matters more than the exception type.

Two properties get controls of their own, because both are places where a plausible
implementation is silently wrong:

* **fails closed** — an unverifiable comment list must refuse, paired with a
  control proving a verifiable-and-empty list still allows. Without the control,
  a guard that refuses everything would pass the fail-closed test.
* **outermost** — the guard must sit outside the cognition wrapper, so a refusal
  costs no network call. Tested by ordering two wrappers and asserting the inner
  one never runs.
"""

from __future__ import annotations

import pytest

from langford.dedupe import (
    DuplicateTopLevelRefused,
    ENV_FLAG,
    install_duplicate_guard,
    refusal_reason,
    self_top_level_ids,
)

ME = "langford"


def _c(cid, author=ME, parent=None):
    return {"id": cid, "parent_id": parent, "author": {"username": author}}


class FakeClient:
    def __init__(self, comments=None, raise_on_get=False):
        self._comments = comments if comments is not None else []
        self._raise_on_get = raise_on_get
        self.create_calls: list[dict] = []
        self.get_calls: list[str] = []

    def get_comments(self, post_id):
        self.get_calls.append(post_id)
        if self._raise_on_get:
            raise RuntimeError("comments API down")
        return self._comments

    def create_comment(self, **kwargs):
        self.create_calls.append(kwargs)
        return {"id": "new-comment"}


class FakeToolkit:
    def __init__(self, client):
        self.client = client


# --- pure policy -------------------------------------------------------------


def test_allows_first_top_level():
    assert refusal_reason(
        self_username=ME, parent_id=None, comments=[_c("a", "other")], fetch_ok=True
    ) is None


def test_refuses_second_top_level():
    reason = refusal_reason(
        self_username=ME, parent_id=None, comments=[_c("mine")], fetch_ok=True
    )
    assert reason is not None
    # The model has to be able to act on this, so the id and the alternative
    # both have to be in the message.
    assert "mine" in reason and "parent_id" in reason


def test_nested_reply_is_never_refused():
    """A nested reply is a different act, even where a top-level already exists."""
    assert refusal_reason(
        self_username=ME, parent_id="c1", comments=[_c("mine")], fetch_ok=True
    ) is None


def test_own_nested_comments_do_not_count_as_top_level():
    assert refusal_reason(
        self_username=ME,
        parent_id=None,
        comments=[_c("mine-nested", parent="x")],
        fetch_ok=True,
    ) is None


def test_other_agents_top_level_does_not_block_me():
    assert refusal_reason(
        self_username=ME,
        parent_id=None,
        comments=[_c("theirs", "eliza-gemma"), _c("also-theirs", "dantic")],
        fetch_ok=True,
    ) is None


def test_fails_closed_when_the_list_cannot_be_fetched():
    reason = refusal_reason(
        self_username=ME, parent_id=None, comments=[], fetch_ok=False
    )
    assert reason is not None and "fails closed" in reason


def test_control_verifiable_and_empty_still_allows():
    """The control for fail-closed: refusing EVERYTHING would also pass that test.

    The old dedupe path reported a failed fetch as ``([], 0)`` — no comments,
    therefore no duplicate, therefore go ahead. These two cases are the exact
    pair it could not distinguish, and they must now decide differently.
    """
    assert refusal_reason(
        self_username=ME, parent_id=None, comments=[], fetch_ok=True
    ) is None


def test_self_top_level_ids_ignores_junk_rows():
    rows = [_c("good"), {"no": "id"}, "not-a-dict", _c("nested", parent="p")]
    assert self_top_level_ids(rows, ME) == ["good"]


# --- installed behaviour -----------------------------------------------------


def test_guard_blocks_the_create_entirely():
    client = FakeClient(comments=[_c("mine")])
    assert install_duplicate_guard(FakeToolkit(client), ME, enabled=True) is True
    with pytest.raises(DuplicateTopLevelRefused):
        client.create_comment(post_id="p1", body="dupe")
    # The point of the whole exercise: nothing was sent.
    assert client.create_calls == []


def test_guard_lets_a_legitimate_first_comment_through():
    client = FakeClient(comments=[])
    install_duplicate_guard(FakeToolkit(client), ME, enabled=True)
    out = client.create_comment(post_id="p1", body="hello")
    assert out == {"id": "new-comment"}
    assert client.create_calls == [{"post_id": "p1", "body": "hello"}]


def test_guard_lets_nested_replies_through():
    client = FakeClient(comments=[_c("mine")])
    install_duplicate_guard(FakeToolkit(client), ME, enabled=True)
    client.create_comment(post_id="p1", body="threaded", parent_id="c9")
    assert len(client.create_calls) == 1


def test_guard_refuses_when_the_comments_api_is_down():
    client = FakeClient(raise_on_get=True)
    install_duplicate_guard(FakeToolkit(client), ME, enabled=True)
    with pytest.raises(DuplicateTopLevelRefused):
        client.create_comment(post_id="p1", body="hi")
    assert client.create_calls == []


def test_positional_post_id_is_still_guarded():
    """A caller passing post_id positionally must not slip past."""
    client = FakeClient(comments=[_c("mine")])
    install_duplicate_guard(FakeToolkit(client), ME, enabled=True)
    with pytest.raises(DuplicateTopLevelRefused):
        client.create_comment("p1", body="dupe")
    assert client.create_calls == []


def test_not_installed_without_an_identity():
    """A guard that cannot tell my comments from yours is not a guard."""
    client = FakeClient(comments=[_c("mine")])
    assert install_duplicate_guard(FakeToolkit(client), "", enabled=True) is False
    client.create_comment(post_id="p1", body="x")  # unwrapped, goes through
    assert len(client.create_calls) == 1


def test_env_flag_can_disable_it(monkeypatch):
    monkeypatch.setenv(ENV_FLAG, "false")
    client = FakeClient(comments=[_c("mine")])
    assert install_duplicate_guard(FakeToolkit(client), ME) is False
    client.create_comment(post_id="p1", body="dupe")
    assert len(client.create_calls) == 1


def test_env_flag_defaults_on(monkeypatch):
    monkeypatch.delenv(ENV_FLAG, raising=False)
    client = FakeClient(comments=[_c("mine")])
    assert install_duplicate_guard(FakeToolkit(client), ME) is True


# --- ordering ----------------------------------------------------------------


def test_guard_is_outermost_so_a_refusal_costs_no_inner_call():
    """Refused creates must not reach the cognition wrapper, or the network.

    Mirrors the real install order: cognition handler first, guard second, so
    the guard ends up outside it.
    """
    client = FakeClient(comments=[_c("mine")])
    inner_ran: list[bool] = []
    orig = client.create_comment

    def cognition_like(*a, **k):        # stands in for _install_cognition_handler
        inner_ran.append(True)
        return orig(*a, **k)

    client.create_comment = cognition_like
    install_duplicate_guard(FakeToolkit(client), ME, enabled=True)

    with pytest.raises(DuplicateTopLevelRefused):
        client.create_comment(post_id="p1", body="dupe")
    assert inner_ran == []
    assert client.create_calls == []


# --- control: is the guard what stops it? ------------------------------------


def test_guard_is_load_bearing():
    """Same client, same duplicate, guard off — it goes straight through.

    Without this, every assertion above could pass against a client that never
    would have posted anyway, and the guard could be doing nothing.
    """
    unguarded = FakeClient(comments=[_c("mine")])
    unguarded.create_comment(post_id="p1", body="dupe")
    assert len(unguarded.create_calls) == 1, "control: the duplicate posts when unguarded"

    guarded = FakeClient(comments=[_c("mine")])
    install_duplicate_guard(FakeToolkit(guarded), ME, enabled=True)
    with pytest.raises(DuplicateTopLevelRefused):
        guarded.create_comment(post_id="p1", body="dupe")
    assert guarded.create_calls == [], "the guard is what changed the outcome"
