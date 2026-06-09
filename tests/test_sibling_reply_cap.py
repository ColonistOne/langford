"""Tests for ``sibling_reply_cap_hit`` — the dispatch-path throttle that
breaks sibling↔sibling notification ping-pong (observed 2026-06-08: an
18-deep smolag↔eliza-gemma chain on post 81779aa1). Cross-stack
equivalence is the point: the same pure function, same defaults, same
edge cases across the four dogfood agents.
"""

from __future__ import annotations

from langford.__main__ import sibling_reply_cap_hit

SIB = frozenset({"eliza-gemma", "smolag", "langford", "dantic"}) - {"langford"}
ME = "langford"


def _c(cid: str, author, parent=None) -> dict:
    return {"id": cid, "parent_id": parent, "author": {"username": author}}


def test_no_comment_id_returns_false():
    assert sibling_reply_cap_hit([], None, ME, sibling_usernames=SIB, cap=1) is False


def test_empty_sibling_set_returns_false():
    cs = [_c("x", "eliza-gemma")]
    assert (
        sibling_reply_cap_hit(cs, "x", ME, sibling_usernames=frozenset(), cap=1)
        is False
    )


def test_non_sibling_sender_never_capped():
    cs = [_c("x", "some-human"), _c("a", ME), _c("b", ME)]
    assert sibling_reply_cap_hit(cs, "x", ME, sibling_usernames=SIB, cap=1) is False


def test_sibling_sender_first_reply_allowed():
    cs = [_c("x", "eliza-gemma")]  # self has 0 comments on the post
    assert sibling_reply_cap_hit(cs, "x", ME, sibling_usernames=SIB, cap=1) is False


def test_sibling_sender_capped_after_one():
    cs = [_c("x", "eliza-gemma"), _c("a", ME)]  # self already commented once
    assert sibling_reply_cap_hit(cs, "x", ME, sibling_usernames=SIB, cap=1) is True


def test_cap_zero_never_replies_to_sibling():
    cs = [_c("x", "eliza-gemma")]
    assert sibling_reply_cap_hit(cs, "x", ME, sibling_usernames=SIB, cap=0) is True


def test_self_sender_excluded():
    cs = [_c("x", ME), _c("a", ME)]
    assert (
        sibling_reply_cap_hit(cs, "x", ME, sibling_usernames=SIB | {ME}, cap=1)
        is False
    )


def test_unknown_comment_id_sender_none():
    cs = [_c("a", ME)]
    assert sibling_reply_cap_hit(cs, "missing", ME, sibling_usernames=SIB, cap=1) is False


def test_negative_cap_disables_guard():
    cs = [_c("x", "eliza-gemma"), _c("a", ME)]
    assert sibling_reply_cap_hit(cs, "x", ME, sibling_usernames=SIB, cap=-1) is False


def test_cap_two_allows_two_then_blocks():
    one = [_c("x", "eliza-gemma"), _c("a", ME)]
    assert sibling_reply_cap_hit(one, "x", ME, sibling_usernames=SIB, cap=2) is False
    two = [_c("x", "eliza-gemma"), _c("a", ME), _c("b", ME)]
    assert sibling_reply_cap_hit(two, "x", ME, sibling_usernames=SIB, cap=2) is True


def test_missing_author_dict_is_safe():
    cs = [{"id": "x"}, {"id": "a", "author": None}]
    assert sibling_reply_cap_hit(cs, "x", ME, sibling_usernames=SIB, cap=1) is False
