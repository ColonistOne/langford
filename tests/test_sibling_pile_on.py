"""Tests for ``sibling_pile_on`` — the engage-loop pile-on throttle.

The filter has three input dimensions:

- ``sibling_ids`` (set): peer dogfood agent IDs to throttle.
- ``post.author.id``: whether the candidate is sibling-authored.
- ``comments`` + ``threshold``: how many sibling commenters before we bow out.

Behaviour matrix below covers the cross-product (empty set, non-sibling
author, sibling author with 0/1/N sibling commenters, threshold edge
cases).
"""

from __future__ import annotations

from langford.__main__ import sibling_pile_on


SIBLINGS = {"sib-1", "sib-2", "sib-3"}


def _post(author_id: str) -> dict:
    return {"id": "p1", "author": {"id": author_id, "username": author_id}}


def _comment(author_id: str) -> dict:
    return {"id": f"c-{author_id}", "author": {"id": author_id, "username": author_id}}


def test_no_throttle_when_sibling_set_empty():
    # Empty sibling_ids → filter is a no-op regardless of inputs.
    assert sibling_pile_on(_post("sib-1"), [_comment("sib-2")], set(), 1) is False


def test_no_throttle_when_post_is_not_sibling_authored():
    post = _post("human-1")
    comments = [_comment("sib-1"), _comment("sib-2")]  # plenty of siblings commenting
    assert sibling_pile_on(post, comments, SIBLINGS, 1) is False


def test_first_sibling_engages_at_threshold_1():
    # Sibling post, no prior sibling commenters → engage allowed.
    assert sibling_pile_on(_post("sib-1"), [], SIBLINGS, 1) is False


def test_second_sibling_bows_out_at_threshold_1():
    # Sibling post, one prior sibling commenter → already at threshold; skip.
    assert sibling_pile_on(_post("sib-1"), [_comment("sib-2")], SIBLINGS, 1) is True


def test_threshold_2_allows_two_before_bowing():
    post = _post("sib-1")
    # 1 sibling commenter, threshold=2 → allowed.
    assert sibling_pile_on(post, [_comment("sib-2")], SIBLINGS, 2) is False
    # 2 sibling commenters, threshold=2 → skip.
    assert sibling_pile_on(
        post, [_comment("sib-2"), _comment("sib-3")], SIBLINGS, 2
    ) is True


def test_threshold_zero_means_hard_exclude():
    # threshold=0 reduces to "skip any sibling-authored post".
    assert sibling_pile_on(_post("sib-1"), [], SIBLINGS, 0) is True


def test_human_comments_do_not_count_toward_threshold():
    # Sibling post, threshold=1, but the only commenter is a human →
    # still allowed (sibling-commenter count is 0).
    post = _post("sib-1")
    comments = [_comment("human-1"), _comment("human-2"), _comment("human-3")]
    assert sibling_pile_on(post, comments, SIBLINGS, 1) is False


def test_missing_author_fields_are_tolerated():
    # Defensive: malformed post / comment objects must not crash.
    assert sibling_pile_on({}, [{}], SIBLINGS, 1) is False
    assert (
        sibling_pile_on(
            {"author": None}, [{"author": None}], SIBLINGS, 1
        )
        is False
    )


def test_negative_threshold_treated_as_hard_exclude():
    # threshold < 0 also means "skip sibling posts" (defensive — no caller
    # should pass this, but env-var parsing could).
    assert sibling_pile_on(_post("sib-2"), [], SIBLINGS, -5) is True
