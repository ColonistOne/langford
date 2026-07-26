"""Refuse duplicate top-level comments at the tool boundary, before they exist.

Langford has produced duplicate top-level replies since 2026-05-02. Three
mitigations were layered on and none of them fixed it, because none of them
addressed generation:

1. **Prompt hardening** (v0.6.2) — a directive telling the model that a second
   top-level comment on a post is a duplicate. An imperative addressed to the
   thing that is already getting it wrong.
2. **A post-dispatch validator** — notices the duplicate *after* dispatch.
3. **`_delete_comment_via_api`** — deletes it, inside a 15-minute author-delete
   window.

That pipeline works, and it is the wrong shape. The duplicate is really posted:
it exists publicly for as long as the round trip takes, it consumes the delete
window, it depends on the supervisor not swapping Langford out between dispatch
and validation, and — worst — **the delete counter reads as success**. A rising
number of cleanly-deleted duplicates looks like a working safety net and is
actually a measure of how often generation is still wrong.

This module removes the sentence instead of catching it. `create_comment` is the
single boundary every agent-authored comment passes through (the langchain tools
resolve it on the client at call time, which is why the cognition handler hooks
the same place). A top-level create onto a post where Langford already has one is
**refused there**, and the model is told what to do instead. Nothing reaches the
network, so there is nothing to detect, delete, or race.

**It fails CLOSED.** If the comment list cannot be fetched, the guard refuses
rather than allowing. That is deliberate and it is the more interesting half:
the old dedupe path reported a failed fetch as ``([], 0)`` — no comments, hence
no duplicate, hence go ahead — so an outage in the *checking* API actively caused
the bad behaviour it was meant to prevent. Refusing on unknown costs a missed
comment during an outage; allowing on unknown costs a duplicate exactly when
Langford is least able to notice.
"""

from __future__ import annotations

import functools
import logging
import os
from typing import Any

logger = logging.getLogger("langford.dedupe")

#: Set to "false" to disable the guard without editing code. Default on.
ENV_FLAG = "LANGFORD_DUPLICATE_GUARD"


class DuplicateTopLevelRefused(RuntimeError):
    """Raised in place of creating a duplicate top-level comment.

    Raised rather than returned so the agent framework surfaces it to the model
    as a tool error: the model gets told *why* and can reply nested instead. A
    silently-swallowed refusal would leave the model believing it had spoken,
    which is a different bug wearing this one's clothes.
    """


def self_top_level_ids(comments: list[dict], self_username: str) -> list[str]:
    """Ids of ``self_username``'s existing TOP-LEVEL comments in ``comments``."""
    out = []
    for c in comments:
        if not isinstance(c, dict):
            continue
        if (c.get("author") or {}).get("username") != self_username:
            continue
        if c.get("parent_id"):
            continue
        cid = c.get("id")
        if isinstance(cid, str) and cid:
            out.append(cid)
    return out


def refusal_reason(
    *,
    self_username: str,
    parent_id: str | None,
    comments: list[dict],
    fetch_ok: bool,
) -> str | None:
    """Why this create must be refused, or None to allow it.

    Pure and total: every branch is decided by the arguments, so the policy can
    be tested without a network, a client, or an LLM.
    """
    if parent_id:
        # A nested reply is never the duplicate this guard exists for. Paired
        # nested duplicates under one parent are a different defect, still
        # handled by the post-dispatch validator.
        return None
    if not fetch_ok:
        return (
            "could not verify whether you already have a top-level comment on this "
            "post (the comment list failed to load), and this guard fails closed"
        )
    existing = self_top_level_ids(comments, self_username)
    if existing:
        return (
            f"you already have a top-level comment on this post ({existing[0]}). "
            f"A second one is a duplicate. Reply to a specific comment by passing "
            f"parent_id, or say nothing"
        )
    return None


def install_duplicate_guard(
    toolkit: Any, self_username: str, *, enabled: bool | None = None
) -> bool:
    """Wrap ``client.create_comment`` to refuse duplicate top-level creates.

    Install AFTER :func:`_install_cognition_handler` so this guard is the
    outermost wrapper: a refused create should never reach the network, and
    therefore never produce a cognition challenge to solve.

    Returns whether the guard was installed.
    """
    if enabled is None:
        enabled = os.environ.get(ENV_FLAG, "true").strip().lower() != "false"
    if not enabled:
        logger.warning(
            "duplicate-guard DISABLED via %s — duplicate top-level comments will "
            "again be caught only after dispatch, by deletion",
            ENV_FLAG,
        )
        return False
    if not self_username:
        # Without an identity the guard cannot tell own comments from anyone
        # else's, and a guard that cannot distinguish is not a guard.
        logger.error("duplicate-guard NOT installed: self_username is empty")
        return False

    client = toolkit.client
    orig = client.create_comment

    @functools.wraps(orig)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        post_id = kwargs.get("post_id") or (args[0] if args else None)
        parent_id = kwargs.get("parent_id")
        if not post_id:
            return orig(*args, **kwargs)

        comments: list[dict] = []
        fetch_ok = True
        try:
            data = client.get_comments(post_id)
            if isinstance(data, list):
                comments = [c for c in data if isinstance(c, dict)]
            elif isinstance(data, dict):
                items = data.get("items") or data.get("comments") or []
                comments = [c for c in items if isinstance(c, dict)]
        except Exception as exc:
            fetch_ok = False
            logger.warning("duplicate-guard: get_comments(%s) failed: %s", post_id, exc)

        reason = refusal_reason(
            self_username=self_username,
            parent_id=parent_id,
            comments=comments,
            fetch_ok=fetch_ok,
        )
        if reason is not None:
            logger.warning(
                "duplicate-guard REFUSED top-level create on %s: %s", post_id, reason
            )
            raise DuplicateTopLevelRefused(
                f"Refusing to post this comment: {reason}."
            )
        return orig(*args, **kwargs)

    client.create_comment = wrapper
    logger.info(
        "duplicate-guard installed for @%s (fails closed on unverifiable state)",
        self_username,
    )
    return True
