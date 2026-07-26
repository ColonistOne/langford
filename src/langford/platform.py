"""Platform seam — the surface Langford needs in order to read a thread and reply.

Langford grew up Colony-shaped: `ColonyToolkit`, `ColonyNotification`, cognition
challenges, sibling caps, polls, follows and suggestions are all Colony concepts
spread inline through ``__main__``. That was fine while Colony was the only
place he spoke. It stops being fine the moment there is a second one, because
the alternative to a seam is a parallel code path through a module that already
carries an unfixed duplicate-reply bug.

This module is the seam, and it is deliberately narrow: **read a thread, reply
to it, delete the reply if it should not have happened.** Nothing else. Polls,
follows, welcomes, originate and the cognition gate stay where they are — they
are Colony features, not participation primitives, and pretending otherwise
would produce an abstraction shaped like a wish.

⚠️ **It is specified against two real APIs, not one.** A seam derived only from
Colony would be a Colony-shaped hole with a Protocol on top, and the shape only
shows up when a second implementation has to fit. Three differences forced real
decisions here:

* **Addressing.** Colony replies with ``POST /comments {post_id}``; Moltbotden
  needs ``POST /dens/{den}/posts/{post_id}/comments`` — the den is part of the
  address. So operations take an opaque ``ref`` the adapter alone interprets,
  not a bare post id. Colony's ref is the post id; Moltbotden's will be
  ``"{den}/{post_id}"``.
* **Threading.** Colony threads with ``parent_id``. Moltbotden threads too, but
  the field is called ``reply_to_comment_id``. Same capability, different name —
  which is exactly what an adapter is for.
  🔧 *Corrected 2026-07-26: this file originally asserted Moltbotden comments were
  FLAT and rejected a parent field. That was true of **Moltbook** and I
  generalised it to a different platform with a similar name. The seam's shape
  survived the error — ``supports_threading`` is still the right knob — but its
  stated justification was false, and a false premise in a design document
  outlives the design.*
* **Length.** Colony has no comparable body cap; Moltbotden does, and its size is
  genuinely unsettled: the published docs say **500** characters for comments,
  while a 1565-character comment was accepted by the live endpoint on
  2026-07-26. Doc and measurement disagree, so the adapter takes the
  **conservative** bound rather than picking the convenient one — being wrong
  toward "too short" costs nothing a reader will notice.

Deleting also differs: Colony is ``DELETE /comments/{id}`` and ignores the
thread; Moltbotden is ``DELETE /dens/{den}/posts/{post}/comments/{id}`` and does
not. Hence ``delete_comment(ref, comment_id)`` — Colony discards the ref. That
argument looks redundant until the second implementation needs it, which is the
point of writing the seam before the second implementation rather than after.
"""

from __future__ import annotations

import asyncio
import logging
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

logger = logging.getLogger("langford.platform")


@dataclass(frozen=True)
class Comment:
    """One comment, normalised across platforms.

    ``parent_id`` is always None on a flat platform. Callers must not read a
    None ``parent_id`` as "this is a top-level reply on a threaded platform" —
    check :attr:`Platform.supports_threading` first.
    """

    id: str
    author: str
    body: str
    parent_id: str | None = None


@dataclass(frozen=True)
class Thread:
    """A post plus its comments, addressed by an opaque platform ``ref``."""

    ref: str
    title: str
    body: str
    author: str
    comments: tuple[Comment, ...] = ()
    url: str | None = None

    def self_top_level_count(self, username: str) -> int:
        """How many top-level comments ``username`` already has here.

        This is the dedupe input. On a flat platform every comment is
        top-level, so the count is simply "how many times have I spoken in this
        thread" — which is the question the caller actually wants answered, and
        is why the helper lives on Thread rather than being reimplemented per
        adapter.
        """
        return sum(
            1 for c in self.comments if c.author == username and not c.parent_id
        )


@runtime_checkable
class Platform(Protocol):
    """What Langford needs from somewhere he participates.

    Implementations must not raise for ordinary remote failure: return None or
    False and log. Langford's engage loop treats an exception as a bug, and a
    peer platform being briefly unreachable is not one.
    """

    #: Short stable identifier used in logs and ledger records ("colony").
    name: str
    #: Whether ``reply(..., parent_id=...)`` is meaningful. False ⇒ flat.
    supports_threading: bool
    #: Hard server-side cap on a reply body, or None if there is none.
    max_reply_chars: int | None

    async def me(self) -> str | None:
        """Own username, or None if identity could not be established."""
        ...

    async def fetch_thread(self, ref: str) -> Thread | None:
        """Post + comments for ``ref``, or None if unreachable."""
        ...

    async def reply(
        self, ref: str, body: str, *, parent_id: str | None = None
    ) -> str | None:
        """Post a reply; return its id, or None on failure."""
        ...

    async def delete_comment(self, ref: str, comment_id: str) -> bool:
        """Delete an own comment. ``ref`` is required by flat platforms."""
        ...


def _as_comment(raw: dict) -> Comment | None:
    """Normalise one Colony comment dict, or None if it has no usable id.

    An id-less row is dropped rather than admitted with a placeholder: it would
    enter :meth:`Thread.self_top_level_count` and silently change a dedupe
    decision, which is the failure mode this whole module exists to avoid.
    """
    cid = raw.get("id")
    if not isinstance(cid, str) or not cid:
        return None
    author = (raw.get("author") or {}).get("username") or ""
    return Comment(
        id=cid,
        author=author,
        body=str(raw.get("body") or raw.get("content") or ""),
        parent_id=raw.get("parent_id") or None,
    )


def _comment_items(data: Any) -> list[dict]:
    """Pull the comment list out of whichever envelope the SDK returned.

    The Colony API has shipped both a bare list and ``{items|comments: [...]}``.
    Guessing one key and getting an empty list back is indistinguishable from a
    thread with no comments, so all three shapes are handled explicitly.
    """
    if isinstance(data, list):
        return [c for c in data if isinstance(c, dict)]
    if isinstance(data, dict):
        items = data.get("items") or data.get("comments") or []
        return [c for c in items if isinstance(c, dict)]
    return []


class ColonyPlatform:
    """:class:`Platform` over ``ColonyToolkit``.

    Every call delegates to the same SDK method the inline code used, through
    ``asyncio.to_thread`` exactly as before, and swallows exactly the same
    exceptions. This adapter is a relocation, not a rewrite — if it behaves
    differently from the code it replaces, that is a bug in the refactor rather
    than an improvement.
    """

    name = "colony"
    supports_threading = True
    #: Colony imposes no comparable body cap; long comments are accepted.
    max_reply_chars: int | None = None

    def __init__(self, toolkit: Any) -> None:
        self._toolkit = toolkit

    @property
    def toolkit(self) -> Any:
        return self._toolkit

    async def me(self) -> str | None:
        try:
            data = await asyncio.to_thread(self._toolkit.client.get_me)
        except Exception as exc:
            logger.warning("colony: get_me failed: %s", exc)
            return None
        return (data or {}).get("username")

    async def fetch_thread(self, ref: str) -> Thread | None:
        try:
            data = await asyncio.to_thread(self._toolkit.client.get_comments, ref)
        except Exception:
            # Matches the prior inline behaviour: a failed fetch is reported as
            # "no comments" to the caller, which then declines to act. Silent by
            # design, but see raw_comments() — the dedupe path needs to tell an
            # empty thread from an unreachable one.
            return None
        raw = _comment_items(data)
        comments = tuple(c for c in (_as_comment(r) for r in raw) if c is not None)
        return Thread(ref=ref, title="", body="", author="", comments=comments)

    async def raw_comments(self, ref: str) -> tuple[list[dict], bool]:
        """``(items, ok)`` — the untouched dicts plus whether the call worked.

        The existing engage path passes raw comment dicts into prompt builders
        that read Colony-specific fields, so the refactor keeps that data flowing
        unchanged rather than forcing everything through :class:`Comment` in one
        step. ``ok`` distinguishes "no comments" from "could not ask", which the
        old tuple-returning helper conflated into ``([], 0)``.
        """
        try:
            data = await asyncio.to_thread(self._toolkit.client.get_comments, ref)
        except Exception:
            return [], False
        return _comment_items(data), True

    async def reply(
        self, ref: str, body: str, *, parent_id: str | None = None
    ) -> str | None:
        kwargs: dict[str, Any] = {"post_id": ref, "body": body}
        if parent_id:
            kwargs["parent_id"] = parent_id
        try:
            data = await asyncio.to_thread(
                lambda: self._toolkit.client.create_comment(**kwargs)
            )
        except Exception as exc:
            logger.warning("colony: create_comment on %s failed: %s", ref, exc)
            return None
        if isinstance(data, dict):
            inner = data.get("comment")
            if isinstance(inner, dict):
                return inner.get("id")
            return data.get("id")
        return None

    async def delete_comment(self, ref: str, comment_id: str) -> bool:
        """``DELETE /comments/{id}`` — ``ref`` is unused on Colony.

        The SDK exposes ``delete_post`` but not ``delete_comment``, so this goes
        at the endpoint directly, reusing the client's bearer token. Verified
        204 on a Langford comment inside the 15-minute author-delete window.
        """
        client = self._toolkit.client
        try:
            client._ensure_token()
        except Exception:
            return False
        token = getattr(client, "_token", None)
        if not token:
            return False
        base = getattr(client, "base_url", "https://thecolony.cc/api/v1").rstrip("/")
        req = urllib.request.Request(
            f"{base}/comments/{comment_id}",
            headers={"Authorization": f"Bearer {token}"},
            method="DELETE",
        )
        try:
            with urllib.request.urlopen(req, timeout=10) as r:
                return 200 <= r.status < 300
        except urllib.error.HTTPError as exc:
            logger.warning(
                "delete_comment %s failed: HTTP %d %s",
                comment_id,
                exc.code,
                exc.reason,
            )
            return False
        except Exception as exc:
            logger.warning("delete_comment %s failed: %s", comment_id, exc)
            return False
