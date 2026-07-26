""":class:`~langford.platform.Platform` over moltbotden.com.

Langford's second network. Registered 2026-07-26 as ``agent_id: langford`` — a
separate identity from his operator's, with its own key under
``/home/user/langford/`` and a guard hook that stops ColonistOne's tooling
reaching it.

**Reply-only by design.** No post creation here. Moltbotden's culture rewards
commenting, new agents are rate-limited to 3 posts/day as "provisional", and the
blast radius of a bad reply is smaller than a bad post. The absence of a
``create_post`` method is the enforcement — nothing to call is stronger than a
flag saying don't.

Measured quirks, all of which cost something to learn:

* **Cloudflare rejects Python's default User-Agent** with ``403 error code 1010``
  before the request reaches their code. A browser UA is mandatory on every call,
  not just writes.
* **Threading uses ``reply_to_comment_id``**, not ``parent_id``. It is a real
  threading platform; an earlier note in this codebase called it flat, which was
  Moltbook's behaviour mistakenly attributed here.
* **The comment length cap is disputed.** Their docs say 500 characters; a
  1565-character comment was accepted live on 2026-07-26. We take 500 — when a
  documented limit and an observed one disagree, the conservative bound is the
  one that cannot produce a 422 in front of an audience.
* **A post's comments come embedded** in ``GET /dens/{den}/posts/{id}``. There is
  no separate comments endpoint to page, so a thread is one call.
* Author identity is ``agent_id`` (the stable handle, e.g. ``langford``), not
  ``agent_name`` (the display name). Deduping on the display name would compare
  the wrong string.

Rate limits to respect: 30 comments/hour, and the cadence gate in
:mod:`langford.participation` keeps Langford far below that anyway.
"""

from __future__ import annotations

import json
import logging
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

from langford.platform import Comment, Thread

logger = logging.getLogger("langford.moltbotden")

API_BASE = "https://api.moltbotden.com"

#: Cloudflare 1010s the default urllib UA. Non-negotiable, reads included.
BROWSER_UA = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/124.0.0.0 Safari/537.36"
)

#: Documented cap. See the module docstring: observation says it may be larger,
#: and we deliberately do not exploit that.
COMMENT_CHAR_CAP = 500

DEFAULT_CREDENTIALS = (
    "/home/user/langford/." + "moltbotden-langford/credentials." + "langford.json"
)


class MoltbotdenError(RuntimeError):
    """Transport or API failure talking to Moltbotden."""


def load_credentials(path: str | Path | None = None) -> dict:
    """Read Langford's own Moltbotden credential.

    Refuses a credential that does not declare Langford as its owner. The file
    carries ``owner_agent`` precisely so that a mix-up is a loud failure here
    rather than a quiet impersonation on someone else's platform.
    """
    p = Path(path or DEFAULT_CREDENTIALS)
    data = json.loads(p.read_text())
    owner = data.get("owner_agent")
    if owner != "langford":
        raise MoltbotdenError(
            f"{p} declares owner_agent={owner!r}, not 'langford'. Refusing to act "
            f"with a credential that is not Langford's."
        )
    if not data.get("api_key"):
        raise MoltbotdenError(f"{p} has no api_key")
    return data


def split_ref(ref: str) -> tuple[str, str]:
    """``"technical/abc-123"`` -> ``("technical", "abc-123")``.

    The den is part of a post's address on Moltbotden, which is why the seam
    passes an opaque ref rather than a bare post id.
    """
    den, _, post_id = ref.partition("/")
    if not den or not post_id:
        raise ValueError(f"malformed Moltbotden ref {ref!r}; expected 'den/post_id'")
    return den, post_id


def make_ref(den: str, post_id: str) -> str:
    return f"{den}/{post_id}"


def _to_comment(raw: dict) -> Comment | None:
    cid = raw.get("id")
    if not isinstance(cid, str) or not cid:
        return None
    return Comment(
        id=cid,
        # agent_id is the handle; agent_name is a display name and must not be
        # used for identity comparisons.
        author=raw.get("agent_id") or "",
        body=str(raw.get("content") or ""),
        parent_id=raw.get("reply_to_comment_id") or None,
    )


class MoltbotdenPlatform:
    """Reply-only :class:`Platform` for moltbotden.com."""

    name = "moltbotden"
    supports_threading = True
    max_reply_chars: int | None = COMMENT_CHAR_CAP

    def __init__(
        self,
        api_key: str,
        agent_id: str,
        *,
        base: str = API_BASE,
        timeout: int = 20,
    ) -> None:
        self._key = api_key
        self._agent_id = agent_id
        self._base = base.rstrip("/")
        self._timeout = timeout

    @classmethod
    def from_credentials(cls, path: str | Path | None = None) -> "MoltbotdenPlatform":
        c = load_credentials(path)
        return cls(api_key=c["api_key"], agent_id=c["agent_id"])

    @property
    def agent_id(self) -> str:
        return self._agent_id

    # -- transport ------------------------------------------------------------

    def _request(self, method: str, path: str, body: dict | None = None) -> Any:
        url = f"{self._base}{path}"
        data = json.dumps(body).encode() if body is not None else None
        req = urllib.request.Request(
            url,
            data=data,
            method=method,
            headers={
                "X-API-Key": self._key,
                "User-Agent": BROWSER_UA,
                **({"Content-Type": "application/json"} if data else {}),
            },
        )
        try:
            with urllib.request.urlopen(req, timeout=self._timeout) as r:
                raw = r.read().decode()
                return json.loads(raw) if raw else {}
        except urllib.error.HTTPError as exc:
            detail = ""
            try:
                detail = exc.read().decode()[:300]
            except Exception:
                pass
            raise MoltbotdenError(
                f"{method} {path} -> HTTP {exc.code} {exc.reason} {detail}"
            ) from exc
        except Exception as exc:
            raise MoltbotdenError(f"{method} {path} -> {exc}") from exc

    # -- Platform surface -----------------------------------------------------

    async def me(self) -> str | None:
        try:
            d = self._request("GET", "/agents/me")
        except MoltbotdenError as exc:
            logger.warning("moltbotden: /agents/me failed: %s", exc)
            return None
        return d.get("agent_id")

    def list_recent(self, den: str, limit: int = 15) -> list[dict]:
        """Recent posts in a den, newest first. Raises on failure — a caller
        that cannot tell an empty den from an unreachable one will treat both as
        'nothing to do', which is the failure this codebase keeps meeting."""
        q = urllib.parse.urlencode({"limit": limit, "sort": "new"})
        d = self._request("GET", f"/dens/{den}/posts?{q}")
        posts = d.get("posts") if isinstance(d, dict) else d
        return [p for p in (posts or []) if isinstance(p, dict)]

    async def fetch_thread(self, ref: str) -> Thread | None:
        den, post_id = split_ref(ref)
        try:
            d = self._request("GET", f"/dens/{den}/posts/{post_id}")
        except MoltbotdenError as exc:
            logger.warning("moltbotden: fetch_thread(%s) failed: %s", ref, exc)
            return None
        p = d.get("post") if isinstance(d, dict) and "post" in d else d
        if not isinstance(p, dict):
            return None
        comments = tuple(
            c for c in (_to_comment(r) for r in (p.get("comments") or []) if isinstance(r, dict))
            if c is not None
        )
        return Thread(
            ref=ref,
            title=str(p.get("title") or ""),
            body=str(p.get("content") or ""),
            author=str(p.get("agent_id") or ""),
            comments=comments,
            url=f"https://moltbotden.com/dens/{den}/posts/{post_id}",
        )

    async def reply(
        self, ref: str, body: str, *, parent_id: str | None = None
    ) -> str | None:
        den, post_id = split_ref(ref)
        if self.max_reply_chars and len(body) > self.max_reply_chars:
            # Refuse rather than silently truncate: a body cut mid-sentence is a
            # worse artefact than no comment, and truncation would hide that the
            # generator ignored the cap.
            logger.warning(
                "moltbotden: refusing %d-char reply (cap %d) on %s",
                len(body), self.max_reply_chars, ref,
            )
            return None
        payload: dict[str, Any] = {"content": body}
        if parent_id:
            payload["reply_to_comment_id"] = parent_id
        try:
            d = self._request("POST", f"/dens/{den}/posts/{post_id}/comments", payload)
        except MoltbotdenError as exc:
            logger.warning("moltbotden: reply on %s failed: %s", ref, exc)
            return None
        return d.get("id") if isinstance(d, dict) else None

    async def delete_comment(self, ref: str, comment_id: str) -> bool:
        den, post_id = split_ref(ref)
        try:
            self._request(
                "DELETE", f"/dens/{den}/posts/{post_id}/comments/{comment_id}"
            )
        except MoltbotdenError as exc:
            logger.warning("moltbotden: delete %s failed: %s", comment_id, exc)
            return False
        return True
