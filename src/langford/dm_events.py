"""Direct-message notifications, read from the fields the API sends.

Two defects, one loop (open since 2026-09-03):

* langchain-colony 0.17.0 builds a notification from ``id``, ``type`` and
  ``message`` only, then finds the sender by matching the notification's time to
  a conversation's ``last_message_at`` within 5 minutes. When one sender floods
  a thread (magician-local: 120+ byte-identical copies of one note), the older
  notifications sit far from the conversation's latest message, match nothing,
  and arrive with no sender username and no body. The model then read the
  display name "Magician" out of the notification text and called
  ``colony_send_message("Magician")``: 404, retried, a GPU call each time.
  The API has carried the sender (``actor``) and the exact message
  (``message_id``) on every DM notification since at least 2026-10-20, so the
  time match is no longer needed.
* Every copy of the flood was a fresh notification, so every copy got a model
  run and a reply.

This module keeps each DM notification's ``actor`` and ``message_id``, fills
the sender from ``actor``, reads the body from the message with that id, and
gives the dispatcher two refusals: no sender username means no dispatch (the
model must never guess a recipient), and a byte-identical DM already handled
from the same sender within the window is skipped.
"""

from __future__ import annotations

import hashlib
import logging
import time
from pathlib import Path
from typing import Any

from langchain_colony import ColonyEventPoller, ColonyNotification
from langchain_colony._response import as_list

logger = logging.getLogger("langford")

DM_TYPES = {"direct_message", "dm"}

# notification id -> {"actor_username", "message_id"} for DM notifications.
_DM_FIELDS: dict[str, dict[str, Any]] = {}
_INSTALLED = False


def install_actor_capture() -> None:
    """Wrap ``ColonyNotification.from_api`` so DM notifications keep ``actor`` and ``message_id``.

    Idempotent. The sender fields are filled from ``actor`` at parse time, so a
    later failed enrichment can no longer leave them empty.
    """
    global _INSTALLED
    if _INSTALLED:
        return
    original = ColonyNotification.from_api.__func__  # the function under the classmethod

    def from_api(cls: type[ColonyNotification], data: dict) -> ColonyNotification:
        notif = original(cls, data)
        if notif.notification_type in DM_TYPES and isinstance(data, dict):
            raw_actor = data.get("actor")
            actor: dict[str, Any] = raw_actor if isinstance(raw_actor, dict) else {}
            if actor.get("username"):
                notif.sender_id = actor.get("id") or None
                notif.sender_username = actor["username"]
                notif.sender_display_name = actor.get("display_name") or None
                notif.sender_user_type = actor.get("user_type") or None
            _DM_FIELDS[notif.id] = {"actor_username": actor.get("username"), "message_id": data.get("message_id")}
        return notif

    ColonyNotification.from_api = classmethod(from_api)  # type: ignore[method-assign]
    _INSTALLED = True


def _body_by_id(thread: Any, message_id: str) -> str | None:
    for m in as_list(thread, "get_conversation"):
        if isinstance(m, dict) and m.get("id") == message_id and m.get("body"):
            return m["body"]
    return None


class ActorAwarePoller(ColonyEventPoller):
    """``ColonyEventPoller`` that reads a DM by its ``message_id`` from its ``actor``'s thread.

    Falls back to the library's time match when the notification lacks those
    fields, but never lets that match overwrite a sender that ``actor`` named.
    """

    def _from_fields(self, notif: ColonyNotification) -> tuple[str | None, str | None]:
        f = _DM_FIELDS.get(notif.id) or {}
        return f.get("actor_username"), f.get("message_id")

    def _keep_actor(self, notif: ColonyNotification, saved: tuple) -> None:
        if saved[1]:  # actor named a sender: it wins over any time-matched conversation
            notif.sender_id, notif.sender_username, notif.sender_display_name, notif.sender_user_type = saved

    def _populate_dm(self, notif: ColonyNotification, conversations: Any) -> None:
        username, mid = self._from_fields(notif)
        if username and mid:
            try:
                body = _body_by_id(self.client.get_conversation(username), mid)
            except Exception as exc:  # noqa: BLE001 — fall back below
                logger.warning("dm: get_conversation(%s) failed: %s", username, exc)
                body = None
            if body:
                notif.body, notif.body_truncated = body, False
                return
        saved = (notif.sender_id, notif.sender_username, notif.sender_display_name, notif.sender_user_type)
        super()._populate_dm(notif, conversations)
        self._keep_actor(notif, saved)

    async def _populate_dm_async(self, notif: ColonyNotification, conversations: Any) -> None:
        username, mid = self._from_fields(notif)
        if username and mid:
            try:
                body = _body_by_id(await self._call_async(self.client.get_conversation, username), mid)
            except Exception as exc:  # noqa: BLE001 — fall back below
                logger.warning("dm: get_conversation(%s) failed: %s", username, exc)
                body = None
            if body:
                notif.body, notif.body_truncated = body, False
                return
        saved = (notif.sender_id, notif.sender_username, notif.sender_display_name, notif.sender_user_type)
        await super()._populate_dm_async(notif, conversations)
        self._keep_actor(notif, saved)


class HandledDms:
    """Byte-identical DMs already handled, per sender, persisted one per line: ``epoch<TAB>sender<TAB>sha256``."""

    def __init__(self, path: str | Path, window_sec: int = 7 * 86400) -> None:
        self.path = Path(path)
        self.window_sec = window_sec
        self._seen: dict[tuple[str, str], float] = {}
        now = time.time()
        try:
            for line in self.path.read_text().splitlines():
                parts = line.split("\t")
                if len(parts) == 3 and now - float(parts[0]) <= window_sec:
                    self._seen[(parts[1], parts[2])] = float(parts[0])
        except (OSError, ValueError):
            pass

    @staticmethod
    def _key(sender: str, text: str) -> tuple[str, str]:
        return sender.lower(), hashlib.sha256(text.encode("utf-8")).hexdigest()

    def seen(self, sender: str, text: str) -> bool:
        ts = self._seen.get(self._key(sender, text))
        return ts is not None and time.time() - ts <= self.window_sec

    def record(self, sender: str, text: str) -> None:
        key, now = self._key(sender, text), time.time()
        self._seen[key] = now
        try:
            with self.path.open("a") as f:
                f.write(f"{now:.0f}\t{key[0]}\t{key[1]}\n")
        except OSError as exc:
            logger.warning("dm: could not persist handled DM to %s: %s", self.path, exc)


def dm_dispatch_refusal(notif: ColonyNotification, handled: HandledDms | None) -> str | None:
    """Why a DM notification must not reach the model, or None to dispatch (and record it)."""
    if notif.notification_type not in DM_TYPES:
        return None
    if not notif.sender_username:
        return f"DM notification {notif.id} has no sender username; not guessing a recipient"
    text = notif.body or notif.message or ""
    if handled is not None and text:
        if handled.seen(notif.sender_username, text):
            return f"identical DM from @{notif.sender_username} already handled"
        handled.record(notif.sender_username, text)
    return None
