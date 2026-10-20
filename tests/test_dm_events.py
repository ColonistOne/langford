"""DM notifications: sender from ``actor``, body by ``message_id``, no guessed recipients, no reply per flood copy.

The 2026-09-03..10-20 loop: a DM notification far (in time) from its
conversation's latest message matched nothing, arrived with no sender, and the
model replied to the display name "Magician" -> 404, again and again; and each
of magician-local's 120+ identical copies got its own run. See dm_events.
"""

from __future__ import annotations

import asyncio
import time

import pytest
from langchain_colony import ColonyEventPoller, ColonyNotification

from langford import dm_events as dm

dm.install_actor_capture()

ACTOR = {"id": "u-mag", "username": "magician-local", "display_name": "Magician", "user_type": "agent"}


def raw_dm(nid: str, message_id: str | None = "m2", actor: dict | None = ACTOR, created: str = "2026-10-20T02:15:00Z") -> dict:
    d = {"id": nid, "notification_type": "direct_message", "message": "Magician: SAP P0 note...",
         "conversation_id": "c1", "message_id": message_id, "created_at": created, "is_read": False}
    if actor is not None:
        d["actor"] = actor
    return d


# The conversation's latest message is 3 h after the notification: the library's 5-minute time match fails.
CONVERSATIONS = {"items": [{"id": "c1", "other_user": ACTOR, "last_message_at": "2026-10-20T05:15:00Z",
                            "last_message_preview": "preview of the newest message"}]}
THREAD = {"id": "c1", "other_user": ACTOR, "messages": [
    {"id": "m1", "body": "an older message", "sender": {"id": "u-mag"}, "created_at": "2026-10-20T01:00:00Z"},
    {"id": "m2", "body": "the SAP P0 note, in full", "sender": {"id": "u-mag"}, "created_at": "2026-10-20T02:14:59Z"},
    {"id": "m3", "body": "the newest message", "sender": {"id": "u-mag"}, "created_at": "2026-10-20T05:15:00Z"},
]}


class FakeClient:
    def __init__(self, notifications: list[dict] | None = None) -> None:
        self.notifications = notifications or []
        self.conversation_calls: list[str] = []

    def get_notifications(self, unread_only: bool = True):
        return {"items": self.notifications}

    def list_conversations(self):
        return CONVERSATIONS

    def get_conversation(self, username: str):
        self.conversation_calls.append(username)
        if username != "magician-local":
            raise RuntimeError("404 Not Found")
        return THREAD

    def mark_notifications_read(self):
        return {}


def test_actor_fills_the_sender_at_parse_time() -> None:
    n = ColonyNotification.from_api(raw_dm("n1"))
    assert n.sender_username == "magician-local"
    assert n.sender_display_name == "Magician"
    assert dm._DM_FIELDS["n1"]["message_id"] == "m2"


def test_non_dm_notifications_are_untouched() -> None:
    n = ColonyNotification.from_api({"id": "n9", "notification_type": "mention", "message": "x", "actor": ACTOR})
    assert n.sender_username is None and "n9" not in dm._DM_FIELDS


def test_install_is_idempotent() -> None:
    dm.install_actor_capture()
    dm.install_actor_capture()
    assert ColonyNotification.from_api(raw_dm("n2")).sender_username == "magician-local"


def _enrich(poller_cls, notif: ColonyNotification, client: FakeClient) -> ColonyNotification:
    p = poller_cls(client=client)
    asyncio.run(p._populate_dm_async(notif, client.list_conversations()))
    return notif


def test_body_is_the_notified_message_not_the_newest() -> None:
    n = _enrich(dm.ActorAwarePoller, ColonyNotification.from_api(raw_dm("n3")), FakeClient())
    assert n.body == "the SAP P0 note, in full" and n.body_truncated is False
    assert n.sender_username == "magician-local"


def test_control_the_library_alone_loses_the_body_on_this_input() -> None:
    # Same notification, plain library poller, and a sender-less parse (what 0.17.0 sees): nothing matches.
    plain = ColonyNotification(id="n4", notification_type="direct_message", message="Magician: SAP P0 note...",
                               created_at="2026-10-20T02:15:00Z")
    _enrich(ColonyEventPoller, plain, FakeClient())
    assert plain.sender_username is None and plain.body is None


def test_without_message_id_the_time_match_cannot_overwrite_the_actor() -> None:
    other = {"id": "u-x", "username": "someone-else", "display_name": "X", "user_type": "agent"}
    client = FakeClient()
    convs = {"items": [{"id": "c9", "other_user": other, "last_message_at": "2026-10-20T02:15:01Z",
                        "last_message_preview": "x"}]}
    n = ColonyNotification.from_api(raw_dm("n5", message_id=None))
    p = dm.ActorAwarePoller(client=client)
    asyncio.run(p._populate_dm_async(n, convs))
    assert n.sender_username == "magician-local"


def test_sync_path_matches_the_async_one() -> None:
    n = ColonyNotification.from_api(raw_dm("n6"))
    dm.ActorAwarePoller(client=FakeClient())._populate_dm(n, CONVERSATIONS)
    assert n.body == "the SAP P0 note, in full"


def test_refusal_without_a_sender(tmp_path) -> None:
    n = ColonyNotification(id="n7", notification_type="direct_message", message="Magician: hi")
    assert "no sender username" in dm.dm_dispatch_refusal(n, dm.HandledDms(tmp_path / "h.txt"))


def test_identical_repeat_is_refused_and_distinct_text_is_not(tmp_path) -> None:
    h = dm.HandledDms(tmp_path / "h.txt")
    mk = lambda nid, body: ColonyNotification(id=nid, notification_type="direct_message", sender_username="magician-local", body=body)
    assert dm.dm_dispatch_refusal(mk("a", "same note"), h) is None
    assert "already handled" in dm.dm_dispatch_refusal(mk("b", "same note"), h)
    assert dm.dm_dispatch_refusal(mk("c", "a different note"), h) is None


def test_same_text_from_another_sender_is_dispatched(tmp_path) -> None:
    h = dm.HandledDms(tmp_path / "h.txt")
    a = ColonyNotification(id="a", notification_type="direct_message", sender_username="magician-local", body="hello")
    b = ColonyNotification(id="b", notification_type="direct_message", sender_username="someone-else", body="hello")
    assert dm.dm_dispatch_refusal(a, h) is None and dm.dm_dispatch_refusal(b, h) is None


def test_handled_survives_a_restart_and_expires(tmp_path) -> None:
    f = tmp_path / "h.txt"
    dm.HandledDms(f).record("magician-local", "note")
    assert dm.HandledDms(f).seen("magician-local", "note")
    key = dm.HandledDms._key("magician-local", "old")
    f.write_text(f.read_text() + f"{time.time() - 8 * 86400:.0f}\t{key[0]}\t{key[1]}\n")
    assert not dm.HandledDms(f).seen("magician-local", "old")


def test_non_dm_is_never_refused(tmp_path) -> None:
    n = ColonyNotification(id="m", notification_type="mention", message="x")
    assert dm.dm_dispatch_refusal(n, dm.HandledDms(tmp_path / "h.txt")) is None


def test_a_flood_poll_dispatches_once(tmp_path) -> None:
    flood = [raw_dm(f"f{i}") for i in range(5)] + [raw_dm("nosender", actor=None, message_id=None)]
    client = FakeClient(flood)
    poller = dm.ActorAwarePoller(client=client, mark_read=True)
    handled = dm.HandledDms(tmp_path / "h.txt")
    dispatched: list[str] = []

    @poller.on()
    async def on_event(notif: ColonyNotification) -> None:
        if dm.dm_dispatch_refusal(notif, handled) is None:
            dispatched.append(notif.id)

    asyncio.run(poller.poll_once_async())
    assert dispatched == ["f0"]


def test_handle_event_never_runs_the_model_for_a_senderless_dm(tmp_path) -> None:
    import langford.__main__ as lm

    reached: list[str] = []

    class RecordingAgent:  # _handle_event catches exceptions, so record instead of raising
        def __getattr__(self, name):
            reached.append(name)
            raise AttributeError(name)

    n = ColonyNotification(id="z", notification_type="direct_message", message="Magician: hi")
    asyncio.run(lm._handle_event(RecordingAgent(), n, handled_dms=dm.HandledDms(tmp_path / "h.txt")))
    assert reached == [], reached


def test_handle_event_skips_a_repeat(tmp_path) -> None:
    import langford.__main__ as lm

    reached: list[str] = []

    class RecordingAgent:  # _handle_event catches exceptions, so record instead of raising
        def __getattr__(self, name):
            reached.append(name)
            raise AttributeError(name)

    h = dm.HandledDms(tmp_path / "h.txt")
    h.record("magician-local", "same note")
    n = ColonyNotification(id="r", notification_type="direct_message", sender_username="magician-local", body="same note")
    asyncio.run(lm._handle_event(RecordingAgent(), n, handled_dms=h))
    assert reached == [], reached
