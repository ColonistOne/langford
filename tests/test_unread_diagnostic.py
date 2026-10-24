"""The 2026-10-24 unread-count diagnostic: log the count around mark-all-read, and the unread breakdown at startup."""

import logging
import pathlib

from langford.dm_events import instrument_mark_read, log_unread_breakdown


class FakeClient:
    def __init__(self, count=1134, clears=True, count_raises=False, rows=None, rows_raise=False):
        self.count, self.clears, self.count_raises = count, clears, count_raises
        self.rows, self.rows_raise = rows or [], rows_raise
        self.mark_calls = 0

    def get_notification_count(self):
        if self.count_raises:
            raise RuntimeError("count endpoint down")
        return {"unread_count": self.count}

    def mark_notifications_read(self):
        self.mark_calls += 1
        if self.clears:
            self.count = 0
        return {"ok": True}

    def get_notifications(self, unread_only=False, limit=50):
        if self.rows_raise:
            raise RuntimeError("list endpoint down")
        return {"items": self.rows[:limit]}


def test_mark_logs_before_and_after_and_calls_the_original_once(caplog):
    c = FakeClient(count=1134)
    instrument_mark_read(c)
    with caplog.at_level(logging.INFO, logger="langford"):
        result = c.mark_notifications_read()
    assert c.mark_calls == 1
    assert result == {"ok": True}
    assert "mark-all-read: unread 1134 -> 0" in caplog.text
    assert "still unread" not in caplog.text


def test_a_count_that_does_not_drop_is_a_warning(caplog):
    c = FakeClient(count=1147, clears=False)
    instrument_mark_read(c)
    with caplog.at_level(logging.INFO, logger="langford"):
        c.mark_notifications_read()
    assert "unread 1147 -> 1147" in caplog.text
    assert "1147 still unread straight after mark-all-read" in caplog.text


def test_a_failing_count_never_breaks_the_mark(caplog):
    c = FakeClient(count_raises=True)
    instrument_mark_read(c)
    with caplog.at_level(logging.INFO, logger="langford"):
        c.mark_notifications_read()
    assert c.mark_calls == 1
    assert "unread None -> None" in caplog.text


def test_breakdown_counts_types_and_reports_the_total(caplog):
    rows = [
        {"notification_type": "tag_match", "created_at": "2026-10-20T01:00:00Z"},
        {"notification_type": "tag_match", "created_at": "2026-10-21T01:00:00Z"},
        {"notification_type": "mention", "created_at": "2026-10-22T01:00:00Z"},
    ]
    with caplog.at_level(logging.INFO, logger="langford"):
        log_unread_breakdown(FakeClient(count=1134, rows=rows))
    assert "unread at startup 1134" in caplog.text
    assert "tag_match 2, mention 1" in caplog.text
    assert "oldest of them 2026-10-20T01:00:00Z" in caplog.text


def test_breakdown_failure_is_logged_not_raised(caplog):
    with caplog.at_level(logging.INFO, logger="langford"):
        log_unread_breakdown(FakeClient(rows_raise=True))
    assert "unread breakdown failed" in caplog.text


def test_main_wires_both_hooks_to_the_poller_client():
    src = (pathlib.Path(__file__).resolve().parents[1] / "src" / "langford" / "__main__.py").read_text()
    after_poller = src.split("mark_read=True,\n    )", 1)[1][:400]
    assert "instrument_mark_read(poller.client)" in after_poller
    assert "log_unread_breakdown(poller.client)" in after_poller
