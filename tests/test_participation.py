"""Tests for the cadence gate, the decision ledger, and the Moltbotden adapter.

The property under test is not "Langford posts". It is **that every run leaves a
typed record**, so that at a 1–2 day cadence an observer can tell "ran and chose
silence" from "never ran". Those two produce identical output on the platform and
the ledger is the only thing that separates them — which is why
``test_every_path_records_exactly_one_row`` enumerates the paths rather than
spot-checking a couple.
"""

from __future__ import annotations

import asyncio
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from langford.moltbotden import (
    COMMENT_CHAR_CAP,
    MoltbotdenError,
    MoltbotdenPlatform,
    load_credentials,
    make_ref,
    split_ref,
)
from langford.participation import (
    COULD_NOT_REACH,
    DECLINED_ALREADY_REPLIED,
    DECLINED_CADENCE,
    DECLINED_DAILY_CAP,
    DECLINED_EMPTY_COMPOSE,
    DECLINED_NO_CANDIDATE,
    POSTED,
    REFUSED_REPETITIVE,
    REFUSED_TOO_LONG,
    REFUSED_UNGROUNDED,
    CadenceGate,
    Refusal,
    run_once,
    usable_reply,
)
from langford.platform import Comment, Platform

ME = "langford"
T0 = datetime(2026, 7, 26, 12, 0, tzinfo=timezone.utc)


def gate(tmp_path: Path, *, at=T0, **kw) -> CadenceGate:
    return CadenceGate(
        ledger_path=tmp_path / "moltbotden.jsonl", _clock=lambda: at, **kw
    )


def _run(coro):
    return asyncio.run(coro)


def rows(g: CadenceGate) -> list[dict]:
    return [json.loads(l) for l in g.ledger_path.read_text().splitlines() if l.strip()]


# --- adapter shape -----------------------------------------------------------


def test_adapter_satisfies_the_protocol():
    assert isinstance(MoltbotdenPlatform("k", ME), Platform)


def test_adapter_declares_moltbotdens_real_capabilities():
    """These contradict Colony's, which is the entire reason the seam exists.

    Threading is TRUE here: Moltbotden threads via `reply_to_comment_id`. An
    earlier note in this repo called it flat — that was Moltbook's behaviour
    attributed to the wrong platform.
    """
    p = MoltbotdenPlatform("k", ME)
    assert p.name == "moltbotden"
    assert p.supports_threading is True
    assert p.max_reply_chars == COMMENT_CHAR_CAP == 500


def test_refs_carry_the_den_because_the_address_needs_it():
    assert split_ref(make_ref("technical", "abc")) == ("technical", "abc")
    for bad in ("no-slash", "/missing-den", "den/"):
        with pytest.raises(ValueError):
            split_ref(bad)


def test_credentials_must_declare_langford_as_owner(tmp_path):
    """A mix-up should be a loud failure here, not a quiet impersonation there."""
    good = tmp_path / "ok.json"
    good.write_text(json.dumps({"owner_agent": "langford", "api_key": "k",
                                "agent_id": "langford"}))
    assert load_credentials(good)["agent_id"] == "langford"

    wrong = tmp_path / "wrong.json"
    wrong.write_text(json.dumps({"owner_agent": "colonist-one", "api_key": "k"}))
    with pytest.raises(MoltbotdenError, match="not 'langford'"):
        load_credentials(wrong)

    keyless = tmp_path / "keyless.json"
    keyless.write_text(json.dumps({"owner_agent": "langford"}))
    with pytest.raises(MoltbotdenError, match="no api_key"):
        load_credentials(keyless)


# --- cadence policy ----------------------------------------------------------


def test_first_ever_run_is_allowed(tmp_path):
    assert gate(tmp_path).blocked_reason() is None


def test_daily_cap_then_interval(tmp_path):
    g = gate(tmp_path)
    g.record(POSTED, ref="d/1")
    assert g.blocked_reason() == DECLINED_DAILY_CAP

    # Next day, but inside the 30h interval -> cadence, not cap.
    g2 = gate(tmp_path, at=T0 + timedelta(hours=20), max_per_day=5)
    assert g2.blocked_reason() == DECLINED_CADENCE

    # Past the interval -> allowed again.
    g3 = gate(tmp_path, at=T0 + timedelta(hours=31), max_per_day=5)
    assert g3.blocked_reason() is None


def test_declines_do_not_count_against_the_cadence(tmp_path):
    """A run that chose silence must not push the next visit out.

    Otherwise a stretch of quiet days silently converts into a longer and longer
    gap — the cadence would decay without anyone deciding to change it.
    """
    g = gate(tmp_path)
    for _ in range(5):
        g.record(DECLINED_NO_CANDIDATE)
    assert g.blocked_reason() is None
    assert g.last_post_at() is None


def test_corrupt_ledger_line_does_not_shrink_history(tmp_path):
    g = gate(tmp_path)
    g.record(POSTED, ref="d/1")
    with g.ledger_path.open("a") as fh:
        fh.write("{not json\n")
    assert g.last_post_at() is not None
    assert g.blocked_reason() == DECLINED_DAILY_CAP


# --- the orchestrator: one typed row per run ---------------------------------


class FakePlatform:
    name = "moltbotden"
    supports_threading = True
    max_reply_chars = 500

    def __init__(self, posts=None, thread=None, reply_id="c1", raise_list=False):
        self.agent_id = ME
        self._posts = posts if posts is not None else []
        self._thread = thread
        self._reply_id = reply_id
        self._raise_list = raise_list
        self.replies: list[tuple] = []

    def list_recent(self, den, limit=10):
        if self._raise_list:
            raise RuntimeError("den unreachable")
        return self._posts

    async def fetch_thread(self, ref):
        return self._thread

    async def reply(self, ref, body, *, parent_id=None):
        self.replies.append((ref, body))
        return self._reply_id


def _thread(ref="technical/p1", comments=()):
    from langford.platform import Thread
    return Thread(ref=ref, title="t", body="b", author="someone", comments=tuple(comments))


def _post(pid="p1", author="someone"):
    return {"id": pid, "agent_id": author}


def test_happy_path_posts_and_records(tmp_path):
    p = FakePlatform(posts=[_post()], thread=_thread())
    g = gate(tmp_path)
    row = _run(run_once(p, g, lambda t: "a real reply", dens=["technical"]))
    assert row["decision"] == POSTED
    assert p.replies == [("technical/p1", "a real reply")]
    assert rows(g)[-1]["comment_id"] == "c1"


@pytest.mark.parametrize(
    "make,expected",
    [
        (lambda: (FakePlatform(raise_list=True), lambda t: "x"), COULD_NOT_REACH),
        (lambda: (FakePlatform(posts=[]), lambda t: "x"), DECLINED_NO_CANDIDATE),
        (lambda: (FakePlatform(posts=[_post()], thread=None), lambda t: "x"), COULD_NOT_REACH),
        (lambda: (FakePlatform(posts=[_post()], thread=_thread()), lambda t: ""), DECLINED_EMPTY_COMPOSE),
        (lambda: (FakePlatform(posts=[_post()], thread=_thread()), lambda t: "z" * 600), REFUSED_TOO_LONG),
        (lambda: (FakePlatform(posts=[_post()], thread=_thread(), reply_id=None), lambda t: "x"), COULD_NOT_REACH),
        (lambda: (FakePlatform(posts=[_post()], thread=_thread()),
                  lambda t: Refusal(REFUSED_UNGROUNDED, {"why": "invented a figure"})),
         REFUSED_UNGROUNDED),
    ],
    ids=["den-unreachable", "no-candidate", "thread-unreachable",
         "model-chose-silence", "too-long", "reply-failed", "ungrounded"],
)
def test_every_path_records_exactly_one_row(tmp_path, make, expected):
    """No path may exit without a record — that is the whole contract."""
    platform, compose = make()
    g = gate(tmp_path)
    row = _run(run_once(platform, g, compose, dens=["technical"]))
    assert row["decision"] == expected
    assert len(rows(g)) == 1, "exactly one row per run"


def test_every_decision_value_is_emitted_by_some_fixture(tmp_path):
    """Verdict coverage, not branch coverage.

    Added when REFUSED_UNGROUNDED was introduced and the parametrised list above
    kept passing without it — a new terminal state that no fixture produced, and
    nothing went red. "The branch is reachable" and "some test reached it" are
    different claims, and only the second is evidence.

    Self-contained on purpose: collecting verdicts from the other tests via a
    shared fixture would make this pass or fail on test ORDER, and a coverage
    check whose result depends on ordering is not a coverage check.

    RETRACTED is excluded by name: it is applied to an existing row after the
    fact by a maintainer, so run_once cannot emit it. Naming the exclusion keeps
    the hole visible instead of letting the expected set quietly shrink.
    """
    from langford.participation import DECISIONS, RETRACTED

    ok = FakePlatform(posts=[_post()], thread=_thread())
    producers = {
        POSTED: (ok, lambda t: "a real reply", {}),
        COULD_NOT_REACH: (FakePlatform(raise_list=True), lambda t: "x", {}),
        DECLINED_NO_CANDIDATE: (FakePlatform(posts=[]), lambda t: "x", {}),
        DECLINED_EMPTY_COMPOSE: (ok, lambda t: "", {}),
        REFUSED_TOO_LONG: (ok, lambda t: "z" * 600, {}),
        REFUSED_UNGROUNDED: (
            ok, lambda t: Refusal(REFUSED_UNGROUNDED, {"why": "invented a figure"}), {}),
        # max_per_day must be raised here or the daily cap fires first and this
        # "cadence" fixture silently tests the cap instead — a producer that
        # emits the wrong verdict is worse than none, since it reads as coverage.
        DECLINED_CADENCE: (ok, lambda t: "x",
                           {"min_interval_hours": 999, "max_per_day": 99}),
        DECLINED_DAILY_CAP: (ok, lambda t: "x", {"max_per_day": 0}),
        # This one had NO producer anywhere in the suite before this test was
        # written — a terminal state that existed, was reachable, and had never
        # been shown to be reached. Exactly the hole the test is for.
        REFUSED_REPETITIVE: (
            ok, lambda t: Refusal(REFUSED_REPETITIVE, {"why": "same post again"}), {}),
        DECLINED_ALREADY_REPLIED: (
            FakePlatform(
                posts=[_post()],
                thread=_thread(comments=[Comment(id="c0", author=ME, body="mine")]),
            ),
            lambda t: "x", {}),
    }
    expected = set(DECISIONS) - {RETRACTED}
    assert set(producers) == expected, (
        f"no producer for {sorted(expected - set(producers))} — a decision value "
        "exists that no test can make the code emit")

    emitted = set()
    for want, (platform, compose, gate_kw) in producers.items():
        g = gate(tmp_path / want, **gate_kw)
        if want in (DECLINED_CADENCE, DECLINED_DAILY_CAP):
            g.record(POSTED, ref="technical/seed")  # prior activity to be blocked by
        row = _run(run_once(platform, g, compose, dens=["technical"]))
        emitted.add(row["decision"])
        assert row["decision"] == want, f"wanted {want}, got {row['decision']}"

    missing = expected - emitted
    assert not missing, (
        f"decision value(s) {sorted(missing)} are defined but never produced — "
        "an unemitted verdict is untested however green the suite looks")


def test_unreachable_is_not_filed_as_silence(tmp_path):
    """The distinction the whole ledger exists for.

    'Nothing to say' and 'could not ask' must never share a decision value —
    conflating them is how a broken integration reads as a quiet one.
    """
    g = gate(tmp_path)
    _run(run_once(FakePlatform(raise_list=True), g, lambda t: "x", dens=["d"]))
    _run(run_once(FakePlatform(posts=[]), g, lambda t: "x", dens=["d"]))
    got = [r["decision"] for r in rows(g)]
    assert got == [COULD_NOT_REACH, DECLINED_NO_CANDIDATE]
    assert got[0] != got[1]


def test_never_speaks_twice_in_one_thread(tmp_path):
    from langford.platform import Comment
    p = FakePlatform(
        posts=[_post()],
        thread=_thread(comments=[Comment("c0", ME, "already said this")]),
    )
    g = gate(tmp_path)
    row = _run(run_once(p, g, lambda t: "again", dens=["technical"]))
    assert row["decision"] == DECLINED_ALREADY_REPLIED
    assert p.replies == []


def test_skips_his_own_posts_as_candidates(tmp_path):
    p = FakePlatform(posts=[_post(author=ME)], thread=_thread())
    row = _run(run_once(p, gate(tmp_path), lambda t: "x", dens=["technical"]))
    assert row["decision"] == DECLINED_NO_CANDIDATE


def test_replies_to_colonist_one_like_any_other_user(tmp_path):
    """Operator decision, 2026-07-26: ColonistOne is not special here.

    Deliberately asserted so a later pass does not "fix" it into an exclusion —
    it cuts against the don't-self-engage norm for dogfood agents on purpose.
    """
    p = FakePlatform(posts=[_post(author="colonist-one")], thread=_thread())
    row = _run(
        run_once(p, gate(tmp_path), lambda t: "replying to my operator",
                 dens=["technical"])
    )
    assert row["decision"] == POSTED
    assert p.replies[0][1] == "replying to my operator"


def test_cadence_block_short_circuits_before_any_network_call(tmp_path):
    g = gate(tmp_path)
    g.record(POSTED, ref="d/1")
    p = FakePlatform(raise_list=True)   # would blow up if consulted
    row = _run(run_once(p, g, lambda t: "x", dens=["technical"]))
    assert row["decision"] == DECLINED_DAILY_CAP


def test_run_once_accepts_an_async_compose(tmp_path):
    """The real compose is async — a sync LLM call would block the shared loop.

    Without this the async path is untested and would fail as "compose returned
    a coroutine, which is truthy", posting the repr of a coroutine object.
    """
    async def compose(thread):
        return "composed off-thread"

    p = FakePlatform(posts=[_post()], thread=_thread())
    row = _run(run_once(p, gate(tmp_path), compose, dens=["technical"]))
    assert row["decision"] == POSTED
    assert p.replies[0][1] == "composed off-thread"


def test_async_compose_can_still_decline(tmp_path):
    async def compose(thread):
        return None

    p = FakePlatform(posts=[_post()], thread=_thread())
    row = _run(run_once(p, gate(tmp_path), compose, dens=["technical"]))
    assert row["decision"] == DECLINED_EMPTY_COMPOSE
    assert p.replies == []


# --- usable_reply: the sanitiser that was missing when Langford posted a repr ---

class FakeOut:
    """Stands in for a LangChain AIMessage."""
    def __init__(self, content, done_reason=None, eval_count=None):
        self.content = content
        self.response_metadata = {}
        if done_reason:
            self.response_metadata["done_reason"] = done_reason
        if eval_count:
            self.response_metadata["eval_count"] = eval_count


def test_usable_reply_accepts_a_normal_reply():
    assert usable_reply(FakeOut("a real reply"), 500) == "a real reply"


def test_usable_reply_rejects_the_exact_production_failure():
    """content='' + done_reason='length' — what actually shipped a repr.

    The old code did `(out.content or str(out))`, so an empty string fell through
    to the object's repr, which was then truncated to the cap and published to a
    peer platform under Langford's name.
    """
    out = FakeOut("", done_reason="length", eval_count=4096)
    assert usable_reply(out, 500) is None


def test_usable_reply_never_falls_back_to_the_object():
    class NoContent:
        response_metadata = {}
        def __repr__(self):
            return "AIMessage(content='' additional_kwargs={})"
    assert usable_reply(NoContent(), 500) is None


def test_usable_reply_rejects_a_truncated_generation_even_with_text():
    """A cut-off thought is not a short thought."""
    out = FakeOut("this sentence was going somewhere and then", done_reason="length")
    assert usable_reply(out, 500) is None


def test_usable_reply_rejects_serialised_objects_that_slip_through():
    out = FakeOut("content='hi' additional_kwargs={} response_metadata={}")
    assert usable_reply(out, 500) is None


def test_usable_reply_refuses_over_cap_rather_than_truncating():
    """The adapter's own docstring: refuse, never truncate.

    Truncating here meant the adapter's refusal could never fire, so its stated
    policy was unreachable code.
    """
    out = FakeOut("z" * 600)
    assert usable_reply(out, 500) is None


def test_usable_reply_honours_pass_and_empty():
    assert usable_reply(FakeOut("PASS"), 500) is None
    assert usable_reply(FakeOut("   "), 500) is None


def test_usable_reply_control_boundary_cases_still_allowed():
    """Controls: rejecting everything would pass every test above."""
    assert usable_reply(FakeOut("z" * 500), 500) == "z" * 500      # exactly at cap
    assert usable_reply(FakeOut("passable point actually"), 500) is not None  # not PASS
    assert usable_reply(FakeOut("fine"), None) == "fine"           # no cap configured
    assert usable_reply(FakeOut("done", done_reason="stop"), 500) == "done"


def test_a_retracted_post_does_not_consume_the_cadence(tmp_path):
    """A comment that was deleted never stood, so it bought no silence.

    Langford's first Moltbotden comment was malformed and removed. Letting it
    gate the next 30 hours would mean a broken post purchased exactly the same
    quiet as a good one.
    """
    g = gate(tmp_path)
    g.record(POSTED, ref="technical/p1", comment_id="c1", retracted=True)
    assert g.blocked_reason() is None
    assert g.last_post_at() is None
    assert g.posts_today() == 0
    # and he may speak in that thread again, since he never really did
    assert g.replied_refs() == set()


def test_control_a_standing_post_still_blocks(tmp_path):
    """Without this, `retracted` could be swallowing every post."""
    g = gate(tmp_path)
    g.record(POSTED, ref="technical/p1", comment_id="c1")
    assert g.blocked_reason() == DECLINED_DAILY_CAP
    assert g.replied_refs() == {"technical/p1"}
