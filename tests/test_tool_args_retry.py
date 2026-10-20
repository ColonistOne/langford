"""A tool call rejected for invalid JSON arguments is retried once, at once, with a quoting note.

2026-10-19/20: four replies lost to "llama-server returned invalid tool call
arguments for colony_comment_on_post: json: cannot unmarshal number 4e473658
into Go value of type float64" (an id written bare parses as a number).
"""

from __future__ import annotations

import asyncio

from langchain_core.messages import HumanMessage

import langford.__main__ as m

REJECTED = Exception(
    'llama-server returned invalid tool call arguments for "colony_comment_on_post": '
    "json: cannot unmarshal number 4e473658 into Go value of type float64 (status code: -1)"
)


class ScriptedAgent:
    def __init__(self, outcomes: list) -> None:
        self.outcomes = list(outcomes)
        self.payloads: list[dict] = []

    async def ainvoke(self, payload: dict):
        self.payloads.append(payload)
        out = self.outcomes.pop(0)
        if isinstance(out, BaseException):
            raise out
        return out


def _no_sleep(monkeypatch) -> list[float]:
    slept: list[float] = []

    async def fake_sleep(d: float) -> None:
        slept.append(d)

    monkeypatch.setattr(m.asyncio, "sleep", fake_sleep)
    return slept


def test_rejected_tool_call_is_retried_once_with_the_note(monkeypatch) -> None:
    slept = _no_sleep(monkeypatch)
    agent = ScriptedAgent([REJECTED, {"messages": ["ok"]}])
    first = {"messages": [HumanMessage(content="the event")]}
    out = asyncio.run(m._invoke_agent_with_retry(agent, first))
    assert out == {"messages": ["ok"]}
    assert len(agent.payloads) == 2 and slept == []
    retry_msgs = agent.payloads[1]["messages"]
    assert retry_msgs[0].content == "the event"
    assert "double quotes" in retry_msgs[-1].content
    assert len(first["messages"]) == 1, "the caller's payload is not mutated"


def test_only_one_nudge_per_call(monkeypatch) -> None:
    _no_sleep(monkeypatch)
    agent = ScriptedAgent([REJECTED, REJECTED, {"messages": ["never reached"]}])
    try:
        asyncio.run(m._invoke_agent_with_retry(agent, {"messages": [HumanMessage(content="e")]}))
    except Exception as exc:
        assert "invalid tool call arguments" in str(exc)
    else:
        raise AssertionError("a second rejection must propagate")
    assert len(agent.payloads) == 2


def test_other_errors_still_propagate_at_once(monkeypatch) -> None:
    _no_sleep(monkeypatch)
    agent = ScriptedAgent([ValueError("something else"), {"messages": ["x"]}])
    try:
        asyncio.run(m._invoke_agent_with_retry(agent, {"messages": []}))
    except ValueError:
        pass
    else:
        raise AssertionError("non-retryable errors must propagate")
    assert len(agent.payloads) == 1


def test_transient_load_errors_keep_their_backoff(monkeypatch) -> None:
    slept = _no_sleep(monkeypatch)
    agent = ScriptedAgent([Exception("model failed to load"), {"messages": ["ok"]}])
    asyncio.run(m._invoke_agent_with_retry(agent, {"messages": []}))
    assert slept == [30.0] and len(agent.payloads[1]["messages"]) == 0
