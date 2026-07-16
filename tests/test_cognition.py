"""Tests for proof-of-cognition challenge handling.

Langford solves the optional "Cognition Check" the server may attach to a
create response and answers it at the client layer, transparent to the agent.
These cover the pure extract/parse helpers and the client-wrap install path
with fakes (no live Colony, no Ollama).
"""

from __future__ import annotations

from typing import Any

from langford.__main__ import (
    _extract_cognition_challenge,
    _install_cognition_handler,
    _maybe_answer_cognition,
    _parse_cognition_answer,
)

CHALLENGE = {
    "status": "requested",
    "prompt": "eAch] TId/E pOOl hOLdS EIght s[HEl~Ls; th-eRe Are fOu[RtEEN PoOL^S. HOW maNY?",
    "token": "tok-abc",
    "difficulty": 1,
}


# --- _extract_cognition_challenge -----------------------------------------


def test_extract_none_when_no_cognition():
    assert _extract_cognition_challenge({"id": "c1"}) is None


def test_extract_none_when_cognition_null():
    assert _extract_cognition_challenge({"id": "c1", "cognition": None}) is None


def test_extract_none_without_token():
    assert _extract_cognition_challenge({"id": "c1", "cognition": {"prompt": "x"}}) is None


def test_extract_none_for_non_dict():
    assert _extract_cognition_challenge("nope") is None
    assert _extract_cognition_challenge(None) is None


def test_extract_returns_block_when_token_and_prompt_present():
    got = _extract_cognition_challenge({"id": "c1", "cognition": CHALLENGE})
    assert got is CHALLENGE


# --- _parse_cognition_answer ----------------------------------------------


def test_parse_bare_number():
    assert _parse_cognition_answer("112") == "112"


def test_parse_takes_last_integer_after_working():
    assert _parse_cognition_answer("eight times fourteen = 112") == "112"


def test_parse_ignores_think_words_no_digits():
    assert _parse_cognition_answer("<think>eight and fourteen</think>\n112") == "112"


def test_parse_with_trailing_punctuation():
    assert _parse_cognition_answer("The answer is 9.") == "9"


def test_parse_none_when_no_digits():
    assert _parse_cognition_answer("I cannot solve this") is None
    assert _parse_cognition_answer("") is None


# --- fakes ----------------------------------------------------------------


class _FakeLLM:
    """Returns a canned solve reply; records the prompt it was handed."""

    def __init__(self, reply: str) -> None:
        self.reply = reply
        self.seen: list[Any] = []

    def invoke(self, messages: Any) -> Any:
        self.seen.append(messages)

        class _Resp:
            content = self.reply

        return _Resp()


class _FakeClient:
    def __init__(self, create_resp: Any) -> None:
        self._create_resp = create_resp
        self.raw_calls: list[tuple[str, str, Any]] = []
        self.answer_result = {"status": "proved", "reason": "ok", "attempts_remaining": 0}

    def create_comment(self, **_kwargs: Any) -> Any:
        return self._create_resp

    def create_post(self, **_kwargs: Any) -> Any:
        return self._create_resp

    def _raw_request(self, method: str, path: str, body: Any = None) -> Any:
        self.raw_calls.append((method, path, body))
        return self.answer_result


class _FakeToolkit:
    def __init__(self, client: _FakeClient) -> None:
        self.client = client


# --- _maybe_answer_cognition ----------------------------------------------


def test_maybe_answer_comment_posts_to_comment_endpoint():
    client = _FakeClient(None)
    llm = _FakeLLM("112")
    resp = {"id": "c1", "cognition": CHALLENGE}
    _maybe_answer_cognition(client, llm, "comment", resp)
    assert client.raw_calls == [
        ("POST", "/comments/c1/cognition", {"token": "tok-abc", "answer": "112"})
    ]


def test_maybe_answer_post_posts_to_post_endpoint():
    client = _FakeClient(None)
    llm = _FakeLLM("The total is 170")
    resp = {"id": "p1", "cognition": CHALLENGE}
    _maybe_answer_cognition(client, llm, "post", resp)
    assert client.raw_calls == [
        ("POST", "/posts/p1/cognition", {"token": "tok-abc", "answer": "170"})
    ]


def test_maybe_answer_noop_without_challenge():
    client = _FakeClient(None)
    llm = _FakeLLM("112")
    _maybe_answer_cognition(client, llm, "comment", {"id": "c1"})
    assert client.raw_calls == []


def test_maybe_answer_noop_when_llm_gives_no_number():
    client = _FakeClient(None)
    llm = _FakeLLM("I don't know")
    _maybe_answer_cognition(client, llm, "comment", {"id": "c1", "cognition": CHALLENGE})
    assert client.raw_calls == []


# --- _install_cognition_handler -------------------------------------------


def test_install_wraps_create_comment_and_answers():
    resp = {"id": "c1", "cognition": CHALLENGE}
    client = _FakeClient(resp)
    toolkit = _FakeToolkit(client)
    llm = _FakeLLM("112")

    _install_cognition_handler(toolkit, llm)  # type: ignore[arg-type]
    out = client.create_comment(post_id="p1", body="hi")

    assert out is resp  # original create response passed through unchanged
    assert client.raw_calls == [
        ("POST", "/comments/c1/cognition", {"token": "tok-abc", "answer": "112"})
    ]


def test_install_create_post_unchallenged_is_noop():
    resp = {"id": "p1"}  # no cognition block
    client = _FakeClient(resp)
    toolkit = _FakeToolkit(client)
    llm = _FakeLLM("112")

    _install_cognition_handler(toolkit, llm)  # type: ignore[arg-type]
    out = client.create_post(title="t", body="b", colony="general")

    assert out is resp
    assert client.raw_calls == []


def test_install_handler_never_raises_into_create():
    """A failure inside the handler must not break the create."""
    resp = {"id": "c1", "cognition": CHALLENGE}
    client = _FakeClient(resp)
    toolkit = _FakeToolkit(client)

    class _BoomLLM:
        def invoke(self, _messages: Any) -> Any:
            raise RuntimeError("ollama down")

    _install_cognition_handler(toolkit, _BoomLLM())  # type: ignore[arg-type]
    out = client.create_comment(post_id="p1", body="hi")
    assert out is resp  # create still returns despite the solver blowing up
