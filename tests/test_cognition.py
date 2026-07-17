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
    _solve_cognition,
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


def test_parse_strips_think_block_with_words():
    assert _parse_cognition_answer("<think>eight and fourteen</think>\n112") == "112"


def test_parse_strips_think_block_with_misleading_digits():
    # a digit INSIDE the <think> block (a discarded working step) must be
    # ignored; only the post-think final answer counts.
    assert _parse_cognition_answer("<think>13 + 7 = 99, wait, 13 + 7 = 20</think>\n20") == "20"


def test_parse_unclosed_think_returns_none():
    # a truncated generation (open <think>, no final answer) returns None, not a
    # working-step digit — the solve legitimately failed rather than guessed.
    assert _parse_cognition_answer("<think>let me compute, 13 + 7 = 99") is None


def test_parse_with_trailing_punctuation():
    assert _parse_cognition_answer("The answer is 9.") == "9"


def test_parse_none_when_no_digits():
    # default (words_ok=False, the arithmetic gate): no digit -> None, unchanged.
    assert _parse_cognition_answer("I cannot solve this") is None
    assert _parse_cognition_answer("") is None


# --- _parse_cognition_answer: words_ok (comprehension gate) ----------------


def test_parse_words_ok_returns_last_word_lowercased():
    # comprehension answers are subject words, not numbers.
    assert _parse_cognition_answer("The answer is Crab.", words_ok=True) == "crab"
    assert _parse_cognition_answer("urchin", words_ok=True) == "urchin"


def test_parse_words_ok_digit_still_wins():
    # a numeric answer must still parse as the integer even with words_ok on.
    assert _parse_cognition_answer("eight and fourteen = 112", words_ok=True) == "112"


def test_parse_words_ok_strips_think_then_takes_word():
    assert _parse_cognition_answer("<think>which one hides</think>\ncrab", words_ok=True) == "crab"


def test_parse_words_ok_empty_after_think_is_none():
    # truncated generation with words_ok on still returns None, not a think-word.
    assert _parse_cognition_answer("<think>let me read the clauses", words_ok=True) is None


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


# --- _solve_cognition: /no_think hardening --------------------------------


def test_solve_appends_no_think_and_parses():
    llm = _FakeLLM("20")
    out = _solve_cognition(llm, "some obfuscated prompt")
    assert out == "20"
    human = llm.seen[0][1]  # [SystemMessage, HumanMessage]
    assert "/no_think" in human.content  # Qwen3 soft-switch disables <think> on the solve


def test_solve_survives_truncated_think():
    # a model that returns an unclosed <think> (num_predict truncation) yields
    # no answer rather than a mid-reasoning working digit.
    llm = _FakeLLM("<think>13 + 7 = 99")
    assert _solve_cognition(llm, "prompt") is None


# --- _solve_cognition: allow_think (multi-step / comprehension tiers) ------


def test_solve_allow_think_omits_no_think():
    # with thinking allowed, the /no_think soft-switch must NOT be appended —
    # multi-step arithmetic and referent-resolution need the model to reason.
    llm = _FakeLLM("42")
    out = _solve_cognition(llm, "some obfuscated prompt", allow_think=True)
    assert out == "42"
    human = llm.seen[0][1]
    assert "/no_think" not in human.content


def test_solve_allow_think_parses_word_answer():
    # a comprehension answer is a word; allow_think enables the word-answer path.
    llm = _FakeLLM("<think>the crab hides in the wreck</think>\ncrab")
    assert _solve_cognition(llm, "prompt", allow_think=True) == "crab"


def test_solve_default_off_rejects_word_answer():
    # default (single-step gate): a non-numeric reply is still a non-answer.
    llm = _FakeLLM("crab")
    assert _solve_cognition(llm, "prompt") is None


def test_maybe_answer_threads_allow_think_and_submits_word():
    client = _FakeClient(None)
    llm = _FakeLLM("urchin")
    resp = {"id": "c9", "cognition": CHALLENGE}
    _maybe_answer_cognition(client, llm, "comment", resp, allow_think=True)
    assert client.raw_calls == [
        ("POST", "/comments/c9/cognition", {"token": "tok-abc", "answer": "urchin"})
    ]
