#!/usr/bin/env python3
"""A/B the /no_think placement. Nothing is published.

2 of 14 generations in the abstention fixture hit the 4096-token ceiling and
produced nothing. The requested reply is 400 characters — roughly 100 tokens —
so a ceiling hit is not a slightly-long reply, it is a runaway producing ~40x
what was asked.

**The hypothesis.** This codebase already documents the failure and its fix, in
`_solve_cognition`: the Qwen3 `/no_think` soft-switch stops the model "burning
its num_predict budget inside a <think> block and truncating before the answer
lands". That call site spells it `prompt + "\\n\\n/no_think"` — own line, blank
line before. The reply and post prompts instead append `" /no_think"` inline, so
the prompt ends:

    ...reply with exactly: PASS /no_think

If the switch is only recognised on its own line, the prompts have been asking a
thinking model for a short answer with a 4096-token budget, which is exactly the
shape of the observed failure.

It is a hypothesis. It could equally be that placement is irrelevant and the
ceiling hits are variance in a small sample. That is what this measures.

Same case, same repeats, one variable:

  A_inline   "...PASS /no_think"      (what ships today)
  B_ownline  "...PASS\\n\\n/no_think"  (the form the cognition solver uses)

Reported: ceiling rate, whether the raw output contains a <think> block, and
output length. The <think> check is the discriminating one — if inline
generations think and own-line ones do not, the mechanism is established rather
than inferred from a rate.

    colony-agent-lock langford-eval uv run python -u eval_nothink.py
"""
from __future__ import annotations

import json
import os
import pathlib
import statistics
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).parent / "src"))

from langchain_ollama import ChatOllama  # noqa: E402

from langford.moltbotden import COMMENT_CHAR_CAP  # noqa: E402
from langford.prompts import reply_prompt  # noqa: E402

MODEL = os.environ.get("OLLAMA_MODEL", "qwen3.6:27b")
BASE = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
TEMP = float(os.environ.get("LANGFORD_TEMPERATURE", "0.7"))
REPS = int(os.environ.get("NOTHINK_REPS", "6"))
OUT = pathlib.Path(__file__).parent / "eval_nothink_results.json"


class C:
    def __init__(self, author, body):
        self.author, self.body = author, body


# The case that hit the ceiling in both recorded runs: a demand for first-hand
# telemetry. Plausibly the hardest to answer, so the most likely to spiral.
CASE = dict(
    author="clawgrcbot",
    body=("Everyone post your actual numbers: what p99 did you measure after "
          "moving off per-IP buckets, and on what hardware? Numbers only, no "
          "theory please."),
    comments=[],
)


def variants() -> dict[str, str]:
    base = reply_prompt(author=CASE["author"], body=CASE["body"],
                        comments=CASE["comments"], cap=COMMENT_CHAR_CAP)
    assert base.endswith(" /no_think"), "shipped prompt no longer ends inline"
    stripped = base[: -len(" /no_think")]
    return {"A_inline": base, "B_ownline": stripped + "\n\n/no_think"}


def main() -> int:
    llm = ChatOllama(model=MODEL, base_url=BASE, temperature=TEMP, num_predict=4096)
    v = variants()
    print(f"model={MODEL} temp={TEMP} reps={REPS}", flush=True)
    for k, p in v.items():
        print(f"  {k}: prompt ends {p[-28:]!r}", flush=True)
    rows = []
    for name, prompt in v.items():
        for rep in range(REPS):
            t0 = time.time()
            try:
                out = llm.invoke(prompt)
                content = getattr(out, "content", "") or ""
                meta = getattr(out, "response_metadata", None) or {}
                err = None
            except Exception as exc:
                content, meta, err = "", {}, f"{type(exc).__name__}: {exc}"
            ceiling = meta.get("done_reason") == "length"
            thinking = "<think>" in content or "</think>" in content
            rows.append({
                "variant": name, "rep": rep, "ceiling": ceiling,
                "has_think_block": thinking, "chars": len(content),
                "eval_count": meta.get("eval_count"), "error": err,
                "head": content[:160], "seconds": round(time.time() - t0, 1),
            })
            print(f"  {name:10} rep{rep} ceiling={ceiling!s:5} think={thinking!s:5} "
                  f"chars={len(content):5} ({rows[-1]['seconds']}s)", flush=True)

    summary = {}
    for name in v:
        rs = [r for r in rows if r["variant"] == name]
        summary[name] = {
            "n": len(rs),
            "ceiling_hits": sum(r["ceiling"] for r in rs),
            "think_blocks": sum(r["has_think_block"] for r in rs),
            "median_chars": int(statistics.median(r["chars"] for r in rs)),
        }
    OUT.write_text(json.dumps({"model": MODEL, "case": CASE["body"][:80],
                               "summary": summary, "rows": rows}, indent=2))
    print("\n" + json.dumps(summary, indent=2))
    a, b = summary["A_inline"], summary["B_ownline"]
    if a["think_blocks"] > b["think_blocks"]:
        print("\nMECHANISM ESTABLISHED: the inline switch is not suppressing "
              "thinking and the own-line one is.")
    elif a["ceiling_hits"] == b["ceiling_hits"] == 0:
        print("\nINCONCLUSIVE: neither variant hit the ceiling, so this sample "
              "cannot separate them. The 2/14 was rarer than these reps can see.")
    else:
        print("\nNO MECHANISM SHOWN: placement did not change the think-block "
              "rate. Ceiling hits are something else — do not 'fix' the "
              "placement on the strength of a rate difference alone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
