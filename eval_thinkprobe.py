#!/usr/bin/env python3
"""Control-test the think-detector, and find where 4096 tokens went.

The /no_think A/B reported think_blocks=0 for BOTH variants and I nearly read
that as "the switch works". But one generation evaluated 4096 tokens and
returned ZERO characters of content with no <think> marker. Tokens do not
vanish. Either they went somewhere `content` does not show, or the detector
cannot see what it was built to see.

`has_think_block` looked for "<think>" in content. If thinking never lands in
content, that check returns 0 whether the model thought or not — a detector
that cannot fire, reporting clean. Zero from an instrument never shown to
produce non-zero is the failure this whole codebase is about.

So: the SAME prompt with the switch REMOVED, which should make the model think.

  * If <think> now appears -> the detector works and the switch works.
  * If content is empty again with a high eval_count -> thinking is invisible in
    `content`, the earlier zeros meant nothing, and the ceiling hits are almost
    certainly thinking that the detector could never have caught.

Also dumps every field of the response object, because "where did 4096 tokens
go" is answerable by looking rather than reasoning.
"""
from __future__ import annotations
import json, os, pathlib, sys, time
sys.path.insert(0, str(pathlib.Path(__file__).parent / "src"))
from langchain_ollama import ChatOllama
from langford.moltbotden import COMMENT_CHAR_CAP
from langford.prompts import reply_prompt

MODEL=os.environ.get("OLLAMA_MODEL","qwen3.6:27b")
BASE=os.environ.get("OLLAMA_BASE_URL","http://localhost:11434")
OUT=pathlib.Path(__file__).parent/"eval_thinkprobe_results.json"
CASE=dict(author="clawgrcbot",
  body=("Everyone post your actual numbers: what p99 did you measure after moving "
        "off per-IP buckets, and on what hardware? Numbers only, no theory please."),
  comments=[])

def main()->int:
    llm=ChatOllama(model=MODEL, base_url=BASE, temperature=0.7, num_predict=4096)
    base=reply_prompt(author=CASE["author"], body=CASE["body"],
                      comments=CASE["comments"], cap=COMMENT_CHAR_CAP)
    assert base.endswith(" /no_think")
    thinking_on = base[:-len(" /no_think")]          # THE CONTROL: switch removed
    rows=[]
    for label, prompt in (("switch_REMOVED_control", thinking_on),
                          ("switch_present", base)):
        for rep in range(3):
            t0=time.time(); out=llm.invoke(prompt)
            content=getattr(out,"content","") or ""
            meta=getattr(out,"response_metadata",None) or {}
            extra=getattr(out,"additional_kwargs",None) or {}
            rows.append({"variant":label,"rep":rep,"chars":len(content),
                "eval_count":meta.get("eval_count"),
                "done_reason":meta.get("done_reason"),
                "has_think_marker":("<think>" in content or "</think>" in content),
                "meta_keys":sorted(meta.keys()),
                "additional_kwargs_keys":sorted(extra.keys()),
                "other_attrs":[a for a in dir(out)
                               if not a.startswith("_") and a in
                               ("reasoning","thinking","reasoning_content")],
                "head":content[:200],"seconds":round(time.time()-t0,1)})
            r=rows[-1]
            print(f"  {label:24} rep{rep} chars={r['chars']:5} eval={r['eval_count']} "
                  f"done={r['done_reason']} think_marker={r['has_think_marker']} "
                  f"({r['seconds']}s)", flush=True)
    OUT.write_text(json.dumps({"model":MODEL,"rows":rows},indent=2))
    ctrl=[r for r in rows if r["variant"].endswith("control")]
    fired=any(r["has_think_marker"] for r in ctrl)
    print("\nmeta keys seen:", sorted({k for r in rows for k in r["meta_keys"]}))
    print("additional_kwargs keys:", sorted({k for r in rows for k in r["additional_kwargs_keys"]}))
    print("reasoning-ish attrs on the response:", sorted({a for r in rows for a in r["other_attrs"]}) or "none")
    if fired:
        print("\nDETECTOR WORKS: <think> appears when the switch is removed. The "
              "earlier zeros were real and the switch is doing its job.")
    else:
        print("\n** DETECTOR IS VACUOUS **: no <think> marker even with thinking "
              "enabled. Every think_blocks=0 I have reported means nothing, and "
              "the ceiling hits were never testable by that check.")
    return 0

if __name__=="__main__":
    raise SystemExit(main())
