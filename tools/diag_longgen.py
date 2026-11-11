"""Long-generation consistency probe: does a decode step ever disagree with a re-prefill?

The failure it was written for: Qwen3.6 served by this platform produced 14 correct tokens and then
``!`` for the rest of the sequence.  ``logprobs`` showed the sampled token's logprob going ``nan`` at
exactly that step while a *re-prefill* of the same prefix still had finite, sensible logits - so one
decode step returned NaN logits, greedy then latched onto token id 0, and the poisoned state was fed
back for ever.  Reading the text alone cannot tell that apart from "the model rambled", which is why
this probe reports three things instead:

* ``MISMATCH`` - positions where re-prefilling the generated prefix picks a different token than the
  decode step did.  A near-tie produces a few of these; a *monotone* increase means the two code
  paths carry different state (the GDN/attention state hand-off), not just rounding.
* ``NAN_AT`` - the first decode position whose logprob is NaN.  That is a numerical blow-up, not a
  quality problem, and it is the signal that matters.
* ``REPEAT`` - distinct-4-gram ratio over head and tail, so a collapse is visible even when the text
  is only ever read in a log file.

Usage (needs an NPU, a model directory, and the ops layer on ``PYTHONPATH``)::

    export VLLM_PLUGINS=ascend_platform ASCEND_PLATFORM_OPS=ascend_attention
    PYTHONPATH=/path/to/AscendAttention/src python tools/diag_longgen.py --model <path> --max-new 64

Exit code 0 when every decode logprob was finite and the argmax never disagreed; 1 otherwise, so it
can gate a regression run.
"""
from __future__ import annotations

import argparse
import json
import math
import sys


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", required=True, help="local model directory")
    parser.add_argument(
        "--instruction",
        default=(
            "Write a 400-word story about a lighthouse keeper who finds a message in a bottle. "
            "Do not repeat yourself."
        ),
    )
    parser.add_argument("--max-new", type=int, default=64)
    parser.add_argument("--max-model-len", type=int, default=2048)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument("--json-out", default="", help="write the token ids and per-step rows here")
    parser.add_argument(
        "--raw-prompt",
        dest="chat_template",
        action="store_false",
        help="send the instruction verbatim instead of applying the model's chat template",
    )
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)

    from vllm import LLM, SamplingParams

    llm = LLM(
        model=args.model,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enforce_eager=True,
        trust_remote_code=True,
        disable_log_stats=True,
    )
    tokenizer = llm.get_tokenizer()
    if args.chat_template:
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": args.instruction}],
            add_generation_prompt=True,
            tokenize=True,
        )
        if hasattr(prompt, "keys"):  # fast tokenizers hand back a BatchEncoding
            prompt = prompt["input_ids"]
        prompt_ids = list(prompt)
    else:
        prompt_ids = list(tokenizer(args.instruction, add_special_tokens=False)["input_ids"])
    print(f"PROMPT_TOKENS {len(prompt_ids)}")

    gen = llm.generate(
        [{"prompt_token_ids": prompt_ids}],
        SamplingParams(temperature=0.0, max_tokens=args.max_new, logprobs=5),
    )[0]
    out = gen.outputs[0]
    generated = list(out.token_ids)
    print(f"TEXT {out.text[:200]!r}")

    def repeat(tokens, n=4):
        grams = [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]
        return 1.0 - len(set(grams)) / max(len(grams), 1)

    print(f"REPEAT head={repeat(generated[:64]):.3f} tail={repeat(generated[-64:]):.3f}")

    res2 = llm.generate(
        [{"prompt_token_ids": prompt_ids + generated}],
        SamplingParams(temperature=0.0, max_tokens=1, prompt_logprobs=5),
    )[0]
    pre = res2.prompt_logprobs

    rows, mismatch, nan_at = [], [], []
    for k, token in enumerate(generated):
        idx = len(prompt_ids) + k
        dec = out.logprobs[k] if out.logprobs else None
        row = {"k": k, "token": token, "lp_dec": None, "lp_pre": None, "pre_top": None}
        if dec is not None and token in dec:
            row["lp_dec"] = dec[token].logprob
            if row["lp_dec"] is None or math.isnan(row["lp_dec"]):
                nan_at.append(k)
        if idx < len(pre) and pre[idx] is not None:
            pre_row = pre[idx]
            row["pre_top"] = max(pre_row, key=lambda t: pre_row[t].logprob)
            if token in pre_row:
                row["lp_pre"] = pre_row[token].logprob
            if row["pre_top"] != token:
                mismatch.append(k)
        rows.append(row)

    print(f"MISMATCH {len(mismatch)}/{len(generated)} positions {mismatch[:20]}")
    print(f"NAN_AT {nan_at[:10]} (count {len(nan_at)})")
    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as handle:
            json.dump({"prompt_ids": prompt_ids, "generated": generated, "rows": rows}, handle)
        print(f"WROTE {args.json_out}")
    return 0 if not mismatch and not nan_at else 1


if __name__ == "__main__":
    sys.exit(main())
