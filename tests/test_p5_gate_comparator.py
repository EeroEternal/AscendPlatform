"""The P5 gate's judge logic, tested — because the runbook's script is code too.

[`docs/notes/p5-gate.md`](../docs/notes/p5-gate.md) tells whoever has the card to capture two
``/v1/completions`` responses and compare them.  If that comparison is wrong, the gate either passes
something it should not or fails something it should not, and the person holding the card cannot tell
which.  So the judge is a tested artifact here: same semantics as the snippet, exercised on synthetic
payloads that differ in exactly the ways a real pair can.

The comparison is *bitwise*, not "close": ``float.hex()`` round-trips a double exactly, so equal hex
means equal bits.  Four decimals — the hardware checklist's first-light criterion — cannot separate
"the port is semantically wrong" from "the reduction order changed".
"""

from __future__ import annotations

import json
import math
import pathlib
import re
import shutil
import subprocess
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest


def _response(tokens, logprobs, top=None):
    """A minimal ``/v1/completions`` body: what the gate actually parses."""
    return {
        "choices": [
            {
                "logprobs": {
                    "tokens": list(tokens),
                    "token_logprobs": list(logprobs),
                    "top_logprobs": list(top) if top is not None else [None] * len(tokens),
                }
            }
        ]
    }


def flatten(body):
    """``[(token, token_logprob.hex(), ((alt, alt.hex()), ...)), ...]`` — the runbook's ``flat()``."""
    logprobs = body["choices"][0]["logprobs"]
    flat = []
    for index, token in enumerate(logprobs["tokens"]):
        alternatives = (logprobs.get("top_logprobs") or [None] * len(logprobs["tokens"]))[index] or {}
        flat.append(
            (
                token,
                float(logprobs["token_logprobs"][index]).hex(),
                tuple(sorted((name, float(value).hex()) for name, value in alternatives.items())),
            )
        )
    return flat


def compare(left, right):
    """``(ok, message)`` — the gate's verdict, with the first divergence named."""
    a, b = flatten(left), flatten(right)
    for index, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return False, f"token {index} diverges:\n  left : {x}\n  right: {y}"
    if len(a) != len(b):
        return False, f"different token counts: {len(a)} vs {len(b)}"
    return True, "bitwise identical"


# ---------------------------------------------------------------------------------------------
# The verdicts the person on the box will see
# ---------------------------------------------------------------------------------------------


def test_identical_responses_are_bitwise_identical():
    body = _response(["The", " capital"], [-0.125, -1.75], [{"The": -0.125}, {" capital": -1.75}])
    ok, message = compare(body, json.loads(json.dumps(body)))
    assert ok and message == "bitwise identical"


def test_a_one_ulp_difference_fails_and_is_reported_as_such():
    """The whole point: ``round(x, 4)`` would call these equal."""
    left = _response(["a"], [-0.125])
    right = _response(["a"], [-0.12500000000000003])
    assert round(left["choices"][0]["logprobs"]["token_logprobs"][0], 4) == round(
        right["choices"][0]["logprobs"]["token_logprobs"][0], 4
    ), "the four-decimal criterion the checklist uses cannot see this difference"
    ok, message = compare(left, right)
    assert not ok
    assert "token 0 diverges" in message


def test_a_plausible_but_wrong_number_is_reported_at_the_divergence_point():
    """Semantic breakage: same token, logprob off by 1e-3 (head layout / KV stride class of bug)."""
    left = _response(["a", "b", "c"], [-0.1, -0.2, -0.3])
    right = _response(["a", "b", "c"], [-0.1, -0.201, -0.3])
    ok, message = compare(left, right)
    assert not ok
    assert "token 1 diverges" in message, "the verdict must point at the first bad token, not the tail"


def test_a_different_token_count_fails_even_when_the_prefix_matches():
    left = _response(["a", "b"], [-0.1, -0.2])
    right = _response(["a"], [-0.1])
    ok, message = compare(left, right)
    assert not ok and "different token counts" in message


def test_top_k_alternatives_are_part_of_the_verdict():
    """Two runs can agree on the chosen token while disagreeing on the distribution behind it."""
    left = _response(["a"], [-0.1], [{"a": -0.1, "b": -2.0}])
    right = _response(["a"], [-0.1], [{"a": -0.1, "b": -2.0000001}])
    ok, message = compare(left, right)
    assert not ok and "token 0 diverges" in message


def test_a_response_without_top_logprobs_still_compares():
    left = _response(["a", "b"], [-0.1, -0.2])
    ok, _message = compare(left, left)
    assert ok


def test_hex_is_the_judge_so_serialisation_must_round_trip():
    """Guards the assumption: ``json`` writes a float with enough digits to read it back exactly."""
    value = -0.12500000000000003
    restored = json.loads(json.dumps({"v": value}))["v"]
    assert float(restored).hex() == value.hex()


def test_the_runbook_and_this_file_agree_on_the_judge():
    """A cheap consistency check: the doc must not grow a different criterion than the tested one."""
    doc = pathlib.Path(__file__).resolve().parents[1] / "docs" / "notes" / "p5-gate.md"
    text = doc.read_text(encoding="utf-8")
    assert "float.hex()" in text
    assert "top_logprobs" in text
    assert "逐位" in text
    assert "round(x, 4)" in text or "4 位小数" in text


# =============================================================================================
# the runbook's *actual* script: extracted from the note and executed, not re-implemented
# =============================================================================================

ROOT = pathlib.Path(__file__).resolve().parents[1]
NOTE = ROOT / "docs" / "notes" / "p5-gate.md"


def _runbook_script() -> str:
    """The one fenced ``python`` block in the note that defines ``flat()``.

    The tests above re-implement that function so their assertions read cleanly; this one takes the
    artifact the operator will actually paste into ``/tmp/p5_compare.py`` and runs *it*.  Otherwise
    the note could drift into something broken while the test file stays green - the exact failure
    mode the note's own paragraph promises cannot happen.
    """
    text = NOTE.read_text(encoding="utf-8")
    blocks = [block for block in re.findall(r"```python\n(.*?)```", text, re.S) if "def flat(" in block]
    assert len(blocks) == 1, f"expected exactly one comparator block in {NOTE.name}, found {len(blocks)}"
    return blocks[0]


def _compare(tmp_path: pathlib.Path, left: dict, right: dict) -> subprocess.CompletedProcess:
    script = tmp_path / "p5_compare.py"
    script.write_text(_runbook_script(), encoding="utf-8")
    left_path = tmp_path / "a.json"
    right_path = tmp_path / "b.json"
    left_path.write_text(json.dumps(left), encoding="utf-8")
    right_path.write_text(json.dumps(right), encoding="utf-8")
    return subprocess.run(
        [sys.executable, str(script), str(left_path), str(right_path)],
        capture_output=True,
        text=True,
        timeout=120,
    )


def test_the_runbooks_own_script_passes_an_identical_pair(tmp_path):
    body = _response(["a", "b"], [-0.5, -1.25], [{"x": -0.5}, {"y": -2.0}])
    result = _compare(tmp_path, body, body)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "PASS" in result.stdout


def test_the_runbooks_own_script_fails_one_ulp(tmp_path):
    body = _response(["a"], [-0.5])
    nudged = _response(["a"], [math.nextafter(-0.5, 0.0)])
    result = _compare(tmp_path, body, nudged)
    assert result.returncode == 1
    assert "分叉" in result.stdout


def test_the_runbooks_own_script_fails_a_length_mismatch(tmp_path):
    result = _compare(tmp_path, _response(["a"], [-0.5]), _response(["a", "b"], [-0.5, -1.0]))
    assert result.returncode == 1
    assert "长度不同" in result.stdout


def test_the_runbooks_own_script_fails_a_topk_difference(tmp_path):
    left = _response(["a"], [-0.5], [{"x": -0.5, "y": -1.0}])
    right = _response(["a"], [-0.5], [{"x": -0.5, "z": -1.0}])
    result = _compare(tmp_path, left, right)
    assert result.returncode == 1
    assert "分叉" in result.stdout


# =============================================================================================
# the judge reads three keys off the wire; those keys belong to vLLM, not to us
# =============================================================================================

#: What the runbook's ``flat()`` reads from ``choices[0].logprobs``.
WIRE_KEYS = ("tokens", "token_logprobs", "top_logprobs")


def test_the_keys_the_judge_reads_are_the_keys_vllm_023_serialises():
    """A vLLM point release could rename one of these and the gate would compare nothing.

    ``flat()`` indexes ``lp["tokens"]`` / ``lp["token_logprobs"]`` and ``lp.get("top_logprobs")``.
    If 0.24 renamed them, a run against a *newer* engine would raise ``KeyError`` on the capture —
    or, worse for ``top_logprobs`` (read with ``.get``), silently compare only the sampled token.
    The schema is imported from the engine actually installed, so this cannot drift unnoticed.
    """
    pytest.importorskip("vllm", reason="no vLLM here: the wire format is checked in the vLLM venv")
    protocol = pytest.importorskip(
        "vllm.entrypoints.openai.completion.protocol",
        reason="the completion protocol module moved in this vLLM",
    )
    schema = protocol.CompletionLogProbs
    missing = [key for key in WIRE_KEYS if key not in schema.model_fields]
    assert not missing, (
        f"vLLM's CompletionLogProbs has no {missing}; the P5 runbook reads "
        f"{list(WIRE_KEYS)} and would compare nothing (docs/notes/p5-gate.md §3)"
    )
    body = {"tokens": ["a"], "token_logprobs": [-0.5], "top_logprobs": [{"a": -0.5}], "text_offset": [0]}
    parsed = schema.model_validate(body)
    assert parsed.tokens == ["a"] and parsed.token_logprobs == [-0.5]


def test_the_judge_reads_the_logprobs_from_the_first_choice():
    """``choices[0]`` is what vLLM returns for a single-prompt request; the runbook hard-codes it."""
    pytest.importorskip("vllm", reason="no vLLM here")
    protocol = pytest.importorskip("vllm.entrypoints.openai.completion.protocol")
    assert "choices" in protocol.CompletionResponse.model_fields


# =============================================================================================
# the capture half: run the runbook's own bash block against a fake /v1/completions
# =============================================================================================

CAPTURE_BLOCK = re.compile(r"```bash\n(?P<body>.*?)```", re.S)


def _capture_script() -> str:
    """The bash block from the runbook that curls the endpoint and prints the token count."""
    text = NOTE.read_text(encoding="utf-8")
    for match in CAPTURE_BLOCK.finditer(text):
        body = match.group("body")
        if "curl" in body and "/v1/completions" in body:
            return body
    raise AssertionError(f"no capture block found in {NOTE.name}")


class _FakeCompletions(BaseHTTPRequestHandler):
    """Answers with a body in the shape vLLM 0.23 returns (three logprob keys, 16 tokens)."""

    payload: bytes = b""

    def do_POST(self) -> None:  # noqa: N802 - stdlib spelling
        length = int(self.headers.get("Content-Length", "0"))
        _FakeCompletions.request_body = self.rfile.read(length)
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(self.payload)))
        self.end_headers()
        self.wfile.write(self.payload)

    def log_message(self, *args) -> None:  # keep the test output clean
        return


def _serve(payload: bytes):
    import threading

    _FakeCompletions.payload = payload
    server = HTTPServer(("127.0.0.1", 0), _FakeCompletions)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, thread


def test_the_runbooks_capture_block_fetches_and_counts(tmp_path):
    """It has to work on the card, where a typo costs the session — so run it here first.

    The block is the one piece of the runbook that no test touched: ``curl`` with a multi-line
    single-quoted JSON body, then a ``python3 -c`` one-liner that reads the file back.  A quoting
    mistake in either half is invisible to a reader and fatal at the gate.
    """
    if shutil.which("curl") is None or shutil.which("bash") is None:
        pytest.skip("no curl/bash on this machine")
    tokens = ["Paris"] + [f"t{i}" for i in range(15)]
    body = _response(tokens, [-0.25] * 16, [{"Paris": -0.25}] * 16)
    server, _thread = _serve(json.dumps(body).encode())
    try:
        host, port = server.server_address[0], server.server_address[1]
        script = _capture_script().replace("http://127.0.0.1:8000", f"http://{host}:{port}")
        script = script.replace("/tmp/p5-$LABEL.json", str(tmp_path / "out.json"))
        asset = tmp_path / "capture.sh"
        asset.write_text("set -euo pipefail\n" + script, encoding="utf-8")
        result = subprocess.run(["bash", str(asset)], capture_output=True, text=True, timeout=120)
        assert result.returncode == 0, result.stdout + result.stderr
        assert f"{len(tokens)} tokens" in result.stdout, result.stdout
        written = json.loads((tmp_path / "out.json").read_text(encoding="utf-8"))
        assert written["choices"][0]["logprobs"]["tokens"] == tokens
        assert _FakeCompletions.request_body, "the request never reached the endpoint"
        request = json.loads(_FakeCompletions.request_body)
        assert request["logprobs"] == 5 and request["temperature"] == 0
    finally:
        server.shutdown()


def test_the_capture_block_asks_for_the_settings_the_gate_documents():
    """``temperature=0`` / ``top_p=1`` / ``logprobs=5`` / ``max_tokens>=16`` are part of the proof."""
    block = _capture_script()
    start = block.index("-d '") + len("-d '")
    payload = block[start : block.index("'", start)]
    request = json.loads(payload)
    assert request["temperature"] == 0
    assert request["top_p"] == 1
    assert request["logprobs"] == 5
    assert request["max_tokens"] >= 16
    assert request["model"]
