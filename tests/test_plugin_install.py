"""The ``vllm.general_plugins`` entry point's side effects: logging visibility, install-once.

Everything here runs without vLLM (the entry point returns ``False`` when vLLM is absent), because the
part that matters on a *bring-up* is what an operator can see: the banner the runbook greps, and the
fact that a second call into the same process changes nothing.
"""

from __future__ import annotations

import logging

import pytest

from ascend_platform.engine import plugin



# =============================================================================================
# the banner the runbook greps has to reach the log
# =============================================================================================


@pytest.fixture
def clean_logging():
    """Restore the process logging state: a test that installs a handler must not leak it."""
    package = logging.getLogger("ascend_platform")
    root = logging.getLogger()
    before = (list(package.handlers), package.level, package.propagate, list(root.handlers))
    made_visible = list(plugin._LOGS_MADE_VISIBLE)
    package.handlers.clear()
    root.handlers.clear()
    plugin._LOGS_MADE_VISIBLE[0] = False
    try:
        yield package, root
    finally:
        package.handlers[:] = before[0]
        package.setLevel(before[1])
        package.propagate = before[2]
        root.handlers[:] = before[3]
        plugin._LOGS_MADE_VISIBLE[0] = made_visible[0]


def test_the_destination_is_only_created_when_nothing_would_print(clean_logging, monkeypatch):
    """The predicate, without pretending pytest has no handler: pytest's own root handler means "no"."""
    assert plugin._logs_need_a_destination() is False
    monkeypatch.setattr(plugin, "_logs_need_a_destination", lambda: True)
    assert plugin._ensure_visible_logs() is True


def test_the_plugin_gives_its_banner_a_destination_under_vllm(clean_logging, monkeypatch):
    """vLLM puts its only handler on the ``vllm`` logger (``propagate: False``), so a third-party
    logger has no destination: INFO is dropped, WARNING+ goes to ``logging.lastResort``.  The
    runbook's first triage step greps for an INFO banner - it has to be printable."""
    package, _root = clean_logging
    monkeypatch.setattr(plugin, "_logs_need_a_destination", lambda: True)
    assert plugin._ensure_visible_logs() is True
    assert package.handlers, "the banner still has nowhere to go"
    assert package.level == logging.INFO
    assert package.propagate is False, "a propagating logger would print twice under a root handler"
    stream = package.handlers[0]
    assert isinstance(stream, logging.StreamHandler)


def test_the_log_fix_is_a_no_op_a_second_time(clean_logging, monkeypatch):
    package, _root = clean_logging
    monkeypatch.setattr(plugin, "_logs_need_a_destination", lambda: True)
    assert plugin._ensure_visible_logs() is True
    handlers = list(package.handlers)
    assert plugin._ensure_visible_logs() is False
    assert package.handlers == handlers, "a second call must not add a second handler"


def test_a_process_that_configured_logging_is_left_alone(clean_logging):
    """pytest, a user's ``dictConfig``, an operator's log file: none of them want our handler."""
    package, root = clean_logging
    root.addHandler(logging.NullHandler())
    assert plugin._logs_need_a_destination() is False
    assert plugin._ensure_visible_logs() is False
    assert not package.handlers


def test_the_log_level_follows_vllm_logging_level(clean_logging, monkeypatch):
    package, _root = clean_logging
    monkeypatch.setattr(plugin, "_logs_need_a_destination", lambda: True)
    monkeypatch.setenv("VLLM_LOGGING_LEVEL", "DEBUG")
    assert plugin._ensure_visible_logs() is True
    assert package.level == logging.DEBUG
    assert package.handlers[0].level == logging.DEBUG
