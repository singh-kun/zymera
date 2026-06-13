"""Content-policy gate: the two-axis (real-person / NSFW) screening rules.

These tests encode the non-negotiable responsible-use policy (CLAUDE.md).
"""

import pytest

from zymera.registry.policy import Decision, PolicyError, PolicyGate


def test_real_person_blocked_in_both_modes():
    # Axis 1 is never bypassable — neither mode allows real people.
    for mode in ("sfw", "nsfw"):
        gate = PolicyGate(mode)
        by_flag = gate.screen({"name": "x", "poi": True})
        by_term = gate.screen({"name": "celebrity lookalike lora", "tags": ["celebrity"]})
        assert by_flag.allowed is False and by_flag.axis == "real_person", mode
        assert by_term.allowed is False and by_term.axis == "real_person", mode


def test_explicit_real_person_flag_blocked():
    gate = PolicyGate("nsfw")
    assert gate.screen({"name": "synthetic", "real_person": True}).allowed is False


def test_nsfw_blocked_in_sfw_allowed_in_nsfw():
    entry = {"name": "synthetic-persona-lora", "nsfw": True}
    assert PolicyGate("sfw").screen(entry).allowed is False
    assert PolicyGate("sfw").screen(entry).axis == "nsfw"
    assert PolicyGate("nsfw").screen(entry).allowed is True


def test_nsfw_keyword_detected():
    entry = {"name": "tasteful art", "tags": ["nude", "explicit"]}
    assert PolicyGate("sfw").screen(entry).allowed is False


def test_sfw_synthetic_always_allowed():
    entry = {"name": "detail-tweaker", "tags": ["quality", "detail"], "family": "sdxl"}
    assert PolicyGate("sfw").screen(entry) == Decision(True, None, "allowed")
    assert PolicyGate("nsfw").screen(entry).allowed is True


def test_nsfw_mode_does_not_unlock_real_person():
    # The critical invariant: NSFW mode lifts only axis 2, never axis 1.
    entry = {"name": "actor nsfw lora", "nsfw": True, "tags": ["actor"]}
    decision = PolicyGate("nsfw").screen(entry)
    assert decision.allowed is False
    assert decision.axis == "real_person"


def test_check_raises_policy_error():
    gate = PolicyGate("sfw")
    with pytest.raises(PolicyError):
        gate.check({"name": "celebrity", "poi": True})


def test_filter_drops_blocked():
    gate = PolicyGate("sfw")
    entries = [
        {"name": "ok", "tags": ["quality"]},
        {"name": "bad-celeb", "poi": True},
        {"name": "nsfw-one", "nsfw": True},
    ]
    kept = gate.filter(entries)
    assert [e["name"] for e in kept] == ["ok"]


def test_invalid_mode_rejected():
    with pytest.raises(ValueError):
        PolicyGate("maybe")
