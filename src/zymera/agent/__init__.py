"""Optional Claude-powered planner/executor (the ``zymera auto`` command).

Thin reasoning layer over the deterministic Phase-A building blocks. Active only
when ``ANTHROPIC_API_KEY`` is set and the ``anthropic`` SDK is installed; the
heuristic planner (:mod:`zymera.planner`) is the always-available fallback.

The "multi-agent pattern" is: a Planner (Claude or heuristic) emits a
``GenerationPlan``; an Executor consumes that same contract to screen + download
assets, save a recipe, and optionally run generation. The PolicyGate guards
every asset the agent can see or fetch.
"""

from __future__ import annotations

from zymera.agent.run import run_auto

__all__ = ["run_auto"]
