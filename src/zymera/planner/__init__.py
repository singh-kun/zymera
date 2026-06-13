"""Planning: turn a natural-language requirement into a GenerationPlan.

``heuristic.plan`` is the deterministic, no-LLM path (also the fallback for the
optional Claude-powered planner in :mod:`zymera.agent`). Both paths share the
:class:`GenerationPlan` contract.
"""

from __future__ import annotations

from zymera.planner.heuristic import plan
from zymera.planner.types import GenerationPlan, section_for_phase

__all__ = ["plan", "GenerationPlan", "section_for_phase"]
