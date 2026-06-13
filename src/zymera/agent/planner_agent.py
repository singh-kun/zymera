"""Claude-powered planner: requirement + capabilities + catalog -> GenerationPlan.

Runs a manual tool-use loop (Anthropic Python SDK). Claude may call
``search_assets`` to explore the catalog, then calls ``submit_plan`` to emit the
final plan. Shares the ``GenerationPlan`` contract with the heuristic planner.
"""

from __future__ import annotations

import json
import logging

from zymera.agent.tools import PLANNER_TOOLS, plan_from_submit, run_search_assets

log = logging.getLogger(__name__)

_SYSTEM = """You are Zymera's generation planner. Given a user's requirement, the \
detected GPU capabilities, and an asset catalog, choose the best phase, preset, \
prompt style, and any LoRA adapters, then call submit_plan.

Rules:
- phase1=image, phase2=identity image, phase3=video, phase4=talking video. Infer \
from the requirement (video/animation -> phase3; talking/speaking -> phase4; \
"same face"/identity -> phase2; otherwise phase1).
- Pick a preset by VRAM tier: tiny(<6GB)->quantized if bitsandbytes else low_vram; \
small(6-8GB)->low_vram; medium(8-12GB)->balanced; large(12GB+)->quality; cpu->sd15.
- Only add a LoRA when the requirement clearly calls for it, and only one whose \
family matches the base model the chosen phase+preset will use (phase3/4 and the \
VRAM presets for phase2 are sd15; phase1 is sdxl unless preset=sd15). Use \
search_assets to find compatible LoRAs.
- Responsible use is NON-NEGOTIABLE: this tool is for synthetic identities only. \
The catalog you can search already excludes real-person assets; never try to \
target real, identifiable people.
- Keep param_overrides minimal. Provide a one-line rationale."""

_DEFAULT_MODEL = "claude-opus-4-8"


def is_available() -> bool:
    """True when the Anthropic SDK is importable and an API key is set."""
    import os

    if not os.environ.get("ANTHROPIC_API_KEY"):
        return False
    try:
        import anthropic  # noqa: F401
    except ImportError:
        return False
    return True


def plan_with_claude(requirement, profile, catalog, cfg, client=None):
    """Produce a GenerationPlan via Claude. Raises on unrecoverable failure so the
    caller can fall back to the heuristic planner."""
    from zymera.registry.policy import PolicyGate

    if client is None:
        import anthropic

        client = anthropic.Anthropic()
    model = cfg.get("agent.model", _DEFAULT_MODEL)
    max_tokens = cfg.get("agent.max_tokens", 4096)
    gate = PolicyGate(cfg.get("registry.content_mode", "sfw"))

    user_text = (
        f"Requirement: {requirement}\n\n"
        f"Capabilities: {profile.summary()}; tier={profile.tier}; "
        f"bitsandbytes={'yes' if profile.has_bitsandbytes else 'no'}.\n\n"
        f"Catalog (searchable): {len(catalog.names())} assets. "
        "Use search_assets to inspect LoRAs by family/type."
    )
    messages = [{"role": "user", "content": user_text}]

    for _ in range(8):  # bounded loop; submit_plan normally ends it in 1-2 rounds
        response = client.messages.create(
            model=model,
            max_tokens=max_tokens,
            system=_SYSTEM,
            tools=PLANNER_TOOLS,
            messages=messages,
        )
        tool_uses = [b for b in response.content if b.type == "tool_use"]
        if not tool_uses:
            # Nudge the model to use a tool rather than chatting.
            messages.append({"role": "assistant", "content": response.content})
            messages.append({"role": "user", "content": "Call submit_plan now."})
            continue

        messages.append({"role": "assistant", "content": response.content})
        results = []
        plan = None
        for tu in tool_uses:
            if tu.name == "submit_plan":
                plan = plan_from_submit(requirement, tu.input)
                results.append({"type": "tool_result", "tool_use_id": tu.id,
                                "content": "plan accepted"})
            elif tu.name == "search_assets":
                found = run_search_assets(catalog, gate, tu.input)
                results.append({"type": "tool_result", "tool_use_id": tu.id,
                                "content": json.dumps(found)})
            else:
                results.append({"type": "tool_result", "tool_use_id": tu.id,
                                "content": f"unknown tool {tu.name}", "is_error": True})
        if plan is not None:
            log.debug("Claude planner produced: %s", plan.rationale)
            return plan
        messages.append({"role": "user", "content": results})

    raise RuntimeError("planner did not submit a plan within the loop budget")
