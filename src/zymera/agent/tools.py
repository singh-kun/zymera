"""Agent toolset: Anthropic tool schemas + the deterministic functions behind them.

The planner agent is given two tools:
- ``search_assets`` — query the (already policy-filtered) asset catalog. The LLM
  can only ever see assets that pass the content policy, so it cannot propose a
  blocked one.
- ``submit_plan`` — emit the final ``GenerationPlan`` (schema mirrors the
  dataclass). The loop ends when this is called.

Live Civitai/HF *search* is intentionally not exposed to the LLM: the catalog
(curated + the user's ``configs/registry.json``) is the safe surface. Civitai
assets are fetched by ``model_id`` only when a catalog entry exists, so every
download still passes ``PolicyGate`` with authoritative metadata.
"""

from __future__ import annotations

from typing import Any

# ---- Tool: search_assets -------------------------------------------------

SEARCH_ASSETS_TOOL: dict[str, Any] = {
    "name": "search_assets",
    "description": (
        "Search the asset catalog for base models, LoRAs, VAEs, and adapters. "
        "Results are pre-filtered by the content policy (real-person assets are "
        "always excluded; NSFW only appears when content_mode=nsfw). Use this to "
        "find LoRAs compatible with the chosen base-model family before planning."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Free-text query (name/repo/tags)"},
            "family": {"type": "string", "enum": ["sdxl", "sd15"],
                       "description": "Restrict to a base-model family"},
            "type": {"type": "string",
                     "enum": ["checkpoint", "lora", "vae", "ip_adapter", "motion_adapter"]},
        },
        "additionalProperties": False,
    },
}

# ---- Tool: submit_plan (mirrors GenerationPlan) --------------------------

SUBMIT_PLAN_TOOL: dict[str, Any] = {
    "name": "submit_plan",
    "description": "Submit the final generation plan. Call exactly once when ready.",
    "input_schema": {
        "type": "object",
        "properties": {
            "phase": {"type": "string", "enum": ["phase1", "phase2", "phase3", "phase4"],
                      "description": "phase1=image, phase2=identity image, phase3=video, phase4=talking video"},
            "preset": {"type": "string",
                       "description": "Base preset: balanced|quality|fast|low_vram|quantized|sd15"},
            "style": {"type": "string", "description": "Prompt style name, or omit for none"},
            "loras": {
                "type": "array",
                "description": "LoRA adapters to apply (catalog names + scale)",
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "scale": {"type": "number"},
                    },
                    "required": ["name"],
                    "additionalProperties": False,
                },
            },
            "param_overrides": {
                "type": "object",
                "description": "Dot-path config overrides, e.g. {\"image.steps\": 40}",
                "additionalProperties": True,
            },
            "rationale": {"type": "string", "description": "One-line why this plan fits"},
        },
        "required": ["phase", "preset", "rationale"],
        "additionalProperties": False,
    },
}

PLANNER_TOOLS = [SEARCH_ASSETS_TOOL, SUBMIT_PLAN_TOOL]


def run_search_assets(catalog, gate, tool_input: dict) -> list[dict]:
    """Execute a ``search_assets`` tool call: catalog search + policy filter."""
    results = catalog.search(
        query=tool_input.get("query"),
        family=tool_input.get("family"),
        type=tool_input.get("type"),
    )
    allowed = gate.filter(results)
    # Trim to the fields the planner needs (keeps tokens + context tight).
    return [
        {k: e.get(k) for k in ("name", "type", "family", "vram_gb", "tags") if k in e}
        for e in allowed
    ]


def plan_from_submit(requirement: str, tool_input: dict):
    """Build a GenerationPlan from a ``submit_plan`` tool call's input."""
    from zymera.planner.types import GenerationPlan

    loras = [
        {"name": l["name"], "scale": float(l.get("scale", 1.0))}
        for l in (tool_input.get("loras") or [])
        if l.get("name")
    ]
    return GenerationPlan(
        requirement=requirement,
        phase=tool_input.get("phase", "phase1"),
        preset=tool_input.get("preset"),
        style=tool_input.get("style") or None,
        loras=loras,
        param_overrides=tool_input.get("param_overrides") or {},
        assets=[l["name"] for l in loras],
        rationale=tool_input.get("rationale", ""),
    )
