"""Agent layer: tool schemas, search filtering, and the planner loop with a
mocked Anthropic client (no network, no token spend)."""

import json
from types import SimpleNamespace

from zymera.agent import tools
from zymera.agent.planner_agent import plan_with_claude
from zymera.config import Config
from zymera.planner.types import GenerationPlan
from zymera.registry.catalog import Catalog
from zymera.registry.policy import PolicyGate


def test_tool_schemas_wellformed():
    for tool in tools.PLANNER_TOOLS:
        assert "name" in tool and "input_schema" in tool
        assert tool["input_schema"]["type"] == "object"


def test_search_assets_filters_policy():
    cat = Catalog({"bad-celeb": {"type": "lora", "source": "hf", "repo": "x/y",
                                 "family": "sdxl", "poi": True}})
    gate = PolicyGate("sfw")
    results = tools.run_search_assets(cat, gate, {"type": "lora", "family": "sdxl"})
    names = {r["name"] for r in results}
    assert "bad-celeb" not in names          # real-person filtered out
    assert "lcm-lora-sdxl" in names           # built-in SFW lora present


def test_plan_from_submit_builds_plan():
    plan = tools.plan_from_submit("anime portrait", {
        "phase": "phase1", "preset": "low_vram", "style": "anime",
        "loras": [{"name": "lcm-lora-sd15", "scale": 0.7}],
        "rationale": "fits 6GB",
    })
    assert isinstance(plan, GenerationPlan)
    assert plan.phase == "phase1"
    assert plan.loras[0]["name"] == "lcm-lora-sd15"
    assert plan.assets == ["lcm-lora-sd15"]


class _Block:
    def __init__(self, **kw):
        self.__dict__.update(kw)


class _FakeMessages:
    """Returns a search_assets call, then a submit_plan call."""

    def __init__(self):
        self.calls = 0

    def create(self, **kwargs):
        self.calls += 1
        if self.calls == 1:
            block = _Block(type="tool_use", name="search_assets", id="t1",
                           input={"type": "lora", "family": "sdxl"})
        else:
            block = _Block(type="tool_use", name="submit_plan", id="t2",
                           input={"phase": "phase1", "preset": "balanced",
                                  "rationale": "test"})
        return SimpleNamespace(content=[block])


class _FakeClient:
    def __init__(self):
        self.messages = _FakeMessages()


def test_plan_with_claude_loop(monkeypatch):
    cfg = Config()
    catalog = Catalog()
    profile = SimpleNamespace(summary=lambda: "Test GPU", tier="medium",
                              has_bitsandbytes=False)
    plan = plan_with_claude("an sdxl portrait", profile, catalog, cfg,
                            client=_FakeClient())
    assert isinstance(plan, GenerationPlan)
    assert plan.phase == "phase1"
    assert plan.preset == "balanced"
    assert plan.requirement == "an sdxl portrait"
