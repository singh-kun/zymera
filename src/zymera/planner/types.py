"""The GenerationPlan contract shared by the heuristic and LLM planners."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

# Which config section a phase's model/lora/params live under.
_SECTION_FOR_PHASE = {
    "phase1": "image", "image": "image",
    "phase2": "identity_image", "identity-image": "identity_image",
    "identity_image": "identity_image",
    "phase3": "video", "video": "video",
    "phase4": "video", "talking-video": "video", "talking_video": "video",
}


def section_for_phase(phase: str) -> str:
    return _SECTION_FOR_PHASE.get(phase, "image")


@dataclass
class GenerationPlan:
    """A concrete, executable generation plan.

    ``loras`` entries are ``{"name": <catalog-name-or-repo>, "scale": float,
    "weight_name": str | None}``. ``assets`` lists catalog names the executor
    must ``AssetManager.ensure()`` before running (LoRAs, custom checkpoints).
    """

    requirement: str
    phase: str = "phase1"
    preset: str | None = None
    style: str | None = None
    base_model: str | None = None
    base_family: str | None = None
    loras: list[dict[str, Any]] = field(default_factory=list)
    param_overrides: dict[str, Any] = field(default_factory=dict)
    assets: list[str] = field(default_factory=list)
    rationale: str = ""

    def to_config_overrides(self) -> dict[str, Any]:
        """Flatten the plan into dot-path config overrides (for ``Config.set``)."""
        ov: dict[str, Any] = dict(self.param_overrides)
        if self.style:
            ov["prompt.style"] = self.style
        section = section_for_phase(self.phase)
        if self.base_model:
            ov[f"{section}.model"] = self.base_model
            if self.base_family:
                ov[f"{section}.family"] = self.base_family
        if self.loras:
            ov[f"{section}.lora.enabled"] = True
            ov[f"{section}.lora.adapters"] = self.loras
        return ov

    def apply_to(self, cfg) -> None:
        for key, value in self.to_config_overrides().items():
            cfg.set(key, value)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "GenerationPlan":
        fields = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in data.items() if k in fields})
