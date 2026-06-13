"""Content-policy gate for downloadable assets (models, LoRAs, adapters).

Two **orthogonal** axes that must never be conflated:

**Axis 1 — Real-person (ALWAYS blocked, every mode, non-negotiable).**
Zymera is for synthetic identities only (see CLAUDE.md "Responsible use").
Any asset that depicts, is named after, or is trained on a real, identifiable
person without consent is refused unconditionally. No mode or flag bypasses this.

**Axis 2 — SFW vs NSFW (mode-gated, a separate setting).**
NSFW of a *synthetic* persona is permitted by the policy — the ban is on real
people, not on nudity. NSFW is therefore controlled by an explicit opt-in
``content_mode`` ("sfw" default blocks NSFW; "nsfw" allows it). Enabling NSFW
mode only lifts the Axis-2 filter; it never touches Axis 1.

Authoritative signals, in order of trust:
1. Explicit boolean flags on the entry: ``real_person`` / ``poi`` (Civitai's
   "person of interest" flag) and ``nsfw``.
2. Keyword heuristics over name/tags/trigger-words/description — a safety net
   that catches obvious cases the flags miss. It cannot know every real name,
   which is exactly why a curated catalog + the explicit flags come first.
"""

from __future__ import annotations

from dataclasses import dataclass

# Axis 1: real-person indicators (lowercase substring match over the text blob).
REAL_PERSON_TERMS = (
    "celebrity",
    "celeb ",
    "real person",
    "real-person",
    "realperson",
    "real life person",
    "real-life person",
    "actor",
    "actress",
    "politician",
    "lookalike",
    "look-alike",
    "look alike",
    "likeness of",
    "deepfake",
    "deep fake",
    "famous person",
    "person of interest",
)

# Axis 2: NSFW indicators (only consulted for the SFW/NSFW decision).
NSFW_TERMS = (
    "nsfw",
    "explicit",
    "hardcore",
    "porn",
    "hentai",
    "nude",
    "nudity",
    "xxx",
    "adult content",
)


@dataclass(frozen=True)
class Decision:
    """Outcome of screening one asset entry."""

    allowed: bool
    axis: str | None  # "real_person" | "nsfw" | None
    reason: str


class PolicyError(RuntimeError):
    """Raised when a blocked asset is requested for download/use."""

    def __init__(self, decision: Decision, name: str = ""):
        self.decision = decision
        self.name = name
        prefix = f"Asset '{name}' blocked" if name else "Asset blocked"
        super().__init__(f"{prefix} [{decision.axis}]: {decision.reason}")


def _text_blob(entry: dict) -> str:
    """Flatten the searchable text of an entry to a single lowercase string."""
    parts: list[str] = []
    for key in ("name", "title", "description"):
        value = entry.get(key)
        if isinstance(value, str):
            parts.append(value)
    for key in ("tags", "trigger_words", "trained_words"):
        value = entry.get(key)
        if isinstance(value, (list, tuple)):
            parts.extend(str(v) for v in value)
        elif isinstance(value, str):
            parts.append(value)
    return " ".join(parts).lower()


class PolicyGate:
    """Screen asset entries against the two-axis content policy."""

    def __init__(self, content_mode: str = "sfw"):
        mode = (content_mode or "sfw").lower()
        if mode not in ("sfw", "nsfw"):
            raise ValueError(f"content_mode must be 'sfw' or 'nsfw', got {content_mode!r}")
        self.content_mode = mode

    @property
    def allow_nsfw(self) -> bool:
        return self.content_mode == "nsfw"

    def screen(self, entry: dict) -> Decision:
        """Return a Decision. Axis 1 (real-person) is evaluated first and is
        never overridable; Axis 2 (NSFW) depends on ``content_mode``."""
        blob = _text_blob(entry)

        # --- Axis 1: real-person — hard block, any mode ---
        if entry.get("real_person") or entry.get("poi"):
            return Decision(False, "real_person",
                            "flagged as depicting a real person (real_person/poi)")
        for term in REAL_PERSON_TERMS:
            if term in blob:
                return Decision(False, "real_person",
                                f"matched real-person indicator '{term.strip()}'")

        # --- Axis 2: NSFW — mode-gated ---
        is_nsfw = bool(entry.get("nsfw")) or any(term in blob for term in NSFW_TERMS)
        if is_nsfw and not self.allow_nsfw:
            return Decision(False, "nsfw",
                            "NSFW asset; set registry.content_mode=nsfw (or --nsfw) "
                            "to allow synthetic NSFW content")

        return Decision(True, None, "allowed")

    def check(self, entry: dict) -> None:
        """Screen and raise PolicyError if blocked."""
        decision = self.screen(entry)
        if not decision.allowed:
            raise PolicyError(decision, entry.get("name", ""))

    def filter(self, entries: list[dict]) -> list[dict]:
        """Return only the entries that pass screening (used for search)."""
        return [e for e in entries if self.screen(e).allowed]
