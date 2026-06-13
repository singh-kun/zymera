"""Asset registry: catalog, content-policy gate, and downloader.

Public surface:
- ``Catalog``        — named model/LoRA/adapter entries (built-in + user JSON).
- ``PolicyGate``     — two-axis content screening (real-person + SFW/NSFW).
- ``Decision``       — result of a policy screen.
- ``PolicyError``    — raised when a blocked asset is requested.
- ``AssetManager``   — resolves a catalog name to a local path, downloading on
                       demand from HuggingFace or Civitai, policy-gated.
"""

from __future__ import annotations

from zymera.registry.catalog import Catalog
from zymera.registry.manager import AssetManager
from zymera.registry.policy import Decision, PolicyError, PolicyGate

__all__ = ["Catalog", "AssetManager", "PolicyGate", "Decision", "PolicyError"]
