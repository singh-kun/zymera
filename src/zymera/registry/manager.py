"""AssetManager: resolve a catalog name to a local path, downloading on demand.

Sources:
- ``hf``      — HuggingFace Hub via ``huggingface_hub`` (already a dependency).
- ``civitai`` — Civitai REST API (``requests``); honours ``CIVITAI_API_KEY`` for
                gated/authenticated files.

Every resolution is screened by :class:`~zymera.registry.policy.PolicyGate`
*before* any bytes are fetched. Civitai downloads are re-screened against live
API metadata (``poi`` / ``nsfw``) as defence-in-depth against a stale catalog.

Heavy/optional imports (``huggingface_hub``, ``requests``) live inside methods,
per project convention.
"""

from __future__ import annotations

import hashlib
import logging
import os
from pathlib import Path
from typing import Any

from zymera.registry.catalog import Catalog
from zymera.registry.policy import PolicyError, PolicyGate

log = logging.getLogger(__name__)

CIVITAI_API = "https://civitai.com/api/v1"
DEFAULT_ASSETS_DIR = Path.home() / ".cache" / "zymera" / "assets"


class AssetManager:
    """Download + cache assets, gated by content policy."""

    def __init__(
        self,
        catalog: Catalog | None = None,
        assets_dir: str | Path | None = None,
        content_mode: str = "sfw",
        gate: PolicyGate | None = None,
    ):
        self.catalog = catalog or Catalog.load()
        self.assets_dir = Path(assets_dir) if assets_dir else DEFAULT_ASSETS_DIR
        self.gate = gate or PolicyGate(content_mode)

    @classmethod
    def from_config(cls, cfg) -> "AssetManager":
        catalog = Catalog.load(cfg.get("paths.registry_file"))
        return cls(
            catalog=catalog,
            assets_dir=cfg.get("paths.assets_dir") or DEFAULT_ASSETS_DIR,
            content_mode=cfg.get("registry.content_mode", "sfw"),
        )

    # ------------------------------------------------------------------ resolve
    def _entry(self, name_or_entry: str | dict) -> dict[str, Any]:
        if isinstance(name_or_entry, dict):
            return name_or_entry
        return self.catalog.resolve(name_or_entry)

    def ensure(self, name_or_entry: str | dict) -> str:
        """Return a local path (or HF repo id) usable by diffusers loaders,
        downloading if needed. Raises :class:`PolicyError` if blocked."""
        entry = self._entry(name_or_entry)
        self.gate.check(entry)
        source = entry.get("source")
        if source == "hf":
            return self._ensure_hf(entry)
        if source == "civitai":
            return str(self._ensure_civitai(entry))
        raise ValueError(
            f"Asset '{entry.get('name', '?')}' has unknown source {source!r} "
            "(expected 'hf' or 'civitai')"
        )

    # ----------------------------------------------------------------- HF source
    def _ensure_hf(self, entry: dict) -> str:
        repo = entry.get("repo")
        if not repo:
            raise ValueError(f"hf asset '{entry.get('name', '?')}' is missing 'repo'")
        weight_name = entry.get("weight_name")
        if not weight_name:
            # No single file specified: let diffusers resolve from the repo id.
            # (Used for full pipelines and standard-layout LoRA repos.)
            return repo
        from huggingface_hub import hf_hub_download

        path = hf_hub_download(
            repo_id=repo,
            filename=weight_name,
            subfolder=entry.get("subfolder"),
            cache_dir=str(self.assets_dir / "hf"),
        )
        self._verify_sha256(Path(path), entry.get("sha256"), entry.get("name", repo))
        return path

    # ------------------------------------------------------------ Civitai source
    def _ensure_civitai(self, entry: dict) -> Path:
        import requests

        model_id = entry.get("model_id")
        if not model_id:
            raise ValueError(
                f"civitai asset '{entry.get('name', '?')}' is missing 'model_id'"
            )
        headers = {}
        token = os.environ.get("CIVITAI_API_KEY")
        if token:
            headers["Authorization"] = f"Bearer {token}"

        meta = requests.get(
            f"{CIVITAI_API}/models/{model_id}", headers=headers, timeout=30
        )
        meta.raise_for_status()
        data = meta.json()

        # Re-screen with live metadata before downloading anything.
        live = {
            **entry,
            "poi": data.get("poi", entry.get("poi")),
            "nsfw": data.get("nsfw", entry.get("nsfw")),
            "tags": list(data.get("tags", [])) + list(entry.get("tags", [])),
            "description": data.get("description") or entry.get("description", ""),
        }
        decision = self.gate.screen(live)
        if not decision.allowed:
            raise PolicyError(decision, entry.get("name", str(model_id)))

        version = self._pick_version(data, entry.get("version_id"))
        file_info = self._pick_file(version)
        url = file_info["downloadUrl"]
        filename = file_info.get("name", f"{model_id}.safetensors")
        dest = self.assets_dir / "civitai" / str(model_id) / filename
        if dest.is_file():
            log.debug("Civitai asset cached: %s", dest)
            return dest

        dest.parent.mkdir(parents=True, exist_ok=True)
        log.info("Downloading Civitai asset '%s' -> %s", entry.get("name", model_id), dest)
        self._stream_download(url, dest, headers)
        expected = (file_info.get("hashes") or {}).get("SHA256")
        self._verify_sha256(dest, expected, entry.get("name", filename))
        return dest

    @staticmethod
    def _pick_version(data: dict, version_id: int | None) -> dict:
        versions = data.get("modelVersions") or []
        if not versions:
            raise ValueError(f"Civitai model {data.get('id')} has no versions")
        if version_id is not None:
            for v in versions:
                if v.get("id") == version_id:
                    return v
            raise ValueError(f"Civitai version {version_id} not found")
        return versions[0]

    @staticmethod
    def _pick_file(version: dict) -> dict:
        files = version.get("files") or []
        if not files:
            raise ValueError("Civitai version has no downloadable files")
        # Prefer the primary file, else the first.
        for f in files:
            if f.get("primary"):
                return f
        return files[0]

    @staticmethod
    def _stream_download(url: str, dest: Path, headers: dict) -> None:
        import requests

        with requests.get(url, headers=headers, stream=True, timeout=120) as resp:
            resp.raise_for_status()
            tmp = dest.with_suffix(dest.suffix + ".part")
            with open(tmp, "wb") as fh:
                for chunk in resp.iter_content(chunk_size=1 << 20):
                    if chunk:
                        fh.write(chunk)
            tmp.replace(dest)

    @staticmethod
    def _verify_sha256(path: Path, expected: str | None, name: str) -> None:
        if not expected:
            return
        h = hashlib.sha256()
        with open(path, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
        actual = h.hexdigest()
        if actual.lower() != expected.lower():
            path.unlink(missing_ok=True)
            raise ValueError(
                f"SHA256 mismatch for asset '{name}': expected {expected}, got {actual}"
            )
