"""Identity management: metadata + reference images on disk.

Identities must be synthetic personas or people who have given explicit
consent for their likeness to be used. See README "Responsible use".
"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass, field
from pathlib import Path

METADATA_FILE = "metadata.json"
IMAGES_DIR = "images"


@dataclass
class Identity:
    identity_id: str
    root: Path
    reference_images: list[str] = field(default_factory=list)
    attributes: dict = field(default_factory=dict)

    def reference_paths(self) -> list[Path]:
        return [self.root / IMAGES_DIR / name for name in self.reference_images]

    def primary_reference(self) -> Path | None:
        for path in self.reference_paths():
            if path.is_file():
                return path
        return None


class IdentityStore:
    def __init__(self, root: str | Path):
        self.root = Path(root)

    def create(
        self,
        identity_id: str,
        image_paths: list[str | Path] | None = None,
        attributes: dict | None = None,
    ) -> Identity:
        identity_dir = self.root / identity_id
        images_dir = identity_dir / IMAGES_DIR
        images_dir.mkdir(parents=True, exist_ok=True)

        names: list[str] = []
        for i, src in enumerate(image_paths or []):
            src = Path(src)
            if not src.is_file():
                raise FileNotFoundError(f"Reference image not found: {src}")
            name = f"ref_{i}{src.suffix.lower() or '.jpg'}"
            shutil.copy(src, images_dir / name)
            names.append(name)

        identity = Identity(identity_id, identity_dir, names, attributes or {})
        self._write_metadata(identity)
        return identity

    def add_reference(self, identity_id: str, image_path: str | Path) -> Identity:
        identity = self.load(identity_id)
        src = Path(image_path)
        if not src.is_file():
            raise FileNotFoundError(f"Reference image not found: {src}")
        name = f"ref_{len(identity.reference_images)}{src.suffix.lower() or '.jpg'}"
        (identity.root / IMAGES_DIR).mkdir(parents=True, exist_ok=True)
        shutil.copy(src, identity.root / IMAGES_DIR / name)
        identity.reference_images.append(name)
        self._write_metadata(identity)
        return identity

    def load(self, identity_id: str) -> Identity:
        metadata_path = self.root / identity_id / METADATA_FILE
        if not metadata_path.is_file():
            known = ", ".join(self.list_ids()) or "(none)"
            raise KeyError(f"Identity '{identity_id}' not found. Known identities: {known}")
        meta = json.loads(metadata_path.read_text(encoding="utf-8"))
        return Identity(
            identity_id=meta.get("identity_id", identity_id),
            root=self.root / identity_id,
            reference_images=meta.get("reference_images", []),
            attributes=meta.get("attributes", {}),
        )

    def list_ids(self) -> list[str]:
        if not self.root.is_dir():
            return []
        return sorted(p.parent.name for p in self.root.glob(f"*/{METADATA_FILE}"))

    def _write_metadata(self, identity: Identity) -> None:
        payload = {
            "identity_id": identity.identity_id,
            "reference_images": identity.reference_images,
            "attributes": identity.attributes,
        }
        (identity.root / METADATA_FILE).write_text(
            json.dumps(payload, indent=2), encoding="utf-8"
        )
