"""Catalog merge/resolve/search and AssetManager policy gating."""

import json

import pytest

from zymera.registry.catalog import BUILTIN_ASSETS, Catalog
from zymera.registry.manager import AssetManager
from zymera.registry.policy import PolicyError


def test_builtin_catalog_loads():
    cat = Catalog()
    assert "sdxl-base" in cat.names()
    assert "lcm-lora-sdxl" in cat.names()
    entry = cat.resolve("sdxl-base")
    assert entry["name"] == "sdxl-base"
    assert entry["family"] == "sdxl"


def test_user_registry_merges(tmp_path):
    path = tmp_path / "registry.json"
    path.write_text(json.dumps({"assets": {
        "my-lora": {"type": "lora", "source": "hf", "repo": "me/lora", "family": "sdxl"},
        "sdxl-base": {"vram_gb": 9.9},  # override one field of a built-in
    }}))
    cat = Catalog.load(path)
    assert "my-lora" in cat.names()
    assert cat.resolve("my-lora")["repo"] == "me/lora"
    # deep-merge keeps untouched built-in fields
    base = cat.resolve("sdxl-base")
    assert base["vram_gb"] == 9.9
    assert base["repo"] == BUILTIN_ASSETS["sdxl-base"]["repo"]


def test_resolve_unknown_raises():
    with pytest.raises(KeyError):
        Catalog().resolve("nope")


def test_search_by_family_type_tags():
    cat = Catalog()
    loras = cat.search(type="lora", family="sdxl")
    assert all(e["type"] == "lora" and e["family"] == "sdxl" for e in loras)
    speed = cat.search(type="lora", tags=["lcm"])
    assert any(e["name"] == "lcm-lora-sdxl" for e in speed)


def test_manager_blocks_real_person(tmp_path):
    cat = Catalog({"bad": {"type": "lora", "source": "hf", "repo": "x/y", "poi": True}})
    mgr = AssetManager(catalog=cat, assets_dir=tmp_path, content_mode="nsfw")
    with pytest.raises(PolicyError):
        mgr.ensure("bad")


def test_manager_hf_without_weight_returns_repo(tmp_path):
    # No weight_name → diffusers resolves from the repo id; no download here.
    cat = Catalog()
    mgr = AssetManager(catalog=cat, assets_dir=tmp_path)
    assert mgr.ensure("lcm-lora-sdxl") == "latent-consistency/lcm-lora-sdxl"


def test_manager_unknown_source(tmp_path):
    cat = Catalog({"weird": {"type": "lora", "source": "ftp", "repo": "x"}})
    mgr = AssetManager(catalog=cat, assets_dir=tmp_path)
    with pytest.raises(ValueError, match="unknown source"):
        mgr.ensure("weird")
