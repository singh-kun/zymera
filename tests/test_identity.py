import pytest

from zymera.identity import IdentityStore


def _make_image(path):
    from PIL import Image

    Image.new("RGB", (64, 64), color=(120, 90, 200)).save(path)
    return path


def test_create_load_list(tmp_path):
    ref = _make_image(tmp_path / "ref.png")
    store = IdentityStore(tmp_path / "identities")

    created = store.create("persona_a", [ref], attributes={"note": "fully synthetic"})
    assert created.reference_images == ["ref_0.png"]

    loaded = store.load("persona_a")
    assert loaded.identity_id == "persona_a"
    assert loaded.attributes == {"note": "fully synthetic"}
    assert loaded.primary_reference().is_file()
    assert store.list_ids() == ["persona_a"]


def test_create_without_images(tmp_path):
    store = IdentityStore(tmp_path)
    identity = store.create("prompt_only")
    assert identity.reference_images == []
    assert identity.primary_reference() is None


def test_add_reference(tmp_path):
    store = IdentityStore(tmp_path)
    store.create("persona_b")
    store.add_reference("persona_b", _make_image(tmp_path / "extra.jpg"))
    assert store.load("persona_b").reference_images == ["ref_0.jpg"]


def test_load_missing_raises_with_known_ids(tmp_path):
    store = IdentityStore(tmp_path)
    store.create("existing")
    with pytest.raises(KeyError, match="existing"):
        store.load("ghost")


def test_missing_image_raises(tmp_path):
    store = IdentityStore(tmp_path)
    with pytest.raises(FileNotFoundError):
        store.create("bad", [tmp_path / "nope.jpg"])
