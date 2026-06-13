"""CapabilityProfile tiering and graceful probing without a GPU."""

from zymera.capabilities import CapabilityProfile, probe


def _profile(vram, cuda=True, bnb=False):
    return CapabilityProfile(has_cuda=cuda, vram_gb=vram, has_bitsandbytes=bnb,
                             gpu_name="Test GPU")


def test_tier_boundaries():
    assert _profile(0, cuda=False).tier == "cpu"
    assert _profile(4).tier == "tiny"
    assert _profile(6).tier == "small"
    assert _profile(8).tier == "medium"
    assert _profile(16).tier == "large"


def test_summary_strings():
    assert "CPU only" in _profile(0, cuda=False).summary()
    assert "tier=small" in _profile(6).summary()


def test_probe_runs_without_crashing():
    # On a CPU-only CI box this must still return a usable profile.
    prof = probe()
    assert isinstance(prof, CapabilityProfile)
    assert prof.tier in ("cpu", "tiny", "small", "medium", "large")
