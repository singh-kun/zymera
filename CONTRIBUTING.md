# Contributing to Zymera

## Development setup

```bash
git clone https://github.com/singh-kun/zymera.git
cd zymera
pip install torch --index-url https://download.pytorch.org/whl/cu130
pip install -r requirements.txt
pip install -e .
python -m pytest tests -q   # all tests should pass without a GPU
```

## Conventions (from CLAUDE.md)

- Heavy imports (`torch`, `diffusers`, `TTS`, `PIL`) go inside functions, never
  at module top level in `config/prompts/identity/cli/pipeline`.
- Every new tunable gets a key in `DEFAULTS` in `config.py` with a sensible
  default. Expose as a CLI flag only if it is used frequently (`--set` covers
  the rest).
- Stage contract: set `section`, implement `_load()` returning the pipeline,
  implement `run(...)` returning raw artifacts (PIL images / frames). All file
  IO belongs in `pipeline.py`.
- Video models must be SD1.5-family — AnimateDiff is incompatible with SDXL.
- VRAM-limited presets (`fast`, `low_vram`, `quantized`) must keep phase2 on
  SD1.5 + img2img. A unit test in `tests/test_config.py` guards this.
- Never add `flash-attn` as a dependency — it won't build on Windows. PyTorch
  SDPA already dispatches flash kernels on Ampere/Ada automatically.
- Downloadable assets (models/LoRAs/adapters) belong in the catalog
  (`zymera.registry.catalog` built-ins or `configs/registry.json`), never
  hardcoded in a stage. Every download must go through `AssetManager`, which
  screens it with `PolicyGate` first.
- The agent layer is a thin wrapper over deterministic functions — anything
  `zymera auto` can do must also work without an LLM (the heuristic planner).
  Keep the single model id in `agent.model` (default `claude-opus-4-8`).

## Adding a stage

1. Create `src/zymera/stages/<name>.py` with a class inheriting `base.Stage`.
2. Register it in `src/zymera/stages/__init__.py`.
3. Add the phase mapping in `pipeline.py`.
4. Add a config section and `DEFAULTS` entry in `config.py`.

## Responsible use (non-negotiable)

Zymera is for **synthetic identities only** — fully AI-generated personas or
people who have given explicit consent. Pull requests that add code, examples,
prompts, tests, or identities targeting real people without consent (celebrities
included) will be rejected.

The `PolicyGate` (`src/zymera/registry/policy.py`) enforces this on every asset
download along two independent axes: **real-person content is always blocked**
(any mode), and **NSFW is a separate `content_mode` opt-in** that only permits
NSFW of synthetic personas. Do not weaken axis 1, and never let the NSFW toggle
affect it — `tests/test_policy.py` guards this.

## Pull request checklist

- [ ] `python -m pytest tests -q` passes
- [ ] New tunables are in `DEFAULTS` with a sensible default
- [ ] Heavy imports are inside functions (not module top level)
- [ ] No hardcoded generation parameters outside `config.py`
- [ ] No real people used as identity references
