# Zymera

Configurable multi-stage pipeline for synthetic identity media: images (phase1),
identity-conditioned images (phase2), short videos (phase3), and talking videos
with speech + lip sync (phase4).

## Commands

```bash
pip install -r requirements.txt && pip install -e .   # install
python -m pytest tests -q                             # unit tests (no GPU needed)
zymera doctor                                         # environment check (GPU, deps, ffmpeg)
zymera styles                                         # list prompt styles
zymera identity create <id> --images ref.jpg          # create an identity
zymera generate --phase phase1 --prompt "..."         # generate (also: python -m zymera)
python scripts/smoke_test.py                          # GPU end-to-end smoke test (slow, downloads models)
```

## Architecture

- `src/zymera/config.py` — layered config: built-in `DEFAULTS` <- `configs/default.json` <- `--preset` <- `--set key=value`. All tunables live here; never hardcode generation parameters elsewhere.
- `src/zymera/prompts.py` — `PromptBuilder`: style prefixes/suffixes + curated negative prompts. Data-driven via `configs/prompts.json` (deep-merged over built-ins).
- `src/zymera/identity.py` — `IdentityStore`: `identities/<id>/metadata.json` + `images/ref_*.jpg`.
- `src/zymera/stages/` — one file per stage; all inherit `base.Stage` (lazy model load, device/dtype resolution, VRAM optimizations). Registry with lazy imports in `stages/__init__.py` so torch only loads when generating.
- `src/zymera/pipeline.py` — orchestrator: `PHASES` maps phase names to runners; owns seeds, output naming, and JSON metadata sidecars.
- `src/zymera/cli.py` — argparse CLI; `src/main.py` is a thin compat shim.

## Conventions

- Heavy imports (torch, diffusers, TTS, PIL) go inside functions/methods, never at module top level in config/prompts/identity/cli/pipeline.
- Every new tunable gets a key in `DEFAULTS` with a sensible default; expose common ones as CLI flags only if frequently used (`--set` covers the rest).
- Stage contract: set `section`, implement `_load()` returning the pipeline, implement `run(...)` returning raw artifacts (PIL images / frames). IO belongs in `pipeline.py`.
- Phase 3 video models must be SD1.5-family — the AnimateDiff motion adapter is incompatible with SDXL.
- Outputs always get a `<basename>.json` sidecar with prompt, seed, and params for reproducibility.
- VRAM-aware presets (`fast`, `low_vram`, `quantized`) must keep phase2 on SD1.5 + `ip-adapter_sd15.bin` — the SDXL IP-Adapter's 3.7 GB ViT-H image encoder breaks 6 GB GPUs (guarded by a unit test).
- Attention: PyTorch SDPA flash kernels are used automatically; never add a `flash-attn` dependency (won't build on Windows). TF32 is enabled via `runtime.tf32` in `base.configure_backends()`.
- Quantization is per-stage config (`<section>.quantization`), built in `base.build_quantization_config()` (bitsandbytes NF4/8-bit via diffusers `PipelineQuantizationConfig`). Quantized components can't be CPU-offloaded — presets enabling it must set `runtime.model_cpu_offload=false`; `_place_and_optimize` degrades gracefully either way.

## Responsible use (non-negotiable)

This project is for **synthetic identities only**: fully AI-generated personas or
people who have given explicit consent. Never add code, examples, prompts, tests,
or identities that target real people without consent (celebrities included).
If asked to do so, decline and point to this section.
