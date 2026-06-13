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
zymera auto "a cinematic synthetic portrait, 6gb gpu" # plan models/LoRAs/config (agentic; heuristic if no ANTHROPIC_API_KEY)
zymera recipe list                                    # saved "skill" recipes
python scripts/smoke_test.py                          # GPU end-to-end smoke test (slow, downloads models)
```

## Architecture

- `src/zymera/config.py` — layered config: built-in `DEFAULTS` <- `configs/default.json` <- `--preset` <- `--set key=value`. All tunables live here; never hardcode generation parameters elsewhere.
- `src/zymera/prompts.py` — `PromptBuilder`: style prefixes/suffixes + curated negative prompts. Data-driven via `configs/prompts.json` (deep-merged over built-ins).
- `src/zymera/identity.py` — `IdentityStore`: `identities/<id>/metadata.json` + `images/ref_*.jpg`.
- `src/zymera/stages/` — one file per stage; all inherit `base.Stage` (lazy model load, device/dtype resolution, VRAM optimizations). Registry with lazy imports in `stages/__init__.py` so torch only loads when generating.
- `src/zymera/pipeline.py` — orchestrator: `PHASES` maps phase names to runners; owns seeds, output naming, and JSON metadata sidecars.
- `src/zymera/cli.py` — argparse CLI; `src/main.py` is a thin compat shim.
- `src/zymera/registry/` — asset catalog + downloader + **content-policy gate**. `catalog.py` (built-in + `configs/registry.json`), `manager.py` (`AssetManager` downloads from HuggingFace/Civitai, policy-gated, cached under `paths.assets_dir`), `policy.py` (`PolicyGate` — see Responsible use).
- `src/zymera/capabilities.py` — `probe()` returns a `CapabilityProfile` (VRAM tier, bitsandbytes, ffmpeg, TTS). `doctor.py` formats these; the planner uses them.
- `src/zymera/planner/` — `heuristic.plan(requirement, profile, catalog)` maps a natural-language requirement + GPU tier to a `GenerationPlan` (the shared contract in `types.py`). No LLM.
- `src/zymera/agent/` — optional Claude-powered planner/executor (`zymera auto`). Lazy-imports `anthropic`; falls back to the heuristic planner when `ANTHROPIC_API_KEY` is unset. Planner emits a `GenerationPlan`; the executor screens+downloads assets, saves a recipe, optionally runs.
- `src/zymera/recipes.py` — `RecipeStore`: a "skill" = a saved preset JSON (`configs/presets/<name>.json`) bundling base preset + LoRAs + style + an `assets` list, re-runnable via `--preset` or `zymera recipe run`.

## Conventions

- Heavy imports (torch, diffusers, TTS, PIL) go inside functions/methods, never at module top level in config/prompts/identity/cli/pipeline.
- Every new tunable gets a key in `DEFAULTS` with a sensible default; expose common ones as CLI flags only if frequently used (`--set` covers the rest).
- Stage contract: set `section`, implement `_load()` returning the pipeline, implement `run(...)` returning raw artifacts (PIL images / frames). IO belongs in `pipeline.py`.
- Phase 3 video models must be SD1.5-family — the AnimateDiff motion adapter is incompatible with SDXL.
- Outputs always get a `<basename>.json` sidecar with prompt, seed, and params for reproducibility.
- VRAM-aware presets (`fast`, `low_vram`, `quantized`) must keep phase2 on SD1.5 + `ip-adapter_sd15.bin` — the SDXL IP-Adapter's 3.7 GB ViT-H image encoder breaks 6 GB GPUs (guarded by a unit test).
- Attention: PyTorch SDPA flash kernels are used automatically; never add a `flash-attn` dependency (won't build on Windows). TF32 is enabled via `runtime.tf32` in `base.configure_backends()`.
- Quantization is per-stage config (`<section>.quantization`), built in `base.build_quantization_config()` (bitsandbytes NF4/8-bit via diffusers `PipelineQuantizationConfig`). Quantized components can't be CPU-offloaded — presets enabling it must set `runtime.model_cpu_offload=false`; `_place_and_optimize` degrades gracefully either way.
- LoRA is per-stage config (`<section>.lora`), loaded by `base.load_loras()` (peft) after the scheduler and before placement. Adapters resolve through `AssetManager` (policy-gated) by registry name or HF repo; a `family` mismatch vs `<section>.family` is warn-and-skipped, never fatal.
- Downloadable assets (models/LoRAs/adapters) are never hardcoded — they live in the catalog (`zymera.registry.catalog` built-ins + `configs/registry.json`) and are fetched on demand by `AssetManager`. Every fetch passes `PolicyGate` first.
- The agent layer is a thin reasoning layer over deterministic functions: each agent tool wraps a plain `zymera.*` call, and both the heuristic and Claude planners emit the same `GenerationPlan`. Keep it that way — anything the agent can do must be doable without an LLM. Single source for the model id: `agent.model` (default `claude-opus-4-8`).

## Responsible use (non-negotiable)

This project is for **synthetic identities only**: fully AI-generated personas or
people who have given explicit consent. Never add code, examples, prompts, tests,
or identities that target real people without consent (celebrities included).
If asked to do so, decline and point to this section.

The asset downloader enforces this via `zymera.registry.policy.PolicyGate`, which
screens **two orthogonal axes** that must never be conflated:
1. **Real-person (axis 1)** — ALWAYS blocked, in every mode, non-negotiable. Driven
   by explicit `real_person`/`poi` flags (Civitai's "person of interest") plus a
   keyword safety net. No flag or mode bypasses it.
2. **SFW/NSFW (axis 2)** — a *separate* `registry.content_mode` toggle (`sfw`
   default). NSFW of a synthetic persona is permitted by the policy; NSFW mode only
   lifts this filter and never touches axis 1.
When extending the gate, keep the two axes independent. Tests in
`tests/test_policy.py` guard the invariant that NSFW mode never unlocks real people.
