# Zymera — Synthetic Identity Media Pipeline

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org)
[![PyTorch](https://img.shields.io/badge/torch-2.1%2B-orange)](https://pytorch.org)

An open-source, fully configurable multi-stage pipeline for generating media of
**synthetic identities**: high-quality images, identity-consistent images, short
videos, and talking videos with speech and lip sync.

> **Responsible use** — This project is strictly for **fully AI-generated personas**
> or people who have given **explicit consent** to the use of their likeness and
> voice. Do not generate media of real people without consent. You are responsible
> for complying with local law and platform policies. Label generated content as
> synthetic where required.

---

## What it does

| Phase | Command | Output | Key models |
|-------|---------|--------|------------|
| `phase1` | `--phase phase1` | 1024px image | SDXL + DPM++ Karras |
| `phase2` | `--phase phase2` | Identity-consistent image | SD1.5 img2img (or IP-Adapter on 8 GB+) |
| `phase3` | `--phase phase3` | Short video clip | AnimateDiff + IP-Adapter |
| `phase4` | `--phase phase4` | Talking video with audio | Coqui TTS + ffmpeg (Wav2Lip pluggable) |

---

## Installation

```bash
# 1. CUDA-enabled PyTorch for your platform (see https://pytorch.org)
pip install torch --index-url https://download.pytorch.org/whl/cu130

# 2. All other dependencies
pip install -r requirements.txt
pip install -e .

# 3. Verify
zymera doctor
```

**Requirements:** Python 3.10+, NVIDIA GPU with CUDA.
**VRAM:** 6 GB minimum (`--preset low_vram`); 8 GB+ recommended for SDXL quality.
**Disk:** ~15 GB for model caches (downloaded automatically from Hugging Face).

---

## Quickstart

```bash
# Phase 1: generate a synthetic face
zymera generate --phase phase1 --style photorealistic \
  --prompt "a woman in her late 20s, long black hair, soft smile, emerald blouse, sunlit loft"

# Register that face as an identity
zymera identity create nova --images outputs/image_anon_<...>.png --note "fully synthetic"

# Phase 2: put that identity in a new scene
zymera generate --phase phase2 --identity nova \
  --prompt "wearing a red satin dress, studio backdrop, medium close-up" --style studio_portrait

# Phase 3: short video clip
zymera generate --phase phase3 --identity nova \
  --prompt "slowly smiling at the camera" --output clip.mp4

# Phase 4: talking video
zymera generate --phase phase4 --identity nova \
  --prompt "speaking warmly to the camera" \
  --text "Hello! Welcome to the demo." --output talk.mp4
```

Outputs land in `outputs/`. Each file has a `.json` sidecard recording the full
prompt, seed, and parameters — reproduce any result exactly with `--seed <N>`.

---

## Auto mode (agentic planner)

Let Zymera pick the phase, preset, models, LoRAs, and style for you from a plain
description, then download what's needed and save it as a re-runnable **recipe**:

```bash
# Plan only (prints what it would do)
zymera auto "a cinematic synthetic portrait, anime-leaning, on my 6GB laptop"

# Plan, download assets, save a recipe, and generate
zymera auto "a synthetic woman speaking warmly to camera" \
  --save nova_talk --run --prompt "..." --text "Hello!" --yes
```

- **With `ANTHROPIC_API_KEY` set** (and `pip install anthropic`), a Claude planner
  reasons about the requirement and selects compatible assets.
- **Without a key**, a deterministic heuristic planner (GPU tier + keywords) runs —
  no network, no cost. Same plan contract either way.

A **recipe** is just a preset, so anything you save is reproducible:

```bash
zymera recipe list
zymera recipe show nova_talk
zymera recipe run  nova_talk --prompt "..." --text "Hello!"
```

### LoRA adapters

LoRAs come from the asset catalog (`configs/registry.json`, deep-merged over a
built-in set) and load via peft. Enable per stage:

```bash
zymera generate --phase phase1 --prompt "..." \
  --set image.lora.enabled=true \
  --set 'image.lora.adapters=[{"name":"lcm-lora-sdxl","scale":0.8}]'
```

`name` is a catalog entry or a HuggingFace repo. Sources are `hf` or `civitai`
(set `CIVITAI_API_KEY` for gated files). A LoRA whose `family` doesn't match the
base model is skipped with a warning rather than crashing.

> **Content policy (enforced on every download).** Two independent axes:
> **real-person** assets are *always* blocked (synthetic identities only — see
> Responsible use); **NSFW** is a separate opt-in via `--set registry.content_mode=nsfw`
> (or `zymera auto --nsfw`) and only ever permits NSFW of *synthetic* personas.
> NSFW mode never unlocks real people.

---

## Configuration

Settings are layered; later layers win:

```
built-in defaults  ←  configs/default.json  ←  --preset  ←  --set key=value
```

### Presets

| Preset | Phase 1 | Phase 2 | Use when |
|--------|---------|---------|----------|
| `balanced` | SDXL 1024px | SDXL IP-Adapter | 8 GB+ VRAM, default quality |
| `quality` | SDXL 1024px, more steps | SDXL IP-Adapter | best output, slow |
| `fast` | SDXL 832px, fewer steps | SD1.5 img2img | quick iterations |
| `low_vram` | SDXL 768px | SD1.5 img2img | 6 GB GPUs |
| `quantized` | SDXL NF4 4-bit | SD1.5 img2img | SDXL quality on 6 GB |
| `sd15` | SD1.5 512px | SD1.5 img2img | CPU or very low VRAM |

```bash
zymera generate --phase phase1 --prompt "..." --preset quantized
```

### Overriding any value

```bash
zymera generate --phase phase1 --prompt "..." \
  --preset quality \
  --set image.scheduler=EulerAncestralDiscreteScheduler \
  --set runtime.model_cpu_offload=false \
  --set image.steps=50
```

### Common CLI flags

`--steps`, `--guidance-scale`, `--width`, `--height`, `--frames`, `--num-images`,
`--seed`, `--style`, `--offload on|off`

### Config sections

| Section | Controls |
|---------|----------|
| `runtime` | device, dtype, seed, TF32, VRAM (offload / slicing / tiling) |
| `image` | phase1 model, VAE, scheduler, steps, resolution, quantization |
| `identity_image` | phase2 method (`img2img` / `ip_adapter`), strength, steps |
| `video` | SD1.5 checkpoint, motion adapter, frames, fps, IP-Adapter |
| `speech` | TTS model, speaker_wav for voice cloning |
| `compose` | lip-sync method, codecs, intermediate cleanup |
| `image/identity_image/video.lora` | LoRA adapters (`enabled`, `adapters`, `fuse`) |
| `registry` | `content_mode` (`sfw`/`nsfw`); `paths.assets_dir`, `paths.registry_file` |
| `agent` | Claude planner `model` (default `claude-opus-4-8`) and `max_tokens` |

---

## Performance on small GPUs (6 GB)

- **Flash attention** is on automatically — PyTorch SDPA dispatches flash kernels
  on RTX 30/40-series GPUs. No `flash-attn` package needed.
- **TF32** (`runtime.tf32 = true`, default) speeds up all matmuls on Ampere/Ada
  with no quality impact (~15% faster denoising on RTX 40xx).
- **NF4 4-bit quantization** (`--preset quantized`, needs `bitsandbytes`): shrinks
  the SDXL UNet from 5.1 GB → 1.7 GB, enabling full 1024px quality without CPU
  offload on 6 GB cards.
- **OOM escalation path**: `fast` → `quantized` → `low_vram` → `sd15`

Check everything with `zymera doctor`.

---

## Prompt styles

```bash
zymera styles          # list all styles
```

Built-in styles: `photorealistic`, `studio_portrait`, `cinematic`, `fashion_editorial`,
`anime`, `natural_light_portrait`, `product_lifestyle`, `none`.

Add your own in `configs/prompts.json` (no code change needed):

```json
{
  "styles": {
    "my_style": {
      "prefix": "dark fantasy concept art, ",
      "suffix": ", dramatic rim lighting, ultra detailed",
      "negative": "photo, realistic, blurry"
    }
  }
}
```

---

## Identities

```
identities/<id>/
├── images/ref_0.png     # reference image(s)
└── metadata.json
```

```bash
zymera identity create nova --images face.png --note "fully synthetic persona"
zymera identity add    nova --image face2.png
zymera identity list
```

Phase 2 requires at least one reference image. The recommended workflow is to
generate a face you like in phase1 and register it as the identity.

---

## Lip sync

Default: loops video to speech duration and muxes audio (ffmpeg). For true lip
sync, install [Wav2Lip](https://github.com/Rudrabha/Wav2Lip) and set:

```json
{
  "compose": {
    "lipsync": {
      "method": "wav2lip",
      "command": [
        "python", "/path/to/Wav2Lip/inference.py",
        "--checkpoint_path", "/path/to/wav2lip_gan.pth",
        "--face", "{video}", "--audio", "{audio}", "--outfile", "{output}"
      ]
    }
  }
}
```

## Voice cloning

With a consented voice sample:

```bash
zymera generate --phase phase4 --identity nova --prompt "..." --text "Hello!" \
  --set speech.model=tts_models/multilingual/multi-dataset/xtts_v2 \
  --set speech.speaker_wav=voices/nova.wav
```

---

## Project layout

```
configs/
  default.json          user config overrides
  prompts.json          custom styles
  presets/              JSON preset files (e.g. sd15.json)
src/zymera/
  config.py             layered config system
  prompts.py            PromptBuilder: styles + negative prompts
  identity.py           IdentityStore
  capabilities.py       GPU/dep probing (CapabilityProfile)
  pipeline.py           phase orchestration, seeds, metadata sidecars
  cli.py                CLI entry point
  doctor.py             environment health checks
  recipes.py            RecipeStore — saved "skill" presets
  registry/             asset catalog, downloader, content-policy gate
  planner/              heuristic requirement -> GenerationPlan
  agent/                optional Claude planner/executor (zymera auto)
  stages/               one file per stage
    base.py             Stage base class, quantization, backend config
    text2image.py       phase1
    identity_image.py   phase2
    video.py            phase3
    tts.py              phase4 speech
    compose.py          phase4 A/V composition
tests/                  unit tests (no GPU required)
scripts/
  smoke_test.py         end-to-end GPU test (all 4 phases)
```

---

## Testing

```bash
# Fast unit tests — no GPU needed
python -m pytest tests -q

# Full GPU smoke test
python scripts/smoke_test.py --preset low_vram

# Specific phases only
python scripts/smoke_test.py --preset low_vram --phases phase1 phase2 --traceback
```

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| CUDA out of memory | `--preset low_vram` → `--preset quantized` → `--preset sd15`; reduce `--frames` for video |
| Slow first run | Models download on first use (~7 GB SDXL, ~2 GB SD1.5); cached in `~/.cache/huggingface/hub` |
| Video errors / wrong UNet shape | `video.model` must be SD1.5-family — AnimateDiff motion adapter is incompatible with SDXL |
| No audio in phase4 | Check TTS and ffmpeg rows in `zymera doctor` |
| Verbose output | Use `zymera -v generate ...` to enable debug logs |
| Download hangs | Set `HF_HUB_ENABLE_HF_TRANSFER=1` (hf_transfer is in requirements.txt) |

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md).

---

## License

MIT — see [LICENSE](LICENSE).
