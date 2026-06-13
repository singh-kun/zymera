# Responsible AI Use — Zymera

## Purpose

Zymera is a tool for generating **synthetic identity media**: fully AI-generated
personas, characters, and avatars. It is **not** a tool for generating media of
real, identifiable people without their explicit, documented consent.

This document describes the policy, its technical enforcement, and the reasoning
behind it. All contributors and users are expected to read and follow it.

---

## What Zymera is for

- Fully AI-generated characters and personas (no real-world counterpart).
- Stylised avatars, concept art, and synthetic actors for creative projects.
- Research and development on generative pipelines where no real person is depicted.
- Consensual use: a real person who has explicitly consented to having their likeness
  used as an identity reference, and has provided that consent in writing.

## What Zymera is not for

- Generating images or video of celebrities, public figures, or any identifiable
  real person without their explicit, documented consent.
- Creating non-consensual intimate imagery (NCII) of any person, real or synthetic,
  that could be used to harm, harass, or deceive.
- Producing media intended to impersonate, defame, or deceive audiences about the
  identity or statements of a real person (deepfakes for deception).
- Bypassing or weakening the content-policy gate (`PolicyGate`) in any mode.

---

## The two-axis content policy

The asset downloader enforces policy on every download through `PolicyGate`
(`src/zymera/registry/policy.py`). It screens two independent axes that must
**never** be conflated:

### Axis 1 — Real-person content (ALWAYS blocked)

Any asset whose metadata indicates it depicts or is conditioned on a real,
identifiable person is blocked unconditionally, in every operating mode, without
exception. This includes:

- Assets flagged with `"poi": true` or `"real_person": true` in the catalog.
- Assets whose name, tags, trigger words, or description contain real-person
  indicators (named individuals, "celebrity", "realistic likeness of …", etc.)
  detected by the keyword safety net in `policy.py`.

**No flag, mode, config key, or command-line argument bypasses Axis 1.** If you
believe a block is a false positive on a genuinely synthetic asset, open an issue
with the catalog entry's source URL; the fix is to correct the catalog, not to
weaken the gate.

### Axis 2 — SFW / NSFW (mode-gated, separate from Axis 1)

Adult/explicit content of a *synthetic* persona is gated by `registry.content_mode`
(default: `sfw`). NSFW mode permits explicit content of synthetic characters only:

| `content_mode` | Synthetic SFW | Synthetic NSFW | Real-person |
|---|---|---|---|
| `sfw` (default) | Allowed | Blocked | Always blocked |
| `nsfw` | Allowed | Allowed | Always blocked |

To enable NSFW mode for an asset search or generation run:

```cmd
python -m zymera auto "..." --nsfw
python -m zymera generate ... --set registry.content_mode=nsfw
```

Axis 2 changes **only** which synthetic assets are permitted. It does not touch,
relax, or influence Axis 1 in any way. The tests in `tests/test_policy.py` guard
this invariant — do not modify them to make real-person tests pass.

---

## For contributors

- **Never add** code, examples, prompts, tests, identities, or catalog entries that
  target real people without consent. Pull requests that do so will be rejected.
- **Never weaken Axis 1.** If a new catalog source or download path bypasses
  `PolicyGate.screen()`, it is a bug — fix it before merging.
- **Extend `tests/test_policy.py`** when you add new asset sources or gate logic.
  The test that asserts "real-person entry is blocked in BOTH modes" must always
  pass.
- When adding voice/TTS assets for phase4, apply the same two-axis screen.
- The gate's `Decision` object includes a `reason` string — surface it in CLI
  output so users understand exactly why an asset was blocked.

## For users

- If you supply your own identity images (`zymera identity create`), ensure the
  subject has given explicit written consent or is a fully synthetic persona.
- If you use `configs/registry.json` to add custom catalog entries, apply the
  same `"poi"` / `"real_person"` flags that the built-in catalog uses.
- If `CIVITAI_API_KEY` is set, Zymera can download from Civitai — the same
  policy gate applies. Do not use API keys to circumvent screening.

---

## Legal context (informational, not legal advice)

The following laws and regulations are relevant to AI-generated media of real people.
Consult a qualified legal professional for advice in your jurisdiction.

- **Non-consensual intimate imagery (NCII)**: Many jurisdictions criminalise the
  creation or distribution of synthetic intimate images of real people without consent
  (e.g., UK Online Safety Act 2023, US DEFIANCE Act, various US state laws).
- **Defamation / false light**: Realistic synthetic media that places a real person in
  a false context may give rise to civil liability.
- **Right of publicity**: Using a person's likeness for commercial purposes without
  consent may infringe their publicity rights.
- **EU AI Act**: The EU AI Act classifies deepfake generation as high-risk in certain
  contexts and requires disclosure that content is AI-generated.
- **Platform terms**: Major hosting and distribution platforms prohibit non-consensual
  synthetic media of real people in their terms of service.

Zymera's technical policy is stricter than any single legal requirement — it blocks
all real-person content, not just intimate or commercial content. This is intentional.

---

## Reporting concerns

If you discover that Zymera is being used in a way that violates this policy, or if
you find a bypass in the `PolicyGate`, please open an issue at:

https://github.com/singh-kun/zymera/issues

Label it `responsible-use` or `security` as appropriate.
