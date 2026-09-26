# Radiance 3.5.0 readiness report

Updated 2026-09-27. Assessed against the local working tree on `release/cleanup`.

**Assessment: 8/10 — controlled pilot, not approved for general production.**
This is an engineering judgment, not a certification. Nuke handoff and Resolve
import and hosted release CI are verified. Long-video windowing remains the
functional limitation. The maintainer requested release as v3.5 on 2026-09-27
with temporal windowing explicitly experimental and disabled by default.
Release authorization does not certify that experimental path for production.

## Completed

- Verified Nuke 17.1v2 with the installed interactive license. Fixed terminal
  listener dispatch, then passed real signed EXR handoff: Read node, frames
  1001-1003, raw scene-linear mode, and pixel values 0.18, 1.0 and 4.0. The
  real test used terminal mode; GUI dispatch has regression coverage.
- Fixed Windows Resolve scripting-library discovery inside the isolated import
  worker. A real Resolve Studio test imported an EXR still and a 121-frame EXR
  sequence, with Media Pool frame counts of 1 and 121. The host environment is
  unchanged and native failures remain isolated from ComfyUI.
- Rejected incompatible ControlNet/window combinations before the model receives
  mismatched frame ranges. ControlNet still works when the clip is not split.
- Labelled temporal windowing experimental in its controls and runtime message.
  The default remains off. More overlap is no longer described as a guarantee
  of coherent joins.
- Rebuilt and verified the 3.5.0 wheel, with all 47
  required frontend assets, the Nuke listener and the Resolve worker. Listener
  inclusion is now checked in CI. Tests and scratch data are
  excluded. No release has been published.
- Merged the latest main-branch fixes for packed audio/video latents, saved
  workflow/gizmo locations and the dashboard button.
- Passed hosted CI on the merged code (`842317b`), including Python 3.10–3.13
  on Windows and Linux, real runtime dependencies, packaging, browser rendering
  and JavaScript. [Branch CI](https://github.com/FXTD-Studios/radiance-beta/actions/runs/36269560499)
  and [PR CI](https://github.com/FXTD-Studios/radiance-beta/actions/runs/36269595975).
- Opened [draft PR #89](https://github.com/FXTD-Studios/radiance-beta/pull/89)
  for release review.

## Validation

| Check | Result |
| --- | --- |
| Latest full Python suite, including main-branch integration | 4,698 passed; 91 skipped; 3 slow tests deselected; 28 subtests passed |
| Branch-inclusive coverage | 68.07%; required floor 54% |
| Nuke listener follow-up: DCC regressions | 26 passed, including GUI and terminal dispatch; static/release checks passed |
| Focused sampler, DCC and reference checks | 105 passed |
| Separate slow tests | 3 passed |
| JavaScript | 324 passed; 5 skipped |
| Syntax and release hygiene | Passed |
| Real Resolve import | Still and 121-frame sequence passed |
| Nuke acceptance | Real signed handoff passed in 17.1v2 using the interactive license |
| Video quality | Full-refresh boat clips have clean sampled joins at 121 and 241 frames; the 121-frame train with shortened windows failed; 31-latent-frame train windows and native causal windows also failed |

Several approaches were rejected after real renders: moving boundaries,
first-frame references, related initial noise and gradual ancestral noise.
They are absent from the release. The full-refresh research candidate uses
fresh Gaussian noise above sigma 0.8 and Euler below it, with half-window
overlap. Its 121- and 241-frame boat clips have clean sampled joins. The train
clip with 16 latent frames per window distorted near the end; the same sampling
method without window splitting produced a coherent train control. A native
31-latent-frame-window train test also distorted the train and tracks. Native
ComfyUI causal context windows still changed the train geometry. A 241-frame
640x352 global pass had severe texture artifacts and was rejected before
refinement. The candidate is not shipped.

Native ComfyUI LCM followed by Euler matched the candidate's mathematical
probe, and an opt-in integration passed focused checks before being set aside
after failing visual acceptance. These diagnostic EXRs contain display-encoded
values, not scene-linear mastered output. Finite EXRs, low frame differences and one clean
scene are insufficient to establish production readiness.

Release review also repaired an impossible `Imath>=3.1.0` publish dependency,
aligned publish-test dependencies with CI, made failed-test artifact names unique
per operating system, and enabled the full matrix on public mirrors. Hosted
CI then exposed and verified repairs for Python 3.10 TOML parsing, Windows
3.13 invalid-path rejection, missing ffprobe/Hugging Face test dependencies,
and subprocess imports from a checkout named radiance-beta.

## Still required

1. Resolve temporal-window video coherence and pass visual acceptance at real
   resolution, including longer clips. Use `temporal_window=0` for current work;
   that path has higher memory requirements and is not a long-video fix.
2. Verify the final v3.5.0 release commit and package. Earlier release-branch CI
   passed; the Viewer audio, workflow restoration and keyboard fixes require
   validation on the final public release branch before publication.

An API-key issue was previously listed without a confirmed failing service.
That was incorrect and has been removed as a production blocker. Local Radiance
rendering does not require an API key.

Other declared limitations, including the optional WebGPU backend's feature
gaps, remain in [KNOWN_ISSUES.md](../KNOWN_ISSUES.md). The tested viewer path is
WebGL. Resolve import acceptance does not certify colour or timeline round trips.

Detailed evidence and earlier repairs: [development record](DEVELOPMENT.md).
