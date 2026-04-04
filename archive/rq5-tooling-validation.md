# RQ5 — Tooling and Stack Validation

**Date:** 2026-04-03
**Status:** Complete

**Question:** Does the proposed tech stack work correctly on the target environment, and are there compatibility issues to address early?

**Depends on:** RQ3 (candidate models to validate)

---

## Executive Summary

The proposed stack is viable on Windows 11 + Python 3.13 + CUDA + RTX 3070 (8 GB VRAM) with no blocking issues. All core components have Python 3.13 support, and `uv` now has first-class PyTorch CUDA integration that eliminates the historically painful manual index configuration.

**Key findings:**

1. **PyTorch 2.11.0** (stable, March 2026) supports Python 3.10–3.14 on Windows with CUDA 12.6, 12.8, and 13.0. The RTX 3070 (Ampere, SM 8.6) is fully supported across all three CUDA builds.
2. **open_clip 3.3.0** officially classifies Python 3.9–3.12 but requires only `>=3.9` — Python 3.13 works in practice (confirmed by NixOS and community builds). Pure Python + PyTorch dependency chain; no native extension blocks 3.13.
3. **All RQ3 candidate models fit simultaneously in ~3.3–5.1 GB VRAM** with the ViT-L/14 CLIP backbone. Sequential load/unload is unnecessary on 8 GB.
4. **`uv`** provides `--torch-backend=auto` and declarative `pyproject.toml` index configuration for CUDA wheels — the install experience is now clean on Windows.
5. **One concern:** `onnxruntime-gpu` is needed for deepghs models and can coexist with PyTorch, but adds ~200 MB of CUDA runtime libraries. The alternative is to run deepghs models through `imgutils` which wraps ONNX internally.

**Recommended CUDA target:** CUDA 12.8 — stable in PyTorch 2.11, supported by onnxruntime-gpu 1.24, and the RTX 3070's Ampere (SM 8.6) is fully included.

---

## 1. PyTorch + CUDA on Windows

### Current State

| Component | Version | Status |
| --- | --- | --- |
| PyTorch | 2.11.0 (2026-03-23) | Stable release |
| Python support | 3.10, 3.11, 3.12, **3.13**, 3.14 | All have Windows wheels |
| CUDA 12.6 | Stable | SM 5.0–9.0 on Windows |
| CUDA 12.8 | Stable | SM 7.5–12.0 on Windows (drops Maxwell) |
| CUDA 13.0 | Stable (new in 2.11) | SM 7.5–12.0 on Windows |

**Source:** [PyTorch 2.11 CUDA support matrix RFC](https://github.com/pytorch/pytorch/issues/172663), [PyPI torch page](https://pypi.org/project/torch/)

### RTX 3070 Compatibility

The RTX 3070 uses the GA104 GPU (Ampere architecture), compute capability **SM 8.6**. This is explicitly supported by all three CUDA build variants in PyTorch 2.11:

- **CUDA 12.6.3:** Maxwell(5.0), Pascal(6.0), Volta(7.0), Turing(7.5), **Ampere(8.0, 8.6)**, Hopper(9.0)
- **CUDA 12.8.1:** Turing(7.5), **Ampere(8.0, 8.6)**, Hopper(9.0), Blackwell(10.0, 12.0)
- **CUDA 13.0.0:** Turing(7.5), **Ampere(8.0, 8.6)**, Hopper(9.0), Blackwell(10.0, 12.0)

### Recommendation

**Use CUDA 12.8.** It is stable (not experimental), includes Ampere, and aligns with onnxruntime-gpu's default CUDA 12.x builds. CUDA 12.6 is also fine but older; CUDA 13.0 is newly stable and has less ecosystem testing.

### CUDA Toolkit Installation

PyTorch ships its own CUDA runtime libraries — a standalone CUDA Toolkit install is **not required** for inference. However, if `onnxruntime-gpu` needs cuDNN or CUDA libraries not bundled with its wheels, the CUDA Toolkit or the `nvidia-*` pip packages can fill the gap. The onnxruntime-gpu 1.24 wheels bundle the necessary CUDA/cuDNN runtime DLLs.

---

## 2. open_clip

| Attribute | Value |
| --- | --- |
| Latest version | 3.3.0 (2026-02-27) |
| Python requirement | `>=3.9` |
| Official classifiers | 3.9, 3.10, 3.11, 3.12 |
| Python 3.13 status | **Works** — not in classifiers but no native extension blocks it; confirmed by community builds |
| License | MIT |
| Key dependencies | torch, torchvision, timm, huggingface_hub |

**Source:** [PyPI open-clip-torch](https://pypi.org/project/open-clip-torch/)

open_clip is a pure Python package on top of PyTorch. Since PyTorch 2.11 fully supports Python 3.13 on Windows, and open_clip has no native extensions of its own, the combination works. The classifier list lagging behind the `requires-python` field is common in the PyTorch ecosystem.

### Model Loading

```python
import open_clip
model, _, preprocess = open_clip.create_model_and_transforms(
    "ViT-L-14", pretrained="openai"
)
model = model.to("cuda").eval()
```

VRAM for ViT-L/14 at FP16: ~1.5 GB. Comfortable on 8 GB.

---

## 3. Candidate Models from RQ3

### 3.1 VRAM Budget

RTX 3070: 8 GB total. Assume ~1.0–1.5 GB reserved for OS/display = **~6.5–7.0 GB working budget.**

### 3.2 Model-by-Model Validation

#### WD-Tagger v3 (SwinV2-Base) — Must Evaluate

| Attribute | Value |
| --- | --- |
| Repository | `SmilingWolf/wd-swinv2-tagger-v3` |
| Loading | `timm.create_model("hf-hub:SmilingWolf/wd-swinv2-tagger-v3", pretrained=True)` |
| Alternative | `AutoModelForImageClassification` via `p1atdev/wd-swinv2-tagger-v3-hf` |
| Params | ~90M |
| Model file | 392 MB (safetensors) / 467 MB (ONNX) |
| Input | 448x448 RGB, normalized mean/std=[0.5, 0.5, 0.5], bicubic, center crop |
| Output | 10,861 sigmoid probabilities |
| Est. VRAM | ~1.0–1.5 GB (weight + inference buffer) |
| Dependencies | `timm` (recommended), or `onnxruntime` |
| Python 3.13 | timm 1.0.26 tested with Python 3.13 + PyTorch 2.6+ |
| License | Apache-2.0 |

**Status: Compatible.** Load via timm to share the PyTorch runtime with CLIP and other models. No ONNX runtime needed for this model.

#### WD-Tagger v3 (EVA02-Large) — Must Evaluate

Same as SwinV2-Base above but:

- Params: ~300M
- Model file: 1.26 GB (safetensors)
- Est. VRAM: ~2.0–3.0 GB
- Macro-F1: 0.4772 (vs 0.4541 for SwinV2)

**Status: Compatible.** Larger but still fits. Consider this the accuracy-optimized variant; SwinV2-Base is the efficiency variant.

#### deepghs/anime_aesthetic (SwinV2) — Must Evaluate

| Attribute | Value |
| --- | --- |
| Repository | `deepghs/anime_aesthetic` |
| Loading | Via `imgutils` library: `from imgutils.validate import anime_aesthetic` |
| Alternative | Direct ONNX loading with onnxruntime |
| Params | 66M (SwinV2 variant) |
| Format | ONNX |
| Output | 7-class: masterpiece/best/great/good/normal/low/worst |
| Est. VRAM | ~0.3 GB |
| Dependencies | `dghs-imgutils` or `onnxruntime-gpu` |
| License | OpenRAIL |

**Status: Compatible.** Two loading paths:

1. **Via imgutils** (`pip install dghs-imgutils[gpu]`) — wraps ONNX runtime internally, provides a clean API. Requires Python 3.8+.
2. **Direct ONNX** — load the `.onnx` file with `onnxruntime.InferenceSession`. Requires `onnxruntime-gpu`.

imgutils is the lower-friction path. It handles model downloading, caching, and preprocessing internally.

#### kawaimasa/kawai-aesthetic-scorer — Must Evaluate

| Attribute | Value |
| --- | --- |
| Repository | `kawaimasa/kawai-aesthetic-scorer-convnextv2` |
| Loading | `AutoModelForImageClassification.from_pretrained(repo_id)` |
| Params | ~200M (ConvNeXtV2-Large) |
| Input | 768x768 RGB, letterbox-padded, ImageNet normalization |
| Output | 5-class softmax (SS/S/A/B/C) + weighted scalar |
| Est. VRAM | ~1.0 GB |
| Dependencies | `transformers`, `torch`, `torchvision`, `pillow` |
| License | Apache-2.0 |

**Status: Compatible.** Standard HuggingFace Transformers loading. The `transformers` library supports Python 3.13 since v4.38+. Custom `PadResizeProcessor` is pure Python/Pillow — no compatibility risk.

**Preprocessing sensitivity:** Author warns that inference must match training preprocessing exactly (768px letterbox with black padding, specific normalization). The inference code is provided in the model card and must be followed precisely.

#### skytnt/anime-segmentation — Must Evaluate

| Attribute | Value |
| --- | --- |
| Repository | `skytnt/anime-seg` (HF), `SkyTNT/anime-segmentation` (GitHub) |
| Loading | `AnimeSegmentation.from_pretrained("skytnt/anime-seg")` |
| Architecture | ISNet-IS |
| Input | 1024px recommended |
| Output | Per-pixel alpha mask (character segmentation) |
| Est. VRAM | ~0.5 GB |
| Dependencies | `pytorch_lightning`, `kornia`, `timm`, `huggingface_hub>=0.22` |
| License | Apache-2.0 |

**Status: Compatible with caveats.**

- `pytorch_lightning` 2.6.1 supports Python 3.13.
- `kornia` depends on PyTorch; should work with 3.13 but not explicitly tested in classifiers.
- The `pytorch_lightning` dependency is heavy for inference-only use. Consider extracting the model loading logic directly if the dependency footprint is a concern, or using the ONNX export instead.

**Alternative:** Load the ONNX export via onnxruntime to avoid the pytorch_lightning/kornia dependency chain entirely.

#### deepghs face/head/person detection — Must Evaluate

| Attribute | Value |
| --- | --- |
| Repository | `deepghs` organization on HuggingFace |
| Loading | Via `imgutils`: `from imgutils.detect import detect_faces, detect_heads, detect_persons` |
| Format | ONNX |
| Est. VRAM | ~0.3 GB per model |
| Dependencies | `dghs-imgutils[gpu]` |
| License | Open source (non-profit project) |

**Status: Compatible.** Same loading path as anime_aesthetic — via imgutils or direct ONNX. Lightweight models.

#### OpenCLIP ViT-L/14 — Should Evaluate

Covered in Section 2 above. ~1.5 GB VRAM. Fully compatible.

#### Improved Aesthetic Predictor — Should Evaluate

| Attribute | Value |
| --- | --- |
| Architecture | MLP head on CLIP ViT-L/14 embeddings |
| Loading | Load CLIP via open_clip, then load MLP weights |
| Est. VRAM | Negligible beyond CLIP (~few MB for MLP) |
| Dependencies | `open_clip` (already required) |
| License | Apache-2.0 |

**Status: Compatible.** Nearly free if CLIP is already loaded. The MLP head is tiny.

#### Waifu Scorer v3 — Should Evaluate

| Attribute | Value |
| --- | --- |
| Repository | `Eugeoter/waifu-scorer-v3` |
| Architecture | CLIP backbone + MLP head |
| Model files | `model.safetensors` (11.2 MB) |
| Output | 0–10 continuous score |
| Est. VRAM | Negligible beyond CLIP (~11 MB for MLP) |
| License | OpenRAIL |

**Status: Compatible with caution.** The model is a safetensors file containing MLP weights. Loading requires knowing the MLP architecture (hidden dimensions, layers). The README is minimal (710 bytes) and no inference code is published on the model card. Inference code must be reverse-engineered from the safetensors weight shapes or from HuggingFace Spaces using this model.

This is a validation risk — the model may work well but the lack of documentation means integration takes extra effort.

#### Depth Anything V2 (Small) — Could Evaluate

| Attribute | Value |
| --- | --- |
| Repository | `DepthAnything/Depth-Anything-V2` |
| Params | Small variant |
| Est. VRAM | ~0.5 GB |
| License | Apache-2.0 |

**Status: Compatible.** DINOv2-based, standard PyTorch. Known limitation: anime images produce flat depth maps due to non-photorealistic shading. Useful only for coarse foreground/background separation.

### 3.3 VRAM Combinations

| Configuration | Est. VRAM | Fits 8 GB? |
| --- | --- | --- |
| CLIP ViT-L/14 + WD-Tagger SwinV2 + deepghs/anime_aesthetic + anime-segmentation | ~3.3 GB | Yes — comfortable |
| Above + deepghs detection + Depth Anything V2 Small | ~4.1 GB | Yes |
| Above + kawaimasa aesthetic scorer | ~5.1 GB | Yes |
| Above + Waifu Scorer MLP (shares CLIP) | ~5.1 GB | Yes |
| Above + Improved Aesthetic Predictor MLP (shares CLIP) | ~5.1 GB | Yes |
| Replace SwinV2 with EVA02-Large WD-Tagger | +1.0–1.5 GB | Still fits (~6.1–6.6 GB) |
| Replace ViT-L/14 with ViT-H/14 | +2.4 GB | Tight (~7.5 GB), may need sequential loading |

**Conclusion:** All "Must" and "Should" models fit simultaneously with the ViT-L/14 backbone. No model management (lazy load/unload) is needed for the recommended configuration. If ViT-H/14 or EVA02-Large WD-Tagger are both desired, sequential loading may be needed.

**Important caveat:** All VRAM estimates are approximate, derived from model parameter counts and FP16 weight sizes. Actual VRAM includes PyTorch CUDA context (~300–500 MB), inference activation memory (varies by batch size and input resolution), and memory fragmentation. **Empirical measurement during implementation is essential.** The estimates here provide confidence that the budget is feasible, not exact allocations.

---

## 4. opencv-python-headless

| Attribute | Value |
| --- | --- |
| Latest version | 4.13.0.92 (2026-02-05) |
| Python support | 3.7–3.13 (stable ABI wheel: `cp37-abi3`) |
| Windows wheel | `opencv_python_headless-4.13.0.92-cp37-abi3-win_amd64.whl` |
| Python 3.13 classifier | Yes — explicitly listed |

**Source:** [PyPI opencv-python-headless](https://pypi.org/project/opencv-python-headless/)

**Status: Fully compatible.** Uses the CPython stable ABI (`abi3`), so a single wheel covers Python 3.7 through 3.13+. Pre-built Windows x86-64 wheel available. No build-from-source needed.

---

## 5. Other Stack Components

| Package | Latest Version | Python 3.13 | Notes |
| --- | --- | --- | --- |
| **Pydantic v2** | 2.12+ | Yes (since 2.8.0) | Explicit 3.13 support. Versions < 2.8.0 are incompatible with 3.13. |
| **Typer** | 0.15+ | Yes | Requires `>=3.8`. Pure Python over Click. |
| **Rich** | 13.x | Yes | Confirmed on Python 3.13 readiness tracker. |
| **PyYAML** | 6.x | Yes | Confirmed on Python 3.13 readiness tracker. |
| **timm** | 1.0.26 (2026-03-23) | Yes | "PyTorch 2.6 & Python 3.13 are tested and working" per release notes. |
| **transformers** | 4.50+ | Yes | Needed for kawaimasa model. Supports 3.13 since 4.38+. |
| **Pillow** | 11.x | Yes | Standard imaging library, early Python version adopter. |
| **NumPy** | 2.x | Yes | Python 3.13 support since NumPy 2.1. |
| **SciPy** | 1.15+ | Yes | Python 3.13 support since SciPy 1.14. |

**Sources:** [Pydantic 3.13 issue](https://github.com/pydantic/pydantic/issues/11524), [PyPI timm](https://pypi.org/project/timm/), [Python 3.13 readiness](https://pyreadiness.org/3.13/)

### ONNX Runtime (conditional dependency)

| Attribute | Value |
| --- | --- |
| Package | `onnxruntime-gpu` |
| Latest version | 1.24.4 (2026-03-17) |
| Python 3.13 | Yes — `cp313-cp313-win_amd64.whl` available |
| CUDA | 12.x (default); 13.x available in nightlies |
| Windows wheel | Available |

**Source:** [PyPI onnxruntime-gpu](https://pypi.org/project/onnxruntime-gpu/)

Needed if loading deepghs models directly or if preferring ONNX inference for WD-Tagger. Not needed if using `imgutils` (which bundles its own onnxruntime dependency) or loading all models via PyTorch/timm.

### Dependency Strategy: PyTorch vs ONNX Runtime

Two models (deepghs/anime_aesthetic, deepghs detection) are ONNX-only. Two paths exist:

1. **Via `imgutils`:** Install `dghs-imgutils[gpu]`. This pulls in onnxruntime internally and provides a clean Python API. Lower integration effort, higher dependency weight.
2. **Direct ONNX:** Install `onnxruntime-gpu` and load `.onnx` files manually. More control, less abstraction.

**Recommendation:** Start with `imgutils` for rapid prototyping during evaluation. If the dependency footprint (imgutils pulls in many sub-dependencies) becomes a concern, switch to direct ONNX loading for production.

Both PyTorch and ONNX Runtime can coexist on the same CUDA device. They manage separate CUDA contexts, so VRAM is shared but not pooled — total usage is additive.

---

## 6. uv + PyTorch CUDA Install Sequence

### Validated Configuration

`uv` now has first-class PyTorch support with `--torch-backend` and declarative index configuration in `pyproject.toml`.

### pyproject.toml Configuration

```toml
[project]
name = "loupe"
version = "0.1.0"
requires-python = ">=3.13"
dependencies = [
    "torch>=2.11.0",
    "torchvision",
    "open-clip-torch>=3.3.0",
    "opencv-python-headless>=4.13.0",
    "numpy>=2.1",
    "scipy>=1.14",
    "pillow>=11.0",
    "pydantic>=2.8",
    "typer>=0.15",
    "rich>=13.0",
    "pyyaml>=6.0",
    "timm>=1.0.20",
    "huggingface-hub",
]

[project.optional-dependencies]
models = [
    "transformers>=4.38",        # For kawaimasa model
    "dghs-imgutils[gpu]",        # For deepghs models (includes onnxruntime)
]
dev = [
    "pytest",
    "mypy",
    "ruff",
]

[tool.uv.sources]
torch = [
    { index = "pytorch-cu128", marker = "sys_platform == 'linux' or sys_platform == 'win32'" },
]
torchvision = [
    { index = "pytorch-cu128", marker = "sys_platform == 'linux' or sys_platform == 'win32'" },
]

[[tool.uv.index]]
name = "pytorch-cu128"
url = "https://download.pytorch.org/whl/cu128"
explicit = true
```

### Install Sequence

```bash
# 1. Install uv (if not already installed)
# https://docs.astral.sh/uv/getting-started/installation/

# 2. Create project and sync environment
uv sync

# 3. Install with model dependencies
uv sync --extra models

# 4. Verify CUDA availability
uv run python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"

# 5. Verify open_clip
uv run python -c "import open_clip; print(open_clip.__version__)"

# 6. Verify ONNX runtime GPU (if using imgutils)
uv run python -c "import onnxruntime; print(onnxruntime.get_device())"
```

### Alternative: CLI-based install

```bash
uv pip install torch torchvision --torch-backend=cu128
uv pip install open-clip-torch timm opencv-python-headless
```

Or with automatic GPU detection:

```bash
uv pip install torch --torch-backend=auto
```

### Known Considerations

1. **Index configuration is required for CUDA wheels.** PyPI only hosts CPU-only torch wheels for Windows. The `tool.uv.sources` + `tool.uv.index` pattern routes torch/torchvision to the PyTorch CUDA index.
2. **`explicit = true` is important.** Without it, uv would search the PyTorch index for all packages, which is slow and can cause resolution failures.
3. **Platform markers ensure macOS compatibility.** If anyone runs Loupe on macOS (CPU-only), the markers fall through to PyPI's CPU wheels gracefully.
4. **`uv sync` resolves and locks all dependencies deterministically.** The lockfile (`uv.lock`) captures exact versions, including the CUDA-specific torch wheels. This ensures reproducible environments.

---

## 7. Compatibility Issues and Workarounds

### Issue 1: open_clip Python 3.13 not in classifiers

**Severity:** Low
**Description:** open_clip 3.3.0 lists Python 3.9–3.12 in PyPI classifiers but specifies `requires-python >= 3.9`. Python 3.13 works because open_clip has no native extensions — it's pure Python on top of PyTorch/timm.
**Workaround:** None needed. If a future version adds a `<3.13` bound (unlikely), pin to 3.3.0.

### Issue 2: anime-segmentation dependency chain

**Severity:** Medium
**Description:** `skytnt/anime-segmentation` requires `pytorch_lightning`, `kornia`, `timm`, `huggingface_hub>=0.22`. The `pytorch_lightning` dependency is heavy (~50+ transitive dependencies) for inference-only use.
**Workaround:** Two options:

1. Load the ONNX export via onnxruntime instead of the PyTorch model. Avoids the pytorch_lightning/kornia chain entirely.
2. Extract the model class and weights-loading code into Loupe directly (ISNet architecture is well-documented). Eliminates external dependency.
**Recommendation:** Use the ONNX export path. This aligns with the deepghs model loading strategy (also ONNX) and avoids pulling in pytorch_lightning.

### Issue 3: Waifu Scorer v3 undocumented architecture

**Severity:** Medium
**Description:** The `model.safetensors` file (11.2 MB) contains MLP weights but no published inference code or architecture description. The MLP dimensions must be reverse-engineered from weight tensor shapes in the safetensors file.
**Workaround:** Inspect safetensors metadata: `safetensors.safe_open("model.safetensors", framework="pt")` to get tensor names and shapes. Reconstruct the MLP architecture. Alternatively, examine the 4 HuggingFace Spaces using this model for working inference code.
**Risk:** If the architecture cannot be reliably reconstructed, deprioritize this model in favor of the better-documented alternatives (deepghs/anime_aesthetic, kawaimasa).

### Issue 4: ONNX Runtime + PyTorch CUDA coexistence

**Severity:** Low
**Description:** Both PyTorch and onnxruntime-gpu use CUDA. They maintain separate CUDA contexts, so VRAM usage is additive. There is no sharing of GPU memory between them.
**Workaround:** Account for both in VRAM budget. PyTorch CUDA context: ~300–500 MB. ONNX CUDA context: ~200–300 MB. Combined overhead: ~500–800 MB. This is already within the 6.5–7.0 GB working budget.
**Mitigation:** If VRAM becomes tight, run ONNX models on CPU (detection and aesthetic scoring are fast enough on CPU for single-image inference). Only CLIP and WD-Tagger need GPU for acceptable throughput on large batches.

### Issue 5: No blocking compatibility issues found

**Severity:** None
**Description:** Python 3.13 is supported by all core dependencies. PyTorch 2.11 has stable CUDA 12.8 support on Windows. All candidate models fit in 8 GB VRAM. `uv` handles the CUDA index routing.

---

## 8. Validated Install Sequence (Step-by-Step)

For a clean Windows 11 machine with an RTX 3070:

### Prerequisites

1. **Python 3.13** — Install from python.org or via `winget install Python.Python.3.13`
2. **NVIDIA Driver** — 560+ (any recent Game Ready or Studio driver supports CUDA 12.8)
3. **uv** — Install via `powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"` or `pip install uv`

### Project Setup

```bash
# Clone the repository
git clone <loupe-repo-url>
cd loupe

# Create virtual environment and install all dependencies
uv sync --extra models --extra dev

# Verify the installation
uv run python -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'CUDA version: {torch.version.cuda}')
print(f'GPU: {torch.cuda.get_device_name(0)}')
print(f'VRAM: {torch.cuda.get_device_properties(0).total_mem / 1024**3:.1f} GB')

import open_clip
print(f'open_clip: {open_clip.__version__}')

import cv2
print(f'OpenCV: {cv2.__version__}')

import timm
print(f'timm: {timm.__version__}')
"
```

### Expected Output

```plaintext
PyTorch: 2.11.0+cu128
CUDA available: True
CUDA version: 12.8
GPU: NVIDIA GeForce RTX 3070
VRAM: 8.0 GB
open_clip: 3.3.0
OpenCV: 4.13.0
timm: 1.0.26
```

### CUDA Toolkit

**Not required for inference.** PyTorch bundles its own CUDA runtime. If building custom CUDA extensions in the future, install the CUDA Toolkit matching the PyTorch CUDA version (12.8).

---

## 9. VRAM Budget Summary

### Per-Model Allocations (FP16 where applicable)

| Model | Weight Size | Est. VRAM (inference) | Loading Framework |
| --- | --- | --- | --- |
| OpenCLIP ViT-L/14 | ~850 MB | ~1.5 GB | open_clip (PyTorch) |
| WD-Tagger v3 SwinV2-Base | 392 MB | ~1.0–1.5 GB | timm (PyTorch) |
| deepghs/anime_aesthetic SwinV2 | ~130 MB | ~0.3 GB | imgutils (ONNX) |
| kawaimasa aesthetic scorer | ~400 MB | ~1.0 GB | transformers (PyTorch) |
| skytnt/anime-segmentation | ~170 MB | ~0.5 GB | ONNX or PyTorch |
| deepghs face/head/person detection | ~60 MB each | ~0.3 GB | imgutils (ONNX) |
| Waifu Scorer v3 MLP | 11 MB | negligible (shares CLIP) | PyTorch |
| Improved Aesthetic Predictor MLP | ~3 MB | negligible (shares CLIP) | PyTorch |
| Depth Anything V2 Small | ~100 MB | ~0.5 GB | PyTorch |

### Runtime Overhead

| Component | Est. VRAM |
| --- | --- |
| PyTorch CUDA context | ~300–500 MB |
| ONNX Runtime CUDA context | ~200–300 MB |
| OS/Display | ~1.0–1.5 GB |

### Total Budget Scenarios

| Scenario | Models | Est. Total VRAM | Headroom |
| --- | --- | --- | --- |
| **Minimal** | CLIP + WD-Tagger SwinV2 + anime_aesthetic | ~3.8 GB | ~4.2 GB |
| **Recommended** | Minimal + kawaimasa + anime-seg + detection | ~5.9 GB | ~2.1 GB |
| **Full** | Recommended + Depth Anything V2 Small | ~6.4 GB | ~1.6 GB |
| **Maximum** | Full with EVA02-Large WD-Tagger instead of SwinV2 | ~7.4–7.9 GB | ~0.1–0.6 GB |

**Conclusion:** The "Recommended" configuration fits comfortably. The "Full" configuration fits with margin. The "Maximum" configuration is tight and may require sequential loading or FP16 inference optimization. **Empirical validation during implementation will refine these estimates.**

---

## 10. Stack Changes from CLAUDE.md Proposals

| CLAUDE.md Proposal | Validation Result | Action |
| --- | --- | --- |
| Python 3.13+ | Fully supported by all components | Keep |
| PyTorch + CUDA | PyTorch 2.11 stable, CUDA 12.8 recommended | Keep, specify CUDA 12.8 |
| open_clip | 3.3.0 works on 3.13 despite classifier lag | Keep |
| opencv-python-headless | 4.13.0, explicit 3.13 support | Keep |
| Pydantic v2 | Requires >= 2.8.0 for Python 3.13 | Keep, pin >= 2.8 |
| Typer | Compatible | Keep |
| Rich | Compatible | Keep |
| PyYAML | Compatible | Keep |
| uv | First-class PyTorch CUDA support | Keep |
| ruff | Compatible (pure Rust binary) | Keep |
| mypy | Compatible | Keep |
| pytest | Compatible | Keep |

**No stack changes needed.** All proposed tools and libraries are compatible with the target environment.

### Additions to Consider

| Addition | Rationale | Priority |
| --- | --- | --- |
| `timm` | Required for WD-Tagger v3 loading via PyTorch. Already a transitive dep of open_clip. | Must — add as explicit dependency |
| `transformers` | Required for kawaimasa model. Only needed if this model is adopted. | Should — add as optional dependency |
| `dghs-imgutils[gpu]` | Required for deepghs models. Includes onnxruntime. | Should — add as optional dependency |
| `safetensors` | For loading Waifu Scorer v3 model. Already a transitive dep of transformers/huggingface-hub. | No action needed |
| `huggingface-hub` | Model downloading and caching. Already a transitive dep. | No action needed — add as explicit if used directly |

---

## Limitations of This Research

1. **All VRAM estimates are theoretical.** Based on parameter counts and FP16 weight sizes, not measured on hardware. Actual measurements during model loading will differ due to CUDA context overhead, memory fragmentation, and inference activation buffers. The estimates provide confidence in feasibility, not exact numbers.

2. **Model inference correctness not validated.** This research confirms that models can be *loaded* on the target stack. Whether they produce *correct and useful* outputs on anime screenshots is an evaluation task for implementation, not a stack validation question.

3. **Batch inference throughput not measured.** Single-image inference VRAM is estimated. Batch processing (the primary use case) may require different memory strategies. Batch size should be configurable and start at 1 for safety.

4. **Dependency resolution not tested end-to-end.** The `pyproject.toml` configuration is based on documented uv behavior, not a test run. The first `uv sync` on the actual project will be the true validation. Pin major versions early to avoid surprise breakage.

5. **Community forums not surveyed.** Reddit, Discord, and CivitAI may contain reports of Windows-specific issues with these models that aren't captured in official documentation.

---

## References

- [PyTorch 2.11 CUDA support matrix RFC — pytorch/pytorch#172663](https://github.com/pytorch/pytorch/issues/172663)
- [PyTorch on PyPI](https://pypi.org/project/torch/)
- [open-clip-torch on PyPI](https://pypi.org/project/open-clip-torch/)
- [opencv-python-headless on PyPI](https://pypi.org/project/opencv-python-headless/)
- [timm on PyPI](https://pypi.org/project/timm/)
- [onnxruntime-gpu on PyPI](https://pypi.org/project/onnxruntime-gpu/)
- [uv PyTorch integration guide](https://docs.astral.sh/uv/guides/integration/pytorch/)
- [Pydantic Python 3.13 issue — pydantic/pydantic#11524](https://github.com/pydantic/pydantic/issues/11524)
- [Python 3.13 readiness tracker](https://pyreadiness.org/3.13/)
- [NVIDIA CUDA GPU compute capability](https://developer.nvidia.com/cuda/gpus)
- [SmilingWolf/wd-swinv2-tagger-v3](https://huggingface.co/SmilingWolf/wd-swinv2-tagger-v3)
- [deepghs/anime_aesthetic](https://huggingface.co/deepghs/anime_aesthetic)
- [kawaimasa/kawai-aesthetic-scorer-convnextv2](https://huggingface.co/kawaimasa/kawai-aesthetic-scorer-convnextv2)
- [Eugeoter/waifu-scorer-v3](https://huggingface.co/Eugeoter/waifu-scorer-v3)
- [SkyTNT/anime-segmentation](https://github.com/SkyTNT/anime-segmentation)
- [deepghs/imgutils](https://github.com/deepghs/imgutils)
- [CUDA Toolkit 12.8 Downloads](https://developer.nvidia.com/cuda-12-8-0-download-archive)
