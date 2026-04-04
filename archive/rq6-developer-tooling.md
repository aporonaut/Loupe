# RQ6 — Developer Tooling and Libraries

**Date:** 2026-04-03
**Status:** Complete

## Research Question

What developer-facing tools and libraries best support Loupe's development workflow, and are there better alternatives to what's currently proposed?

## Scope and Constraints

- **Target environment:** Windows 11, Python 3.13+, NVIDIA RTX 3070 (8 GB VRAM), CUDA
- **Heavy dependencies:** PyTorch (CUDA), OpenCV, open_clip, NumPy/SciPy
- **Project type:** CLI tool with batch image processing, Pydantic v2 data models, `src/loupe/` layout
- **Proposed stack (from CLAUDE.md):** uv, Typer, Rich, PyYAML+Pydantic, pytest, Pillow+OpenCV, mypy, ruff

Each section evaluates the proposed tool, considers alternatives, and makes a recommendation with rationale.

## Methodology

Research conducted via web searches across official documentation, PyPI, GitHub repositories (issue trackers, release history, commit activity), community forums (Python Discourse, Hacker News), and practitioner articles. Each category was researched independently with 3–8 search queries and 4–10 source page fetches per topic. Claims from training knowledge are marked `[T]` and should be verified before acting on them. All other claims are sourced from web research conducted on the date above.

---

## 1. Package Management: uv

**Proposed:** uv (Astral)

### Packages Current State

uv is at version 0.11.3 (2026-04-01), with 82,600+ GitHub stars and 14,800+ dependent projects [^1]. The 0.x version is a deliberate policy choice — the CLI interface is considered stable and the tool is widely used in production [^2]. Commercially backed by Astral (same company behind ruff), with multiple releases per month.

### PyTorch + CUDA on Windows

This is the highest-risk area. PyTorch publishes CUDA-enabled wheels on dedicated indexes (e.g., `https://download.pytorch.org/whl/cu128`), not PyPI. uv handles this via `tool.uv.sources` + `tool.uv.index` in `pyproject.toml` [^3]:

```toml
[project]
dependencies = [
    "torch>=2.9.1",
    "torchvision>=0.24.1",
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

**Key gotchas:**

- Every PyTorch ecosystem package needing CUDA must be listed individually in `tool.uv.sources` — omitting one causes silent fallback to CPU-only PyPI builds [^3]
- `uv add --index` for PyTorch packages can fail on Windows (GitHub #11532 [^4]) — manual `pyproject.toml` editing is more reliable
- `uv run` may resolve CPU-only torch if CUDA index routing is gated behind optional extras (GitHub #18478 [^5]) — structure dependencies so CUDA is not optional
- Always verify CUDA is active after install with `torch.cuda.is_available()`

**Future:** PyTorch 2.8+ introduced experimental wheel variant support (PEP 817 [^6], Draft status). Astral co-developed this and ships an experimental variant-enabled uv build [^7]. When PEP 817 matures, `uv pip install torch` will auto-select the right CUDA variant without custom index configuration. Not production-ready yet (likely 6–12 months away).

### Package Management Alternatives

| Tool | PyTorch CUDA on Windows | Notes |
| ------ | ------------------------ | ------- |
| pip + venv | Reliable — PyTorch's official install path | No lockfile, no project management, slower |
| Poetry | Active bugs with PyTorch CUDA resolution (GitHub #10443 [^8], PyTorch #152121 [^9]) | **Avoid for this use case** |
| PDM | Works but thin Windows+CUDA community coverage | `pdm-plugin-torch` available |
| Conda/Mamba | Excellent CUDA support — manages CUDA toolkit itself | Different ecosystem, heavier |

### Package Management Recommendation

**Keep uv.** It handles PyTorch+CUDA on Windows once the `pyproject.toml` is correctly configured. The configuration requires care (every CUDA package must be explicitly routed), but the documentation is thorough and the pattern is well-established. The speed advantage (10–100x over pip [^1]) is meaningful for a project with heavy binary dependencies. Poetry should be avoided for this stack. pip is the safe fallback if uv proves problematic.

**Action items:**

- Configure `tool.uv.sources` and `tool.uv.index` as shown above from the start
- Do not use `uv add --index` for PyTorch packages — edit `pyproject.toml` manually
- Add a post-install verification step (`python -c "import torch; print(torch.cuda.is_available())"`)

---

## 2. CLI Framework: Typer

**Proposed:** Typer

### CLI Current State

Typer is at version 0.24.1 (2026-02-21), under the FastAPI GitHub organization (maintained by Sebastián Ramírez). 19,144 GitHub stars, MIT licensed, commits landing daily [^10]. Python 3.13 and 3.14 supported. Rich is a required dependency (not optional) [^11].

Typer meets Loupe's core requirements: subcommands, type-annotation-driven API, path validation via `typer.Argument`/`typer.Option`, and Rich is bundled for help rendering.

**Limitations relevant to Loupe:**

- **No built-in config file support.** The `--config` flag pattern requires `typer-config` (third-party, 39 GitHub stars, v1.5.1, actively maintained [^12]). It provides `@use_yaml_config()` / `@use_toml_config()` decorators that add a `--config` option with CLI-override semantics.
- **No Union type support** in parameter annotations — inherited from Click. Unlikely to affect Loupe's parameter surface.
- **Progress bars are not a Typer feature** — use `rich.progress` directly in command functions. Typer exposes a legacy `typer.progressbar()` but recommends Rich directly [^13].
- Still pre-1.0 after 6+ years, though the API has been stable in practice.

### Primary Alternative: cyclopts

cyclopts (v4.10.1, 2026-03-23) is a ground-up CLI framework inspired by Typer but addressing its limitations [^14]. 1,110 GitHub stars, Apache 2.0 licensed, very actively developed (10 releases in 6 weeks, v5 alpha in progress).

**Advantages over Typer:**

- **Built-in config file support** — `cyclopts.config.Yaml` and `.Toml` as first-class features, no third-party package needed [^15]
- **Full Union and Literal type support** — handles `Union[int, str]` natively where Typer raises `AssertionError`
- **Docstring-driven help** — automatically parses NumPy, Google, and reST docstrings to populate help text [^16]. Aligns with Loupe's NumPy docstring convention
- **Pydantic model support** as parameter types
- **No Click dependency** — independent implementation, no inherited limitations

**Risks:**

- Smaller community (1.1k vs 19k stars) — fewer tutorials, SO answers, third-party integrations
- v5 breaking change cycle is underway — API still evolving
- Less battle-tested in production

### Other Alternatives

- **Click** (8.3.1): Typer's foundation. Decorator-based API is more verbose, no type-annotation-driven parameters. No advantage over Typer for Loupe's use case.
- **argparse** (stdlib): No Rich integration, maximum boilerplate, no compelling advantage when heavier dependencies are already accepted.

### CLI Recommendation

**Keep Typer**, with cyclopts as a noted alternative worth revisiting if Typer's config file gap or maintenance trajectory becomes problematic.

Rationale: Typer's ecosystem maturity, community size, and stable maintenance under the FastAPI organization make it the lower-risk choice. The config file gap is covered by `typer-config`. cyclopts is technically superior in several areas (config, docstring parsing, type support) but the smaller community and active breaking-change cycle (v4→v5) make it a higher-risk dependency for a greenfield project that will lean on community resources.

If the project finds `typer-config` insufficient or Typer's maintenance falters, cyclopts is the clear migration target.

**Action items:**

- Add `typer` and `typer-config` as dependencies
- Use `rich.progress.Progress` directly for batch processing display (not `typer.progressbar()`)

---

## 3. Configuration Format

**Proposed:** PyYAML + Pydantic v2

### Assessment

Loupe's configuration covers per-analyzer settings (enabled/disabled, confidence thresholds, analyzer-specific parameters), dimension weights, model selection, and batch processing options. This is relatively flat, simple nesting — no anchors, multi-document streams, or complex key types needed.

### TOML as Alternative

TOML is the Python ecosystem standard for configuration (`pyproject.toml`). `tomllib` is in the standard library since Python 3.11 (read-only, zero dependencies). `pydantic-settings` v2.13+ has native TOML support via `TomlConfigSettingsSource` using stdlib `tomllib` — no extra parsing library needed [^17].

The Pydantic + TOML integration pattern:

```python
from pydantic_settings import BaseSettings, SettingsConfigDict

class LoupeConfig(BaseSettings):
    model_config = SettingsConfigDict(toml_file='config/default.toml')
```

Multiple files can be provided as a list, merged in priority order — mapping directly to Loupe's layered config design (default → user → CLI overrides) [^18].

### Security Consideration

YAML deserialization vulnerabilities remain a live concern. CVE-2025-50460 demonstrated ongoing exploitation of unsafe `yaml.load()` in production code [^19]. While `yaml.safe_load()` mitigates this, it requires developer discipline at every call site. TOML has no equivalent attack surface — the format specification does not support arbitrary object construction.

### Configuration Recommendation

**Switch from PyYAML to TOML.**

Rationale:

- Loupe's config complexity does not require YAML-specific features
- `tomllib` is stdlib — zero parsing dependencies
- `pydantic-settings` native TOML support makes the integration cleaner than PyYAML + manual Pydantic loading
- No deserialization security surface
- Consistent with `pyproject.toml` already in the project
- If Loupe needs to generate config files programmatically, `tomli-w` is a small, focused dependency

**Action items:**

- Add `pydantic-settings` as a dependency (replaces PyYAML)
- Use `.toml` extension for config files (`config/default.toml`)
- Update CLAUDE.md to reflect TOML instead of YAML

---

## 4. Output Formatting: Rich

**Proposed:** Rich

### Formatting Current State

Rich is at version 14.3.3 (2026-02-xx), with Python 3.13 and 3.14 support confirmed. Actively maintained by Textualize [^20].

### Feature Coverage

Rich covers all of Loupe's display needs:

- **Progress bars with ETA**: `rich.progress` supports customizable columns (elapsed, ETA, speed, percentage) and multiple concurrent progress bars
- **Table output**: `rich.table` for formatted ranked results
- **Colored text**: Full color/style support via `rich.console` and `rich.text`

### Output Formatting Alternatives

| Need | Alternative | Trade-off |
| ------ | ------------ | ----------- |
| Progress bars | tqdm | ~5x faster update loop, but less visual customization. Irrelevant for Loupe — each iteration is seconds of analysis, not microseconds |
| Tables | tabulate | Simple, zero-dependency. No color, no live update |
| Colors | colorama / raw ANSI | Manual, no rich markup |

Using alternatives means pulling 2–3 separate libraries to replace what Rich does as one package.

### Dependency Cost

Rich is a **required dependency** of Typer (the `typer-slim` variant without Rich has been discontinued [^11]). Using Rich for progress bars and tables adds zero additional dependencies — it is already in the dependency tree.

### Output Formatting Recommendation

**Keep Rich.** It is already a transitive dependency of Typer, covers all display needs in a single API, and is actively maintained. No action needed beyond using `rich.progress` and `rich.table` in the CLI layer.

---

## 5. Testing Stack

**Proposed:** pytest

### Core Framework

pytest is uncontested for Python testing in 2026. No alternative warrants consideration.

### Recommended Plugin Set

| Plugin | Purpose | Rationale |
| -------- | --------- | ----------- |
| **pytest-benchmark** (v5.2.3) | Analyzer inference time tracking | Calibrates iterations, collects statistical timing data (min/max/mean/stddev), stores results as JSON for historical comparison. Default wall-clock timer is appropriate for "how long does the user wait" measurement. For precise GPU kernel timing, supports custom timers via `--benchmark-timer` [^21] |
| **pytest-xdist** | Parallel test execution | Analyzer tests are naturally independent. `pytest -n auto` parallelizes CPU-only unit tests. GPU tests should be excluded from parallel runs (single GPU contention) [^22] |
| **pytest-cov** (v7.1.0) | Coverage collection | Preferred over raw `coverage` when using pytest-xdist — handles merging coverage from multiple workers automatically [^23] |

### Deferred / Conditional Plugins

| Plugin | Purpose | When to add |
| -------- | --------- | ------------- |
| **pytest-datadir** / **pytest-datadir-ng** | Per-test fixture isolation | Add if tests need to modify fixture images. For read-only reference images, a simple conftest fixture pointing to `tests/fixtures/` is sufficient |
| **pytest-arraydiff** | NumPy array comparison with tolerance | Add when testing analyzer feature extraction consistency against stored reference values |
| **pytest-randomly** | Randomize test order | Add to verify test isolation — especially useful with xdist |

### GPU/PyTorch Mocking Strategy

Separate tests into two tiers:

**Unit tests (fast, no GPU):** Mock model loading with `pytest.monkeypatch` or `unittest.mock.patch`. Use pre-computed `AnalyzerResult` fixtures to test engine orchestration, scoring, aggregation, and sidecar I/O. These run with xdist.

```python
# Example: patch CUDA availability
monkeypatch.setattr("torch.cuda.is_available", lambda: False)
```

**Integration tests (slow, GPU optional):** Run real analyzers on reference images. Mark with `@pytest.mark.gpu` or `@pytest.mark.slow`. Skip when GPU is unavailable:

```python
@pytest.mark.skipif(not torch.cuda.is_available(), reason="No GPU")
def test_style_analyzer_real_inference(sample_image):
    ...
```

### Testing Recommendation

**Keep pytest. Add pytest-benchmark, pytest-xdist, and pytest-cov from the start.** Add pytest-datadir, pytest-arraydiff, and pytest-randomly as the test suite matures and their value becomes concrete.

---

## 6. Image Handling: Pillow + OpenCV

**Proposed:** Both Pillow and OpenCV (`opencv-python-headless`)

### Division of Responsibility

| Capability | Pillow | OpenCV |
| ----------- | -------- | -------- |
| Broad format support (JPEG, PNG, TIFF, WebP, etc.) | Strong | Limited |
| EXIF/metadata extraction | Yes (`Image.getexif()`) | No — `imread` discards metadata |
| CV algorithms (edge detection, saliency, frequency analysis) | No | Yes |
| Color space conversions (HSV, LAB, YCrCb) | Limited | Comprehensive |
| NumPy ndarray native | No (requires conversion) | Yes |
| Histogram operations, morphological transforms | No | Yes |

**Can OpenCV replace Pillow?** Partially — it can load images but cannot extract metadata, which Loupe needs for `LoupeResult` (image dimensions, format).

**Can Pillow replace OpenCV?** No — Pillow lacks saliency detection, Canny edge detection, frequency-domain analysis, and advanced color-space operations. These are core to Loupe's analyzers.

### Conversion Overhead

`np.asarray(pil_image)` creates a view on PIL's internal buffer without copying data — near-zero cost [^24]. The only overhead is an RGB→BGR channel swap if needed for OpenCV functions (`array[:, :, ::-1]` or `cv2.cvtColor()`). Negligible for per-image analysis.

### Python 3.13 Windows Compatibility

- **opencv-python-headless 4.13.0.92** (2026-02-05): Uses `cp37-abi3` stable ABI tag — works with any CPython ≥ 3.7 including 3.13. Windows amd64 wheel available [^25].
- **Pillow 12.2.0**: Dedicated `cp313` wheels for Windows win32, win_amd64, and win_arm64 [^26].

Both libraries are fully compatible with the target environment.

### Recommended Loading Pattern

```plaintext
Pillow (Image.open) → extract metadata → np.asarray() → RGB→BGR if needed → OpenCV operations
```

**Rationale:** Pillow handles format detection and metadata extraction in one step. `np.asarray()` is near-zero cost. The image enters the system once; all analyzers receive the same ndarray. Avoids OpenCV `imread` quirks (silent `None` on bad paths, limited format support, no metadata access).

### Image Handling Recommendation

**Keep both.** The division is clear: Pillow owns image loading and metadata extraction; OpenCV owns all CV operations. Conversion overhead is negligible. Both have confirmed Python 3.13 Windows wheels.

**Action items:**

- Define the loading boundary in `src/loupe/io/image.py`: Pillow loads, extracts metadata, converts to ndarray once
- All analyzers receive ndarray (not PIL.Image) — no per-analyzer format conversion

---

## 7. Type Checking

**Proposed:** mypy

### mypy Current State (v1.20.0, 2026-03-31)

- Python 3.13 fully supported with mypyc-compiled binary wheels [^27]
- Significant performance improvements in v1.18+ (~40% speedup) and v1.20 (binary cache, SQLite cache) [^28]
- Still 3–5x slower than pyright on large codebases [^29]
- **Pydantic v2 plugin:** Exists but has ongoing stability issues — crashes with generic models and forward-referenced constraints [^30], regressions between Pydantic minor versions [^31], false positives with `extra="forbid"` in recursive models [^32]. Straightforward Pydantic usage works; complex patterns can trigger crashes.
- **PyTorch typing:** Partial. Tensor operator result types are not always correctly inferred (`x**x`, `x//x` require `# type: ignore`) [^33]. This is a PyTorch annotation problem, not a mypy problem.

### pyright Current State (v1.1.408+)

- Python 3.13 fully supported [^34]
- 3–5x faster than mypy — TypeScript-based, lazy/JIT type evaluator [^29]
- **Pydantic v2 support without a plugin** — leverages `__dataclass_transform__` (PEP 681). Handles constructor signatures, field types, model inheritance. Some dynamic Pydantic features (validators, computed fields) are not fully visible, but sufficient for Loupe's straightforward data models [^35]. No plugin means nothing to break between versions.
- **PyTorch typing:** Same incomplete stubs as mypy — the gap is in PyTorch's annotations, not the checker.
- **Editor integration:** Powers Pylance for VS Code — autocompletion, hover info, go-to-definition, real-time errors. This is pyright's strongest advantage [^29].

### basedpyright (v1.38.4)

Community fork of pyright by DetachHead. 3,156 GitHub stars [^36]. Adds:

- Stricter defaults (all rules as warning/error)
- 13+ additional diagnostic rules (e.g., `reportAny` for explicit Any usage)
- Pip-installable without Node.js (vs pyright which needs Node.js or a pip wrapper)
- Baseline system for incremental adoption of strict checking [^36]

The Positron IDE team's March 2026 evaluation described it as "the most mature" of the newer type checkers [^37]. However, it depends on a single primary maintainer — a risk factor.

### Emerging Tools

- **ty** (Astral, Rust-based): Beta since December 2025, 10–60x faster than mypy/pyright without caching [^38]. Still 0.0.x, not production-ready. Plans first-class Pydantic support.
- **pyrefly** (Meta, Rust-based): Beta March 2026. Built-in Pydantic support [^39]. Neither is ready for production use today.

### Comparative Summary

| Criterion | mypy 1.20 | pyright 1.1.408 |
| ----------- | ----------- | ----------------- |
| Speed | Moderate (improved) | Fast (3–5x faster) |
| Pydantic v2 | Plugin (fragile edge cases) | Native via PEP 681 (stable) |
| PyTorch | Partial (stubs incomplete) | Partial (same stubs) |
| VS Code integration | None (pair with Pylance) | Excellent (is Pylance) |
| CI installation | `pip install mypy` | `pip install pyright` (needs Node.js) or basedpyright |
| Configuration | `pyproject.toml [tool.mypy]` | `pyproject.toml [tool.pyright]` |

### Recommendation

**Switch from mypy to pyright** (or basedpyright).

Rationale:

- **Pydantic v2 without a plugin** eliminates the fragile mypy plugin dependency — the most significant practical advantage for Loupe's Pydantic-heavy data model layer
- **3–5x faster** — meaningful for developer iteration speed
- **Pylance integration** provides real-time type checking in VS Code, which mypy cannot offer on its own
- **PyTorch typing is equally incomplete** in both — no advantage to mypy here
- **basedpyright** is worth considering over standard pyright for its pip-installable CLI (no Node.js in CI) and baseline system for incremental strictness adoption

If choosing basedpyright: accept the single-maintainer risk in exchange for easier CI installation and stricter defaults. If choosing standard pyright: install via `pip install pyright` (which wraps the npm package) or ensure Node.js is available in CI.

**Action items:**

- Add `pyright` (or `basedpyright`) as a dev dependency
- Configure in `pyproject.toml` under `[tool.pyright]`
- Update CLAUDE.md verification commands: `uv run pyright src/` instead of `uv run mypy src/`
- Use Pylance extension in VS Code for real-time feedback

---

## 8. Additional Tooling

### pre-commit

**Recommendation: Add.**

pre-commit ensures format/lint/type checks run automatically before every commit, preventing unformatted or failing code from entering the repository. Strong uv integration exists:

- Install: `uv tool install pre-commit --with pre-commit-uv` [^40]
- `pre-commit-uv` patches pre-commit to use uv for hook environment installation (faster)
- `astral-sh/ruff-pre-commit` provides official ruff hooks [^41]

Suggested hooks:

```yaml
repos:
  - repo: https://github.com/pre-commit/pre-commit-hooks
    hooks:
      - id: trailing-whitespace
      - id: end-of-file-fixer
      - id: check-toml
      - id: check-added-large-files
  - repo: https://github.com/astral-sh/ruff-pre-commit
    hooks:
      - id: ruff
        args: [--fix]
      - id: ruff-format
```

Type checking as a pre-commit hook is optional — it can be slow for large codebases. Consider running it in CI only.

### Task Runner: just

**Recommendation: Add.**

uv has no built-in task runner (second most upvoted feature request, GitHub #5903 [^42]). `just` is a Rust-based, cross-platform task runner that works well on Windows — no Make quirks, no `.PHONY`, simple syntax [^43].

```just
format:
    ruff format .

lint:
    ruff check .

typecheck:
    uv run pyright src/

test:
    uv run pytest

verify: format lint typecheck test
```

This standardizes the "Key Commands" from CLAUDE.md as executable definitions.

### Documentation Generation

**Recommendation: Defer.**

Loupe is in the research phase with no public API surface. When the API stabilizes, **mkdocs + mkdocs-material + mkdocstrings** is the lighter-weight, more modern choice. mkdocstrings' Griffe parser handles NumPy-style docstrings natively [^44] — aligning with Loupe's docstring convention. Sphinx + napoleon is the mature alternative but requires RST authoring.

### Dependency Scanning: pip-audit

**Recommendation: Add (in CI).**

pip-audit (by Trail of Bits) uses the transparent PyPA advisory database, is fully open source, and has 45% better vulnerability recall than safety per 2025 OpenSSF Scorecard benchmarks [^45]. Useful for a project with PyTorch, OpenCV, and other heavy binary dependencies with large transitive dependency trees. Integrates as a pre-commit hook or CI step.

---

## Summary Recommendation Table

| Category | Proposed | Recommendation | Action |
| ---------- | ---------- | --------------- | -------- |
| Package management | uv | **Keep** | Configure `tool.uv.sources` for PyTorch CUDA index routing |
| CLI framework | Typer | **Keep** | Add `typer-config` for `--config` support. Note cyclopts as future alternative |
| Configuration | PyYAML + Pydantic | **Switch to TOML + pydantic-settings** | Eliminates PyYAML dependency; uses stdlib `tomllib` + native Pydantic integration |
| Output formatting | Rich | **Keep** | Already a Typer transitive dependency; zero additional cost |
| Testing | pytest | **Keep + add plugins** | Add pytest-benchmark, pytest-xdist, pytest-cov from the start |
| Image handling | Pillow + OpenCV | **Keep both** | Pillow loads + metadata; OpenCV for CV operations. Define boundary in `io/image.py` |
| Type checking | mypy | **Switch to pyright** (or basedpyright) | Faster, plugin-free Pydantic support, Pylance integration |
| Additional: pre-commit | (not proposed) | **Add** | Enforce ruff format/lint before commits |
| Additional: task runner | (not proposed) | **Add just** | Cross-platform task runner for dev commands |
| Additional: docs | (not proposed) | **Defer** | mkdocs + mkdocstrings when API stabilizes |
| Additional: dep scanning | (not proposed) | **Add pip-audit** | CI step for vulnerability scanning |

## Open Questions

1. **pyright vs basedpyright:** The single-maintainer risk of basedpyright should be weighed against its practical advantages (pip-installable without Node.js, baseline system). Either is a sound choice over mypy for this stack.

2. **Typer vs cyclopts:** If `typer-config` proves insufficient for Loupe's layered config needs (default → user → CLI override), cyclopts' built-in config support makes it the natural migration target.

3. **uv wheel variants (PEP 817):** When this matures, the PyTorch CUDA index configuration in `pyproject.toml` can be simplified significantly. Monitor PEP status.

4. **ty (Astral's Rust-based type checker):** Currently beta (0.0.x). If it reaches stability, it would be a natural complement to ruff and uv in the Astral toolchain. Worth revisiting in 6–12 months.

---

## References

[^1]: astral-sh/uv GitHub repository. <https://github.com/astral-sh/uv>
[^2]: "Versioning," uv documentation. <https://docs.astral.sh/uv/reference/policies/versioning/>
[^3]: "Using uv with PyTorch," Astral official documentation. <https://docs.astral.sh/uv/guides/integration/pytorch/>
[^4]: "`uv add --index` fails to install PyTorch CUDA packages on Windows," GitHub issue #11532. <https://github.com/astral-sh/uv/issues/11532>
[^5]: "When running, uv run reinstalls torch to a version without [CUDA]," GitHub issue #18478. <https://github.com/astral-sh/uv/issues/18478>
[^6]: PEP 817, "Wheel Variants: Beyond Platform Tags." <https://peps.python.org/pep-0817/>
[^7]: "An experimental, variant-enabled build of uv," Astral blog, August 2025. <https://astral.sh/blog/wheel-variants>
[^8]: "Poetry does not install torch as defined in extra-group," GitHub issue #10443. <https://github.com/python-poetry/poetry/issues/10443>
[^9]: "[poetry] 2.7.0+cpu includes cuda as a dependency," GitHub issue #152121. <https://github.com/pytorch/pytorch/issues/152121>
[^10]: Typer on PyPI. <https://pypi.org/project/typer/>
[^11]: Typer PR #1522: typer-slim now requires Rich. <https://github.com/fastapi/typer/pull/1522>
[^12]: typer-config on GitHub. <https://github.com/maxb2/typer-config>
[^13]: Typer progress bar documentation. <https://typer.tiangolo.com/tutorial/progressbar/>
[^14]: cyclopts on GitHub. <https://github.com/BrianPugh/cyclopts>
[^15]: cyclopts config file documentation. <https://cyclopts.readthedocs.io/en/latest/config_file.html>
[^16]: cyclopts vs Typer comparison. <https://cyclopts.readthedocs.io/en/latest/vs_typer/README.html>
[^17]: pydantic-settings on PyPI (v2.13.1). <https://pypi.org/project/pydantic-settings/>
[^18]: pydantic-settings TOML configuration. <https://docs.pydantic.dev/latest/concepts/pydantic_settings/>
[^19]: CVE-2025-50460: PyYAML RCE in ms-swift. <https://www.miggo.io/vulnerability-database/cve/CVE-2025-50460>
[^20]: Rich on PyPI (v14.3.3). <https://pypi.org/project/rich/>
[^21]: pytest-benchmark on PyPI (v5.2.3). <https://pypi.org/project/pytest-benchmark/>
[^22]: pytest-xdist on PyPI. <https://pypi.org/project/pytest-xdist/>
[^23]: pytest-cov distributed testing with xdist. <https://pytest-cov.readthedocs.io/en/latest/xdist.html>
[^24]: "Fast Pillow image import to NumPy and OpenCV arrays," Uploadcare. <https://uploadcare.com/blog/fast-import-of-pillow-images-to-numpy-opencv-arrays/>
[^25]: opencv-python-headless 4.13.0.92 on PyPI. <https://pypi.org/project/opencv-python-headless/>
[^26]: Pillow 12.2.0 on PyPI. <https://pypi.org/project/Pillow/>
[^27]: Mypy 1.20 Released, Mypy Blog. <https://mypy-lang.blogspot.com/2026/03/mypy-120-released.html>
[^28]: Mypy Blog 2025, performance improvements. <https://mypy-lang.blogspot.com/2025/>
[^29]: Pyright mypy comparison. <https://github.com/microsoft/pyright/blob/main/docs/mypy-comparison.md>
[^30]: Pydantic issue #12185, mypy plugin crash with generic constraints. <https://github.com/pydantic/pydantic/issues/12185>
[^31]: Pydantic issue #11727, regression with type alias and root model. <https://github.com/pydantic/pydantic/issues/11727>
[^32]: Pydantic issue #11329, false "Unexpected keyword argument" errors. <https://github.com/pydantic/pydantic/issues/11329>
[^33]: PyTorch issue #145838, operator type inference gaps. <https://github.com/pytorch/pytorch/issues/145838>
[^34]: Pyright releases on GitHub. <https://github.com/microsoft/pyright/releases>
[^35]: Pydantic VS Code integration docs. <https://docs.pydantic.dev/latest/integrations/visual_studio_code/>
[^36]: basedpyright on GitHub. <https://github.com/DetachHead/basedpyright>
[^37]: "How we chose Positron's Python type checker," Positron blog, March 2026. <https://positron.posit.co/blog/posts/2026-03-31-python-type-checkers/>
[^38]: ty announcement, Astral blog. <https://astral.sh/blog/ty>
[^39]: Pyrefly experimental Pydantic support. <https://pyrefly.org/en/docs/pydantic/>
[^40]: "Using uv with pre-commit," uv documentation. <https://docs.astral.sh/uv/guides/integration/pre-commit/>
[^41]: astral-sh/ruff-pre-commit on GitHub. <https://github.com/astral-sh/ruff-pre-commit>
[^42]: "Using uv run as a task runner," GitHub issue #5903. <https://github.com/astral-sh/uv/issues/5903>
[^43]: "Justfile became my favorite task runner." <https://tduyng.com/blog/justfile-my-favorite-task-runner/>
[^44]: mkdocstrings docstring parsers (Griffe). <https://mkdocstrings.github.io/griffe/reference/docstrings/>
[^45]: "Safety and pip-audit: Comparing Security Tools," Six Feet Up. <https://sixfeetup.com/blog/safety-pip-audit-python-security-tools>
