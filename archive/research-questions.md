# Loupe Research Questions

This document defines the research questions that must be answered before implementation begins. Each RQ is self-contained — a new session can pick up any RQ by reading this document alongside the project's `CLAUDE.md`.

Results for each RQ should be written to `docs/research/rq<N>-<slug>.md` (e.g., `rq1-analysis-dimensions.md`).

**Research scope note:** The question, context, and scope sections below are guiding information, not hard boundaries. Further analysis, sourcing, and readings that fall outside the stated scope but provide value towards answering the question are worthwhile and expected. Do what you need to do to research and synthesize thorough answers.

## Dependency Map

```plaintext
RQ1 (Dimensions) ──► RQ2 (Classical CV vs. Learned per dimension)
RQ3 (Anime Models) ──► RQ5 (Tooling Validation — needs candidate models)
RQ4 (Aggregate Scoring) — independent
RQ6 (Developer Tooling) — independent
```

RQ1, RQ3, RQ4, and RQ6 can all start in parallel. RQ2 requires RQ1's output. RQ5 requires RQ3's output.

---

## RQ1 — Analysis Dimensions

**Question:** What aesthetic dimensions should Loupe measure, and what specific sub-properties comprise each dimension?

**Context:** The CLAUDE.md proposes five dimensions: composition, color, detail, style, and subject. This research should validate, refine, and/or expand that list.

**Scope:**

- For each candidate dimension, define what it measures concretely for anime frames (not photography — anime has distinct visual properties like cel shading, limited color palettes, line art)
- Identify sub-properties per dimension (e.g., composition might include rule-of-thirds adherence, visual balance, depth layering, leading lines)
- Evaluate whether any proposed dimensions overlap enough to merge, or whether missing dimensions should be added (candidates: lighting quality, visual clarity/noise, line art quality, background detail vs. character detail)
- For each dimension, describe what "scores high" and "scores low" looks like in anime screenshot terms

**Output:** A validated list of dimensions with sub-properties, each with a brief rationale for inclusion. Flag any dimensions that are conceptually sound but may be infeasible to measure (to be resolved by RQ2).

**Status:** Complete — see `rq1-analysis-dimensions.md`

---

## RQ2 — Classical CV vs. Learned Models per Dimension

**Depends on:** RQ1 (needs the finalized dimension list)

**Question:** For each analysis dimension, what is the best technical approach — classical computer vision, learned/neural models, or a hybrid?

**Context:** Some dimensions (composition geometry, color harmony math) have well-established classical CV techniques. Others (style recognition, aesthetic "feel") likely require learned representations. The choice affects complexity, speed, GPU requirements, and accuracy.

**Scope:**

- For each dimension from RQ1, survey available techniques:
  - Classical CV: What algorithms apply? How reliable are they for anime content specifically? What libraries implement them?
  - Learned models: What pretrained models exist? Do they generalize to anime, or do they need anime-specific training?
  - Hybrid: Can classical features feed into a learned scorer?
- For each technique, assess:
  - Expected accuracy/reliability on anime frames
  - Computational cost (CPU-only vs. GPU-required, inference time per image)
  - Implementation complexity
  - Dependency weight (does it pull in a large framework?)
- Recommend an approach per dimension with rationale

**Output:** Per-dimension recommendation table: dimension, recommended approach, candidate techniques/models, key trade-offs, confidence level in the recommendation.

**Status:** Complete — see [rq2-classical-vs-learned.md](rq2-classical-vs-learned.md)

---

## RQ3 — Anime-Tuned Models Landscape

**Question:** What pretrained models exist that are trained on or tuned for anime/illustration content, and which are viable for Loupe's use cases?

**Context:** General-purpose vision models (CLIP, aesthetic predictors) underperform on anime style discrimination. The anime/illustration ML community (centered around Danbooru-tagged datasets) has produced specialized models. This research maps that landscape.

**Scope:**

- **CLIP fine-tunes**: Models trained on anime/Danbooru datasets. Evaluate embedding quality for style discrimination within the anime domain (not anime-vs-photo, but distinguishing aesthetic qualities between anime frames)
- **Aesthetic scoring models**: NIMA variants, CLIP-based aesthetic predictors (e.g., improved-aesthetic-predictor), any anime-specific aesthetic scorers. Do general aesthetic predictors correlate with anime screenshot quality, or do they need domain-specific training?
- **Anime tagging models**: WD-Tagger (and its versions/successors), other Danbooru-trained classifiers. Could tag outputs feed into Loupe's analysis (e.g., detecting composition elements, color mood, scene type)?
- **Other relevant models**: Saliency detection models, depth estimation models, segmentation models that work on anime art style
- For each model, document:
  - Model name, source/repository, and version
  - Training data and approach
  - License (commercial use? modification? redistribution?)
  - Model size and VRAM requirements
  - Community adoption and maintenance status
  - Known strengths and weaknesses on anime content

**Output:** A catalog of viable models organized by use case, with recommendations for which to evaluate further in RQ5.

**Status:** Complete — see `rq3-anime-models.md`

---

## RQ4 — Aggregate Scoring Strategy

**Question:** How should per-dimension scores combine into a single sortable aggregate score?

**Context:** The user's workflow is: analyze a batch of hundreds of images, sort by aggregate score, review top-down. The aggregate determines what "sorts to the top" — it's the most UX-critical computation in the system. The approach should handle the fact that different images are good for different reasons (a moody close-up vs. a sweeping landscape both deserve high placement, but for different dimensional strengths).

**Scope:**

- Survey approaches:
  - **Weighted sum**: Simple, interpretable, user-configurable weights. Risk: flattens dimensional identity — a mediocre-everywhere image can outscore an image that's exceptional on two dimensions but weak on others
  - **Geometric mean**: Penalizes zeros/low scores more than arithmetic mean — rewards balanced profiles
  - **Rank-based aggregation**: Score relative to the batch rather than absolute. Handles cross-dimension scale differences naturally
  - **Percentile-based**: Normalize each dimension to percentile within the batch, then combine
  - **Max-of-top-N**: Reward images that excel in any K dimensions rather than requiring breadth
  - **Learned combination**: Train a small model on user preferences (future scope, but worth understanding feasibility)
- For each approach, assess:
  - Does it reward "exceptional in a few dimensions" or "good across all dimensions"? Which does the user's use case favor?
  - Sensitivity to dimension count — does adding a new analyzer change rankings of existing images?
  - Configurability — can the user adjust behavior without deep technical knowledge?
  - Edge cases — how does it handle missing dimensions (analyzer disabled or failed)?
- Consider whether the aggregate should be a single number or a composite (e.g., aggregate + "profile type" tag like "composition-driven" or "color-driven")

**Output:** Recommended approach with rationale, including fallback/alternative. Define the formula or algorithm precisely enough to implement. Address configurability.

**Status:** Complete — see [rq4-aggregate-scoring.md](rq4-aggregate-scoring.md)

---

## RQ5 — Tooling and Stack Validation

**Depends on:** RQ3 (needs candidate models to validate)

**Question:** Does the proposed tech stack work correctly on the target environment, and are there compatibility issues to address early?

**Context:** The target environment is Windows 11, Python 3.13+, NVIDIA RTX 3070 (8GB VRAM), CUDA. Some combinations of Python version + PyTorch + CUDA + Windows have known issues.

**Scope:**

- **PyTorch + CUDA**: Confirm current PyTorch supports Python 3.13 on Windows with CUDA. Identify the correct install command and CUDA toolkit version
- **open_clip**: Confirm compatibility with current PyTorch and Python 3.13. Test that model loading and inference work on Windows + CUDA
- **Candidate models from RQ3**: For each recommended model, verify it can be loaded and run inference on the RTX 3070 (8GB VRAM). Document VRAM usage per model. Identify if multiple models can coexist in VRAM or need sequential load/unload
- **opencv-python-headless**: Confirm Python 3.13 wheel availability on Windows
- **Other stack components**: Pydantic v2, Typer, Rich, PyYAML — confirm no compatibility issues (these are pure Python and unlikely to have issues, but verify)
- Document the exact install sequence that produces a working environment

**Output:** Validated install sequence, VRAM budget per model, any compatibility issues found with workarounds. Flag any stack changes needed (e.g., if Python 3.13 is blocked, what's the fallback?).

**Status:** Complete — see [rq5-tooling-validation.md](rq5-tooling-validation.md)

---

## RQ6 — Developer Tooling and Libraries

**Question:** What developer-facing tools and libraries best support Loupe's development workflow, and are there better alternatives to what's currently proposed?

**Context:** The CLAUDE.md proposes a specific set of tools (ruff, mypy, pytest, uv) and libraries (Typer, Rich, Pydantic, PyYAML). This research should validate those choices and identify alternatives worth considering, as well as any additional tooling that would benefit the project.

**Scope:**

- **Package management**: `uv` is proposed. Confirm it handles PyTorch + CUDA installs cleanly on Windows (this is historically painful). Are there gotchas with uv's lockfile and GPU-specific package indices?
- **CLI framework**: Typer is proposed. Alternatives: click (Typer's foundation), cyclopts, argparse. Evaluate against needs: subcommands, progress bars (Rich integration), path validation, config file support
- **Configuration**: PyYAML is proposed for config parsing, Pydantic for validation. Consider: TOML (stdlib in 3.11+) vs. YAML — TOML avoids the PyYAML dependency and is arguably safer. Does YAML offer anything Loupe specifically needs (complex nesting, anchors)?
- **Output formatting**: Rich is proposed for CLI display. Confirm it handles the needed output: progress bars for batch processing, table output for score reports, colored/formatted text. Any lighter alternatives worth considering?
- **Testing**: pytest is proposed. Evaluate plugin needs: pytest-benchmark (for performance regression on analyzers), pytest-datadir or similar (for test fixture images), coverage tooling
- **Image handling**: Pillow is proposed alongside OpenCV. Clarify division of responsibility — when does Loupe use Pillow vs. OpenCV for image operations? Is carrying both justified, or can one handle all needs?
- **Type checking**: mypy is proposed. Consider pyright/basedpyright as an alternative — faster, better inference, stricter. What works best with the stack (PyTorch typing, Pydantic v2 plugin)?
- **Any additional tooling** that would benefit the project: pre-commit hooks, documentation generation, dependency scanning, etc.

**Output:** For each tool category, a recommendation (keep proposed choice or switch) with brief rationale. Flag any tools that need special configuration for this project's constraints (Windows, CUDA, large binary dependencies).

**Status:** Complete — see [rq6-developer-tooling.md](rq6-developer-tooling.md)
