# Loupe

Modular aesthetic analysis tool for anime screenshots. Loupe measures frames across six independent aesthetic dimensions — composition, color, detail, lighting, subject, and style — producing structured per-dimension scores and tags. The human remains the curator; Loupe provides the structured data to accelerate a review pass across large screenshot collections.

## Installation

Requires Python 3.13+ and a CUDA-capable GPU (recommended).

```bash
# Clone and install
git clone <repo-url> && cd loupe
uv sync

# Download models (~2 GB, one-time)
uv run loupe setup
```

For development (linting, type checking, tests):

```bash
uv sync --extra dev
```

Loupe uses PyTorch with CUDA 12.8. The `uv sync` command handles PyTorch index routing automatically via `pyproject.toml`.

### cuDNN note (Windows)

ONNX Runtime needs cuDNN 9.x for GPU acceleration. Loupe automatically finds the cuDNN bundled with PyTorch — no separate install needed. If you see CUDA fallback warnings, your PyTorch installation may be missing CUDA support.

## Quickstart

```bash
# Analyze a single image
uv run loupe analyze screenshot.png

# Analyze a directory
uv run loupe analyze screenshots/

# View rankings
uv run loupe rank screenshots/

# Batch summary statistics
uv run loupe report screenshots/

# Prefix filenames with rank numbers for easier review
uv run loupe rank screenshots/ --rename

# Re-rank with a different preset
uv run loupe rank screenshots/ --preset composition

# List all tags Loupe can produce
uv run loupe tags
```

### Output

Analysis results are written as JSON sidecar files in a `.loupe/` directory alongside the images:

```text
screenshots/
├── image.png
├── .loupe/
│   └── image.png.json    # Full analysis result
```

Delete `.loupe/` to cleanly remove all Loupe artifacts. Images are never modified.

## Analyzers

Each analyzer measures one aesthetic dimension, producing a score (0.0-1.0) and contextual tags.

### Composition

Spatial arrangement — rule of thirds, visual balance, symmetry, leading lines, diagonal dominance, negative space, depth layering, framing. All classical CV (OpenCV + NumPy), no model dependencies.

### Color

Palette design — Matsuda harmony scoring across 8 templates, palette extraction via K-means in OkLab color space, saturation balance, color contrast, temperature consistency, diversity, colorfulness. Fully classical.

### Detail

Visual complexity — edge density, spatial frequency, texture richness (GLCM), shading granularity, line work quality, rendering clarity. Region-separated analysis (character vs background) using anime segmentation.

### Lighting

Illumination design — contrast ratio, light directionality, rim/edge lighting, shadow quality, atmospheric effects, highlight/shadow balance. Supplemented with WD-Tagger lighting tags.

### Subject

Focal emphasis — saliency strength, figure-ground separation, depth-of-field detection, negative space utilization, subject completeness, subject scale. Requires anime segmentation and character detection models.

### Style

Artistic identity — aesthetic quality score (anime aesthetic scorer), layer consistency (experimental). Categorical tags from WD-Tagger (art style tags) and CLIP zero-shot classification (style categories). Aesthetic tier labels (masterpiece through worst).

## Configuration

Loupe uses layered TOML configuration:

1. **Defaults**: `config/default.toml`
2. **User config**: `~/.config/loupe/config.toml` (or `--config` flag)
3. **CLI overrides**: `--preset`, `--force`, etc.

### Scoring presets

Presets control the relative weight of each dimension in the aggregate score:

| Preset        | Composition | Color | Detail | Lighting | Subject | Style |
| ------------- | ----------- | ----- | ------ | -------- | ------- | ----- |
| `balanced`    | 1.0         | 1.0   | 1.0    | 1.0      | 1.0     | 0.5   |
| `composition` | 3.0         | 1.0   | 1.0    | 1.0      | 1.0     | 0.5   |
| `visual`      | 1.0         | 2.0   | 2.0    | 1.0      | 1.0     | 0.5   |

Style is weighted at 0.5 by default because the aesthetic scorer provides limited discriminative signal for intra-anime quality comparison.

### Per-analyzer configuration

Each analyzer can be enabled/disabled and configured independently:

```toml
[analyzers.color]
enabled = true
confidence_threshold = 0.25

[analyzers.color.params]
n_clusters = 6  # K-means palette clusters

[analyzers.detail.params]
bg_weight = 0.6    # Background region weight
char_weight = 0.4  # Character region weight

[analyzers.style.params]
tagger_threshold = 0.35       # WD-Tagger confidence threshold
aesthetic_weight = 0.70        # Aesthetic scorer contribution
layer_consistency_weight = 0.30  # Layer consistency contribution
```

## Models

Loupe uses several pretrained models for analysis. All are downloaded via `loupe setup` and cached locally — analysis runs fully offline after setup.

| Model | Purpose | Used by |
| ----- | ------- | ------- |
| [anime-segmentation](https://github.com/SkyTNT/anime-segmentation) (ONNX) | Character mask | Detail, Lighting, Subject, Style |
| [WD-Tagger v3](https://huggingface.co/SmilingWolf/wd-swinv2-tagger-v3) (SwinV2) | Tag prediction | Style, Lighting |
| [deepghs detection](https://huggingface.co/deepghs) (ONNX) | Face/head/person boxes | Subject |
| [deepghs aesthetic](https://huggingface.co/deepghs/anime_aesthetic) (ONNX) | Aesthetic quality | Style |
| [CLIP ViT-L/14](https://github.com/mlfoundations/open_clip) (OpenAI) | Style embeddings | Style |

Total VRAM usage: ~5.1 GB (fits RTX 3070 8 GB comfortably).

## Performance

On an RTX 3070 with CUDA, typical throughput is ~1.4 seconds per image (~170 images in 4 minutes). The time splits roughly:

- Model inference (7 passes): ~60% of per-image time
- Classical CV (color K-means, composition, detail): ~40%
- Scoring/I/O: negligible

## Known limitations

- **Style dimension has low variance** (std ~0.02 across diverse images) — the aesthetic scorer provides limited discriminative power for intra-anime comparison. Style is downweighted to 0.5 in the default preset.
- **Subject floors at 0.1 for environment shots** — when the segmentation model finds no character, subject scores `environment_focus` at 0.1. This is by design but penalizes intentional environment/object-focused compositions.
- **Segmentation fails on non-standard art styles** — painterly, watercolor, or heavily stylized frames may not have characters detected even when figures are visible.
- **Scores are not comparable across art styles** — a Kyoto Animation frame and a Madhouse frame have fundamentally different visual profiles. Rankings are most meaningful within a single title or similar style.
- **Loupe measures visual properties, not narrative significance** — a dramatically important scene with poor composition will score low. The human review pass accounts for this.

## Development

```bash
# Format
ruff format .

# Lint
ruff check .

# Type check
uv run pyright src/

# Run tests
uv run pytest

# Run benchmarks
uv run pytest tests/test_benchmarks.py --benchmark-only

# Build
uv build
```

## License

TBD
