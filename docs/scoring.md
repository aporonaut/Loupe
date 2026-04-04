# Scoring Reference

Loupe computes an aggregate score for each image by combining per-dimension analyzer scores using a Weighted Arithmetic Mean (WAM).

## Formula

Each enabled analyzer produces an independent score `s_i` in [0.0, 1.0]. The aggregate score is:

```text
aggregate = sum(w_i * s_i) / sum(w_i)
```

Where `w_i` is the weight assigned to dimension `i`. Weights are normalized internally, so only their relative proportions matter -- weights of `[1, 1, 2]` and `[0.5, 0.5, 1]` produce identical results.

If an analyzer is disabled or its weight is zero, it is excluded from both the numerator and denominator. This means the aggregate score adjusts proportionally rather than penalizing missing dimensions.

## Presets

Presets are named weight profiles that control the relative importance of each dimension:

| Preset | Composition | Color | Detail | Lighting | Subject | Style |
| --- | --- | --- | --- | --- | --- | --- |
| `balanced` | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 0.5 |
| `composition` | 3.0 | 1.0 | 1.0 | 1.0 | 1.0 | 0.5 |
| `visual` | 1.0 | 2.0 | 2.0 | 1.0 | 1.0 | 0.5 |

- **balanced** (default): Equal weight to all dimensions except style (0.5).
- **composition**: Emphasizes spatial arrangement. Useful when sorting for well-framed shots.
- **visual**: Emphasizes color and detail. Useful when sorting for visually rich frames.

Style is weighted at 0.5 in all presets because the aesthetic scorer has very low variance across anime screenshots (std ~0.02), providing limited discriminative signal for intra-anime comparison.

### Using presets

```bash
# Analyze with the default balanced preset
loupe analyze shots/

# Rank with a different preset (recomputes aggregates, does not re-analyze)
loupe rank shots/ --preset composition
```

The `rank` command can apply a different preset without re-running analysis. It reads the per-dimension scores from existing sidecar files and recomputes the aggregate with the new weights.

## Custom weights

Override weights in your config TOML file:

```toml
[scoring]
preset = "balanced"  # or omit to use custom weights below

[scoring.weights]
composition = 2.0
color = 1.5
detail = 1.0
lighting = 1.0
subject = 1.0
style = 0.0  # exclude style entirely
```

Setting a weight to 0.0 excludes that dimension from the aggregate score.

## JSON output fields

The `scoring` object in the JSON sidecar contains:

```json
{
  "scoring": {
    "method": "weighted_mean",
    "version": "1.0",
    "weights": {
      "composition": 0.181818,
      "color": 0.181818,
      "detail": 0.181818,
      "lighting": 0.181818,
      "subject": 0.181818,
      "style": 0.090909
    },
    "contributions": {
      "composition": 0.131273,
      "color": 0.123636,
      "detail": 0.107818,
      "lighting": 0.129273,
      "subject": 0.115818,
      "style": 0.043636
    },
    "reliable": true
  }
}
```

| Field | Description |
| --- | --- |
| `method` | Always `"weighted_mean"` in version 1.0 |
| `version` | Scoring algorithm version |
| `weights` | Normalized weights used (sum to 1.0). Reflects the preset after normalization |
| `contributions` | Each dimension's weighted contribution to the aggregate: `contribution_i = weight_i * score_i` |
| `reliable` | `true` if at least 2 dimensions contributed. `false` if only 0 or 1 analyzers ran (the aggregate may not be meaningful) |

The aggregate score equals the sum of all `contributions` values.

## Score calibration

Analyzer scores are on an absolute 0.0--1.0 scale, not relative to other images. A score of 0.7 in composition means the same thing whether you've analyzed 10 images or 10,000. This means:

- Scores are comparable within a batch -- higher is better for ranking.
- Scores across different shows or art styles may cluster differently. A high-detail show may have all detail scores in the 0.6--0.8 range, while a minimalist show clusters around 0.3--0.5. Both are valid.
- The aggregate score is best used for **sort-and-review** within a session, not as an absolute quality judgment.
