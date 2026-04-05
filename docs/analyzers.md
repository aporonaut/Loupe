<!-- markdownlint-disable MD024 -->

# Analyzer Reference

Loupe measures anime screenshots across six independent aesthetic dimensions, each handled by a dedicated analyzer module. Every analyzer produces:

- **Score** (0.0 to 1.0): An absolute quality measurement for that dimension. Scores are not relative to other images. A 0.7 means the same thing regardless of what else has been analyzed.
- **Tags**: Contextual labels with confidence values (0.0 to 1.0) describing specific properties detected in the image. Tags are emitted only when their confidence exceeds the configured threshold (default 0.25).
- **Metadata**: Internal sub-property scores and diagnostics, available in the JSON sidecar file for detailed inspection.

Each analyzer combines multiple sub-properties into a single dimension score using a weighted arithmetic mean. Sub-property weights are documented per analyzer below and can be overridden via configuration.

---

## Composition

**Source:** `src/loupe/analyzers/composition.py`

### What it measures

Spatial arrangement of visual elements within the frame. Eight sub-properties capture different aspects of compositional structure:

| Sub-property | Weight | What it measures |
| --- | --- | --- |
| Rule of thirds | 0.20 | Saliency proximity to the four power-point intersections of the thirds grid |
| Visual balance | 0.20 | Center-of-mass deviation from frame center and quadrant energy variance |
| Leading lines | 0.15 | Convergence of detected line segments toward the saliency centroid |
| Negative space | 0.15 | Distribution of low-edge-density blocks (sweet spot: 30 to 60% "empty") |
| Symmetry | 0.10 | Bilateral symmetry via horizontal flip-and-compare on edge map (IoU) |
| Diagonal dominance | 0.10 | Proportion of line energy from segments in the 30 to 60 degree band |
| Depth layering | 0.05 | Laplacian variance variation across horizontal strips (sharpness gradient) |
| Framing | 0.05 | Ratio of border vs center edge density (in-frame framing elements) |

Saliency is approximated via edge density (Canny edges + box filter + center-bias Gaussian) rather than a learned model. This works well for anime's clean line art, where high-edge subjects stand out against low-edge backgrounds.

All computations are classical CV (OpenCV + NumPy + SciPy). No model dependencies.

### Score interpretation

| Range | Meaning |
| --- | --- |
| 0.7+ | Strong compositional structure: clear subject placement, visual flow, intentional spatial design |
| 0.4 to 0.7 | Moderate composition. Functional framing without strong structural design |
| Below 0.4 | Weak composition. Cluttered, center-heavy, or lacking spatial structure |

### Tags

| Tag | Condition | Description |
| --- | --- | --- |
| `rule_of_thirds` | Score >= 0.6 | Subject placed near third-line intersections |
| `centered` | Center proximity >= 0.85 | Subject placed near frame center |
| `balanced` | Score >= 0.7 | Visual weight evenly distributed |
| `symmetric` | Score >= 0.5 | Bilateral symmetry detected |
| `strong_leading_lines` | Score >= 0.4 | Prominent converging lines |
| `diagonal_composition` | Score >= 0.4 | Strong diagonal line structure |
| `open_composition` | Score >= 0.7 | Significant negative space |
| `framed_subject` | Score >= 0.4 | In-frame elements border the subject |

### Known quirks

- Saliency is edge-density-based, not learned. Works well for anime's flat-shading aesthetic but may underperform on scenes with uniformly busy or uniformly flat edge distributions.
- Depth layering is the weakest sub-property due to anime's frequent uniform focus across depth planes.
- `rule_of_thirds` and `centered` are complementary tags. A frame can score well on thirds but not be centered, or vice versa.

---

## Color

**Source:** `src/loupe/analyzers/color.py`

### What it measures

Chromatic properties and palette design. Seven sub-properties evaluate color usage:

| Sub-property | Weight | What it measures |
| --- | --- | --- |
| Harmony | 0.25 | Matsuda harmony score. Fits the hue histogram to 8 template types via Cohen-Or rotation optimization |
| Vivid color | 0.20 | Hasler & Susstrunk colorfulness metric (opponent color channel statistics) |
| Palette cohesion | 0.15 | Mean pairwise OkLab distance between K-means cluster centroids |
| Color contrast | 0.15 | Luminance range (95th--5th percentile in OkLab L) combined with chroma spread |
| Saturation balance | 0.10 | Coefficient of variation and histogram entropy of OkLCh chroma channel |
| Palette diversity | 0.10 | Simpson's diversity index on 12-sector hue histogram |
| Color temperature | 0.05 | Dominance ratio of warm (0 to 60, 300 to 360 deg) vs cool (120 to 270 deg) hues |

Palette extraction uses K-means clustering in OkLab color space (configurable clusters, default 6) with automatic merging of shading variants (OkLab distance < 0.05). All perceptual computations use OkLab/OkLCh, which provides more uniform perceptual distances than CIELAB.

Fully classical, no model dependencies.

### Score interpretation

| Range | Meaning |
| --- | --- |
| 0.7+ | Harmonious, intentional palette. Colors work together with clear design intent |
| 0.4 to 0.7 | Functional color usage. No major clashes but no strong palette design |
| Below 0.4 | Muddy, clashing, or overly desaturated palette |

### Tags

| Tag | Condition | Description |
| --- | --- | --- |
| `harmonic_i` | Best-fit Matsuda template | Identity harmony (single narrow hue arc) |
| `harmonic_V` | Best-fit Matsuda template | Complementary harmony (single wide hue arc) |
| `harmonic_I` | Best-fit Matsuda template | Analogous pair harmony (two narrow arcs 180 deg apart) |
| `harmonic_L` | Best-fit Matsuda template | Split-complementary harmony (narrow + wide arcs 90 deg apart) |
| `harmonic_T` | Best-fit Matsuda template | Triadic harmony (three narrow arcs 120 deg apart) |
| `harmonic_Y` | Best-fit Matsuda template | Analogous triad harmony (wide + narrow arcs 180 deg apart) |
| `harmonic_X` | Best-fit Matsuda template | Tetradic harmony (four narrow arcs 90 deg apart) |
| `harmonic_N` | Best-fit Matsuda template | No clear harmony (hue distribution fits N-type template best) |
| `warm_palette` | Warm hue ratio > 0.6 | Dominant warm color temperature |
| `cool_palette` | Cool hue ratio > 0.6 | Dominant cool color temperature |
| `neutral_palette` | Temperature score < 0.5 | No dominant temperature bias |
| `vivid` | Score >= 0.6 | High overall colorfulness |
| `muted` | Score <= 0.25 | Low overall colorfulness |
| `monochromatic` | Diversity <= 0.15 | Single-hue palette |
| `limited_palette` | Diversity <= 0.35 | Restricted color diversity |
| `diverse_palette` | Diversity >= 0.7 | Wide color diversity |

### Known quirks

- Exactly one `harmonic_*` tag is always emitted (the best-fit template). `harmonic_N` doesn't mean "bad"; it means the hue distribution is broad and doesn't cluster into a simpler template pattern.
- Color contrast uses overall image statistics without segmentation. Segmentation-enhanced contrast (comparing foreground vs background) is a potential future improvement.
- Achromatic images (near-zero chroma) score 1.0 on harmony (trivially harmonious) and 0.0 on diversity.

---

## Detail

**Source:** `src/loupe/analyzers/detail.py`

### What it measures

Visual complexity and rendering effort. Six sub-properties measure different facets of detail:

| Sub-property | Weight | What it measures |
| --- | --- | --- |
| Edge density | 0.20 | Laplacian variance. Higher variance indicates more edges and visual complexity |
| Shading granularity | 0.20 | Tonal histogram peak count and entropy. More peaks means more shading levels |
| Line work quality | 0.20 | Sobel gradient magnitude at Canny edges (sharpness) and edge density (coverage) |
| Rendering clarity | 0.15 | Global + local Laplacian variance. Sharp patches vs blurry patches |
| Spatial frequency | 0.15 | High-frequency energy ratio via 2D FFT (outer 60% of frequency space) |
| Texture richness | 0.10 | GLCM entropy at multiple distances and angles (skimage graycomatrix) |

**Region separation:** When a segmentation mask is available, each sub-property is measured independently on foreground (character) and background regions. The combined score uses configurable region weights (default: 60% background, 40% character). This reflects the observation that background detail is often the stronger differentiator for anime wallpaper quality.

Edge cases: If character fills >90% of frame, 100% character weight is used. If no character is detected, 100% background weight is used.

### Score interpretation

| Range | Meaning |
| --- | --- |
| 0.7+ | Rich visual complexity: fine line work, detailed textures, sophisticated shading |
| 0.4 to 0.7 | Moderate detail. Functional rendering without exceptional complexity |
| Below 0.4 | Low detail. Flat rendering, simple shapes, minimal texture |

### Tags

| Tag | Condition | Description |
| --- | --- | --- |
| `high_detail` | Overall score >= 0.7 | Very high overall visual complexity |
| `rich_background` | Background region avg >= 0.6 | Background region has high detail |
| `detailed_character` | Character region avg >= 0.6 | Character region has high detail |
| `sharp_rendering` | Rendering clarity >= 0.6 | High rendering clarity across the frame |
| `complex_shading` | Shading granularity >= 0.6 | Many tonal levels in shading |
| `fine_line_work` | Line work quality >= 0.6 | Sharp, clean edge definition |

### Known quirks

- Motion blur can inflate edge density and spatial frequency metrics. The multi-sub-property approach dilutes this effect. A motion-blurred frame may score high on detail but likely lower on composition and subject clarity.
- GLCM texture richness is less informative on anime's flat color regions. It performs better on detailed backgrounds (foliage, cityscapes) than character regions. Weighted lower by default (0.10).
- The bg/char region weights (60/40) can be adjusted via `analyzers.detail.params.bg_weight` and `char_weight` in config.

---

## Lighting

**Source:** `src/loupe/analyzers/lighting.py`

### What it measures

Illumination design and quality. Six sub-properties analyze how light is used in the frame:

| Sub-property | Weight | What it measures |
| --- | --- | --- |
| Contrast ratio | 0.25 | Robust percentile contrast on V channel (95th minus 5th percentile) |
| Rim/edge lighting | 0.20 | Boundary luminance differential. Compares brightness at character boundary ring vs nearby background |
| Shadow quality | 0.15 | Gradient magnitude at shadow boundaries. Scores both soft and hard shadows as intentional |
| Atmospheric lighting | 0.15 | Bloom/glow detection via comparison of V channel with Gaussian-blurred V |
| Highlight/shadow balance | 0.15 | Tri-zone histogram analysis. Shadow, midtone, and highlight proportions and their entropy |
| Light directionality | 0.10 | Luminance asymmetry across a 3x3 grid |

Classical CV operates on the HSV V (value/brightness) channel. Rim-light detection uses the shared segmentation mask to identify character boundaries.

WD-Tagger predictions supplement the tag output with learned lighting labels (backlighting, sunlight, moonlight, lens flare, etc.) that classical CV cannot reliably detect.

### Score interpretation

| Range | Meaning |
| --- | --- |
| 0.7+ | Dramatic, intentional illumination: clear lighting design with visual interest |
| 0.4 to 0.7 | Adequate lighting. Functional but not distinctive |
| Below 0.4 | Flat or uncontrolled lighting. Narrow tonal range, no discernible lighting design |

### Tags

| Tag | Condition | Description |
| --- | --- | --- |
| `high_contrast` | Contrast >= 0.7 | Strong tonal range |
| `low_contrast` | Contrast < 0.3 | Narrow tonal range |
| `dramatic_lighting` | Overall score >= 0.7 | Overall lighting score is very high |
| `flat_lighting` | Overall score < 0.25 | Overall lighting score is very low |
| `rim_lit` | Rim score >= 0.3 | Rim/edge lighting detected on character boundary |
| `soft_shadows` | Shadow >= 0.6 + atmospheric >= 0.3 | Gradual shadow boundaries |
| `hard_shadows` | Shadow >= 0.6 + atmospheric < 0.3 | Sharp shadow boundaries |
| `atmospheric` | Atmospheric >= 0.3 | Bloom or glow effects detected |
| `directional_light` | Directionality >= 0.3 | Strong luminance gradient across frame |
| `balanced_exposure` | Balance >= 0.8 | Even tonal zone distribution |
| `backlighting` | WD-Tagger confidence >= threshold | Backlighting detected |
| `sunlight` | WD-Tagger confidence >= threshold | Sunlight detected |
| `moonlight` | WD-Tagger confidence >= threshold | Moonlight detected |
| `lens_flare` | WD-Tagger confidence >= threshold | Lens flare detected |
| `light_rays` | WD-Tagger confidence >= threshold | Light rays detected |
| `glowing` | WD-Tagger confidence >= threshold | Glow effect detected |

### Known quirks

- Light directionality uses a coarse 3x3 grid. Multi-source or non-physical anime lighting can produce ambiguous directionality readings.
- Atmospheric bloom detection may false-positive on large bright regions that aren't bloom effects (e.g., white backgrounds).
- Rim-light detection requires the segmentation mask. Without it, `rim_edge_lighting` defaults to 0.0.
- Shadow quality scores both very soft and very hard shadows as intentional. The V-shaped scoring curve has its minimum around gradient magnitude ~35 (the "neither" zone).
- Metadata includes `tonality` classification (high_key, low_key, balanced) and `light_direction` (e.g., "top_right") for downstream use.

---

## Subject

**Source:** `src/loupe/analyzers/subject.py`

### What it measures

How effectively the frame emphasizes its subject. Six sub-properties assess focal emphasis:

| Sub-property | Weight | What it measures |
| --- | --- | --- |
| Figure-ground separation | 0.25 | OkLab color difference between foreground/background mean colors + Sobel gradient at mask boundary |
| Saliency strength | 0.15 | Spectral residual saliency (Hou & Zhang 2007) concentration within subject mask, adjusted for subject size |
| DOF effect | 0.15 | Laplacian variance ratio between subject and background regions (sharp subject vs blurry background) |
| Negative space utilization | 0.15 | Distribution of quiet (low-edge-density) background blocks around subject |
| Subject completeness | 0.15 | Frame edge contact ratio. Low contact means the subject is fully within frame |
| Subject scale | 0.15 | Subject area as fraction of frame, scored with preference for medium-to-closeup range (15 to 50%) |

Requires the shared segmentation mask to identify the subject. When no mask is available (or the mask is empty), the analyzer returns a 0.1 floor score with the `environment_focus` tag.

### Score interpretation

| Range | Meaning |
| --- | --- |
| 0.7+ | Strong focal emphasis: clear subject with good separation from background |
| 0.4 to 0.7 | Moderate emphasis. Subject present but framing could be stronger |
| Below 0.4 | Weak subject emphasis. Unclear subject, or environment/landscape shot |
| 0.1 | Floor score for environment shots (no character detected) |

### Tags

| Tag | Condition | Description |
| --- | --- | --- |
| `extreme_closeup` | Subject > 60% of frame | Subject dominates the frame |
| `closeup` | Subject 30 to 60% of frame | Close framing on subject |
| `medium_shot` | Subject 15 to 30% of frame | Standard character framing |
| `wide_shot` | Subject 5 to 15% of frame | Subject with environmental context |
| `very_wide` | Subject < 5% of frame | Subject small in frame |
| `environment_focus` | No character detected | Landscape or scenery shot |
| `strong_separation` | FG separation >= 0.7 | Strong figure-ground separation |
| `shallow_dof` | DOF score >= 0.5 | Significant DOF blur differential |
| `complete_subject` | Completeness >= 0.7 | Subject fully within frame (not cropped) |

### Known quirks

- The 0.1 floor for environment shots is intentional but penalizes environment-focused compositions. Users sorting by aggregate score should be aware that landscape shots will sort lower. The `environment_focus` tag helps identify these.
- Multi-subject compositions use the union of all character masks. Individual subject emphasis is not measured separately.
- Saliency uses OpenCV's spectral residual method, which has moderate reliability on anime content.
- Subject scale scoring prefers the medium-to-closeup range (15 to 50% of frame). Extreme close-ups (>60%) and very wide shots (<5%) score lower.

---

## Style

**Source:** `src/loupe/analyzers/style.py`

### What it measures

Artistic identity through two scored sub-properties and categorical style tags. This is the least mature analyzer. The tagging component is solid, but the scoring component uses proxies rather than direct style coherence measurements.

**Scored sub-properties:**

| Sub-property | Weight | What it measures |
| --- | --- | --- |
| Aesthetic quality | 0.70 | deepghs/anime_aesthetic scorer output (SwinV2, ONNX), a Danbooru quality proxy |
| Layer consistency | 0.30 | Intra-layer rendering uniformity via edge density uniformity, gradient consistency, and palette coherence (experimental) |

Layer consistency measures how consistent the rendering is *within* each layer (foreground and background independently). Cross-layer differences are not penalized, as anime routinely uses different rendering approaches for characters and backgrounds.

**Style tags** are categorical and do not affect the numerical score. They come from three sources:

1. **Aesthetic tier tags:** from the aesthetic scorer's output distribution (masterpiece through worst)
2. **WD-Tagger style tags:** art style tags passed through when confident (flat_color, gradient, realistic, cel_shading, soft_shading)
3. **CLIP zero-shot categories:** broad style classification via CLIP ViT-L/14 (naturalistic, geometric/abstract, painterly, digital modern, retro cel)

### Score interpretation

| Range | Meaning |
| --- | --- |
| 0.7+ | Top aesthetic tier with consistent rendering |
| 0.4 to 0.7 | Typical range for most anime screenshots. Adequate quality and consistency |
| Below 0.4 | Low aesthetic quality or inconsistent rendering |

**Important:** Style scores have very low variance across images (std ~0.02 in testing). The aesthetic scorer provides limited discriminative power for intra-anime quality comparison. Style is weighted at 0.5 in the default scoring preset for this reason.

### Tags

| Tag | Condition | Description |
| --- | --- | --- |
| `aesthetic_masterpiece` | Tier probability >= 0.3 | Top-tier aesthetic quality |
| `aesthetic_best` | Tier probability >= 0.3 | Excellent aesthetic quality |
| `aesthetic_great` | Tier probability >= 0.3 | Great aesthetic quality |
| `aesthetic_good` | Tier probability >= 0.3 | Good aesthetic quality |
| `aesthetic_normal` | Tier probability >= 0.3 | Average aesthetic quality |
| `aesthetic_low` | Tier probability >= 0.3 | Below-average aesthetic quality |
| `aesthetic_worst` | Tier probability >= 0.3 | Poor aesthetic quality |
| `consistent_rendering` | Layer consistency >= 0.7 | Uniform quality across layers |
| `inconsistent_rendering` | Layer consistency < 0.3 | Quality varies across layers |
| `flat_color` | WD-Tagger confidence >= threshold | Flat color style |
| `gradient` | WD-Tagger confidence >= threshold | Gradient shading |
| `realistic` | WD-Tagger confidence >= threshold | Realistic rendering |
| `cel_shading` | WD-Tagger confidence >= threshold | Cel-shaded style |
| `soft_shading` | WD-Tagger confidence >= threshold | Soft shading technique |
| `naturalistic_anime` | CLIP probability >= threshold | Naturalistic anime style |
| `geometric_abstract_anime` | CLIP probability >= threshold | Geometric/abstract anime style |
| `painterly_anime` | CLIP probability >= threshold | Painterly anime style |
| `digital_modern_anime` | CLIP probability >= threshold | Modern digital anime style |
| `retro_cel_anime` | CLIP probability >= threshold | Retro cel animation style |

### Known quirks

- The aesthetic quality score is entangled with other quality aspects (detail, color appeal, composition). It is a proxy, not a pure style coherence measure.
- Layer consistency is experimental and unvalidated. Its contribution is weighted low (0.30) to limit impact from potential false signals.
- Cross-layer consistency is deliberately ignored. Genuinely incoherent rendering that spans both layers may not be detected.
- CLIP zero-shot categories are broad and may not capture fine-grained style distinctions within anime sub-genres.

---

## Complete Tag Reference

All tags across all six dimensions. This matches the output of `loupe tags`.

| Dimension | Tag | Description |
| --- | --- | --- |
| composition | `rule_of_thirds` | Subject placed near third-line intersections |
| composition | `centered` | Subject placed near frame center |
| composition | `balanced` | Visual weight evenly distributed |
| composition | `symmetric` | Bilateral symmetry detected |
| composition | `strong_leading_lines` | Prominent converging lines |
| composition | `diagonal_composition` | Strong diagonal line structure |
| composition | `open_composition` | Significant negative space |
| composition | `framed_subject` | In-frame elements border the subject |
| color | `harmonic_i` | Matsuda i-type harmony (identity) |
| color | `harmonic_V` | Matsuda V-type harmony (complementary) |
| color | `harmonic_I` | Matsuda I-type harmony (analogous pair) |
| color | `harmonic_L` | Matsuda L-type harmony (split-complementary) |
| color | `harmonic_T` | Matsuda T-type harmony (triadic) |
| color | `harmonic_Y` | Matsuda Y-type harmony (analogous triad) |
| color | `harmonic_X` | Matsuda X-type harmony (tetradic) |
| color | `harmonic_N` | Matsuda N-type harmony (no clear harmony) |
| color | `warm_palette` | Dominant warm color temperature |
| color | `cool_palette` | Dominant cool color temperature |
| color | `neutral_palette` | No dominant temperature bias |
| color | `vivid` | High overall colorfulness |
| color | `muted` | Low overall colorfulness |
| color | `monochromatic` | Single-hue palette |
| color | `limited_palette` | Restricted color diversity |
| color | `diverse_palette` | Wide color diversity |
| detail | `high_detail` | Overall detail score above 0.7 |
| detail | `rich_background` | Background region has high detail |
| detail | `detailed_character` | Character region has high detail |
| detail | `sharp_rendering` | High rendering clarity |
| detail | `complex_shading` | Many tonal levels in shading |
| detail | `fine_line_work` | High line work quality |
| lighting | `high_contrast` | Strong tonal range |
| lighting | `low_contrast` | Narrow tonal range |
| lighting | `dramatic_lighting` | Overall lighting score above 0.7 |
| lighting | `flat_lighting` | Overall lighting score below 0.25 |
| lighting | `rim_lit` | Rim/edge lighting on character boundary |
| lighting | `soft_shadows` | Gradual shadow boundaries |
| lighting | `hard_shadows` | Sharp shadow boundaries |
| lighting | `atmospheric` | Bloom or glow effects detected |
| lighting | `directional_light` | Strong luminance gradient |
| lighting | `balanced_exposure` | Even tonal zone distribution |
| lighting | `backlighting` | WD-Tagger: backlighting detected |
| lighting | `sunlight` | WD-Tagger: sunlight detected |
| lighting | `moonlight` | WD-Tagger: moonlight detected |
| lighting | `lens_flare` | WD-Tagger: lens flare detected |
| lighting | `light_rays` | WD-Tagger: light rays detected |
| lighting | `glowing` | WD-Tagger: glow effect detected |
| subject | `extreme_closeup` | Subject >60% of frame area |
| subject | `closeup` | Subject 30 to 60% of frame area |
| subject | `medium_shot` | Subject 15 to 30% of frame area |
| subject | `wide_shot` | Subject 5 to 15% of frame area |
| subject | `very_wide` | Subject <5% of frame area |
| subject | `environment_focus` | No character detected |
| subject | `strong_separation` | Strong figure-ground separation |
| subject | `shallow_dof` | Significant DOF blur differential |
| subject | `complete_subject` | Subject fully within frame |
| style | `aesthetic_masterpiece` | Top-tier aesthetic quality |
| style | `aesthetic_best` | Excellent aesthetic quality |
| style | `aesthetic_great` | Great aesthetic quality |
| style | `aesthetic_good` | Good aesthetic quality |
| style | `aesthetic_normal` | Average aesthetic quality |
| style | `aesthetic_low` | Below-average aesthetic quality |
| style | `aesthetic_worst` | Poor aesthetic quality |
| style | `consistent_rendering` | Uniform quality across layers |
| style | `inconsistent_rendering` | Quality varies across layers |
| style | `flat_color` | WD-Tagger: flat color style |
| style | `gradient` | WD-Tagger: gradient shading |
| style | `realistic` | WD-Tagger: realistic rendering |
| style | `cel_shading` | WD-Tagger: cel-shaded style |
| style | `soft_shading` | WD-Tagger: soft shading technique |
| style | `naturalistic_anime` | CLIP: naturalistic anime style |
| style | `geometric_abstract_anime` | CLIP: geometric/abstract anime style |
| style | `painterly_anime` | CLIP: painterly anime style |
| style | `digital_modern_anime` | CLIP: modern digital anime style |
| style | `retro_cel_anime` | CLIP: retro cel animation style |
