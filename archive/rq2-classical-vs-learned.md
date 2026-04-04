# RQ2 — Classical CV vs. Learned Models per Dimension

**Date:** 2026-04-03
**Status:** Complete
**Depends on:** RQ1 (Analysis Dimensions)
**Question:** For each analysis dimension, what is the best technical approach — classical computer vision, learned/neural models, or a hybrid?

---

## 1. Methodology

### Approach

For each of the six dimensions validated in RQ1, this research surveys three categories of technique:

1. **Classical CV** — deterministic algorithms using image processing primitives (edge detection, frequency analysis, histogram statistics, geometric measurements). Implemented with OpenCV, NumPy, SciPy, scikit-image.
2. **Learned models** — pretrained neural networks that produce predictions from image input. Sourced from the model inventory cataloged in RQ3.
3. **Hybrid** — a learned model handles one sub-task (typically subject/region identification), then classical CV performs the measurement.

Each technique is assessed against four criteria from the RQ2 specification:

- **Accuracy/reliability on anime frames** — does the technique produce meaningful results on anime's distinctive visual properties (flat shading, strong outlines, painted lighting, discrete palettes)?
- **Computational cost** — CPU-only vs. GPU-required, approximate inference time per image.
- **Implementation complexity** — lines of code, algorithmic subtlety, tuning surface.
- **Dependency weight** — does it require PyTorch, large model downloads, or additional libraries beyond the core stack?

Additionally, techniques are assessed against constraints established by RQ4 (Aggregate Scoring):

- Scores must be **absolute** (0.0–1.0), not batch-relative.
- Scores must be **decomposable** — per-dimension contributions to the aggregate must be transparent.
- Scores must be **batch-independent** — adding or removing images from the batch must not change existing scores.

### Knowledge Sources

This research draws on training knowledge of the computer vision and image quality assessment literature, the anime production domain, and the model inventory from RQ3. Claims derived from training knowledge are marked [T] where verification against current sources would strengthen confidence. The core algorithms (edge detection, FFT analysis, color harmony models, saliency detection) are well-established and stable; version-specific library APIs and model availability are the primary areas where [T]-marked claims should be verified during RQ5 (Tooling Validation).

### Key Constraints from Prior Research

- **From RQ3 (Models):** Available anime-specific models include WD-Tagger v3 (tagging), deepghs/anime_aesthetic (scoring), kawaimasa/kawai-aesthetic-scorer (scoring), skytnt/anime-segmentation (character masks), deepghs detection suite (face/head/person), OpenCLIP ViT-L/14 (embeddings). All fit within an 8GB VRAM budget.
- **From RQ4 (Scoring):** Weighted Arithmetic Mean aggregation. Each analyzer must produce calibrated absolute scores on [0.0, 1.0]. No batch-dependent normalization. Missing dimensions handled via proportional aggregation.
- **From RQ6 (Tooling):** OpenCV is a native dependency (classical CV = low friction). PyTorch/CUDA requires explicit routing via `tool.uv.sources`. TOML config means simple parameter types. Pyright for type checking.

---

## 2. Executive Summary

| # | Dimension | Recommended Approach | Primary Technique | Learned Components | Confidence |
| --- | ----------- | --------------------- | ------------------- | ------------------- | ------------ |
| 1 | **Composition** | Hybrid (classical-heavy) | Classical geometry on saliency map | Saliency model (U2-Net or edge-density proxy) | HIGH |
| 2 | **Color** | Classical | Matsuda harmony, palette extraction, colorfulness metrics | None required (WD-Tagger tags optional supplement) | HIGH |
| 3 | **Detail** | Hybrid (classical + segmentation) | Edge density, spatial frequency, texture entropy, shading analysis | Anime segmentation for region separation | HIGH |
| 4 | **Lighting** | Hybrid (classical-heavy) | Luminance statistics, contrast, shadow/rim detection | Anime segmentation for boundary analysis; WD-Tagger lighting tags | MEDIUM |
| 5 | **Subject** | Hybrid (learned detection + classical measurement) | Figure-ground contrast, DOF detection, saliency concentration | Anime segmentation + deepghs detection for subject identification | HIGH |
| 6 | **Style** | Learned (primarily) | Anime aesthetic scorer + WD-Tagger style tags | Aesthetic scorer, WD-Tagger, CLIP zero-shot | LOW-MEDIUM |

**Pattern:** Four of six dimensions are primarily classical CV with learned models serving as infrastructure (saliency maps, segmentation masks). Only Style requires learned models as the primary scoring mechanism. Color is fully classical with no learned component needed.

**Shared infrastructure:** The anime segmentation model (skytnt/anime-segmentation) is used by four dimensions (Detail, Lighting, Subject, Style). It should be loaded once by the engine and its output shared across analyzers.

---

## 3. Resolution of RQ1 Open Questions

RQ1 deferred five questions to RQ2. Each is addressed here and elaborated in the relevant dimension section.

### 3.1 Style Coherence Measurement (RQ1 §6.1)

**Question:** What technical approaches can quantify style coherence from a single frame? Is CLIP embedding analysis sufficient, or does this require multi-frame context?

**Resolution:** True style coherence measurement from a single frame is **not feasible for v1**. CLIP embedding analysis requires multi-frame context (comparing a frame's embedding to a production-specific centroid), which Loupe's single-frame architecture cannot provide. The most viable v1 approach is:

- **Style tags** (categorical): WD-Tagger v3 style-relevant tags + CLIP zero-shot classification. HIGH feasibility.
- **Style quality score** (0.0–1.0): Use an anime aesthetic scorer (deepghs/anime_aesthetic or kawaimasa) as a proxy. This measures entangled aesthetic quality, not pure coherence. MODERATE feasibility.
- **Experimental layer consistency** (supplementary): Compare rendering properties (edge density, color palette, gradient smoothness) across foreground and background regions within the frame. Novel and unvalidated — include with low weight and flag as experimental.

The Style dimension should carry a lower default weight in the aggregate score and be documented as the least mature analyzer. See §9 for full analysis.

### 3.2 Background vs. Character Detail Weighting (RQ1 §6.2)

**Question:** Should background and character detail be measured separately and combined, or measured holistically?

**Resolution:** **Measure separately using segmentation, then combine with configurable weights.**

Rationale:

- Background and character detail are artistically independent. Detailed backgrounds with simply-rendered characters (common in Makoto Shinkai films) and detailed characters on flat backgrounds (common in dialogue scenes) are both valid aesthetic approaches.
- For the wallpaper use case, background detail often matters more — the background fills the majority of the frame.
- The anime segmentation model (skytnt/anime-segmentation) provides the foreground/background mask, making the overall approach hybrid.

Default weighting: **60% background / 40% character**. Configurable via analyzer parameters. When no character is detected (landscape shots), use 100% background. When character fills nearly the entire frame (extreme close-up), use 100% character. See §6 for full analysis.

### 3.3 Lighting Detection in Anime (RQ1 §6.3)

**Question:** How reliably can classical CV techniques detect light direction and atmospheric effects in anime frames, where shadows and highlights are artistic choices rather than physical phenomena?

**Resolution:** **Partially reliable — sub-property dependent.**

Techniques that detect the *presence and properties* of lighting features work well:

- Contrast ratio, highlight/shadow balance, histogram statistics — purely statistical, medium-independent. **HIGH reliability.**
- Rim/edge lighting detection — anime rim lights are high-contrast boundary effects, readily detectable with segmentation + luminance analysis. **HIGH reliability.**
- Shadow edge softness — anime shadow boundaries have clear stylistic signatures (hard cel vs. soft gradient). **MEDIUM-HIGH reliability.**

Techniques that *infer physical lighting parameters* are less reliable:

- Light directionality — coarse grid-based luminance comparison gives approximate direction. Fine-grained gradient analysis assumes physical light transport that anime doesn't follow. **MEDIUM reliability.**
- Atmospheric effects beyond bloom — light shafts and haze are harder. **MEDIUM reliability.**

Overall, Lighting is the dimension where classical CV faces the most conceptual mismatch with anime content. The recommended approach focuses on statistical and feature-detection methods rather than physical lighting inference. See §7 for full analysis.

### 3.4 Cross-Dimension Normalization (RQ1 §6.4)

**Question:** With six dimensions, should sub-properties be individually scored and averaged per dimension, or should each dimension produce a single holistic score?

**Resolution:** **Each dimension produces a single holistic score (0.0–1.0), computed as a weighted combination of its sub-property scores.** Sub-property scores are recorded in `AnalyzerResult.metadata` for transparency but do not directly enter the aggregate.

Rationale:

- RQ4's Weighted Arithmetic Mean operates on dimension-level scores. Exposing sub-properties to the aggregate would create a two-level weighting problem (sub-property weights within dimensions × dimension weights across dimensions) that is harder for users to configure.
- Sub-property weights within a dimension are an analyzer implementation detail — the analyzer's author determines, for example, that color harmony should weight more than palette diversity within the Color score.
- Reporting sub-properties in metadata preserves transparency: the user can inspect *why* a dimension scored as it did, and the engine can expose this in the CLI output via Rich formatting.

Default sub-property weights within each dimension are specified in the per-dimension sections below. All are configurable via analyzer parameters.

### 3.5 Anime-Tuned Models for Composition (RQ1 §6.5)

**Question:** Do general composition analysis models generalize to anime's distinctive composition patterns?

**Resolution:** **The question is largely moot because the recommended approach does not use composition-specific neural models.**

Composition analysis uses classical geometric measurements (thirds alignment, symmetry, line angles, negative space ratios) that are **medium-independent** — they operate on spatial relationships, edges, and intensity distributions, not on learned visual statistics. A line at 45° is a diagonal whether it's a road in a photograph or an architectural element in an anime background. Edge detection works *better* on anime than on photography because anime has stronger, cleaner contour lines.

The one component with anime-transfer risk is **saliency detection** (identifying where the viewer's attention falls). Photo-trained saliency models (U2-Net, BASNet) may not perfectly capture anime-specific visual conventions. However:

- Salient object detection models capture figure-ground separation, which is often *more* pronounced in anime (strong outlines delineate subjects).
- An anime-specific fallback exists: edge-density saliency proxy — compute local edge density + center-bias Gaussian prior. This exploits anime's bimodal detail distribution (high-edge characters against low-edge backgrounds) and requires no learned model.

The recommended phased approach: start with the edge-density proxy, validate on anime test images, upgrade to U2-Net if insufficient. See §4 for full analysis.

---

## 4. Composition — Hybrid (Classical-Heavy)

### 4.1 Recommended Approach

**Classical geometry on saliency/attention map.** The pipeline decomposes into two stages:

1. **Saliency detection** (learned or heuristic): Identify where visual attention falls in the frame. This is the only component with anime-domain risk.
2. **Geometric measurement** (classical): Given the saliency map, compute each sub-property using spatial analysis. All measurements are medium-independent.

### 4.2 Saliency Options

| Option | Method | Anime Reliability | Cost | Dependencies |
| -------- | -------- | ------------------- | ------ | ------------- |
| Edge-density proxy | Canny edge density + center-bias Gaussian | MEDIUM-HIGH for anime | Very low (CPU, <10ms) | OpenCV only |
| Spectral residual | `cv2.saliency.StaticSaliencySpectralResidual` [T] | MEDIUM | Very low (CPU, <15ms) | OpenCV (contrib) |
| U2-Net | Learned salient object detection | MEDIUM-HIGH (general) | Moderate (GPU ~50–200ms) [T] | PyTorch, ~44MB model |

**Recommendation:** Start with the **edge-density proxy** for v1. It exploits anime's flat-shading aesthetic (high-edge subjects against low-edge backgrounds), requires no model download, and runs on CPU. If validation reveals insufficient quality on specific composition types, upgrade to U2-Net (shared with Subject analyzer).

### 4.3 Per-Sub-Property Techniques

| Sub-Property | Technique | Key Implementation | Anime Reliability | Cost |
| --- | --- | --- | --- | --- |
| **Rule of thirds** | Saliency-weighted Gaussian proximity to 4 power points | Place 2D Gaussians (σ ≈ 8% of diagonal) at grid intersections; score = `Σ S(x,y)·G(x,y) / Σ S(x,y)` | HIGH (if saliency is good) | Very low |
| **Visual balance** | Saliency/luminance center-of-mass deviation + quadrant variance | `scipy.ndimage.center_of_mass(S)` → deviation from frame center; quadrant energy variance | HIGH | Very low |
| **Symmetry** | Bilateral flip-and-compare on Canny edge map | `cv2.flip(edges, 1)` → IoU of edge pixels, or SSIM on edge maps | HIGH | Very low |
| **Depth layering** | Laplacian variance across horizontal strips | `cv2.Laplacian(strip, cv2.CV_64F).var()` per strip; score = coefficient of variation | LOW-MEDIUM | Very low |
| **Leading lines** | LSD line detection + convergence toward saliency centroid | `cv2.createLineSegmentDetector()` → filter by length → cluster line intersections → check overlap with saliency peak | MEDIUM-HIGH | Low |
| **Diagonal dominance** | Length-weighted angle histogram of LSD lines | Bin line angles; diagonal energy = energy in 30°–60° / total energy | HIGH | Low (reuses LSD) |
| **Negative space** | Edge-density block thresholding | `cv2.Canny` → `cv2.boxFilter` for local density → threshold low-density regions → compute ratio and spatial distribution | HIGH | Low |
| **Framing** | Border-region edge density ratio | Edge density in outer 15% border strips vs. center region; high border + low center = framing | MEDIUM | Low |

### 4.4 Anime-Specific Considerations

- **Depth layering is the weakest sub-property.** Anime frequently uses uniform focus across all depth planes. The Laplacian variance approach detects composited depth-of-field blur when present (increasingly common in modern digital anime) but misses depth conveyed through other means (overlap, scale, atmospheric color). Assign lower default weight.
- **Centered compositions are valid.** Anime frequently centers subjects, especially in dialogue. The composition analyzer should detect and tag centered compositions alongside rule-of-thirds — both are valid patterns. The thirds score should not penalize centering; they should be complementary measurements.
- **Dutch angles** are common in anime for dramatic tension. Well-captured by the diagonal dominance measurement.
- **Edge-based analysis works well on anime** because anime's line art produces stronger, cleaner edges than photographic content.

### 4.5 Computational Profile

- **Saliency (edge-density proxy):** <10ms CPU
- **All geometric measurements combined:** ~20–50ms CPU
- **Total per image:** <60ms CPU, no GPU required
- **Dependencies:** OpenCV, NumPy, SciPy (all in core stack)

### 4.6 Suggested Default Sub-Property Weights

| Sub-Property | Default Weight | Rationale |
| --- | --- | --- |
| Rule of thirds | 0.20 | Primary composition signal, but not universally applicable |
| Visual balance | 0.20 | Universally meaningful across composition styles |
| Symmetry | 0.10 | Important when present, but most frames are asymmetric |
| Depth layering | 0.05 | Low anime reliability — keep weight low |
| Leading lines | 0.15 | Strong signal in backgrounds; absent in close-ups (correctly) |
| Diagonal dominance | 0.10 | Captures dynamic compositions; absent in static scenes (correctly) |
| Negative space | 0.15 | Clean signal in anime; meaningful for wallpaper use case |
| Framing | 0.05 | Useful but narrow — only applies to specific shot types |

---

## 5. Color — Classical

### 5.1 Recommended Approach

**Fully classical.** Color is the dimension where classical CV has its strongest advantage. Every sub-property has a well-established, mathematically grounded technique that produces interpretable, continuous scores. No learned models are needed for the core measurements.

Rationale:

1. Color harmony is formally defined via Matsuda's geometric model — no training data or domain bias.
2. Palette extraction via K-means is a textbook technique, and anime's cel-shaded discrete color regions make it *easier* than photography.
3. All statistical metrics (cohesion, saturation, diversity, colorfulness) are direct computations on pixel distributions — fast, deterministic, and interpretable.
4. No anime-domain training data is needed. A harmonious palette is harmonious regardless of medium.

### 5.2 Color Space

**OkLab/OkLCh is the primary color space** for all perceptual computations.

| Space | Perceptual Uniformity | Library Support | Recommendation |
| --- | --- | --- | --- |
| **OkLab/OkLCh** | Excellent (best available) | Manual conversion (~20 lines NumPy) or `colour-science` | Primary — all harmony, palette, and distance computations |
| **CIELAB** | Good (blue region issues) | Native in OpenCV (`COLOR_BGR2Lab`) | Fallback if OkLab adds unacceptable complexity |
| **HSV** | Poor | Native in OpenCV | Quick prototyping only; avoid for scoring |
| **sRGB** | Not perceptually uniform | Universal | Only for Hasler & Süsstrunk colorfulness metric |

OkLab conversion from linear sRGB is straightforward — two 3×3 matrix multiplications with a cube-root nonlinearity between them. This can be implemented in ~20 lines of NumPy without external dependencies [T]. The `colour-science` library also provides OkLab but adds a large dependency tree.

**Implementation note:** Input must be **linear sRGB** (gamma-decoded). Gamma decode: `linear = np.where(srgb <= 0.04045, srgb / 12.92, ((srgb + 0.055) / 1.055) ** 2.4)`.

### 5.3 Per-Sub-Property Techniques

| Sub-Property | Technique | Key Implementation | Anime Reliability | Cost |
| --- | --- | --- | --- | --- |
| **Color harmony** | Matsuda template fitting (Cohen-Or algorithm) | Extract chroma-weighted hue histogram in OkLCh. For each of 8 template types × rotation angles, compute total angular distance from hues to nearest template sector. Score = 1 − (min_cost / max_theoretical_cost) | HIGH | Low-medium (2,880 template evaluations, all vectorized NumPy) |
| **Palette extraction** | K-means in OkLab | `sklearn.cluster.KMeans(n_clusters=K)` on downsampled OkLab pixels, K=5–8. Merge clusters within CIEDE2000 threshold. Output: centroids + pixel proportions | HIGH | Low (downsample first) |
| **Palette cohesion** | Pairwise OkLab distance variance | Compute Euclidean distances between all palette centroid pairs in OkLab. Cohesion = inverse of normalized variance. Optionally: separate FG/BG palette cohesion if segmentation available | HIGH | Very low (operates on K centroids, not pixels) |
| **Saturation balance** | OkLCh chroma statistics | Chroma histogram: mean, variance, skewness, entropy. Balanced = low variance + moderate entropy | HIGH | Very low |
| **Color contrast** | CIEDE2000 between subject/background dominant colors | Segment via saliency or segmentation mask. Compute delta-E between FG and BG dominant palette entries. Also: luminance contrast ratio | HIGH | Low (requires region separation) |
| **Color temperature** | Warm/cool hue ratio + OkLab b-axis mean | Classify OkLCh hues as warm (0°–60°, 300°–360°) or cool (120°–270°). Score temperature consistency (how strongly the frame leans one direction) | HIGH | Very low |
| **Palette diversity** | Simpson's diversity index on hue bins | Bin OkLCh hues into 12 sectors. D = 1 − Σ(pᵢ²). Ranges 0 (monochromatic) to ~1 (maximally diverse) | HIGH | Very low |
| **Vivid color** | Hasler & Süsstrunk colorfulness metric | In RGB: `rg = R−G`, `yb = 0.5(R+G)−B`, `colorfulness = √(σ_rg² + σ_yb²) + 0.3·√(μ_rg² + μ_yb²)` [T] | HIGH | Very low |

### 5.4 Matsuda Harmony — Implementation Detail

The Cohen-Or et al. (2006) [^1] formalization of Matsuda's 8 harmonic hue templates is the scoring algorithm:

1. Extract the image's hue distribution as a chroma-weighted histogram in OkLCh. Desaturated pixels (chroma below threshold) are excluded — they carry no hue information.
2. For each of 8 template types, each parameterized by sector geometries on the hue wheel:
   - **Type i** — single narrow arc (~18°)
   - **Type V** — single wider arc (~93°)
   - **Type I** — two narrow arcs 180° apart (complementary)
   - **Type L** — one narrow + one wide, ~180° apart
   - **Type T** — split-complementary variant
   - **Type Y** — triadic variant
   - **Type X** — double-complementary
   - **Type N** — rectangle
3. For each template type, rotate in ~1° steps across 360°. At each rotation angle α, compute the harmony cost: `H(T, α) = Σ cᵢ · d(hᵢ, T_α)` where cᵢ is the chroma weight and d is the shortest angular distance from hue hᵢ to the nearest sector boundary of template T at rotation α.
4. The best-fitting template and rotation minimize this cost.
5. Normalize: `score = 1.0 − (min_cost / max_theoretical_cost)`.

**Performance note:** 8 types × 360 rotations = 2,880 evaluations. Each evaluation is a vectorized NumPy operation over the hue histogram (typically 360 bins). Total computation: ~10–50ms on CPU. No GPU needed.

**Tag generation:** The best-fit template type produces tags: `harmonic_complementary` (type I), `harmonic_analogous` (type V), `harmonic_triadic` (type Y), etc.

### 5.5 Anime-Specific Considerations

- **Cel shading produces clean K-means clusters.** Anime's flat color regions with sharp boundaries make palette extraction more reliable than on photography.
- **Deliberately limited palettes.** K=5–8 is appropriate for anime. Higher K extracts shading variants rather than distinct palette entries. Merge clusters within a CIEDE2000 threshold (~5) to collapse shadow/base variants of the same hue.
- **Foreground/background palette separation.** If segmentation data is available from the Subject analyzer, compute cross-layer palette cohesion — how well the character palette and background palette fit the same Matsuda template. This is a more nuanced, anime-appropriate measure than whole-image harmony alone.
- **Color temperature is descriptive, not evaluative.** Warm is not better than cool. Score temperature *consistency* (how clearly the frame commits to a temperature) and *distinctiveness* (how far from neutral).

### 5.6 Computational Profile

- **Palette extraction:** ~20–50ms CPU (downsample + K-means)
- **Matsuda harmony scoring:** ~10–50ms CPU (vectorized template fitting)
- **All other sub-properties:** <10ms CPU each
- **Total per image:** ~50–120ms CPU, no GPU required
- **Dependencies:** OpenCV, NumPy, SciPy, scikit-learn (KMeans). Optional: `colour-science` for CIEDE2000 and CCT.

### 5.7 Suggested Default Sub-Property Weights

| Sub-Property | Default Weight | Rationale |
| --- | --- | --- |
| Color harmony | 0.25 | Highest-value signal — captures intentional palette design |
| Palette cohesion | 0.15 | Measures cross-layer unity; important for anime compositing quality |
| Saturation balance | 0.10 | Controlled saturation indicates design intent |
| Color contrast | 0.15 | Subject-background separation; ties to readability |
| Color temperature | 0.05 | Descriptive — consistency scored, not temperature itself |
| Palette diversity | 0.10 | Moderate weight — both low and high diversity can be excellent |
| Vivid color | 0.20 | Directly captures chromatic impact; validated by AADB as independent attribute |

---

## 6. Detail — Hybrid (Classical + Segmentation)

### 6.1 Recommended Approach

**Classical CV for all detail measurement, with learned segmentation for region separation.** The anime segmentation model (skytnt/anime-segmentation) separates foreground (character) from background regions. Classical metrics are computed per region and combined with configurable weights.

### 6.2 Per-Sub-Property Techniques

| Sub-Property | Technique | Key Implementation | Anime Reliability | Cost |
| --- | --- | --- | --- | --- |
| **Edge density** | Laplacian variance | `cv2.Laplacian(gray, cv2.CV_64F).var()` — higher variance = more edges = more detail | HIGH | Very low |
| **Spatial frequency** | High-frequency energy ratio via FFT | `np.fft.fft2` → radial energy binning → ratio of energy above cutoff to total | MEDIUM-HIGH | Low |
| **Texture richness** | GLCM entropy | `skimage.feature.graycomatrix` at multiple distances [1,3,5] and 4 angles → entropy of normalized GLCM. Quantize to 64 levels first, downsample to ~512px | MEDIUM | Low-medium |
| **Shading granularity** | V-histogram mode count + entropy | Convert to HSV → V channel histogram → smooth → `scipy.signal.find_peaks` for tone count. Entropy of V histogram for continuous score | HIGH | Very low |
| **Line work quality** | Edge sharpness at detected edges | `cv2.Canny` for edge positions → `cv2.Sobel` magnitude at edge pixels → mean sharpness. Supplement: skeletonize + connected components for line continuity | HIGH | Low |
| **Rendering clarity** | Global Laplacian variance + local blur detection | Global: reuse edge density metric (dual-purpose). Local: patch-wise Laplacian variance → flag regions below threshold | HIGH | Low |
| **Region separation** | Anime segmentation | skytnt/anime-segmentation (ISNet-IS, Apache-2.0) → character mask | HIGH | Moderate (GPU, ~50–100ms) [T] |

### 6.3 Background vs. Character Detail (Resolved)

Compute all detail sub-metrics independently for the background region and the character region:

```plaintext
detail_background = weighted_combination(sub_scores on background pixels)
detail_character = weighted_combination(sub_scores on character pixels)
detail_overall = bg_weight * detail_background + char_weight * detail_character
```

**Default weights:** 60% background / 40% character (configurable).

**Edge cases:**

- No character detected (landscape/scenery): 100% background score.
- Character fills >90% of frame (extreme close-up): 100% character score.
- Multiple characters: all character pixels are grouped as one foreground region.

### 6.4 IQA Models — Not Recommended for Detail

| Model | Assessment |
| --- | --- |
| **BRISQUE / NIQE** | Not recommended. These models measure deviation from Natural Scene Statistics. Anime systematically violates NSS — flat fills and sharp synthetic edges are "defects" to BRISQUE but "correct rendering" in anime. Would require anime-specific retraining (out of scope). |
| **Aesthetic scorers** | Not useful for detail specifically. They capture overall aesthetic appeal, not decomposable detail metrics. A minimalist frame and a hyper-detailed frame may score similarly. |
| **CLIP** | Not sensitive to detail level differences within the anime domain. Two anime frames at different detail levels with similar content produce similar embeddings. |

### 6.5 Anime-Specific Considerations

- **Key frames vs. in-betweens** have very different detail levels. Key frames receive more rendering passes — additional shadow tones, highlight details, fabric folds. This is exactly the variance the Detail analyzer should capture.
- **Motion blur and compositing effects** can inflate edge density metrics. Frames with heavy motion blur may register as "high detail" by edge metrics despite being perceptually blurry. The Laplacian variance metric partially handles this (blur reduces variance), and the multi-sub-property approach naturally dilutes the effect.
- **GLCM on anime** is less informative than on natural textures because anime's flat regions produce degenerate GLCM distributions. It works better on detailed backgrounds (foliage, cityscapes) than on character regions. Weight accordingly.

### 6.6 Computational Profile

- **Classical measurements (all combined):** ~50–100ms CPU
- **Anime segmentation (shared):** ~50–100ms GPU [T]
- **Total per image:** ~100–200ms (segmentation may be shared with other analyzers)
- **Dependencies:** OpenCV, NumPy, SciPy, scikit-image (GLCM, skeletonize). Segmentation model: PyTorch.

### 6.7 Suggested Default Sub-Property Weights

| Sub-Property | Default Weight | Rationale |
| --- | --- | --- |
| Edge density | 0.20 | Primary complexity signal; reliable on anime |
| Spatial frequency | 0.15 | Complements edge density with frequency-domain perspective |
| Texture richness | 0.10 | Less reliable on anime flat regions; useful for backgrounds |
| Shading granularity | 0.20 | Directly targets anime rendering quality (tone count, shading complexity) |
| Line work quality | 0.20 | Important quality differentiator for anime specifically |
| Rendering clarity | 0.15 | Captures sharpness; dual-purpose with edge density |

---

## 7. Lighting — Hybrid (Classical-Heavy)

### 7.1 Recommended Approach

**Classical CV for all lighting measurement, with anime segmentation for boundary-dependent analysis and WD-Tagger tags as supplement.** Lighting is the dimension where classical CV faces the most conceptual mismatch with anime's non-physical lighting, but statistical and feature-detection approaches still extract meaningful signals.

The key insight: focus on detecting the *presence and properties* of lighting features (contrast, rim light, shadow characteristics, atmospheric effects) rather than inferring *physical lighting parameters* (light source position, intensity falloff). Anime lighting is artistic, not physical — but its visual effects are measurable.

### 7.2 Per-Sub-Property Techniques

| Sub-Property | Technique | Key Implementation | Anime Reliability | Cost |
| --- | --- | --- | --- | --- |
| **Contrast ratio** | Percentile contrast on V channel | `np.percentile(V, 95) - np.percentile(V, 5)` — robust to outliers (specular highlights, black outlines) | HIGH | Very low |
| **Light directionality** | Grid luminance comparison | Divide V into 3×3 grid → compute mean per cell → identify dominant gradient direction. Output as categorical tag | MEDIUM | Very low |
| **Rim/edge lighting** | Boundary luminance differential | Dilate segmentation mask → compute mean V in boundary ring vs. nearby background. Positive differential = rim light present | HIGH | Low (requires segmentation) |
| **Shadow quality** | Shadow edge softness | Threshold V channel → detect shadow regions → compute gradient magnitude at shadow boundaries. High gradient = hard shadows; low = soft | MEDIUM-HIGH | Low |
| **Atmospheric lighting** | Bloom/glow detection | Compare V with `cv2.GaussianBlur(V, large_ksize)` → regions where original > blurred by threshold = bloom sources. Score = bloom coverage + intensity | MEDIUM | Low |
| **Highlight/shadow balance** | Tri-zone ratio | Proportion of V pixels in shadows (<0.3), midtones (0.3–0.7), highlights (>0.7). Classify: high-key, low-key, balanced | HIGH | Very low |

### 7.3 WD-Tagger Lighting Tags (Supplement)

WD-Tagger v3 produces Danbooru tags including lighting-relevant terms [T — exact tag names need verification against selected_tags.csv]:

- `backlighting`, `rim_lighting`, `sunlight`, `moonlight`, `spotlight`
- `dramatic_lighting`, `lens_flare`, `light_rays`, `glowing`

These are **supplementary signals**, not primary score inputs. Use them for tag generation alongside classical CV-derived tags. A WD-Tagger `rim_lighting` prediction can corroborate the classical rim-light detection, increasing tag confidence.

### 7.4 Anime-Specific Challenges

**What works well (medium-independent):**

- Contrast ratio, histogram statistics, tri-zone balance — pure pixel statistics with no physical assumptions.
- Rim/edge lighting detection — anime rim lights are high-contrast effects at subject boundaries, readily detectable.
- Shadow edge softness — anime shadow boundaries have clear stylistic signatures (hard cel vs. soft gradient).

**What works with reduced reliability:**

- Light directionality — coarse grid analysis gives approximate direction, but anime's multi-source or non-physical lighting can produce ambiguous results. Express as a confidence-tagged categorical label, not a precise angle.
- Atmospheric effects beyond bloom — light shafts are detectable via Hough on thresholded brightness, but false positives from other bright elongated regions are likely. Haze detection via dark channel prior works but is less reliable for anime's stylized fog.

**What doesn't transfer from photography:**

- Physical light source estimation (face shading analysis, specular highlight geometry).
- Inverse rendering approaches.
- Any technique assuming Lambertian reflectance or physically-based light transport.

### 7.5 Computational Profile

- **Classical measurements (all combined):** ~30–60ms CPU
- **Anime segmentation (shared):** ~50–100ms GPU [T] (already loaded for Detail/Subject)
- **WD-Tagger inference (shared):** part of shared tagger pass
- **Total per image:** ~30–60ms incremental (segmentation shared)
- **Dependencies:** OpenCV, NumPy, SciPy. Segmentation model: PyTorch (shared).

### 7.6 Suggested Default Sub-Property Weights

| Sub-Property | Default Weight | Rationale |
| --- | --- | --- |
| Contrast ratio | 0.25 | Most reliable and universally meaningful signal |
| Light directionality | 0.10 | Lower weight due to reduced anime reliability |
| Rim/edge lighting | 0.20 | Distinctive anime technique; reliably detectable |
| Shadow quality | 0.15 | Captures shading sophistication |
| Atmospheric lighting | 0.15 | Important for high-production-value frames; moderate reliability |
| Highlight/shadow balance | 0.15 | Reliable; captures overall tonal character |

---

## 8. Subject — Hybrid (Learned Detection + Classical Measurement)

### 8.1 Recommended Approach

**Learned models identify the subject; classical CV measures how well the frame emphasizes it.** This is the cleanest hybrid decomposition — learned models answer "what is the subject?" and classical CV answers "how effectively is it emphasized?"

### 8.2 Subject Identification (Learned)

| Model | Role | Output | Anime Reliability | Cost |
| --- | --- | --- | --- | --- |
| **skytnt/anime-segmentation** | Character mask extraction | Soft/binary character mask | HIGH (for characters) | Moderate (GPU, ~50–100ms) [T] |
| **deepghs face/person detection** | Subject location + scale | Bounding boxes for face/head/person | HIGH | Low-moderate (GPU, ~30–50ms) [T] |

These two models are complementary:

- Anime-segmentation provides a pixel-level mask — where exactly the character is.
- deepghs detection provides bounding boxes and focal points — where the face is (almost always where the viewer's eye should go).

**When no subject is detected** (no character mask, no face/person boxes): the analyzer reports low/null subject emphasis scores and tags the frame as `environment_focus` or `no_clear_subject`. This is information, not failure.

### 8.3 Per-Sub-Property Techniques

| Sub-Property | Technique | Key Implementation | Anime Reliability | Cost |
| --- | --- | --- | --- | --- |
| **Saliency strength** | Spectral residual saliency concentration within subject mask | `cv2.saliency.StaticSaliencySpectralResidual` [T] → measure fraction of saliency mass within subject mask | MODERATE | Very low |
| **Figure-ground separation** | Lab Delta-E + boundary edge strength | Convert to Lab → compute mean color in FG and BG regions → CIEDE2000 distance. Also: Sobel magnitude at mask boundary | HIGH | Low |
| **DOF effect** | Laplacian variance ratio (subject vs. background) | Per-region Laplacian variance. DOF strength = subject_variance / background_variance | GOOD | Low |
| **Negative space utilization** | Low-complexity area distribution around subject | Edge density thresholding (shared with Composition) → measure quiet area ratio and spatial distribution relative to subject centroid | GOOD | Low |
| **Subject completeness** | Mask vs. frame boundary contact | Percentage of subject mask pixels touching frame edges. Low contact = complete subject. High contact = cropped | HIGH | Trivial |
| **Subject scale** | Mask area / frame area | Direct ratio from segmentation mask or detection bounding box | HIGH | Trivial |

### 8.4 Anime-Specific Considerations

- **Anime-segmentation is purpose-built** for this domain. Unlike general salient object detection, it specifically identifies drawn characters — the most common subject in anime screenshots.
- **Multi-subject compositions:** Multiple detected characters form one "subject region" (union of masks). The emphasis measurements then assess how well the group is emphasized as a whole.
- **Landscape/scenery shots:** When no character is detected, subject emphasis scores are inherently low. This is correct — these frames score on other dimensions (composition, color, detail) rather than subject emphasis.
- **Scale categorization is useful for tagging:** extreme close-up (>60% frame area), close-up (30–60%), medium (15–30%), wide (5–15%), very wide (<5%).

### 8.5 Computational Profile

- **Anime segmentation (shared):** ~50–100ms GPU [T]
- **deepghs detection:** ~30–50ms GPU [T]
- **Classical measurements (all combined):** ~20–40ms CPU
- **Total per image:** ~100–190ms (segmentation shared with Detail/Lighting/Style)
- **Dependencies:** PyTorch (segmentation, detection), OpenCV, NumPy.

### 8.6 Suggested Default Sub-Property Weights

| Sub-Property | Default Weight | Rationale |
| --- | --- | --- |
| Saliency strength | 0.15 | Moderate reliability; complements figure-ground |
| Figure-ground separation | 0.25 | Most direct measure of emphasis; high reliability on anime |
| DOF effect | 0.15 | Strong signal when present; absent in many anime frames |
| Negative space utilization | 0.15 | Meaningful for wallpaper use case |
| Subject completeness | 0.15 | Important for screenshot quality — awkward crops reduce quality |
| Subject scale | 0.15 | Informs emphasis; extreme values (very small or very large) are informative |

---

## 9. Style — Learned (Primarily)

### 9.1 Recommended Approach

Style is the most challenging dimension. RQ1 recommended splitting it into **style tagging** (categorical, feasible) and **style coherence scoring** (0.0–1.0, uncertain feasibility). This research confirms that split and provides a concrete v1 approach.

**v1 architecture:**

1. **Style tags** (HIGH confidence): WD-Tagger v3 style-relevant tags + CLIP zero-shot classification → categorical labels.
2. **Style quality score** (MODERATE confidence): Anime aesthetic scorer → proxy score for style quality (not pure coherence).
3. **Layer consistency sub-score** (LOW confidence, experimental): Classical CV rendering property comparison across segmented regions.

### 9.2 Style Tagging (Categorical — HIGH Feasibility)

| Method | Tags Produced | Anime Reliability | Cost |
| --- | --- | --- | --- |
| **WD-Tagger v3** style tag extraction | Art style: `flat_color`, `gradient`, `realistic`, `sketch`, `watercolor_(medium)`. Rendering: `cel_shading`, `soft_shading`, `hard_shadow`, `chromatic_aberration`, `bloom`. Quality: `detailed`, `simple_background` [T] | Very high (Danbooru-trained) | Shared with other tagger uses |
| **CLIP zero-shot classification** | Broader style categories: "naturalistic anime", "geometric abstract anime", "painterly anime", "digital modern anime", "retro cel anime" | Moderate-high (with anime-tuned CLIP) | Shared with CLIP uses |

WD-Tagger provides granular style-relevant tags from Danbooru's vocabulary. CLIP zero-shot provides broader stylistic categorization that Danbooru tags don't cover. Together they produce a rich categorical description of the frame's artistic approach.

### 9.3 Style Quality Score (Proxy — MODERATE Feasibility)

The anime aesthetic scorers from RQ3 produce quality scores that partially correlate with style execution quality:

| Model | Training | Output | Pros | Cons |
| --- | --- | --- | --- | --- |
| **deepghs/anime_aesthetic** (SwinV2) | Anime, 7-tier, AUC 0.82 | Quality tier prediction | Published benchmarks, anime-specific | Entangled with other quality aspects |
| **kawaimasa/kawai-aesthetic-scorer** (ConvNeXtV2) | 60K manually labeled anime images | Aesthetic score | Best-documented, Apache-2.0, large labeled dataset | Also entangled with overall quality |

**What these actually measure:** A blend of detail quality, color appeal, composition, rendering quality, and (to some extent) style consistency. A frame with inconsistent rendering (some areas detailed, others rough) tends to score lower. But so does a deliberately minimalist frame that is stylistically coherent.

**Recommendation:** Use deepghs/anime_aesthetic as the primary style quality score for v1. Document clearly that this is an "aesthetic quality proxy," not a pure coherence measure. Normalize its output to [0.0, 1.0].

### 9.4 Layer Consistency (Experimental — LOW Feasibility)

This is the most promising approach for measuring actual within-frame style coherence, but it is novel and unvalidated:

1. **Segment** the frame into foreground/background using anime-segmentation.
2. **Compute per-region visual properties:**
   - Detail density: edge density, GLCM entropy, high-frequency FFT energy
   - Color properties: dominant hue histogram, mean saturation
   - Rendering approach: gradient smoothness (local gradient variance — smooth shading vs. hard cel edges), edge sharpness
3. **Measure intra-layer consistency:** How similar are the rendering properties *within* the foreground layer? *Within* the background layer? High internal consistency = coherent rendering within each layer.

**Critical caveat:** Cross-layer consistency (foreground vs. background) should **not** be penalized. Anime *routinely* uses different rendering approaches for characters and backgrounds — this is a feature of the medium, not a defect. Only measure consistency *within* each layer.

**Assessment:** This is experimental. Include with low weight in the style score and flag as "experimental" in documentation. Its value will be determined empirically once the analyzer runs on real anime screenshots.

### 9.5 Approaches Evaluated and Not Recommended for v1

| Approach | Why Not |
| --- | --- |
| **Gram matrix style comparison** | Computationally expensive (CNN forward pass + O(C²) per region). VGG-19 features may not capture anime-relevant style properties. Shares the intentional-contrast problem. Marginal expected benefit over simpler layer consistency. |
| **CLIP embedding consistency** (crop regions + compare) | CLIP embeddings capture semantic content as much as style. Different regions with different content produce different embeddings regardless of style consistency. Cannot isolate the "style" component without training a projection — out of scope. |
| **True coherence vs. production reference** | Requires multi-frame context (comparing frame embedding to cluster centroid of the same production). Not feasible in Loupe's single-frame architecture. Defer entirely to a future version that could accept a "reference set" of frames from the same production. |

### 9.6 Anime-Specific Considerations

- **Style is the weakest dimension.** The tagging component is solid and useful for filtering/sorting. The scoring component is a best-available proxy, not a direct measurement of the target property.
- **The aggregate score weight for Style should be lower by default** to reflect this lower confidence.
- **Anime aesthetic scorers are the best available tool** — they are trained on anime data with human preference labels, making them the most domain-appropriate learned models available for any quality-related assessment.

### 9.7 Computational Profile

- **WD-Tagger inference (shared):** ~50–100ms GPU [T]
- **CLIP inference (shared):** ~20–50ms GPU [T]
- **Aesthetic scorer:** ~50–100ms GPU [T]
- **Layer consistency (classical):** ~30–50ms CPU
- **Total per image:** ~150–300ms (most shared with other analyzers)
- **Dependencies:** PyTorch (all learned models). Segmentation model shared.

### 9.8 Suggested Default Sub-Property Weights

| Component | Default Weight | Rationale |
| --- | --- | --- |
| Aesthetic quality score (scorer) | 0.70 | Most reliable signal available; anime-trained |
| Layer consistency (experimental) | 0.30 | Supplementary signal; lower weight due to uncertainty |

Style tags are categorical and do not contribute to the numerical score.

---

## 10. Shared Infrastructure and Computational Budget

### 10.1 Shared Models

Several learned models are used across multiple dimensions. The engine should load each once and share outputs:

| Model | Used By | Load Once? | VRAM (est.) |
| --- | --- | --- | --- |
| **skytnt/anime-segmentation** (ISNet-IS) | Detail, Lighting, Subject, Style | Yes | ~200MB [T] |
| **WD-Tagger v3** (SwinV2 or EVA02-Large) | Style (tags), Lighting (supplement), Color (supplement) | Yes | ~400–800MB [T] |
| **deepghs face/person detection** | Subject | Yes | ~100–300MB [T] |
| **Anime aesthetic scorer** (deepghs or kawaimasa) | Style | Yes | ~200–400MB [T] |
| **OpenCLIP ViT-L/14** (for CLIP zero-shot) | Style | Yes | ~900MB [T] |

**Total estimated VRAM:** ~1.8–2.7GB [T], well within the 8GB RTX 3070 budget. Sequential load/unload is likely unnecessary.

### 10.2 Shared Computations

| Computation | Produced By | Consumed By |
| --- | --- | --- |
| Grayscale image | Image loader | Composition, Detail, Lighting |
| HSV V channel | First use | Lighting, Detail (shading granularity) |
| OkLab/OkLCh conversion | Color analyzer | Color (all sub-properties) |
| Canny edge map | First use | Composition, Detail, Lighting |
| Segmentation mask | Engine (anime-segmentation) | Detail, Lighting, Subject, Style |
| LSD line segments | Composition | Composition (leading lines, diagonal dominance) |
| Saliency map | Composition or Subject | Composition, Subject, Color (contrast) |
| WD-Tagger predictions | Engine (WD-Tagger) | Style, Lighting |

### 10.3 Per-Image Time Budget

| Stage | Time (GPU) | Time (CPU) | Notes |
| --- | --- | --- | --- |
| Image loading + preprocessing | ~10ms | ~10ms | Pillow → ndarray → grayscale, HSV, OkLab |
| Anime segmentation | ~50–100ms | N/A (GPU only) | Shared across 4 dimensions |
| WD-Tagger inference | ~50–100ms | N/A | Shared across Style + supplements |
| Aesthetic scorer | ~50–100ms | N/A | Style dimension |
| CLIP inference | ~20–50ms | N/A | Style dimension (zero-shot) |
| deepghs detection | ~30–50ms | N/A | Subject dimension |
| All classical measurements | ~150–300ms | ~300–500ms | All 6 dimensions combined |
| Aggregate scoring | <1ms | <1ms | Weighted arithmetic mean |
| **Total** | **~360–700ms** | N/A | **~1–2 images/sec** |

This estimate assumes sequential GPU inference and parallel-capable classical computation. With batched GPU inference across an image set, throughput could be higher. The target is <1 second per image on the RTX 3070 with CUDA, which appears achievable.

---

## 11. Per-Dimension Recommendation Summary

This table consolidates the RQ2 output format specified in the research questions.

| Dimension | Approach | Candidate Techniques | Key Trade-offs | Confidence |
| --- | --- | --- | --- | --- |
| **Composition** | Hybrid (classical-heavy) | Edge-density saliency proxy → rule of thirds, balance, symmetry, leading lines, diagonal dominance, negative space (all classical geometry via OpenCV/NumPy) | Saliency quality on anime is the main risk. Classical geometry transfers perfectly. Depth layering is weakest sub-property. | HIGH |
| **Color** | Classical | K-means palette in OkLab, Matsuda harmony scoring (Cohen-Or), Hasler & Süsstrunk colorfulness, Simpson diversity, CIEDE2000 contrast | No learned component → no model management, no GPU needed, no domain-transfer risk. Requires implementing Matsuda scoring (no mature library exists). | HIGH |
| **Detail** | Hybrid (classical + segmentation) | Laplacian variance, FFT high-freq ratio, GLCM entropy, V-histogram mode count, edge sharpness; anime-segmentation for region split | Segmentation model adds GPU dependency. GLCM less informative on anime flat regions. Background/character weighting is configurable but default may need tuning. | HIGH |
| **Lighting** | Hybrid (classical-heavy) | Percentile contrast, grid luminance (directionality), boundary luminance differential (rim light), shadow edge softness, bloom/glow detection, tri-zone ratio; WD-Tagger tags | Anime's non-physical lighting limits directionality and atmospheric detection accuracy. Statistical measures (contrast, balance) are fully reliable. | MEDIUM |
| **Subject** | Hybrid (learned + classical) | anime-segmentation + deepghs detection → figure-ground Delta-E, DOF ratio, saliency concentration, subject scale/completeness | Strongest hybrid design — learned models are anime-specific and well-validated. Classical measurements are direct. Falls back gracefully when no subject detected. | HIGH |
| **Style** | Learned (primarily) | Anime aesthetic scorer (deepghs/kawaimasa) as quality proxy; WD-Tagger + CLIP zero-shot for style tags; experimental layer consistency | Weakest dimension. Score is a quality proxy, not true coherence. Layer consistency is unvalidated. Style tags are high-value categorical output. | LOW-MEDIUM |

---

## 12. Implementation Priorities

Based on feasibility, confidence, and dependency structure:

### Phase 1 — Foundation (highest confidence, no model dependency)

1. **Color analyzer** — fully classical, no models needed, highest confidence. Validates the analyzer framework and scoring pipeline.
2. **Composition analyzer** — classical geometry with edge-density saliency proxy. No model download required for v1.

### Phase 2 — Model-Dependent Dimensions

1. **Subject analyzer** — requires anime-segmentation and deepghs detection. Validates the model loading infrastructure.
2. **Detail analyzer** — requires anime-segmentation (shared with Subject). Validates region-separated measurement.
3. **Lighting analyzer** — requires anime-segmentation (shared). Lower confidence but incremental once segmentation is available.

### Phase 3 — Style (Lowest Confidence)

1. **Style analyzer** — requires aesthetic scorer, WD-Tagger, and CLIP. Most models, most uncertainty. Benefits from empirical data gathered during Phase 1–2 development.

This ordering minimizes risk: the most confident dimensions ship first, validating the architecture, before the less certain dimensions are implemented. Each phase builds on infrastructure established by the previous one.

---

## 13. Limitations

- **Training knowledge basis.** This research draws on knowledge with a May 2025 cutoff. Specific library APIs (OpenCV saliency module, skimage GLCM), model availability (deepghs detection suite versions), and VRAM measurements should be verified during RQ5 (Tooling Validation). Claims marked [T] are the primary verification targets.
- **No empirical validation.** All technique assessments are based on algorithmic analysis and domain reasoning, not on running the techniques against anime test images. Empirical validation during implementation may reveal that some techniques perform better or worse than predicted — particularly saliency detection on anime, GLCM texture analysis on flat-shaded content, and lighting directionality estimation.
- **Anime style diversity.** Assessments like "HIGH anime reliability" are generalizations across a diverse medium. Specific visual styles (extreme minimalism, heavy digital effects, 3D-rendered anime) may challenge techniques that work well on mainstream cel-shaded content.
- **Style dimension uncertainty.** The Style analyzer's scoring component is acknowledged as a proxy, not a direct measurement of the target property. Its value relative to the other five dimensions will only be clear after empirical testing.
- **Sub-property weights are educated defaults.** The suggested weights within each dimension are starting points based on domain reasoning. They should be tuned against user preference data once the system is operational.

---

## References

[^1]: Cohen-Or, D., et al. "Color Harmonization." SIGGRAPH 2006. <https://igl.ethz.ch/projects/color-harmonization/harmonization.pdf> — Computational formalization of Matsuda's harmonic hue templates.
