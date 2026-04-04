# RQ1 — Analysis Dimensions

**Date:** 2026-04-03
**Status:** Complete
**Question:** What aesthetic dimensions should Loupe measure, and what specific sub-properties comprise each dimension?

---

## 1. Methodology

### Search Strategy

Research was conducted across five knowledge domains:

1. **Computational aesthetics and IQA literature** — survey papers on image aesthetic assessment, multi-attribute scoring frameworks, and the dimensions researchers have validated as independently measurable
2. **Anime production and visual design** — how anime frames are constructed (layout, key animation, compositing/photography), the distinct visual properties of anime as a medium, and what the sakuga community values in frame quality
3. **Color theory and composition** — formalized models for color harmony (Matsuda, Itten), established composition principles (rule of thirds, golden ratio, Gestalt), and their computational formalization
4. **Anime/illustration ML community** — Danbooru tag taxonomy, how the anime tagging ecosystem categorizes visual properties, and what aesthetic dimensions are already captured in existing classification systems
5. **Practical curation** — what anime wallpaper communities and screenshot curators value when selecting high-quality frames

### Source Coverage

Key sources informing this analysis:

- **AADB (Aesthetics and Attributes Database)** [^1] — 10,000 images scored across 11 aesthetic attributes by professional photographers. Attributes span composition (rule of thirds, balancing elements, symmetry, repetition), color (harmony, vividness), lighting, focus (depth of field), and content (object emphasis, motion blur). The most granular multi-attribute dataset in the IQA literature.
- **UNIAA (Unified Multi-modal Image Aesthetic Assessment)** [^2] — proposes six core aesthetic dimensions: Content/Theme, Composition, Color, Light, Focus, and Sentiment. Evaluates through question-based assessment rather than scalar scoring.
- **EVA dataset** — categorizes aesthetics into Light and Color, Composition and Depth, Quality (distortions/artifacts), and Semantic Content.
- **CADB (Composition Aesthetics DataBase)** — focuses on composition attributes: Rule of Thirds, Symmetry, Color Harmony, Depth of Field, Motion Blur.
- **NIMA (Neural Image Assessment)** [^3] — predicts aesthetic score distributions using CNNs. Identifies colorfulness, contrast, composition, lighting, and subject as key perceptual features influencing aesthetic judgment.
- **Matsuda color harmony model** [^4] — defines 8 harmonic hue template types over the hue wheel, producing 80 color schemes. Provides a formal geometric framework for quantifying color harmony as angular relationships in HSV/OkLCh color space.
- **Anime production resources** — Washi's analysis of depth and compositing in anime [^5], Shinobi Creative's documentation of T-light (T光) effects [^6], analysis of anime framing and blocking techniques [^7], and the Anime Analytica overview of anime visual styles [^8].
- **Danbooru tag system** [^9] — extensive composition tag group (angles, framing, view types), quality tags (masterpiece, best_quality), and color/style tags. The most comprehensive taxonomy of anime visual properties in existence, though oriented toward content description rather than aesthetic quality measurement.

### Scope Decisions

- **Single-frame analysis only.** Temporal properties (animation fluidity, framerate modulation, motion quality) are out of scope — Loupe analyzes screenshots, not sequences.
- **Anime-specific grounding.** All dimensions are defined in terms of anime visual properties, not photography. Where IQA literature provides a photographic framework, it is adapted to the anime domain with explicit notation of where the analogy breaks down.
- **Intrinsic properties preferred.** Following the CLAUDE.md constraint about cross-show comparability, dimensions should measure intrinsic visual properties (gradient complexity, palette coherence, composition geometry) rather than style-relative qualities where possible.

---

## 2. Anime as a Distinct Visual Domain

Before defining dimensions, it is essential to understand why anime requires domain-specific aesthetic analysis rather than direct application of photographic IQA frameworks.

### Structural Properties of Anime Frames

Anime frames are composites of distinct visual layers with fundamentally different rendering approaches [^5]:

- **Background art** — often painted (watercolor, gouache, or digital painting), with rich texture, atmospheric perspective, and environmental detail. Background art quality varies enormously between productions and is a major differentiator of visual prestige.
- **Character art** — defined by line art (inked contours) and flat or gradient cel shading. Color palettes for characters are deliberately limited and pre-designed by the color designer (色彩設計). Shading is typically 2-3 tone levels rather than continuous gradients.
- **Compositing layer (photography/撮影)** — the final stage where layers are combined and post-processing effects applied: lighting effects (T-lights/T光, rim lighting, specular highlights), depth-of-field blur, atmospheric fog, color grading, and lens simulation effects [^6]. Modern anime derives much of its visual polish from this stage.

This layered construction means that "detail" in anime is not a single property — background detail, character rendering complexity, and compositing sophistication are semi-independent axes.

### How Anime Differs from Photography

| Property | Photography | Anime |
| --- | --- | --- |
| Color palette | Continuous, captured from scene | Deliberately designed, often limited |
| Shading | Continuous gradients from real lighting | Discrete tone levels (cel shading), 2-3 steps typical |
| Depth of field | Optical, from lens properties | Simulated in compositing; varies by production budget |
| Line art | Not applicable | Primary structural element defining forms |
| Lighting | Captured from environment | Constructed through painted shadows + compositing effects |
| Detail distribution | Uniform across frame (lens captures everything) | Intentionally uneven — detailed backgrounds with simpler character rendering, or vice versa |
| Texture | Physical surface texture captured by sensor | Painted/digital texture, often stylized or simplified |

### Studio Style Variation

Different studios produce visually distinct work [^8]:

- **Kyoto Animation** — soft, naturalistic lighting; meticulous environmental detail; polished character rendering; subtle color grading that emphasizes warmth and atmosphere
- **Ufotable** — advanced digital compositing; high-contrast cinematic lighting; seamless 2D/3D integration; heavy post-processing effects
- **Studio Ghibli** — hand-painted backgrounds with watercolor textures; naturalistic color palettes; environmental storytelling through background detail
- **Shaft** — geometric and abstract compositions; unconventional framing; bold color choices; stylistic experimentation over naturalism

This variance means that a single "quality" standard cannot universally apply. Loupe's dimensions must measure properties that are meaningful *within* any given style — compositional balance, color harmony, and lighting quality are relevant regardless of whether the frame is a Ghibli landscape or a Shaft dialogue scene.

---

## 3. Candidate Dimension Analysis

Eight candidate dimensions were evaluated: the five proposed in the CLAUDE.md (Composition, Color, Detail, Style, Subject) plus three additional candidates identified in the research questions (Lighting, Visual Clarity, Line Art Quality).

### 3.1 Composition

**Definition:** The spatial arrangement and structural design of visual elements within the frame — how objects, characters, backgrounds, and negative space are organized to create visual impact, guide the viewer's eye, and establish depth.

**Sub-properties:**

| Sub-property | Description | Anime context |
| --- | --- | --- |
| Rule of thirds | Subject placement at or near third-line intersections | Common in character-focused shots; key frames often place eyes at upper-third intersections |
| Visual balance | Distribution of visual weight across the frame | Asymmetric balance (weighted by character position, background elements) is more common in anime than strict symmetry |
| Symmetry | Mirror-like or radial symmetry in frame layout | Used deliberately for dramatic or architectural shots; rarer in everyday scenes |
| Depth layering | Separation of foreground, midground, and background planes | Anime achieves this through differential blur, parallax movement cues, and atmospheric perspective in painted backgrounds [^5] |
| Leading lines | Lines within the image that direct the eye toward the subject | Architectural elements, perspective lines, character gaze direction, weapon/arm angles |
| Negative space | Intentional use of empty areas to emphasize the subject or create mood | Common in dialogue scenes, emotional beats, and minimalist compositions |
| Framing | Use of in-frame elements (doors, windows, foliage) to border the subject | A frequent anime cinematographic technique, especially in slice-of-life genres |
| Diagonal dominance | Presence of strong diagonal lines creating dynamism or tension | Action scenes, Dutch angles, dramatic tilted compositions (particularly Shaft productions) |

**Scores high:** A frame with clear structural intent — the eye is naturally guided to the subject, foreground and background create depth, visual weight is distributed with purpose (whether symmetric or deliberately asymmetric), and the arrangement creates either dynamic energy or contemplative stillness appropriate to the content.

**Scores low:** A visually flat frame where elements are placed without compositional intent — the subject is centered with no supporting structure, no depth separation between layers, empty areas feel accidental rather than purposeful, and no clear visual path guides the eye.

**Independence:** Composition is the most structurally distinct dimension. It concerns *where* elements are placed, not *what* they look like. A poorly colored frame can be beautifully composed; a richly detailed frame can have no compositional structure.

**Feasibility:** HIGH. Composition properties (thirds alignment, symmetry, visual balance, depth layers) are among the most established computational measurements in IQA literature. AADB independently scores rule_of_thirds, balancing_element, symmetry, and repetition [^1]. Edge detection, saliency maps, and spatial frequency analysis provide strong tooling foundations.

---

### 3.2 Color

**Definition:** The chromatic properties of the frame — palette design, harmony relationships between colors, saturation dynamics, contrast, and the overall color mood. In anime, color is a designed property (chosen by the color designer), not an incidental capture from the environment.

**Sub-properties:**

| Sub-property | Description | Anime context |
| --- | --- | --- |
| Color harmony | How well the palette conforms to established harmony models (complementary, analogous, triadic, split-complementary) | Anime palettes are pre-designed, so strong harmony is a mark of intentional color design. Matsuda's harmonic templates [^4] provide a formal measurement framework |
| Palette cohesion | How well colors across all frame layers (background, characters, effects) work together | The background art may use a different palette register than character art — cohesion across layers is a distinct quality |
| Saturation balance | Distribution and control of saturation across the frame | Anime often uses saturation strategically — desaturated backgrounds with saturated character accents, or uniformly muted palettes for mood |
| Color contrast | Tonal and chromatic difference between key elements | High contrast between subject and background aids readability; dramatic color contrast creates visual impact |
| Color temperature | Warm/cool balance and its consistency or intentional variation | Color grading in the compositing stage shifts temperature for mood — warm golden tones for nostalgia, cool blues for tension |
| Palette diversity | Number of distinct hue groups present and their distribution | Some frames achieve impact through restricted palettes (near-monochromatic); others through rich, diverse color |
| Vivid color | Intensity and vibrancy of the overall color impression | AADB separates vivid_color from color_harmony [^1] — a harmonious palette can be muted, and a vivid palette can be disharmonious |

**Scores high:** A frame where the color palette feels intentional and unified — colors across background, characters, and effects support the same mood; harmony relationships are present between dominant hues; saturation is controlled rather than arbitrary; and the overall chromatic impression is distinctive and memorable.

**Scores low:** A frame with muddy, conflicting, or arbitrary colors — hues that clash without purpose, inconsistent saturation across elements, no discernible color mood, or a palette that feels generic rather than designed.

**Independence:** Color is independent of composition (the same spatial arrangement can have wildly different color treatments) and largely independent of detail (a simple, low-detail frame can have excellent color design). Some correlation with lighting exists — lighting affects perceived color — but color harmony measures the palette relationships themselves, not the illumination.

**Feasibility:** HIGH. Color harmony quantification is well-formalized through Matsuda templates and geometric models in HSV/OkLCh color space [^4]. Palette extraction, histogram analysis, and harmony scoring are established computational techniques. AADB validates color_harmony and vivid_color as independently ratable attributes [^1].

---

### 3.3 Detail

**Definition:** The visual complexity, rendering effort, and technical execution quality of the frame — how much visual information is present, how finely it is rendered, and how well the rendering is executed across all frame layers.

This dimension is broadened from the CLAUDE.md's "detail density" to encompass rendering quality more generally. The rationale: in anime, the *quality* of rendering matters as much as the *quantity* of detail. A frame with moderate detail but crisp, confident execution often looks better than a frame with high detail density but sloppy rendering.

**Sub-properties:**

| Sub-property | Description | Anime context |
| --- | --- | --- |
| Background detail density | Amount of visual information in the background layer | Ranges from minimalist color fields to Ghibli-level painted environments with foliage, architecture, and atmospheric detail |
| Character rendering complexity | Sophistication of character shading, facial detail, and form definition | Key frames often receive more rendering passes — additional shadow tones, highlight details, fabric folds |
| Shading granularity | Number of tone levels and smoothness of transitions in cel shading | Two-tone vs. three-tone vs. gradient shading; soft vs. hard shadow edges; the presence of ambient occlusion shading |
| Texture richness | Presence and quality of surface textures (fabric, hair, environmental surfaces) | Some productions render fabric weave, hair strand detail, and surface imperfections; others use flat fills |
| Edge density | Spatial frequency of edges and contours across the frame | A proxy for overall visual complexity; high edge density indicates intricate rendering |
| Line work quality | Crispness, consistency, and expressiveness of drawn lines | Confident line weight variation, clean intersections, intentional thick/thin transitions (see Section 4.2 for why this is merged here rather than standalone) |
| Rendering clarity | Sharpness and cleanliness of the rendered output | Absence of encoding artifacts, compression noise, or unintentional blur; clean anti-aliasing on lines (see Section 4.3 for merge rationale) |

**Scores high:** A frame that exhibits high rendering effort and execution quality — rich background painting with environmental texture, multi-tone character shading with nuanced shadow placement, crisp and confident line work, and fine details like fabric folds, hair strands, or atmospheric particles.

**Scores low:** A frame with minimal rendering effort — flat, unshaded character art; simple gradient or solid-color backgrounds; thick, uniform line weight without variation; and an overall impression of visual simplicity or rushed production.

**Independence:** Detail is independent of composition (a cluttered, poorly composed frame can be highly detailed) and largely independent of color (a monochromatic frame can be extremely detailed). Some correlation with lighting — detailed rendering often includes detailed shadow work — but detail measures the *amount and quality* of rendered information, while lighting measures the *illumination design*.

**Feasibility:** HIGH. Spatial frequency analysis, edge detection (Canny, Sobel), texture entropy (GLCM), and gradient magnitude distributions are established classical CV techniques for measuring visual complexity. The challenge is weighting background vs. character detail appropriately, which may require saliency-guided region-specific measurement.

---

### 3.4 Lighting

**Definition:** The quality, design, and impact of illumination in the frame — how light sources (real or implied) create contrast, atmosphere, depth, and visual hierarchy. In anime, lighting is a constructed property applied during the compositing/photography (撮影) stage [^5] [^6].

**Rationale for addition:** Lighting is identified as an independent dimension in both AADB (`good_lighting`) [^1] and UNIAA (`Light`) [^2]. The IQA literature consistently separates lighting from color — a well-lit frame with a dull palette is different from a flat-lit frame with rich colors. In anime specifically, the photography/compositing stage applies lighting effects (T-lights, rim lights, volumetric light shafts, shadow mapping) as a distinct creative layer [^6], making it conceptually and practically separable from base color design.

**Sub-properties:**

| Sub-property | Description | Anime context |
| --- | --- | --- |
| Contrast ratio | Luminance range between highlights and shadows | High contrast creates drama and depth; low contrast creates softness or flatness |
| Light directionality | Presence and clarity of implied light source direction | Key lighting, side lighting, backlighting — each creates different shadow patterns and depth cues |
| Rim/edge lighting | Light outlining subjects to separate them from backgrounds | Common in anime compositing; T-light effects (T光) add specular highlights and rim glow [^6] |
| Shadow quality | Design and rendering of shadows (hard vs. soft, detailed vs. flat) | Well-designed shadow shapes add form and depth; flat, arbitrary shadows reduce dimensionality |
| Atmospheric lighting | Volumetric effects, light shafts, environmental glow, haze | Compositing adds god rays, dust motes in light beams, window light scatter — hallmarks of high-production-value anime |
| Highlight/shadow balance | Distribution of light and dark areas across the frame | A well-balanced frame uses light to create visual hierarchy without crushing shadows or blowing highlights |

**Scores high:** A frame where lighting creates depth, atmosphere, and visual hierarchy — clear light direction establishing form, rim lighting separating subjects from backgrounds, atmospheric effects adding environment, and shadow design that enhances rather than flattens the image. Ufotable's cinematic compositing exemplifies high lighting scores.

**Scores low:** A frame with flat, uniform illumination — no discernible light direction, no atmospheric depth, shadows that are either absent or arbitrarily placed, and an overall impression of visual flatness regardless of the underlying detail or color quality.

**Independence:** Lighting is separable from color (a well-lit frame can have a limited palette; a richly colored frame can be flatly lit), from composition (lighting design is about illumination, not spatial arrangement), and from detail (a simple scene can be dramatically lit; a detailed scene can have flat lighting). Moderate correlation with subject emphasis — lighting often highlights the subject — but the mechanism differs (subject measures *what* draws the eye; lighting measures *how illumination is designed*).

**Feasibility:** MEDIUM-HIGH. Luminance histogram analysis, gradient-based shadow detection, and local contrast measurement are established. Detecting light directionality and atmospheric effects is more challenging but feasible through luminance gradient analysis and local intensity patterns. The AADB validates `good_lighting` as an independently ratable attribute [^1].

---

### 3.5 Subject

**Definition:** The clarity, emphasis, and visual dominance of the frame's primary focal region — how effectively the viewer's attention is directed to a coherent area of interest, and how well that region is separated from its surroundings. "Subject" here means the perceptual focus of the frame, not necessarily a single entity — two characters in a fight, a group in formation, or even a meaningful object can constitute the subject as long as they form a unified focal region.

**Sub-properties:**

| Sub-property | Description | Anime context |
| --- | --- | --- |
| Saliency strength | How strongly the focal region attracts visual attention relative to the rest of the frame | Measured through computational saliency maps; a concentrated saliency peak (even if spanning multiple characters) indicates clear focus |
| Figure-ground separation | Visual distinctness between the subject and background | In anime, this is often achieved through color contrast, blur, or lighting differences between character and background layers |
| Depth-of-field effect | Simulated optical blur that isolates the subject | Applied in compositing; not all anime uses this, but when present, it strongly directs focus. AADB scores `shallow_depth_of_field` independently [^1] |
| Negative space utilization | How empty space around the subject frames and emphasizes it | Effective negative space directs attention and creates breathing room; excessive empty space without compositional purpose feels barren |
| Subject completeness | Whether the subject is fully visible and well-framed (not awkwardly cropped) | A character cut off at an unnatural point, or a landscape with the focal element at the extreme edge, reduces subject clarity |
| Subject scale | Size of the subject relative to the frame | Medium shots to close-ups generally have clearer subject focus than wide establishing shots; subject scale affects how much of the frame the focal point occupies |

**Scores high:** A frame with a clear, coherent focal region — the subject (whether a single character, two fighters mid-clash, or a meaningful object) immediately draws the eye, is well-separated from the background through focus, lighting, or color contrast, and occupies the frame with intentional presence. A character close-up with soft background blur; two characters locked in combat against a motion-blurred background; a landscape with a clear point of visual interest against a harmonious but uncompetitive background.

**Scores low:** A frame where no element clearly dominates — the eye wanders without finding a focal point, multiple elements compete equally for attention, or the putative subject is lost in a busy background with insufficient separation.

**Independence:** Subject emphasis is related to but distinct from composition. Composition concerns the *spatial arrangement* of all elements; subject concerns the *perceptual dominance* of a focal region. A well-composed landscape with no single subject (intentionally even visual weight) scores high on composition but low on subject emphasis — and that is correct and desirable for that type of frame. AADB validates `object_emphasis` as independent from composition attributes [^1].

**Feasibility:** HIGH. Saliency detection is a mature field with strong models (Itti-Koch, deep learning saliency predictors). Depth-of-field detection via blur estimation is well-established. Figure-ground separation can be measured through local contrast and edge analysis at subject boundaries. The primary challenge is determining *what* the subject is in a frame without human annotation — saliency models handle this well for frames with clear focal points but struggle with evenly-distributed compositions (which should correctly score low on this dimension).

---

### 3.6 Style

**Definition:** The artistic identity, coherence, and execution quality of the frame's visual style — how consistently and effectively the frame realizes its intended aesthetic approach.

**Reframing note:** The CLAUDE.md positions style as "CLIP-based style tagging, aesthetic scoring." This research recommends splitting the style dimension into two distinct functions:

1. **Style tagging** (categorical, not scored) — identifying *what* style is present (e.g., "naturalistic," "geometric/abstract," "painterly," "cel-classic," "digital-modern"). This is classification, not quality assessment.
2. **Style coherence scoring** (0.0–1.0) — measuring *how well* the frame executes its style. This is the scorable dimension.

The rationale: a Shaft geometric composition and a KyoAni naturalistic scene represent different styles, neither inherently better. But within any style, execution quality varies — consistent rendering across layers, intentional rather than accidental visual choices, and unity between character art, background art, and compositing effects.

**Sub-properties (for the scored coherence dimension):**

| Sub-property | Description | Anime context |
| --- | --- | --- |
| Layer cohesion | Visual consistency between background art, character art, and effects | Jarring disparity between a hand-painted background and simple character art (or vice versa) reduces cohesion; deliberate stylistic integration increases it |
| Rendering consistency | Uniformity of shading approach, line weight, and detail level within the same visual layer | Mixed rendering approaches within a single character (some parts detailed, others rough) indicate inconsistency unless deliberate |
| Intentionality | Degree to which visual choices appear deliberate rather than accidental or default | Strong style scores feel like every visual element was chosen; weak scores feel generic or randomly assembled |
| Aesthetic distinctiveness | How much the frame's visual identity stands out from a generic baseline | Frames with a recognizable artistic voice (whether bold or subtle) score higher than visually generic frames |

**Scores high:** A frame where every visual element feels unified under a coherent artistic vision — the background art, character rendering, color design, and compositing effects all serve the same visual identity. Whether that identity is Ghibli's painterly naturalism or Shaft's geometric abstraction, the execution is consistent and intentional.

**Scores low:** A frame that feels visually incoherent or generic — mismatched rendering quality between layers, generic compositing applied without artistic consideration, or an overall impression that the frame could belong to any production without distinction.

**Independence:** Style coherence is conceptually independent from the other dimensions — a frame can be well-composed, well-colored, and well-lit but still feel stylistically generic. Conversely, a frame with strong stylistic identity may deliberately break composition rules or use dissonant color. In practice, moderate correlations with other dimensions are expected (high-style-coherence frames tend to also score well elsewhere because intentional production quality is holistic), but the dimension captures something the others do not: *artistic unity*.

**Feasibility:** LOW-MEDIUM. This is the most challenging dimension to automate. Style tagging (the categorical function) is feasible through CLIP-based classification with anime-tuned models. Style coherence scoring (the scored function) is harder — it requires measuring consistency and intentionality, which are abstract properties. Possible approaches:

- **CLIP embedding coherence**: measure how tightly the frame's embedding clusters with other frames from the same production (requires multi-frame context, complicating single-frame analysis)
- **Learned aesthetic scoring**: use anime-tuned aesthetic predictors as a proxy for style quality (conflates style with overall quality)
- **Layer consistency analysis**: compare rendering properties (detail density, color palette, edge characteristics) across detected regions (background vs. foreground) for cohesion

This dimension should be flagged for RQ2 to determine technical feasibility. If style coherence proves infeasible to automate reliably, the dimension can be reduced to style tagging (categorical labels only, no 0.0–1.0 score).

---

## 4. Overlap Analysis and Boundary Decisions

### 4.1 Lighting vs. Color — KEEP SEPARATE

**Concern:** Lighting affects perceived color. Are they truly separable?

**Resolution:** They measure different properties. **Color** measures the designed palette — what hues are chosen and how they relate to each other (harmony, saturation, diversity). **Lighting** measures how illumination is applied — direction, contrast, atmosphere, shadow design. A flat-lit scene can have excellent color harmony (muted palette with careful hue relationships). A dramatically lit scene can have a dull palette (monochromatic with high luminance contrast). The IQA literature consistently separates them: AADB has both `color_harmony`/`vivid_color` and `good_lighting` as independent attributes [^1]; UNIAA lists Color and Light as separate dimensions [^2].

**Implementation note:** Color analysis should operate primarily in chrominance channels (hue, saturation) while lighting analysis operates primarily in luminance channels (value, contrast). This makes them technically separable even when perceptually intertwined.

### 4.2 Line Art Quality — MERGE INTO Detail

**Concern:** Line art quality is a distinctive and important property of anime art. Should it be its own dimension?

**Resolution:** Merge as a sub-property of Detail. Rationale:

1. **Low discriminative variance within a show.** Line art quality is largely constant across frames from the same production — it reflects the studio's house style, the key animator's approach, and the digital pipeline settings. Between two screenshots from the same episode, line art quality varies little. Detail density and rendering effort vary significantly between frames (key frames vs. in-betweens, background detail shots vs. close-ups).
2. **Measurement overlap.** The computational signals for line art quality (edge sharpness, contour consistency, line weight distribution) substantially overlap with detail measurement signals (edge density, spatial frequency, texture analysis).
3. **Conceptual containment.** Line work quality is one aspect of rendering execution quality, which is what the Detail dimension measures. Separating it creates a narrow dimension that lacks sufficient sub-properties for a rich 0.0–1.0 score.

Line art quality is retained as a sub-property of Detail (line work quality) to ensure it is not ignored.

### 4.3 Visual Clarity — MERGE INTO Detail

**Concern:** Sharpness, noise, and artifact absence are important for screenshot quality. Should this be its own dimension?

**Resolution:** Merge as a sub-property of Detail. Rationale:

1. **Low variance given Loupe's input assumptions.** The CLAUDE.md specifies that input images are "clean screenshots without subtitles, network logos, or letterboxing — pre-screened by the user for cleanliness." If the user is pre-screening, severe quality issues (heavy compression, noise, blur from bad encoding) are already filtered out. The remaining variance in clarity is small.
2. **Correlation with detail.** In the absence of quality defects, visual clarity and detail density are positively correlated — cleaner rendering shows more detail. Measuring them separately would produce highly correlated scores.
3. **Dimension dilution.** Adding a dimension that barely varies across pre-screened inputs dilutes the aggregate score's discriminative power.

Rendering clarity is retained as a sub-property of Detail to capture the residual variance (some frames are slightly sharper than others, some have subtle compression artifacts).

### 4.4 Subject vs. Composition — KEEP SEPARATE

**Concern:** Focal point detection and spatial arrangement are closely related. Are they truly separable?

**Resolution:** They measure different things. **Composition** asks "how are all elements arranged in the frame?" **Subject** asks "does one element clearly dominate visual attention?" These diverge in important cases:

- A well-composed landscape with intentionally even visual weight across the frame scores high on composition, low on subject emphasis — correctly.
- A poorly composed frame where a character is awkwardly placed but still the obvious focal point scores low on composition, high on subject emphasis — correctly.
- A well-composed character shot with clear subject isolation scores high on both — correctly.

This separation matters for the user's workflow: some wallpaper-worthy frames are landscapes (strong composition, low subject emphasis); others are character portraits (strong subject emphasis, varied composition quality). The aggregate scoring should reward both.

### 4.5 Style — REFRAME, FLAG FOR RQ2

**Concern:** Style is qualitatively different from other dimensions. Can it be meaningfully scored 0.0–1.0?

**Resolution:** Split into two functions:

1. **Style tags** (categorical) — produced alongside the score, classifying the frame's visual approach. Not scored, not contributing to aggregate.
2. **Style coherence** (scored 0.0–1.0) — measuring execution quality and visual unity. This IS scored and contributes to aggregate.

This is flagged for RQ2 because the scoring function's technical feasibility is uncertain. If RQ2 determines that style coherence cannot be reliably automated, the fallback is to retain style tagging only and exclude style from the scored aggregate.

---

## 5. Final Validated Dimension List

| # | Dimension | What It Measures | Key Sub-properties | Feasibility | Status |
| --- | ----------- | ------------------ | ------------------- | ------------- | -------- |
| 1 | **Composition** | Spatial arrangement and structural design | Rule of thirds, visual balance, symmetry, depth layering, leading lines, negative space, framing, diagonal dominance | HIGH | Validated |
| 2 | **Color** | Palette design, harmony, and chromatic impact | Color harmony, palette cohesion, saturation balance, color contrast, color temperature, palette diversity, vivid color | HIGH | Validated |
| 3 | **Detail** | Visual complexity, rendering quality, technical execution | Background detail density, character rendering complexity, shading granularity, texture richness, edge density, line work quality, rendering clarity | HIGH | Validated (broadened from original "detail density") |
| 4 | **Lighting** | Illumination design and atmospheric effects | Contrast ratio, light directionality, rim/edge lighting, shadow quality, atmospheric lighting, highlight/shadow balance | MEDIUM-HIGH | Added (not in original five) |
| 5 | **Subject** | Focal point clarity and figure-ground separation | Saliency strength, figure-ground separation, depth-of-field effect, negative space utilization, subject completeness, subject scale | HIGH | Validated |
| 6 | **Style** | Artistic coherence and execution quality | Layer cohesion, rendering consistency, intentionality, aesthetic distinctiveness | LOW-MEDIUM | Reframed (split: tagging + coherence scoring); flagged for RQ2 feasibility assessment |

### Dimensions Considered and Rejected

| Candidate | Disposition | Rationale |
| --- | --- | --- |
| **Visual Clarity** | Merged into Detail (as "rendering clarity" sub-property) | Low variance given pre-screened inputs; high correlation with detail density; would dilute aggregate |
| **Line Art Quality** | Merged into Detail (as "line work quality" sub-property) | Low per-frame variance within productions; measurement overlaps with detail signals; conceptually contained within rendering quality |
| **Sentiment / Mood** | Not adopted | UNIAA includes Sentiment as a dimension, but it is highly subjective and context-dependent. A "sad" frame is not aesthetically worse than a "happy" frame. Mood is partially captured through color temperature and lighting atmosphere, which are already covered. Explicit mood scoring would be unreliable and inappropriate for Loupe's sort-by-quality use case |
| **Content / Subject Matter** | Not adopted | AADB includes `interesting_content` and UNIAA includes Content/Theme. But what constitutes "interesting" is entirely viewer-dependent. A character close-up is not inherently more interesting than a landscape. Loupe should measure visual execution quality, not content preference — the user's review pass handles content judgment |

---

## 6. Open Questions for RQ2

1. **Style coherence measurement**: What technical approaches can quantify style coherence from a single frame? Is CLIP embedding analysis sufficient, or does this require multi-frame context from the same production?
2. **Background vs. character detail weighting**: The Detail dimension includes both background and character rendering quality. Should these be measured separately and combined, or measured holistically? If separately, what weighting reflects user preference for wallpaper-quality screenshots?
3. **Lighting detection in anime**: Anime lighting is painted/composited, not physically simulated. How reliably can classical CV techniques detect light direction and atmospheric effects in anime frames, where shadows and highlights are artistic choices rather than physical phenomena?
4. **Cross-dimension normalization**: With six dimensions, the aggregate score (RQ4) needs to handle the case where some dimensions have more sub-properties than others. Should sub-properties be individually scored and averaged per dimension, or should each dimension produce a single holistic score?
5. **Anime-tuned models for composition**: General composition analysis models are trained on photography. Do they generalize to anime's distinctive composition patterns (extreme close-ups, Dutch angles, deliberate asymmetry)?

---

## 7. Limitations

- **Photography-centric literature bias**: The IQA literature is overwhelmingly focused on photographs. While the fundamental dimensions (composition, color, lighting, etc.) transfer to anime, the specific sub-property definitions and measurement approaches may need adaptation. This adaptation is deferred to RQ2.
- **Lack of anime-specific IQA research**: No academic papers were found that specifically address computational aesthetic assessment of anime or illustration content. The dimension definitions in this document bridge general IQA frameworks with anime production knowledge, but this bridge is based on domain reasoning rather than empirical validation on anime datasets.
- **Single-frame limitation**: Some aesthetic properties that matter for anime appreciation (animation fluidity, scene pacing, visual storytelling across cuts) are excluded because Loupe operates on individual frames. This is a deliberate scope boundary, not a gap.
- **Subjectivity of style**: The Style dimension's feasibility is uncertain, and its subjective nature may limit automated measurement accuracy. This is acknowledged and flagged for RQ2.

---

## References

[^1]: Kong, S., et al. "Photo Aesthetics Ranking Network with Attributes and Content Adaptation." ECCV 2016. <https://ar5iv.labs.arxiv.org/html/1606.01621> — Defines the AADB dataset with 11 aesthetic attributes scored by professional photographers.

[^2]: Zhou, Y., et al. "UNIAA: A Unified Multi-modal Image Aesthetic Assessment Baseline and Benchmark." 2024. <https://arxiv.org/html/2404.09619v1> — Proposes 6-dimension aesthetic framework with question-based assessment.

[^3]: Talebi, H. and Milanfar, P. "NIMA: Neural Image Assessment." IEEE TIP, 2018. <https://arxiv.org/abs/1709.05424> — CNN-based aesthetic score distribution prediction.

[^4]: Matsuda, Y. "Color Design." Asakura Shoten, 1995. Referenced via <https://github.com/xzr139/colorwheel> and Cohen-Or et al. "Color Harmonization" (<https://igl.ethz.ch/projects/color-harmonization/harmonization.pdf>) — 8 harmonic hue templates defining color harmony as geometric relationships on the hue wheel.

[^5]: "Depth in Anime – Photography, Compositing and Animation." Washi Blog, 2017. <https://washiblog.wordpress.com/2017/03/12/depth-in-anime-photography-compositing-and-animation/> — Analysis of how anime achieves depth through layered compositing.

[^6]: "What is T-light (T光) effects? Understanding T-light effects in Japanese animation." Shinobi Creative. <https://shinobicreative.com/en/blogs/animator-guidance-blogs/what-is-t-lightt%E5%85%89-effects-understanding-t-light-effects-in-japanese-animation> — Documentation of anime-specific lighting effects in the compositing stage.

[^7]: "Framing and Blocking in Anime." I Drink and Watch Anime, 2020. <https://drunkenanimeblog.com/2020/04/29/framing-and-blocking-in-anime/> — Analysis of anime composition and cinematography techniques.

[^8]: "The Anime Look: Visual Styles & Aesthetics." Anime Analytica. <https://animeanalytica.com/the-anime-look-visual-styles-aesthetics/> — Overview of anime visual styles across studios and eras.

[^9]: Danbooru tag system. Referenced via <https://gigazine.net/gsc_news/en/20221119-danbooru-tag-novelai-waifu-diffusion/> and <https://civitai.com/articles/25464/common-style-tags-recognized-by-illustrious-and-other-danbooru-based-models> — Comprehensive anime image tagging taxonomy including composition, quality, and style tags.

Additional sources consulted:

- "A Survey on Image Aesthetic Assessment." Li et al., 2021. <https://arxiv.org/abs/2103.11616>
- "Image Aesthetic Assessment: An Experimental Survey." Deng et al., 2016. <https://arxiv.org/abs/1610.00838>
- "Explaining Automatic Image Assessment." 2025. <https://arxiv.org/html/2502.01873v1>
- "Staging in Animation: Visual Composition Guide." Animotions Studio. <https://animotionsstudio.com/animation-staging/>
- "The Unseen Art of Anime Storyboarding." Anime Herald, 2025. <https://www.animeherald.com/2025/10/25/the-unseen-art-of-anime-storyboarding/>
- "Animation fundamentals — Layout and storyboard." Full Frontal. <https://fullfrontal.moe/animation-fundamentals-layout-and-storyboard/>
- "How Lighting Imparts Emotion in an Animated Scene." Envato Tuts+. <https://photography.tutsplus.com/tutorials/how-lighting-in-an-animation-creates-mood-ambience-and-immerses-the-viewer--cms-41429>
- "Sakuga Animation in Anime: The Art of Emphasizing Movement." Nihon Narratives. <https://nihonnarratives.substack.com/p/sakuga-animation>
- BRISQUE: "No-Reference Image Quality Assessment in the Spatial Domain." <https://live.ece.utexas.edu/publications/2012/TIP%20BRISQUE.pdf>
