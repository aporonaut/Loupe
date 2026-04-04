# RQ4 — Aggregate Scoring Strategy

**Research date:** 2026-04-03
**Status:** Complete

**Question:** How should per-dimension scores combine into a single sortable aggregate score?

---

## Scope and Motivation

Loupe measures anime screenshots across multiple independent aesthetic dimensions (composition, color, detail, style, subject), each producing a score on [0.0, 1.0]. The user's workflow is: analyze a batch of hundreds of images, sort by aggregate score, review top-down. The aggregate determines what "sorts to the top" — it is the most UX-critical computation in the system.

This research surveys aggregation methods from multi-criteria decision analysis (MCDA), image quality assessment (IQA), practical ranking systems, and social choice theory. It evaluates each against Loupe's specific constraints: incremental batch processing, user-configurable behavior, interpretable rankings, and the need to handle images that are "good for different reasons."

**Deliberately excluded:** Training a learned aggregation model (requires preference data Loupe doesn't have), GUI/interactive weight tuning (out of project scope), auto-classification of images into quality tiers (Loupe sorts, it does not threshold).

---

## Methodology

Research was conducted across three parallel tracks:

1. **Aggregation theory** — Academic MCDA literature, social choice theory (Arrow's theorem), power mean family, compensatory vs. non-compensatory taxonomy, IQA system survey
2. **Practical scoring systems** — How real systems combine multi-dimensional scores: IQA models (NIMA, MUSIQ, LAION), recommendation engines (Netflix, YouTube), review aggregators (Metacritic, Rotten Tomatoes), product scoring rubrics (wine, coffee), sports judging (figure skating, gymnastics, diving), search engine ranking (Elasticsearch/Lucene)
3. **Normalization and edge cases** — Score distribution handling, batch composition stability, missing dimensions, dimension correlation, interpretability requirements

Sources include academic papers, official documentation, technical blogs, and open-source repositories. Training knowledge was used for foundational MCDA literature where web verification was not possible — these claims are marked with `[^T]`.

---

## Key Finding: IQA Systems Do Not Decompose and Reaggregate

A significant finding is that major image quality assessment systems — NIMA[^1], MUSIQ[^2], BRISQUE[^3], LAION aesthetic predictor[^4], and anime-community aesthetic models (WaifuDiffusion aesthetic, cafe_aesthetic) — all predict a **single scalar score end-to-end**. None decompose quality into named dimensions and then aggregate. Loupe's multi-dimensional approach is closer to the MCDA tradition than the IQA tradition, which is an advantage for interpretability but means the aggregation problem is genuinely Loupe's to solve — there is no established IQA precedent to follow.[^T]

The notable exception is **VMAF** (Netflix's video quality metric), which combines VIF, DLM, and motion sub-scores using a trained SVM/random forest.[^5] This is a "learned combination" approach that requires human subjective quality ratings as training data.

---

## Survey of Aggregation Methods

### 1. Weighted Arithmetic Mean (WAM)

**Formula:** `S = Sigma(w_i * s_i)` where `Sigma(w_i) = 1`

The baseline approach. Each dimension contributes proportionally to its weight. Fully **compensatory** — excellence in one dimension perfectly offsets weakness in another.

| Property | Assessment |
| --- | --- |
| Monotonic | Yes — improving any dimension always improves the aggregate |
| Batch-independent | Yes — each image's score depends only on its own dimension scores |
| Decomposable | Yes — each dimension's contribution is `w_i * s_i`, trivially inspectable |
| Interpretable | Excellent — users intuitively understand weighted averages |
| Compensation | Full — a score profile of [0.9, 0.9, 0.1] averages to 0.63 (equal weights, 3 dims) |

**Strengths:** Simple, transparent, user-configurable via weights, stable across batches.
**Weaknesses:** An image mediocre everywhere can outscore an image exceptional on two dimensions but weak on one. The full compensation property means a stunning composition with terrible color can score the same as a balanced-but-unremarkable image.[^T]

**Used by:** Wine scoring (Robert Parker 100-point scale — additive rubric with fixed dimension budgets)[^6], coffee cupping (SCA 100-point scale)[^7], Metacritic (weighted average of critic scores with secret weights)[^8].

**Known failure in analogous domains:** Wine scoring's additive rubric has been criticized for biasing toward "big" wines that score well on every dimension, penalizing subtle wines that excel on fewer dimensions.[^6] This is directly analogous to Loupe's "different images good for different reasons" concern.

### 2. Weighted Geometric Mean (WGM)

**Formula:** `S = Product(s_i ^ w_i)` where `Sigma(w_i) = 1`

**Partially compensatory** — low scores are penalized more heavily than in WAM. A score of zero in any dimension drives the aggregate to zero.

| Property | Assessment |
| --- | --- |
| Monotonic | Yes, for positive scores |
| Batch-independent | Yes |
| Decomposable | Moderate — log-space decomposition: `log(S) = Sigma(w_i * log(s_i))` |
| Interpretable | Moderate — less intuitive than WAM but still understandable |
| Compensation | Partial — [0.9, 0.9, 0.1] gives geometric mean ~0.43 vs. WAM's 0.63 |

**Key property:** By the AM-GM inequality, WGM <= WAM always, with equality only when all scores are identical. The gap between WAM and WGM measures how "balanced" a score profile is.[^T]

**Practical issue:** Requires all scores > 0. A score of exactly 0.0 zeroes out the entire aggregate. Needs a floor (e.g., scores in [0.01, 1.0]).[^T]

**Precedent:** The UN Human Development Index switched from arithmetic to geometric mean in 2010 specifically because of the reduced compensation property — poor health should not be fully offset by high income.[^9]

### 3. Power Mean (Generalized Mean)

**Formula:** `M_p(s) = (Sigma(w_i * s_i^p))^(1/p)` where `Sigma(w_i) = 1`

A parameterized family that includes WAM, WGM, and several other means as special cases:

| Parameter p | Resulting Mean | Compensation Behavior |
| --- | --- | --- |
| p -> -infinity | Minimum | None — only worst dimension matters |
| p = -1 | Weighted harmonic mean | Very low — harshly penalizes weakness |
| p -> 0 | Weighted geometric mean | Low — penalizes weakness substantially |
| p = 1 | Weighted arithmetic mean | Full — standard compensation |
| p = 2 | Weighted quadratic mean (RMS) | Above full — rewards peaks |
| p -> +infinity | Maximum | Infinite — only best dimension matters |

**The single most important property:** `M_p` is monotonically increasing in `p`.[^10] This means `min <= M_{-1} <= M_0 (geometric) <= M_1 (arithmetic) <= M_2 <= max`. The parameter `p` is a single knob that continuously controls the compensation/penalty trade-off.

| Property | Assessment |
| --- | --- |
| Monotonic | Yes, for all p |
| Batch-independent | Yes |
| Decomposable | Yes — contribution of each dimension is visible in the weighted sum before the root |
| Interpretable | Good with presets; raw `p` value is less intuitive |
| Compensation | Continuously parameterized by `p` |

**Strengths:** Subsumes WAM and WGM as special cases. One knob (`p`) controls the fundamental aesthetic question: "should balanced images rank higher than specialist images?" Mathematically clean with well-understood properties.[^10]

**References:** Hardy, Littlewood, & Polya (1934) *Inequalities* — definitive mathematical reference.[^10] Grabisch et al. (2009) *Aggregation Functions* — comprehensive treatment in the aggregation/MCDA context.[^11]

### 4. Rank-Based Aggregation (Borda Count)

**Approach:** Each image gets a rank per dimension. Final score = sum or average of ranks.

| Property | Assessment |
| --- | --- |
| Monotonic | Yes, within a fixed batch |
| **Batch-independent** | **No** — adding/removing images changes all ranks |
| Decomposable | Moderate |
| Interpretable | Good — "ranked 3rd in composition, 7th in color" |
| Compensation | Partial — compresses magnitude differences |

**Critical drawback for Loupe:** Batch-dependent. Ranks are relative to the set of images being analyzed. Adding 50 new images to an existing set of 200 can change the rankings of the original 200. This conflicts with Loupe's sidecar model where each image receives a persistent, stable score.[^T]

**Subject to Arrow's impossibility theorem.** No ordinal (rank-based) aggregation system can simultaneously satisfy unrestricted domain, Pareto efficiency, independence of irrelevant alternatives, and non-dictatorship.[^12] Cardinal methods (score-based, like WAM or power mean) escape this impossibility.[^13]

**Verdict:** Useful as a secondary view but unsuitable as the primary aggregation method.

### 5. Percentile Normalization + Combination

**Approach:** Convert each dimension's raw scores to percentiles within the batch, then combine.

Shares rank-based methods' **batch-dependency problem** — percentiles are relative to the batch composition. Also destroys magnitude information in a different way than pure ranks: a dimension where all images score similarly gets as much percentile spread as one with genuine variance.[^T]

**Verdict:** Same fundamental incompatibility with Loupe's incremental workflow as rank-based methods. The scale-mismatch problem it solves should be addressed at the analyzer level (calibrated outputs) rather than at the aggregation level.

### 6. Max-of-Top-K / OWA Operators

**Approach:** Score = average of the top K dimension scores, or more generally, assign weights to the *ordered* scores (best dimension gets highest weight, worst gets lowest).

**Theoretical basis:** Ordered Weighted Averaging (OWA) operators.[^14] OWA assigns weights to score positions (1st-best, 2nd-best, etc.) rather than to named dimensions. Weighting the top positions heavily produces a "best-of" effect.

| Property | Assessment |
| --- | --- |
| Monotonic | **Not always** — improving a low-scoring dimension may not improve the aggregate if it doesn't enter the top K |
| Batch-independent | Yes |
| Interpretable | Moderate — "best 3 of 5 dimensions" is clear, but the user can't easily predict which dimensions matter for a given image |

**Relevant use case:** A user looking for reference frames for a specific aesthetic property wants "best composition, period" (K=1). A user looking for wallpapers wants balance across dimensions (high K or a different method entirely). This suggests OWA could serve as an alternative aggregation mode rather than the default.

**Verdict:** Viable as a secondary option (e.g., `--preset specialist`), not as the primary method.

### 7. L_p Distance from Ideal

**Approach:** Measure weighted L_p distance from each image's score vector to the ideal point (1.0, 1.0, ..., 1.0). Score = 1 - distance (or similar transformation).

**Formula:** `d_p = (Sigma(w_i * |1 - s_i|^p))^(1/p)`, then `S = 1 - d_p`

Mathematically related to the power mean family — both are parameterized families spanning the same compensation spectrum, but in opposite directions (`p=2` in L_p penalizes large deviations, equivalent to lower `p` in the power mean).[^T]

**Verdict:** Functionally equivalent to power mean with a different parameterization. The "distance from ideal" framing may be less intuitive than "average score." No practical advantage over power mean for Loupe's use case.

### 8. TOPSIS

**Approach:** Rank by relative closeness to the ideal solution (best in each dimension across the batch) and distance from the anti-ideal (worst in each dimension).[^15]

**Batch-dependent** — the ideal and anti-ideal points are determined from the batch. Adding images can shift these reference points and change all rankings.

**Verdict:** Well-established in MCDA for one-shot decisions ("choose the best from this set") but incompatible with Loupe's incremental scoring model. Rejected.

### 9. Learned Combination

**Approach:** Train a small model to predict aggregate score from per-dimension scores, using user preference data as supervision.

**Requires training data** that Loupe doesn't have (user keep/discard labels). Cold start problem. With only 5 input dimensions, a linear model suffices — which reduces to learning WAM weights from data.[^T]

**Practical path:** Start with configurable WAM or power mean. In a future version, add a `loupe train` command that fits a linear model from user keep/discard decisions, effectively learning personalized weights. This is explicitly deferred — the aggregation method should work well without training data.

**Verdict:** Deferred to future scope. The foundation (weighted combination with configurable weights) supports later addition of learned weights.

### Comparison Matrix

| Method | Batch-Independent | Monotonic | Decomposable | Compensatory | Configurable | Implementation Complexity |
| --- | --- | --- | --- | --- | --- | --- |
| WAM (p=1) | Yes | Yes | Full | Full | Weights | Trivial |
| WGM (p->0) | Yes | Yes | Moderate | Partial | Weights | Low |
| **Power mean** | **Yes** | **Yes** | **Full** | **Tunable (p)** | **Weights + p** | **Low** |
| Borda / rank | No | Yes* | Moderate | Partial | Limited | Low |
| Percentile | No | Yes* | Low | Depends | Limited | Low |
| OWA / top-K | Yes | Not always | Moderate | Tunable | K parameter | Low |
| L_p distance | Yes | Yes | Full | Tunable (p) | Weights + p | Low |
| TOPSIS | No | Yes | Moderate | Partial | Weights | Moderate |
| Learned | Yes | Not guaranteed | No | Implicit | Trained | High |

*Within a fixed batch only.

---

## The "Good for Different Reasons" Problem

This is Loupe's central aggregation challenge: a moody close-up with exceptional color harmony and subject focus but unconventional composition, and a sweeping landscape with strong composition but moderate color, both deserve high placement — but for different dimensional strengths.

### How Real Systems Address This

**Elasticsearch's `best_fields` mode** uses the maximum score across fields, with an optional `tie_breaker` parameter to blend in non-best-field scores.[^16] This is functionally `score = max(dimensions) + tie_breaker * sum(other dimensions)`.

**Sports judging** separates fundamentally different kinds of quality. Figure skating's ISU system has distinct Technical and Program Component scores that are added but independently visible.[^17] Diving uses multiplicative combination (difficulty x execution) — high difficulty with poor execution scores poorly.[^T]

**Profile/archetype matching** — Some recommendation systems define archetypes (clusters of dimensional profiles) and score each item by how well it matches its best-fitting archetype.[^T] A quiet atmospheric image would be evaluated against an "atmospheric" archetype where detail density weight is low, while an action key frame is evaluated against a "dynamic" archetype where detail density matters more.

**Pareto frontier** — An image is Pareto-optimal if no other image is better on *every* dimension simultaneously.[^18] Images on the frontier are "incomparable" — they represent different trade-offs. In practice, pure Pareto ranking produces too many ties when dimensions >= 5 (most images end up on or near the frontier). Works better as a filter to identify clearly dominated images than as a complete ranking.[^T]

### Problem Assessment for Loupe

The power mean with `p=1` (arithmetic mean) already partially addresses this problem — an image exceptional on two dimensions and average on three others will score above an image that's merely average everywhere, because the exceptional scores pull the average up. Lowering `p` (toward geometric mean) would penalize the image's weaker dimensions more, which is counterproductive for the "specialist" case.

The most practical enhancement is a **peak bonus**: blend the weighted mean with the maximum dimension score. This ensures images with at least one exceptional dimension get a boost:

```plaintext
score = (1 - alpha) * weighted_mean(scores) + alpha * max(scores)
```

Where `alpha` controls how much peak performance matters (0 = pure average, 1 = best dimension only). This is simple, intuitive, and directly addresses the concern. However, it adds a second configuration parameter beyond weights, increasing complexity.

**Recommendation:** Start with the power mean at `p=1` (WAM) as the default. The peak bonus is a candidate enhancement if empirical testing shows that specialist images consistently sort too low. Do not implement archetype matching or Pareto filtering in v1 — they add substantial complexity for uncertain benefit.

---

## Score Normalization: The Batch Composition Problem

### The Problem

If per-dimension scores have different distributions in practice (e.g., composition clusters around [0.6, 0.8] while color spreads across [0.2, 0.9]), the wider-spread dimension dominates the aggregate even at equal weights. This is the **commensurability problem** in MCDA.[^T]

Post-hoc normalization methods (min-max, z-score, percentile, quantile) all introduce **batch dependency**: adding or removing images changes normalization parameters, which changes rankings of existing images. This violates **independence of irrelevant alternatives** — a desirable property for Loupe's incremental workflow where the same image analyzed in two different batches should get the same score.[^T]

### The Solution: Calibrated Analyzers

Design each analyzer to produce scores on a meaningful absolute scale. Define what 0.0 and 1.0 mean for each dimension concretely — anchored to reference points, not relative to a batch.

This is the approach used by NIMA[^1] and CLIP-based aesthetic predictors[^4] — they predict absolute quality scores, not relative rankings. Loupe should follow the same principle.

| Normalization Approach | Batch-Independent | Preserves Magnitude | Requires Calibration |
| --- | --- | --- | --- |
| None (raw scores) | Yes | Yes | Yes — analyzers must be well-calibrated |
| Min-max | No | Yes | No |
| Z-score | No | Partially | No |
| Percentile | No | No | No |
| **Global baseline** | **Yes** | **Yes** | **Yes — one-time calibration set** |

**Fallback:** If some analyzers prove difficult to calibrate to a meaningful absolute scale, use **global baseline normalization** — establish distribution parameters (mean, std) from a representative calibration set of ~50-100 anime screenshots, then normalize all future scores against those fixed parameters. This preserves batch independence because the baseline is fixed.[^T]

**Recommendation:** Calibrated raw scores (no normalization at aggregation time). Each analyzer's development includes calibrating its output to use the [0.0, 1.0] range meaningfully. Global baseline normalization as the fallback for dimensions where absolute calibration is impractical.

---

## Handling Missing Dimensions

Three scenarios: analyzer disabled in config, analyzer fails for a specific image, new analyzer added in a future version.

### Recommended Approach: Proportional Aggregation

```plaintext
aggregate = Sigma(w_i * s_i) / Sigma(w_i)    for all dimensions i where s_i is available
```

This naturally adjusts when dimensions are missing — the denominator shrinks proportionally. It is equivalent to a weighted average over available dimensions only.[^T]

| Strategy | Pros | Cons |
| --- | --- | --- |
| **Proportional (recommended)** | Simple, principled, no invented data | Image missing a weak dimension gets unearned boost |
| Imputation (fill defaults) | All images have same dimension count | Imputed values are fiction, adds noise |
| Separate aggregates by dimension set | Cleanest comparison | Fragments the ranking |

**Practical mitigation:** In Loupe's workflow, all images in a batch are typically analyzed with the same configuration, so missing dimensions are the exception (failures), not the rule. For the version-upgrade case (new analyzer added), re-analysis with `--force` is the right answer.

**Minimum dimension threshold:** Consider requiring at least N-1 of N dimensions before computing an aggregate. If only 1 of 5 analyzers succeeded, the "aggregate" is noise. Flag low-dimension aggregates in the output rather than silently presenting them.

---

## Dimension Correlation

If two dimensions are correlated (e.g., composition and subject clarity both depend on focal point placement), a weighted sum double-counts the shared underlying factor.

### Detection

After analyzers are implemented and run on real data, compute pairwise Spearman correlations between dimension scores. Correlations above r = 0.7 suggest meaningful redundancy.[^T]

### Correlation Assessment for Loupe

With 5 planned dimensions covering genuinely different properties — composition geometry (spatial analysis), color palette (color-space analysis), detail texture (frequency analysis), style embedding (CLIP), subject saliency (attention modeling) — the *methods* are sufficiently different that extreme redundancy is unlikely. Moderate correlation (r = 0.3-0.5) is expected and acceptable.[^T]

**Recommendation:** Do not pre-engineer correlation handling. Monitor empirically once analyzers exist. If strong correlations emerge (r > 0.7), options in order of preference:

1. **Merge the correlated dimensions** into one if they're measuring the same underlying property
2. **Adjust weights** — reduce combined weight allocation for correlated dimensions
3. **Redundancy-adjusted weighting** — scale each weight by `(1 - avg_correlation_with_others)`[^19]

Avoid PCA rotation or Choquet integrals[^19] — the interpretability cost is too high for Loupe's "debuggable rankings" goal.

---

## Dimension Dilution

Adding a 6th dimension (e.g., lighting quality) means existing dimensions lose relative influence if weights are renormalized to sum to 1.0.

### Recommended Mitigation: Fixed Weight Budget

Weights always sum to 1.0. Adding a dimension means consciously reallocating from existing dimensions. This makes trade-offs explicit: "lighting gets 10%, taken from detail (15% -> 10%) and style (15% -> 10%)."[^T]

If the dimension count grows beyond 6-7, consider **hierarchical weighting** (AHP-style)[^20]: group dimensions into categories, weight categories first, then weight dimensions within each category. Adding a dimension within a category only dilutes other dimensions in that category, not the whole set.

Example hierarchy (illustrative, not a recommendation for specific values):

```plaintext
Structural (40%):
  Composition (25%)
  Subject clarity (15%)
Visual quality (35%):
  Color harmony (15%)
  Detail density (10%)
  Lighting (10%)
Stylistic (25%):
  Style coherence (25%)
```

**Recommendation:** Fixed weight budget for v1 (5 dimensions, flat weighting is manageable). Revisit with hierarchical weighting if dimension count grows.

---

## Interpretability and Debuggability

For users to trust the ranking, they must be able to understand why image A ranked above image B. This imposes requirements on the aggregation method.[^21]

### Required Properties

1. **Monotonicity** — Improving on any single dimension never decreases the aggregate
2. **Decomposability** — The contribution of each dimension to the aggregate can be isolated and displayed
3. **Stability** — Small changes in input produce proportionally small changes in output
4. **Transparency of weights** — Users can see what's being prioritized

WAM satisfies all four. WGM satisfies all four (with log-space decomposition). Power mean satisfies all four. Neural/learned aggregation violates 2 and 4.[^T]

### Contribution Breakdown

For WAM, the contribution of each dimension is trivially decomposable:

```plaintext
Image: frame_0042.png
Aggregate: 0.74
  Composition:  0.82 x 0.30 = 0.246  (33.2%)
  Color:        0.71 x 0.25 = 0.178  (24.0%)
  Detail:       0.65 x 0.20 = 0.130  (17.6%)
  Style:        0.78 x 0.15 = 0.117  (15.8%)
  Subject:      0.69 x 0.10 = 0.069  ( 9.3%)
```

**Pairwise comparison** ("why did A rank above B?") becomes a dimension-by-dimension delta showing where each image wins or loses.

**Recommendation:** Include per-dimension contribution data in every `LoupeResult`. This is cheap to compute and critical for user trust. The CLI's `loupe rank` output should show the top contributor(s) alongside the aggregate.

---

## Recommendation

### Primary Aggregation: Weighted Arithmetic Mean

After surveying nine aggregation families, the weighted arithmetic mean (WAM) is recommended as Loupe's primary aggregation method.

**Formula:**

```plaintext
aggregate = Sigma(w_i * s_i) / Sigma(w_i)    for available dimensions i
```

**Rationale:**

1. **Batch-independent.** Each image's score depends only on its own dimension scores and the configured weights. Adding images to a batch never changes existing scores. This is essential for Loupe's sidecar model and incremental workflow.
2. **Fully decomposable.** Per-dimension contributions are trivially computed, enabling the contribution breakdowns and pairwise comparisons that make rankings debuggable.
3. **Monotonic.** Improving any dimension always improves the aggregate. No counterintuitive ranking behavior.
4. **Simple to configure.** Weights map directly to "I care more about X than Y." Users do not need to understand mathematical properties to adjust behavior.
5. **Handles missing dimensions gracefully.** Proportional aggregation (dividing by sum of available weights) is a natural fallback.
6. **No batch-dependent normalization required.** With calibrated analyzers producing meaningful absolute scores, raw scores combine directly.

**Why not power mean?** The power mean family is mathematically elegant and subsumes WAM as a special case. However, it introduces a second configuration parameter (`p`) whose practical meaning — "how much do I penalize weak dimensions?" — is harder to reason about than weights alone. The added complexity does not justify the benefit for v1, especially because:

- The `p` parameter interacts with weights in non-obvious ways (the effect of `p` depends on the score distribution, which depends on analyzers not yet built)
- Preset names ("balanced," "specialist," "consistent") help but still require users to develop intuition about which preset fits their needs
- WAM with well-chosen weights already produces reasonable rankings for both "balanced" and "specialist" images

**Power mean is the recommended upgrade path** if WAM proves insufficient in practice. The implementation should be structured so that switching from WAM to power mean is a one-line change in the scoring function.

### Configuration Design

**Default weights** — Equal weights (1/N per dimension) as the starting point. Empirical tuning against real batches should produce better defaults, but equal weights are the principled starting point when no preference data exists.

**User configuration** — Per-dimension weights in YAML config, expressed as relative importance values (not required to sum to 1.0 — the system normalizes internally). This lets users write intuitive values:

```yaml
scoring:
  weights:
    composition: 3    # high importance
    color: 2          # moderate
    detail: 2         # moderate
    style: 1          # lower
    subject: 2        # moderate
```

Internally normalized: `w_i = raw_w_i / Sigma(raw_w_i)`

**Presets** — Ship 2-3 named presets as CLI flags for quick switching without config editing:

| Preset | Weights (relative) | Use Case |
| --- | --- | --- |
| `balanced` (default) | Equal weights | General-purpose sorting |
| `composition` | Composition 3x, others 1x | Wallpaper/framing-focused review |
| `visual` | Color + Detail 2x, others 1x | Visual quality-focused review |

Presets are overridden by explicit weight configuration. The preset list should grow based on user feedback, not speculation.

**CLI integration:**

```plaintext
loupe rank <path>                         # uses default weights
loupe rank <path> --preset composition    # uses composition preset
loupe rank <path> --config my-weights.yaml  # uses custom weights
```

### What to Include in LoupeResult

The aggregate score in the sidecar JSON should include:

```json
{
  "aggregate_score": 0.74,
  "scoring_method": "weighted_mean",
  "scoring_version": "1.0",
  "weights_used": {
    "composition": 0.30,
    "color": 0.25,
    "detail": 0.20,
    "style": 0.15,
    "subject": 0.10
  },
  "contributions": {
    "composition": 0.246,
    "color": 0.178,
    "detail": 0.130,
    "style": 0.117,
    "subject": 0.069
  }
}
```

Including `scoring_method` and `scoring_version` allows future migration if the aggregation method changes. Including `weights_used` makes each score self-documenting — the user can see exactly what weights produced this ranking. Including `contributions` enables pairwise comparison and "why did this rank here?" explanations without recomputation.

### What to Defer

| Feature | Rationale for Deferral |
| --- | --- |
| Power mean (`p` parameter) | Upgrade path if WAM proves insufficient; requires empirical data on score distributions to tune `p` meaningfully |
| Peak bonus (`alpha * max(scores)`) | Addresses "specialist" images; wait for evidence that WAM under-ranks them before adding complexity |
| Hierarchical weighting (AHP) | Unnecessary at 5 dimensions; revisit if dimension count exceeds 6-7 |
| Learned weights (`loupe train`) | Requires preference data; natural extension once users have keep/discard history |
| Correlation-adjusted weighting | Monitor empirically first; unlikely to be needed with 5 methodologically distinct analyzers |
| Archetype/profile matching | Interesting but substantial complexity; not justified without evidence of need |
| Pareto filtering | Too many ties at 5 dimensions to be useful as primary ranking |

---

## Limitations and Open Questions

**Calibration difficulty.** The recommendation to use calibrated raw scores (no batch-dependent normalization) pushes complexity into analyzer development. Each analyzer must map its outputs to a [0.0, 1.0] range where scores are meaningful in absolute terms. For classical CV analyzers (composition geometry, color harmony math), this requires defining reference points and fitting score curves. For CLIP-based analyzers, raw embedding distances or cosine similarities must be mapped to calibrated scores — the mapping is model-specific and may need adjustment per model version. This is the hardest part of the recommended approach and should be addressed during each analyzer's development (RQ2 scope).

**Weight defaults are speculative.** Equal weights are principled but almost certainly suboptimal. Good defaults require running analyzers on real batches and evaluating whether the resulting rankings match human judgment. A small calibration study (50-100 images, human-ranked) would anchor defaults, but this is implementation-phase work.

**The "specialist vs. balanced" question remains open empirically.** WAM with equal weights favors balanced images. Whether this matches the Loupe user's actual preferences for wallpaper selection — or whether specialist images are systematically under-ranked — can only be determined through use. The power mean upgrade path exists for this reason.

**Single aggregate may not be sufficient long-term.** The research on profile/archetype matching and Pareto frontiers suggests that a single sortable number, regardless of aggregation method, may ultimately be too reductive for a tool whose value proposition is multi-dimensional analysis. Future versions might complement the aggregate with profile tags ("composition-driven," "color-driven") or multi-column sort support. This is beyond v1 scope but worth noting as a design direction.

---

## Source Coverage

| Source Type | Count | Coverage |
| --- | --- | --- |
| Academic papers (MCDA, IQA, social choice) | 12 | Strong — foundational MCDA and IQA literature well-covered |
| Official documentation / technical blogs | 5 | Moderate — practical systems documented via public sources |
| Open-source repositories | 3 | Moderate — key IQA implementations examined |
| Domain-specific rubrics (wine, coffee, sports) | 4 | Adequate — analogous scoring systems surveyed |
| Anime-specific aesthetic systems | 2 | Thin — no multi-dimensional anime scoring systems found (this appears to be a genuine gap) |

---

## References

[^1]: Talebi, H. & Milanfar, P. (2018). "NIMA: Neural Image Assessment." IEEE Transactions on Image Processing, 27(8). <https://arxiv.org/abs/1709.05424>

[^2]: Ke, J. et al. (2021). "MUSIQ: Multi-scale Image Quality Transformer." <https://arxiv.org/abs/2108.05997>

[^3]: Mittal, A. et al. (2012). "No-Reference Image Quality Assessment in the Spatial Domain." IEEE Transactions on Image Processing.

[^4]: Schuhmann, C. et al. "Improved Aesthetic Predictor." <https://github.com/christophschuhmann/improved-aesthetic-predictor> — LAION aesthetic scoring blog: <https://laion.ai/blog/laion-aesthetics/>

[^5]: Li, Z. et al. (2018). "VMAF: The Journey Continues." Netflix Technology Blog. Repository: <https://github.com/Netflix/vmaf>

[^6]: Robert Parker Wine Advocate rating system. <https://www.robertparker.com/resources/robert-parkers-rating-system> — Academic analysis of wine scoring biases: AAWE Working Paper No. 23, <https://www.wine-economics.org/dt_catalog/aawe-wp23/>

[^7]: Specialty Coffee Association (SCA) Cupping Protocols. <https://sca.coffee/research/protocols-best-practices>

[^8]: Metacritic scoring methodology. <https://www.metacritic.com/about-metascores>

[^9]: UNDP (2010). Human Development Report Technical Notes — switch from arithmetic to geometric mean in HDI calculation.

[^10]: Hardy, G.H., Littlewood, J.E., & Polya, G. (1934). *Inequalities.* Cambridge University Press.

[^11]: Grabisch, M., Marichal, J.-L., Mesiar, R., & Pap, E. (2009). *Aggregation Functions.* Cambridge University Press.

[^12]: Arrow, K.J. (1951). *Social Choice and Individual Values.* Yale University Press.

[^13]: Blackorby, C., Donaldson, D., & Weymark, J.A. (1984). "Social choice with interpersonal utility comparisons." International Economic Review, 25(2).

[^14]: Yager, R.R. (1988). "On ordered weighted averaging aggregation operators in multicriteria decisionmaking." IEEE Transactions on Systems, Man, and Cybernetics, 18(1). <https://doi.org/10.1109/21.87068>

[^15]: Hwang, C.L. & Yoon, K. (1981). *Multiple Attribute Decision Making: Methods and Applications.* Springer. Survey: Behzadian, M. et al. (2012). "A state-of-the-art survey of TOPSIS applications." Expert Systems with Applications, 39(17).

[^16]: Elasticsearch multi-match query documentation. <https://www.elastic.co/guide/en/elasticsearch/reference/current/query-dsl-multi-match-query.html>

[^17]: ISU Judging System for figure skating. <https://www.isu.org/figure-skating/rules/fsk-judging-system>

[^18]: Deb, K. et al. (2002). "A fast and elitist multiobjective genetic algorithm: NSGA-II." IEEE Transactions on Evolutionary Computation. <https://ieeexplore.ieee.org/document/996017>

[^19]: Marichal, J.-L. (2000). "An axiomatic approach of the discrete Choquet integral as a tool to aggregate interacting criteria." IEEE Transactions on Fuzzy Systems. <https://doi.org/10.1109/91.868950> — Grabisch, M. (1996). "The application of fuzzy integrals in multicriteria decision making." EJOR. <https://doi.org/10.1016/0377-2217(95)00300-2>

[^20]: Saaty, T.L. (1980). *The Analytic Hierarchy Process.* McGraw-Hill. Modern overview: Vaidya, O.S. & Kumar, S. (2006). <https://doi.org/10.1016/j.ejor.2007.01.016>

[^21]: Liu, T.Y. (2009). "Learning to Rank for Information Retrieval." Foundations and Trends in Information Retrieval. <https://doi.org/10.1561/1500000016>

[^T]: Based on Claude's training knowledge — not verified against current sources in this session.
