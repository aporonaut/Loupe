# RQ3 — Anime-Tuned Models Landscape

**Date:** 2026-04-03
**Status:** Complete

**Question:** What pretrained models exist that are trained on or tuned for anime/illustration content, and which are viable for Loupe's use cases?

---

## Executive Summary

The anime-tuned model ecosystem is fragmented but functional. Coverage exists across all four categories Loupe needs — CLIP embeddings, aesthetic scoring, tagging, and auxiliary vision — though documentation quality is poor across the board.

**Key findings:**

1. **No dedicated anime CLIP fine-tune dominates.** The best option is DanbooruCLIP [^1] (ViT-L/14, Danbooru+Pixiv, 7.6K downloads/month), but general CLIP with anime-trained heads is the more practical pattern.
2. **Anime aesthetic scoring is immature.** No model has published benchmarks on anime quality ranking. The best candidates are deepghs/anime_aesthetic [^6] (7-tier, AUC 0.82, documented) and kawaimasa/kawai-aesthetic-scorer [^7] (60K manually labeled images, best-documented). The domain gap from photo-trained scorers is significant — the SD/NovelAI community independently built anime-specific alternatives. [^T]
3. **WD-Tagger v3 is the clear winner for tagging** [^15] — 79K downloads/month, Apache-2.0, 10,861 tags including composition, lighting, and color descriptors useful for Loupe's dimensions.
4. **Auxiliary models are mixed.** Anime-specific segmentation [^20] and face detection [^24] have good solutions. Depth estimation struggles with anime's flat shading. [^T]
5. **All recommended models fit simultaneously in 8GB VRAM** (~3.8–5.6 GB total depending on combination).

**Recommended models for RQ5 evaluation:**

| Priority | Model | Category | Rationale |
| --- | --- | --- | --- |
| Must | WD-Tagger v3 (SwinV2 or EVA02-Large) [^15] | Tagging | De facto standard, Apache-2.0, feeds multiple Loupe dimensions |
| Must | deepghs/anime_aesthetic (SwinV2) [^6] | Aesthetic | Only anime quality scorer with published benchmarks (AUC 0.82) |
| Must | kawaimasa/kawai-aesthetic-scorer [^7] | Aesthetic | Best-documented, 60K labeled images, Apache-2.0 |
| Must | skytnt/anime-segmentation [^20] | Auxiliary | Purpose-built anime character extraction, Apache-2.0 |
| Must | deepghs face/head/person detection [^24] | Auxiliary | Comprehensive anime detection suite |
| Should | OpenCLIP ViT-L/14 [^4] | CLIP | Foundation backbone, reusable across dimensions |
| Should | Improved Aesthetic Predictor [^10] | Aesthetic | Baseline comparison, nearly free if CLIP loaded |
| Should | Waifu Scorer v3 [^8] | Aesthetic | CLIP+MLP trained on anime, promising but undocumented |
| Could | Depth Anything V2 (Small) [^21] | Auxiliary | Coarse depth signal, known anime limitations |
| Could | Camie-Tagger v2 [^18] | Tagging | Better rare-tag performance, but GPL-3.0 concern |

---

## Methodology

### Sources Consulted

- **HuggingFace Hub** — API searches with 20+ query terms across all categories, individual model cards for all significant models, organization pages (SmilingWolf, deepghs, cafeai, Eugeoter, kawaimasa, Blackroot, skytnt, Camais03, NovelAI, animetimm, pixai-labs)
- **GitHub** — Project READMEs and code for: improved-aesthetic-predictor [^10], SkyTNT/anime-segmentation [^20], wdv3-timm [^30], clipbooru [^3], DeepDanbooru [^19], Depth-Anything-V2 [^21], SAM2 [^22], U2-Net [^23], YOLO anime face detectors [^25] [^26] [^27]
- **Web search** — Community reports on anime-domain performance for general-purpose models (SAM, MiDaS, Depth Anything V2)
- **Model config files** — config.json and selected_tags.csv from WD-Tagger repos for architecture and vocabulary details

### Coverage Assessment

| Source Type | Coverage |
| --- | --- |
| HuggingFace Hub (API + model cards) | Strong — primary discovery source |
| GitHub repositories | Moderate — key repos accessed |
| Academic papers | Referenced via model cards; no direct paper fetches |
| Community forums (Reddit, Discord, CivitAI) | Not accessed — gap for empirical performance reports |

### Epistemic Markers

Claims from training knowledge that could not be verified against current sources are marked `[^T]`.

---

## Category 1: CLIP Models for Anime

### CLIP Landscape Overview

**No major standalone anime CLIP fine-tune exists as a general-purpose embedding model.** The community's approach has been to use general CLIP (which includes anime in its web-crawled training data) and train task-specific heads on top. The discriminative understanding lives in the heads, not a fine-tuned encoder.

This means Loupe should use a general CLIP backbone and build or leverage anime-trained heads rather than seeking a single anime CLIP model.

### Model Catalog

#### OysterQAQ/DanbooruCLIP

| Attribute | Value |
| --- | --- |
| **Repository** | `OysterQAQ/DanbooruCLIP` [^1] |
| **Architecture** | CLIP ViT-L/14 |
| **Training data** | Danbooru2021 + Pixiv (added July 2023). Captions from tag hierarchy: character > series > general tags. Long tag lists (>20) split into two captions. |
| **License** | Not specified |
| **Model size** | ~428M params [^T] |
| **VRAM (FP16)** | ~1.5 GB [^T] |
| **Downloads** | 7,592/month |
| **Last updated** | 2023-07-17 (inactive) |

**Strengths:** Largest anime CLIP download count. Danbooru2021 is massive and well-tagged. Thoughtful caption construction from hierarchical tags.

**Weaknesses:** No published benchmarks. License unspecified. Tag-concatenation training (not natural language) may limit zero-shot flexibility. Inactive since 2023.

**Intra-anime quality discrimination:** Uncertain — trained to associate images with tags, not rank quality. Embeddings capture style/content but likely don't encode quality gradients without a scoring head.

#### dudcjs2779/anime-style-tag-clip

| Attribute | Value |
| --- | --- |
| **Repository** | `dudcjs2779/anime-style-tag-clip` [^2] |
| **Architecture** | EVA02-Base (Patch16, 224px) via OpenCLIP |
| **Training data** | 29,187 anime image-tag pairs |
| **License** | MIT |
| **VRAM (FP16)** | ~0.5 GB [^T] |
| **Downloads** | 6/month |
| **Validation** | R@1: 0.877, R@5: 0.968 |

Small training set limits generalization (acknowledged by author). [^2] Minimal adoption. MIT license is favorable.

#### v2ray/clipbooru

| Attribute | Value |
| --- | --- |
| **Repository** | `v2ray/clipbooru` [^3] |
| **Architecture** | CLIP + classification head for Danbooru tag prediction |
| **License** | MIT |
| **Model size** | 0.3B params, BF16 |
| **Downloads** | 29/month |
| **Last updated** | 2025-04-08 (active) |

Recently updated. Dual-purpose (embeddings + tags) is interesting for Loupe. Very low adoption.

#### General-Purpose CLIP Models (Reference)

| Model | Params | VRAM (FP16) | Notes |
| --- | --- | --- | --- |
| ViT-L/14 (OpenAI/OpenCLIP) [^4] | ~428M | ~1.5 GB | Strong baseline, widely used |
| ViT-H/14 (LAION-2B) [^4] | ~986M | ~3.9 GB | Best quality, tight on 8GB VRAM |
| ViT-bigG/14 (LAION) [^4] | ~2.5B | ~5 GB | Best embeddings, very tight on 8GB |
| EVA02-L (merged2b) [^4] | ~428M | ~1.5 GB | Often outperforms ViT-L [^T] |

General CLIP distinguishes anime from photos well but struggles with intra-anime quality discrimination. [^T]

### Anime-Specific Embeddings via WD-Tagger

The deepghs/wd14_tagger_with_embeddings project [^5] provides modified ONNX versions of WD-Tagger models that output both tags and embeddings:

| Base Model | Tags | Embedding Dim |
| --- | --- | --- |
| WD v3 EVA02-Large | 10,861 | 1024 |
| WD v3 ViT-Large | 10,861 | 1024 |
| WD v3 SwinV2 | 10,861 | 1024 |
| WD v3 ViT | 10,861 | 768 |
| WD v3 ConvNeXt | 10,861 | 1024 |

These anime-native embeddings may be more useful than general CLIP for anime similarity/clustering tasks. Worth investigating in RQ5.

### Category 1 Recommendations for RQ5

- **Must evaluate:** OpenCLIP ViT-L/14 [^4] (baseline backbone), deepghs/wd14_tagger_with_embeddings [^5] (anime-native)
- **Worth evaluating:** DanbooruCLIP [^1] (if license can be clarified)
- **Deprioritize:** anime-style-tag-clip [^2] (too small), clipbooru [^3] (too new/untested)

---

## Category 2: Aesthetic Scoring Models

### Aesthetic Scoring Landscape Overview

The aesthetic scoring landscape splits into photo-trained models and anime-specific models. **The domain gap is significant and well-established** — photo-trained models learn photographic quality signals (exposure, bokeh, natural lighting) that are irrelevant or anti-correlated with anime quality. The entire SD/NovelAI ecosystem independently built anime-specific alternatives, which is strong evidence general models were found inadequate. [^T]

The domain gap lives primarily in the scoring head, not the feature extractor — which is why the CLIP+MLP architecture (general CLIP encoder → anime-trained MLP head) is the dominant pattern.

### Tier 1: Anime-Specific (High Relevance)

#### deepghs/anime_aesthetic

| Attribute | Value |
| --- | --- |
| **Repository** | `deepghs/anime_aesthetic` [^6] |
| **Architecture** | CAFormer-S36 and SwinV2 variants |
| **Training data** | Danbooru-sourced (not further documented) |
| **Output** | 7-class: `masterpiece`, `best`, `great`, `good`, `normal`, `low`, `worst` |
| **License** | OpenRAIL |
| **Maintenance** | Active (part of deepghs/imgutils ecosystem) |

| Variant | FLOPs | Params | Accuracy | AUC |
| --- | --- | --- | --- | --- |
| caformer_s36_v0_ls0.2 | 22.10G | 37.22M | 34.68% | 0.7725 |
| swinv2pv3_v0_448_ls0.2 | 46.20G | 65.94M | 40.32% | 0.8188 |
| swinv2pv3_v0_448_ls0.2_x | 46.20G | 65.94M | 40.88% | 0.8214 |

**Strengths:** Only anime quality scorer with published benchmarks. [^6] 7-tier output maps well to quality gradients. AUC 0.82 means reliable rank-ordering ~82% of the time. Lightweight (37–66M). ONNX format. Part of well-maintained imgutils ecosystem.

**Weaknesses:** ~40% raw accuracy (7-tier is hard). Danbooru quality tags are noisy (mix quality with popularity). ONNX-only may be less flexible.

**Loupe relevance:** Directly provides the quality-grade signal that tagger tags lack. Best-documented candidate for overall quality dimension.

#### kawaimasa/kawai-aesthetic-scorer-convnextv2

| Attribute | Value |
| --- | --- |
| **Repository** | `kawaimasa/kawai-aesthetic-scorer-convnextv2` [^7] |
| **Architecture** | ConvNeXtV2-Large (facebook/convnextv2-large-22k-384 base) |
| **Training data** | 60,000 manually labeled anime/illustration images, 5 quality tiers |
| **Output** | 5-class softmax (SS/S/A/B/C) + weighted scalar: `weights = [1.0, 0.9, 0.5, -0.5, -1.0]` |
| **Input** | 768x768 black-padded letterbox, specific normalization |
| **Training** | Progressive resizing (384→512→768px), single annotator |
| **License** | Apache-2.0 |
| **Params** | ~200M |
| **VRAM** | ~1 GB |
| **Downloads** | 42/month |

**Strengths:** Best-documented anime aesthetic model. Largest verified training set (60K). Clear preprocessing requirements. Honest about subjectivity. Apache-2.0. [^7]

**Weaknesses:** Single annotator's preferences baked in. Requires exact preprocessing match. Author explicitly warns it cannot do objective quality assessment.

#### Waifu Scorer v3 (Eugeoter)

| Attribute | Value |
| --- | --- |
| **Repository** | `Eugeoter/waifu-scorer-v3` [^8], `Eugeoter/waifu-scorer-v4-beta` [^9] |
| **Architecture** | CLIP backbone + MLP head |
| **Output** | 0–10 continuous score |
| **License** | OpenRAIL (v3), Apache-2.0 (v4-beta) |
| **VRAM (FP16)** | ~1.5 GB (with CLIP) [^T] |
| **Maintenance** | Active (v4 in development) |

**Strengths:** Purpose-built for anime. Proven CLIP+MLP architecture. v3 has 23 likes and is used in 4 Spaces. If CLIP is already loaded, the MLP head is nearly free.

**Weaknesses:** Training data undocumented. No published benchmarks. "Waifu" framing may bias toward character-focused images. v4 READMEs are empty. [^9]

**Intra-anime quality discrimination:** Most architecturally appropriate model for direct anime scoring. Empirical testing in RQ5 is essential.

#### skytnt/anime-aesthetic

| Attribute | Value |
| --- | --- |
| **Repository** | `skytnt/anime-aesthetic` [^11] |
| **Architecture** | Unknown (ONNX only, 112MB) |
| **License** | Not specified |
| **Adoption** | 100+ HuggingFace Spaces, 18 likes |
| **Maintenance** | Unknown — no model card, 3+ years since update |

Widely adopted but completely opaque. Black-box risk. [^11]

#### Blackroot/Anime-Aesthetic-Predictor-Medium

| Attribute | Value |
| --- | --- |
| **Repository** | `Blackroot/Anime-Aesthetic-Predictor-Medium` [^12] |
| **Architecture** | ConvNeXt |
| **Training data** | 20,000 labeled anime images |
| **Output** | 0–10 continuous score |
| **Training** | 12 epochs, ~6 hours on RTX 3090 Ti |
| **License** | MIT |
| **Training code** | `CoffeeVampir3/easy-aesthetic-predictor` [^13] |

MIT licensed with training code. If off-the-shelf models prove inadequate, this provides a template for training a custom Loupe-specific scorer.

### Tier 2: General-Purpose (Photo-Trained)

#### Improved Aesthetic Predictor (LAION / Schuhmann)

| Attribute | Value |
| --- | --- |
| **Repository** | `christophschuhmann/improved-aesthetic-predictor` (GitHub) [^10], `shunk031/aesthetics-predictor-v2-sac-logos-ava1-l14-linearMSE` (HF) [^14] |
| **Architecture** | MLP head on CLIP ViT-L/14 embeddings |
| **Training data** | SAC+Logos+AVA1 (~490K photos + AI images) [^T] |
| **Output** | 1–10 continuous score |
| **License** | Apache-2.0 |
| **VRAM (FP16)** | ~1.5 GB (CLIP) + negligible MLP |
| **Downloads** | 9,067/month (shunk031 v2 SAC variant) [^14] |

The foundational model. De facto standard in the SD community. CLIP backbone provides dual-use (embeddings + scoring from single model load). Multiple variants: v1, v2-linearMSE, v2-reluMSE. [^14]

**Anime domain gap:** Produces compressed, low-variance scores on anime content compared to photos. [^T] Should not be the primary quality signal but useful as one dimension in a multi-signal system, and as a baseline for comparison.

#### Other General Models (Not Recommended)

| Model | Why Not |
| --- | --- |
| NIMA (MobileNet/Inception + EMD) [^T] | Photo-only (AVA), old (2017), no anime relevance |
| CAFE Aesthetic (BEiT-base) [^35] | Binary output too coarse, AGPL-3.0, 3.5K training images |
| Aesthetic Shadow v1/v2 (ViT 1.1B) [^36] [^37] | Oversized, undocumented, CC-BY-NC-4.0 / unknown license |
| Q-Align (mPLUG-Owl2 7B+) [^T] | Exceeds 8GB VRAM budget |
| rsinema/aesthetic-scorer [^38] | Photo-trained (PARA), zero adoption, dimensions don't map to anime |
| aesthetic-anime-v2 (minizhu) [^39] | 1.1B params, CC-BY-NC-4.0, undocumented, 11 downloads/month |

### Key Analysis

**No model has published anime-quality benchmarks against a standardized test set.** The anime community lacks an equivalent of AVA. Danbooru quality tags are the closest proxy but are noisy (mix quality with popularity/recency). Empirical evaluation in RQ5 is required before committing to any model.

**Custom training is feasible and may be necessary.** The CLIP+MLP pattern makes training a custom anime aesthetic head straightforward (~hours on consumer GPU). Blackroot's training code [^13] provides a template. This is within scope as a fallback.

### Category 2 Recommendations for RQ5

- **Must evaluate:** deepghs/anime_aesthetic [^6] (SwinV2), kawaimasa/kawai-aesthetic-scorer [^7]
- **Should evaluate:** Waifu Scorer v3 [^8], Improved Aesthetic Predictor [^14] (as baseline)
- **Deprioritize:** skytnt/anime-aesthetic [^11] (opaque), Aesthetic Shadow [^36] (oversized), NIMA (photo-only)

---

## Category 3: Anime Tagging Models

### Anime Tagging Landscape Overview

The tagging ecosystem is dominated by **WD-Tagger v3** (SmilingWolf). [^15] The Danbooru tag vocabulary includes aesthetically useful tags — composition framing, color/mood descriptors, lighting, scene type — but **does not include quality-grade tags** like "masterpiece" or "best quality" (those are Stable Diffusion prompt artifacts, not Danbooru metadata).

### WD-Tagger v3 Family (SmilingWolf) — Primary Recommendation

**Source:** SmilingWolf on HuggingFace [^15]
**Code:** wdv3-timm (PyTorch) [^30], wdv3-jax (training) [^31]
**Demo:** wd-tagger Space [^32]
**License:** Apache 2.0

#### Training Data

- **Source:** Danbooru images up to ID 7,220,105
- **Cutoff:** 2024-02-28
- **Split:** IDs modulo 0000-0899 (training), 0950-0999 (validation)
- **Filtering:** Images with <10 general tags excluded; tags with <600 images excluded
- **Infrastructure:** Google TPU via TRC program, JAX-CV framework
- **Loss:** Tag frequency-based scaling for class imbalance (v2.0 of each model)

#### Tag Vocabulary

- **Total classes:** 10,861
- **Categories:** Rating (4: general/sensitive/questionable/explicit), General (~10,850+), Character (small number)
- **Selection:** Tags in 600+ images, on images with 10+ tags

#### Architecture Variants

| Variant | HuggingFace | Params (est.) | ONNX Size | Safetensors | Macro-F1 | Threshold | Downloads/mo |
| --- | --- | --- | --- | --- | --- | --- | --- |
| **EVA02-Large** | `SmilingWolf/wd-eva02-large-tagger-v3` | ~300M | 1.26 GB | 1.26 GB | 0.4772 | 0.5296 | 8,922 |
| **ViT-Large** | `SmilingWolf/wd-vit-large-tagger-v3` | ~300M | 1.26 GB | 1.26 GB | 0.4674 | 0.2606 | 4,550 |
| **SwinV2-Base** | `SmilingWolf/wd-swinv2-tagger-v3` | ~90M | 467 MB | 392 MB | 0.4541 | 0.2653 | 18,607 |
| **ViT-Base** | `SmilingWolf/wd-vit-tagger-v3` | ~85M | 379 MB | 378 MB | 0.4402 | 0.2614 | 45,928 |
| **ConvNeXt-Base** | `SmilingWolf/wd-convnext-tagger-v3` | ~90M | 395 MB | 395 MB | 0.4419 | 0.2682 | 1,207 |

**Note on F1 scores:** v3 Macro-F1 (~0.44–0.48) appears lower than v2 (~0.68–0.69) because v3 has 10,861 tags vs. v2's 9,083. More tags means more rare tags pulling down the macro average. v3 is not worse; the evaluation is over a harder tag set.

#### Input/Output

- **Input:** 448x448 RGB, normalized mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], bicubic, center crop (crop_pct=1.0)
- **Output:** Sigmoid probabilities for 10,861 tags
- **Formats:** ONNX (requires onnxruntime >= 1.17.0, flexible batch), Safetensors (timm), MessagePack (JAX)

#### Inference Approaches

**Via timm (recommended for PyTorch):**

```python
import timm
model = timm.create_model("hf-hub:SmilingWolf/wd-swinv2-tagger-v3", pretrained=True)
```

**Via ONNX:** Using onnxruntime with CPU or CUDA execution provider.

**Via HuggingFace Transformers:** `p1atdev/wd-swinv2-tagger-v3-hf` [^16] (98M params, BF16, 20.2K downloads/month) loads via `AutoModelForImageClassification`.

#### VRAM

- EVA02-Large / ViT-Large: ~1.3 GB weight + inference = ~2–3 GB. Comfortable on RTX 3070.
- SwinV2 / ViT / ConvNeXt: ~400 MB weight = ~1–1.5 GB. Very comfortable.

#### Recommendation

**SwinV2-Base** for general use (good accuracy, moderate size, second-highest downloads). **EVA02-Large** for maximum accuracy if 1.26 GB model load is acceptable.

### WD-Tagger v2 Family — Superseded

Five variants (MOAT, SwinV2, ViT, ConvNeXt, ConvNeXtV2). [^15] TF-Keras based. 9,083 tags. Danbooru up to ID 5,944,504. Downloads dropped to double digits. **Skip for new development.**

### animetimm (Danbooru v4 Taggers) — Emerging Alternative

**Source:** animetimm on HuggingFace [^17]
**License:** GPL-3.0

A 2025 project training on "Danbooru v4" dataset with 12,476 tags (9,225 general + 3,247 character + 4 rating). 27+ model variants.

| Model | Params | Tags | Macro-F1@0.40 |
| --- | --- | --- | --- |
| `eva02_large_patch14_448.dbv4-full` | 316.8M | 12,476 | 0.570 |
| `convnextv2_huge.dbv4-full` | 692.6M | 12,476 | 0.580 |

**Unique feature:** Per-tag optimal thresholds (each of 12,476 tags has individually optimized threshold vs. WD-Tagger's single global threshold). [^17]

**Concern:** **GPL-3.0 license** — if Loupe imports or links against these models, Loupe may need to be GPL. The 692M ConvNeXtV2-Huge will be demanding on RTX 3070 (1.2T FLOPS). Much smaller community adoption (149 downloads/month).

### Camie-Tagger v2

| Attribute | Value |
| --- | --- |
| **Repository** | `Camais03/camie-tagger-v2` [^18] |
| **Architecture** | ViT with cross-attention, dual-pool (patch mean + CLS token) |
| **Params** | 143M (66% smaller than v1's 424M) |
| **Training data** | Danbooru 2024, 2M images, multi-resolution (384→512px) |
| **Tag vocabulary** | **70,527 tags** (30,841 general, 26,968 character, 5,364 copyright, 7,007 artist, 323 meta, 4 rating, 20 year) |
| **Performance** | Micro-F1: 67.3%, Macro-F1: 50.6% |
| **License** | **GPL-3.0** |
| **Formats** | Safetensors, ONNX, PyTorch |
| **VRAM** | ~0.6–1 GB |
| **Python** | Requires 3.11.9 specifically |
| **Downloads** | 83/month |

**Key differentiator:** Instance-Aware Repeat Factor Sampling (IRFS) for long-tail handling — much better on rare tags than WD-Tagger (macro-F1 50.6% vs. ~44–48%). [^18] Largest tag vocabulary (70,527). Artist recognition +22pp, character detection +7.7pp over WD-Tagger.

**Caveats:** GPL-3.0 is more restrictive than Apache-2.0. Python 3.11.9 requirement conflicts with Loupe's 3.13+ target. Only 83 downloads/month.

### Other Taggers (Low Priority)

| Model | Tags | License | Status | Notes |
| --- | --- | --- | --- | --- |
| JoyTag [^33] | 5,813 | Apache-2.0 | Dormant | ViT-B/16, 845 dl/mo, fewer tags than WD |
| PixAI Tagger v0.9 [^34] | 13,461 | Apache-2.0 | Active | Opaque training, 35 dl/mo |
| ML-Danbooru [^40] | ~12K | Not specified | Dormant | Inferior to WD-Tagger on every axis |
| DeepDanbooru [^19] | ~9,000 | MIT | Obsolete | Superseded by WD-Tagger |

### Danbooru Tags Relevant to Loupe's Dimensions

The v3 vocabulary (10,861 tags) includes aesthetically useful tags: [^15]

**Composition/Framing:**
`full_body` (624K), `upper_body` (595K), `cowboy_shot` (438K), `from_behind` (194K), `from_side` (171K), `dutch_angle` (105K), `from_above` (81K), `from_below` (69K), `close-up` (37K)

**Color/Mood:**
`monochrome` (548K), `greyscale` (432K), `dark` (13K), `sepia` (7K), `muted_color` (5K), `colorful` (5K)

**Lighting:**
`sunlight` (66K), `lens_flare` (33K), `backlighting` (29K), `sunbeam` (9K), `dappled_sunlight` (8K)

**Scene/Setting:**
`outdoors` (440K), `sky` (357K), `day` (295K), `indoors` (282K), `night` (96K), `scenery` (43K)

**Visual effects:**
`blurry` (140K), `depth_of_field` (74K), `blurry_background` (74K)

**Not present:** `masterpiece`, `best_quality`, `highres`, `vibrant`, `dramatic_lighting`. These are SD prompt artifacts, not Danbooru metadata. For quality assessment, use deepghs/anime_aesthetic. [^6]

### Can Tag Confidence Serve as Aesthetic Signal?

**Partially, with caveats.** Tag confidence reflects how clearly a visual property is present, not how aesthetically pleasing it is. A badly composed image with obvious backlighting scores high on `backlighting`. Tags provide structured descriptive data for dimensional analysis (composition, lighting, scene type) but should not be used as quality proxies. Use a dedicated aesthetic scorer for quality.

### ONNX vs. PyTorch

All major taggers support both. For Loupe: if PyTorch is already loaded for CLIP, use **timm** for WD-Tagger (avoids adding onnxruntime). If seeking minimal dependencies for auxiliary models, **ONNX with onnxruntime-gpu** is lighter.

### Category 3 Recommendations for RQ5

- **Must evaluate:** WD-Tagger v3 SwinV2-Base and EVA02-Large [^15], deepghs/anime_aesthetic [^6] (SwinV2)
- **Worth evaluating:** animetimm EVA02-Large [^17] (per-tag thresholds, but GPL), deepghs/wd14_tagger_with_embeddings [^5] (anime embeddings)
- **Deprioritize:** DeepDanbooru [^19], WD-Tagger v2, ML-Danbooru [^40]

---

## Category 4: Auxiliary Vision Models

### Auxiliary Vision Models Landscape Overview

Auxiliary models (segmentation, depth, saliency, detection) support Loupe's analyzers indirectly — providing spatial analysis that feeds into composition, subject, and detail scoring. The key question is whether standard photographic models generalize to anime's flat colors, black outlines, and non-photorealistic shading.

**Summary:** Segmentation and detection have good anime-specific solutions. Depth estimation struggles. Saliency works reasonably, especially when adapted for anime.

### Segmentation

#### skytnt/anime-segmentation (Recommended)

| Attribute | Value |
| --- | --- |
| **Repository** | `skytnt/anime-seg` (HuggingFace) [^20], `SkyTNT/anime-segmentation` (GitHub) [^20] |
| **Architecture** | ISNet-IS (primary), also supports U2Net, MODNet, InSPyReNet variants |
| **Training data** | Combined AniSeg + character_bg_seg_data, cleaned with DeepDanbooru + manual review |
| **Formats** | PyTorch checkpoint, ONNX export |
| **License** | Apache-2.0 |
| **Input** | 1024px recommended |
| **Dependencies** | pytorch_lightning, kornia, timm, huggingface_hub >= 0.22 |

Purpose-built for anime character segmentation. Produces high-quality masks for separating characters from backgrounds. Directly useful for subject analyzer (focal region, negative space, depth layering). [^20]

#### SAM / SAM 2

Strong zero-shot generalization but **no published evaluation on anime**. [^22] SAM's strength is promptable (click/box) segmentation — interactive, not batch-friendly. Not recommended as primary; skytnt/anime-segmentation is purpose-built and lighter.

**VRAM:** SAM ViT-H ~2.5GB; SAM 2 Large ~3GB [^T]

### Depth Estimation

#### Depth Anything V2

| Attribute | Value |
| --- | --- |
| **Source** | `DepthAnything/Depth-Anything-V2` (GitHub) [^21] |
| **Architecture** | DINOv2-based, multiple sizes (Small/Base/Large/Giant) |
| **License** | Apache-2.0 |
| **VRAM** | Small ~0.5GB, Base ~1GB, Large ~2GB [^T] |

**Anime performance:** Users report **anime images appear flat**. [^T] Expected — anime uses flat color fills and stylized shading, not photorealistic depth cues. Can detect gross depth (foreground character vs. background) but struggles with subtle depth layering.

**Application note:** AniDepth (SIGGRAPH 2025) uses Depth Anything V2 for anime by converting frames to depth maps as a bridge domain — suggesting *some* utility with domain-aware post-processing. [^T]

#### MiDaS v3.1

Trained on 10+ photographic depth datasets. [^28] No anime evaluation found. Expected to share Depth Anything's limitations. Multiple model sizes (BEiT-Large down to LeViT).

**Verdict for depth:** Weakest auxiliary category for anime. Classical CV approaches (blur gradient analysis, scale-based layering) may be more reliable than learned depth for anime composition analysis. A depth model could still provide a coarse foreground/background signal for the subject analyzer.

### Saliency Detection

#### U2-Net

| Attribute | Value |
| --- | --- |
| **Source** | `xuebinqin/U-2-Net` (GitHub) [^23] |
| **Architecture** | Nested U-structure with skip connections |
| **Params** | ~44M (full), ~4.7M (lite) [^T] |
| **VRAM** | <1 GB [^T] |
| **License** | Apache-2.0 [^T] |

General-purpose saliency. Trained on photos but the approach (pixel-level saliency regression) generalizes better to anime than depth estimation — saliency still correlates with contrast, color prominence, and spatial position in anime. The skytnt/anime-segmentation project includes U2Net adapted for anime. [^20]

#### InSPyReNet

Image-pyramid-based saliency. Multi-resolution outputs. Available in skytnt/anime-segmentation project. [^20] [^T]

**Verdict:** Saliency models work reasonably on anime, especially when fine-tuned. Useful for subject analyzer (focal point, visual weight distribution).

### Face and Character Detection

#### deepghs Detection Suite

Comprehensive anime detection toolkit, non-profit, fully open-source, accessible via `imgutils` Python library: [^24] [^29]

| Model | Purpose |
| --- | --- |
| anime_face_detection [^24] | Anime face localization |
| anime_head_detection [^24] | Head position (works when face not visible) |
| anime_person_detection [^24] | Full-body character detection |

#### YOLO-based Anime Face Detectors

| Model | Architecture | Training Data | Performance |
| --- | --- | --- | --- |
| yolov8_animeface [^25] | YOLOv8x6 | 10K annotated safebooru images | Precision 0.957, Recall 0.924 |
| yolov5_anime [^26] | YOLOv5x/s | 5,845 annotated pixiv images | Good |
| AnimeHeadDetector [^27] | YOLOv3 | — | Heads (not just faces) |

**Verdict:** Strong ecosystem. deepghs suite is most comprehensive (face+head+person). YOLO detectors offer higher precision for face-only. Useful for subject analyzer (focal point anchoring, subject clarity).

### Additional deepghs Models

| Model | Purpose | Loupe Relevance |
| --- | --- | --- |
| anime_style_ages [^41] | Art era classifier (7 classes: 1970s- through 2020s). 71% accuracy, 0.93 AUC | Style dimension |
| anime_classification [^24] | Image type classifier | Filtering |
| anime_real_cls [^24] | Anime vs. real classifier | Low (input pre-screened) |
| ccip [^24] | Character similarity encoding | Low (outside scope) |

### Category 4 Recommendations for RQ5

- **Must evaluate:** skytnt/anime-segmentation [^20], deepghs face/head/person detection [^24]
- **Worth evaluating:** Depth Anything V2 Small [^21] (coarse depth), U2-Net [^23] (saliency), deepghs/anime_style_ages [^41]
- **Deprioritize:** SAM [^22] (not batch-friendly), MiDaS [^28] (redundant with Depth Anything)

---

## Cross-Category Analysis

### VRAM Budget

RTX 3070: 8GB total. ~1.5GB for OS/display = **~6.5GB working budget.**

| Configuration | Est. VRAM | Feasible? |
| --- | --- | --- |
| CLIP ViT-L/14 (1.5GB) + WD-Tagger SwinV2 (~1GB) + deepghs/anime_aesthetic (~0.3GB) + anime-segmentation (~0.5GB) | ~3.3 GB | Yes — comfortable |
| Above + deepghs face detection (~0.3GB) + Depth Anything V2 Small (~0.5GB) | ~4.1 GB | Yes |
| Above + kawaimasa aesthetic (~1GB) | ~5.1 GB | Yes |
| Above + Waifu Scorer MLP (<0.01GB, shares CLIP) | ~5.1 GB | Yes |
| Using CLIP ViT-H/14 instead of ViT-L/14 | +2.4 GB | Pushes to limits |

**Sequential load/unload likely unnecessary** with ViT-L/14 backbone. All recommended models fit simultaneously. If ViT-H/14 is desired, sequential processing may be needed.

**Note:** All VRAM estimates approximate — actual measurements needed in RQ5.

### Integration Complexity

| Framework | Models | Notes |
| --- | --- | --- |
| PyTorch (via timm) | WD-Tagger v3 [^15], CLIP (OpenCLIP) [^4], Depth Anything V2 [^21] | Primary — all coexist |
| ONNX Runtime | deepghs models [^24], skytnt models [^20], WD-Tagger (alt) | Lightweight, CPU-capable |
| PyTorch (custom) | Waifu Scorer [^8], kawaimasa [^7], anime-segmentation [^20] | Standard PyTorch |

Most models support both PyTorch and ONNX. Strategy: PyTorch for GPU-heavy models (CLIP, WD-Tagger), ONNX for lightweight auxiliary models (deepghs detection/aesthetic).

### Loupe Dimension Coverage

| Loupe Dimension | Primary Model(s) | Signal Type |
| --- | --- | --- |
| **Composition** | WD-Tagger [^15] (framing/angle tags), anime-segmentation [^20] (layout), saliency [^23] (visual weight) | Tags + spatial maps |
| **Color** | CLIP embeddings [^4], WD-Tagger [^15] (color mood tags) | Embeddings + tags |
| **Detail** | WD-Tagger [^15] (quality tags), deepghs/anime_aesthetic [^6] (quality tier) | Tags + classification |
| **Style** | CLIP embeddings [^4], WD-Tagger [^15] (style/artist tags), deepghs/anime_style_ages [^41] | Embeddings + tags + classification |
| **Subject** | anime-segmentation [^20] (character masks), deepghs detection [^24], saliency [^23] | Spatial maps + bounding boxes |
| **Lighting** | WD-Tagger [^15] (lighting tags) | Tags |
| **Overall Aesthetic** | deepghs/anime_aesthetic [^6], kawaimasa [^7], Waifu Scorer [^8] | Scores + tiers |

No single model covers all dimensions. Loupe needs a multi-model approach. The **WD-Tagger + CLIP + deepghs ecosystem** combination covers the broadest surface area with the least VRAM.

---

## Limitations and Open Questions

1. **No cross-model evaluation on anime exists.** All recommendations are based on architecture analysis, training data assessment, and community adoption — not empirical head-to-head testing. RQ5 must include comparative evaluation.

2. **Danbooru quality tag reliability is unvalidated.** Multiple models train on these tags. Whether they correlate with aesthetic quality vs. popularity/recency/artist-fame is an open question.

3. **The anime CLIP gap.** No dedicated contrastive CLIP fine-tune dominates. General CLIP has some anime understanding but its embedding space wasn't optimized for intra-anime discrimination. DanbooruCLIP [^1] is the best option but has unclear licensing and is inactive.

4. **Training data transparency is poor.** Waifu Scorer [^8], Aesthetic Shadow [^36], and skytnt/anime-aesthetic [^11] all hide training data. This makes bias assessment impossible.

5. **No SigLIP or BLIP anime fine-tunes exist** for aesthetic scoring. These newer architectures are untested in the anime domain.

6. **Community forums not surveyed.** Reddit, Discord, CivitAI contain valuable anecdotal performance reports — a follow-up search could surface additional models or insights.

7. **Camie-Tagger Python constraint.** The Python 3.11.9 requirement [^18] may be training-only or may affect inference — needs investigation.

8. **GPL license implications.** animetimm [^17] and Camie-Tagger [^18] are GPL-3.0. If Loupe imports their code, Loupe may need to be GPL. Loading models as external assets (not distributed) may satisfy GPL depending on interpretation.

9. **Custom training may be necessary.** If existing models underperform on anime screenshot ranking, the CLIP+MLP pattern and Blackroot's training code [^13] provide a feasible path to a custom scorer.

---

## References

[^1]: OysterQAQ, "DanbooruCLIP," HuggingFace Hub. <https://huggingface.co/OysterQAQ/DanbooruCLIP>
[^2]: dudcjs2779, "anime-style-tag-clip," HuggingFace Hub. <https://huggingface.co/dudcjs2779/anime-style-tag-clip>
[^3]: v2ray, "clipbooru," HuggingFace Hub. <https://huggingface.co/v2ray/clipbooru> — GitHub: <https://github.com/LagPixelLOL/clipbooru>
[^4]: mlfoundations, "OpenCLIP," GitHub. <https://github.com/mlfoundations/open_clip>
[^5]: deepghs, "wd14_tagger_with_embeddings," HuggingFace Hub. <https://huggingface.co/deepghs/wd14_tagger_with_embeddings>
[^6]: deepghs, "anime_aesthetic," HuggingFace Hub. <https://huggingface.co/deepghs/anime_aesthetic>
[^7]: kawaimasa, "kawai-aesthetic-scorer-convnextv2," HuggingFace Hub. <https://huggingface.co/kawaimasa/kawai-aesthetic-scorer-convnextv2>
[^8]: Eugeoter, "waifu-scorer-v3," HuggingFace Hub. <https://huggingface.co/Eugeoter/waifu-scorer-v3>
[^9]: Eugeoter, "waifu-scorer-v4-beta," HuggingFace Hub. <https://huggingface.co/Eugeoter/waifu-scorer-v4-beta>
[^10]: Schuhmann, C., "improved-aesthetic-predictor," GitHub. <https://github.com/christophschuhmann/improved-aesthetic-predictor>
[^11]: skytnt, "anime-aesthetic," HuggingFace Hub. <https://huggingface.co/skytnt/anime-aesthetic>
[^12]: Blackroot, "Anime-Aesthetic-Predictor-Medium," HuggingFace Hub. <https://huggingface.co/Blackroot/Anime-Aesthetic-Predictor-Medium>
[^13]: CoffeeVampir3, "easy-aesthetic-predictor," GitHub. <https://github.com/CoffeeVampir3/easy-aesthetic-predictor>
[^14]: shunk031, "aesthetics-predictor-v2-sac-logos-ava1-l14-linearMSE," HuggingFace Hub. <https://huggingface.co/shunk031/aesthetics-predictor-v2-sac-logos-ava1-l14-linearMSE>
[^15]: SmilingWolf, WD-Tagger v3 series, HuggingFace Hub. <https://huggingface.co/SmilingWolf> — Individual models: wd-eva02-large-tagger-v3, wd-vit-tagger-v3, wd-swinv2-tagger-v3, wd-vit-large-tagger-v3, wd-convnext-tagger-v3
[^16]: p1atdev, "wd-swinv2-tagger-v3-hf," HuggingFace Hub. <https://huggingface.co/p1atdev/wd-swinv2-tagger-v3-hf>
[^17]: animetimm, Danbooru v4 tagger series, HuggingFace Hub. <https://huggingface.co/animetimm>
[^18]: Camais03, "camie-tagger-v2," HuggingFace Hub. <https://huggingface.co/Camais03/camie-tagger-v2>
[^19]: KichangKim, "DeepDanbooru," GitHub. <https://github.com/KichangKim/DeepDanbooru> — ONNX port: <https://huggingface.co/skytnt/deepdanbooru_onnx>
[^20]: SkyTNT, "anime-segmentation," GitHub. <https://github.com/SkyTNT/anime-segmentation> — HuggingFace: <https://huggingface.co/skytnt/anime-seg>
[^21]: DepthAnything, "Depth-Anything-V2," GitHub. <https://github.com/DepthAnything/Depth-Anything-V2>
[^22]: Meta, "SAM 2," GitHub. <https://github.com/facebookresearch/sam2>
[^23]: Qin et al., "U2-Net," GitHub. <https://github.com/xuebinqin/U-2-Net>
[^24]: deepghs, HuggingFace organization page and imgutils library. <https://huggingface.co/deepghs> — GitHub: <https://github.com/deepghs/imgutils>
[^25]: Fuyucch1, "yolov8_animeface," GitHub. <https://github.com/Fuyucch1/yolov8_animeface>
[^26]: zymk9, "yolov5_anime," GitHub. <https://github.com/zymk9/yolov5_anime>
[^27]: grapeot, "AnimeHeadDetector," GitHub. <https://github.com/grapeot/AnimeHeadDetector>
[^28]: isl-org, "MiDaS," GitHub. <https://github.com/isl-org/MiDaS>
[^29]: deepghs, "imgutils," GitHub. <https://github.com/deepghs/imgutils>
[^30]: neggles, "wdv3-timm," GitHub. <https://github.com/neggles/wdv3-timm>
[^31]: SmilingWolf, "wdv3-jax," GitHub. <https://github.com/SmilingWolf/wdv3-jax>
[^32]: SmilingWolf, "wd-tagger" demo Space, HuggingFace Spaces. <https://huggingface.co/spaces/SmilingWolf/wd-tagger>
[^33]: fancyfeast, "joytag," HuggingFace Hub. <https://huggingface.co/fancyfeast/joytag>
[^34]: deepghs, "pixai-tagger-v0.9-onnx," HuggingFace Hub. <https://huggingface.co/deepghs/pixai-tagger-v0.9-onnx> — Original: <https://huggingface.co/pixai-labs/pixai-tagger-v0.9>
[^35]: cafeai, "cafe_aesthetic," HuggingFace Hub. <https://huggingface.co/cafeai/cafe_aesthetic>
[^36]: shadowlilac, "aesthetic-shadow," HuggingFace Hub. <https://huggingface.co/shadowlilac/aesthetic-shadow>
[^37]: RE-N-Y, "aesthetic-shadow-v2," HuggingFace Hub. <https://huggingface.co/RE-N-Y/aesthetic-shadow-v2>
[^38]: rsinema, "aesthetic-scorer," HuggingFace Hub. <https://huggingface.co/rsinema/aesthetic-scorer>
[^39]: minizhu, "aesthetic-anime-v2," HuggingFace Hub. <https://huggingface.co/minizhu/aesthetic-anime-v2>
[^40]: 7eu7d7, "ML-Danbooru," HuggingFace Hub. <https://huggingface.co/7eu7d7/ML-Danbooru>
[^41]: deepghs, "anime_style_ages," HuggingFace Hub. <https://huggingface.co/deepghs/anime_style_ages>
[^T]: Based on Claude's training knowledge — not verified against current sources in this session.
