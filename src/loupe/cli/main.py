"""Typer CLI entry point and subcommands."""

from __future__ import annotations

import statistics
from pathlib import Path  # noqa: TC003 — Typer needs Path at runtime
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from loupe.core.config import load_config
from loupe.core.engine import Engine
from loupe.core.models import LoupeResult  # noqa: TC001 — used at runtime in helpers
from loupe.io.sidecar import read_result

app = typer.Typer(
    name="loupe",
    help="Modular aesthetic analysis tool for anime screenshots.",
    no_args_is_help=True,
)
console = Console()

IMAGE_EXTENSIONS = frozenset({".jpg", ".jpeg", ".png", ".webp", ".tiff", ".bmp"})

DIMENSIONS = ("composition", "color", "detail", "lighting", "subject", "style")


def _collect_images(path: Path) -> list[Path]:
    """Collect image files from a path (single file or directory)."""
    if path.is_file():
        return [path]
    if path.is_dir():
        return sorted(p for p in path.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS)
    return []


def _register_analyzers(engine: Engine) -> None:
    """Register all available analyzers on an engine instance."""
    from loupe.analyzers.color import ColorAnalyzer
    from loupe.analyzers.composition import CompositionAnalyzer
    from loupe.analyzers.detail import DetailAnalyzer
    from loupe.analyzers.lighting import LightingAnalyzer
    from loupe.analyzers.style import StyleAnalyzer
    from loupe.analyzers.subject import SubjectAnalyzer

    engine.register_analyzer(ColorAnalyzer())
    engine.register_analyzer(CompositionAnalyzer())
    engine.register_analyzer(DetailAnalyzer())
    engine.register_analyzer(LightingAnalyzer())
    engine.register_analyzer(SubjectAnalyzer())
    engine.register_analyzer(StyleAnalyzer())


def _top_dimensions(result: LoupeResult, n: int = 2) -> list[str]:
    """Get the top N contributing dimensions for a result."""
    contributions = result.scoring.contributions
    if not contributions:
        return []
    ranked = sorted(contributions, key=lambda d: contributions[d], reverse=True)
    return ranked[:n]


def _profile_tag(result: LoupeResult) -> str:
    """Generate a brief profile tag like 'composition-driven' or 'balanced'."""
    contributions = result.scoring.contributions
    if not contributions:
        return ""
    ranked = sorted(contributions.items(), key=lambda x: x[1], reverse=True)
    if len(ranked) < 2:
        return f"{ranked[0][0]}-driven"
    top_val = ranked[0][1]
    second_val = ranked[1][1]
    # If the top dimension contributes significantly more than the second
    if top_val > 0 and second_val > 0 and top_val / second_val > 1.5:
        return f"{ranked[0][0]}-driven"
    return "balanced"


def _print_analyze_summary(result: LoupeResult, *, verbose: bool = False) -> None:
    """Print a rich per-image analysis summary."""
    console.print(f"\n[bold]{result.image_path.name}[/bold]")

    # Dimension scores table
    table = Table(show_header=True, box=None, padding=(0, 1))
    table.add_column("Dimension", style="cyan", min_width=12)
    table.add_column("Score", justify="right", min_width=6)
    table.add_column("Tags", style="dim")

    for ar in result.analyzer_results:
        # Color-code score
        score = ar.score
        if score >= 0.7:
            score_str = f"[green]{score:.3f}[/green]"
        elif score >= 0.4:
            score_str = f"[yellow]{score:.3f}[/yellow]"
        else:
            score_str = f"[red]{score:.3f}[/red]"

        if verbose:
            tag_names = [t.name for t in ar.tags]
        else:
            # Show top 3 tags by confidence
            top_tags = sorted(ar.tags, key=lambda t: t.confidence, reverse=True)[:3]
            tag_names = [t.name for t in top_tags]

        tags_str = ", ".join(tag_names) if tag_names else ""
        table.add_row(ar.analyzer, score_str, tags_str)

    console.print(table)

    # Aggregate line
    agg = result.aggregate_score
    profile = _profile_tag(result)
    profile_str = f"  [dim]({profile})[/dim]" if profile else ""
    console.print(f"  [bold]Aggregate: {agg:.3f}[/bold]{profile_str}")


@app.command()
def analyze(
    path: Annotated[Path, typer.Argument(help="Image file or directory to analyze.")],
    config: Annotated[
        Path | None, typer.Option("--config", help="Path to config TOML file.")
    ] = None,
    preset: Annotated[
        str | None,
        typer.Option(
            "--preset", help="Scoring preset (balanced, composition, visual)."
        ),
    ] = None,
    force: Annotated[
        bool, typer.Option("--force", help="Re-analyze even if sidecar exists.")
    ] = False,
    verbose: Annotated[
        bool, typer.Option("--verbose", "-v", help="Show all tags per dimension.")
    ] = False,
) -> None:
    """Analyze a single image or all images in a directory."""
    if verbose:
        import logging

        loupe_logger = logging.getLogger("loupe")
        loupe_logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(name)s: %(message)s"))
        loupe_logger.addHandler(handler)

    images = _collect_images(path)
    if not images:
        console.print(f"[red]No images found at: {path}[/red]")
        raise typer.Exit(1)

    try:
        cfg = load_config(config_path=config, preset=preset)
    except Exception as e:
        console.print(f"[red]Configuration error: {e}[/red]")
        raise typer.Exit(1) from None

    engine = Engine(cfg)
    _register_analyzers(engine)

    # Load models before starting the progress bar
    console.print("[dim]Loading models...[/dim]")
    engine.ensure_models_loaded()

    from rich.progress import (
        BarColumn,
        MofNCompleteColumn,
        Progress,
        TextColumn,
        TimeRemainingColumn,
    )

    skipped = 0
    results: list[LoupeResult] = []

    with Progress(
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeRemainingColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Analyzing", total=len(images))
        for img in images:
            result = engine.analyze(img, force=force)
            if result is not None:
                results.append(result)
            else:
                skipped += 1
            progress.advance(task)

    # Summary
    console.print(f"\n[green]Analyzed {len(results)} image(s)[/green]", end="")
    if skipped:
        console.print(f"  [dim]({skipped} skipped \u2014 already analyzed)[/dim]")
    else:
        console.print()

    # Single image: always show per-dimension detail.
    # Batch: show score range summary (verbose adds top/bottom 3).
    if len(images) == 1:
        for result in results:
            _print_analyze_summary(result, verbose=verbose)
    elif results:
        scores = [r.aggregate_score for r in results]
        console.print(f"  Score range: {min(scores):.3f}\u2013{max(scores):.3f}")
        if verbose:
            ranked = sorted(results, key=lambda r: r.aggregate_score, reverse=True)
            console.print("\n[bold]Top 3:[/bold]")
            for result in ranked[:3]:
                _print_analyze_summary(result, verbose=False)
            console.print("\n[bold]Bottom 3:[/bold]")
            for result in ranked[-3:]:
                _print_analyze_summary(result, verbose=False)
        console.print(
            "  [dim]Use 'loupe rank' to view full rankings"
            " or 'loupe analyze <image>' for per-image detail.[/dim]"
        )


@app.command()
def rank(
    path: Annotated[Path, typer.Argument(help="Directory with analyzed images.")],
    preset: Annotated[
        str | None, typer.Option("--preset", help="Scoring preset.")
    ] = None,
    limit: Annotated[
        int | None, typer.Option("--limit", "-n", help="Show top N images only.")
    ] = None,
    rename: Annotated[
        bool,
        typer.Option(
            "--rename",
            help="Prefix filenames with rank number (e.g. 001_image.png).",
        ),
    ] = False,
) -> None:
    """List images sorted by aggregate score."""
    images = _collect_images(path)
    if not images:
        console.print(f"[red]No images found at: {path}[/red]")
        raise typer.Exit(1)

    scored: list[tuple[Path, float, list[str], str]] = []
    for img in images:
        result = read_result(img)
        if result is None:
            continue

        # If a preset is specified, recompute aggregate with those weights
        if preset is not None:
            from loupe.core.config import SCORING_PRESETS
            from loupe.core.scoring import compute_aggregate

            weights = SCORING_PRESETS.get(preset, SCORING_PRESETS["balanced"])
            agg, scoring = compute_aggregate(result.analyzer_results, weights)
            result.aggregate_score = agg
            result.scoring = scoring

        top_dims = _top_dimensions(result, n=2)
        profile = _profile_tag(result)
        scored.append((img, result.aggregate_score, top_dims, profile))

    if not scored:
        console.print(
            "[yellow]No analyzed images found. Run 'loupe analyze' first.[/yellow]"
        )
        raise typer.Exit(1)

    scored.sort(key=lambda x: x[1], reverse=True)

    if limit is not None:
        scored = scored[:limit]

    table = Table(title="Image Rankings")
    table.add_column("#", style="dim", width=4)
    table.add_column("Image", style="bold")
    table.add_column("Score", justify="right")
    table.add_column("Top Dimensions", style="cyan")
    table.add_column("Profile", style="dim")

    pad = len(str(len(scored)))
    for i, (img_path, score, top_dims, profile) in enumerate(scored, 1):
        if score >= 0.7:
            score_str = f"[green]{score:.3f}[/green]"
        elif score >= 0.4:
            score_str = f"[yellow]{score:.3f}[/yellow]"
        else:
            score_str = f"[red]{score:.3f}[/red]"

        dims_str = ", ".join(top_dims)
        table.add_row(str(i), img_path.name, score_str, dims_str, profile)

    console.print(table)

    if rename:
        _rename_with_rank(scored, pad)


def _rename_with_rank(
    scored: list[tuple[Path, float, list[str], str]],
    pad: int,
) -> None:
    """Prefix filenames with zero-padded rank numbers.

    Strips any existing rank prefix (digits followed by underscore)
    before applying the new one so the command is idempotent.
    """
    import re

    renamed = 0
    for i, (img_path, _score, _dims, _profile) in enumerate(scored, 1):
        name = img_path.name
        # Strip existing rank prefix like "001_"
        stripped = re.sub(r"^\d+_", "", name)
        new_name = f"{i:0{pad}d}_{stripped}"
        if new_name == name:
            continue
        new_path = img_path.parent / new_name
        img_path.rename(new_path)
        # Also rename sidecar if it exists
        sidecar = img_path.parent / ".loupe" / f"{name}.json"
        if sidecar.exists():
            sidecar.rename(sidecar.parent / f"{new_name}.json")
        renamed += 1

    console.print(f"[green]Renamed {renamed} file(s) with rank prefix.[/green]")


@app.command()
def report(
    path: Annotated[Path, typer.Argument(help="Directory with analyzed images.")],
) -> None:
    """Summarize existing sidecar results for a directory."""
    images = _collect_images(path)
    results: list[LoupeResult] = []

    for img in images:
        result = read_result(img)
        if result is not None:
            results.append(result)

    if not results:
        console.print("[yellow]No analyzed images found.[/yellow]")
        raise typer.Exit(1)

    console.print(f"[bold]Report for {path}[/bold]\n")
    console.print(f"  Images analyzed: {len(results)}\n")

    # Aggregate score statistics
    agg_scores = [r.aggregate_score for r in results]
    console.print("[bold]Aggregate Score Distribution[/bold]")
    _print_score_stats("  aggregate", agg_scores)
    console.print()

    # Per-dimension statistics
    console.print("[bold]Per-Dimension Scores[/bold]")
    dim_table = Table(show_header=True, box=None, padding=(0, 1))
    dim_table.add_column("Dimension", style="cyan", min_width=12)
    dim_table.add_column("Min", justify="right")
    dim_table.add_column("Median", justify="right")
    dim_table.add_column("Mean", justify="right")
    dim_table.add_column("Max", justify="right")
    dim_table.add_column("Std Dev", justify="right", style="dim")

    for dim in DIMENSIONS:
        scores: list[float] = [
            ar.score for r in results for ar in r.analyzer_results if ar.analyzer == dim
        ]
        if not scores:
            continue
        dim_table.add_row(
            dim,
            f"{min(scores):.3f}",
            f"{statistics.median(scores):.3f}",
            f"{statistics.mean(scores):.3f}",
            f"{max(scores):.3f}",
            f"{statistics.stdev(scores):.3f}" if len(scores) > 1 else "\u2014",
        )

    console.print(dim_table)
    console.print()

    # Dimension correlation summary — which dimensions tend to score together?
    console.print("[bold]Dimension Correlations (Pearson)[/bold]")
    dim_scores: dict[str, list[float]] = {}
    for dim in DIMENSIONS:
        scores_list: list[float] = [
            ar.score for r in results for ar in r.analyzer_results if ar.analyzer == dim
        ]
        if len(scores_list) == len(results):
            dim_scores[dim] = scores_list

    if len(dim_scores) >= 2:
        corr_table = Table(show_header=True, box=None, padding=(0, 1))
        corr_table.add_column("Pair", style="cyan")
        corr_table.add_column("r", justify="right")

        pairs: list[tuple[str, str, float]] = []
        dim_names = sorted(dim_scores.keys())
        for i, d1 in enumerate(dim_names):
            for d2 in dim_names[i + 1 :]:
                r = _pearson(dim_scores[d1], dim_scores[d2])
                pairs.append((d1, d2, r))

        # Show strongest correlations first
        pairs.sort(key=lambda x: abs(x[2]), reverse=True)
        for d1, d2, r in pairs:
            if abs(r) >= 0.3:
                style = "[green]" if r > 0 else "[red]"
                corr_table.add_row(f"{d1} ↔ {d2}", f"{style}{r:+.3f}[/]")
            else:
                corr_table.add_row(f"{d1} ↔ {d2}", f"[dim]{r:+.3f}[/dim]")

        console.print(corr_table)
    else:
        console.print("  [dim]Not enough data for correlations.[/dim]")


def _print_score_stats(label: str, scores: list[float]) -> None:
    """Print min/median/mean/max for a list of scores."""
    console.print(
        f"{label}:  "
        f"min={min(scores):.3f}  "
        f"median={statistics.median(scores):.3f}  "
        f"mean={statistics.mean(scores):.3f}  "
        f"max={max(scores):.3f}"
    )


def _pearson(x: list[float], y: list[float]) -> float:
    """Compute Pearson correlation coefficient between two lists."""
    n = len(x)
    if n < 2:
        return 0.0
    mx = statistics.mean(x)
    my = statistics.mean(y)
    num = sum((xi - mx) * (yi - my) for xi, yi in zip(x, y, strict=True))
    dx = sum((xi - mx) ** 2 for xi in x) ** 0.5
    dy = sum((yi - my) ** 2 for yi in y) ** 0.5
    if dx == 0 or dy == 0:
        return 0.0
    return num / (dx * dy)


# -- Tag reference data for the `tags` command --

_TAG_REFERENCE: dict[str, list[tuple[str, str]]] = {
    "composition": [
        ("rule_of_thirds", "Subject placed near third-line intersections"),
        ("centered", "Subject placed near frame center"),
        ("balanced", "Visual weight evenly distributed"),
        ("symmetric", "Bilateral symmetry detected"),
        ("strong_leading_lines", "Prominent converging lines"),
        ("diagonal_composition", "Strong diagonal line structure"),
        ("open_composition", "Significant negative space"),
        ("framed_subject", "In-frame elements border the subject"),
    ],
    "color": [
        ("harmonic_i", "Matsuda i-type harmony (identity)"),
        ("harmonic_V", "Matsuda V-type harmony (complementary)"),
        ("harmonic_I", "Matsuda I-type harmony (analogous pair)"),
        ("harmonic_L", "Matsuda L-type harmony (split-complementary)"),
        ("harmonic_T", "Matsuda T-type harmony (triadic)"),
        ("harmonic_Y", "Matsuda Y-type harmony (analogous triad)"),
        ("harmonic_X", "Matsuda X-type harmony (tetradic)"),
        ("harmonic_N", "Matsuda N-type harmony (no clear harmony)"),
        ("warm_palette", "Dominant warm color temperature"),
        ("cool_palette", "Dominant cool color temperature"),
        ("neutral_palette", "No dominant temperature bias"),
        ("vivid", "High overall colorfulness"),
        ("muted", "Low overall colorfulness"),
        ("monochromatic", "Single-hue palette"),
        ("limited_palette", "Restricted color diversity"),
        ("diverse_palette", "Wide color diversity"),
    ],
    "detail": [
        ("high_detail", "Overall detail score above 0.7"),
        ("rich_background", "Background region has high detail"),
        ("detailed_character", "Character region has high detail"),
        ("sharp_rendering", "High rendering clarity"),
        ("complex_shading", "Many tonal levels in shading"),
        ("fine_line_work", "High line work quality"),
    ],
    "lighting": [
        ("high_contrast", "Strong tonal range"),
        ("low_contrast", "Narrow tonal range"),
        ("dramatic_lighting", "Overall lighting score above 0.7"),
        ("flat_lighting", "Overall lighting score below 0.25"),
        ("rim_lit", "Rim/edge lighting on character boundary"),
        ("soft_shadows", "Gradual shadow boundaries"),
        ("hard_shadows", "Sharp shadow boundaries"),
        ("atmospheric", "Bloom or glow effects detected"),
        ("directional_light", "Strong luminance gradient"),
        ("balanced_exposure", "Even tonal zone distribution"),
        ("backlighting", "WD-Tagger: backlighting detected"),
        ("sunlight", "WD-Tagger: sunlight detected"),
        ("moonlight", "WD-Tagger: moonlight detected"),
        ("lens_flare", "WD-Tagger: lens flare detected"),
        ("light_rays", "WD-Tagger: light rays detected"),
        ("glowing", "WD-Tagger: glow effect detected"),
    ],
    "subject": [
        ("extreme_closeup", "Subject >60% of frame area"),
        ("closeup", "Subject 30-60% of frame area"),
        ("medium_shot", "Subject 15-30% of frame area"),
        ("wide_shot", "Subject 5-15% of frame area"),
        ("very_wide", "Subject <5% of frame area"),
        ("environment_focus", "No character detected"),
        ("strong_separation", "Strong figure-ground separation"),
        ("shallow_dof", "Significant DOF blur differential"),
        ("complete_subject", "Subject fully within frame"),
    ],
    "style": [
        ("aesthetic_masterpiece", "Top-tier aesthetic quality"),
        ("aesthetic_best", "Excellent aesthetic quality"),
        ("aesthetic_great", "Great aesthetic quality"),
        ("aesthetic_good", "Good aesthetic quality"),
        ("aesthetic_normal", "Average aesthetic quality"),
        ("aesthetic_low", "Below-average aesthetic quality"),
        ("aesthetic_worst", "Poor aesthetic quality"),
        ("consistent_rendering", "Uniform quality across layers"),
        ("inconsistent_rendering", "Quality varies across layers"),
        ("flat_color", "WD-Tagger: flat color style"),
        ("gradient", "WD-Tagger: gradient shading"),
        ("realistic", "WD-Tagger: realistic rendering"),
        ("cel_shading", "WD-Tagger: cel-shaded style"),
        ("soft_shading", "WD-Tagger: soft shading technique"),
        ("naturalistic_anime", "CLIP: naturalistic anime style"),
        ("geometric_abstract_anime", "CLIP: geometric/abstract anime style"),
        ("painterly_anime", "CLIP: painterly anime style"),
        ("digital_modern_anime", "CLIP: modern digital anime style"),
        ("retro_cel_anime", "CLIP: retro cel animation style"),
    ],
}


@app.command()
def tags() -> None:
    """List all available tags across enabled analyzers."""
    for dim in DIMENSIONS:
        dim_tags = _TAG_REFERENCE.get(dim, [])
        if not dim_tags:
            continue
        console.print(f"\n[bold cyan]{dim}[/bold cyan]  ({len(dim_tags)} tags)")
        tag_table = Table(show_header=False, box=None, padding=(0, 2))
        tag_table.add_column("Tag", style="bold")
        tag_table.add_column("Description", style="dim")
        for name, desc in dim_tags:
            tag_table.add_row(name, desc)
        console.print(tag_table)

    total = sum(len(t) for t in _TAG_REFERENCE.values())
    console.print(f"\n[dim]{total} tags across {len(DIMENSIONS)} dimensions[/dim]")


@app.command()
def setup() -> None:
    """Download all model files required for analysis.

    Pre-downloads models from HuggingFace Hub to the local cache
    so that subsequent ``loupe analyze`` runs don't require network
    access. This may take several minutes on the first run.
    """
    from loupe.models.manager import ModelManager

    console.print("[bold]Downloading Loupe models...[/bold]\n")
    console.print("This may take several minutes on first run.\n")

    try:
        ModelManager.download_all()
        console.print("\n[green]All models downloaded successfully.[/green]")
    except Exception as e:
        console.print(f"\n[red]Error downloading models: {e}[/red]")
        console.print(
            "[yellow]Hint: Run 'loupe setup' with network access "
            "to download required models.[/yellow]"
        )
        raise typer.Exit(1) from None


if __name__ == "__main__":
    app()
