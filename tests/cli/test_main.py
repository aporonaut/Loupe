"""Smoke tests for the CLI."""

from typer.testing import CliRunner

from loupe.cli.main import app

runner = CliRunner()


class TestCLI:
    def test_help(self) -> None:
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        assert "loupe" in result.output.lower() or "aesthetic" in result.output.lower()

    def test_analyze_help(self) -> None:
        result = runner.invoke(app, ["analyze", "--help"])
        assert result.exit_code == 0
        assert "analyze" in result.output.lower()

    def test_rank_help(self) -> None:
        result = runner.invoke(app, ["rank", "--help"])
        assert result.exit_code == 0

    def test_report_help(self) -> None:
        result = runner.invoke(app, ["report", "--help"])
        assert result.exit_code == 0

    def test_tags(self) -> None:
        result = runner.invoke(app, ["tags"])
        assert result.exit_code == 0

    def test_no_args_shows_help(self) -> None:
        result = runner.invoke(app, [])
        # Typer returns exit code 0 for --help, but 2 for no_args_is_help
        assert result.exit_code in (0, 2)
