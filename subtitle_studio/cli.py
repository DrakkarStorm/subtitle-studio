"""Typer CLI entry point — `pipeline` command."""

from __future__ import annotations

import logging
import os
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import typer
from dotenv import load_dotenv
from rich.console import Console
from rich.logging import RichHandler
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TaskID, TextColumn, TimeElapsedColumn

from . import __version__
from .pipeline import PipelineConfigError, PipelineStepError, run_pipeline

load_dotenv(override=False)

app = typer.Typer(
    name="subtitle-studio",
    help="Extract, verify and translate YouTube subtitles from a video file.",
    no_args_is_help=True,
    pretty_exceptions_show_locals=False,  # avoid leaking ANTHROPIC_API_KEY in tracebacks
    pretty_exceptions_short=True,
)

console = Console()
err_console = Console(stderr=True)

SUPPORTED_EXTENSIONS = {".mp4", ".mkv", ".mov", ".avi"}


def _version_callback(value: bool) -> None:
    if value:
        typer.echo(f"subtitle-studio {__version__}")
        raise typer.Exit()


def _configure_logging(verbose: int, quiet: bool) -> None:
    """Configure the root logger with a RichHandler on stderr.

    Priority: --quiet > --verbose. Default level is WARNING (emits warnings
    such as "Claude response truncated" or missing translation indices).
    """
    if quiet:
        level = logging.ERROR
    elif verbose >= 2:
        level = logging.DEBUG
    elif verbose == 1:
        level = logging.INFO
    else:
        level = logging.WARNING

    logging.basicConfig(
        level=level,
        format="%(message)s",
        handlers=[
            RichHandler(
                console=err_console,
                show_time=False,
                show_path=False,
                rich_tracebacks=False,
                markup=False,
            )
        ],
        force=True,
    )


@app.command()
def pipeline(
    video: Path = typer.Argument(..., help="Video file to process"),
    context: str = typer.Option(
        "",
        "--context",
        "-c",
        help="Video topic (e.g. 'Kubernetes tutorial') — improves detection accuracy",
    ),
    backend: str = typer.Option(
        "local",
        "--backend",
        "-b",
        help="Whisper backend: local or api",
    ),
    output: Path | None = typer.Option(
        None,
        "--output",
        "-o",
        help="Output directory (default: <stem>_YYYYMMDD_HHMMSS/ in the current working directory)",
    ),
    model: str = typer.Option(
        "",
        "--model",
        "-m",
        help="Claude model for the pipeline — detection and translation (default: claude-haiku-4-5-20251001)",
        envvar="ANTHROPIC_MODEL",
    ),
    short: bool = typer.Option(
        False,
        "--short",
        "-s",
        help="YouTube Shorts format (9:16 vertical) — lines ≤32 chars, adapted CPL thresholds",
    ),
    target_lang: str = typer.Option(
        "en",
        "--target-lang",
        "-l",
        help="Target translation language: en, es, de, pt (default: en)",
    ),
    max_cps: float | None = typer.Option(
        None,
        "--max-cps",
        help="CPS error threshold (default: 21.0 chars/s)",
        min=0.001,
    ),
    warn_cps: float | None = typer.Option(
        None,
        "--warn-cps",
        help="CPS warning threshold (default: 18.0 chars/s)",
        min=0.001,
    ),
    max_cpl: int | None = typer.Option(
        None,
        "--max-cpl",
        help="CPL error threshold — verification only, does not change line wrapping (default: 42 chars)",
        min=1,
    ),
    warn_cpl: int | None = typer.Option(
        None,
        "--warn-cpl",
        help="CPL warning threshold (default: 40 chars)",
        min=1,
    ),
    min_gap: int | None = typer.Option(
        None,
        "--min-gap",
        help="Minimum gap between subtitles in ms (default: 80 ms)",
        min=1,
    ),
    verbose: int = typer.Option(
        0,
        "--verbose",
        "-v",
        count=True,
        help="Increase log verbosity (-v: INFO, -vv: DEBUG)",
    ),
    quiet: bool = typer.Option(
        False,
        "--quiet",
        "-q",
        help="Errors only (takes precedence over --verbose)",
    ),
    check_guidelines: bool = typer.Option(
        False,
        "--check-guidelines",
        help=(
            "Audit YouTube guidelines (CPS/CPL/duration/gap) in landscape mode. "
            "Read-only: emits warnings in the report, never auto-fixes or blocks. "
            "No-op with --short (Shorts mode always audits)."
        ),
    ),
    _version: bool | None = typer.Option(
        None,
        "--version",
        "-V",
        callback=_version_callback,
        is_eager=True,
        help="Print the version and exit",
    ),
) -> None:
    """Extract, verify and translate YouTube subtitles from a video file."""
    _configure_logging(verbose, quiet)

    # Startup validation — fail fast before any computation
    if not os.environ.get("ANTHROPIC_API_KEY"):
        err_console.print("[red]Error: ANTHROPIC_API_KEY is missing.[/red]")
        err_console.print("Set it in a .env file or export it in your shell.")
        raise typer.Exit(code=1)

    video = video.resolve()
    if not video.exists():
        err_console.print(f"[red]Error: file not found: {video}[/red]")
        raise typer.Exit(code=1)
    if video.suffix.lower() not in SUPPORTED_EXTENSIONS:
        err_console.print(
            f"[red]Error: unsupported extension: {video.suffix!r}[/red]\n"
            f"Accepted extensions: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
        )
        raise typer.Exit(code=1)
    if backend not in ("local", "api"):
        err_console.print("[red]Error: --backend must be 'local' or 'api'.[/red]")
        raise typer.Exit(code=1)

    # Model resolution: --model > $ANTHROPIC_MODEL (via Typer envvar) > default constant
    from .detect.detector import MODEL as _DEFAULT_MODEL
    from .translate.translate import SUPPORTED_LANGS

    if target_lang not in SUPPORTED_LANGS:
        err_console.print(f"[red]Error: --target-lang must be one of: {', '.join(SUPPORTED_LANGS.keys())}[/red]")
        raise typer.Exit(code=1)

    resolved_model = model or _DEFAULT_MODEL

    # Warn if CPS/CPL/duration flags were passed without --short or --check-guidelines,
    # because they are silently ignored in landscape mode.
    threshold_flags = {
        "--max-cps": max_cps,
        "--warn-cps": warn_cps,
        "--max-cpl": max_cpl,
        "--warn-cpl": warn_cpl,
        "--min-gap": min_gap,
    }
    active_thresholds = [name for name, value in threshold_flags.items() if value is not None]
    if active_thresholds and not short and not check_guidelines:
        err_console.print(
            f"[yellow]Warning:[/yellow] {', '.join(active_thresholds)} "
            "are ignored in landscape mode. Pass --short or --check-guidelines to enable the audit."
        )

    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            TimeElapsedColumn(),
            console=err_console,
            transient=False,
        ) as progress:
            steps = ("Extraction", "Verification", "Translation")
            tasks: dict[str, TaskID] = {
                name: progress.add_task(f"[dim]{name}[/dim]", total=1, start=False) for name in steps
            }

            @contextmanager
            def step_ctx(name: str) -> Iterator[None]:
                task_id = tasks[name]
                progress.start_task(task_id)
                progress.update(task_id, description=f"[cyan]{name}…[/cyan]")
                try:
                    yield
                except BaseException:
                    progress.update(task_id, description=f"[red]{name} — failed[/red]", completed=1)
                    raise
                else:
                    progress.update(task_id, description=f"[green]{name} ✓[/green]", completed=1)

            result_dir = run_pipeline(
                video,
                output_dir=output,
                context=context,
                backend=backend,
                model=resolved_model,
                shorts=short,
                target_lang=target_lang,
                max_cps=max_cps,
                warn_cps=warn_cps,
                max_cpl=max_cpl,
                warn_cpl=warn_cpl,
                min_gap=min_gap,
                check_guidelines=check_guidelines,
                step_ctx=step_ctx,
            )

        console.print(
            Panel(
                f"Artifacts in: [bold]{result_dir}[/bold]",
                title="[green]Pipeline complete[/green]",
                border_style="green",
            )
        )
    except PipelineConfigError as exc:
        err_console.print(f"[red]Configuration error: {exc}[/red]")
        raise typer.Exit(code=1) from None
    except PipelineStepError as exc:
        err_console.print(Panel(str(exc), title="[red]Pipeline failed[/red]", border_style="red"))
        raise typer.Exit(code=1) from None
    except KeyboardInterrupt:
        err_console.print("\n[yellow]Pipeline interrupted.[/yellow]")
        raise typer.Exit(code=130) from None
