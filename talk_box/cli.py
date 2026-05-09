"""Talk Box CLI: command-line interface for managing and testing AI assistants."""

from __future__ import annotations

import sys

import click
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

# ---------------------------------------------------------------------------
# CLI group
# ---------------------------------------------------------------------------


@click.group()
@click.version_option(package_name="talk-box")
def main() -> None:
    """Talk Box: build production AI assistants.

    Use 'talk-box COMMAND --help' for details on each command.
    """


# ---------------------------------------------------------------------------
# info
# ---------------------------------------------------------------------------


@main.command()
@click.option("--personas", is_flag=True, help="Show all available personas.")
@click.option("--models", is_flag=True, help="Show registered model profiles.")
@click.option("--ollama", is_flag=True, help="Detect and list Ollama models.")
def info(
    *,
    personas: bool,
    models: bool,
    ollama: bool,
) -> None:
    """Show Talk Box configuration, personas, and models.

    With no flags, displays a summary of the installed version,
    persona count, and model count.
    """
    console = Console()

    if personas:
        _show_personas(console)
        return

    if models:
        _show_models(console)
        return

    if ollama:
        _show_ollama(console)
        return

    # Default: summary overview
    _show_summary(console)


def _show_summary(console: Console) -> None:
    """Print a summary panel."""
    import talk_box as tb
    from talk_box.personas import list_personas, persona_categories

    version = tb.__version__
    persona_names = list_personas()
    categories = persona_categories()
    model_profiles = tb.list_models()
    providers = sorted({m.provider for m in model_profiles})

    table = Table(show_header=False, box=None, padding=(0, 2))
    table.add_column(style="bold cyan")
    table.add_column()
    table.add_row("Version", version)
    table.add_row(
        "Python", f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    )
    table.add_row("Personas", f"{len(persona_names)} across {len(categories)} categories")
    table.add_row("Models", f"{len(model_profiles)} profiles ({', '.join(providers)})")

    # Check Ollama
    ollama_status = tb.detect_ollama()
    if ollama_status.available:
        table.add_row("Ollama", f"running ({len(ollama_status.models)} models)")
    else:
        table.add_row("Ollama", "not detected")

    console.print(Panel(table, title="[bold]Talk Box[/bold]", border_style="blue"))


def _show_personas(console: Console) -> None:
    """Print persona table grouped by category."""
    from talk_box.personas import get_persona, persona_categories

    categories = persona_categories()
    table = Table(title="Available Personas", border_style="blue")
    table.add_column("Category", style="bold cyan")
    table.add_column("Name", style="green")
    table.add_column("Description")

    for category, names in categories.items():
        for i, name in enumerate(names):
            persona = get_persona(name)
            cat_label = category if i == 0 else ""
            table.add_row(cat_label, name, persona.description or "—")

    console.print(table)


def _show_models(console: Console) -> None:
    """Print model profiles table."""
    import talk_box as tb

    profiles = tb.list_models()
    table = Table(title="Registered Model Profiles", border_style="blue")
    table.add_column("Provider", style="bold cyan")
    table.add_column("Model", style="green")
    table.add_column("Context", justify="right")
    table.add_column("Tools")
    table.add_column("Vision")
    table.add_column("Cost")

    for p in profiles:
        ctx = f"{p.context_window:,}" if p.context_window else "—"
        tools = _bool_icon(p.supports_tools)
        vision = _bool_icon(p.supports_vision)
        cost = p.cost_tier.value if p.cost_tier else "—"
        table.add_row(p.provider, p.model, ctx, tools, vision, cost)

    console.print(table)


def _show_ollama(console: Console) -> None:
    """Detect Ollama and list available models."""
    import talk_box as tb

    status = tb.detect_ollama()
    if not status.available:
        console.print("[yellow]Ollama is not running or not installed.[/yellow]")
        console.print("Install from https://ollama.com and run 'ollama serve'.")
        return

    if not status.models:
        console.print("[green]Ollama is running[/green] but no models are pulled.")
        console.print("Run 'ollama pull llama3.2' to get started.")
        return

    table = Table(title="Ollama Models", border_style="green")
    table.add_column("Model", style="bold green")
    table.add_column("Size")

    for model in sorted(status.models):
        table.add_row(model, "—")

    console.print(table)


def _bool_icon(value: bool | None) -> str:
    """Convert a bool to a display icon."""
    if value is True:
        return "✓"
    if value is False:
        return "✗"
    return "—"


# ---------------------------------------------------------------------------
# test
# ---------------------------------------------------------------------------


@main.command("test")
@click.argument("persona")
@click.option(
    "--model",
    "-m",
    multiple=True,
    required=True,
    help="Model string (provider:model). Can be specified multiple times.",
)
@click.option(
    "--judge",
    "-j",
    default=None,
    help="Judge model string for scoring (e.g., 'anthropic:claude-sonnet-4-6').",
)
@click.option(
    "--output",
    "-o",
    default=None,
    type=click.Path(),
    help="Write scorecard JSON to this path.",
)
@click.option(
    "--threshold",
    "-t",
    default=None,
    type=float,
    help="Minimum passing score (0.0–1.0). Exit 1 if any score is below.",
)
@click.option(
    "--no-guards",
    is_flag=True,
    help="Skip default persona guardrails during eval.",
)
def test_cmd(
    persona: str,
    model: tuple[str, ...],
    judge: str | None,
    output: str | None,
    threshold: float | None,
    no_guards: bool,
) -> None:
    """Run eval suite for a persona across one or more models.

    Evaluates the persona's test queries against each model, scores
    with a judge, and prints a scorecard.

    Examples:

        talk-box test code_reviewer -m anthropic:claude-sonnet-4-6

        talk-box test customer_support_tier1 -m ollama:llama4 -m anthropic:claude-sonnet-4-6 -t 0.85
    """
    import talk_box as tb

    console = Console()

    # Validate persona exists
    try:
        tb.get_persona(persona)
    except KeyError:
        console.print(f"[red]Unknown persona: {persona}[/red]")
        console.print(f"Available: {', '.join(tb.list_personas())}")
        raise SystemExit(1)

    console.print(
        f"[bold]Evaluating[/bold] [cyan]{persona}[/cyan] across {len(model)} model(s)...\n"
    )

    results = tb.eval_suite(
        persona,
        models=list(model),
        judge=judge,
        default_guards=not no_guards,
        scorecard_path=output,
    )

    # Print results table
    table = Table(title=f"Eval: {persona}", border_style="blue")
    table.add_column("Model", style="bold cyan")
    table.add_column("Score", justify="right")
    table.add_column("Dimensions")

    for variant_name, variant_results in results.by_variant.items():
        scores = variant_results.scores
        if scores:
            mean = sum(s.score for s in scores) / len(scores)
            dims = ", ".join(f"{s.dimension.value}={s.score:.2f}" for s in scores)
            score_style = "green" if (threshold is None or mean >= threshold) else "red"
            table.add_row(variant_name, f"[{score_style}]{mean:.2f}[/{score_style}]", dims)
        else:
            table.add_row(variant_name, "—", "no scores")

    console.print(table)

    if output:
        console.print(f"\nScorecard written to [bold]{output}[/bold]")

    # Exit with error if below threshold
    if threshold is not None:
        all_scores = [s.score for s in results.scores]
        if all_scores and min(all_scores) < threshold:
            console.print(
                f"\n[red]FAIL[/red]: minimum score {min(all_scores):.2f} "
                f"< threshold {threshold:.2f}"
            )
            raise SystemExit(1)
        if all_scores:
            console.print(f"\n[green]PASS[/green]: all scores ≥ {threshold:.2f}")


# ---------------------------------------------------------------------------
# personas
# ---------------------------------------------------------------------------


@main.command()
@click.argument("name", required=False)
def personas(name: str | None) -> None:
    """List personas or show details for a specific persona.

    Examples:

        talk-box personas

        talk-box personas code_reviewer
    """
    console = Console()

    if name is None:
        _show_personas(console)
        return

    from talk_box.personas import get_persona

    try:
        persona = get_persona(name)
    except KeyError:
        from talk_box.personas import list_personas

        console.print(f"[red]Unknown persona: {name}[/red]")
        console.print(f"Available: {', '.join(list_personas())}")
        raise SystemExit(1)

    table = Table(show_header=False, box=None, padding=(0, 2))
    table.add_column(style="bold cyan")
    table.add_column()
    table.add_row("Name", persona.name)
    table.add_row("Display Name", persona.display_name or "—")
    table.add_row("Category", persona.category)
    table.add_row("Description", persona.description or "—")
    table.add_row("Role", persona.persona_role)
    table.add_row("Expertise", persona.expertise or "—")

    if persona.recommended_models:
        table.add_row("Models", ", ".join(str(m) for m in persona.recommended_models))
    if persona.tools:
        table.add_row("Tools", ", ".join(persona.tools))
    if persona.avoid_topics:
        table.add_row("Avoid", ", ".join(persona.avoid_topics))
    if persona.tags:
        table.add_row("Tags", ", ".join(persona.tags))
    if persona.test_queries:
        table.add_row("Test Queries", str(len(persona.test_queries)))

    console.print(
        Panel(
            table, title=f"[bold]{persona.display_name or persona.name}[/bold]", border_style="blue"
        )
    )


# ---------------------------------------------------------------------------
# models
# ---------------------------------------------------------------------------


@main.command()
@click.option("--provider", "-p", default=None, help="Filter by provider.")
@click.option("--tools-only", is_flag=True, help="Only models with tool support.")
@click.option("--vision-only", is_flag=True, help="Only models with vision support.")
def models(
    *,
    provider: str | None,
    tools_only: bool,
    vision_only: bool,
) -> None:
    """List registered model profiles.

    Examples:

        talk-box models

        talk-box models --provider anthropic --tools-only
    """
    import talk_box as tb

    console = Console()
    profiles = tb.list_models(
        provider=provider,
        supports_tools=True if tools_only else None,
        supports_vision=True if vision_only else None,
    )

    if not profiles:
        console.print("[yellow]No matching model profiles found.[/yellow]")
        return

    table = Table(title="Model Profiles", border_style="blue")
    table.add_column("Provider", style="bold cyan")
    table.add_column("Model", style="green")
    table.add_column("Context", justify="right")
    table.add_column("Tools")
    table.add_column("Vision")
    table.add_column("Cost")

    for p in profiles:
        ctx = f"{p.context_window:,}" if p.context_window else "—"
        tools = _bool_icon(p.supports_tools)
        vision = _bool_icon(p.supports_vision)
        cost = p.cost_tier.value if p.cost_tier else "—"
        table.add_row(p.provider, p.model, ctx, tools, vision, cost)

    console.print(table)
    console.print(f"\n[dim]{len(profiles)} model(s)[/dim]")


# ---------------------------------------------------------------------------
# config
# ---------------------------------------------------------------------------


@main.group()
def config() -> None:
    """View and manage Talk Box configuration.

    Configuration is resolved from multiple layers (last wins):
    built-in defaults → global config → project config → profile → env vars → CLI flags.
    """


@config.command("show")
def config_show() -> None:
    """Show the resolved configuration.

    Displays the merged result of all config layers with the source
    of each setting.
    """
    from talk_box.config import global_config_dir, load_config, project_config_path

    console = Console()
    cfg = load_config()

    table = Table(show_header=False, box=None, padding=(0, 2))
    table.add_column(style="bold cyan")
    table.add_column()

    table.add_row("Model", cfg.default_model or "[dim]not set[/dim]")
    table.add_row("Persona", cfg.default_persona or "[dim]not set[/dim]")
    table.add_row(
        "Temperature", str(cfg.temperature) if cfg.temperature is not None else "[dim]not set[/dim]"
    )
    table.add_row("Guardrails", ", ".join(cfg.guardrails) if cfg.guardrails else "[dim]none[/dim]")
    table.add_row("Allow Cloud", str(cfg.allow_cloud))
    table.add_row("TUI Mode", cfg.mode.value)
    table.add_row("Profiles", ", ".join(cfg.profiles.keys()) if cfg.profiles else "[dim]none[/dim]")
    table.add_row("Trusted Cmds", ", ".join(cfg.trusted_commands))

    project = project_config_path()
    table.add_row("Project Config", str(project) if project else "[dim]none found[/dim]")
    table.add_row("Global Config", str(global_config_dir() / "config.yml"))

    console.print(Panel(table, title="[bold]Configuration[/bold]", border_style="blue"))


@config.command("get")
@click.argument("key")
def config_get(key: str) -> None:
    """Get the value of a configuration key.

    Examples:

        talk-box config get default_model

        talk-box config get allow_cloud
    """
    from talk_box.config import load_config

    console = Console()
    cfg = load_config()

    # Map key names to config attributes
    key_map = {
        "default_model": cfg.default_model,
        "model": cfg.default_model,
        "default_persona": cfg.default_persona,
        "persona": cfg.default_persona,
        "temperature": cfg.temperature,
        "guardrails": cfg.guardrails,
        "allow_cloud": cfg.allow_cloud,
        "mode": cfg.mode.value,
        "trusted_commands": cfg.trusted_commands,
    }

    if key not in key_map:
        console.print(f"[red]Unknown key: {key}[/red]")
        console.print(f"Available: {', '.join(sorted(key_map.keys()))}")
        raise SystemExit(1)

    value = key_map[key]
    if isinstance(value, list):
        console.print(", ".join(str(v) for v in value))
    elif value is None:
        console.print("[dim]not set[/dim]")
    else:
        console.print(str(value))


@config.command("set")
@click.argument("key")
@click.argument("value")
def config_set(key: str, value: str) -> None:
    """Set a configuration value in the project config.

    Creates or updates ``talk-box.yml`` in the current directory.

    Examples:

        talk-box config set default_model ollama:llama3.3

        talk-box config set allow_cloud false
    """
    from pathlib import Path

    from yaml12 import read_yaml, write_yaml

    console = Console()
    config_path = Path("talk-box.yml")

    # Load existing or start fresh
    if config_path.is_file():
        try:
            data = read_yaml(config_path)
            if not isinstance(data, dict):
                data = {}
        except (OSError, ValueError):
            data = {}
    else:
        data = {}

    # Type coercion for known keys
    bool_keys = {"allow_cloud"}
    float_keys = {"temperature"}
    list_keys = {"guardrails", "trusted_commands"}

    if key in bool_keys:
        data[key] = value.lower() in ("true", "1", "yes")
    elif key in float_keys:
        try:
            data[key] = float(value)
        except ValueError:
            console.print(f"[red]Invalid float value: {value}[/red]")
            raise SystemExit(1)
    elif key in list_keys:
        data[key] = [v.strip() for v in value.split(",")]
    else:
        data[key] = value

    write_yaml(data, config_path)
    console.print(f"[green]Set {key} = {data[key]}[/green] in {config_path}")


@config.command("list")
def config_list() -> None:
    """List all named profiles.

    Profiles are stored in ``~/.config/talk-box/profiles/``.
    """
    from talk_box.config import list_profiles, load_profile

    console = Console()
    names = list_profiles()

    if not names:
        console.print("[dim]No profiles found.[/dim]")
        console.print("Create one with: talk-box config set-profile NAME --model MODEL")
        return

    table = Table(title="Named Profiles", border_style="blue")
    table.add_column("Name", style="bold cyan")
    table.add_column("Model", style="green")
    table.add_column("Persona")
    table.add_column("Temp", justify="right")

    for name in names:
        try:
            p = load_profile(name)
            table.add_row(
                name,
                p.model or "—",
                p.persona or "—",
                str(p.temperature) if p.temperature is not None else "—",
            )
        except Exception:
            table.add_row(name, "[red]error[/red]", "", "")

    console.print(table)
