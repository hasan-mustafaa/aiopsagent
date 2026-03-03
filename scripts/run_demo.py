"""End-to-end demo for LogSentry Agent — warm-up → fault → detect → remediate."""

from __future__ import annotations

import argparse
import logging
import os
import time
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from rich.box import ROUNDED
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.rule import Rule
from rich.table import Table

from src.main import build_pipeline, load_config

console = Console()
logger = logging.getLogger(__name__)

_FAULT_TYPES = ["crash", "latency_spike", "connection_failure", "memory_leak", "oom"]
_SERVICES = [
    "transaction-validator",
    "fraud-check-service",
    "document-processor",
    "title-search-service",
]

_FAULT_DESCRIPTIONS = {
    "crash":              "Process stops responding — metrics collapse, error rate spikes",
    "latency_spike":      "P99 latency multiplies 15x — upstream callers time out",
    "connection_failure": "Listener goes down — ECONNREFUSED floods error logs",
    "memory_leak":        "Heap grows unbounded — GC pressure elevates CPU and latency",
    "oom":                "OOM killer fires — memory pegged at ceiling, throughput near zero",
}


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="LogSentry Agent — end-to-end AIOps demo",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python scripts/run_demo.py\n"
            "  python scripts/run_demo.py --fault crash --service fraud-check-service\n"
            "  python scripts/run_demo.py --duration 120 --dry-run\n"
        ),
    )
    parser.add_argument("--duration", type=int, default=60, help="Run time in seconds")
    parser.add_argument("--fault", dest="fault_type", default="latency_spike", choices=_FAULT_TYPES, help="Fault type")
    parser.add_argument("--service", dest="target_service", default="title-search-service", choices=_SERVICES, help="Target service")
    parser.add_argument("--dry-run", action="store_true", help="Log actions without executing")
    parser.add_argument("--no-llm", action="store_true", help="Detection only (no agent)")
    parser.add_argument("--config", type=Path, default=Path("config/config.yaml"), help="Config path")
    return parser.parse_args()


def print_banner() -> None:
    """Print ASCII banner."""
    lines = [
        "  ██╗      ██████╗  ██████╗ ███████╗███████╗███╗   ██╗████████╗██████╗ ██╗   ██╗",
        "  ██║     ██╔═══██╗██╔════╝ ██╔════╝██╔════╝████╗  ██║╚══██╔══╝██╔══██╗╚██╗ ██╔╝",
        "  ██║     ██║   ██║██║  ███╗███████╗█████╗  ██╔██╗ ██║   ██║   ██████╔╝ ╚████╔╝ ",
        "  ██║     ██║   ██║██║   ██║╚════██║██╔══╝  ██║╚██╗██║   ██║   ██╔══██╗  ╚██╔╝  ",
        "  ███████╗╚██████╔╝╚██████╔╝███████║███████╗██║ ╚████║   ██║   ██║  ██║   ██║   ",
        "  ╚══════╝ ╚═════╝  ╚═════╝ ╚══════╝╚══════╝╚═╝  ╚═══╝   ╚═╝   ╚═╝  ╚═╝   ╚═╝  ",
    ]
    console.print()
    for line in lines:
        console.print(line, style="bold cyan")
    console.print()
    console.print("  AI-Powered AIOps  |  Detect · Diagnose · Remediate  |  v1.0", style="dim")
    console.print()


def print_pipeline_summary(results: dict[str, Any]) -> None:
    """Print results table."""
    console.print(Rule("[bold white]Demo Summary", style="cyan"))
    console.print()

    table = Table(show_header=True, header_style="bold cyan", box=ROUNDED, padding=(0, 1), min_width=72)
    table.add_column("Stage", style="bold", width=22, no_wrap=True)
    table.add_column("Result", min_width=48)

    fault = results.get("fault_injected", {})
    table.add_row(
        "Fault Injected",
        f"[bold red]{fault.get('type', '-')}[/bold red] on [yellow]{fault.get('service', '-')}[/yellow]\n"
        f"[dim]{_FAULT_DESCRIPTIONS.get(fault.get('type', ''), '')}[/dim]",
    )

    ttd = results.get("time_to_detection_s")
    score = results.get("anomaly_score")
    if ttd is not None:
        table.add_row("Anomaly Detection", f"[bold red]DETECTED[/bold red] in [bold]{ttd}s[/bold]  (score={score:.3f})")
    else:
        table.add_row("Anomaly Detection", "[dim]Not detected[/dim]")

    steps = results.get("agent_steps", 0)
    actions = results.get("actions_taken", [])
    if steps:
        actions_str = ", ".join(f"[cyan]{a}[/cyan]" for a in actions) if actions else "[dim]none[/dim]"
        table.add_row("Agent Reasoning", f"{steps} step(s) — {actions_str}")
    else:
        table.add_row("Agent Reasoning", "[dim]Skipped (--no-llm or unavailable)[/dim]")

    status = results.get("resolution_status", "unresolved")
    if status == "resolved":
        status_str = "[bold green]RESOLVED[/bold green]"
    elif status == "escalated":
        status_str = "[bold orange1]ESCALATED[/bold orange1]"
    elif "detection-only" in str(status):
        status_str = "[dim]Detection only[/dim]"
    else:
        status_str = "[bold red]UNRESOLVED[/bold red]"
    table.add_row("Resolution", status_str)

    rca = results.get("rca_summary")
    if rca:
        rca_display = rca[:140] + "..." if len(rca) > 140 else rca
        table.add_row("RCA Summary", f"[italic]{rca_display}[/italic]")

    console.print(table)
    console.print()
    console.print("  [dim]Open dashboard:[/dim]  [bold cyan]streamlit run src/dashboard/app.py[/bold cyan]")
    console.print()


def run_demo(
    duration: int = 60,
    fault_type: str = "latency_spike",
    target_service: str = "title-search-service",
    dry_run: bool = False,
    no_llm: bool = False,
    config_path: Path = Path("config/config.yaml"),
) -> dict[str, Any]:
    """Run the full demo pipeline."""
    load_dotenv()

    if no_llm:
        os.environ.pop("OPENAI_API_KEY", None)
        os.environ.pop("ANTHROPIC_API_KEY", None)

    config = load_config(config_path)
    config["simulator"]["metrics_interval_seconds"] = 2
    pipeline = build_pipeline(config, dry_run=dry_run)

    results: dict[str, Any] = {
        "fault_injected":      {"service": target_service, "type": fault_type},
        "detected_at":         None,
        "time_to_detection_s": None,
        "anomaly_score":       None,
        "agent_steps":         0,
        "actions_taken":       [],
        "resolution_status":   "unresolved",
        "rca_summary":         None,
    }

    console.print(Rule("[bold yellow]Phase 1 — Warm-Up", style="yellow"))
    console.print("Training ML models on baseline data...\n", style="dim")

    with Progress(SpinnerColumn(), TextColumn("  [progress.description]{task.description}"), TimeElapsedColumn(), console=console, transient=False) as progress:
        task = progress.add_task("Training...", total=None)
        pipeline._warm_up()
        for svc in pipeline._services:
            pipeline._dash["metric_history"][svc] = []
        pipeline._dash["anomaly_events"] = []
        progress.update(task, description="[green]Complete")

    console.print()

    console.print(Rule("[bold red]Phase 2 — Fault Injection", style="red"))
    fault_desc = _FAULT_DESCRIPTIONS.get(fault_type, "")
    console.print(f"  [bold red]{fault_type}[/bold red] on [yellow]{target_service}[/yellow]\n  [dim]{fault_desc}[/dim]\n")

    fault_inject_time = time.time()
    pipeline.inject_fault(target_service, fault_type, duration=float(duration))
    console.print(f"  [bold red]Fault active.[/bold red] Monitoring {duration}s...\n")

    console.print(Rule("[bold cyan]Phase 3 — Detection Loop", style="cyan"))
    console.print(f"  Ensemble detector scanning every {config['simulator']['metrics_interval_seconds']}s.\n", style="dim")

    tick_interval = config["simulator"]["metrics_interval_seconds"]
    end_time = time.time() + duration
    agent_done = False

    with Progress(SpinnerColumn(), TextColumn("  [progress.description]{task.description}"), TimeElapsedColumn(), console=console, transient=True) as progress:
        task = progress.add_task("Scanning...", total=None)

        while time.time() < end_time and not agent_done:
            pipeline._tick()

            target_anomalies = [e for e in pipeline._dash.get("anomaly_events", []) if e.get("service") == target_service]

            if target_anomalies and results["detected_at"] is None:
                latest = target_anomalies[-1]
                results["detected_at"] = time.time()
                results["time_to_detection_s"] = round(results["detected_at"] - fault_inject_time, 1)
                results["anomaly_score"] = round(latest.get("anomaly_score", 0.0), 3)
                triggered = latest.get("triggered_metrics", [])
                progress.update(task, description=f"[bold red]ANOMALY[/bold red] (score={results['anomaly_score']:.3f}, triggered={triggered})")

            agent_results = pipeline._dash.get("agent_results", [])
            if agent_results and not agent_done:
                latest_agent = agent_results[-1]
                trace = latest_agent.get("reasoning_trace", [])
                results["agent_steps"] = len(trace)
                results["actions_taken"] = [s["action"]["action"] for s in trace if s.get("action")]
                results["resolution_status"] = (
                    "resolved" if latest_agent.get("resolved")
                    else "escalated" if latest_agent.get("escalated")
                    else "in_progress"
                )
                rca = latest_agent.get("rca_report") or {}
                results["rca_summary"] = rca.get("summary") or rca.get("root_cause_type", "") or "See trace."
                agent_done = True
                progress.update(task, description="[green]Agent complete")

            if not pipeline._fault_injector.active_faults():
                if results["resolution_status"] == "unresolved":
                    results["resolution_status"] = "resolved"
                break

            time.sleep(tick_interval)

    if no_llm and results["detected_at"] is not None:
        results["resolution_status"] = "detection-only"

    pipeline._save_dashboard_state()
    console.print()
    return results


def main() -> None:
    """Entry point."""
    logging.basicConfig(level=logging.WARNING, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s", datefmt="%H:%M:%S")

    args = parse_args()
    print_banner()

    mode_tags = []
    if args.dry_run:
        mode_tags.append("[dim]dry-run[/dim]")
    if args.no_llm:
        mode_tags.append("[dim]no-llm[/dim]")
    mode_str = " | ".join(mode_tags) if mode_tags else "[dim]live[/dim]"

    config_panel = (
        f"  Fault     [bold red]{args.fault_type}[/bold red]\n"
        f"  Service   [yellow]{args.target_service}[/yellow]\n"
        f"  Duration  {args.duration}s\n"
        f"  Mode      {mode_str}"
    )
    console.print(Panel(config_panel, title="[bold]Run Configuration", border_style="cyan", expand=False))
    console.print()

    results = run_demo(
        duration=args.duration,
        fault_type=args.fault_type,
        target_service=args.target_service,
        dry_run=args.dry_run,
        no_llm=args.no_llm,
        config_path=args.config,
    )

    print_pipeline_summary(results)


if __name__ == "__main__":
    main()
