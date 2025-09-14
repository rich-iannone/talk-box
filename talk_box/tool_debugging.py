"""
Talk Box Tool Debugging Utilities

Easy-to-use debugging interfaces for analyzing tool performance,
troubleshooting issues, and monitoring tool behavior in real-time.
"""

from datetime import datetime
from typing import Any, Optional

from rich.console import Console
from rich.table import Table

from .tool_observability import ObservabilityLevel, ToolObserver, get_global_observer
from .tools import ToolStatus

console = Console()


class ToolDebugger:
    """
    High-level debugging interface for Talk Box tools.

    Provides easy methods to inspect tool performance, analyze errors,
    and monitor tool execution in real-time.
    """

    def __init__(self, observer: Optional[ToolObserver] = None):
        self.observer = observer or get_global_observer()

    def show_performance_dashboard(self) -> None:
        """Display a comprehensive performance dashboard."""
        summary = self.observer.get_performance_summary()

        console.print(
            "\n[bold blue]🔧 Talk Box Tool Performance Dashboard[/bold blue]", justify="center"
        )
        console.print("=" * 60)

        # Overall stats
        stats_table = Table(title="Overall Statistics")
        stats_table.add_column("Metric", style="cyan")
        stats_table.add_column("Value", style="green")

        stats_table.add_row("Total Executions", str(summary["total_executions"]))
        stats_table.add_row("Success Rate", f"{summary['success_rate']:.1f}%")
        stats_table.add_row("Error Rate", f"{summary['error_rate']:.1f}%")
        stats_table.add_row("Average Duration", f"{summary['avg_duration_ms']:.1f}ms")
        stats_table.add_row("Tools Used", str(summary["tools_used"]))

        console.print(stats_table)

        # Most used tools
        if summary["most_used_tools"]:
            usage_table = Table(title="Most Used Tools")
            usage_table.add_column("Tool Name", style="cyan")
            usage_table.add_column("Executions", style="green")

            for tool in summary["most_used_tools"]:
                usage_table.add_row(tool["name"], str(tool["count"]))

            console.print(usage_table)

        # Slowest tools
        if summary["slowest_tools"]:
            slow_table = Table(title="Slowest Tools (Average Duration)")
            slow_table.add_column("Tool Name", style="cyan")
            slow_table.add_column("Avg Duration (ms)", style="yellow")

            for tool in summary["slowest_tools"]:
                slow_table.add_row(tool["name"], f"{tool['avg_duration_ms']:.1f}")

            console.print(slow_table)

    def show_tool_details(self, tool_name: str) -> None:
        """Show detailed information about a specific tool."""
        metrics = self.observer.get_metrics(tool_name)

        if not metrics:
            console.print(f"[red]No data found for tool: {tool_name}[/red]")
            return

        tool_metrics = list(metrics.values())[0]

        console.print(f"\n[bold blue]🔍 Tool Details: {tool_name}[/bold blue]")
        console.print("=" * 50)

        details_table = Table()
        details_table.add_column("Metric", style="cyan")
        details_table.add_column("Value", style="green")

        details_table.add_row("Total Executions", str(tool_metrics.total_executions))
        details_table.add_row("Success Rate", f"{tool_metrics.success_rate():.1f}%")
        details_table.add_row("Error Rate", f"{tool_metrics.error_rate:.1f}%")
        details_table.add_row("Avg Duration", f"{tool_metrics.avg_duration_ms:.1f}ms")
        details_table.add_row("Min Duration", f"{tool_metrics.min_duration_ms:.1f}ms")
        details_table.add_row("Max Duration", f"{tool_metrics.max_duration_ms:.1f}ms")

        if tool_metrics.last_execution:
            details_table.add_row(
                "Last Execution", tool_metrics.last_execution.strftime("%Y-%m-%d %H:%M:%S")
            )

        console.print(details_table)

        # Recent executions
        recent = self.observer.get_executions(tool_name=tool_name, limit=5)
        if recent:
            console.print(f"\n[bold]Recent Executions ({len(recent)} of last 5)[/bold]")
            exec_table = Table()
            exec_table.add_column("Time", style="cyan")
            exec_table.add_column("Status", style="green")
            exec_table.add_column("Duration (ms)", style="yellow")
            exec_table.add_column("Error", style="red")

            for execution in recent:
                status_color = "green" if execution.status == ToolStatus.SUCCESS else "red"
                status_text = f"[{status_color}]{execution.status.value}[/{status_color}]"
                duration = f"{execution.duration_ms:.1f}" if execution.duration_ms else "N/A"
                error = (
                    execution.error_message[:50] + "..."
                    if execution.error_message and len(execution.error_message) > 50
                    else (execution.error_message or "")
                )

                exec_table.add_row(
                    execution.start_time.strftime("%H:%M:%S"), status_text, duration, error
                )

            console.print(exec_table)

    def show_error_analysis(self, tool_name: Optional[str] = None) -> None:
        """Show detailed error analysis."""
        analysis = self.observer.get_error_analysis(tool_name)

        title = f"Error Analysis: {tool_name}" if tool_name else "Global Error Analysis"
        console.print(f"\n[bold red]🚨 {title}[/bold red]")
        console.print("=" * 50)

        if analysis["total_errors"] == 0:
            console.print("[green]No errors found! 🎉[/green]")
            return

        # Error summary
        console.print(f"[bold]Total Errors:[/bold] {analysis['total_errors']}")

        # Error types
        if analysis["error_types"]:
            console.print("\n[bold]Error Types:[/bold]")
            for error_type, count in analysis["error_types"].items():
                console.print(f"  • {error_type}: {count}")

        # Common patterns
        if analysis["common_patterns"]:
            console.print("\n[bold]Most Common Error Messages:[/bold]")
            for i, pattern in enumerate(analysis["common_patterns"][:5], 1):
                console.print(f"  {i}. [{pattern['count']}x] {pattern['message']}")

        # Recent errors
        if analysis["recent_errors"]:
            console.print("\n[bold]Recent Errors (last 10):[/bold]")
            error_table = Table()
            error_table.add_column("Time", style="cyan")
            error_table.add_column("Tool", style="yellow")
            error_table.add_column("Type", style="red")
            error_table.add_column("Message", style="red")

            for error in analysis["recent_errors"][:10]:
                time_str = datetime.fromisoformat(error["time"]).strftime("%m-%d %H:%M:%S")
                raw_msg = error.get("error")
                if raw_msg:
                    message = raw_msg[:60] + "..." if len(raw_msg) > 60 else raw_msg
                else:
                    message = "(no message)"

                error_table.add_row(
                    time_str,
                    error.get("tool", "-"),
                    error.get("type") or "Unknown",
                    message,
                )

            console.print(error_table)

    def show_live_monitoring(self, duration_seconds: int = 30) -> None:
        """Show live monitoring of tool executions."""
        console.print(f"\n[bold green]📡 Live Tool Monitoring ({duration_seconds}s)[/bold green]")
        console.print("=" * 50)
        console.print("Watching for tool executions... (Press Ctrl+C to stop)")

        executions_seen = set()

        def execution_listener(event: str, data: Any) -> None:
            if event == "execution_finished" and data.execution_id not in executions_seen:
                executions_seen.add(data.execution_id)

                status_color = "green" if data.status == ToolStatus.SUCCESS else "red"
                status_icon = "✅" if data.status == ToolStatus.SUCCESS else "❌"

                time_str = data.end_time.strftime("%H:%M:%S") if data.end_time else "unknown"
                duration = f"{data.duration_ms:.1f}ms" if data.duration_ms else "N/A"

                console.print(
                    f"[{status_color}]{status_icon} {time_str} | {data.tool_name} | {duration}[/{status_color}]"
                )

                if data.error_message:
                    console.print(f"    [red]Error: {data.error_message}[/red]")

        # Add listener
        self.observer.add_listener(execution_listener)

        try:
            import time

            time.sleep(duration_seconds)
        except KeyboardInterrupt:
            console.print("\n[yellow]Monitoring stopped.[/yellow]")

    def export_debug_report(self, filename: Optional[str] = None) -> str:
        """Export a comprehensive debug report."""
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"talk_box_debug_report_{timestamp}.json"

        report_data = self.observer.export_data()

        with open(filename, "w") as f:
            f.write(report_data)

        console.print(f"[green]Debug report exported to: {filename}[/green]")
        return filename

    def reset_all_data(self) -> None:
        """Reset all observability data (useful for testing)."""
        from .tool_observability import reset_observability

        reset_observability()
        console.print("[yellow]All observability data has been reset.[/yellow]")


# Convenience functions for quick debugging


def debug_dashboard() -> None:
    """Show the tool performance dashboard."""
    debugger = ToolDebugger()
    debugger.show_performance_dashboard()


def debug_tool(tool_name: str) -> None:
    """Show detailed information about a specific tool."""
    debugger = ToolDebugger()
    debugger.show_tool_details(tool_name)


def debug_errors(tool_name: Optional[str] = None) -> None:
    """Show error analysis for all tools or a specific tool."""
    debugger = ToolDebugger()
    debugger.show_error_analysis(tool_name)


def live_monitor(duration: int = 30) -> None:
    """Start live monitoring of tool executions."""
    debugger = ToolDebugger()
    debugger.show_live_monitoring(duration)


def export_debug_report(filename: Optional[str] = None) -> str:
    """Export a comprehensive debug report."""
    debugger = ToolDebugger()
    return debugger.export_debug_report(filename)


def configure_debug_mode(level: ObservabilityLevel = ObservabilityLevel.DEBUG) -> None:
    """Configure observability for maximum debugging detail."""
    from .tool_observability import configure_observability

    observer = configure_observability(
        level=level,
        max_executions=5000,  # Store more executions for debugging
        retention_days=14,  # Keep data longer
        enable_memory_profiling=True,
    )

    console.print(f"[green]Debug mode configured with level: {level.value}[/green]")
    console.print("Use debug_dashboard() to view tool performance")
    return observer


# Example usage patterns
def debug_example() -> None:
    """Show example debugging workflows."""
    console.print("""
[bold blue]Talk Box Tool Debugging Examples[/bold blue]

1. [green]Quick Performance Overview:[/green]
   import talk_box as tb
   tb.debug_dashboard()

2. [green]Analyze Specific Tool:[/green]
   tb.debug_tool('web_search')

3. [green]Check for Errors:[/green]
   tb.debug_errors()
   tb.debug_errors('file_operations')  # specific tool

4. [green]Live Monitoring:[/green]
   tb.live_monitor(60)  # watch for 60 seconds

5. [green]Export Debug Report:[/green]
   tb.export_debug_report()

6. [green]Enable Debug Mode:[/green]
   tb.configure_debug_mode()

7. [green]Custom Analysis:[/green]
   debugger = tb.ToolDebugger()
   observer = debugger.observer
   metrics = observer.get_metrics()
   executions = observer.get_executions(limit=100)
    """)
