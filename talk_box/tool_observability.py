"""
Talk Box Tool Observability and Debugging

Comprehensive monitoring, logging, and debugging utilities for Talk Box tools.
Provides insights into tool performance, usage patterns, and failure analysis.
"""

from __future__ import annotations

import json
import logging
import threading
from collections import defaultdict, deque
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set
from weakref import WeakSet

from .tools import ToolStatus


class ObservabilityLevel(Enum):
    """Different levels of observability detail."""

    NONE = "none"  # No observability
    BASIC = "basic"  # Basic metrics and errors
    DETAILED = "detailed"  # Detailed timing and parameters
    DEBUG = "debug"  # Everything including internal state


@dataclass
class ToolExecution:
    """Complete record of a tool execution."""

    tool_name: str
    execution_id: str
    start_time: datetime
    end_time: Optional[datetime] = None
    duration_ms: Optional[float] = None
    status: ToolStatus = ToolStatus.PENDING
    parameters: Dict[str, Any] = field(default_factory=dict)
    result_summary: Optional[str] = None
    error_message: Optional[str] = None
    error_type: Optional[str] = None
    memory_usage_mb: Optional[float] = None
    conversation_id: Optional[str] = None
    user_id: Optional[str] = None
    tags: Set[str] = field(default_factory=set)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        data = asdict(self)
        # Convert datetime objects to ISO format
        if self.start_time:
            data["start_time"] = self.start_time.isoformat()
        if self.end_time:
            data["end_time"] = self.end_time.isoformat()
        # Convert set to list
        data["tags"] = list(self.tags)
        return data


@dataclass
class ToolMetrics:
    """Aggregated metrics for a tool."""

    tool_name: str
    total_executions: int = 0
    successful_executions: int = 0
    failed_executions: int = 0
    avg_duration_ms: float = 0.0
    min_duration_ms: float = float("inf")
    max_duration_ms: float = 0.0
    last_execution: Optional[datetime] = None
    error_rate: float = 0.0
    most_common_errors: List[str] = field(default_factory=list)

    def success_rate(self) -> float:
        """Calculate success rate as percentage."""
        if self.total_executions == 0:
            return 0.0
        return (self.successful_executions / self.total_executions) * 100


class ToolObserver:
    """
    Observes and records tool executions for monitoring and debugging.

    Features:
    - Real-time execution tracking
    - Performance metrics collection
    - Error pattern analysis
    - Memory usage monitoring
    - Configurable retention policies
    """

    def __init__(
        self,
        level: ObservabilityLevel = ObservabilityLevel.BASIC,
        max_executions: int = 1000,
        retention_days: int = 7,
        enable_memory_profiling: bool = False,
    ):
        self.level = level
        self.max_executions = max_executions
        self.retention_days = retention_days
        self.enable_memory_profiling = enable_memory_profiling

        # Storage
        self._executions: deque = deque(maxlen=max_executions)
        self._metrics: Dict[str, ToolMetrics] = {}
        self._active_executions: Dict[str, ToolExecution] = {}

        # Thread safety
        self._lock = threading.RLock()

        # Listeners for real-time monitoring
        self._listeners: WeakSet = WeakSet()

        # Setup logging
        self.logger = logging.getLogger(f"{__name__}.ToolObserver")

    def start_execution(
        self,
        tool_name: str,
        execution_id: str,
        parameters: Dict[str, Any],
        conversation_id: Optional[str] = None,
        user_id: Optional[str] = None,
        tags: Optional[Set[str]] = None,
    ) -> ToolExecution:
        """Start tracking a tool execution."""
        if self.level == ObservabilityLevel.NONE:
            return None

        execution = ToolExecution(
            tool_name=tool_name,
            execution_id=execution_id,
            start_time=datetime.now(timezone.utc),
            parameters=parameters
            if self.level in [ObservabilityLevel.DETAILED, ObservabilityLevel.DEBUG]
            else {},
            conversation_id=conversation_id,
            user_id=user_id,
            tags=tags or set(),
        )

        with self._lock:
            self._active_executions[execution_id] = execution

        if self.level == ObservabilityLevel.DEBUG:
            self.logger.debug(f"Started execution {execution_id} for tool {tool_name}")

        return execution

    def finish_execution(
        self,
        execution_id: str,
        status: ToolStatus,
        result_summary: Optional[str] = None,
        error_message: Optional[str] = None,
        error_type: Optional[str] = None,
        memory_usage_mb: Optional[float] = None,
    ) -> Optional[ToolExecution]:
        """Finish tracking a tool execution."""
        if self.level == ObservabilityLevel.NONE:
            return None

        with self._lock:
            execution = self._active_executions.pop(execution_id, None)

        if not execution:
            self.logger.warning(f"No active execution found for ID: {execution_id}")
            return None

        # Update execution record
        execution.end_time = datetime.now(timezone.utc)
        execution.duration_ms = (execution.end_time - execution.start_time).total_seconds() * 1000
        execution.status = status
        execution.result_summary = (
            result_summary
            if self.level in [ObservabilityLevel.DETAILED, ObservabilityLevel.DEBUG]
            else None
        )
        execution.error_message = error_message
        execution.error_type = error_type
        execution.memory_usage_mb = memory_usage_mb

        # Store execution
        self._executions.append(execution)

        # Update metrics
        self._update_metrics(execution)

        # Notify listeners
        self._notify_listeners("execution_finished", execution)

        if self.level == ObservabilityLevel.DEBUG:  # pragma: no cover
            self.logger.debug(  # pragma: no cover
                f"Finished execution {execution_id}: {status.value} in {execution.duration_ms:.2f}ms"
            )
        elif status == ToolStatus.ERROR:
            self.logger.error(f"Tool {execution.tool_name} failed: {error_message}")

        return execution

    def _update_metrics(self, execution: ToolExecution) -> None:
        """Update aggregated metrics for a tool."""
        tool_name = execution.tool_name

        if tool_name not in self._metrics:
            self._metrics[tool_name] = ToolMetrics(tool_name=tool_name)

        metrics = self._metrics[tool_name]
        metrics.total_executions += 1
        metrics.last_execution = execution.end_time

        if execution.status == ToolStatus.SUCCESS:
            metrics.successful_executions += 1
        elif execution.status == ToolStatus.ERROR:
            metrics.failed_executions += 1

        # Update duration stats
        if execution.duration_ms is not None:
            metrics.min_duration_ms = min(metrics.min_duration_ms, execution.duration_ms)
            metrics.max_duration_ms = max(metrics.max_duration_ms, execution.duration_ms)

            # Calculate rolling average
            total_duration = metrics.avg_duration_ms * (metrics.total_executions - 1)
            metrics.avg_duration_ms = (
                total_duration + execution.duration_ms
            ) / metrics.total_executions

        # Update error rate
        metrics.error_rate = (metrics.failed_executions / metrics.total_executions) * 100

    def _notify_listeners(self, event: str, data: Any) -> None:
        """Notify registered listeners of events."""
        for listener in self._listeners:
            try:
                listener(event, data)
            except Exception as e:  # pragma: no cover
                self.logger.error(f"Listener notification failed: {e}")  # pragma: no cover

    def get_metrics(self, tool_name: Optional[str] = None) -> Dict[str, ToolMetrics]:
        """Get metrics for tools."""
        with self._lock:
            if tool_name:
                return (
                    {tool_name: self._metrics.get(tool_name)} if tool_name in self._metrics else {}
                )
            return dict(self._metrics)

    def get_executions(
        self,
        tool_name: Optional[str] = None,
        status: Optional[ToolStatus] = None,
        limit: Optional[int] = None,
        since: Optional[datetime] = None,
    ) -> List[ToolExecution]:
        """Get execution records with filtering."""
        with self._lock:
            executions = list(self._executions)

        # Apply filters
        if tool_name:
            executions = [e for e in executions if e.tool_name == tool_name]
        if status:
            executions = [e for e in executions if e.status == status]
        if since:
            executions = [e for e in executions if e.start_time >= since]

        # Sort by start time (newest first)
        executions.sort(key=lambda e: e.start_time, reverse=True)

        if limit:
            executions = executions[:limit]

        return executions

    def get_error_analysis(self, tool_name: Optional[str] = None) -> Dict[str, Any]:
        """Get detailed error analysis."""
        executions = self.get_executions(tool_name=tool_name, status=ToolStatus.ERROR)

        if not executions:
            return {"total_errors": 0, "error_types": {}, "common_patterns": []}

        # Analyze error patterns
        error_types = defaultdict(int)
        error_messages = defaultdict(int)

        for execution in executions:
            if execution.error_type:
                error_types[execution.error_type] += 1
            if execution.error_message:
                error_messages[execution.error_message] += 1

        # Find common patterns
        common_patterns = []
        for msg, count in sorted(error_messages.items(), key=lambda x: x[1], reverse=True)[:5]:
            common_patterns.append({"message": msg, "count": count})

        return {
            "total_errors": len(executions),
            "error_types": dict(error_types),
            "common_patterns": common_patterns,
            "recent_errors": [
                {
                    "tool": e.tool_name,
                    "time": e.start_time.isoformat(),
                    "error": e.error_message,
                    "type": e.error_type,
                }
                for e in executions[:10]
            ],
        }

    def get_performance_summary(self) -> Dict[str, Any]:
        """Get overall performance summary."""
        with self._lock:
            total_executions = sum(m.total_executions for m in self._metrics.values())
            successful_executions = sum(m.successful_executions for m in self._metrics.values())
            failed_executions = sum(m.failed_executions for m in self._metrics.values())

        if total_executions == 0:
            return {
                "total_executions": 0,
                "success_rate": 0.0,
                "avg_duration_ms": 0.0,
                "tools_used": 0,
                "most_used_tools": [],
                "slowest_tools": [],
                "error_rate": 0.0,
            }

        # Calculate averages
        avg_duration = (
            sum(m.avg_duration_ms * m.total_executions for m in self._metrics.values())
            / total_executions
        )

        # Find top tools by usage
        most_used = sorted(self._metrics.values(), key=lambda m: m.total_executions, reverse=True)[
            :5
        ]

        # Find slowest tools
        slowest = sorted(self._metrics.values(), key=lambda m: m.avg_duration_ms, reverse=True)[:5]

        return {
            "total_executions": total_executions,
            "success_rate": (successful_executions / total_executions) * 100,
            "error_rate": (failed_executions / total_executions) * 100,
            "avg_duration_ms": avg_duration,
            "tools_used": len(self._metrics),
            "most_used_tools": [
                {"name": m.tool_name, "count": m.total_executions} for m in most_used
            ],
            "slowest_tools": [
                {"name": m.tool_name, "avg_duration_ms": m.avg_duration_ms} for m in slowest
            ],
        }

    def cleanup_old_data(self) -> None:
        """Remove old execution data based on retention policy."""
        cutoff = datetime.now(timezone.utc) - timedelta(days=self.retention_days)

        with self._lock:
            # Filter executions
            old_executions = deque()
            for execution in self._executions:
                if execution.start_time >= cutoff:
                    old_executions.append(execution)

            removed_count = len(self._executions) - len(old_executions)
            self._executions = old_executions

        if removed_count > 0:  # pragma: no cover
            self.logger.info(
                f"Cleaned up {removed_count} old execution records"
            )  # pragma: no cover

    def add_listener(self, callback: Callable[[str, Any], None]) -> None:
        """Add a listener for real-time events."""
        self._listeners.add(callback)

    def export_data(self, format: str = "json") -> str:
        """Export observability data for external analysis."""
        if format != "json":
            raise ValueError("Only JSON format is currently supported")

        data = {
            "summary": self.get_performance_summary(),
            "metrics": {name: asdict(metrics) for name, metrics in self.get_metrics().items()},
            "recent_executions": [e.to_dict() for e in self.get_executions(limit=100)],
            "error_analysis": self.get_error_analysis(),
            "exported_at": datetime.now(timezone.utc).isoformat(),
        }

        return json.dumps(data, indent=2, default=str)


@contextmanager
def tool_execution_context(
    observer: ToolObserver, tool_name: str, execution_id: str, parameters: Dict[str, Any], **kwargs
):
    """Context manager for tracking tool executions."""
    execution = observer.start_execution(tool_name, execution_id, parameters, **kwargs)

    try:
        yield execution
        if execution:
            observer.finish_execution(execution_id, ToolStatus.SUCCESS)
    except Exception as e:
        if execution:
            observer.finish_execution(
                execution_id, ToolStatus.ERROR, error_message=str(e), error_type=type(e).__name__
            )
        raise


# Global observer instance
_global_observer: Optional[ToolObserver] = None


def get_global_observer() -> ToolObserver:
    """Get the global tool observer instance."""
    global _global_observer
    if _global_observer is None:  # pragma: no cover
        _global_observer = ToolObserver()  # pragma: no cover
    return _global_observer


def configure_observability(
    level: ObservabilityLevel = ObservabilityLevel.BASIC, **kwargs
) -> ToolObserver:
    """Configure global observability settings."""
    global _global_observer
    _global_observer = ToolObserver(level=level, **kwargs)
    return _global_observer


def reset_observability() -> None:
    """Reset observability data (useful for testing)."""
    global _global_observer
    if _global_observer:  # pragma: no cover
        _global_observer._executions.clear()  # pragma: no cover
        _global_observer._metrics.clear()  # pragma: no cover
        _global_observer._active_executions.clear()  # pragma: no cover
