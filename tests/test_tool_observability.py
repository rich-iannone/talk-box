import json
import pytest
import asyncio
from datetime import datetime, timezone, timedelta
from unittest.mock import Mock, patch
from collections import deque
import threading
import time

import talk_box as tb

from talk_box.tool_observability import (
    ObservabilityLevel,
    ToolExecution,
    ToolMetrics,
    ToolObserver,
    tool_execution_context,
    get_global_observer,
    configure_observability,
    reset_observability,
    _global_observer,
)
from talk_box.tools import ToolStatus


def test_tool_execution_creation():
    """Test basic ToolExecution creation."""
    start_time = datetime.now(timezone.utc)
    execution = ToolExecution(
        tool_name="test_tool",
        execution_id="exec_123",
        start_time=start_time,
        parameters={"param1": "value1"},
        tags={"tag1", "tag2"},
    )

    assert execution.tool_name == "test_tool"
    assert execution.execution_id == "exec_123"
    assert execution.start_time == start_time
    assert execution.parameters == {"param1": "value1"}
    assert execution.tags == {"tag1", "tag2"}
    assert execution.status == ToolStatus.PENDING


def test_tool_execution_to_dict():
    """Test ToolExecution serialization to dictionary."""
    start_time = datetime.now(timezone.utc)
    end_time = start_time + timedelta(seconds=1)

    execution = ToolExecution(
        tool_name="test_tool",
        execution_id="exec_123",
        start_time=start_time,
        end_time=end_time,
        duration_ms=1000.0,
        status=ToolStatus.SUCCESS,
        parameters={"param1": "value1"},
        result_summary="Success result",
        tags={"tag1", "tag2"},
    )

    result = execution.to_dict()

    assert result["tool_name"] == "test_tool"
    assert result["execution_id"] == "exec_123"
    assert result["start_time"] == start_time.isoformat()
    assert result["end_time"] == end_time.isoformat()
    assert result["duration_ms"] == 1000.0
    assert result["status"] == ToolStatus.SUCCESS
    assert result["parameters"] == {"param1": "value1"}
    assert result["result_summary"] == "Success result"
    assert isinstance(result["tags"], list)
    assert set(result["tags"]) == {"tag1", "tag2"}


def test_tool_execution_to_dict_none_times():
    """Test ToolExecution to_dict with None datetime values."""
    execution = ToolExecution(
        tool_name="test_tool",
        execution_id="exec_123",
        start_time=datetime.now(timezone.utc),
        end_time=None,
    )

    result = execution.to_dict()
    assert result["end_time"] is None
    assert isinstance(result["start_time"], str)


def test_tool_metrics_creation():
    """Test ToolMetrics initialization."""
    metrics = ToolMetrics(tool_name="test_tool")

    assert metrics.tool_name == "test_tool"
    assert metrics.total_executions == 0
    assert metrics.successful_executions == 0
    assert metrics.failed_executions == 0
    assert metrics.avg_duration_ms == 0.0
    assert metrics.min_duration_ms == float("inf")
    assert metrics.max_duration_ms == 0.0
    assert metrics.last_execution is None
    assert metrics.error_rate == 0.0
    assert metrics.most_common_errors == []


def test_tool_metrics_success_rate_zero_executions():
    """Test success rate calculation with zero executions."""
    metrics = ToolMetrics(tool_name="test_tool")
    assert metrics.success_rate() == 0.0


def test_tool_metrics_success_rate_calculation():
    """Test success rate calculation with various scenarios."""
    metrics = ToolMetrics(tool_name="test_tool")

    # 100% success
    metrics.total_executions = 10
    metrics.successful_executions = 10
    assert metrics.success_rate() == 100.0

    # 50% success
    metrics.successful_executions = 5
    assert metrics.success_rate() == 50.0

    # 0% success
    metrics.successful_executions = 0
    assert metrics.success_rate() == 0.0


def test_tool_observer_creation_defaults():
    """Test ToolObserver creation with default parameters."""
    observer = ToolObserver()

    assert observer.level == ObservabilityLevel.BASIC
    assert observer.max_executions == 1000
    assert observer.retention_days == 7
    assert observer.enable_memory_profiling is False
    assert isinstance(observer._executions, deque)
    assert observer._executions.maxlen == 1000
    assert isinstance(observer._metrics, dict)
    assert isinstance(observer._active_executions, dict)
    assert hasattr(observer._lock, "acquire")  # Check it's a lock-like object


def test_tool_observer_creation_custom_params():
    """Test ToolObserver creation with custom parameters."""
    observer = ToolObserver(
        level=ObservabilityLevel.DEBUG,
        max_executions=500,
        retention_days=14,
        enable_memory_profiling=True,
    )

    assert observer.level == ObservabilityLevel.DEBUG
    assert observer.max_executions == 500
    assert observer.retention_days == 14
    assert observer.enable_memory_profiling is True
    assert observer._executions.maxlen == 500


def test_tool_observer_start_execution_none_level():
    """Test start_execution returns None when level is NONE."""
    observer = ToolObserver(level=ObservabilityLevel.NONE)
    result = observer.start_execution("test_tool", "exec_123", {"param": "value"})
    assert result is None


def test_tool_observer_start_execution_basic_level():
    """Test start_execution with BASIC level."""
    observer = ToolObserver(level=ObservabilityLevel.BASIC)
    execution = observer.start_execution(
        "test_tool",
        "exec_123",
        {"param": "value"},
        conversation_id="conv_456",
        user_id="user_789",
        tags={"tag1"},
    )

    assert execution.tool_name == "test_tool"
    assert execution.execution_id == "exec_123"
    assert execution.parameters == {}  # Empty for BASIC level
    assert execution.conversation_id == "conv_456"
    assert execution.user_id == "user_789"
    assert execution.tags == {"tag1"}
    assert "exec_123" in observer._active_executions


def test_tool_observer_start_execution_detailed_level():
    """Test start_execution with DETAILED level."""
    observer = ToolObserver(level=ObservabilityLevel.DETAILED)
    execution = observer.start_execution("test_tool", "exec_123", {"param": "value"})

    assert execution.parameters == {"param": "value"}  # Parameters included for DETAILED


def test_tool_observer_start_execution_debug_level():
    """Test start_execution with DEBUG level and logging."""
    observer = ToolObserver(level=ObservabilityLevel.DEBUG)

    with patch.object(observer.logger, "debug") as mock_debug:
        execution = observer.start_execution("test_tool", "exec_123", {"param": "value"})

        assert execution.parameters == {"param": "value"}  # Parameters included for DEBUG
        mock_debug.assert_called_once_with("Started execution exec_123 for tool test_tool")


def test_tool_observer_finish_execution_none_level():
    """Test finish_execution returns None when level is NONE."""
    observer = ToolObserver(level=ObservabilityLevel.NONE)
    result = observer.finish_execution("exec_123", ToolStatus.SUCCESS)
    assert result is None


def test_tool_observer_finish_execution_not_found():
    """Test finish_execution with non-existent execution ID."""
    observer = ToolObserver()

    with patch.object(observer.logger, "warning") as mock_warning:
        result = observer.finish_execution("nonexistent", ToolStatus.SUCCESS)

        assert result is None
        mock_warning.assert_called_once_with("No active execution found for ID: nonexistent")


def test_tool_observer_finish_execution_success():
    """Test successful execution finish."""
    observer = ToolObserver(level=ObservabilityLevel.DETAILED)

    # Start execution
    start_time = datetime.now(timezone.utc)
    execution = observer.start_execution("test_tool", "exec_123", {"param": "value"})

    # Mock the listeners to test notification
    mock_listener = Mock()
    observer.add_listener(mock_listener)

    # Finish execution
    with patch.object(observer.logger, "debug") as mock_debug:
        result = observer.finish_execution(
            "exec_123",
            ToolStatus.SUCCESS,
            result_summary="Task completed",
            memory_usage_mb=50.0,
        )

    assert result.status == ToolStatus.SUCCESS
    assert result.result_summary == "Task completed"
    assert result.memory_usage_mb == 50.0
    assert result.end_time is not None
    assert result.duration_ms > 0
    assert "exec_123" not in observer._active_executions
    assert len(observer._executions) == 1

    # Check metrics updated
    assert "test_tool" in observer._metrics
    metrics = observer._metrics["test_tool"]
    assert metrics.total_executions == 1
    assert metrics.successful_executions == 1
    assert metrics.failed_executions == 0

    # Check listener was notified
    mock_listener.assert_called_once_with("execution_finished", result)


def test_tool_observer_finish_execution_error():
    """Test error execution finish with logging."""
    observer = ToolObserver()

    # Start execution
    execution = observer.start_execution("test_tool", "exec_123", {"param": "value"})

    # Finish with error
    with patch.object(observer.logger, "error") as mock_error:
        result = observer.finish_execution(
            "exec_123",
            ToolStatus.ERROR,
            error_message="Something went wrong",
            error_type="ValueError",
        )

    assert result.status == ToolStatus.ERROR
    assert result.error_message == "Something went wrong"
    assert result.error_type == "ValueError"

    # Check error logging
    mock_error.assert_called_once_with("Tool test_tool failed: Something went wrong")

    # Check metrics updated
    metrics = observer._metrics["test_tool"]
    assert metrics.total_executions == 1
    assert metrics.successful_executions == 0
    assert metrics.failed_executions == 1
    assert metrics.error_rate == 100.0

    def test_update_metrics_duration_calculations(self):
        """Test metrics duration calculations with multiple executions."""
        observer = ToolObserver()

        # Create multiple executions with different durations
        durations = [100.0, 200.0, 300.0]

        for i, duration in enumerate(durations):
            execution = observer.start_execution("test_tool", f"exec_{i}", {})
            execution.end_time = execution.start_time + timedelta(milliseconds=duration)
            execution.duration_ms = duration
            execution.status = ToolStatus.SUCCESS

            observer._update_metrics(execution)

        metrics = observer._metrics["test_tool"]
        assert metrics.total_executions == 3
        assert metrics.min_duration_ms == 100.0
        assert metrics.max_duration_ms == 300.0
        assert metrics.avg_duration_ms == 200.0  # (100 + 200 + 300) / 3

    def test_notify_listeners_error_handling(self):
        """Test listener notification with error handling."""
        observer = ToolObserver()

        # Add a failing listener
        failing_listener = Mock(side_effect=Exception("Listener failed"))
        observer.add_listener(failing_listener)

        with patch.object(observer.logger, "error") as mock_error:
            observer._notify_listeners("test_event", "test_data")

            mock_error.assert_called_once_with("Listener notification failed: Listener failed")

    def test_get_metrics_specific_tool(self):
        """Test getting metrics for a specific tool."""
        observer = ToolObserver()

        # Create execution for tool
        execution = observer.start_execution("test_tool", "exec_123", {})
        observer.finish_execution("exec_123", ToolStatus.SUCCESS)

        # Get specific tool metrics
        metrics = observer.get_metrics("test_tool")
        assert "test_tool" in metrics
        assert len(metrics) == 1

        # Get non-existent tool
        empty_metrics = observer.get_metrics("nonexistent_tool")
        assert empty_metrics == {}

    def test_get_metrics_all_tools(self):
        """Test getting metrics for all tools."""
        observer = ToolObserver()

        # Create executions for multiple tools
        for tool_name in ["tool1", "tool2", "tool3"]:
            execution = observer.start_execution(tool_name, f"exec_{tool_name}", {})
            observer.finish_execution(f"exec_{tool_name}", ToolStatus.SUCCESS)

        metrics = observer.get_metrics()
        assert len(metrics) == 3
        assert all(tool in metrics for tool in ["tool1", "tool2", "tool3"])

    def test_get_executions_no_filters(self):
        """Test getting all executions without filters."""
        observer = ToolObserver()

        # Create multiple executions
        for i in range(3):
            execution = observer.start_execution(f"tool_{i}", f"exec_{i}", {})
            observer.finish_execution(f"exec_{i}", ToolStatus.SUCCESS)

        executions = observer.get_executions()
        assert len(executions) == 3
        # Should be sorted by start_time (newest first)
        assert executions[0].tool_name == "tool_2"

    def test_get_executions_with_filters(self):
        """Test getting executions with various filters."""
        observer = ToolObserver()

        # Create executions with different tools and statuses
        tools_and_statuses = [
            ("tool1", ToolStatus.SUCCESS),
            ("tool1", ToolStatus.ERROR),
            ("tool2", ToolStatus.SUCCESS),
            ("tool2", ToolStatus.ERROR),
        ]

        start_time = datetime.now(timezone.utc)

        for i, (tool, status) in enumerate(tools_and_statuses):
            execution = observer.start_execution(tool, f"exec_{i}", {})
            observer.finish_execution(f"exec_{i}", status)

        # Filter by tool name
        tool1_executions = observer.get_executions(tool_name="tool1")
        assert len(tool1_executions) == 2
        assert all(e.tool_name == "tool1" for e in tool1_executions)

        # Filter by status
        error_executions = observer.get_executions(status=ToolStatus.ERROR)
        assert len(error_executions) == 2
        assert all(e.status == ToolStatus.ERROR for e in error_executions)

        # Filter by limit
        limited_executions = observer.get_executions(limit=2)
        assert len(limited_executions) == 2

        # Filter by since (should return all since they're recent)
        since_executions = observer.get_executions(since=start_time - timedelta(minutes=1))
        assert len(since_executions) == 4

    def test_get_error_analysis_no_errors(self):
        """Test error analysis with no errors."""
        observer = ToolObserver()

        analysis = observer.get_error_analysis()
        assert analysis["total_errors"] == 0
        assert analysis["error_types"] == {}
        assert analysis["common_patterns"] == []

    def test_get_error_analysis_with_errors(self):
        """Test error analysis with multiple errors."""
        observer = ToolObserver()

        # Create errors with different types and messages
        error_data = [
            ("ValueError", "Invalid input"),
            ("ValueError", "Invalid input"),  # Duplicate for pattern analysis
            ("TypeError", "Wrong type"),
            ("ConnectionError", "Network failed"),
        ]

        for i, (error_type, error_msg) in enumerate(error_data):
            execution = observer.start_execution("test_tool", f"exec_{i}", {})
            observer.finish_execution(
                f"exec_{i}", ToolStatus.ERROR, error_message=error_msg, error_type=error_type
            )

        analysis = observer.get_error_analysis()

        assert analysis["total_errors"] == 4
        assert analysis["error_types"]["ValueError"] == 2
        assert analysis["error_types"]["TypeError"] == 1
        assert analysis["error_types"]["ConnectionError"] == 1

        # Check common patterns (most frequent first)
        patterns = analysis["common_patterns"]
        assert len(patterns) > 0
        assert patterns[0]["message"] == "Invalid input"
        assert patterns[0]["count"] == 2

        # Check recent errors
        recent = analysis["recent_errors"]
        assert len(recent) == 4
        assert all("tool" in err and "time" in err for err in recent)

    def test_get_performance_summary_no_data(self):
        """Test performance summary with no data."""
        observer = ToolObserver()

        summary = observer.get_performance_summary()

        assert summary["total_executions"] == 0
        assert summary["success_rate"] == 0.0
        assert summary["avg_duration_ms"] == 0.0
        assert summary["tools_used"] == 0
        assert summary["most_used_tools"] == []
        assert summary["slowest_tools"] == []
        assert summary["error_rate"] == 0.0

    def test_get_performance_summary_with_data(self):
        """Test performance summary with execution data."""
        observer = ToolObserver()

        # Create executions for multiple tools with different patterns
        tools_data = [
            ("fast_tool", 50.0, ToolStatus.SUCCESS, 5),  # Fast, successful, many uses
            ("slow_tool", 500.0, ToolStatus.SUCCESS, 2),  # Slow, successful, few uses
            ("error_tool", 100.0, ToolStatus.ERROR, 3),  # Average, errors
        ]

        exec_id = 0
        for tool_name, duration, status, count in tools_data:
            for _ in range(count):
                execution = observer.start_execution(tool_name, f"exec_{exec_id}", {})
                execution.end_time = execution.start_time + timedelta(milliseconds=duration)
                execution.duration_ms = duration
                execution.status = status

                observer._update_metrics(execution)
                observer._executions.append(execution)
                exec_id += 1

        summary = observer.get_performance_summary()

        assert summary["total_executions"] == 10  # 5 + 2 + 3
        assert summary["success_rate"] == 70.0  # (5 + 2) / 10 * 100
        assert summary["error_rate"] == 30.0  # 3 / 10 * 100
        assert summary["tools_used"] == 3

        # Most used tools (by execution count)
        most_used = summary["most_used_tools"]
        assert most_used[0]["name"] == "fast_tool"
        assert most_used[0]["count"] == 5

        # Slowest tools (by average duration)
        slowest = summary["slowest_tools"]
        assert slowest[0]["name"] == "slow_tool"
        assert slowest[0]["avg_duration_ms"] == 500.0

    def test_cleanup_old_data(self):
        """Test cleanup of old execution data."""
        observer = ToolObserver(retention_days=1)

        # Create old and new executions
        old_time = datetime.now(timezone.utc) - timedelta(days=2)
        new_time = datetime.now(timezone.utc)

        # Add old execution
        old_execution = ToolExecution("old_tool", "old_exec", old_time)
        observer._executions.append(old_execution)

        # Add new execution
        new_execution = ToolExecution("new_tool", "new_exec", new_time)
        observer._executions.append(new_execution)

        with patch.object(observer.logger, "info") as mock_info:
            observer.cleanup_old_data()

            # Should have removed 1 old execution
            mock_info.assert_called_once_with("Cleaned up 1 old execution records")

        # Only new execution should remain
        assert len(observer._executions) == 1
        assert observer._executions[0].execution_id == "new_exec"

    def test_add_listener(self):
        """Test adding listeners for real-time events."""
        observer = ToolObserver()

        listener1 = Mock()
        listener2 = Mock()

        observer.add_listener(listener1)
        observer.add_listener(listener2)

        # Test notification reaches both listeners
        observer._notify_listeners("test_event", "test_data")

        listener1.assert_called_once_with("test_event", "test_data")
        listener2.assert_called_once_with("test_event", "test_data")

    def test_export_data_json(self):
        """Test exporting observability data to JSON."""
        observer = ToolObserver()

        # Create some test data
        execution = observer.start_execution("test_tool", "exec_123", {"param": "value"})
        observer.finish_execution("exec_123", ToolStatus.SUCCESS, result_summary="Success")

        # Export data
        exported = observer.export_data()
        data = json.loads(exported)

        assert "summary" in data
        assert "metrics" in data
        assert "recent_executions" in data
        assert "error_analysis" in data
        assert "exported_at" in data

        # Verify structure
        assert data["summary"]["total_executions"] == 1
        assert "test_tool" in data["metrics"]
        assert len(data["recent_executions"]) == 1
        assert data["error_analysis"]["total_errors"] == 0

    def test_export_data_invalid_format(self):
        """Test export with invalid format raises error."""
        observer = ToolObserver()

        with pytest.raises(ValueError, match="Only JSON format is currently supported"):
            observer.export_data("xml")


def test_context_manager_success():
    """Test context manager with successful execution."""
    observer = ToolObserver()

    with tool_execution_context(observer, "test_tool", "exec_123", {"param": "value"}) as execution:
        assert execution is not None
        assert execution.tool_name == "test_tool"
        assert execution.execution_id == "exec_123"

    # Should have finished successfully
    assert len(observer._executions) == 1
    assert observer._executions[0].status == ToolStatus.SUCCESS


def test_context_manager_exception():
    """Test context manager with exception."""
    observer = ToolObserver()

    with pytest.raises(ValueError, match="Test error"):
        with tool_execution_context(observer, "test_tool", "exec_123", {"param": "value"}):
            raise ValueError("Test error")

    # Should have finished with error
    assert len(observer._executions) == 1
    execution = observer._executions[0]
    assert execution.status == ToolStatus.ERROR
    assert execution.error_message == "Test error"
    assert execution.error_type == "ValueError"


def test_context_manager_none_level():
    """Test context manager with NONE observability level."""
    observer = ToolObserver(level=ObservabilityLevel.NONE)

    with tool_execution_context(observer, "test_tool", "exec_123", {"param": "value"}) as execution:
        assert execution is None

    # Should have no executions recorded
    assert len(observer._executions) == 0


def test_get_global_observer_singleton():
    """Test global observer singleton behavior."""
    # Reset to ensure clean state
    reset_observability()

    observer1 = get_global_observer()
    observer2 = get_global_observer()

    # Should return the same instance
    assert observer1 is observer2
    assert isinstance(observer1, ToolObserver)


def test_configure_observability():
    """Test configuring global observability."""
    observer = configure_observability(
        level=ObservabilityLevel.DEBUG, max_executions=500, retention_days=14
    )

    assert observer.level == ObservabilityLevel.DEBUG
    assert observer.max_executions == 500
    assert observer.retention_days == 14

    # Should be the same as global observer
    assert get_global_observer() is observer


def test_reset_observability():
    """Test resetting observability data."""
    from talk_box.tool_observability import reset_observability

    reset_observability()
    observer = get_global_observer()

    # Add some data
    execution = observer.start_execution("test_tool", "exec_123", {})
    observer.finish_execution("exec_123", ToolStatus.SUCCESS)

    # At least one execution should be recorded (exact count can vary if global state changed elsewhere)
    assert len(observer._executions) >= 1
    assert len(observer._metrics) == 1

    # Reset
    reset_observability()

    # Data should be cleared
    assert len(observer._executions) == 0
    assert len(observer._metrics) == 0
    assert len(observer._active_executions) == 0


def test_reset_observability_none_observer():
    """Test reset when no global observer exists."""
    global _global_observer
    original = _global_observer
    _global_observer = None

    try:
        # Should not raise error
        reset_observability()
    finally:
        _global_observer = original


def test_threading_safety():
    """Test thread safety of observer operations."""
    observer = ToolObserver()
    results = []

    def worker(thread_id):
        for i in range(10):
            exec_id = f"thread_{thread_id}_exec_{i}"
            execution = observer.start_execution(f"tool_{thread_id}", exec_id, {})
            observer.finish_execution(exec_id, ToolStatus.SUCCESS)
            results.append(exec_id)

    threads = []
    for i in range(3):
        thread = threading.Thread(target=worker, args=(i,))
        threads.append(thread)
        thread.start()

    for thread in threads:
        thread.join()

    # Should have 30 total executions (3 threads * 10 executions)
    assert len(observer._executions) == 30
    assert len(results) == 30

    # Check metrics for each tool
    metrics = observer.get_metrics()
    assert len(metrics) == 3  # 3 different tools
    for tool_name, tool_metrics in metrics.items():
        assert tool_metrics.total_executions == 10


def test_memory_profiling_disabled():
    """Test behavior when memory profiling is disabled."""
    observer = ToolObserver(enable_memory_profiling=False)

    execution = observer.start_execution("test_tool", "exec_123", {})
    result = observer.finish_execution("exec_123", ToolStatus.SUCCESS, memory_usage_mb=100.0)

    # Memory usage should still be recorded if provided
    assert result.memory_usage_mb == 100.0


def test_large_execution_history():
    """Test handling of large execution history with maxlen."""
    observer = ToolObserver(max_executions=5)

    # Add more executions than the limit
    for i in range(10):
        execution = observer.start_execution("test_tool", f"exec_{i}", {})
        observer.finish_execution(f"exec_{i}", ToolStatus.SUCCESS)

    # Should only keep the last 5 executions
    assert len(observer._executions) == 5

    # Should be the most recent ones
    execution_ids = [e.execution_id for e in observer._executions]
    expected_ids = [f"exec_{i}" for i in range(5, 10)]
    assert execution_ids == expected_ids


def test_complex_filtering_scenario():
    """Test complex execution filtering scenarios."""
    observer = ToolObserver()

    # Create varied execution history
    base_time = datetime.now(timezone.utc)
    scenarios = [
        ("tool1", ToolStatus.SUCCESS, base_time - timedelta(hours=2)),
        ("tool1", ToolStatus.ERROR, base_time - timedelta(hours=1)),
        ("tool2", ToolStatus.SUCCESS, base_time - timedelta(minutes=30)),
        ("tool2", ToolStatus.SUCCESS, base_time - timedelta(minutes=10)),
        ("tool3", ToolStatus.ERROR, base_time),
    ]

    for i, (tool, status, start_time) in enumerate(scenarios):
        execution = ToolExecution(
            tool_name=tool, execution_id=f"exec_{i}", start_time=start_time, status=status
        )
        observer._executions.append(execution)
        observer._update_metrics(execution)

    # Test multiple filters combined
    recent_tool1_errors = observer.get_executions(
        tool_name="tool1",
        status=ToolStatus.ERROR,
        since=base_time - timedelta(hours=1, minutes=30),
        limit=1,
    )

    assert len(recent_tool1_errors) == 1
    assert recent_tool1_errors[0].tool_name == "tool1"
    assert recent_tool1_errors[0].status == ToolStatus.ERROR


# =============================================================================
# INTEGRATION TESTS
# =============================================================================
# The following test classes were moved from test_tool_observability.py
# to consolidate all observability testing in one place.


def test_basic_observability():
    """Test basic observability functionality."""
    # Configure detailed observability for this test
    observer = ToolObserver(level=ObservabilityLevel.DETAILED)

    # Start an execution
    execution = observer.start_execution(
        tool_name="test_tool",
        execution_id="test_123",
        parameters={"param1": "value1"},
        conversation_id="conv_1",
        user_id="user_1",
    )

    assert execution is not None
    assert execution.tool_name == "test_tool"
    assert execution.execution_id == "test_123"
    assert execution.parameters == {"param1": "value1"}
    assert execution.status == ToolStatus.PENDING

    # Finish the execution
    finished_execution = observer.finish_execution(
        execution_id="test_123", status=ToolStatus.SUCCESS, result_summary="Test completed"
    )

    assert finished_execution is not None
    assert finished_execution.status == ToolStatus.SUCCESS
    assert finished_execution.result_summary == "Test completed"
    assert finished_execution.duration_ms is not None
    assert finished_execution.duration_ms > 0


def test_metrics_aggregation():
    """Test that metrics are properly aggregated."""
    from talk_box.tool_observability import reset_observability

    reset_observability()
    observer = tb.get_global_observer()

    # Execute multiple operations
    for i in range(5):
        execution = observer.start_execution(
            tool_name="test_tool", execution_id=f"test_{i}", parameters={"iteration": i}
        )

        status = ToolStatus.SUCCESS if i < 4 else ToolStatus.ERROR
        observer.finish_execution(
            execution_id=f"test_{i}",
            status=status,
            error_message="Test error" if status == ToolStatus.ERROR else None,
        )

    # Check metrics
    metrics = observer.get_metrics("test_tool")
    assert "test_tool" in metrics

    tool_metrics = metrics["test_tool"]
    assert tool_metrics.total_executions == 5
    assert tool_metrics.successful_executions == 4
    assert tool_metrics.failed_executions == 1
    assert tool_metrics.success_rate() == 80.0
    assert tool_metrics.error_rate == 20.0


def test_execution_filtering():
    """Test filtering of execution records."""
    from talk_box.tool_observability import reset_observability

    reset_observability()
    observer = tb.get_global_observer()

    # Create executions with different tools and statuses
    tools_and_statuses = [
        ("tool_a", ToolStatus.SUCCESS),
        ("tool_a", ToolStatus.ERROR),
        ("tool_b", ToolStatus.SUCCESS),
        ("tool_b", ToolStatus.SUCCESS),
        ("tool_a", ToolStatus.ERROR),
    ]

    for i, (tool, status) in enumerate(tools_and_statuses):
        execution = observer.start_execution(
            tool_name=tool, execution_id=f"exec_{i}", parameters={}
        )
        observer.finish_execution(execution_id=f"exec_{i}", status=status)

    # Test filtering by tool
    tool_a_executions = observer.get_executions(tool_name="tool_a")
    assert len(tool_a_executions) == 3

    tool_b_executions = observer.get_executions(tool_name="tool_b")
    assert len(tool_b_executions) == 2

    # Test filtering by status
    error_executions = observer.get_executions(status=ToolStatus.ERROR)
    assert len(error_executions) == 2

    success_executions = observer.get_executions(status=ToolStatus.SUCCESS)
    assert len(success_executions) == 3

    # Test limit
    limited_executions = observer.get_executions(limit=3)
    assert len(limited_executions) == 3


def test_error_analysis():
    """Test error analysis functionality."""
    observer = tb.get_global_observer()

    # Create some errors
    error_scenarios = [
        ("ValueError", "Invalid input value"),
        ("ConnectionError", "Network connection failed"),
        ("ValueError", "Invalid input value"),  # Duplicate
        ("TimeoutError", "Operation timed out"),
    ]

    for i, (error_type, error_message) in enumerate(error_scenarios):
        execution = observer.start_execution(
            tool_name="error_tool", execution_id=f"error_{i}", parameters={}
        )
        observer.finish_execution(
            execution_id=f"error_{i}",
            status=ToolStatus.ERROR,
            error_message=error_message,
            error_type=error_type,
        )

    # Analyze errors
    analysis = observer.get_error_analysis("error_tool")

    assert analysis["total_errors"] == 4
    assert "ValueError" in analysis["error_types"]
    assert analysis["error_types"]["ValueError"] == 2
    assert "ConnectionError" in analysis["error_types"]
    assert analysis["error_types"]["ConnectionError"] == 1

    # Check common patterns
    common_patterns = analysis["common_patterns"]
    assert len(common_patterns) > 0

    # Most common should be "Invalid input value" (appears twice)
    most_common = common_patterns[0]
    assert most_common["message"] == "Invalid input value"
    assert most_common["count"] == 2


def test_performance_summary():
    """Test performance summary generation."""
    # Ensure clean state: reset observer & registry side-effects
    from talk_box.tool_observability import reset_observability

    reset_observability()
    observer = tb.get_global_observer()

    # Create mixed executions
    for i in range(10):
        tool_name = f"tool_{i % 3}"  # 3 different tools
        execution = observer.start_execution(
            tool_name=tool_name, execution_id=f"perf_{i}", parameters={}
        )

        status = ToolStatus.SUCCESS if i < 8 else ToolStatus.ERROR
        observer.finish_execution(execution_id=f"perf_{i}", status=status)

    summary = observer.get_performance_summary()

    assert summary["total_executions"] == 10
    assert summary["success_rate"] == 80.0
    assert summary["error_rate"] == 20.0
    assert summary["tools_used"] == 3
    assert len(summary["most_used_tools"]) <= 5
    assert len(summary["slowest_tools"]) <= 5


def test_observability_levels():
    """Test different observability levels."""
    # Test NONE level
    observer = ToolObserver(level=ObservabilityLevel.NONE)
    execution = observer.start_execution("test", "123", {})
    assert execution is None

    finished = observer.finish_execution("123", ToolStatus.SUCCESS)
    assert finished is None

    # Test BASIC level (parameters not stored)
    observer = ToolObserver(level=ObservabilityLevel.BASIC)
    execution = observer.start_execution("test", "123", {"param": "value"})
    assert execution is not None
    assert execution.parameters == {}  # Not stored in BASIC mode

    # Test DETAILED level (parameters stored)
    observer = ToolObserver(level=ObservabilityLevel.DETAILED)
    execution = observer.start_execution("test", "123", {"param": "value"})
    assert execution is not None
    assert execution.parameters == {"param": "value"}  # Stored in DETAILED mode


def test_debugger_creation():
    """Test that ToolDebugger can be created."""
    debugger = tb.ToolDebugger()
    assert debugger.observer is not None

    # Test with custom observer
    custom_observer = ToolObserver(level=ObservabilityLevel.DEBUG)
    custom_debugger = tb.ToolDebugger(observer=custom_observer)
    assert custom_debugger.observer is custom_observer


def test_debug_functions():
    """Test convenience debug functions."""
    observer = tb.get_global_observer()

    # Create some test data
    execution = observer.start_execution("test_tool", "test_123", {"param": "value"})
    observer.finish_execution("test_123", ToolStatus.SUCCESS)

    # Test that debug functions don't crash
    # (We can't easily test output in unit tests, but we can ensure they run)
    try:
        tb.debug_dashboard()
        tb.debug_tool("test_tool")
        tb.debug_errors()
        tb.debug_errors("test_tool")

        # These should complete without error
        assert True
    except Exception as e:
        pytest.fail(f"Debug functions failed: {e}")


@pytest.mark.asyncio
async def test_tool_integration():
    """Test that tools properly integrate with observability."""
    reset_observability()

    # Create a simple test tool
    @tb.tool(name="integration_test_tool", description="A test tool for integration testing")
    def test_tool(context: tb.ToolContext, value: int) -> tb.ToolResult:
        if value < 0:
            raise ValueError("Value must be positive")
        return tb.ToolResult(data=value * 2)

    # Execute the tool
    context = tb.ToolContext()
    tool = tb.get_global_registry().get_tool("integration_test_tool")

    # Successful execution
    result = await tool.execute(context, value=5)
    assert result.success
    assert result.data == 10

    # Failed execution
    result = await tool.execute(context, value=-1)
    assert not result.success
    assert "Value must be positive" in result.error

    # Check that observability captured these executions
    observer = tb.get_global_observer()
    executions = observer.get_executions(tool_name="integration_test_tool")
    assert len(executions) == 2

    # Check metrics
    metrics = observer.get_metrics("integration_test_tool")
    tool_metrics = metrics["integration_test_tool"]
    assert tool_metrics.total_executions == 2
    assert tool_metrics.successful_executions == 1
    assert tool_metrics.failed_executions == 1
