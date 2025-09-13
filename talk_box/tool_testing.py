"""
Talk Box Tool Testing Framework

Comprehensive utilities for testing Talk Box tools including:
- Mock contexts and environments
- Tool validation and verification
- Integration testing helpers
- Performance testing utilities
"""

import contextlib
import inspect
import io
import logging
import time
from datetime import datetime
from typing import Any, Dict, List, Optional

from .tools import TalkBoxTool, ToolContext, ToolRegistry, ToolResult

__all__ = [
    # Test contexts
    "MockToolContext",
    "create_test_context",
    # Test utilities
    "ToolTester",
    "ToolValidator",
    "ToolPerformanceTester",
    # Assertions
    "assert_tool_success",
    "assert_tool_failure",
    "assert_tool_result_equals",
    # Fixtures
    "mock_registry",
    "capture_tool_logs",
]


class MockToolContext(ToolContext):
    """
    A mock tool context for testing that provides predictable responses
    and captures interactions for verification.
    """

    def __init__(
        self,
        conversation_id: str = "test-conv",
        user_id: str = "test-user",
        session_id: str = "test-session",
        conversation_history: Optional[List[Dict[str, Any]]] = None,
        user_metadata: Optional[Dict[str, Any]] = None,
        mock_responses: Optional[Dict[str, Any]] = None,
        **kwargs,
    ):
        super().__init__(
            conversation_id=conversation_id,
            user_id=user_id,
            session_id=session_id,
            conversation_history=conversation_history or [],
            user_metadata=user_metadata or {},
            **kwargs,
        )
        self.mock_responses = mock_responses or {}
        self.interactions_log: List[Dict[str, Any]] = []

    def get_user_preference(self, key: str, default: Any = None) -> Any:
        """Mock user preference retrieval with logging."""
        self.interactions_log.append(
            {
                "action": "get_user_preference",
                "key": key,
                "default": default,
                "timestamp": datetime.utcnow(),
            }
        )

        # Check mock responses first
        if f"preference.{key}" in self.mock_responses:
            return self.mock_responses[f"preference.{key}"]

        return super().get_user_preference(key, default)

    def log_tool_usage(self, tool_name: str, parameters: Dict[str, Any]) -> None:
        """Enhanced tool usage logging for testing."""
        self.interactions_log.append(
            {
                "action": "log_tool_usage",
                "tool_name": tool_name,
                "parameters": parameters,
                "timestamp": datetime.utcnow(),
            }
        )
        super().log_tool_usage(tool_name, parameters)

    def set_mock_response(self, key: str, value: Any) -> None:
        """Set a mock response for testing."""
        self.mock_responses[key] = value

    def get_interactions(self, action_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get logged interactions, optionally filtered by action type."""
        if action_type:
            return [log for log in self.interactions_log if log["action"] == action_type]
        return self.interactions_log.copy()


def create_test_context(
    user_preferences: Optional[Dict[str, Any]] = None,
    conversation_messages: Optional[List[Dict[str, Any]]] = None,
    mock_responses: Optional[Dict[str, Any]] = None,
    **context_kwargs,
) -> MockToolContext:
    """
    Create a test context with common testing defaults.

    Args:
        user_preferences: User preferences to set
        conversation_messages: Conversation history to set
        mock_responses: Mock responses for context methods
        **context_kwargs: Additional context arguments

    Returns:
        Configured MockToolContext for testing
    """
    user_metadata = {"preferences": user_preferences or {}}

    return MockToolContext(
        conversation_history=conversation_messages or [],
        user_metadata=user_metadata,
        mock_responses=mock_responses or {},
        **context_kwargs,
    )


class ToolTester:
    """
    Comprehensive tool testing utility with validation and verification.
    """

    def __init__(self, tool: TalkBoxTool):
        self.tool = tool
        self.test_results: List[Dict[str, Any]] = []

    async def test_with_params(
        self,
        context: ToolContext,
        params: Dict[str, Any],
        expected_success: bool = True,
        expected_data: Any = None,
        expected_error: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Test tool with specific parameters and validate results.

        Returns:
            Test result dictionary with outcome and metrics
        """
        test_start = time.time()

        try:
            result = await self.tool.execute(context, **params)
            test_duration = time.time() - test_start

            test_result = {
                "success": True,
                "test_duration": test_duration,
                "tool_result": result,
                "params": params,
                "validation": {},
            }

            # Validate success expectation
            if result.success != expected_success:
                test_result["validation"]["success_mismatch"] = {
                    "expected": expected_success,
                    "actual": result.success,
                }

            # Validate expected data
            if expected_data is not None and result.data != expected_data:
                test_result["validation"]["data_mismatch"] = {
                    "expected": expected_data,
                    "actual": result.data,
                }

            # Validate expected error
            if expected_error is not None:
                if expected_error not in (result.error or ""):
                    test_result["validation"]["error_mismatch"] = {
                        "expected": expected_error,
                        "actual": result.error,
                    }

        except Exception as e:  # pragma: no cover
            test_duration = time.time() - test_start  # pragma: no cover
            test_result = {  # pragma: no cover
                "success": False,
                "test_duration": test_duration,
                "exception": str(e),
                "params": params,
                "validation": {"unexpected_exception": True},
            }

        self.test_results.append(test_result)
        return test_result

    async def run_test_suite(self, test_cases: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Run a comprehensive test suite with multiple test cases.

        Args:
            test_cases: List of test case dictionaries with:
                - context: ToolContext to use
                - params: Parameters to pass to tool
                - expected_success: Expected success status
                - expected_data: Expected result data
                - expected_error: Expected error message
                - description: Test case description

        Returns:
            Test suite results with summary statistics
        """
        suite_start = time.time()
        results = []

        for i, test_case in enumerate(test_cases):
            case_result = await self.test_with_params(
                context=test_case["context"],
                params=test_case["params"],
                expected_success=test_case.get("expected_success", True),
                expected_data=test_case.get("expected_data"),
                expected_error=test_case.get("expected_error"),
            )
            case_result["case_index"] = i
            case_result["description"] = test_case.get("description", f"Test case {i + 1}")
            results.append(case_result)

        suite_duration = time.time() - suite_start

        # Calculate summary statistics
        total_tests = len(results)
        successful_tests = sum(1 for r in results if r["success"])
        validation_failures = sum(1 for r in results if r["success"] and r["validation"])

        return {
            "suite_duration": suite_duration,
            "total_tests": total_tests,
            "successful_tests": successful_tests,
            "failed_tests": total_tests - successful_tests,
            "validation_failures": validation_failures,
            "results": results,
            "summary": {
                "success_rate": successful_tests / total_tests if total_tests > 0 else 0,
                "avg_test_duration": sum(r["test_duration"] for r in results) / total_tests
                if total_tests > 0
                else 0,
            },
        }


class ToolValidator:
    """
    Validates tool definitions and implementations for best practices.
    """

    @staticmethod
    def validate_tool(tool: TalkBoxTool) -> Dict[str, Any]:
        """
        Validate a tool for common issues and best practices.

        Returns:
            Validation report with issues and recommendations
        """
        issues = []
        warnings = []
        recommendations = []

        # Check basic properties
        if not tool.name:
            issues.append("Tool name is empty or None")

        if not tool.description:
            issues.append("Tool description is empty or None")
        elif len(tool.description) < 10:
            warnings.append("Tool description is very short (< 10 characters)")

        # Check function signature
        try:
            sig = inspect.signature(tool.func)
            has_context = any(param.annotation == ToolContext for param in sig.parameters.values())
            if not has_context:
                issues.append("Tool function should accept ToolContext as first parameter")

        except Exception as e:  # pragma: no cover
            issues.append(f"Could not inspect function signature: {e}")  # pragma: no cover

        # Check parameters
        if not tool.parameters or not tool.parameters.get("properties"):
            warnings.append("Tool has no parameters defined")

        # Check for examples
        if not tool.examples:
            recommendations.append("Consider adding usage examples to help users")

        # Check for tags
        if not tool.tags:
            recommendations.append("Consider adding tags to improve tool discovery")

        return {
            "valid": len(issues) == 0,
            "issues": issues,
            "warnings": warnings,
            "recommendations": recommendations,
            "score": max(0, 100 - len(issues) * 30 - len(warnings) * 10),
        }

    @staticmethod
    def validate_registry(registry: ToolRegistry) -> Dict[str, Any]:
        """
        Validate an entire tool registry.

        Returns:
            Registry validation report
        """
        tools = registry.get_all_tools()
        tool_reports = {}

        total_issues = 0
        total_warnings = 0

        for tool in tools:
            report = ToolValidator.validate_tool(tool)
            tool_reports[tool.name] = report
            total_issues += len(report["issues"])
            total_warnings += len(report["warnings"])

        # Check for name collisions
        name_collisions = []
        names = [tool.name for tool in tools]
        for name in set(names):
            if names.count(name) > 1:  # pragma: no cover
                name_collisions.append(name)  # pragma: no cover

        return {
            "total_tools": len(tools),
            "valid_tools": sum(1 for report in tool_reports.values() if report["valid"]),
            "total_issues": total_issues,
            "total_warnings": total_warnings,
            "name_collisions": name_collisions,
            "tool_reports": tool_reports,
            "avg_score": sum(report["score"] for report in tool_reports.values()) / len(tools)
            if tools
            else 0,
        }


class ToolPerformanceTester:
    """
    Performance testing and benchmarking for tools.
    """

    def __init__(self, tool: TalkBoxTool):
        self.tool = tool

    async def benchmark_execution(
        self, context: ToolContext, params: Dict[str, Any], iterations: int = 10
    ) -> Dict[str, Any]:
        """
        Benchmark tool execution performance.

        Returns:
            Performance metrics including timing statistics
        """
        execution_times = []
        memory_usage = []

        for _ in range(iterations):
            start_time = time.perf_counter()

            try:
                result = await self.tool.execute(context, **params)
                end_time = time.perf_counter()
                execution_times.append(end_time - start_time)

            except Exception as e:  # pragma: no cover
                # Still record timing for failed executions
                end_time = time.perf_counter()  # pragma: no cover
                execution_times.append(end_time - start_time)  # pragma: no cover

        return {
            "iterations": iterations,
            "total_time": sum(execution_times),
            "avg_time": sum(execution_times) / len(execution_times),
            "min_time": min(execution_times),
            "max_time": max(execution_times),
            "median_time": sorted(execution_times)[len(execution_times) // 2],
            "execution_times": execution_times,
        }


# Test assertion helpers
def assert_tool_success(result: ToolResult, message: str = "Tool execution should succeed"):
    """Assert that a tool result indicates success."""
    if not result.success:
        raise AssertionError(f"{message}. Error: {result.error}")


def assert_tool_failure(result: ToolResult, message: str = "Tool execution should fail"):
    """Assert that a tool result indicates failure."""
    if result.success:
        raise AssertionError(f"{message}. Tool unexpectedly succeeded with result: {result.data}")


def assert_tool_result_equals(
    result: ToolResult, expected: Any, message: str = "Tool result data mismatch"
):
    """Assert that a tool result data equals expected value."""
    if result.data != expected:
        raise AssertionError(f"{message}. Expected: {expected}, Got: {result.data}")


# Testing fixtures and context managers
@contextlib.contextmanager
def mock_registry():
    """
    Context manager that creates a temporary tool registry for testing.

    Usage:
        with mock_registry() as registry:
            # Register test tools
            # Run tests
            pass
    """
    from .tools import _global_registry, clear_global_registry

    # Save current registry
    original_registry = _global_registry

    # Clear and yield new registry
    clear_global_registry()
    try:
        yield _global_registry
    finally:
        # Restore original registry
        globals()["_global_registry"] = original_registry


@contextlib.contextmanager
def capture_tool_logs(level: int = logging.INFO):
    """
    Context manager to capture tool execution logs for testing.

    Usage:
        with capture_tool_logs() as log_capture:
            # Execute tools
            pass
        logs = log_capture.getvalue()
    """
    logger = logging.getLogger("talk_box.tools")
    original_level = logger.level

    # Create string buffer to capture logs
    log_buffer = io.StringIO()
    handler = logging.StreamHandler(log_buffer)
    handler.setLevel(level)

    logger.addHandler(handler)
    logger.setLevel(level)

    try:
        yield log_buffer
    finally:
        logger.removeHandler(handler)
        logger.setLevel(original_level)
