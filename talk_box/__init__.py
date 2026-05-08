from importlib.metadata import PackageNotFoundError, version

try:  # pragma: no cover
    __version__ = version("talk-box")
except PackageNotFoundError:  # pragma: no cover
    __version__ = "0.0.0"

__author__ = "Richard Iannone"
__email__ = "riannone@me.com"

# Core imports for easy access
from talk_box.attachments import AttachmentMetadata, Attachments
from talk_box.builder import BuilderTypes, ChatBot
from talk_box.builtin_tools import get_builtin_tool, load_selected_tools, load_tool_box
from talk_box.consensus import (
    ConsensusResult,
    ConsensusStrategy,
    Disagreement,
    ModelResponse,
    consensus,
    find_disagreements,
)
from talk_box.context_window import (
    ContextWindow,
    FitResult,
    FitStrategy,
    PromptFitResult,
    estimate_tokens,
)
from talk_box.conversation import (
    Conversation,
    Message,
    ToolEnabledConversation,
    create_tool_conversation,
)
from talk_box.eval import (
    EvalCase,
    EvalDimension,
    EvalResult,
    EvalResults,
    EvalScore,
    eval,
    eval_model_update,
    eval_regression,
    eval_suite,
    scorecard_table,
    sweep_table,
)
from talk_box.guardrails import (
    Guard,
    GuardAction,
    GuardPhase,
    GuardPipeline,
    GuardPipelineResult,
    GuardResult,
    disclaimer_required,
    guardrail,
    keyword_block,
    max_input_length,
    max_response_length,
    must_cite_sources,
    no_pii,
    resolve_guards,
    tone_check,
)
from talk_box.models import (
    CostTier,
    ModelProfile,
    OllamaStatus,
    detect_ollama,
    get_model_profile,
    list_models,
    list_ollama_models,
    model_profiles_table,
    register_model,
    sync_ollama_models,
)
from talk_box.pathways import Pathways
from talk_box.personas import PersonaDefinition, get_persona, list_personas, persona_categories
from talk_box.presets import Preset, PresetManager, PresetNames
from talk_box.prompt_builder import (
    Priority,
    PromptBuilder,
    PromptSection,
    VocabularyTerm,
    architectural_analysis_prompt,
    code_review_prompt,
    debugging_prompt,
)
from talk_box.prompt_optimizer import (
    OptimizationLevel,
    OptimizeResult,
    optimize_prompt,
)
from talk_box.routing import (
    Router,
    RoutingResult,
    RoutingStrategy,
    TaskComplexity,
    classify_complexity,
    route,
)
from talk_box.testing import (
    PathwayTestResults,
    TestResults,
    autotest_avoid_topics,
    autotest_pathways,
)
from talk_box.tool_debugging import (
    ToolDebugger,
    configure_debug_mode,
    debug_dashboard,
    debug_errors,
    debug_tool,
    export_debug_report,
    live_monitor,
)
from talk_box.tool_observability import (
    ObservabilityLevel,
    ToolObserver,
    configure_observability,
    get_global_observer,
)
from talk_box.tools import (
    ToolCategory,
    ToolContext,
    ToolResult,
    get_global_registry,
    tool,
)

# Make key classes available at package level
__all__ = [
    # Core classes
    "ChatBot",
    "Conversation",
    "Message",
    "ToolEnabledConversation",
    "create_tool_conversation",
    # File attachments
    "Attachments",
    "AttachmentMetadata",
    # Conversational pathways
    "Pathways",
    # Guardrails
    "Guard",
    "GuardAction",
    "GuardPhase",
    "GuardPipeline",
    "GuardPipelineResult",
    "GuardResult",
    "guardrail",
    "disclaimer_required",
    "keyword_block",
    "max_input_length",
    "max_response_length",
    "must_cite_sources",
    "no_pii",
    "resolve_guards",
    "tone_check",
    # Context window management
    "ContextWindow",
    "FitResult",
    "FitStrategy",
    "PromptFitResult",
    "estimate_tokens",
    # Evaluation
    "eval",
    "eval_model_update",
    "eval_regression",
    "eval_suite",
    "EvalCase",
    "EvalDimension",
    "EvalResult",
    "EvalResults",
    "EvalScore",
    "scorecard_table",
    "sweep_table",
    # Model profiles
    "ModelProfile",
    "CostTier",
    "OllamaStatus",
    "get_model_profile",
    "list_models",
    "register_model",
    "model_profiles_table",
    # Ollama
    "detect_ollama",
    "list_ollama_models",
    "sync_ollama_models",
    # Prompt engineering
    "PromptBuilder",
    "Priority",
    "PromptSection",
    "VocabularyTerm",
    "architectural_analysis_prompt",
    "code_review_prompt",
    "debugging_prompt",
    # Prompt optimization
    "OptimizationLevel",
    "OptimizeResult",
    "optimize_prompt",
    # Hybrid routing
    "Router",
    "RoutingResult",
    "RoutingStrategy",
    "TaskComplexity",
    "classify_complexity",
    "route",
    # Consensus mode
    "ConsensusResult",
    "ConsensusStrategy",
    "Disagreement",
    "ModelResponse",
    "consensus",
    "find_disagreements",
    # Persona system
    "PersonaDefinition",
    "get_persona",
    "list_personas",
    "persona_categories",
    # Preset management
    "Preset",
    "PresetManager",
    "PresetNames",
    # Builder types
    "BuilderTypes",
    # Testing functions
    "autotest_avoid_topics",
    "autotest_pathways",
    # Testing classes
    "TestResults",
    "PathwayTestResults",
    # Tool Box
    "tool",
    "ToolCategory",
    "ToolContext",
    "ToolResult",
    "get_global_registry",
    "load_tool_box",
    "get_builtin_tool",
    "load_selected_tools",
    # Tool Observability
    "ObservabilityLevel",
    "ToolObserver",
    "configure_observability",
    "get_global_observer",
    # Tool Debugging
    "ToolDebugger",
    "configure_debug_mode",
    "debug_dashboard",
    "debug_errors",
    "debug_tool",
    "export_debug_report",
    "live_monitor",
]
