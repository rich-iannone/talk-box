from importlib.metadata import PackageNotFoundError, version

try:  # pragma: no cover
    __version__ = version("talk-box")
except PackageNotFoundError:  # pragma: no cover
    __version__ = "0.0.0"

__author__ = "Richard Iannone"
__email__ = "riannone@me.com"

# Core imports for easy access
from talk_box.agent import Agent
from talk_box.attachments import AttachmentMetadata, Attachments
from talk_box.builder import BuilderTypes, ChatBot
from talk_box.builtin_tools import get_builtin_tool, load_selected_tools, load_tool_box
from talk_box.capture import (
    CaptureEvent,
    ConversationCapture,
    EventType,
)
from talk_box.cascade import (
    CascadeResult,
    CascadeRound,
    cascade,
    estimate_confidence,
)
from talk_box.comms import (
    AgentMessage,
    Mailbox,
    MessageType,
    broadcast,
    reply,
    send,
)
from talk_box.compliance import (
    export_html,
    export_json,
)
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
from talk_box.diff import (
    DiffResult,
    TurnDiff,
    TurnStatus,
    diff,
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
from talk_box.forgetting import (
    PolicyResult,
    compress_after_n_turns,
    forget_after_resolution,
    retain_only,
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
from talk_box.hitl import (
    HumanReview,
    ReviewDecision,
    ReviewQueue,
    ReviewStatus,
    approve,
    human_review,
    reject,
    revise,
)
from talk_box.mcp_bridge import (
    MCPBridgeServer,
    MCPToolInfo,
    discover_mcp_tools,
    list_mcp_tools,
    mcp_tool_to_talk_box,
    tools_to_mcp_server,
)
from talk_box.memory import (
    LongTermMemory,
    MemoryEntry,
    MemoryStore,
    MemoryTier,
    ShortTermMemory,
    WorkingMemory,
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
from talk_box.replay import (
    ReplayResult,
    ReplayTurn,
    replay,
)
from talk_box.retention import (
    RetentionPolicy,
    apply_retention,
)
from talk_box.routing import (
    Router,
    RoutingResult,
    RoutingStrategy,
    TaskComplexity,
    classify_complexity,
    route,
)
from talk_box.shared_state import (
    SharedState,
    StateChange,
)
from talk_box.skills import (
    SkillDefinition,
    create_skill,
    get_skill,
    list_skills,
    load_skill,
    register_skill,
    skill_categories,
)
from talk_box.structured import (
    ExtractResult,
    extract,
    schema_to_dict,
)
from talk_box.subagents import (
    SubagentResult,
    children,
    delegate,
    parent_name,
    spawn,
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
from talk_box.traits import (
    TraitDefinition,
    apply_trait,
    create_trait,
    get_trait,
    list_traits,
    load_trait,
    register_trait,
    trait_categories,
)

# Make key classes available at package level
__all__ = [
    # Core classes
    "Agent",
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
    # Cascade consensus
    "CascadeResult",
    "CascadeRound",
    "cascade",
    "estimate_confidence",
    # Memory tiers
    "LongTermMemory",
    "MemoryEntry",
    "MemoryStore",
    "MemoryTier",
    "ShortTermMemory",
    "WorkingMemory",
    # Forgetting policies
    "PolicyResult",
    "forget_after_resolution",
    "compress_after_n_turns",
    "retain_only",
    # Conversation capture
    "ConversationCapture",
    "CaptureEvent",
    "EventType",
    # Replay mode
    "ReplayResult",
    "ReplayTurn",
    "replay",
    # Conversation diff
    "DiffResult",
    "TurnDiff",
    "TurnStatus",
    "diff",
    # Compliance export
    "export_json",
    "export_html",
    # Role-based retention
    "RetentionPolicy",
    "apply_retention",
    # Shared state
    "SharedState",
    "StateChange",
    # Subagent spawning
    "SubagentResult",
    "spawn",
    "delegate",
    "children",
    "parent_name",
    # Human-in-the-loop
    "HumanReview",
    "ReviewDecision",
    "ReviewStatus",
    "ReviewQueue",
    "human_review",
    "approve",
    "reject",
    "revise",
    # Agent communication
    "AgentMessage",
    "MessageType",
    "Mailbox",
    "send",
    "broadcast",
    "reply",
    # Skill system
    "SkillDefinition",
    "create_skill",
    "get_skill",
    "list_skills",
    "register_skill",
    "load_skill",
    "skill_categories",
    # Structured outputs
    "ExtractResult",
    "extract",
    "schema_to_dict",
    # Persona traits
    "TraitDefinition",
    "apply_trait",
    "create_trait",
    "get_trait",
    "list_traits",
    "register_trait",
    "load_trait",
    "trait_categories",
    # MCP bridge
    "MCPToolInfo",
    "MCPBridgeServer",
    "tools_to_mcp_server",
    "mcp_tool_to_talk_box",
    "list_mcp_tools",
    "discover_mcp_tools",
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
