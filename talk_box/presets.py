from dataclasses import dataclass
from typing import Any, Optional


@dataclass
class Preset:
    """
    Defines reusable behavior templates for chatbot personality and capabilities.

    The `Preset` class encapsulates pre-configured behavior patterns that can be applied
    to chatbots to instantly configure their personality, expertise, communication style,
    and operational constraints. Presets provide a powerful way to create consistent,
    specialized chatbot behaviors without manual configuration of individual parameters.

    Each preset defines a complete behavioral profile including conversational tone,
    areas of expertise, response verbosity, operational constraints, and optional system
    prompts. This enables rapid deployment of specialized chatbots for specific use cases
    like customer support, technical advisory, creative writing, or data analysis.

    Presets are designed to be both comprehensive and flexible - they provide sensible
    defaults while allowing override of individual settings when applied to chatbots.
    The system prompt can include template variables and conditional logic for dynamic
    behavior adaptation.

    Parameters
    ----------
    name : str
        The unique identifier for this preset. This name is used when applying the preset
        to chatbots via the `preset()` method. Names should be descriptive and follow
        snake_case conventions (e.g., "technical_advisor", "customer_support").
    tone : str
        The conversational tone or style the chatbot should adopt. This affects how
        the chatbot communicates and interacts with users. Examples include:
        - `"professional"`: Formal, business-appropriate communication
        - `"friendly"`: Warm, approachable, conversational style
        - `"authoritative"`: Confident, expert-level communication
        - `"analytical"`: Logical, data-driven, precise communication
        - `"creative"`: Imaginative, expressive, artistic communication
    expertise : str
        The primary areas of knowledge or specialization for this chatbot. This can be
        a single domain or comma-separated list of expertise areas. Examples include:
        - `"python,ml"`: Python programming and machine learning
        - `"customer_service"`: Customer support and service excellence
        - `"data_analysis,statistics"`: Data analysis and statistical methods
        - `"legal_knowledge"`: Legal information and procedures
        - `"creative_writing"`: Creative writing and storytelling
    verbosity : str
        The level of detail and length in responses. This controls how much information
        the chatbot provides and how thoroughly it explains concepts. Options include:
        - `"concise"`: Brief, direct responses focusing on essential information
        - `"detailed"`: Comprehensive responses with explanations and context
        - `"precise"`: Accurate, specific responses with exact information
        - `"descriptive"`: Rich, elaborate responses with vivid details
        - `"thorough"`: Complete, exhaustive responses covering all aspects
    constraints : list[str]
        A list of behavioral constraints or guidelines the chatbot should follow.
        These define what the chatbot should avoid or how it should behave in
        specific situations. Examples include:
        - `["no_slang", "professional_only"]`: Maintain professional language
        - `["family_friendly", "no_controversial_topics"]`: Safe, appropriate content
        - `["evidence_based", "cite_sources"]`: Factual, well-sourced information
        - `["no_personal_advice", "disclaimers_required"]`: Legal/medical boundaries
        - `["positive_tone", "solution_focused"]`: Constructive, helpful approach
    system_prompt : str, optional
        An optional system prompt template that provides detailed instructions to the
        underlying language model. This prompt sets the behavioral foundation and can
        include specific instructions, examples, and formatting guidelines. If not
        provided, behavior will be inferred from other preset parameters.

    Returns
    -------
    Preset
        A new Preset instance with the specified behavioral configuration.

    Preset Application
    -----------------
    Presets are applied to chatbots using the chainable API:

    ```python
    bot = ChatBot().preset("technical_advisor")
    ```

    When applied, presets automatically configure multiple aspects of chatbot behavior
    while allowing individual parameter overrides:

    ```python
    bot = (
        ChatBot()
        .preset("technical_advisor")        # Apply preset
        .temperature(0.3)                   # Override temperature
        .max_tokens(1500)                   # Override token limit
    )
    ```

    Examples
    --------
    ### Creating custom presets for specialized domains

    Define presets for specific business or technical domains:

    ```python
    import talk_box as tb

    # DevOps engineering specialist
    devops_preset = tb.Preset(
        name="devops_engineer",
        tone="authoritative",
        expertise="kubernetes,docker,ci_cd,monitoring",
        verbosity="detailed",
        constraints=["best_practices", "security_conscious", "production_ready"],
        system_prompt="You are a senior DevOps engineer with expertise in containerization, "
                     "orchestration, and CI/CD pipelines. Provide practical, production-ready "
                     "solutions with security and scalability considerations."
    )

    # Marketing content specialist
    marketing_preset = tb.Preset(
        name="marketing_specialist",
        tone="persuasive",
        expertise="copywriting,branding,digital_marketing",
        verbosity="descriptive",
        constraints=["brand_appropriate", "target_audience_aware", "conversion_focused"],
        system_prompt="You are a marketing specialist who creates compelling content. "
                     "Focus on audience engagement, brand consistency, and clear calls to action."
    )

    print(f"DevOps preset: {devops_preset.name}")
    print(f"Expertise areas: {devops_preset.expertise}")
    print(f"Constraints: {', '.join(devops_preset.constraints)}")
    ```

    ### Educational and training presets

    Create presets for educational and training scenarios:

    ```python
    # Programming tutor for beginners
    tutor_preset = tb.Preset(
        name="programming_tutor",
        tone="encouraging",
        expertise="python,programming_fundamentals,debugging",
        verbosity="thorough",
        constraints=["beginner_friendly", "step_by_step", "encouraging", "no_jargon"],
        system_prompt="You are a patient programming tutor. Break down complex concepts "
                     "into simple steps, provide encouragement, and use analogies to help "
                     "beginners understand programming concepts."
    )

    # Research assistant for academics
    research_preset = tb.Preset(
        name="research_assistant",
        tone="scholarly",
        expertise="academic_research,citation,methodology",
        verbosity="precise",
        constraints=["peer_reviewed_sources", "proper_citations", "methodology_aware"],
        system_prompt="You are an academic research assistant. Provide well-sourced "
                     "information with proper citations, maintain scholarly rigor, and "
                     "suggest appropriate research methodologies."
    )

    # Language learning coach
    language_preset = tb.Preset(
        name="language_coach",
        tone="supportive",
        expertise="language_learning,grammar,pronunciation",
        verbosity="detailed",
        constraints=["culturally_sensitive", "progressive_difficulty", "practice_focused"],
        system_prompt="You are a language learning coach. Provide clear explanations "
                     "of grammar rules, suggest practice exercises, and offer cultural "
                     "context for language usage."
    )
    ```

    ### Industry-specific service presets

    Design presets for specific industry applications:

    ```python
    # Healthcare information assistant
    healthcare_preset = tb.Preset(
        name="healthcare_info",
        tone="professional",
        expertise="medical_information,health_education",
        verbosity="thorough",
        constraints=[
            "no_diagnosis",
            "no_treatment_advice",
            "medical_disclaimers",
            "professional_referral"
        ],
        system_prompt="You provide general health information for educational purposes only. "
                     "Always include appropriate medical disclaimers and recommend consulting "
                     "healthcare professionals for personal medical concerns."
    )

    # Financial planning assistant
    finance_preset = tb.Preset(
        name="financial_planner",
        tone="advisory",
        expertise="personal_finance,investment_basics,budgeting",
        verbosity="detailed",
        constraints=[
            "no_specific_investment_advice",
            "risk_awareness",
            "educational_only",
            "regulatory_compliant"
        ],
        system_prompt="You provide general financial education and planning concepts. "
                     "Emphasize the importance of individual circumstances and professional "
                     "financial advice for specific investment decisions."
    )

    # Real estate advisor
    realestate_preset = tb.Preset(
        name="realestate_advisor",
        tone="consultative",
        expertise="real_estate,market_analysis,property_investment",
        verbosity="comprehensive",
        constraints=["market_aware", "location_specific", "risk_transparent"],
        system_prompt="You are a real estate advisor providing market insights and "
                     "property guidance. Consider local market conditions, investment "
                     "risks, and individual financial situations."
    )
    ```

    ### Creative and content generation presets

    Build presets for creative and content creation tasks:

    ```python
    # Storytelling specialist
    storyteller_preset = tb.Preset(
        name="storyteller",
        tone="narrative",
        expertise="creative_writing,character_development,plot_structure",
        verbosity="descriptive",
        constraints=["engaging_narrative", "character_consistency", "genre_appropriate"],
        system_prompt="You are a master storyteller. Create engaging narratives with "
                     "well-developed characters, compelling plots, and vivid descriptions "
                     "that immerse readers in the story world."
    )

    # Technical writer
    techwriter_preset = tb.Preset(
        name="technical_writer",
        tone="clear",
        expertise="documentation,technical_communication,user_guides",
        verbosity="precise",
        constraints=["user_focused", "step_by_step", "accuracy_critical", "accessible"],
        system_prompt="You are a technical writer who creates clear, user-friendly "
                     "documentation. Focus on step-by-step instructions, logical "
                     "organization, and accessibility for your target audience."
    )

    # Social media specialist
    social_preset = tb.Preset(
        name="social_media_manager",
        tone="engaging",
        expertise="social_media,content_strategy,audience_engagement",
        verbosity="concise",
        constraints=["platform_appropriate", "brand_voice", "engagement_focused"],
        system_prompt="You create engaging social media content optimized for specific "
                     "platforms. Maintain brand voice while maximizing audience engagement "
                     "and encouraging interaction."
    )
    ```

    ### Converting presets to configuration dictionaries

    Extract preset data for integration with other systems:

    ```python
    # Convert preset to dictionary for storage or API transmission
    preset_data = tutor_preset.to_dict()

    print("Preset configuration:")
    for key, value in preset_data.items():
        if isinstance(value, list):
            print(f"  {key}: {', '.join(value)}")
        else:
            print(f"  {key}: {value}")

    # Example output structure:
    # {
    #     "name": "programming_tutor",
    #     "tone": "encouraging",
    #     "expertise": "python,programming_fundamentals,debugging",
    #     "verbosity": "thorough",
    #     "constraints": ["beginner_friendly", "step_by_step", "encouraging", "no_jargon"],
    #     "system_prompt": "You are a patient programming tutor..."
    # }
    ```

    ### Preset validation and quality checks

    You can implement validation workflows to ensure preset quality and consistency.
    This includes checking for required fields, naming conventions, constraint formats,
    and system prompt completeness.

    Design Guidelines
    ----------------
    **Naming Conventions**: Use descriptive snake_case names that clearly indicate the preset's
    purpose and domain (e.g., "customer_support", "technical_advisor", "creative_writer").

    **Tone Selection**: Choose tones that match the intended use case and audience expectations.
    Consider how formal or informal the interaction should be.

    **Expertise Specification**: Be specific about domains while keeping them broad enough for
    flexibility. Use comma-separated lists for multiple expertise areas.

    **Constraint Design**: Include constraints that prevent inappropriate behavior for the
    specific use case. Consider legal, ethical, and practical limitations.

    **System Prompt Quality**: Write clear, specific system prompts that provide concrete
    guidance to the language model. Include examples when helpful.

    **Flexibility Balance**: Design presets to be opinionated enough to be useful while
    remaining flexible enough for customization through individual parameter overrides.

    Integration Notes
    ----------------
    - **Serialization**: all preset data serializes to JSON via `to_dict()` for storage and transmission
    - **Immutability**: presets are dataclasses and should be treated as immutable templates
    - **Validation**: consider implementing validation logic for preset consistency and completeness
    - **Versioning**: for production use, consider versioning presets to manage behavioral changes
    - **Documentation**: maintain clear documentation of available presets and their intended use cases

    The Preset class enables rapid deployment of specialized chatbot behaviors, making it easy to
    create consistent, purpose-built AI assistants for specific domains and use cases.
    """

    name: str
    tone: str
    expertise: str
    verbosity: str
    constraints: list[str]
    system_prompt: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        """Convert preset to dictionary format."""
        return {
            "name": self.name,
            "tone": self.tone,
            "expertise": self.expertise,
            "verbosity": self.verbosity,
            "constraints": self.constraints,
            "system_prompt": self.system_prompt,
        }


class PresetNames:
    """
    Predefined preset names for autocomplete and type safety.

    This class provides constants for all available preset names, enabling IDE
    autocomplete and preventing typos in preset names. Use these constants instead
    of string literals when calling the preset() method.

    Values
    ------
    The following preset names are available:

    **Business and Support:**
    - `CUSTOMER_SUPPORT`: Polite, professional customer service interactions
    - `LEGAL_ADVISOR`: Professional legal information with appropriate disclaimers

    **Technical and Development:**
    - `TECHNICAL_ADVISOR`: Authoritative technical guidance with detailed explanations
    - `DATA_ANALYST`: Analytical, evidence-based responses for data science tasks

    **Creative and Content:**
    - `CREATIVE_WRITER`: Imaginative storytelling and creative content generation

    Examples
    --------
    Use these constants instead of strings for better autocomplete:

    ```python
    import talk_box as tb

    # Use constants for autocomplete and type safety
    bot = tb.ChatBot().preset(tb.PresetNames.TECHNICAL_ADVISOR)

    # Instead of error-prone strings
    bot = tb.ChatBot().preset("technical_advisor")  # Risk of typos
    ```
    """

    # Business and Support presets
    CUSTOMER_SUPPORT = "customer_support"
    LEGAL_ADVISOR = "legal_advisor"

    # Technical and Development presets
    TECHNICAL_ADVISOR = "technical_advisor"
    DATA_ANALYST = "data_analyst"

    # Creative and Content presets
    CREATIVE_WRITER = "creative_writer"


class PresetManager:
    """
    Centralized manager for loading, storing, and applying chatbot behavior presets.

    The `PresetManager` class serves as the central registry and orchestrator for all
    behavior presets in the Talk Box system. It provides comprehensive functionality
    for preset lifecycle management including loading default presets, adding custom
    presets, applying presets to chatbot configurations, and managing preset collections
    for different domains or teams.

    The manager automatically loads a curated library of default presets covering common
    use cases like customer support, technical advisory, creative writing, and data analysis.
    It also provides flexible APIs for adding custom presets, removing presets, and
    applying preset configurations to chatbot instances with proper precedence handling.

    The class is designed to be extensible and can be integrated with external preset
    sources like databases, configuration files, or remote preset repositories. It
    handles preset validation, conflict resolution, and provides comprehensive preset
    discovery and introspection capabilities.

    Notes
    -----
    The PresetManager takes no initialization parameters. Default presets are
    automatically loaded during initialization, and the manager is ready for
    immediate use.

    Returns
    -------
    PresetManager
        A new PresetManager instance with default presets loaded and ready for use.

    Core Preset Operations
    ---------------------
    The PresetManager provides comprehensive preset management capabilities:

    - [`get_preset()`](`talk_box.PresetManager.get_preset`): Retrieve presets by name
    - [`add_preset()`](`talk_box.PresetManager.add_preset`): Add custom presets
    - [`remove_preset()`](`talk_box.PresetManager.remove_preset`): Remove presets
    - [`list_presets()`](`talk_box.PresetManager.list_presets`): List available presets
    - [`apply_preset()`](`talk_box.PresetManager.apply_preset`): Apply presets to configurations

    Default Preset Library
    ----------------------
    The manager includes a comprehensive library of default presets:

    - `"customer_support"`: Polite, professional customer service interactions
    - `"technical_advisor"`: Authoritative technical guidance with detailed explanations
    - `"creative_writer"`: Imaginative storytelling and creative content generation
    - `"data_analyst"`: Analytical, evidence-based data insights and interpretation
    - `"legal_advisor"`: Professional legal information with appropriate disclaimers

    Each default preset is carefully crafted with appropriate tone, expertise areas,
    verbosity levels, constraints, and system prompts for optimal performance in
    their respective domains.

    Configuration Application
    ------------------------
    When presets are applied to chatbot configurations, the manager handles intelligent
    merging where existing configuration values take precedence over preset defaults.
    This allows for preset-based initialization while preserving explicit user settings.

    Examples
    --------
    ### Basic preset management operations

    Get and use presets from the default library:

    ```python
    import talk_box as tb

    # Create a preset manager (loads defaults automatically)
    manager = tb.PresetManager()

    # List available presets
    available_presets = manager.list_presets()
    print(f"Available presets: {', '.join(available_presets)}")

    # Get a specific preset
    tech_preset = manager.get_preset("technical_advisor")
    if tech_preset:
        print(f"Tech preset tone: {tech_preset.tone}")
        print(f"Expertise: {tech_preset.expertise}")
        print(f"Constraints: {', '.join(tech_preset.constraints)}")

    # Apply preset to a chatbot
    bot = ChatBot().preset("technical_advisor")
    ```

    ### Adding custom presets for specialized domains

    Create and register custom presets for specific business needs:

    ```python
    # Create a custom preset for e-commerce support
    ecommerce_preset = tb.Preset(
        name="ecommerce_support",
        tone="helpful",
        expertise="product_knowledge,order_management,returns",
        verbosity="detailed",
        constraints=["policy_compliant", "customer_first", "solution_oriented"],
        system_prompt="You are an e-commerce customer support specialist. Help customers "
                     "with orders, products, returns, and general shopping questions. "
                     "Always prioritize customer satisfaction while following company policies."
    )

    # Add to manager
    manager = tb.PresetManager()
    manager.add_preset(ecommerce_preset)

    # Verify it was added
    if "ecommerce_support" in manager.list_presets():
        print("✅ E-commerce preset successfully added")

    # Use the custom preset
    ecommerce_bot = tb.ChatBot().preset("ecommerce_support")
    ```

    ### Building domain-specific preset collections

    Create specialized preset collections for different teams or applications:

    ```python
    # Educational presets
    educational_presets = [
        tb.Preset(
            name="math_tutor",
            tone="patient",
            expertise="mathematics,problem_solving,step_by_step_explanation",
            verbosity="thorough",
            constraints=["show_work", "encourage_learning", "no_direct_answers"],
            system_prompt="You are a math tutor. Guide students through problems step-by-step."
        ),
        tb.Preset(
            name="science_teacher",
            tone="enthusiastic",
            expertise="physics,chemistry,biology,scientific_method",
            verbosity="descriptive",
            constraints=["age_appropriate", "safety_conscious", "experiment_focused"],
            system_prompt="You are a science teacher who makes science exciting and accessible."
        ),
        tb.Preset(
            name="language_coach",
            tone="encouraging",
            expertise="grammar,vocabulary,pronunciation,cultural_context",
            verbosity="detailed",
            constraints=["culturally_sensitive", "progressive_difficulty"],
            system_prompt="You are a language learning coach. Provide clear explanations and practice."
        )
    ]

    # Add all educational presets
    manager = tb.PresetManager()
    for preset in educational_presets:
        manager.add_preset(preset)

    # Create specialized chatbots
    math_bot = tb.ChatBot().preset("math_tutor").temperature(0.3)
    science_bot = tb.ChatBot().preset("science_teacher").temperature(0.7)
    language_bot = tb.ChatBot().preset("language_coach").temperature(0.5)

    print(f"Created {len(educational_presets)} educational chatbots")
    ```

    ### Preset application with configuration merging

    Understand how presets merge with explicit configuration settings:

    ```python
    manager = tb.PresetManager()

    # Start with base configuration
    base_config = {
        "model": "gpt-4-turbo",
        "temperature": 0.5,
        "max_tokens": None,  # Will be filled by preset
        "tools": ["web_search"],
        "preset": None,
        "persona": None
    }

    # Apply technical advisor preset
    tech_config = manager.apply_preset("technical_advisor", base_config)

    print("Configuration after applying preset:")
    for key, value in tech_config.items():
        if value != base_config.get(key):
            print(f"  {key}: {base_config.get(key)} → {value}")
        else:
            print(f"  {key}: {value} (unchanged)")

    # Note: Existing values in base_config take precedence over preset values
    # Only None or missing values get filled from the preset
    ```

    ### Dynamic preset management and updates

    Manage presets dynamically during application runtime:

    ```python
    manager = tb.PresetManager()

    # Check if preset exists before removal
    if manager.get_preset("legal_advisor"):
        print("Legal advisor preset is available")

        # Remove preset (returns True if removed)
        if manager.remove_preset("legal_advisor"):
            print("Legal advisor preset removed")
        else:
            print("Failed to remove preset")

    # Add updated version
    updated_legal_preset = tb.Preset(
        name="legal_advisor_v2",
        tone="professional",
        expertise="legal_information,compliance,risk_assessment",
        verbosity="comprehensive",
        constraints=["disclaimer_required", "no_personal_advice", "cite_sources"],
        system_prompt="You are a legal information assistant. Provide general legal "
                     "information with appropriate disclaimers and source citations."
    )

    manager.add_preset(updated_legal_preset)
    print(f"Updated preset collection: {len(manager.list_presets())} presets available")
    ```

    ### Preset validation and quality assurance

    The PresetManager can be extended with validation workflows to ensure preset quality,
    consistency checks, and comprehensive validation of all presets in the collection.

    ### Integration with external preset sources

    The manager can be extended to work with external preset sources such as JSON files,
    databases, or remote APIs for loading and synchronizing preset collections.

    ### Preset performance monitoring and analytics

    Advanced implementations can track preset usage patterns, popularity metrics, and
    performance analytics to optimize preset collections and identify improvement opportunities.

    Advanced Features
    ----------------
    **Preset Inheritance**: Future versions may support preset inheritance where specialized
    presets can extend base presets with additional or overridden configuration.

    **Conditional Logic**: System prompts can include conditional logic based on context
    or user characteristics for dynamic behavior adaptation.

    **Preset Versioning**: Consider implementing version management for presets to handle
    updates and rollbacks in production environments.

    **Team Collaboration**: The manager can be extended to support team-based preset
    sharing and collaborative preset development workflows.

    **Performance Optimization**: Large preset collections can benefit from lazy loading
    and caching strategies for improved performance.

    Integration Notes
    ----------------
    - **Thread Safety**: `PresetManager` is not thread-safe; use external synchronization for concurrent access
    - **Memory Usage**: all presets are stored in memory; consider external storage for large collections
    - **Persistence**: the manager doesn't automatically persist changes; implement external persistence as needed
    - **Validation**: consider implementing comprehensive preset validation for production use
    - **Namespace Management**: use clear naming conventions to avoid preset name conflicts

    The `PresetManager` class provides a robust foundation for managing chatbot behavior templates,
    enabling consistent deployment of specialized AI assistants across different domains and use cases.
    """

    def __init__(self) -> None:
        """Initialize the preset manager with default presets."""
        self._presets: dict[str, Preset] = {}
        self._load_default_presets()

    def _load_default_presets(self) -> None:
        """Load the default preset library."""
        default_presets = {
            "customer_support": Preset(
                name="customer_support",
                tone="polite",
                expertise="product knowledge",
                verbosity="concise",
                constraints=["no_slang", "no_opinions"],
                system_prompt="You are a helpful customer support representative.",
            ),
            "technical_advisor": Preset(
                name="technical_advisor",
                tone="authoritative",
                expertise="python,ml",
                verbosity="detailed",
                constraints=["no_jokes"],
                system_prompt="You are a senior technical advisor with expertise in Python and machine learning.",
            ),
            "creative_writer": Preset(
                name="creative_writer",
                tone="imaginative",
                expertise="storytelling,creativity",
                verbosity="descriptive",
                constraints=["family_friendly"],
                system_prompt="You are a creative writer who helps with storytelling and creative projects.",
            ),
            "data_analyst": Preset(
                name="data_analyst",
                tone="analytical",
                expertise="data_analysis,statistics",
                verbosity="precise",
                constraints=["evidence_based"],
                system_prompt="You are a data analyst who provides insights based on data and statistical analysis.",
            ),
            "legal_advisor": Preset(
                name="legal_advisor",
                tone="professional",
                expertise="legal_knowledge",
                verbosity="thorough",
                constraints=["no_personal_advice", "disclaimers_required"],
                system_prompt="You are a legal information assistant. Always include appropriate disclaimers.",
            ),
        }

        for name, preset in default_presets.items():
            self._presets[name] = preset

    def get_preset(self, name: str) -> Optional[Preset]:
        """Get a preset by name."""
        return self._presets.get(name)

    def add_preset(self, preset: Preset) -> None:
        """Add a new preset to the manager."""
        self._presets[preset.name] = preset

    def list_presets(self) -> list[str]:
        """List all available preset names."""
        return list(self._presets.keys())

    def remove_preset(self, name: str) -> bool:
        """Remove a preset by name. Returns True if preset was removed."""
        if name in self._presets:
            del self._presets[name]
            return True
        return False

    def apply_preset(self, preset_name: str, config: dict[str, Any]) -> dict[str, Any]:
        """Apply a preset to a configuration dictionary."""
        preset = self.get_preset(preset_name)
        if preset is None:
            raise ValueError(f"Preset '{preset_name}' not found")

        # Create a copy of the config to avoid modifying the original
        updated_config = config.copy()

        # Apply preset values (config values take precedence over preset)
        preset_dict = preset.to_dict()
        for key, value in preset_dict.items():
            if key not in updated_config or updated_config[key] is None:
                updated_config[key] = value

        return updated_config
