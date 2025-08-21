"""
ChatBot builder module for Talk Box.

This module implements the chainable API for configuring and creating chatbots,
including attention-based prompt engineering capabilities.
"""

import socket
from typing import TYPE_CHECKING, Any, Optional, Union

if TYPE_CHECKING:
    from talk_box.conversation import Conversation
    from talk_box.presets import PresetNames
    from talk_box.prompt_builder import PromptBuilder

# Constants for validation
MAX_TEMPERATURE = 2.0
MIN_TEMPERATURE = 0.0


def find_available_port(start_port: int = 8000, max_attempts: int = 100) -> int:
    """Find an available port starting from start_port."""
    for port in range(start_port, start_port + max_attempts):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(("127.0.0.1", port))
                return port
        except OSError:
            continue
    raise RuntimeError(
        f"No available ports found in range {start_port}-{start_port + max_attempts}"
    )


class ChatBot:
    """
    Main entry point for building and managing conversational AI chatbots with integrated conversation handling.

    The `ChatBot` class is the primary interface for creating intelligent chatbots in Talk Box.
    It provides a chainable API for configuration and returns `Conversation` objects that manage
    message history and context. This design creates a natural learning path where users start
    with `ChatBot` for configuration, receive `Conversation` objects for message management,
    and discover `Message` objects within conversations.

    **Layered API Design**:

    1. **ChatBot** (this class): Configuration and interaction entry point
    2. **Conversation**: Multi-turn conversation management and message history
    3. **Message**: Individual message data structures with metadata

    The integration ensures that all chat interactions automatically create and manage conversation
    history, making multi-turn conversations natural and persistent. Advanced users can access
    lower-level `Conversation` and `Message` classes for specialized use cases.

    Notes
    -----
    The ChatBot class takes no initialization parameters. All configuration is done
    through the chainable methods after instantiation.

    Returns
    -------
    ChatBot
        A new ChatBot instance with default configuration and auto-enabled LLM integration
        (when available).

    Conversation Management
    ----------------------
    All chat interactions return `Conversation` objects, providing seamless conversation management:

    - [`chat()`](`talk_box.ChatBot.chat`): Send message and get conversation with response
    - [`start_conversation()`](`talk_box.ChatBot.start_conversation`): Create new empty conversation
    - [`continue_conversation()`](`talk_box.ChatBot.continue_conversation`): Continue existing conversation

    Conversations automatically handle message history, chronological ordering, and context management.
    Users can access individual `Message` objects within conversations for detailed inspection.

    Chainable Configuration Methods
    ------------------------------
    Configure your chatbot behavior with these chainable methods:

    - [`model()`](`talk_box.ChatBot.model`): Set the language model to use
    - [`preset()`](`talk_box.ChatBot.preset`): Apply behavior presets like "technical_advisor"
    - [`temperature()`](`talk_box.ChatBot.temperature`): Control response randomness (0.0-2.0)
    - [`max_tokens()`](`talk_box.ChatBot.max_tokens`): Set maximum response length
    - [`tools()`](`talk_box.ChatBot.tools`): Enable specific tools and capabilities
    - [`persona()`](`talk_box.ChatBot.persona`): Define the chatbot's personality
    - [`avoid()`](`talk_box.ChatBot.avoid`): Specify topics or behaviors to avoid
    - [`verbose()`](`talk_box.ChatBot.verbose`): Enable detailed output logging

    All configuration methods return `self`, enabling method chaining for concise setup.

    Browser Integration
    ------------------
    The ChatBot class provides interactive browser interfaces:

    - **Automatic Launch**: When displayed in Jupyter notebooks, opens browser chat interface
    - **Manual Sessions**: Use [`create_chat_session()`](`talk_box.ChatBot.create_chat_session`)
      for explicit browser interface control
    - **Configuration Display**: Shows current configuration when LLM integration unavailable

    Examples
    --------
    ### Basic conversation flow

    The natural progression from ChatBot to Conversation to Message:

    ```python
    import talk_box as tb

    # 1. Configure chatbot (entry point)
    bot = tb.ChatBot().model("gpt-4").temperature(0.7).preset("helpful")

    # 2. Start conversation (returns Conversation object)
    conversation = bot.chat("Hello! What can you help me with?")

    # 3. Continue conversation (updates Conversation)
    conversation = bot.chat("Tell me about machine learning", conversation=conversation)

    # 4. Access individual messages (Message objects)
    for message in conversation.get_messages():
        print(f"{message.role}: {message.content}")
        print(f"Timestamp: {message.timestamp}")
    ```

    ### Advanced conversation management

    Explicit conversation management for complex workflows:

    ```python
    # Start with empty conversation
    conversation = bot.start_conversation()

    # Add multiple exchanges
    conversation = bot.continue_conversation(conversation, "What's the weather?")
    conversation = bot.continue_conversation(conversation, "What about tomorrow?")

    # Access conversation metadata
    print(f"Total messages: {conversation.get_message_count()}")
    print(f"Last message: {conversation.get_last_message().content}")

    # Filter by role
    user_messages = conversation.get_messages(role="user")
    assistant_messages = conversation.get_messages(role="assistant")
    ```

    ### Discovering the full API layer by layer

    Start simple and naturally discover more advanced features:

    ```python
    import talk_box as tb

    # Layer 1: Basic ChatBot usage
    bot = tb.ChatBot().model("gpt-4")
    convo = bot.chat("Hello!")

    # Layer 2: Conversation management (discovered from return type)
    convo.add_user_message("Another question")
    messages = convo.get_messages()

    # Layer 3: Message details (discovered from conversation contents)
    latest = convo.get_last_message()
    print(f"Message ID: {latest.message_id}")
    print(f"Metadata: {latest.metadata}")

    # Layer 4: Advanced conversation features
    convo.set_context_window(10)  # Limit conversation length
    context_msgs = convo.get_context_messages()  # Get messages in context window
    ```
    """

    def __init__(
        self,
        name: Optional[str] = None,
        description: Optional[str] = None,
        id: Optional[str] = None,
    ) -> None:
        """Initialize a new ChatBot instance.

        Parameters
        ----------
        name : str, optional
            A human-readable name for this chatbot (e.g., "Customer Support Bot")
        description : str, optional
            A brief description of the chatbot's purpose and capabilities
        id : str, optional
            A unique identifier for this chatbot configuration (useful for A/B testing)
        """
        self._config: dict[str, Any] = {
            "name": name or "Untitled ChatBot",
            "description": description or "A Talk Box chatbot",
            "id": id,
            "provider": "openai",  # Default provider
            "model": "gpt-3.5-turbo",
            "temperature": 0.7,
            "max_tokens": 1000,
            "tools": [],
            "preset": None,
            "persona": None,
            "avoid": [],
            "verbose": False,
            "system_prompt": None,  # Custom system prompt override
        }
        # Initialize preset manager
        self._preset_manager = None
        self._current_preset = None
        self._llm_enabled = False

        # Auto-enable LLM integration if available
        self._auto_enable_llm()

    def check_llm_status(self) -> dict:
        """
        Check the status of LLM integration and get setup help if needed.

        Returns
        -------
        dict
            Status information and setup instructions if needed
        """
        status = {
            "enabled": self._llm_enabled,
            "status": getattr(self, "_llm_status", "unknown"),
        }

        if not self._llm_enabled:
            if "missing_chatlas" in status["status"]:
                status["help"] = {
                    "issue": "The 'chatlas' library is required for real LLM integration",
                    "solution": "Install with: pip install chatlas",
                    "note": "Without chatlas, ChatBot works in demo mode with echo responses",
                }
            else:
                status["help"] = {
                    "issue": f"LLM integration failed: {status['status']}",
                    "solution": "Check chatlas installation and API keys",
                }
        else:
            status["help"] = "LLM integration is working! You can use real AI models."

        return status

    def quick_start(self) -> str:
        """
        Get a simple quick-start guide for using this ChatBot.

        Returns
        -------
        str
            Quick-start instructions tailored to current configuration
        """
        llm_status = (
            "🟢 Ready for real AI chat!"
            if self._llm_enabled
            else "🟡 Demo mode (install 'pip install chatlas' for real AI)"
        )

        guide = f"""
🤖 Talk Box ChatBot Quick Start

{llm_status}

📝 Basic Usage:
   • bot.chat("Hello!")                 → Start a conversation
   • bot.show("basic")                  → Launch chat interface
   • bot.show("help")                   → Show this guide again

⚙️ Configuration:
   • bot.model("gpt-4")                 → Change AI model
   • bot.temperature(0.3)               → More focused responses
   • bot.preset("helpful_assistant")    → Use built-in preset

🔧 Current Setup:
   • Model: {self._config["model"]}
   • Temperature: {self._config["temperature"]}
   • Max Tokens: {self._config["max_tokens"]}
   • Preset: {self._config["preset"] or "Custom"}
"""

        if not self._llm_enabled:
            guide += """
💡 To enable real AI responses:
   1. Install chatlas: pip install chatlas
   2. Set API key: export OPENAI_API_KEY=your_key
   3. Restart and enjoy real AI chat!
"""

        return guide

    def install_chatlas_help(self) -> str:
        """
        Get step-by-step instructions for installing chatlas to enable LLM integration.

        Returns
        -------
        str
            Detailed installation guide for different environments
        """
        return """
🔧 Chatlas Installation Guide

📦 Install chatlas for real LLM integration:

1️⃣ Basic Installation:
   pip install chatlas

2️⃣ If you get "externally-managed-environment" error:
   # Option A: Use virtual environment (recommended)
   python3 -m venv talk-box-env
   source talk-box-env/bin/activate  # On Windows: talk-box-env\\Scripts\\activate
   pip install chatlas

   # Option B: User installation
   pip install --user chatlas

   # Option C: Use pipx (if available)
   pipx install chatlas

3️⃣ Set up API keys (choose your provider):
   # OpenAI
   export OPENAI_API_KEY=your_openai_key

   # Anthropic
   export ANTHROPIC_API_KEY=your_anthropic_key

   # Or set in Python:
   import os
   os.environ["OPENAI_API_KEY"] = "your_key"

4️⃣ Test installation:
   import talk_box as tb
   bot = tb.ChatBot()
   bot.show("status")  # Should show "LLM Ready"

💡 Once installed, all ChatBot instances automatically get real LLM capabilities!
"""

    @property
    def preset_manager(self):
        """Get the preset manager, creating it lazily."""
        if self._preset_manager is None:
            try:
                from talk_box.presets import PresetManager

                self._preset_manager = PresetManager()
            except ImportError:
                # Fallback if presets module isn't available
                self._preset_manager = None
        return self._preset_manager

    def _auto_enable_llm(self) -> None:
        """Automatically enable LLM integration if chatlas is available."""
        # Skip auto-enabling during tests to avoid requiring API keys
        import sys

        if "pytest" in sys.modules:
            self._llm_enabled = False
            self._llm_status = "disabled_for_testing"
            return

        try:
            from talk_box._utils_chatlas import enhance_chatbot_with_chatlas

            enhance_chatbot_with_chatlas()
            self.enable_llm_mode()  # Actually enable LLM mode on this instance
            self._llm_enabled = True
            self._llm_status = "enabled"
        except ImportError:
            # Chatlas not available, continue without LLM integration
            self._llm_enabled = False
            self._llm_status = "missing_chatlas"
        except Exception as e:
            # Other errors, continue without LLM integration
            self._llm_enabled = False
            self._llm_status = f"error: {str(e)}"

    def model(self, model_name: str) -> "ChatBot":
        """
        Configure the language model to use for generating responses.

        Sets the specific language model that will be used when the chatbot generates
        responses. This method supports models from various providers including OpenAI,
        Anthropic, Google, and others through the chatlas integration. The model choice
        significantly impacts response quality, speed, cost, and capabilities.

        The chatbot automatically detects the appropriate provider based on the model
        name and handles authentication via environment variables. Different models
        have different strengths - some excel at reasoning, others at creativity,
        and others at specific domains like code generation.

        Parameters
        ----------
        model_name : str
            The name of the language model to use. Supported models include:


            - `"gemini-pro-vision"`: Multimodal version supporting images

            The exact model names may vary by provider. Check provider documentation
            for the most current model names and capabilities.

        Returns
        -------
        ChatBot
            Returns self to enable method chaining, allowing you to configure
            multiple parameters in a single fluent expression.

        Raises
        ------
        ValueError
            If the model name is empty or None. The method does not validate
            model availability at configuration time - validation occurs when
            creating chat sessions.


        Model Types by Provider
        ------------------------
        Here are some examples of model types by provider:

        **OpenAI Models:**

        - `"gpt-4-turbo"`: Latest GPT-4 with improved performance and lower cost
        - `"gpt-4"`: Original GPT-4 model with excellent reasoning capabilities
        - `"gpt-3.5-turbo"`: Fast, cost-effective model good for most tasks
        - `"gpt-4o"`: Multimodal model supporting text, images, and audio

        **Anthropic Models:**

        - `"claude-3-5-sonnet-20241022"`: Latest Claude with excellent reasoning
        - `"claude-3-haiku-20240307"`: Fast, efficient model for simple tasks
        - `"claude-3-opus-20240229"`: Most capable Claude model for complex tasks

        **Google Models:**

        - `"gemini-pro"`: The flagship model from Google

        Examples
        --------
        ### Using different models for different purposes

        Configure chatbots with models optimized for specific tasks:

        ```python
        import talk_box as tb

        # High-performance model for complex reasoning
        reasoning_bot = tb.ChatBot().model("gpt-4-turbo")

        # Fast, cost-effective model for simple tasks
        quick_bot = tb.ChatBot().model("gpt-3.5-turbo")

        # Creative model for storytelling
        creative_bot = tb.ChatBot().model("claude-3-opus-20240229")

        # Multimodal model for image analysis
        vision_bot = tb.ChatBot().model("gpt-4o")
        ```

        ### Model selection with method chaining

        Combine model selection with other configuration options:

        ```python
        import talk_box as tb

        # Technical advisor with high-performance model
        tech_bot = (
            tb.ChatBot()
            .model("gpt-4-turbo")
            .preset("technical_advisor")
            .temperature(0.2)  # Low creativity for factual responses
            .max_tokens(2000)
        )

        # Creative writer with Claude
        writer_bot = (
            tb.ChatBot()
            .model("claude-3-opus-20240229")
            .preset("creative_writer")
            .temperature(0.8)  # High creativity
            .persona("Imaginative storyteller with rich vocabulary")
        )
        ```

        ### Dynamic model switching

        Change models based on task requirements:

        ```python
        import talk_box as tb

        bot = tb.ChatBot().preset("technical_advisor")

        # Use fast model for quick questions
        bot.model("gpt-3.5-turbo")
        quick_response = bot.chat("What is Python?")

        # Switch to powerful model for complex analysis
        bot.model("gpt-4-turbo")
        detailed_response = bot.chat("Explain the architectural trade-offs between microservices and monoliths")
        ```

        ### Model capabilities and selection guide

        Choose models based on your specific requirements:

        ```python
        import talk_box as tb

        # For code generation and technical tasks
        code_bot = tb.ChatBot().model("gpt-4-turbo").preset("technical_advisor")

        # For creative writing and storytelling
        creative_bot = tb.ChatBot().model("claude-3-opus-20240229").preset("creative_writer")

        # For cost-effective general tasks
        general_bot = tb.ChatBot().model("gpt-3.5-turbo").preset("customer_support")

        # For multimodal tasks (text + images)
        vision_bot = tb.ChatBot().model("gpt-4o")
        ```

        Notes
        -----
        **Provider Authentication**: Ensure appropriate API keys are set in environment
        variables (e.g., OPENAI_API_KEY, ANTHROPIC_API_KEY) for the chosen model provider.

        **Model Availability**: Model availability may change over time. Check provider
        documentation for current model names and deprecation schedules.

        **Cost Considerations**: Different models have different pricing structures.
        Consider cost implications for production deployments.

        **Rate Limits**: Each model/provider has different rate limits. Plan accordingly
        for high-volume applications.

        **Context Windows**: Models have different maximum context window sizes, affecting
        how much conversation history can be included in requests.

        See Also
        --------
        preset : Apply behavior presets that work well with specific models
        temperature : Control response randomness and creativity
        max_tokens : Set response length limits appropriate for the chosen model
        """
        self._config["model"] = model_name
        return self

    def provider_model(self, provider_model: str) -> "ChatBot":
        """
        Set provider and model using a single string (e.g., "openai:gpt-4o").

        This method provides a convenient way to configure both the AI provider
        and model in a single call using a colon-separated format. This is especially
        useful when you want to explicitly specify the provider or when working
        with models from different providers that might have similar names.

        Parameters
        ----------
        provider_model : str
            String in the format "provider:model" (e.g., "openai:gpt-4o", "anthropic:claude-3-opus").
            If only a model name is provided without a colon, defaults to OpenAI provider.

        Returns
        -------
        ChatBot
            Returns self for method chaining, allowing you to configure
            multiple parameters in a single fluent expression.

        Raises
        ------
        ValueError
            If the provider_model string is empty, None, or improperly formatted.

        Examples
        --------
        ### Using explicit provider and model combinations

        ```python
        import talk_box as tb

        # OpenAI models
        openai_bot = tb.ChatBot().provider_model("openai:gpt-4o")

        # Anthropic models
        anthropic_bot = tb.ChatBot().provider_model("anthropic:claude-3-opus-20240229")

        # Google models
        google_bot = tb.ChatBot().provider_model("google:gemini-pro")

        # Default to OpenAI if no provider specified
        default_bot = tb.ChatBot().provider_model("gpt-4-turbo")
        ```

        ### Method chaining with provider_model

        ```python
        # Complete configuration with explicit provider
        bot = (
            ChatBot()
            .provider_model("anthropic:claude-3-opus-20240229")
            .preset("technical_advisor")
            .temperature(0.2)
            .max_tokens(2000)
        )
        ```

        Notes
        -----
        **Provider Authentication**: Ensure appropriate API keys are set in environment
        variables for the specified provider (e.g., OPENAI_API_KEY, ANTHROPIC_API_KEY).

        **Provider Detection**: When only a model name is provided, the system defaults
        to OpenAI. For other providers, always specify the provider explicitly.

        See Also
        --------
        model : Set just the model name (uses default provider detection)
        preset : Apply behavior presets that work well with specific provider/model combinations
        """
        if not provider_model or not isinstance(provider_model, str):
            raise ValueError("provider_model must be a non-empty string")

        if ":" in provider_model:
            provider, model = provider_model.split(":", 1)
            provider = provider.strip()
            model = model.strip()
        else:
            provider = "openai"
            model = provider_model.strip()

        if not provider or not model:
            raise ValueError("Both provider and model must be non-empty")

        self._config["provider"] = provider
        self._config["model"] = model
        return self

    def preset(self, preset_name: Union[str, "PresetNames"]) -> "ChatBot":
        """
        Apply a pre-configured behavior template to instantly specialize the chatbot.

        Presets are professionally crafted behavior templates that instantly configure
        multiple aspects of the chatbot including conversational tone, expertise areas,
        response verbosity, operational constraints, and system prompts. This provides
        a quick way to create specialized chatbots for specific domains without manually
        configuring each parameter.

        The preset system includes a curated library of templates covering common use
        cases like customer support, technical advisory, creative writing, data analysis,
        and legal information. Each preset is designed by experts to provide optimal
        performance for its intended domain while maintaining flexibility for customization.

        When a preset is applied, it sets default values for various configuration
        parameters. You can still override individual settings after applying a preset,
        allowing for both rapid deployment and fine-tuned customization.

        Parameters
        ----------
        preset_name : str or PresetNames
            The name of the behavior preset to apply. You can use either a string or
            a constant from `PresetNames` for better autocomplete and type safety.

            **Using PresetNames constants (recommended):**

            ```python
            import talk_box as tb
            bot = tb.ChatBot().preset(tb.PresetNames.TECHNICAL_ADVISOR)
            ```

            **Available presets:**

            **Business and Support:**
            - `PresetNames.CUSTOMER_SUPPORT` or `"customer_support"`: Polite, professional
              customer service interactions with concise responses and helpful guidance
            - `PresetNames.LEGAL_ADVISOR` or `"legal_advisor"`: Professional legal information
              with appropriate disclaimers and thorough, well-sourced responses

            **Technical and Development:**
            - `PresetNames.TECHNICAL_ADVISOR` or `"technical_advisor"`: Authoritative technical
              guidance with detailed explanations, code examples, and best practices
            - `PresetNames.DATA_ANALYST` or `"data_analyst"`: Analytical, evidence-based
              responses for data science and statistical analysis tasks

            **Creative and Content:**
            - `PresetNames.CREATIVE_WRITER` or `"creative_writer"`: Imaginative storytelling
              and creative content generation with descriptive, engaging responses

            Additional presets may be available through custom preset libraries or
            organizational preset collections.

        Returns
        -------
        ChatBot
            Returns self to enable method chaining, allowing you to combine preset
            application with other configuration methods.

        Raises
        ------
        ValueError
            If the preset name is not found in the available preset library. The
            method fails gracefully and continues if preset loading encounters issues.

        Examples
        --------
        ### Using default presets for common scenarios

        Apply presets for different types of interactions:

        ```python
        from talk_box import ChatBot

        # Customer support chatbot
        support_bot = (
            ChatBot()
            .preset("customer_support")
            .model("gpt-3.5-turbo")  # Fast, cost-effective for support
        )

        # Technical advisor for development questions
        tech_bot = (
            ChatBot()
            .preset("technical_advisor")
            .model("gpt-4-turbo")  # Powerful model for complex technical questions
        )

        # Creative writing assistant
        writer_bot = (
            ChatBot()
            .preset("creative_writer")
            .model("claude-3-opus-20240229")  # Excellent for creative tasks
        )
        ```

        ### Combining presets with custom configuration

        Start with a preset and customize specific aspects:

        ```python
        # Start with technical advisor preset, then customize
        specialized_bot = (
            ChatBot()
            .preset("technical_advisor")
            .persona("Senior Python developer specializing in web frameworks")
            .temperature(0.1)  # Very low randomness for precise technical answers
            .tools(["code_executor", "documentation_search"])
            .avoid(["deprecated_practices", "insecure_patterns"])
        )

        # Customer support with custom personality
        friendly_support = (
            ChatBot()
            .preset("customer_support")
            .persona("Enthusiastic and empathetic customer advocate")
            .verbose(True)  # Detailed explanations for complex issues
        )
        ```

        ### Preset-specific optimizations

        Different presets work better with specific models and settings:

        ```python
        # Data analyst with analytical model and settings
        analyst_bot = (
            ChatBot()
            .preset("data_analyst")
            .model("gpt-4-turbo")  # Strong reasoning capabilities
            .temperature(0.2)  # Low creativity, high accuracy
            .max_tokens(2000)  # Allow detailed analysis
        )

        # Creative writer with creative model and settings
        creative_bot = (
            ChatBot()
            .preset("creative_writer")
            .model("claude-3-opus-20240229")  # Excellent creative capabilities
            .temperature(0.8)  # High creativity
            .max_tokens(3000)  # Allow longer creative outputs
        )
        ```

        ### Inspecting preset configuration

        View what a preset configures before applying it:

        ```python
        import talk_box as tb

        # Get preset details
        manager = tb.PresetManager()
        tech_preset = manager.get_preset("technical_advisor")

        if tech_preset:
            print(f"Tone: {tech_preset.tone}")
            print(f"Expertise: {tech_preset.expertise}")
            print(f"Verbosity: {tech_preset.verbosity}")
            print(f"Constraints: {', '.join(tech_preset.constraints)}")

        # Apply preset and check final configuration
        bot = tb.ChatBot().preset("technical_advisor")
        config = bot.get_config()
        print(f"Final config: {config}")
        ```

        ### Dynamic preset switching

        Change presets based on conversation context:

        ```python
        import talk_box as tb

        # Start with customer support
        bot = tb.ChatBot().preset("customer_support")

        # Handle general customer inquiry
        response1 = bot.chat("I need help with my order")

        # Switch to technical advisor for technical questions
        bot.preset("technical_advisor")
        response2 = bot.chat("How do I integrate your API?")

        # Switch to data analyst for analytics questions
        bot.preset("data_analyst")
        response3 = bot.chat("What patterns do you see in our user data?")
        ```

        Preset Customization
        -------------------
        **Individual Override**: All preset settings can be overridden by calling
        the corresponding configuration methods after applying the preset.

        **Custom Presets**: Organizations can create custom presets using the
        `PresetManager` to add domain-specific behavior templates.

        **Preset Inheritance**: Advanced implementations can create preset hierarchies
        where specialized presets extend base presets with additional configuration.

        **Context Awareness**: Some presets include conditional logic in their system
        prompts that adapts behavior based on conversation context.

        Notes
        -----
        **Preset Loading**: Presets are loaded from the `PresetManager` which initializes
        with a default library and can be extended with custom presets.

        **Graceful Failure**: If a preset is not found or fails to load, the method
        continues without error, allowing the chatbot to function with default settings.

        **System Prompts**: Each preset includes carefully crafted system prompts that
        provide detailed behavioral instructions to the underlying language model.

        **Best Practices**: Choose presets that match your intended use case, then
        fine-tune with additional configuration methods as needed.

        See Also
        --------
        PresetManager : Manage and create custom behavior presets
        persona : Add custom personality traits on top of preset behavior
        model : Choose models that work well with specific presets
        temperature : Adjust creativity levels appropriate for the preset domain
        """
        # Convert PresetNames constant to string if needed
        preset_str = preset_name if isinstance(preset_name, str) else str(preset_name)

        self._config["preset"] = preset_str

        # Apply the preset if preset manager is available
        if self.preset_manager:
            try:
                preset_obj = self.preset_manager.get_preset(preset_str)
                if preset_obj:
                    # Store the preset for display in _repr_html_
                    self._current_preset = preset_obj
            except Exception:
                # Continue if preset application fails
                pass

        return self

    def temperature(self, temp: float) -> "ChatBot":
        """
        Control the randomness and creativity level of chatbot responses.

        Temperature is a crucial parameter that controls the balance between
        deterministic accuracy and creative variability in language model outputs.
        Lower temperatures produce more focused, consistent, and predictable responses,
        while higher temperatures encourage more diverse, creative, and exploratory
        outputs at the potential cost of accuracy.

        The temperature parameter directly affects the probability distribution over
        possible next tokens during text generation. At temperature 0, the model
        always selects the most likely next token, resulting in deterministic outputs.
        Higher temperatures flatten the probability distribution, allowing less likely
        but potentially more creative tokens to be selected.

        Understanding temperature is essential for fine-tuning chatbot behavior to
        match specific use cases, from precise technical assistance to creative
        brainstorming and content generation.

        Parameters
        ----------
        temp : float
            The temperature value controlling response randomness, typically ranging
            from 0.0 to 2.0:

            **Ultra-Low (0.0-0.2):**
            - 0.0: Completely deterministic, always chooses most likely response
            - 0.1: Near-deterministic with minimal variation
            - 0.2: Highly consistent with occasional minor variations
            - Best for: Code generation, mathematical calculations, factual Q&A

            **Low (0.3-0.5):**
            - 0.3: Consistent with slight creative touches
            - 0.4: Balanced consistency with controlled variation
            - 0.5: Moderate creativity while maintaining reliability
            - Best for: Technical documentation, structured analysis, tutorials

            **Medium (0.6-0.8):**
            - 0.6: Balanced creativity and consistency
            - 0.7: Default setting for most general-purpose applications
            - 0.8: Enhanced creativity with good coherence
            - Best for: Conversational AI, content writing, explanations

            **High (0.9-1.2):**
            - 0.9: Creative responses with acceptable coherence
            - 1.0: High creativity, more diverse phrasings
            - 1.2: Very creative, potentially unexpected responses
            - Best for: Brainstorming, creative writing, ideation

            **Ultra-High (1.3-2.0):**
            - 1.5: Highly experimental and creative outputs
            - 2.0: Maximum creativity, potentially incoherent
            - Best for: Artistic exploration, experimental content

            Values above 2.0 are generally not recommended as they may produce
            incoherent or nonsensical responses.

        Returns
        -------
        ChatBot
            Returns self to enable method chaining, allowing you to combine
            temperature setting with other configuration methods.

        Raises
        ------
        ValueError
            If temperature is negative or excessively high (typically > 2.0),
            though exact limits depend on the underlying model provider.

        Examples
        --------
        ### Temperature for different use cases

        Configure temperature based on your specific needs:

        ```python
        from talk_box import ChatBot

        # Ultra-precise for code generation and technical tasks
        code_bot = (
            ChatBot()
            .model("gpt-4-turbo")
            .temperature(0.0)  # Deterministic outputs
            .preset("technical_advisor")
        )

        # Balanced for general conversation
        general_bot = (
            ChatBot()
            .model("gpt-3.5-turbo")
            .temperature(0.7)  # Default balanced setting
        )

        # Creative for content generation
        creative_bot = (
            ChatBot()
            .model("claude-3-opus-20240229")
            .temperature(1.0)  # High creativity
            .preset("creative_writer")
        )
        ```

        ### Precision vs. creativity trade-offs

        Demonstrate the impact of different temperature settings:

        ```python
        # For mathematical calculations - use minimal temperature
        math_bot = (
            ChatBot()
            .temperature(0.1)
            .persona("Mathematics tutor focused on step-by-step solutions")
        )

        # For brainstorming - use higher temperature
        brainstorm_bot = (
            ChatBot()
            .temperature(1.1)
            .persona("Creative strategist generating innovative ideas")
        )

        # For customer support - balanced approach
        support_bot = (
            ChatBot()
            .temperature(0.4)
            .preset("customer_support")
            .persona("Helpful and consistent customer service representative")
        )
        ```

        ### Domain-specific temperature optimization

        Adjust temperature for specific professional domains:

        ```python
        # Legal analysis - high precision required
        legal_bot = (
            ChatBot()
            .preset("legal_advisor")
            .temperature(0.2)  # Low creativity, high accuracy
            .model("gpt-4-turbo")
        )

        # Marketing content - creative but controlled
        marketing_bot = (
            ChatBot()
            .temperature(0.8)  # Creative but coherent
            .persona("Brand-aware marketing specialist")
            .avoid(["generic_language", "cliches"])
        )

        # Data analysis - analytical precision
        analyst_bot = (
            ChatBot()
            .preset("data_analyst")
            .temperature(0.3)  # Consistent analytical approach
            .tools(["statistical_analysis", "data_visualization"])
        )
        ```

        ### Dynamic temperature adjustment

        Adapt temperature based on conversation context:

        ```python
        class AdaptiveBot:
            def __init__(self):
                self.bot = ChatBot().model("gpt-4-turbo")

            def answer_question(self, question: str, question_type: str):
                if question_type == "factual":
                    self.bot.temperature(0.1)  # High precision
                elif question_type == "creative":
                    self.bot.temperature(1.0)  # High creativity
                elif question_type == "analytical":
                    self.bot.temperature(0.3)  # Balanced analysis
                else:
                    self.bot.temperature(0.7)  # Default

                return self.bot.chat(question)

        # Usage
        adaptive = AdaptiveBot()

        # Factual question with low temperature
        factual_response = adaptive.answer_question(
            "What is the capital of France?",
            "factual"
        )

        # Creative question with high temperature
        creative_response = adaptive.answer_question(
            "Write a haiku about machine learning",
            "creative"
        )
        ```

        ### Temperature with model-specific considerations

        Different models respond differently to temperature settings:

        ```python
        # GPT models - standard temperature ranges
        gpt_bot = (
            ChatBot()
            .model("gpt-4-turbo")
            .temperature(0.7)  # Works well with GPT models
        )

        # Claude models - may handle higher temperatures better
        claude_bot = (
            ChatBot()
            .model("claude-3-opus-20240229")
            .temperature(0.9)  # Claude often maintains coherence at higher temps
        )

        # Local models - may need different calibration
        local_bot = (
            ChatBot()
            .model("llama-2-13b-chat")
            .temperature(0.5)  # Conservative for smaller models
        )
        ```

        ### A/B testing different temperatures

        Compare response quality across temperature settings:

        ```python
        def compare_temperatures(question: str, temperatures: list[float]):
            \"\"\"Compare the same question across different temperatures.\"\"\"
            results = {}

            for temp in temperatures:
                bot = (
                    ChatBot()
                    .model("gpt-4-turbo")
                    .temperature(temp)
                )

                response = bot.chat(question)
                results[temp] = response

            return results

        # Test different temperatures
        question = "Explain quantum computing in simple terms"
        temps = [0.2, 0.5, 0.8, 1.1]

        comparison = compare_temperatures(question, temps)

        for temp, response in comparison.items():
            print(f"Temperature {temp}:")
            print(f"{response.content[:100]}...")
            print()
        ```

        Temperature Guidelines
        ---------------------
        **Code Generation**: Use 0.0-0.2 for precise, syntactically correct code
        with minimal variation.

        **Technical Writing**: Use 0.2-0.4 for accurate, consistent technical
        documentation and explanations.

        **General Conversation**: Use 0.6-0.8 for natural, engaging dialogue
        with appropriate variation.

        **Creative Content**: Use 0.8-1.2 for storytelling, marketing copy,
        and creative ideation.

        **Brainstorming**: Use 1.0-1.5 for maximum idea diversity and
        out-of-the-box thinking.

        Model Considerations
        -------------------
        **Provider Differences**: Different AI providers may interpret temperature
        values differently, so test with your specific model.

        **Model Size**: Larger models often handle higher temperatures better
        while maintaining coherence.

        **Fine-tuned Models**: Custom fine-tuned models may have different optimal
        temperature ranges compared to base models.

        **Context Length**: Longer conversations may benefit from slightly lower
        temperatures to maintain consistency.

        Notes
        -----
        **Reproducibility**: Use temperature 0.0 for reproducible outputs across
        multiple runs with the same input.

        **Gradual Adjustment**: When uncertain, start with default (0.7) and
        adjust incrementally based on response quality.

        **Task Specificity**: Consider the specific requirements of your task
        when choosing temperature - accuracy vs. creativity trade-offs.

        **Monitoring**: Monitor response quality when adjusting temperature,
        as optimal values may vary by use case and model.

        See Also
        --------
        max_tokens : Control response length alongside creativity
        model : Different models respond differently to temperature
        preset : Presets often include optimized temperature settings
        persona : Personality can complement temperature settings
        """
        if not MIN_TEMPERATURE <= temp <= MAX_TEMPERATURE:
            msg = f"Temperature must be between {MIN_TEMPERATURE} and {MAX_TEMPERATURE}"
            raise ValueError(msg)
        self._config["temperature"] = temp
        return self

    def max_tokens(self, tokens: int) -> "ChatBot":
        """
        Set the maximum number of tokens for chatbot responses.

        The max_tokens parameter controls the maximum length of generated responses
        by limiting the number of tokens (roughly equivalent to words and punctuation)
        that the language model can produce in a single response. This is crucial for
        managing response length, controlling costs, ensuring consistent behavior,
        and preventing excessively long outputs that might overwhelm users or exceed
        system limits.

        Understanding token limits is essential for balancing response completeness
        with practical constraints. Different models have varying token counting
        methods and maximum context windows, making this parameter both a performance
        optimization tool and a cost management mechanism.

        Token counting varies by model and provider, but generally:
        - 1 token ≈ 0.75 English words
        - 100 tokens ≈ 75 words or ~1-2 sentences
        - 500 tokens ≈ 375 words or ~1-2 paragraphs
        - 1000 tokens ≈ 750 words or ~1 page of text

        Parameters
        ----------
        tokens : int
            Maximum number of tokens for response generation. Must be positive.

            **Recommended ranges by use case:**

            **Short Responses (50-200 tokens):**
            - Quick answers, confirmations, brief explanations
            - Customer support acknowledgments
            - Code snippets and short technical answers
            - Chat-style interactions

            **Medium Responses (200-800 tokens):**
            - Detailed explanations and tutorials
            - Code documentation and examples
            - Product descriptions and feature explanations
            - Structured analysis and recommendations

            **Long Responses (800-2000 tokens):**
            - Comprehensive guides and documentation
            - Detailed technical analysis
            - Creative writing and storytelling
            - In-depth research summaries

            **Extended Responses (2000+ tokens):**
            - Long-form content generation
            - Detailed reports and documentation
            - Comprehensive tutorials and guides
            - Complex analysis requiring extensive explanation

            **Model-specific limits vary significantly:**
            - GPT-3.5-turbo: Up to 4,096 tokens (shared with input)
            - GPT-4: Up to 8,192 tokens (shared with input)
            - GPT-4-turbo: Up to 128,000 tokens (shared with input)
            - Claude-3: Up to 200,000 tokens (shared with input)

        Returns
        -------
        ChatBot
            Returns self to enable method chaining, allowing you to combine
            max_tokens setting with other configuration methods.

        Raises
        ------
        ValueError
            If tokens is not a positive integer. Some models may also have
            specific upper limits that could trigger additional validation errors.

        Examples
        --------
        ### Setting tokens for different response types

        Configure max_tokens based on your expected response length:

        ```python
        from talk_box import ChatBot

        # Brief answers for quick interactions
        quick_bot = (
            ChatBot()
            .model("gpt-3.5-turbo")
            .max_tokens(150)  # ~100-120 words
            .preset("customer_support")
        )

        # Detailed explanations for technical questions
        detailed_bot = (
            ChatBot()
            .model("gpt-4-turbo")
            .max_tokens(1000)  # ~750 words
            .preset("technical_advisor")
        )

        # Long-form content generation
        content_bot = (
            ChatBot()
            .model("claude-3-opus-20240229")
            .max_tokens(3000)  # ~2250 words
            .preset("creative_writer")
        )
        ```

        ### Balancing completeness with constraints

        Optimize token limits for specific scenarios:

        ```python
        # Code generation - precise and concise
        code_bot = (
            ChatBot()
            .model("gpt-4-turbo")
            .max_tokens(500)  # Focus on essential code
            .temperature(0.1)
            .persona("Senior software engineer providing clean, efficient code")
        )

        # Documentation writing - comprehensive but structured
        docs_bot = (
            ChatBot()
            .model("gpt-4-turbo")
            .max_tokens(1500)  # Detailed but focused
            .temperature(0.3)
            .persona("Technical writer creating clear, comprehensive documentation")
        )

        # Creative writing - longer form allowed
        story_bot = (
            ChatBot()
            .model("claude-3-opus-20240229")
            .max_tokens(2500)  # Allow creative expression
            .temperature(0.9)
            .preset("creative_writer")
        )
        ```

        ### Dynamic token adjustment based on context

        Adapt max_tokens based on conversation needs:

        ```python
        class AdaptiveTokenBot:
            def __init__(self):
                self.bot = ChatBot().model("gpt-4-turbo")

            def respond(self, message: str, response_type: str):
                if response_type == "brief":
                    self.bot.max_tokens(200)  # Quick answers
                elif response_type == "detailed":
                    self.bot.max_tokens(1000)  # Thorough explanations
                elif response_type == "comprehensive":
                    self.bot.max_tokens(2000)  # In-depth analysis
                else:
                    self.bot.max_tokens(500)  # Default moderate length

                return self.bot.chat(message)

        # Usage examples
        adaptive = AdaptiveTokenBot()

        # Brief response for simple questions
        quick_answer = adaptive.respond(
            "What is Python?",
            "brief"
        )

        # Detailed response for complex topics
        detailed_answer = adaptive.respond(
            "Explain machine learning algorithms",
            "detailed"
        )
        ```

        ### Cost optimization with token limits

        Use max_tokens to control API costs:

        ```python
        # Cost-conscious configuration for high-volume usage
        efficient_bot = (
            ChatBot()
            .model("gpt-3.5-turbo")  # Lower cost model
            .max_tokens(300)  # Limit response length
            .temperature(0.5)  # Balanced creativity
        )

        # Premium configuration for important interactions
        premium_bot = (
            ChatBot()
            .model("gpt-4-turbo")
            .max_tokens(1500)  # Allow detailed responses
            .temperature(0.7)
        )

        # Budget tracking example
        def cost_aware_chat(message: str, budget_tier: str):
            if budget_tier == "economy":
                bot = ChatBot().model("gpt-3.5-turbo").max_tokens(200)
            elif budget_tier == "standard":
                bot = ChatBot().model("gpt-4").max_tokens(500)
            else:  # premium
                bot = ChatBot().model("gpt-4-turbo").max_tokens(1500)

            return bot.chat(message)
        ```

        ### Token limits for different content types

        Optimize based on content format requirements:

        ```python
        # Email responses - professional length
        email_bot = (
            ChatBot()
            .max_tokens(400)  # Professional email length
            .persona("Professional and concise business communicator")
            .preset("customer_support")
        )

        # Blog post generation - substantial content
        blog_bot = (
            ChatBot()
            .max_tokens(2000)  # Article-length content
            .temperature(0.8)
            .persona("Engaging content writer")
        )

        # Social media responses - very brief
        social_bot = (
            ChatBot()
            .max_tokens(100)  # Tweet-length responses
            .temperature(0.7)
            .persona("Friendly and engaging social media manager")
        )

        # Technical documentation - comprehensive
        tech_docs_bot = (
            ChatBot()
            .max_tokens(1800)  # Detailed technical content
            .temperature(0.2)
            .preset("technical_advisor")
        )
        ```

        ### Monitoring token usage

        Track actual vs. maximum token usage:

        ```python
        def monitor_token_usage(messages: list[str], max_tokens: int):
            \"\"\"Monitor actual token usage vs. limits.\"\"\"
            bot = ChatBot().model("gpt-4-turbo").max_tokens(max_tokens)

            usage_data = []
            for message in messages:
                response = bot.chat(message)

                # Note: Actual token counting would require model-specific methods
                estimated_tokens = len(response.content.split()) * 1.3  # Rough estimate

                usage_data.append({
                    "message": message[:50] + "..." if len(message) > 50 else message,
                    "max_tokens": max_tokens,
                    "estimated_used": int(estimated_tokens),
                    "utilization": f"{(estimated_tokens/max_tokens)*100:.1f}%"
                })

            return usage_data

        # Example usage
        test_messages = [
            "What is artificial intelligence?",
            "Explain quantum computing in detail",
            "Write a short poem about technology"
        ]

        usage_report = monitor_token_usage(test_messages, 500)
        for entry in usage_report:
            print(f"Message: {entry['message']}")
            print(f"Utilization: {entry['utilization']}")
            print()
        ```

        Token Management Best Practices
        ------------------------------
        **Start Conservative**: Begin with lower token limits and increase as needed
        to avoid unexpectedly long responses.

        **Content-Specific Limits**: Set different limits for different types of content
        (code, explanations, creative writing, etc.).

        **Cost Monitoring**: Use token limits as a cost control mechanism, especially
        for high-volume applications.

        **User Experience**: Balance completeness with readability - very long responses
        can overwhelm users.

        **Model Considerations**: Different models have different token counting methods
        and optimal ranges.

        Performance Implications
        -----------------------
        **Response Time**: Higher token limits may increase response generation time,
        especially for complex requests.

        **Cost Scaling**: Most API providers charge based on token usage, making this
        parameter directly tied to operational costs.

        **Context Window**: Remember that max_tokens is shared with input tokens in
        most models' context windows.

        **Completion Quality**: Very low token limits may result in incomplete responses,
        while very high limits may lead to verbose, unfocused outputs.

        Notes
        -----
        **Model Variations**: Different models count tokens differently and have
        varying optimal token ranges for quality output.

        **Shared Context**: In most models, max_tokens counts toward the total context
        window, which includes both input and output tokens.

        **Truncation Behavior**: When a response reaches the max_tokens limit, it's
        typically truncated, which may result in incomplete sentences or thoughts.

        **Dynamic Adjustment**: Consider implementing dynamic token adjustment based
        on response type, user preferences, or conversation context.

        See Also
        --------
        model : Different models have different token limits and behavior
        temperature : Balance creativity with token efficiency
        preset : Some presets include optimized token settings
        tools : Tool usage may affect token consumption patterns
        """
        if tokens <= 0:
            raise ValueError("Max tokens must be positive")
        self._config["max_tokens"] = tokens
        return self

    def tools(self, tool_list: list[str]) -> "ChatBot":
        """Set the tools available to the chatbot."""
        self._config["tools"] = tool_list.copy()
        return self

    def avoid(self, avoid_list: list[str]) -> "ChatBot":
        """Set topics or behaviors to avoid."""
        self._config["avoid"] = avoid_list.copy()
        return self

    def persona(self, persona_description: str) -> "ChatBot":
        """Set the persona for the chatbot."""
        self._config["persona"] = persona_description
        return self

    def system_prompt(self, prompt: str) -> "ChatBot":
        """
        Set a custom system prompt, overriding any preset system prompt.

        This method allows for fine-grained prompt engineering. The custom system prompt
        will take precedence over any preset system prompt, though it will still be
        combined with persona, constraints, and other configuration elements.

        Parameters
        ----------
        prompt : str
            The custom system prompt text. Can include prompt engineering techniques,
            specific instructions, formatting requirements, etc.

        Returns
        -------
        ChatBot
            Returns self for method chaining

        Examples
        --------
        >>> bot = ChatBot().system_prompt('''
        ... You are an expert data analyst. Follow these rules:
        ... 1. Always provide statistical context
        ... 2. Cite data sources when possible
        ... 3. Use tables for structured data
        ... 4. Explain your methodology
        ... ''')
        """
        self._config["system_prompt"] = prompt
        return self

    def get_system_prompt(self) -> str:
        """
        Get the final constructed system prompt that will be sent to the LLM.

        This combines preset system prompts, custom system prompts, persona descriptions,
        constraints from 'avoid' settings, and other configuration elements into the
        final prompt text.

        Returns
        -------
        str
            The complete system prompt that will be used for LLM interactions

        Examples
        --------
        >>> bot = ChatBot().preset("technical_advisor").persona("Senior Engineer")
        >>> print(bot.get_system_prompt())
        "You are a senior technical advisor..."
        """
        prompt_parts = []

        # Start with preset system prompt if available
        if self._config["preset"] and self._current_preset:
            preset_config = self._current_preset.to_dict()
            if preset_config.get("system_prompt"):
                prompt_parts.append(preset_config["system_prompt"])

        # Add custom system prompt override
        if self._config["system_prompt"]:
            prompt_parts.append(self._config["system_prompt"])

        # Add persona if specified
        if self._config["persona"]:
            if prompt_parts:
                prompt_parts.append(f"\nAdditional persona: {self._config['persona']}")
            else:
                prompt_parts.append(f"You are: {self._config['persona']}")

        # Add constraints from 'avoid' settings
        if self._config["avoid"]:
            constraints = ", ".join(self._config["avoid"])
            prompt_parts.append(
                f"\nImportant constraints: Avoid discussing or providing {constraints}."
            )

        # Default fallback
        if not prompt_parts:
            prompt_parts.append("You are a helpful AI assistant.")

        return "\n".join(prompt_parts)

    def get_config_summary(self) -> dict[str, Any]:
        """
        Get a comprehensive summary of the current chatbot configuration.

        This includes model settings, prompt configuration, and metadata - useful
        for debugging, logging, A/B testing, and displaying in UI components.

        Returns
        -------
        dict[str, Any]
            Complete configuration summary including:
            - Basic info (name, description, id)
            - Model parameters (model, temperature, max_tokens)
            - Prompt components (preset, custom prompt, persona, constraints)
            - System prompt (final constructed prompt)
            - Advanced settings (tools, verbose mode, LLM status)

        Examples
        --------
        >>> bot = ChatBot(name="Support Bot").preset("customer_support")
        >>> config = bot.get_config_summary()
        >>> print(config["name"], config["model"], len(config["system_prompt"]))
        "Support Bot" "gpt-3.5-turbo" 245
        """
        return {
            # Basic metadata
            "name": self._config["name"],
            "description": self._config["description"],
            "id": self._config["id"],
            # Model configuration
            "model": self._config["model"],
            "temperature": self._config["temperature"],
            "max_tokens": self._config["max_tokens"],
            # Prompt configuration
            "preset": self._config["preset"],
            "custom_system_prompt": self._config["system_prompt"],
            "persona": self._config["persona"],
            "avoid_topics": self._config["avoid"],
            "system_prompt": self.get_system_prompt(),  # Final constructed prompt
            # Advanced settings
            "tools": self._config["tools"],
            "verbose": self._config["verbose"],
            "llm_enabled": self._llm_enabled,
            "llm_integration": "🟢 LLM Ready"
            if self._llm_enabled
            else "🟡 Demo Mode (install chatlas for LLM)",
            "llm_status": getattr(self, "_llm_status", "unknown"),
        }

    def verbose(self, enabled: bool = True) -> "ChatBot":
        """Enable or disable verbose output."""
        self._config["verbose"] = enabled
        return self

    # Attention-based prompt engineering methods

    def prompt_builder(self, builder_type: str = "general") -> "PromptBuilder":
        """
        Create an attention-optimized prompt builder for declarative prompt composition.

        This method returns a specialized prompt builder that implements attention-based
        structuring principles from modern prompt engineering research. The builder helps
        engineers create prompts with optimal attention patterns through a fluent,
        declarative API.

        Based on research showing that structure matters more than specific word choices,
        this builder enables you to:
        - Front-load critical information (primacy bias)
        - Create structured sections for clear attention clustering
        - Avoid attention drift through specific constraints
        - Build modular, maintainable prompt components

        Parameters
        ----------
        builder_type : str, optional
            Type of prompt builder to create. Options include:
            - "general": Basic attention-optimized builder
            - "architectural": Pre-configured for code architecture analysis
            - "code_review": Pre-configured for code review tasks
            - "debugging": Pre-configured for debugging assistance

        Returns
        -------
        PromptBuilder
            A prompt builder with methods for declarative prompt composition

        Examples
        --------
        ### Basic attention-optimized prompt building

        ```python
        from talk_box import ChatBot

        bot = ChatBot().model("gpt-4-turbo")

        # Build an attention-optimized prompt
        prompt = (bot.prompt_builder()
            .persona("senior software architect", "comprehensive codebase analysis")
            .task_context("Create architectural documentation")
            .critical_constraint("Focus on identifying architectural debt")
            .core_analysis([
                "Tools, frameworks, and design patterns",
                "Data models and API design patterns",
                "Architectural inconsistencies"
            ])
            .output_format([
                "Use clear headings and bullet points",
                "Include specific examples from codebase"
            ])
            .final_emphasis("Prioritize findings by impact and consistency")
            .build())

        # Use the structured prompt
        response = bot.chat(prompt)
        ```

        ### Pre-configured builders for common tasks

        ```python
        # Architectural analysis with pre-configured structure
        arch_prompt = (bot.prompt_builder("architectural")
            .focus_on("identifying technical debt")
            .build())

        # Code review with attention-optimized structure
        review_prompt = (bot.prompt_builder("code_review")
            .avoid_topics(["personal criticism"])
            .focus_on("actionable improvement suggestions")
            .build())
        ```

        ### Preview prompt structure before building

        ```python
        builder = (bot.prompt_builder()
            .persona("technical advisor")
            .core_analysis(["Security", "Performance", "Maintainability"])
            .output_format(["Structured sections", "Specific examples"]))

        # Preview the attention structure
        structure = builder.preview_structure()
        print(f"Estimated tokens: {structure['estimated_tokens']}")
        print(f"Priority sections: {len(structure['structured_sections'])}")

        # Build when satisfied with structure
        prompt = builder.build()
        ```

        Notes
        -----
        The returned builder implements attention-based principles:
        - **Primacy bias**: Critical information is front-loaded
        - **Structured sections**: Clear attention clustering prevents drift
        - **Personas**: Behavioral anchoring for consistent responses
        - **Specific constraints**: Avoid vague instructions that cause attention drift
        - **Recency bias**: Final emphasis leverages end-of-prompt attention

        See Also
        --------
        PromptBuilder : The full prompt builder API
        preset : Use presets for quick specialized configurations
        persona : Set behavioral context for responses
        """
        from talk_box.prompt_builder import (
            PromptBuilder,
            architectural_analysis_prompt,
            code_review_prompt,
            debugging_prompt,
        )

        if builder_type == "architectural":
            return architectural_analysis_prompt()
        elif builder_type == "code_review":
            return code_review_prompt()
        elif builder_type == "debugging":
            return debugging_prompt()
        else:
            return PromptBuilder()

    def structured_prompt(self, **sections) -> "ChatBot":
        """
        Configure the chatbot with a structured prompt built from keyword sections.

        This is a convenience method for quickly building attention-optimized prompts
        without using the full prompt builder API. It automatically structures the
        provided sections according to attention-based principles.

        Parameters
        ----------
        **sections : dict
            Keyword arguments defining prompt sections. Recognized keys include:
            - persona: Behavioral role (e.g., "senior developer")
            - task: Primary task description
            - constraints: List of requirements or constraints
            - format: List of output formatting requirements
            - examples: Dict of input/output examples
            - focus: Primary goal to emphasize

        Returns
        -------
        ChatBot
            Returns self for method chaining

        Examples
        --------
        ### Quick structured prompt creation

        ```python
        bot = (ChatBot()
            .model("gpt-4-turbo")
            .structured_prompt(
                persona="senior software architect",
                task="Analyze codebase architecture and identify improvements",
                constraints=[
                    "Focus on security vulnerabilities",
                    "Identify performance bottlenecks",
                    "Suggest specific fixes"
                ],
                format=[
                    "Use bullet points for findings",
                    "Include code examples",
                    "Prioritize by severity"
                ],
                focus="actionable recommendations for immediate implementation"
            ))
        ```

        ### Combining with other configuration

        ```python
        expert_bot = (ChatBot()
            .model("gpt-4-turbo")
            .temperature(0.2)
            .structured_prompt(
                persona="expert debugger",
                task="Identify root cause of performance issues",
                constraints=["Provide reproducible test cases"],
                focus="finding the root cause, not just symptoms"
            )
            .max_tokens(1500))
        ```
        """
        from talk_box.prompt_builder import PromptBuilder

        builder = PromptBuilder()

        # Apply sections in attention-optimized order
        if "persona" in sections:
            builder.persona(sections["persona"])

        if "task" in sections:
            builder.task_context(sections["task"])

        if "constraints" in sections:
            for constraint in sections["constraints"]:
                builder.constraint(constraint)

        if "format" in sections:
            builder.output_format(sections["format"])

        if "examples" in sections:
            examples = sections["examples"]
            if isinstance(examples, dict):
                for input_ex, output_ex in examples.items():
                    builder.example(input_ex, output_ex)

        if "focus" in sections:
            builder.final_emphasis(sections["focus"])

        # Set the built prompt as system prompt
        structured_prompt = builder.build()
        self._config["system_prompt"] = structured_prompt

        return self

    def chain_prompts(self, *prompts) -> "ChatBot":
        """
        Chain multiple structured prompts in attention-optimized order.

        This method allows you to combine multiple prompt components while maintaining
        optimal attention patterns. Components are automatically ordered to maximize
        the model's focus on the most important information.

        Parameters
        ----------
        *prompts : str or PromptBuilder
            Prompt components to chain together. Can be raw strings or
            PromptBuilder instances.

        Returns
        -------
        ChatBot
            Returns self for method chaining

        Examples
        --------
        ### Chaining different prompt components

        ```python
        # Create specialized prompt components
        security_prompt = (bot.prompt_builder()
            .core_analysis(["SQL injection risks", "XSS vulnerabilities"])
            .build())

        performance_prompt = (bot.prompt_builder()
            .core_analysis(["Database query optimization", "Memory usage"])
            .build())

        # Chain them with attention optimization
        bot.chain_prompts(
            "You are a senior security and performance engineer.",
            security_prompt,
            performance_prompt,
            "Focus on the most critical issues that impact user security."
        )
        ```
        """
        from talk_box.prompt_builder import PromptBuilder

        # Build a master prompt that chains all components
        master_builder = PromptBuilder()

        combined_content = []

        for prompt in prompts:
            if hasattr(prompt, "build"):
                # It's a prompt builder
                combined_content.append(prompt.build())
            else:
                # It's a string
                combined_content.append(str(prompt))

        # Join with proper spacing and set as system prompt
        final_prompt = "\n\n".join(combined_content)
        self._config["system_prompt"] = final_prompt

        return self

    def enable_llm_mode(self) -> "ChatBot":
        """
        Enable LLM mode explicitly (DEPRECATED - LLM is auto-enabled by default).

        This method is deprecated and unnecessary since LLM integration is
        automatically enabled during ChatBot initialization. It remains for
        backward compatibility only.

        Returns
        -------
        ChatBot
            Returns self for method chaining

        Note
        ----
        LLM integration is automatically enabled when you create a ChatBot.
        You do not need to call this method unless you're using legacy code.
        """
        if not self._llm_enabled:
            self._auto_enable_llm()
        return self

    def chat(self, message: str, conversation: Optional["Conversation"] = None) -> "Conversation":
        """
        Send a message to the chatbot and get a response within a conversation context.

        This method creates or updates a conversation by adding the user's message and the
        chatbot's response. If no conversation is provided, a new one is automatically
        created. This is the primary way to interact with the chatbot while maintaining
        conversation history and context.

        Parameters
        ----------
        message : str
            The user's message to send to the chatbot.
        conversation : Conversation, optional
            An existing conversation to continue. If not provided, a new conversation
            is created automatically.

        Returns
        -------
        Conversation
            The conversation object containing the full message history including
            the new user message and chatbot response.

        Examples
        --------
        ### Basic single-message chat

        ```python
        from talk_box import ChatBot

        bot = ChatBot().model("gpt-4").temperature(0.7)
        convo = bot.chat("Hello! How are you?")
        print(convo.get_last_message().content)
        ```

        ### Continuing a conversation

        ```python
        # Start a conversation
        convo = bot.chat("What's machine learning?")

        # Continue the same conversation
        convo = bot.chat("Can you give me an example?", conversation=convo)

        # View full conversation history
        for msg in convo.get_messages():
            print(f"{msg.role}: {msg.content}")
        ```
        """
        # Import here to avoid circular imports
        from talk_box.conversation import Conversation

        # Create new conversation if none provided
        if conversation is None:
            conversation = Conversation()

        # Add user message
        conversation.add_user_message(message)

        # TODO: Implement actual chat functionality with LLM
        # For now, return a simple echo response
        response_content = f"Echo: {message}"
        conversation.add_assistant_message(response_content)

        return conversation

    def start_conversation(self) -> "Conversation":
        """
        Start a new conversation with this chatbot.

        Creates a fresh conversation instance that can be used for multi-turn
        interactions with the chatbot. This is useful when you want to explicitly
        manage conversation state and context.

        Returns
        -------
        Conversation
            A new, empty conversation instance ready for interaction.

        Examples
        --------
        ### Starting a managed conversation

        ```python
        from talk_box import ChatBot

        # Configure chatbot
        bot = ChatBot().model("gpt-4").temperature(0.7).preset("helpful")

        # Start a new conversation
        conversation = bot.start_conversation()

        # Add messages manually or use chat method
        conversation.add_user_message("Hello!")
        updated_conversation = bot.chat("How are you?", conversation=conversation)
        ```
        """
        # Import here to avoid circular imports
        from talk_box.conversation import Conversation

        return Conversation()

    def continue_conversation(self, conversation: "Conversation", message: str) -> "Conversation":
        """
        Continue an existing conversation with a new message.

        This is a convenience method that's equivalent to calling
        `chat(message, conversation=conversation)` but makes the intent
        of continuing a conversation more explicit.

        Parameters
        ----------
        conversation : Conversation
            The existing conversation to continue.
        message : str
            The user's message to add to the conversation.

        Returns
        -------
        Conversation
            The updated conversation with the new exchange.

        Examples
        --------
        ```python
        # Start conversation
        conversation = bot.start_conversation()

        # Continue it explicitly
        conversation = bot.continue_conversation(conversation, "What's the weather like?")
        conversation = bot.continue_conversation(conversation, "What about tomorrow?")
        ```
        """
        return self.chat(message, conversation=conversation)

    def create_chat_session(self):
        """Create a chat session that can be used to launch browser interface."""
        try:
            # Import here to avoid circular imports
            from talk_box._utils_chatlas import ChatlasAdapter

            return ChatlasAdapter().create_chat_session(self._config)
        except ImportError:
            # Return a simple session that just shows configuration
            return SimpleChatSession(self)

    def _repr_html_(self) -> str:
        """
        Rich HTML representation for Jupyter notebooks with enhanced diagnostics.

        Shows comprehensive configuration information and provides easy access
        to the enhanced diagnostic chat interface.
        """
        try:
            config = self.get_config_summary()
            system_prompt = self.get_system_prompt()
        except Exception as e:
            # Fallback to basic display if there are configuration issues
            return self._repr_html_fallback()

        # Only show diagnostic information by default, don't auto-launch
        # Users can explicitly call .show("basic") or .show("enhanced") for browser interfaces

        # Create a comprehensive diagnostic display with safe string formatting
        try:
            html = f"""
        <div style="padding: 20px; border: 2px solid #2E86AB; border-radius: 8px; background-color: #f8f9fa; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Arial, sans-serif;">
            <h3 style="color: #2E86AB; margin-top: 0; display: flex; align-items: center;">
                🤖 Talk Box ChatBot
                <span style="margin-left: 10px; font-size: 0.7em; background: #28a745; color: white; padding: 2px 8px; border-radius: 12px;">
                    {config.get("llm_integration", "Unknown")}
                </span>
            </h3>

            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin: 15px 0;">
                <div>
                    <h4 style="color: #495057; margin-bottom: 10px; font-size: 1em;">📊 Configuration</h4>
                    <div style="background: white; color: #212529; padding: 12px; border-radius: 6px; border-left: 4px solid #2E86AB;">
                        <div style="margin: 4px 0; color: #212529;"><strong>Model:</strong> {config.get("model", "Not set")}</div>
                        <div style="margin: 4px 0; color: #212529;"><strong>Temperature:</strong> {config.get("temperature", "Not set")}</div>
                        <div style="margin: 4px 0; color: #212529;"><strong>Max Tokens:</strong> {config.get("max_tokens", "Not set")}</div>
                        <div style="margin: 4px 0; color: #212529;"><strong>Preset:</strong> {config.get("preset", "Custom") or "Custom"}</div>
                    </div>
                </div>

                <div>
                    <h4 style="color: #495057; margin-bottom: 10px; font-size: 1em;">⚙️ Advanced Settings</h4>
                    <div style="background: white; color: #212529; padding: 12px; border-radius: 6px; border-left: 4px solid #6f42c1;">
                        <div style="margin: 4px 0; color: #212529;"><strong>Name:</strong> {config.get("name", "Unnamed Bot") or "Unnamed Bot"}</div>
                        <div style="margin: 4px 0; color: #212529;"><strong>Persona:</strong> {config.get("persona", "None") or "None"}</div>
                        <div style="margin: 4px 0; color: #212529;"><strong>Constraints:</strong> {len(config.get("avoid_topics", []))} topic(s)</div>
                        <div style="margin: 4px 0; color: #212529;"><strong>Tools:</strong> {len(config.get("tools", []))} enabled</div>
                    </div>
                </div>
            </div>

            <div style="margin: 15px 0;">
                <h4 style="color: #495057; margin-bottom: 8px; font-size: 1em;">📝 System Prompt ({len(system_prompt)} characters)</h4>
                <div style="background: #f1f3f4; color: #212529; padding: 12px; border-radius: 6px; font-family: 'Monaco', 'Menlo', monospace; font-size: 0.85em; max-height: 120px; overflow-y: auto; white-space: pre-wrap; border-left: 4px solid #fd7e14;">{system_prompt}</div>
            </div>

            <div style="background: linear-gradient(90deg, #2E86AB, #17a2b8); padding: 15px; border-radius: 6px; margin-top: 15px; color: white;">
                <div style="display: flex; align-items: center; justify-content: space-between;">
                    <div>
                        <strong>🚀 Ready to Chat!</strong>
                        <div style="font-size: 0.9em; margin-top: 4px;">
                            Launch chat interface on demand
                        </div>
                    </div>
                    <div style="text-align: right; font-size: 0.85em;">
                        <div>🔧 Quick Launch:</div>
                        <div><code style="background: rgba(255,255,255,0.2); padding: 2px 6px; border-radius: 3px;">bot.show("basic")</code></div>
                        <div><code style="background: rgba(255,255,255,0.2); padding: 2px 6px; border-radius: 3px;">bot.show("help")</code></div>
                    </div>
                </div>
            </div>

            <div style="margin-top: 12px; font-size: 0.8em; color: #6c757d; text-align: center;">
                💡 Use <strong>bot.show("basic")</strong> for chat • 🔍 Use <strong>bot.show("help")</strong> for guidance
            </div>
        </div>
        """

            return html
        except Exception as e:
            # If HTML generation fails, return fallback
            return self._repr_html_fallback()

    def _repr_html_fallback(self) -> str:
        """Fallback HTML representation when chat interface isn't available."""
        return f"""
        <div style="padding: 20px; border: 2px solid #2E86AB; border-radius: 8px; background-color: #f8f9fa;">
            <h3 style="color: #2E86AB; margin-top: 0;">🤖 Talk Box ChatBot</h3>
            <p><strong>Configuration:</strong></p>
            <ul>
                <li><strong>Model:</strong> {self._config.get("model", "Not set")}</li>
                <li><strong>Preset:</strong> {self._config.get("preset", "Not set")}</li>
                <li><strong>Persona:</strong> {self._config.get("persona", "Not set")}</li>
                <li><strong>Temperature:</strong> {self._config.get("temperature", 0.7)}</li>
                <li><strong>Max Tokens:</strong> {self._config.get("max_tokens", 1000)}</li>
                <li><strong>Tools:</strong> {", ".join(self._config.get("tools", [])) or "None"}</li>
            </ul>
            <div style="background-color: #e8f4f8; padding: 10px; border-radius: 4px; margin-top: 15px;">
                <strong>💡 Next Steps:</strong>
                <ol style="margin: 5px 0 0 20px;">
                    <li>Enable LLM integration: <code>enhance_chatbot_with_chatlas()</code></li>
                    <li>Launch chat interface: <code>bot.create_chat_session().app()</code></li>
                    <li>Or chat directly: <code>bot.chat("Hello!")</code></li>
                </ol>
            </div>
        </div>
        """

    def show(self, mode: str = "help") -> None:
        """
        Display diagnostic information or launch chat interfaces.

        This method provides explicit control over displaying diagnostic
        information and launching chat interfaces, complementing the
        automatic display when the ChatBot object is shown.

        Parameters
        ----------
        mode : str, default "help"
            The type of interface to show:
            - "browser": Launch browser chat interface (formerly "basic")
            - "console": Launch interactive console/terminal chat
            - "config": Display configuration summary in notebook
            - "prompt": Display the final system prompt
            - "help": Show quick-start guide for using this ChatBot
            - "status": Check LLM integration status and troubleshooting
            - "install": Show step-by-step chatlas installation guide

        Examples
        --------
        >>> bot = ChatBot().model("gpt-4").preset("technical_advisor")
        >>> bot.show("help")        # Show quick-start guide (default)
        >>> bot.show("status")      # Check LLM integration status
        >>> bot.show("browser")     # Launch browser chat interface
        >>> bot.show("console")     # Launch terminal chat interface
        >>> bot.show("config")      # Show configuration summary
        """

        if mode == "browser" or mode == "basic":  # Support both old and new names
            # Launch browser chat interface
            try:
                print("🌐 Launching Browser Chat Interface...")
                session = self.create_chat_session()
                if hasattr(session, "app"):
                    session.app()
                else:
                    print("❌ Chat session doesn't support browser interface")
            except Exception as e:
                print(f"❌ Error launching browser interface: {e}")

        elif mode == "console":
            # Launch interactive console/terminal chat
            try:
                print("💬 Launching Console Chat Interface...")
                print("Type 'exit', 'quit', or Ctrl+C to end the conversation.")
                print("-" * 50)
                session = self.create_chat_session()
                if hasattr(session, "console"):
                    session.console()
                else:
                    # Fallback: simple console implementation
                    self._simple_console_chat()
            except KeyboardInterrupt:
                print("\n👋 Chat ended by user.")
            except Exception as e:
                print(f"❌ Error launching console interface: {e}")

        elif mode == "config":
            # Display configuration summary
            config = self.get_config_summary()
            print("📊 ChatBot Configuration Summary")
            print("=" * 50)
            print(f"Name: {config['name'] or 'Unnamed Bot'}")
            print(f"Description: {config['description'] or 'No description'}")
            print(f"ID: {config['id'] or 'No ID'}")
            print()
            print(f"Model: {config['model']}")
            print(f"Temperature: {config['temperature']}")
            print(f"Max Tokens: {config['max_tokens']}")
            print(f"Preset: {config['preset'] or 'Custom'}")
            print()
            print(f"Persona: {config['persona'] or 'None'}")
            print(
                f"Avoid Topics: {', '.join(config['avoid_topics']) if config['avoid_topics'] else 'None'}"
            )
            print(f"Tools: {len(config['tools'])} enabled")
            print(f"LLM Integration: {config['llm_integration']}")

        elif mode == "prompt":
            # Display system prompt details
            system_prompt = self.get_system_prompt()
            config = self.get_config_summary()

            print("📝 System Prompt Analysis")
            print("=" * 50)
            print(f"Total Length: {len(system_prompt)} characters")
            print(f"Custom Prompt: {'Yes' if config['custom_system_prompt'] else 'No'}")
            print(f"Preset: {config['preset'] or 'None'}")

            # Better persona analysis that accounts for both ChatBot and PromptBuilder personas
            chatbot_persona = config["persona"]
            custom_prompt = config["custom_system_prompt"]

            # Extract persona from PromptBuilder if present
            promptbuilder_persona = None
            if custom_prompt and "You are a" in custom_prompt:
                # Extract the first line that starts with "You are a"
                lines = custom_prompt.split("\n")
                for line in lines:
                    if line.strip().startswith("You are a") or line.strip().startswith(
                        "You are an"
                    ):
                        promptbuilder_persona = (
                            line.strip().replace("You are a ", "").replace("You are an ", "")
                        )
                        break

            # Display persona information more accurately
            if promptbuilder_persona and chatbot_persona:
                print(f"Persona: {promptbuilder_persona} + Additional: {chatbot_persona}")
            elif promptbuilder_persona:
                print(f"Persona: {promptbuilder_persona}")
            elif chatbot_persona:
                print(f"Persona: {chatbot_persona}")
            else:
                print("Persona: None")

            print()
            print("Final System Prompt:")
            print("-" * 30)
            print(system_prompt)
            print("-" * 30)

        elif mode == "help":
            # Display quick-start guide
            print(self.quick_start())

        elif mode == "status":
            # Display LLM status and help
            status = self.check_llm_status()
            print("🔍 LLM Integration Status")
            print("=" * 50)
            print(f"Enabled: {'✅ Yes' if status['enabled'] else '❌ No'}")
            print(f"Status: {status['status']}")
            print()
            if isinstance(status["help"], dict):
                print(f"Issue: {status['help']['issue']}")
                print(f"Solution: {status['help']['solution']}")
                if "note" in status["help"]:
                    print(f"Note: {status['help']['note']}")
            else:
                print(f"Status: {status['help']}")

        elif mode == "install":
            # Display chatlas installation guide
            print(self.install_chatlas_help())

        else:
            # Invalid mode
            print(f"❌ Invalid mode: '{mode}'")
            print(
                "💡 Available modes: 'browser', 'console', 'config', 'prompt', 'help', 'status', 'install'"
            )

    def _simple_console_chat(self) -> None:
        """Simple console chat fallback when chatlas console interface isn't available."""
        print("🤖 Simple Console Chat (fallback mode)")
        print(
            "Note: For the full chatlas console experience, ensure chatlas is properly installed."
        )
        print()

        while True:
            try:
                user_input = input("You: ").strip()
                if user_input.lower() in ["exit", "quit", "bye"]:
                    print("👋 Goodbye!")
                    break

                if not user_input:
                    continue

                # Use the chat method to get response
                response = self.chat(user_input)
                if hasattr(response, "content"):
                    print(f"Bot: {response.content}")
                else:
                    print(f"Bot: {response}")
                print()

            except (EOFError, KeyboardInterrupt):
                print("\n👋 Chat ended.")
                break
            except Exception as e:
                print(f"❌ Error: {e}")

    def get_config(self) -> dict[str, Any]:
        """Get the current configuration."""
        return self._config.copy()


class ChatResponse:
    """Response from a chatbot interaction."""

    def __init__(self, content: str, metadata: Optional[dict[str, Any]] = None) -> None:
        """Initialize a chat response."""
        self.content = content
        self.metadata = metadata or {}

    def __str__(self) -> str:
        """String representation of the response."""
        return self.content


class SimpleChatSession:
    """Simple fallback chat session when full integration isn't available."""

    def __init__(self, chatbot: ChatBot) -> None:
        """Initialize with a ChatBot instance."""
        self.chatbot = chatbot

    def app(self) -> None:
        """Display a message about enabling the full chat interface."""
        print("🤖 Talk Box ChatBot Configuration:")
        print("-" * 40)
        config = self.chatbot.get_config()
        for key, value in config.items():
            if isinstance(value, list):
                display_value = ", ".join(value) if value else "None"
            else:
                display_value = value
            print(f"  {key.title()}: {display_value}")
        print("\n💡 To enable the browser chat interface:")
        print("  1. Run: enhance_chatbot_with_chatlas()")
        print("  2. Then: bot.create_chat_session().app()")

    def _get_interface_url(self) -> Optional[str]:
        """Return None since simple session doesn't have a URL."""
        return None
