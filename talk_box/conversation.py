from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional
from uuid import uuid4


@dataclass
class Message:
    """
    Represents a single message in a conversational AI interaction, discovered through Conversation objects.

    The `Message` class is a fundamental data structure in Talk Box that users typically encounter
    when exploring `Conversation` objects returned by `ChatBot` methods. This creates a natural
    discovery path: `ChatBot` → `Conversation` → `Message`, where each layer provides progressively
    more detailed control over conversational AI interactions.

    **Discovery through the layered API**:

    1. Start with [`ChatBot.chat()`](`talk_box.ChatBot.chat`) → returns `Conversation`
    2. Explore [`Conversation.get_messages()`](`talk_box.Conversation.get_messages`) → returns list of `Message`
    3. Access individual `Message` properties for detailed message inspection

    Each message encapsulates all information about individual exchanges in conversations, including
    text content, role information, timestamp data, and extensible metadata for storing additional
    context or processing information. Messages are immutable by design and include automatic
    timestamp generation and unique identification.

    The class supports serialization to and from dictionary format for storage, transmission,
    and integration with external systems. The role-based system follows standard conversational
    AI patterns where different participants in the conversation are clearly identified.

    Parameters
    ----------
    content : str
        The actual text content of the message. This is the primary payload containing
        what was said or communicated in this message exchange.
    role : str
        The role of the message sender, indicating who or what generated this message.
        Standard roles include:
        - `"user"`: Messages from the human user
        - `"assistant"`: Messages from the AI chatbot or assistant
        - `"system"`: System-level messages, instructions, or metadata
        - `"function"`: Messages from function calls or tool executions
        Custom roles can be used for specialized conversation flows.
    timestamp : datetime, optional
        When the message was created. If not provided, automatically set to the current
        time using `datetime.now()`. This enables chronological ordering and time-based
        analysis of conversations.
    metadata : dict[str, Any], optional
        Additional metadata about the message as a dictionary. This extensible field
        can store any additional context, processing information, or custom data.
        Examples include token counts, confidence scores, source information, or
        custom application data. Defaults to an empty dictionary if not provided.
    message_id : str, optional
        Unique identifier for the message. If not provided, automatically generated
        using `uuid4()` to ensure global uniqueness. This enables reliable message
        referencing and tracking across systems.

    Returns
    -------
    Message
        A new Message instance with the specified content and metadata.

    Message Serialization
    --------------------
    Messages support full serialization to and from dictionary format for storage
    and transmission:

    - [`to_dict()`](`talk_box.Message.to_dict`): Convert to dictionary with ISO timestamp
    - [`from_dict()`](`talk_box.Message.from_dict`): Create from dictionary data

    The serialization format preserves all message information including timestamps
    and metadata, enabling reliable persistence and reconstruction.

    Examples
    --------
    ### Creating basic messages

    Create messages for different conversation participants:

    ```python
    from talk_box import Message
    from datetime import datetime

    # User message - most common type
    user_msg = Message(
        content="What are the key principles of machine learning?",
        role="user"
    )

    # Assistant response
    assistant_msg = Message(
        content="Machine learning has three key principles: representation, evaluation, and optimization...",
        role="assistant"
    )

    # System instruction
    system_msg = Message(
        content="You are a helpful AI assistant specializing in technical topics.",
        role="system"
    )

    print(f"User asked: {user_msg.content}")
    print(f"Message ID: {user_msg.message_id}")
    print(f"Created at: {user_msg.timestamp}")
    ```

    ### Working with metadata

    Use metadata to store additional context and processing information:

    ```python
    # Message with rich metadata
    detailed_msg = Message(
        content="Here's the code implementation you requested...",
        role="assistant",
        metadata={
            "model": "gpt-4-turbo",
            "tokens_used": 245,
            "confidence": 0.92,
            "sources": ["python_docs", "stackoverflow"],
            "code_blocks": 2,
            "execution_time": 1.3
        }
    )

    # Access metadata
    print(f"Model used: {detailed_msg.metadata['model']}")
    print(f"Confidence: {detailed_msg.metadata['confidence']}")
    print(f"Sources: {', '.join(detailed_msg.metadata['sources'])}")
    ```

    ### Message serialization and persistence

    Convert messages to/from dictionaries for storage and transmission:

    ```python
    # Create a message
    original_msg = Message(
        content="Serialize this message for storage",
        role="user",
        metadata={"importance": "high", "category": "technical"}
    )

    # Convert to dictionary (for JSON storage, API calls, etc.)
    msg_dict = original_msg.to_dict()
    print("Serialized message:", msg_dict)

    # Reconstruct from dictionary
    restored_msg = Message.from_dict(msg_dict)
    print(f"Restored content: {restored_msg.content}")
    print(f"Same message ID: {restored_msg.message_id == original_msg.message_id}")
    ```

    ### Function call messages

    Create messages for function calls and tool usage:

    ```python
    # Function call request
    function_call = Message(
        content="calculate_statistics",
        role="function",
        metadata={
            "function_name": "calculate_statistics",
            "arguments": {"data": [1, 2, 3, 4, 5], "method": "mean"},
            "call_id": "func_001"
        }
    )

    # Function result
    function_result = Message(
        content="Statistics calculated: mean=3.0, std=1.58",
        role="function",
        metadata={
            "function_name": "calculate_statistics",
            "result": {"mean": 3.0, "std": 1.58},
            "call_id": "func_001",
            "success": True
        }
    )

    print(f"Function: {function_call.metadata['function_name']}")
    print(f"Result: {function_result.content}")
    ```

    ### Custom roles and specialized workflows

    Use custom roles for specialized conversation types:

    ```python
    # Code review workflow
    code_submission = Message(
        content="def fibonacci(n): return n if n <= 1 else fibonacci(n-1) + fibonacci(n-2)",
        role="developer"
    )

    code_review = Message(
        content="Consider using memoization to improve performance for large n values.",
        role="reviewer",
        metadata={
            "review_type": "performance",
            "severity": "suggestion",
            "line_numbers": [1]
        }
    )

    # Documentation workflow
    doc_request = Message(
        content="Please document the fibonacci function",
        role="doc_manager"
    )

    documentation = Message(
        content="Calculates the nth Fibonacci number using recursive approach...",
        role="technical_writer",
        metadata={
            "doc_type": "function_docstring",
            "coverage": "complete"
        }
    )
    ```

    ### Message validation and filtering

    Messages can be validated and filtered based on their attributes:

    ```python
    messages = [user_msg, assistant_msg, system_msg, detailed_msg]

    # Filter by role
    user_messages = [msg for msg in messages if msg.role == "user"]
    assistant_messages = [msg for msg in messages if msg.role == "assistant"]

    # Filter by metadata presence
    messages_with_metadata = [msg for msg in messages if msg.metadata]

    # Filter by timestamp (messages from last hour)
    from datetime import timedelta
    recent_cutoff = datetime.now() - timedelta(hours=1)
    recent_messages = [msg for msg in messages if msg.timestamp > recent_cutoff]

    print(f"User messages: {len(user_messages)}")
    print(f"Messages with metadata: {len(messages_with_metadata)}")
    print(f"Recent messages: {len(recent_messages)}")
    ```

    ### Integration with conversation systems

    Messages are designed to integrate seamlessly with conversation management:

    ```python
    from talk_box import Conversation

    # Create conversation and add messages
    conversation = Conversation()

    # Add messages directly
    conversation.add_message("Hello, I need help with Python", "user")
    conversation.add_message("I'd be happy to help! What specific topic?", "assistant")

    # Or add pre-created Message objects
    detailed_question = Message(
        content="How do I implement a binary search algorithm?",
        role="user",
        metadata={"topic": "algorithms", "difficulty": "intermediate"}
    )
    conversation.messages.append(detailed_question)

    print(f"Conversation has {len(conversation)} messages")
    ```

    Design Notes
    -----------
    - **Immutability**: Messages are dataclasses and should be treated as immutable after creation
    - **Timestamps**: All timestamps are timezone-naive datetime objects in local time
    - **Unique IDs**: Message IDs are UUID4 strings, globally unique across all systems
    - **Metadata Flexibility**: Metadata dictionaries can contain any JSON-serializable data
    - **Role Standards**: While standard roles are recommended, custom roles are fully supported
    - **Serialization Safety**: All fields serialize safely to JSON via the `to_dict()` method

    The Message class provides the foundation for all conversational AI interactions in Talk Box,
    enabling rich, traceable, and extensible communication between users and AI systems.
    """

    content: str
    role: str
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: dict[str, Any] = field(default_factory=dict)
    message_id: str = field(default_factory=lambda: str(uuid4()))

    def to_dict(self) -> dict[str, Any]:
        """Convert message to dictionary format."""
        return {
            "content": self.content,
            "role": self.role,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata,
            "message_id": self.message_id,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Message":
        """Create a message from dictionary data."""
        timestamp = data.get("timestamp")
        if isinstance(timestamp, str):
            timestamp = datetime.fromisoformat(timestamp)
        elif timestamp is None:
            timestamp = datetime.now()

        return cls(
            content=data["content"],
            role=data["role"],
            timestamp=timestamp,
            metadata=data.get("metadata", {}),
            message_id=data.get("message_id", str(uuid4())),
        )


class Conversation:
    """
    Manages sequences of messages in conversational AI interactions, primarily returned by ChatBot methods.

    The `Conversation` class is automatically created and returned by `ChatBot` chat methods, serving as
    the primary container for managing multi-turn conversations between users and AI assistants. This
    integration with `ChatBot` creates a natural progression where users start with chatbot configuration
    and receive `Conversation` objects that handle message history, context management, and persistence.

    **Integration with ChatBot**:

    - Automatically created by [`ChatBot.chat()`](`talk_box.ChatBot.chat`)
    - Returned by [`ChatBot.start_conversation()`](`talk_box.ChatBot.start_conversation`)
    - Updated by [`ChatBot.continue_conversation()`](`talk_box.ChatBot.continue_conversation`)

    This design ensures that users naturally discover conversation management capabilities through
    normal ChatBot usage, while providing advanced users direct access to conversation-level
    operations for specialized workflows.

    The class provides comprehensive functionality for message management, context window control,
    conversation persistence, and history analysis. It automatically handles chronological ordering,
    role-based filtering, and efficient context management for optimal AI performance.

    Conversations serve as containers for sequences of `Message` objects, providing both
    high-level convenience methods and fine-grained control over message operations.
    The class supports automatic context window management to stay within model limitations,
    conversation serialization for persistence, and extensive querying capabilities for
    conversation analysis and processing.

    Parameters
    ----------
    conversation_id : str, optional
        Unique identifier for the conversation. If not provided, automatically generated
        using `uuid4()` to ensure global uniqueness. This enables reliable conversation
        tracking and referencing across systems and sessions.

    Returns
    -------
    Conversation
        A new Conversation instance with empty message history and default settings.

    Core Message Operations
    ----------------------
    The Conversation class provides multiple ways to add and manage messages:

    - [`add_message()`](`talk_box.Conversation.add_message`): Add a message with specified role
    - [`add_user_message()`](`talk_box.Conversation.add_user_message`): Add user messages
    - [`add_assistant_message()`](`talk_box.Conversation.add_assistant_message`): Add AI responses
    - [`add_system_message()`](`talk_box.Conversation.add_system_message`): Add system instructions
    - [`clear_messages()`](`talk_box.Conversation.clear_messages`): Remove all messages

    Message Retrieval and Filtering
    ------------------------------
    Query and filter messages using flexible retrieval methods:

    - [`get_messages()`](`talk_box.Conversation.get_messages`): Get all or role-filtered messages
    - [`get_last_message()`](`talk_box.Conversation.get_last_message`): Get most recent message
    - [`get_message_count()`](`talk_box.Conversation.get_message_count`): Count total messages
    - [`get_context_messages()`](`talk_box.Conversation.get_context_messages`): Get messages within context window

    Context Window Management
    ------------------------
    Control conversation length and model context efficiently:

    - [`set_context_window()`](`talk_box.Conversation.set_context_window`): Set maximum message count
    - [`get_context_messages()`](`talk_box.Conversation.get_context_messages`): Get messages within window

    Context windows automatically manage conversation length by keeping only the most recent
    messages when conversations exceed the specified limit. This ensures optimal performance
    with language models while preserving conversation continuity.

    Serialization and Persistence
    ----------------------------
    Save and restore conversations using built-in serialization:

    - [`to_dict()`](`talk_box.Conversation.to_dict`): Convert to dictionary format
    - [`from_dict()`](`talk_box.Conversation.from_dict`): Create from dictionary data

    The serialization format preserves all conversation metadata, timestamps, and message
    history, enabling reliable persistence across sessions and systems.

    Examples
    --------
    ### Creating and managing basic conversations

    Create a conversation and add messages using convenience methods:

    ```python
    from talk_box import Conversation

    # Create a new conversation
    conversation = Conversation()

    # Add messages using role-specific methods
    conversation.add_user_message("Hello! I need help with Python programming.")
    conversation.add_assistant_message("I'd be happy to help! What specific Python topic would you like to explore?")
    conversation.add_user_message("How do I create and use classes in Python?")

    print(f"Conversation has {len(conversation)} messages")
    print(f"Last message: {conversation.get_last_message().content}")
    ```

    ### Working with system messages and context

    Use system messages to provide context and instructions:

    ```python
    # Create conversation with initial system context
    tech_conversation = Conversation()

    # Set up the assistant's behavior
    tech_conversation.add_system_message(
        "You are a senior Python developer. Provide detailed, practical answers with code examples."
    )

    # Add conversation context
    tech_conversation.add_user_message(
        "I'm building a web API and need help with error handling.",
        metadata={"project": "web_api", "experience_level": "intermediate"}
    )

    tech_conversation.add_assistant_message(
        "For robust API error handling, I recommend using structured exception handling...",
        metadata={"code_examples": True, "topics": ["exceptions", "api_design"]}
    )

    # Get all assistant messages
    assistant_responses = tech_conversation.get_messages(role="assistant")
    print(f"Assistant provided {len(assistant_responses)} responses")
    ```

    ### Context window management for long conversations

    Control conversation length to work within model limitations:

    ```python
    # Create conversation with context window
    long_conversation = Conversation()
    long_conversation.set_context_window(10)  # Keep only last 10 messages

    # Add many messages (simulating a long conversation)
    for i in range(15):
        long_conversation.add_user_message(f"User message {i+1}")
        long_conversation.add_assistant_message(f"Assistant response {i+1}")

    # Check total vs context messages
    all_messages = long_conversation.get_messages()
    context_messages = long_conversation.get_context_messages()

    print(f"Total messages: {len(all_messages)}")
    print(f"Context messages: {len(context_messages)}")
    print(f"First context message: {context_messages[0].content}")
    ```

    ### Conversation analysis and filtering

    Analyze conversation patterns and extract insights:

    ```python
    from datetime import datetime, timedelta

    # Create a conversation with varied message types
    analysis_conversation = Conversation()

    # Add messages with rich metadata
    analysis_conversation.add_user_message(
        "What's the weather like?",
        metadata={"intent": "weather_query", "urgency": "low"}
    )

    analysis_conversation.add_assistant_message(
        "I don't have access to real-time weather data...",
        metadata={"capability": "limitation", "suggestion": "weather_api"}
    )

    analysis_conversation.add_user_message(
        "Can you help me debug this code?",
        metadata={"intent": "code_help", "urgency": "high", "topic": "debugging"}
    )

    # Analyze conversation patterns
    user_messages = analysis_conversation.get_messages(role="user")
    urgent_messages = [
        msg for msg in user_messages
        if msg.metadata.get("urgency") == "high"
    ]

    code_related = [
        msg for msg in analysis_conversation.get_messages()
        if "code" in msg.content.lower() or msg.metadata.get("topic") == "debugging"
    ]

    print(f"Urgent user requests: {len(urgent_messages)}")
    print(f"Code-related messages: {len(code_related)}")
    ```

    ### Conversation persistence and restoration

    Save and restore conversations for session management:

    ```python
    import json
    from pathlib import Path

    # Create and populate a conversation
    original_conversation = Conversation()
    original_conversation.add_user_message("Save this conversation")
    original_conversation.add_assistant_message("I'll help you save this conversation data")

    # Serialize to dictionary
    conversation_data = original_conversation.to_dict()

    # Save to file (example - you might use databases, APIs, etc.)
    save_path = Path("conversation_backup.json")
    with save_path.open("w") as f:
        json.dump(conversation_data, f, indent=2)

    # Later: restore from file
    with save_path.open("r") as f:
        loaded_data = json.load(f)

    # Reconstruct conversation
    restored_conversation = Conversation.from_dict(loaded_data)

    print(f"Original ID: {original_conversation.conversation_id}")
    print(f"Restored ID: {restored_conversation.conversation_id}")
    print(f"Messages match: {len(original_conversation) == len(restored_conversation)}")

    # Clean up
    save_path.unlink()
    ```

    ### Multi-turn conversation workflows

    Build complex conversation workflows with branching logic:

    ```python
    # Customer support conversation workflow
    support_conversation = Conversation()

    # Initial system setup
    support_conversation.add_system_message(
        "You are a helpful customer support agent. Gather information before providing solutions."
    )

    # Customer inquiry
    support_conversation.add_user_message(
        "My order hasn't arrived yet",
        metadata={"category": "shipping", "sentiment": "concerned"}
    )

    # Support response with information gathering
    support_conversation.add_assistant_message(
        "I understand your concern. Can you provide your order number so I can check the status?",
        metadata={"action": "information_gathering", "next_step": "order_lookup"}
    )

    # Customer provides information
    support_conversation.add_user_message(
        "Order #12345",
        metadata={"order_id": "12345", "info_provided": True}
    )

    # Check conversation flow
    last_assistant_msg = support_conversation.get_last_message(role="assistant")
    next_step = last_assistant_msg.metadata.get("next_step")
    print(f"Next action needed: {next_step}")

    # Determine conversation flow based on metadata
    user_messages = support_conversation.get_messages(role="user")
    has_order_info = any(msg.metadata.get("info_provided") for msg in user_messages)

    if has_order_info:
        support_conversation.add_assistant_message(
            "Thank you! I'm looking up order #12345 now...",
            metadata={"action": "order_lookup", "order_id": "12345"}
        )
    ```

    ### Integration with chatbot systems

    Use conversations as the foundation for chatbot interactions:

    ```python
    from talk_box import ChatBot

    # Create chatbot and conversation
    bot = ChatBot().preset("technical_advisor")
    bot_conversation = Conversation()

    # Set up conversation context
    bot_conversation.add_system_message(
        "You are a technical advisor specializing in Python and machine learning."
    )

    # Simulate chat interaction
    def chat_with_bot(user_input: str) -> str:
        # Add user message
        bot_conversation.add_user_message(user_input)

        # Get bot response (simplified - real implementation would use bot.chat())
        response = f"Technical response to: {user_input}"
        bot_conversation.add_assistant_message(response)

        return response

    # Use the chat function
    response1 = chat_with_bot("What is machine learning?")
    response2 = chat_with_bot("How do I start with scikit-learn?")

    print(f"Conversation now has {len(bot_conversation)} messages")
    print(f"Latest response: {bot_conversation.get_last_message().content}")
    ```

    Advanced Features
    ----------------
    **Message Metadata**: Each message can carry rich metadata for application-specific data,
    conversation analysis, and workflow management.

    **Automatic Timestamps**: All messages include creation timestamps for chronological
    analysis and conversation timeline reconstruction.

    **Context Management**: Intelligent context window handling ensures conversations stay
    within model limits while preserving important context.

    **Role-based Organization**: Standard roles (user, assistant, system, function) provide
    clear conversation structure for AI processing.

    **Extensible Design**: The conversation format supports custom roles, metadata schemas,
    and integration patterns for specialized use cases.

    **Memory Efficiency**: Context windows and message filtering enable efficient handling
    of very long conversations without memory issues.

    Integration Notes
    ----------------
    - **Thread Safety**: Conversations are not thread-safe; use external synchronization for concurrent access
    - **Memory Usage**: Long conversations should use context windows to manage memory efficiently
    - **Serialization**: All message content and metadata must be JSON-serializable for persistence
    - **Timestamps**: Uses local time zone-naive datetime objects; consider UTC for distributed systems
    - **ID Uniqueness**: Conversation and message IDs are globally unique UUID4 strings

    The Conversation class provides the foundation for sophisticated conversational AI applications,
    enabling everything from simple chat interfaces to complex multi-turn workflows with rich
    context management and analysis capabilities.
    """

    def __init__(self, conversation_id: Optional[str] = None) -> None:
        """Initialize a new conversation."""
        self.conversation_id = conversation_id or str(uuid4())
        self.messages: list[Message] = []
        self.created_at = datetime.now()
        self.metadata: dict[str, Any] = {}
        self.max_context_length: Optional[int] = None

    def add_message(
        self, content: str, role: str, metadata: Optional[dict[str, Any]] = None
    ) -> Message:
        """Add a new message to the conversation."""
        message = Message(content=content, role=role, metadata=metadata or {})
        self.messages.append(message)
        return message

    def add_user_message(self, content: str, metadata: Optional[dict[str, Any]] = None) -> Message:
        """Add a user message to the conversation."""
        return self.add_message(content, "user", metadata)

    def add_assistant_message(
        self, content: str, metadata: Optional[dict[str, Any]] = None
    ) -> Message:
        """Add an assistant message to the conversation."""
        return self.add_message(content, "assistant", metadata)

    def add_system_message(
        self, content: str, metadata: Optional[dict[str, Any]] = None
    ) -> Message:
        """Add a system message to the conversation."""
        return self.add_message(content, "system", metadata)

    def get_messages(self, role: Optional[str] = None) -> list[Message]:
        """Get all messages, optionally filtered by role."""
        if role is None:
            return self.messages.copy()
        return [msg for msg in self.messages if msg.role == role]

    def get_last_message(self, role: Optional[str] = None) -> Optional[Message]:
        """Get the last message, optionally filtered by role."""
        messages = self.get_messages(role)
        return messages[-1] if messages else None

    def clear_messages(self) -> None:
        """Clear all messages from the conversation."""
        self.messages.clear()

    def set_context_window(self, max_length: int) -> None:
        """Set the maximum context window length."""
        self.max_context_length = max_length

    def get_context_messages(self) -> list[Message]:
        """Get messages within the context window."""
        if self.max_context_length is None:
            return self.messages.copy()

        # Simple implementation: take the last N messages
        return self.messages[-self.max_context_length :]

    def get_message_count(self) -> int:
        """Get the total number of messages."""
        return len(self.messages)

    def to_dict(self) -> dict[str, Any]:
        """Convert conversation to dictionary format."""
        return {
            "conversation_id": self.conversation_id,
            "created_at": self.created_at.isoformat(),
            "metadata": self.metadata,
            "max_context_length": self.max_context_length,
            "messages": [msg.to_dict() for msg in self.messages],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Conversation":
        """Create a conversation from dictionary data."""
        conversation = cls(conversation_id=data["conversation_id"])

        # Set conversation properties
        created_at = data.get("created_at")
        if isinstance(created_at, str):
            conversation.created_at = datetime.fromisoformat(created_at)

        conversation.metadata = data.get("metadata", {})
        conversation.max_context_length = data.get("max_context_length")

        # Load messages
        for msg_data in data.get("messages", []):
            message = Message.from_dict(msg_data)
            conversation.messages.append(message)

        return conversation

    def __len__(self) -> int:
        """Return the number of messages in the conversation."""
        return len(self.messages)

    def __str__(self) -> str:
        """String representation of the conversation."""
        return f"Conversation({self.conversation_id}, {len(self.messages)} messages)"

    def _repr_html_(self) -> str:
        """HTML representation for Jupyter notebook display."""
        if not self.messages:
            return """
            <div style="border: 1px solid #ddd; border-radius: 8px; padding: 15px; background-color: #f9f9f9; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;">
                <h3 style="margin: 0 0 10px 0; color: #666;">💬 Empty Conversation</h3>
                <p style="margin: 0; color: #888;">No messages yet</p>
            </div>
            """

        html_parts = [
            """
            <div style="border: 1px solid #ddd; border-radius: 8px; padding: 15px; background-color: #f9f9f9; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;">
                <h3 style="margin: 0 0 15px 0; color: #333;">💬 Conversation</h3>
            """
        ]

        # Add conversation metadata
        html_parts.append(f"""
            <div style="margin-bottom: 15px; padding: 8px; background-color: #f0f0f0; border-radius: 4px; font-size: 12px; color: #666;">
                <strong>ID:</strong> {self.conversation_id[:8]}... | <strong>Messages:</strong> {len(self.messages)}
            </div>
        """)

        # Add each message with role-based styling
        for i, message in enumerate(self.messages):
            # Role-based styling
            if message.role == "user":
                role_color = "#0066cc"
                role_icon = "👤"
                bg_color = "#e6f3ff"
                border_color = "#0066cc"
            elif message.role == "assistant":
                role_color = "#009900"
                role_icon = "🤖"
                bg_color = "#e6ffe6"
                border_color = "#009900"
            elif message.role == "system":
                role_color = "#cc6600"
                role_icon = "⚙️"
                bg_color = "#fff3e6"
                border_color = "#cc6600"
            else:
                role_color = "#666666"
                role_icon = "📝"
                bg_color = "#f5f5f5"
                border_color = "#999999"

            # Format content with basic HTML escaping and line breaks
            content = (
                message.content.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
            )
            content = content.replace("\n", "<br>")

            # Create scrollable container for long messages
            content_style = "color: #333; line-height: 1.4;"
            if len(content) > 1000:  # Only add scrolling for very long content
                content_style += """
                    max-height: 300px;
                    overflow-y: auto;
                    padding: 12px;
                    border: 1px solid #e0e0e0;
                    border-radius: 6px;
                    background-color: #fafafa;
                    box-shadow: inset 0 1px 3px rgba(0,0,0,0.1);
                    scrollbar-width: thin;
                    scrollbar-color: #ccc #f0f0f0;
                """.replace("\n", " ").strip()

            html_parts.append(f"""
                <div style="margin-bottom: 12px; padding: 12px; background-color: {bg_color}; border-left: 4px solid {border_color}; border-radius: 0 4px 4px 0;">
                    <div style="margin-bottom: 8px;">
                        <strong style="color: {role_color};">{role_icon} {message.role.title()}</strong>
                        <span style="float: right; font-size: 11px; color: #888;">{message.timestamp.strftime("%H:%M:%S")}</span>
                    </div>
                    <div style="{content_style}">{content}</div>
                </div>
            """)

        html_parts.append("</div>")
        return "".join(html_parts)
