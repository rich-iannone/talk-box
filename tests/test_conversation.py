from datetime import datetime
from uuid import UUID

import pytest

from talk_box.conversation import Conversation, Message


class TestMessage:
    """Test cases for the Message class."""

    def test_message_creation(self):
        """Test basic Message creation."""
        content = "Hello, world!"
        role = "user"

        message = Message(content=content, role=role)

        assert message.content == content
        assert message.role == role
        assert isinstance(message.timestamp, datetime)
        assert isinstance(message.metadata, dict)
        assert len(message.metadata) == 0
        # Check that message_id is a valid UUID string
        UUID(message.message_id)

    def test_message_with_metadata(self):
        """Test Message creation with metadata."""
        metadata = {"source": "test", "priority": "high"}
        message = Message(content="Test", role="user", metadata=metadata)

        assert message.metadata == metadata

    def test_message_to_dict(self):
        """Test converting message to dictionary."""
        message = Message(content="Test", role="assistant")
        message_dict = message.to_dict()

        expected_keys = {"content", "role", "timestamp", "metadata", "message_id"}
        assert set(message_dict.keys()) == expected_keys
        assert message_dict["content"] == "Test"
        assert message_dict["role"] == "assistant"

    def test_message_from_dict(self):
        """Test creating message from dictionary."""
        data = {
            "content": "Test message",
            "role": "user",
            "timestamp": "2025-08-16T10:30:00",
            "metadata": {"test": "value"},
            "message_id": "test-id",
        }

        message = Message.from_dict(data)

        assert message.content == "Test message"
        assert message.role == "user"
        assert message.metadata == {"test": "value"}
        assert message.message_id == "test-id"

    def test_message_from_dict_minimal(self):
        """Test creating message from minimal dictionary."""
        data = {"content": "Test", "role": "user"}
        message = Message.from_dict(data)

        assert message.content == "Test"
        assert message.role == "user"
        assert isinstance(message.timestamp, datetime)
        assert isinstance(message.metadata, dict)


class TestConversation:
    """Test cases for the Conversation class."""

    def test_conversation_creation(self):
        """Test basic Conversation creation."""
        conversation = Conversation()

        assert conversation.conversation_id is not None
        assert isinstance(conversation.messages, list)
        assert len(conversation.messages) == 0
        assert isinstance(conversation.created_at, datetime)
        assert isinstance(conversation.metadata, dict)

    def test_conversation_with_id(self):
        """Test Conversation creation with specific ID."""
        conv_id = "test-conversation-123"
        conversation = Conversation(conversation_id=conv_id)

        assert conversation.conversation_id == conv_id

    def test_add_message(self):
        """Test adding messages to conversation."""
        conversation = Conversation()

        message = conversation.add_message("Hello", "user")

        assert len(conversation.messages) == 1
        assert conversation.messages[0] == message
        assert message.content == "Hello"
        assert message.role == "user"

    def test_add_user_message(self):
        """Test adding user message."""
        conversation = Conversation()

        message = conversation.add_user_message("User message")

        assert message.role == "user"
        assert message.content == "User message"

    def test_add_assistant_message(self):
        """Test adding assistant message."""
        conversation = Conversation()

        message = conversation.add_assistant_message("Assistant response")

        assert message.role == "assistant"
        assert message.content == "Assistant response"

    def test_add_system_message(self):
        """Test adding system message."""
        conversation = Conversation()

        message = conversation.add_system_message("System prompt")

        assert message.role == "system"
        assert message.content == "System prompt"

    def test_get_messages(self):
        """Test getting all messages."""
        conversation = Conversation()
        conversation.add_user_message("User 1")
        conversation.add_assistant_message("Assistant 1")
        conversation.add_user_message("User 2")

        all_messages = conversation.get_messages()
        assert len(all_messages) == 3

        user_messages = conversation.get_messages(role="user")
        assert len(user_messages) == 2
        assert all(msg.role == "user" for msg in user_messages)

    def test_get_last_message(self):
        """Test getting the last message."""
        conversation = Conversation()

        # No messages yet
        assert conversation.get_last_message() is None

        conversation.add_user_message("First")
        conversation.add_assistant_message("Second")

        last_message = conversation.get_last_message()
        assert last_message.content == "Second"
        assert last_message.role == "assistant"

        last_user = conversation.get_last_message(role="user")
        assert last_user.content == "First"

    def test_clear_messages(self):
        """Test clearing all messages."""
        conversation = Conversation()
        conversation.add_user_message("Test")
        conversation.add_assistant_message("Response")

        assert len(conversation.messages) == 2

        conversation.clear_messages()
        assert len(conversation.messages) == 0

    def test_context_window(self):
        """Test context window functionality."""
        conversation = Conversation()

        # Add multiple messages
        for i in range(10):
            conversation.add_user_message(f"Message {i}")

        # Set context window
        conversation.set_context_window(5)

        context_messages = conversation.get_context_messages()
        assert len(context_messages) == 5
        # Should be the last 5 messages
        assert context_messages[0].content == "Message 5"
        assert context_messages[-1].content == "Message 9"

    def test_context_window_unlimited(self):
        """Test unlimited context window."""
        conversation = Conversation()

        for i in range(5):
            conversation.add_user_message(f"Message {i}")

        # No context window set
        context_messages = conversation.get_context_messages()
        assert len(context_messages) == 5

    def test_get_message_count(self):
        """Test getting message count."""
        conversation = Conversation()
        assert conversation.get_message_count() == 0

        conversation.add_user_message("Test")
        assert conversation.get_message_count() == 1

    def test_len(self):
        """Test __len__ method."""
        conversation = Conversation()
        assert len(conversation) == 0

        conversation.add_user_message("Test")
        assert len(conversation) == 1

    def test_str(self):
        """Test __str__ method."""
        conversation = Conversation()
        conversation.add_user_message("Test")

        str_repr = str(conversation)
        assert conversation.conversation_id in str_repr
        assert "1 messages" in str_repr

    def test_to_dict(self):
        """Test converting conversation to dictionary."""
        conversation = Conversation()
        conversation.add_user_message("Test message")
        conversation.metadata = {"test": "value"}

        conv_dict = conversation.to_dict()

        expected_keys = {
            "conversation_id",
            "created_at",
            "metadata",
            "max_context_length",
            "messages",
        }
        assert set(conv_dict.keys()) == expected_keys
        assert len(conv_dict["messages"]) == 1

    def test_from_dict(self):
        """Test creating conversation from dictionary."""
        data = {
            "conversation_id": "test-conv",
            "created_at": "2025-08-16T10:30:00",
            "metadata": {"test": "value"},
            "max_context_length": 10,
            "messages": [
                {"content": "Hello", "role": "user"},
                {"content": "Hi there", "role": "assistant"},
            ],
        }

        conversation = Conversation.from_dict(data)

        assert conversation.conversation_id == "test-conv"
        assert conversation.metadata == {"test": "value"}
        assert conversation.max_context_length == 10
        assert len(conversation.messages) == 2
