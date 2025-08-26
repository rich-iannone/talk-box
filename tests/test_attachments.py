from talk_box import ChatBot
from talk_box.attachments import Attachments, AttachmentMetadata


def test_attachments_creation():
    """Test basic attachment creation."""
    attachments = Attachments("README.md")
    assert len(attachments) == 1
    assert len(attachments.files) == 1
    assert attachments.files[0].name == "README.md"


def test_attachments_with_prompt():
    """Test attachment with prompt."""
    attachments = Attachments("README.md").with_prompt("Analyze this file")
    assert attachments.prompt == "Analyze this file"


def test_attachments_multiple_files():
    """Test multiple file attachments."""
    attachments = Attachments("README.md", "pyproject.toml")
    assert len(attachments) == 2
    assert len(attachments.files) == 2


def test_attachments_bool():
    """Test boolean conversion."""
    empty_attachments = Attachments()
    with_files = Attachments("README.md")

    assert not empty_attachments
    assert with_files


def test_attachments_repr():
    """Test string representation."""
    attachments = Attachments("README.md")
    repr_str = repr(attachments)
    assert "README.md" in repr_str

    with_prompt = attachments.with_prompt("Test prompt")
    repr_with_prompt = repr(with_prompt)
    assert "README.md" in repr_with_prompt
    assert "Test prompt" in repr_with_prompt


def test_attachments_html_repr():
    """Test HTML representation for Jupyter notebooks."""
    # Test basic HTML representation
    attachments = Attachments("README.md").with_prompt("Test prompt")
    html = attachments._repr_html_()

    # Check essential HTML elements
    assert "Attachments" in html
    assert "README.md" in html
    assert "Test prompt" in html
    assert "<div" in html
    assert "Files:" in html

    # Test multiple files
    multi_files = Attachments("README.md", "pyproject.toml")
    multi_html = multi_files._repr_html_()
    assert "README.md" in multi_html
    assert "pyproject.toml" in multi_html

    # Test without prompt
    no_prompt = Attachments("README.md")
    no_prompt_html = no_prompt._repr_html_()
    assert "README.md" in no_prompt_html
    # Should not have prompt section when no prompt
    assert no_prompt_html.count("Prompt:") == 0


# ChatBot integration tests
def test_chat_with_attachments():
    """Test chatbot can handle attachments."""
    bot = ChatBot().model("gpt-4")
    attachments = Attachments("README.md").with_prompt("What is this?")

    # This should work in echo mode without LLM
    conversation = bot.chat(attachments)

    # Check that conversation was created
    assert len(conversation.get_messages()) == 2

    # Check user message contains the prompt
    user_msg = conversation.get_messages()[0]
    assert user_msg.content == "What is this?"

    # Check assistant response mentions the file
    assistant_msg = conversation.get_messages()[1]
    assert "README.md" in assistant_msg.content


def test_chat_backward_compatibility():
    """Test that regular text messages still work."""
    bot = ChatBot().model("gpt-4")
    conversation = bot.chat("Hello world")

    assert len(conversation.get_messages()) == 2
    assert conversation.get_messages()[0].content == "Hello world"


def test_chat_conversation_continuation_with_attachments():
    """Test continuing conversation with attachments."""
    bot = ChatBot().model("gpt-4")

    # Start with text
    conv1 = bot.chat("Hello")

    # Continue with attachments
    attachments = Attachments("README.md").with_prompt("What's this file?")
    conv2 = bot.chat(attachments, conversation=conv1)

    # Should have 4 messages total
    assert len(conv2.get_messages()) == 4

    # Check the attachment message
    assert conv2.get_messages()[2].content == "What's this file?"


def test_metadata_creation():
    """Test metadata creation."""
    metadata = AttachmentMetadata(
        filename="test.txt",
        file_type="txt",
        size_bytes=100,
        content_type="text",
        processing_time_ms=0.5,
    )

    assert metadata.filename == "test.txt"
    assert metadata.file_type == "txt"
    assert metadata.size_bytes == 100
    assert metadata.content_type == "text"
    assert metadata.processing_time_ms == 0.5
