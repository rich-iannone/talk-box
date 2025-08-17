# Talk Box

**The best way to generate, test, and deploy LLM chatbots**

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Tests](https://img.shields.io/badge/tests-98%25%20coverage-green.svg)](./tests/)
[![Code Quality](https://img.shields.io/badge/code%20quality-ruff-blue.svg)](https://github.com/astral-sh/ruff)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Talk Box is a Python framework that makes creating conversational AI as intuitive as having a conversation. Start with simple chatbot creation, and naturally discover advanced features like conversation management and message handling as you need them.

## Why Talk Box?

Talk Box uses a **layered API design** that grows with your understanding. You start with `ChatBot` (easy!), naturally discover `Conversation` objects from return values, and find `Message` details within conversations. No overwhelming documentation, just progressive discovery.

```python
from talk_box import ChatBot

# Create a chatbot and start chatting immediately
bot = ChatBot().model("gpt-4").preset("technical_advisor")
conversation = bot.chat("Hello!")
print(conversation.get_last_message().content)

# Conversation automatically manages history
response = bot.chat("Tell me more", conversation)
print(response.get_last_message().content)
```

## Quick Start

### Installation

```bash
pip install talk-box
```

### 30-Second Demo

```python
from talk_box import ChatBot

# Start simple - just create and chat
bot = ChatBot().model("gpt-4").preset("technical_advisor")
conversation = bot.chat("What's machine learning?")
print(conversation.get_last_message().content)

# Naturally discover more features as you need them
conversation = bot.chat("Tell me about neural networks", conversation)
messages = conversation.get_messages()  # Explore message history
latest = conversation.get_last_message()  # Get the latest response

print(f"Conversation has {conversation.get_message_count()} messages")
print(f"Latest message: {latest.content[:100]}...")
```

### Discover the API Layer by Layer

Talk Box is designed for **progressive discovery**: start simple and naturally find more features:

```python
# Layer 1: Basic ChatBot (everyone starts here)
bot = ChatBot().model("gpt-4")
conversation = bot.chat("Hello!")

# Layer 2: Conversation management (discovered from return values)
conversation.add_user_message("Follow-up question")
messages = conversation.get_messages()

# Layer 3: Message details (discovered within conversations)
latest = conversation.get_last_message()
print(f"Role: {latest.role}, Content: {latest.content}")
```

### Test Without Setup

```python
# Works immediately - no API keys needed for testing
bot = ChatBot().preset("creative_writer").temperature(0.8)
conversation = bot.chat("Write a short story about a robot")

# Explore the conversation object
print(f"Messages: {conversation.get_message_count()}")
print(f"Latest: {conversation.get_last_message().content}")

# Chain more interactions
follow_up = bot.chat("Make it funnier", conversation)
```

## Key Features

### **Integrated Conversation Management**

All chat interactions automatically return conversation objects for seamless multi-turn conversations:

```python
bot = ChatBot().model("gpt-4").preset("technical_advisor")

# Every chat returns a conversation object
conversation = bot.chat("What's Python?")
conversation = bot.chat("What about its applications?", conversation)
conversation = bot.chat("Show me code examples", conversation)

# Access the full conversation history
messages = conversation.get_messages()
latest = conversation.get_last_message()
```

### **Built-in Behavior Presets**

Choose from professionally crafted personalities:

- **`technical_advisor`**: Authoritative, detailed, evidence-based
- **`customer_support`**: Polite, helpful, solution-focused  
- **`creative_writer`**: Imaginative, expressive, storytelling
- **`data_analyst`**: Analytical, precise, metrics-driven
- **`legal_advisor`**: Professional, thorough, disclaimer-aware

```python
# Each preset includes tone, expertise, constraints, and system prompts
support_bot = ChatBot().preset("customer_support")
creative_bot = ChatBot().preset("creative_writer") 
analyst_bot = ChatBot().preset("data_analyst")
```

### **Chainable Configuration**

Build exactly the chatbot you need with method chaining:

```python
specialized_bot = (
    ChatBot()
    .model("gpt-4")
    .preset("technical_advisor") 
    .temperature(0.3)
    .persona("Senior Software Engineer")
)

conversation = specialized_bot.chat("Explain microservices")
```

### **Test-First Design**

Start testing immediately, add real LLM integration when ready:

```python
# Works without any API keys - perfect for development
bot = ChatBot().preset("data_analyst").temperature(0.2)
conversation = bot.chat("Analyze this dataset")

# Test your conversation logic
assert conversation.get_message_count() == 2
assert "data" in conversation.get_last_message().content.lower()

# Add real LLM when ready for production
# bot.enable_llm_mode()  # Connects to actual models
```

## Complete Example: Customer Support Bot

```python
from talk_box import ChatBot

# Create a customer support chatbot
support_bot = (
    ChatBot()
    .model("gpt-4")
    .preset("customer_support")
    .temperature(0.3)
    .persona("Helpful support specialist")
)

# Start a conversation
conversation = support_bot.chat("I'm having trouble with my account")
print(f"Bot: {conversation.get_last_message().content}")

# Continue the conversation - history is automatically managed
conversation = support_bot.chat("Can you help me reset it?", conversation)
print(f"Bot: {conversation.get_last_message().content}")

# Explore the conversation
print(f"\nConversation summary:")
print(f"- Total messages: {conversation.get_message_count()}")
print(f"- User messages: {len(conversation.get_messages(role='user'))}")
print(f"- Assistant messages: {len(conversation.get_messages(role='assistant'))}")

# Access individual messages
for i, message in enumerate(conversation.get_messages()):
    print(f"{i+1}. {message.role}: {message.content[:50]}...")
```

## License

MIT License - see [LICENSE](./LICENSE) for details.
