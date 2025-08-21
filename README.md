# Talk Box

**The best way to generate, test, and deploy LLM chatbots with attention-optimized prompt engineering**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Talk Box is a Python framework that transforms prompt engineering from an art into a systematic
engineering discipline. Create effective, maintainable prompts using research-backed attention
mechanisms, then deploy them with powerful conversation management.

## 🎯 Why Talk Box?

- **🧠 Attention-Based Prompt Engineering**: build prompts that leverage how LLMs actually process information
- **📚 Layered API Design**: start simple, discover advanced features as you need them
- **⚡ Multiple Usage Patterns**: from quick structured prompts to complex modular components
- **🔄 Integrated Conversation Management**: seamless multi-turn conversations with full history
- **🛠️ Built-in Behavior Presets**: professional templates for common engineering tasks
- **🧪 Test-First Design**: develop and test without API keys, deploy when ready

## Quick Start

### Installation

```bash
pip install talk-box
```

### 30-Second Demo: Attention-Optimized Prompts

```python
import talk_box as tb

# Create a security-focused chatbot with structured prompts
bot = (
    tb.ChatBot()
    .model("gpt-4.1-mini")
    .system_prompt(
        tb.PromptBuilder()
        .persona("senior security engineer", "web application security")
        .critical_constraint("Focus only on CVSS 7.0+ vulnerabilities")
        .core_analysis([
            "SQL injection risks",
            "Authentication bypass possibilities",
            "Data validation gaps"
        ])
        .output_format([
            "CRITICAL: Immediate security risks",
            "Include specific line numbers and fixes"
        ])
        .final_emphasis("Prioritize vulnerabilities leading to data breaches")
        .build()
    )
)

# Use the structured prompt
bot.show("prompt")
```

<details>
<summary>🔍 View Generated Prompt</summary>

```
📝 System Prompt Analysis
==================================================
Total Length: 402 characters
Custom Prompt: Yes
Preset: None
Persona: senior security engineer with expertise in web application security.

Final System Prompt:
------------------------------
You are a senior security engineer with expertise in web application security.

CRITICAL REQUIREMENTS:
- Focus only on CVSS 7.0+ vulnerabilities

CORE ANALYSIS (Required):
- SQL injection risks
- Authentication bypass possibilities
- Data validation gaps

OUTPUT FORMAT:
- CRITICAL: Immediate security risks
- Include specific line numbers and fixes

Prioritize vulnerabilities leading to data breaches
------------------------------
```

</details>

### Even Simpler: Quick Structured Prompts

```python
import talk_box as tb

# Configure a bot with structured prompts in one go
security_bot = (tb.ChatBot()
    .model("gpt-4.1-mini")
    .structured_prompt(
        persona="cybersecurity expert",
        task="Audit web application security",
        constraints=["Focus on OWASP Top 10", "Include CVSS scores"],
        format=["Critical issues", "Important fixes"],
        focus="preventing data breaches"
    ))

# Now just chat directly - the structure is built in
response = security_bot.chat("Here's my authentication code...")
```

## 🚀 Core Features

### **🧠 Attention-Based Prompt Engineering**

Build prompts that leverage how transformer attention mechanisms actually work:

```python
import talk_box as tb

# Traditional approach (attention-diffused)
old_prompt = "Analyze this code for issues and provide recommendations."

# Attention-optimized approach
bot = tb.ChatBot().model("gpt-4.1-mini")
structured_prompt = (bot.prompt_builder()
    .persona("senior software architect", "security and performance")
    .critical_constraint("Focus on critical security vulnerabilities")
    .core_analysis([
        "SQL injection vulnerabilities",
        "Performance bottlenecks",
        "Code maintainability issues"
    ])
    .output_format([
        "CRITICAL: Security issues requiring immediate fix",
        "IMPORTANT: Performance improvements",
        "Include specific line numbers and remediation code"
    ])
    .final_emphasis("Prioritize by business impact and fix complexity")
    .build())

response = bot.chat(structured_prompt)
```

<details>
<summary>🔍 View Generated Prompt</summary>

```
You are a senior software architect with expertise in security and performance.

CRITICAL REQUIREMENTS:
- Focus on critical security vulnerabilities

CORE ANALYSIS (Required):
- SQL injection vulnerabilities
- Performance bottlenecks
- Code maintainability issues

OUTPUT FORMAT:
- CRITICAL: Security issues requiring immediate fix
- IMPORTANT: Performance improvements
- Include specific line numbers and remediation code

Prioritize by business impact and fix complexity
```

</details>

**Key principles implemented:**

- 🎯 **Front-load critical information** (primacy bias)
- 📋 **Structure creates focus** (attention clustering)
- 🎭 **Personas for behavioral anchoring**
- ⚡ **Specific constraints prevent attention drift**
- 🔚 **Final emphasis leverages recency bias**

### **⚙️ Multiple Usage Patterns**

Choose the approach that fits your workflow:

```python
import talk_box as tb

# Create a base bot first
bot = tb.ChatBot().model("gpt-4.1-mini")

# Pattern 1: Full builder control
prompt = (bot.prompt_builder()
    .persona("security expert")
    .critical_constraint("Focus on critical issues")
    .build())

# Pattern 2: Pre-configured builders for common tasks
arch_prompt = (bot.prompt_builder("architectural")
    .focus_on("microservices migration readiness")
    .build())
```

<details>
<summary>🔍 View Generated Prompt</summary>

```
You are a senior software architect with expertise in comprehensive codebase analysis.

CRITICAL REQUIREMENTS:
- Primary objective: microservices migration readiness

TASK: Create comprehensive architectural documentation

CORE ANALYSIS (Required):
- Tools, frameworks, and design patterns used across the repository
- Data models and API design & versioning patterns
- Any architectural inconsistencies or deviations from language/framework best practices

LEGACY ASSESSMENT:
- Identify conflicting or multiple architectural patterns
- Recommend a best path forward with external source citations
- Distinguish between old and new architectural approaches

ADDITIONAL CONSTRAINTS:
- Primary objective: identifying architectural debt and deviations from expected patterns

OUTPUT FORMAT:
- Use clear headings and bullet points
- Prioritize findings by impact and consistency
- Include specific examples from the codebase
- Reference external best practice sources for any recommendations

Focus your entire response on: microservices migration readiness
```

</details>

```python
import talk_box as tb

# Create a base bot first
bot = tb.ChatBot().model("gpt-4.1-mini")

bot.structured_prompt(
    persona="code reviewer",
    task="Review for bugs and security issues",
    focus="critical errors that could crash the application"
)

# Pattern 4: Modular prompt components
security_component = bot.prompt_builder().core_analysis(["Auth", "Validation"]).build()
performance_component = bot.prompt_builder().core_analysis(["Queries", "Memory"]).build()
bot.chain_prompts(security_component, performance_component)
```

### **🏗️ Pre-configured Engineering Templates**

Start with expert-crafted prompts for common engineering tasks:

```python
import talk_box as tb

# Architectural analysis with attention optimization
arch_bot = (tb.ChatBot()
    .model("gpt-4-turbo")
    .prompt_builder("architectural")
    .focus_on("identifying technical debt and modernization opportunities"))

# Code review with structured feedback
review_bot = (tb.ChatBot()
    .model("gpt-4-turbo")
    .prompt_builder("code_review")
    .avoid_topics(["personal criticism", "style nitpicking"])
    .focus_on("actionable security and performance improvements"))

# Systematic debugging approach
debug_bot = (tb.ChatBot()
    .model("gpt-4-turbo")
    .prompt_builder("debugging")
    .critical_constraint("Always identify root cause, not just symptoms"))
```

### **🔄 Integrated Conversation Management**

All chat interactions automatically return conversation objects for seamless multi-turn conversations:

```python
import talk_box as tb

bot = tb.ChatBot().model("gpt-4").preset("technical_advisor")

# Every chat returns a conversation object with full history
conversation = bot.chat("What's the best way to implement authentication?")
conversation = bot.chat("What about JWT tokens specifically?", conversation)
conversation = bot.chat("Show me a Python example", conversation)

# Access the full conversation history
messages = conversation.get_messages()
latest = conversation.get_last_message()
print(f"Conversation has {conversation.get_message_count()} messages")
```

### **🎛️ Layered API Discovery**

Start simple and naturally discover advanced features as you need them:

```python
import talk_box as tb

# Layer 1: Basic ChatBot (everyone starts here)
bot = tb.ChatBot().model("gpt-4")
conversation = bot.chat("Hello!")

# Layer 2: Conversation management (discovered from return values)
conversation.add_user_message("Follow-up question")
messages = conversation.get_messages()

# Layer 3: Advanced prompt engineering (discovered through methods)
structured_response = (bot.prompt_builder()
    .persona("expert consultant")
    .critical_constraint("Be actionable and specific")
    .output_format(["Key findings", "Recommended next steps"])
    .build())

# Layer 4: Pre-configured patterns for complex tasks
arch_analysis = bot.prompt_builder("architectural").focus_on("scalability concerns")
```

### **🧪 Test Without Setup**

Works immediately with mock responses - no API keys needed for development:

```python
import talk_box as tb

# Explore the full API without external dependencies
bot = tb.ChatBot().preset("creative_writer").temperature(0.8)
conversation = bot.chat("Write a short story about a robot")

# Test attention-based prompts
structured = (bot.prompt_builder()
    .persona("storyteller")
    .core_analysis(["character development", "plot structure"])
    .build())

print(f"Messages: {conversation.get_message_count()}")
print(f"Latest: {conversation.get_last_message().content}")
```

### **Built-in Behavior Presets**

Choose from professionally crafted personalities:

- **`technical_advisor`**: Authoritative, detailed, evidence-based
- **`customer_support`**: Polite, helpful, solution-focused
- **`creative_writer`**: Imaginative, expressive, storytelling
- **`data_analyst`**: Analytical, precise, metrics-driven
- **`legal_advisor`**: Professional, thorough, disclaimer-aware

```python
import talk_box as tb

# Each preset includes tone, expertise, constraints, and system prompts
support_bot = tb.ChatBot().preset("customer_support")
creative_bot = tb.ChatBot().preset("creative_writer")
analyst_bot = tb.ChatBot().preset("data_analyst")
```

### **Chainable Configuration**

Build exactly the chatbot you need with method chaining:

```python
import talk_box as tb

specialized_bot = (
    tb.ChatBot()
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
import talk_box as tb

# Works without any API keys - perfect for development
bot = tb.ChatBot().preset("data_analyst").temperature(0.2)
conversation = bot.chat("Analyze this dataset")

# Test your conversation logic
assert conversation.get_message_count() == 2
assert "data" in conversation.get_last_message().content.lower()

# Add real LLM when ready for production
# bot.enable_llm_mode()  # Connects to actual models
```

## Complete Examples

### **Code Review with Attention Engineering**

```python
import talk_box as tb

# Create an attention-optimized code reviewer
review_bot = (
    tb.ChatBot()
    .model("gpt-4-turbo")
    .prompt_builder("code_review")
    .critical_constraint("Focus on critical security and performance issues")
    .avoid_topics(["style nitpicking", "personal preferences"])
    .output_format([
        "🔴 CRITICAL: Security vulnerabilities requiring immediate attention",
        "🟡 IMPORTANT: Performance issues with business impact",
        "📝 SUGGESTIONS: Code quality improvements",
        "Include specific line numbers and remediation examples"
    ])
)

# Structured review with attention optimization
conversation = review_bot.chat("""
Review this authentication function:

def login(username, password):
    query = f"SELECT * FROM users WHERE username='{username}' AND password='{password}'"
    result = db.execute(query)
    return result.fetchone() is not None
""")

print(f"Review: {conversation.get_last_message().content}")
```

<details>
<summary>🔍 View Generated Prompt</summary>

```
You are a senior software engineer with expertise in code review and best practices.

CRITICAL REQUIREMENTS:
- Focus on critical security and performance issues

CORE ANALYSIS (Required):
- Security: Identify potential security vulnerabilities
- Performance: Suggest optimization opportunities
- Maintainability: Recommend cleaner, more readable code
- Best Practices: Ensure adherence to language conventions
- Testing: Suggest test cases for uncovered scenarios

ADDITIONAL CONSTRAINTS:
- Primary objective: providing constructive, actionable feedback
- Avoid: personal criticism
- Avoid: style nitpicking, personal preferences

OUTPUT FORMAT:
- 🔴 Critical issues (security, bugs)
- 🟡 Improvements (performance, style)
- 🟢 Positive feedback (good practices)
- 🔴 CRITICAL: Security vulnerabilities requiring immediate attention
- 🟡 IMPORTANT: Performance issues with business impact
- 📝 SUGGESTIONS: Code quality improvements
- Include specific line numbers and remediation examples

Focus your entire response on: providing constructive, actionable feedback
```

</details>

### **Customer Support with Progressive Enhancement**

```python
import talk_box as tb

# Start simple
support_bot = tb.ChatBot().preset("customer_support")
conversation = support_bot.chat("I'm having trouble with my account")

# Add structured follow-up when needed
detailed_troubleshooting = (support_bot.prompt_builder()
    .persona("senior technical support specialist")
    .critical_constraint("Provide step-by-step solutions")
    .core_analysis([
        "Account access issues",
        "Authentication problems",
        "Password reset procedures"
    ])
    .output_format([
        "1. Immediate steps to try",
        "2. If that doesn't work, try these alternatives",
        "3. When to escalate to our technical team"
    ])
    .build())

conversation = support_bot.chat(detailed_troubleshooting, conversation)
```

<details>
<summary>🔍 View Generated Prompt</summary>

```
You are a senior technical support specialist.

CRITICAL REQUIREMENTS:
- Provide step-by-step solutions

CORE ANALYSIS (Required):
- Account access issues
- Authentication problems
- Password reset procedures

OUTPUT FORMAT:
- 1. Immediate steps to try
- 2. If that doesn't work, try these alternatives
- 3. When to escalate to our technical team
```

</details>

# Explore the conversation flow

print(f"Conversation summary:")
print(f"- Total messages: {conversation.get_message_count()}")
for i, message in enumerate(conversation.get_messages()):
print(f"{i+1}. {message.role}: {message.content[:80]}...")

```

## License

MIT License - see [LICENSE](./LICENSE) for details.
```
