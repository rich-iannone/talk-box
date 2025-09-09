<div align="center">

![Talk Box Logo](talk-box-logo.png)

**The best way to generate, test, and deploy LLM chatbots with attention-optimized prompt engineering**

[![PyPI](https://img.shields.io/pypi/v/talk-box)](https://pypi.org/project/talk-box/#history)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

[![CI Build](https://github.com/rich-iannone/talk-box/actions/workflows/ci-tests.yaml/badge.svg)](https://github.com/rich-iannone/talk-box/actions/workflows/ci-tests.yaml)
[![Repo Status](https://www.repostatus.org/badges/latest/active.svg)](https://www.repostatus.org/#active)
[![Documentation](https://img.shields.io/badge/docs-project_website-blue.svg)](https://rich-iannone.github.io/talk-box/)

[![Contributor Covenant](https://img.shields.io/badge/Contributor%20Covenant-v2.1%20adopted-ff69b4.svg)](https://www.contributor-covenant.org/version/2/1/code_of_conduct.html)

</div>

Talk Box is a Python framework that transforms prompt engineering from an art into a systematic
engineering discipline. Create effective, maintainable prompts using research-backed attention
mechanisms, then deploy them with powerful conversation management.

## Key Features

Attention-Based Prompt Engineering

- Research-backed attention optimization that works with how LLMs actually process information
- Structured templates for consistent, high-quality outputs across different use cases

Conversation Pathways

- Guide users through complex multi-step processes with intelligent flow management
- Flexible state-based conversations that adapt to natural dialogue patterns

Professional Development Tools

- Modern React-based chat interface for polished user experiences
- File attachment support for analyzing documents, images, and code directly in conversations

Built-in Quality Assurance

- Automated testing framework to verify chatbot behavior and topic compliance
- Adversarial testing with sophisticated scenarios to ensure robust performance

Developer-Friendly Design

- Chainable API that scales from simple prompts to complex conversation systems
- Pre-configured behavior presets for common use cases like customer support and technical assistance

## The Science Behind Structured Prompts

Most prompt engineering is done ad-hoc, leading to inconsistent results and hard-to-maintain systems. Talk Box applies research from transformer attention mechanisms to create a systematic approach.

Having structure matters because LLMs process information through attention patterns. Unstructured prompts scatter attention across irrelevant details, while structured prompts focus attention on what matters most.

**Key Principles**:

- **Primacy Bias**: critical information goes first (personas, constraints)
- **Attention Clustering**: related concepts are grouped together
- **Recency Bias**: final emphasis reinforces the most important outcomes
- **Hierarchical Processing**: clear sections help models organize their reasoning

Here is a schematic of how we consistently structure a prompt:

![Prompt Structure Diagram](prompt-structure.png)

Now let's see how this works in practice.

## Quick Start

### 30-Second Demo: Attention-Optimized Prompts

```python
import talk_box as tb

# Create a security-focused chatbot with structured prompts
bot = (
    tb.ChatBot()
    .provider_model("anthropic:claude-sonnet-4-20250514")
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
    )
)

# View details of the structured prompt
bot.show("prompt")
```

<details>
<summary>🔍 View Generated Prompt</summary>

```
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
```

</details>

### Even Simpler: Quick Structured Prompts with `structured_prompt()`

```python
import talk_box as tb

# Configure a bot with structured prompts in one go
doc_bot = (
    tb.ChatBot()
    .provider_model("anthropic:claude-sonnet-4-20250514")
    .structured_prompt(
        persona="technical writer",
        task="Review and improve documentation",
        constraints=["Focus on clarity and accessibility", "Include code examples"],
        format=["Content improvements", "Structure suggestions"],
        focus="making complex topics easy to understand"
    )
)

# Now just chat directly as the structure is built into the bot's system prompt
response = doc_bot.chat("Here's my API documentation draft...")
```

<details>
<summary>🔍 View Generated Prompt</summary>

```
You are a technical writer with expertise in documentation and user education.

CRITICAL REQUIREMENTS:
- Focus on clarity and accessibility

CORE ANALYSIS (Required):
- Include code examples
- Ensure content improvements
- Provide structure suggestions

OUTPUT FORMAT:
- Content improvements
- Structure suggestions

Prioritize making complex topics easy to understand
```

</details>

## The Core Feature is Attention Optimization

We can build prompts that leverage how transformer attention mechanisms actually work. The `PromptBuilder` class in Talk Box is designed to help you create these optimized prompts easily.

Let's see an example on how to create a language learning bot:

```python
import talk_box as tb

# Traditional approach (attention-diffused)
old_prompt = "Help me learn Spanish and correct my mistakes."

# Attention-optimized approach
bot = (
    tb.ChatBot()
    .provider_model("anthropic:claude-sonnet-4-20250514")
    .system_prompt(
        tb.PromptBuilder()
        .persona("experienced Spanish teacher", "conversational fluency and grammar")
        .critical_constraint("Focus on practical, everyday Spanish usage")
        .core_analysis([
            "Grammar accuracy and common mistakes",
            "Vocabulary building with context",
            "Pronunciation and speaking confidence"
        ])
        .output_format([
            "CORRECCIONES: Grammar fixes with explanations",
            "VOCABULARIO: New words with example sentences",
            "PRÁCTICA: Conversation prompts for next lesson"
        ])
        .final_emphasis("Encourage progress and build confidence through positive reinforcement")
    )
)

# Get learning right in the console
response = bot.show("console")
```

<details>
<summary>🔍 View Generated Prompt</summary>

```
You are an experienced Spanish teacher with expertise in conversational fluency and grammar.

CRITICAL REQUIREMENTS:
- Focus on practical, everyday Spanish usage

CORE ANALYSIS (Required):
- Grammar accuracy and common mistakes
- Vocabulary building with context
- Pronunciation and speaking confidence

OUTPUT FORMAT:
- CORRECCIONES: Grammar fixes with explanations
- VOCABULARIO: New words with example sentences
- PRÁCTICA: Conversation prompts for next lesson

Prioritize encouraging progress and building confidence through positive reinforcement
```

</details>

Looking at the generated prompts, we can identify several key principles that contribute to their effectiveness. The key principles of a successful structured prompt implementation are:

- **Front-loading critical information** (primacy bias)
- **Structure creates focus** (attention clustering)
- **Personas for behavioral anchoring**
- **Specific constraints prevent attention drift**
- **Final emphasis leverages recency bias**

And we've implemented these principles in our prompt engineering process, ensuring that the prompts
you create and use are effective and aligned with these best practices.

## Modern Web Interface

Launch a polished React-based chat interface by using `show("react")`:

```python
import talk_box as tb

# Create and configure a chatbot
bot = (
    tb.ChatBot(
        name="FriendlyBot",
        description="A bot that'll talk about any topic."
    )
    .model("gpt-4")
    .temperature(0.3)
    .persona("You are a friendly conversationalist")
)

bot.show("react")
```

![Talk Box React Interface](talk-box-interface.png)

The React interface provides a professional chat experience with:

- **Modern Design**: clean, responsive interface that works on desktop and mobile
- **Live Typing Indicators**: visual feedback during bot responses
- **Code Block Highlighting**: syntax highlighting for technical conversations
- **Conversation History**: full chat history with easy navigation
- **Bot Configuration Display**: see your bot's settings and persona at a glance

## Intelligent Conversation Pathways

Guide users through complex, multi-step processes with structured conversation flows that adapt to natural dialogue patterns. Pathways serve as intelligent guardrails, ensuring thorough coverage while maintaining conversational flexibility.

```python
import talk_box as tb

# Create a customer onboarding pathway
onboarding_pathway = (
    tb.Pathways(
        title="Customer Onboarding",
        desc="Welcome new customers and set up their accounts",
        activation="User is a new customer or mentions account setup",
        completion_criteria="Customer account is fully configured and ready to use"
    )
    # === STATE: welcome ===
    .state("Welcome customer and collect basic information", id="welcome")
    .required([
        "customer's full name",
        "email address",
        "company or organization name"
    ])
    .success_condition("Customer feels welcomed and basic info is collected")
    .next_state("setup")

    # === STATE: setup ===
    .state("Configure account preferences", id="setup")
    .required([
        "password created and confirmed",
        "notification preferences selected",
        "timezone configured"
    ])
    .optional(["profile photo uploaded", "team member invitations"])
    .success_condition("Account is fully configured")
    .next_state("tour")

    # === STATE: tour ===
    .state("Provide guided tour of key features", id="tour")
    .required(["main features demonstrated", "first task completed"])
    .success_condition("Customer understands how to use core functionality")
)

# Integrate pathway into a chatbot
support_bot = (
    tb.ChatBot()
    .provider_model("openai:gpt-4-turbo")
    .system_prompt(
        tb.PromptBuilder()
        .persona("helpful customer success specialist")
        .pathways(onboarding_pathway)
        .output_format([
            "Ask one focused question at a time",
            "Provide clear, step-by-step guidance",
            "Confirm understanding before moving forward"
        ])
    )
)
```

**Key Benefits of Pathways:**

- **Consistency**: Ensure important steps aren't skipped in complex processes
- **Thoroughness**: Systematically gather all necessary information
- **Flexibility**: Adapt to natural conversation patterns and user needs
- **Error Recovery**: Handle unexpected situations with defined fallback strategies
- **Scalability**: Manage complex multi-step processes that would be overwhelming without structure

## Pre-configured Engineering Templates

Start with expert-crafted prompts for common engineering tasks:

```python
import talk_box as tb

# Architectural analysis with attention optimization
arch_bot = (
    tb.ChatBot()
    .provider_model("anthropic:claude-sonnet-4-20250514")
    .prompt_builder("architectural")
    .focus_on("identifying technical debt and modernization opportunities")
)

# Code review with structured feedback
review_bot = (
    tb.ChatBot()
    .provider_model("anthropic:claude-sonnet-4-20250514")
    .prompt_builder("code_review")
    .avoid_topics(["personal criticism", "style nitpicking"])
    .focus_on("actionable security and performance improvements")
)

# Systematic debugging approach
debug_bot = (
    tb.ChatBot()
    .provider_model("anthropic:claude-sonnet-4-20250514")
    .prompt_builder("debugging")
    .critical_constraint("Always identify root cause, not just symptoms")
)
```

## Integrated Conversation Management

All chat interactions automatically return conversation objects for seamless multi-turn conversations:

```python
import talk_box as tb

bot = tb.ChatBot().model("gpt-4o-mini").preset("technical_advisor")

# Every chat returns a conversation object with full history
conversation = bot.chat("What's the best way to implement authentication?")
conversation = bot.chat("What about JWT tokens specifically?", conversation)
conversation = bot.chat("Show me a Python example", conversation)

# Access the full conversation history
messages = conversation.get_messages()
latest = conversation.get_last_message()
print(f"Conversation has {conversation.get_message_count()} messages")
```

```
Conversation has 6 messages
```

## File Attachments for AI Analysis

Analyze documents, images, and code files directly in your conversations. Talk Box automatically handles different file types and integrates seamlessly with your prompts:

```python
import talk_box as tb

# Step 1: Create a ChatBot for analysis
bot = tb.ChatBot().provider_model("openai:gpt-4-turbo")

# Step 2: Attach a file with analysis prompt
files = tb.Attachments("report.pdf").with_prompt(
    "Summarize the key findings in this report."
)

# Step 3: Get LLM analysis
analysis = bot.chat(files)
```

<div align="center">
<img src="https://rich-iannone.github.io/talk-box/assets/abraham-results.png" width="800px">
</div>

**Supported file types**: PDFs, images (PNG, JPG), text files (Python, JavaScript, Markdown, CSV, JSON), and more. Perfect for code reviews, document analysis, data insights, and visual content evaluation.

## Built-in Behavior Presets

Choose from a few professionally-crafted personalities:

- **`technical_advisor`**: authoritative, detailed, evidence-based
- **`customer_support`**: polite, helpful, solution-focused
- **`creative_writer`**: imaginative, expressive, storytelling
- **`data_analyst`**: analytical, precise, metrics-driven
- **`legal_advisor`**: professional, thorough, disclaimer-aware

```python
import talk_box as tb

support_bot = tb.ChatBot().preset("customer_support")
creative_bot = tb.ChatBot().preset("creative_writer")
analyst_bot = tb.ChatBot().preset("data_analyst")
```

## Automated Avoid Topics Testing

Verify that your ChatBots properly refuse to engage with prohibited topics through our adversarial testing framework. With one function, `autotest_avoid_topics()`, this is all very easy:

```python
import talk_box as tb

# Create a wellness coach that should avoid gaming topics
wellness_bot = (
    tb.ChatBot()
    .provider_model("openai:gpt-4-turbo")
    .system_prompt(
        tb.PromptBuilder()
        .persona("productivity and wellness coach", "work-life balance")
        .avoid_topics(["video games", "gaming"])
        .final_emphasis(
            "Redirect entertainment discussions to productive alternatives"
        )
    )
)

# Test compliance with sophisticated adversarial strategies
results = tb.autotest_avoid_topics(wellness_bot, test_intensity="thorough")

# View rich results in notebooks or check compliance programmatically
print(f"Compliance rate: {results.summary['success_rate']:.1%}")
print(f"Violations found: {results.summary['violation_count']}")
```

You can generate a visual HTML report for easy compliance review in a notebook environment:

```python
results
```

![Avoid Topics Test Results](avoid-topics-test-results.png)

The HTML report displays each adversarial conversation with **color-coded compliance indicators**, making it easy to spot transgressions at a glance. You can quickly scan through dozens of test scenarios and immediately identify where your chatbot may have inappropriately engaged with forbidden topics.

The testing framework uses multiple adversarial strategies:

- **Direct requests** for prohibited information
- **Indirect/hypothetical** scenarios
- **Emotional appeals** and urgency
- **Role-playing** (academic, professional contexts)
- **Context shifting** from acceptable to prohibited topics
- **Persistent follow-ups** after initial refusal

Results provide detailed analysis including conversation transcripts, violation detection with severity ratings, and export capabilities for quality assurance workflows.

**Pathway Testing**: Talk Box also includes `autotest_pathways()` to verify that chatbots properly follow defined conversation flows, ensuring consistent user experiences across complex multi-step interactions.

## Installation

The Talk Box framework can be installed via pip:

```bash
pip install talk-box
```

If you encounter a bug, have usage questions, or want to share ideas to make this package better, please feel free to [open an issue](https://github.com/rich-iannone/talk-box/issues).

## Roadmap

Talk Box is actively evolving to become a comprehensive prompt engineering and LLM deployment framework. Here are the major features and enhancements planned for future releases:

### Advanced Prompt Engineering

- **Dynamic Prompt Adaptation**: automatically adjust prompt structure based on model capabilities and response quality metrics
- **Multi-Modal Prompt Support**: extend structured prompting to include images, audio, and video inputs with attention optimization
- **Advanced Pathway Patterns**: additional conversation flow patterns for complex scenarios like negotiations, troubleshooting, and collaborative planning

### Enhanced Testing & Validation

- **Prompt Performance Analytics**: comprehensive metrics dashboard for prompt effectiveness, response quality, and user satisfaction
- **A/B Testing Framework**: built-in experimentation tools for comparing prompt variations with statistical significance testing
- **Automated Regression Testing**: continuous validation of chatbot behavior across software updates and model changes

### Advanced Analytics & Insights

- **Conversation Intelligence**: automatic extraction of insights, sentiment analysis, and topic modeling from chat histories
- **ROI & Business Impact Tracking**: measure chatbot effectiveness with conversion rates, customer satisfaction, and cost savings
- **Predictive Conversation Analytics**: AI-powered prediction of conversation outcomes and proactive intervention suggestions

### Developer Experience Enhancements

- **Visual Prompt Builder**: interface for creating complex prompt structures without coding
- **Better Prompt Debugging**: visualization of attention patterns, token usage, and reasoning paths

### Emerging Technologies

- **Agentic Workflow Automation**: transformation of chatbots into autonomous agents capable of executing complex, multi-step tasks

## Code of Conduct

Please note that the Talk Box project is released with a [contributor code of conduct](https://www.contributor-covenant.org/version/2/1/code_of_conduct/).
By participating in this project you agree to abide by its terms.

## License

Talk Box is licensed under the MIT license.

© Richard Iannone

## 🏛️ Governance

This project is primarily maintained by [Rich Iannone](https://bsky.app/profile/richmeister.bsky.social). Other authors may occasionally assist with some of these duties.
