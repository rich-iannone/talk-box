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

## Why Talk Box?

- **Attention-Based Prompt Engineering**: build prompts that leverage how LLMs actually process
  information
- **Layered API Design**: start simple, discover advanced features as you need them
- **Multiple Usage Patterns**: from quick structured prompts to complex modular components
- **Integrated Conversation Management**: seamless multi-turn conversations with full history
- **Built-in Behavior Presets**: professional templates for common engineering tasks
- **Test-First Design**: develop and test without API keys, deploy when ready

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
        .build()
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
        .build()
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

## Built-in Behavior Presets

Choose from professionally crafted personalities:

- **`technical_advisor`**: authoritative, detailed, evidence-based
- **`customer_support`**: polite, helpful, solution-focused
- **`creative_writer`**: imaginative, expressive, storytelling
- **`data_analyst`**: analytical, precise, metrics-driven
- **`legal_advisor`**: professional, thorough, disclaimer-aware

```python
import talk_box as tb

# Each preset includes tone, expertise, constraints, and system prompts
support_bot = tb.ChatBot().preset("customer_support")
creative_bot = tb.ChatBot().preset("creative_writer")
analyst_bot = tb.ChatBot().preset("data_analyst")
```

## Automated Avoid Topics Testing

Ensure your ChatBots properly refuse to engage with prohibited topics using sophisticated adversarial testing:

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
        .build()
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

The testing framework provides a **rich visual interface** that makes compliance verification intuitive and thorough:

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

### Installation

The Talk Box framework can be installed via pip:

```bash
pip install talk-box
```

If you encounter a bug, have usage questions, or want to share ideas to make this package better, please feel free to [open an issue](https://github.com/rich-iannone/talk-box/issues).

## Code of Conduct

Please note that the Talk Box project is released with a [contributor code of conduct](https://www.contributor-covenant.org/version/2/1/code_of_conduct/).
By participating in this project you agree to abide by its terms.

## License

Talk Box is licensed under the MIT license.

© Richard Iannone

## 🏛️ Governance

This project is primarily maintained by [Rich Iannone](https://bsky.app/profile/richmeister.bsky.social). Other authors may occasionally assist with some of these duties.
