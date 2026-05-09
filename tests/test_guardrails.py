import pytest

from talk_box.guardrails import (
    Guard,
    GuardAction,
    GuardActivation,
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


# ---------------------------------------------------------------------------
# GuardResult tests
# ---------------------------------------------------------------------------


class TestGuardResult:
    def test_passed(self):
        result = GuardResult.passed()
        assert result.action == GuardAction.PASSED
        assert result.message is None
        assert result.reason is None

    def test_passed_with_reason(self):
        result = GuardResult.passed(reason="All clean")
        assert result.action == GuardAction.PASSED
        assert result.reason == "All clean"

    def test_blocked(self):
        result = GuardResult.blocked("Contains PII")
        assert result.action == GuardAction.BLOCKED
        assert result.reason == "Contains PII"
        assert result.message is None

    def test_rewrite(self):
        result = GuardResult.rewrite("cleaned text", reason="Stripped email")
        assert result.action == GuardAction.REWRITTEN
        assert result.message == "cleaned text"
        assert result.reason == "Stripped email"

    def test_rewrite_without_reason(self):
        result = GuardResult.rewrite("new text")
        assert result.action == GuardAction.REWRITTEN
        assert result.message == "new text"
        assert result.reason is None

    def test_immutable(self):
        result = GuardResult.passed()
        with pytest.raises(Exception):
            result.action = GuardAction.BLOCKED  # type: ignore[misc]


# ---------------------------------------------------------------------------
# @guardrail decorator tests
# ---------------------------------------------------------------------------


class TestGuardrailDecorator:
    def test_bare_decorator(self):
        @guardrail
        def my_guard(text: str) -> GuardResult:
            return GuardResult.passed()

        assert isinstance(my_guard, Guard)
        assert my_guard.name == "my_guard"
        assert my_guard.phase == GuardPhase.BOTH

    def test_decorator_with_name(self):
        @guardrail(name="Custom Name")
        def my_guard(text: str) -> GuardResult:
            return GuardResult.passed()

        assert my_guard.name == "Custom Name"

    def test_decorator_with_phase(self):
        @guardrail(phase=GuardPhase.INPUT)
        def input_only(text: str) -> GuardResult:
            return GuardResult.passed()

        assert input_only.phase == GuardPhase.INPUT

    def test_decorator_with_all_options(self):
        @guardrail(name="PII Filter", phase=GuardPhase.OUTPUT)
        def strip_pii(text: str) -> GuardResult:
            return GuardResult.passed()

        assert strip_pii.name == "PII Filter"
        assert strip_pii.phase == GuardPhase.OUTPUT

    def test_guard_check_method(self):
        @guardrail
        def block_hello(text: str) -> GuardResult:
            if "hello" in text.lower():
                return GuardResult.blocked("Greeting detected")
            return GuardResult.passed()

        result = block_hello.check("Hello world")
        assert result.action == GuardAction.BLOCKED
        assert result.guard_name == "block_hello"

        result = block_hello.check("Goodbye world")
        assert result.action == GuardAction.PASSED
        assert result.guard_name == "block_hello"


# ---------------------------------------------------------------------------
# GuardPipeline tests
# ---------------------------------------------------------------------------


class TestGuardPipeline:
    def test_empty_pipeline_passes(self):
        pipeline = GuardPipeline()
        result = pipeline.run("anything", GuardPhase.INPUT)
        assert result.text == "anything"
        assert result.blocked is False
        assert result.activations == []

    def test_single_passing_guard(self):
        pipeline = GuardPipeline()
        pipeline.add(Guard(name="pass", func=lambda t: GuardResult.passed()))
        result = pipeline.run("test", GuardPhase.INPUT)
        assert result.text == "test"
        assert result.blocked is False
        assert len(result.activations) == 1
        assert result.activations[0].action == GuardAction.PASSED

    def test_single_blocking_guard(self):
        pipeline = GuardPipeline()
        pipeline.add(Guard(name="block", func=lambda t: GuardResult.blocked("nope")))
        result = pipeline.run("test", GuardPhase.INPUT)
        assert result.text is None
        assert result.blocked is True
        assert result.block_reason == "nope"

    def test_rewrite_modifies_text(self):
        pipeline = GuardPipeline()
        pipeline.add(
            Guard(
                name="upper",
                func=lambda t: GuardResult.rewrite(t.upper(), reason="uppercased"),
            )
        )
        result = pipeline.run("hello", GuardPhase.INPUT)
        assert result.text == "HELLO"
        assert result.blocked is False

    def test_guards_chain_rewrites(self):
        """Subsequent guards see the rewritten text."""
        pipeline = GuardPipeline()
        pipeline.add(
            Guard(
                name="add_prefix",
                func=lambda t: GuardResult.rewrite(f"[PREFIX] {t}"),
            )
        )
        pipeline.add(
            Guard(
                name="add_suffix",
                func=lambda t: GuardResult.rewrite(f"{t} [SUFFIX]"),
            )
        )
        result = pipeline.run("hello", GuardPhase.INPUT)
        assert result.text == "[PREFIX] hello [SUFFIX]"

    def test_block_short_circuits(self):
        """A blocking guard stops the pipeline; later guards don't run."""
        pipeline = GuardPipeline()
        ran_second = []
        pipeline.add(Guard(name="block", func=lambda t: GuardResult.blocked("stop")))
        pipeline.add(
            Guard(
                name="never_runs",
                func=lambda t: (ran_second.append(True), GuardResult.passed())[1],
            )
        )
        result = pipeline.run("test", GuardPhase.INPUT)
        assert result.blocked is True
        assert ran_second == []
        assert len(result.activations) == 1

    def test_phase_filtering_input(self):
        """Output-only guards don't run during input phase."""
        pipeline = GuardPipeline()
        pipeline.add(
            Guard(
                name="output_only",
                func=lambda t: GuardResult.blocked("should not fire"),
                phase=GuardPhase.OUTPUT,
            )
        )
        result = pipeline.run("test", GuardPhase.INPUT)
        assert result.blocked is False
        assert result.activations == []

    def test_phase_filtering_output(self):
        """Input-only guards don't run during output phase."""
        pipeline = GuardPipeline()
        pipeline.add(
            Guard(
                name="input_only",
                func=lambda t: GuardResult.blocked("should not fire"),
                phase=GuardPhase.INPUT,
            )
        )
        result = pipeline.run("test", GuardPhase.OUTPUT)
        assert result.blocked is False

    def test_both_phase_runs_always(self):
        """Guards with phase=BOTH run in both phases."""
        pipeline = GuardPipeline()
        pipeline.add(
            Guard(
                name="always",
                func=lambda t: GuardResult.passed(),
                phase=GuardPhase.BOTH,
            )
        )
        result_in = pipeline.run("test", GuardPhase.INPUT)
        result_out = pipeline.run("test", GuardPhase.OUTPUT)
        assert len(result_in.activations) == 1
        assert len(result_out.activations) == 1

    def test_stats_tracking(self):
        pipeline = GuardPipeline()
        pipeline.add(Guard(name="pass_guard", func=lambda t: GuardResult.passed()))
        pipeline.run("a", GuardPhase.INPUT)
        pipeline.run("b", GuardPhase.INPUT)
        stats = pipeline.stats()
        assert stats["pass_guard"]["passed"] == 2
        assert stats["pass_guard"]["blocked"] == 0

    def test_stats_reset(self):
        pipeline = GuardPipeline()
        pipeline.add(Guard(name="g", func=lambda t: GuardResult.passed()))
        pipeline.run("x", GuardPhase.INPUT)
        pipeline.reset_stats()
        assert pipeline.stats()["g"]["passed"] == 0

    def test_activation_records_original_text_on_action(self):
        pipeline = GuardPipeline()
        pipeline.add(
            Guard(
                name="rewriter",
                func=lambda t: GuardResult.rewrite("new"),
            )
        )
        result = pipeline.run("original", GuardPhase.INPUT)
        assert result.activations[0].original_text == "original"

    def test_activation_no_original_text_on_pass(self):
        pipeline = GuardPipeline()
        pipeline.add(Guard(name="p", func=lambda t: GuardResult.passed()))
        result = pipeline.run("msg", GuardPhase.INPUT)
        assert result.activations[0].original_text is None


# ---------------------------------------------------------------------------
# Built-in guards: no_pii
# ---------------------------------------------------------------------------


class TestNoPii:
    def test_no_pii_passes_clean_text(self):
        guard = no_pii()
        result = guard.check("Hello, how are you?")
        assert result.action == GuardAction.PASSED

    def test_detects_email(self):
        guard = no_pii()
        result = guard.check("Contact me at user@example.com please")
        assert result.action == GuardAction.REWRITTEN
        assert "[EMAIL]" in result.message
        assert "user@example.com" not in result.message

    def test_detects_phone(self):
        guard = no_pii()
        result = guard.check("Call me at 555-123-4567")
        assert result.action == GuardAction.REWRITTEN
        assert "[PHONE]" in result.message

    def test_detects_ssn(self):
        guard = no_pii()
        result = guard.check("My SSN is 123-45-6789")
        assert result.action == GuardAction.REWRITTEN
        assert "[SSN]" in result.message

    def test_detects_credit_card(self):
        guard = no_pii()
        result = guard.check("Card: 4111 1111 1111 1111")
        assert result.action == GuardAction.REWRITTEN
        assert "[CARD]" in result.message

    def test_detects_multiple_pii(self):
        guard = no_pii()
        result = guard.check("Email: a@b.com, phone: 555-111-2222")
        assert result.action == GuardAction.REWRITTEN
        assert "[EMAIL]" in result.message
        assert "[PHONE]" in result.message

    def test_block_mode(self):
        guard = no_pii(action="block")
        result = guard.check("My email is user@example.com")
        assert result.action == GuardAction.BLOCKED
        assert "email" in result.reason.lower()

    def test_custom_patterns(self):
        guard = no_pii(patterns=[r"\bMRN-\d+\b"])
        result = guard.check("Patient MRN-12345")
        assert result.action == GuardAction.REWRITTEN
        assert "[PII]" in result.message

    def test_phase_is_both(self):
        guard = no_pii()
        assert guard.phase == GuardPhase.BOTH


# ---------------------------------------------------------------------------
# Built-in guards: max_response_length
# ---------------------------------------------------------------------------


class TestMaxResponseLength:
    def test_short_text_passes(self):
        guard = max_response_length(100)
        result = guard.check("Short text")
        assert result.action == GuardAction.PASSED

    def test_long_text_truncated(self):
        guard = max_response_length(20)
        long_text = "This is a much longer text that exceeds the limit"
        result = guard.check(long_text)
        assert result.action == GuardAction.REWRITTEN
        assert len(result.message) <= 25  # 20 + room for "..."
        assert result.message.endswith("...")

    def test_exact_limit_passes(self):
        guard = max_response_length(5)
        result = guard.check("Hello")
        assert result.action == GuardAction.PASSED

    def test_output_only_phase(self):
        guard = max_response_length(100)
        assert guard.phase == GuardPhase.OUTPUT


# ---------------------------------------------------------------------------
# Built-in guards: tone_check
# ---------------------------------------------------------------------------


class TestToneCheck:
    def test_professional_passes_clean(self):
        guard = tone_check("professional")
        result = guard.check("I appreciate your inquiry. Let me assist you.")
        assert result.action == GuardAction.PASSED

    def test_professional_blocks_casual(self):
        guard = tone_check("professional")
        result = guard.check("lol that's hilarious bruh")
        assert result.action == GuardAction.BLOCKED
        assert "professional" in result.reason

    def test_formal_blocks_casual(self):
        guard = tone_check("formal")
        result = guard.check("Hey, what's up? This is awesome!")
        assert result.action == GuardAction.BLOCKED

    def test_unknown_tone_passes(self):
        guard = tone_check("mysterious")
        result = guard.check("Anything goes here")
        assert result.action == GuardAction.PASSED

    def test_custom_indicators(self):
        guard = tone_check(
            "kid_friendly",
            indicators={"kid_friendly": ["damn", "hell", "crap"]},
        )
        result = guard.check("What the hell is this?")
        assert result.action == GuardAction.BLOCKED

    def test_output_only_phase(self):
        guard = tone_check("professional")
        assert guard.phase == GuardPhase.OUTPUT


# ---------------------------------------------------------------------------
# Built-in guards: disclaimer_required
# ---------------------------------------------------------------------------


class TestDisclaimerRequired:
    def test_disclaimer_present_passes(self):
        disclaimer = "Not financial advice."
        guard = disclaimer_required(disclaimer)
        result = guard.check(f"Here's my analysis.\n\n{disclaimer}")
        assert result.action == GuardAction.PASSED

    def test_disclaimer_missing_appended(self):
        disclaimer = "Not financial advice."
        guard = disclaimer_required(disclaimer)
        result = guard.check("Here's my analysis.")
        assert result.action == GuardAction.REWRITTEN
        assert result.message.endswith(disclaimer)
        assert "Here's my analysis." in result.message

    def test_disclaimer_prepended(self):
        disclaimer = "IMPORTANT: Not financial advice."
        guard = disclaimer_required(disclaimer, position="start")
        result = guard.check("Here's my analysis.")
        assert result.action == GuardAction.REWRITTEN
        assert result.message.startswith(disclaimer)

    def test_output_only_phase(self):
        guard = disclaimer_required("Disclaimer.")
        assert guard.phase == GuardPhase.OUTPUT


# ---------------------------------------------------------------------------
# Built-in guards: must_cite_sources
# ---------------------------------------------------------------------------


class TestMustCiteSources:
    def test_url_citation_passes(self):
        guard = must_cite_sources()
        result = guard.check("See https://example.com for details.")
        assert result.action == GuardAction.PASSED

    def test_bracketed_citation_passes(self):
        guard = must_cite_sources()
        result = guard.check("According to the study [1], this is true.")
        assert result.action == GuardAction.PASSED

    def test_author_year_passes(self):
        guard = must_cite_sources()
        result = guard.check("As noted by (Smith, 2024), the evidence shows...")
        assert result.action == GuardAction.PASSED

    def test_no_citation_blocks(self):
        guard = must_cite_sources()
        result = guard.check("The sky is blue and grass is green.")
        assert result.action == GuardAction.BLOCKED
        assert "citation" in result.reason.lower()

    def test_min_citations(self):
        guard = must_cite_sources(min_citations=2)
        result = guard.check("See [1] for details.")
        assert result.action == GuardAction.BLOCKED
        result = guard.check("See [1] and [2] for details.")
        assert result.action == GuardAction.PASSED

    def test_output_only_phase(self):
        guard = must_cite_sources()
        assert guard.phase == GuardPhase.OUTPUT


# ---------------------------------------------------------------------------
# Built-in guards: max_input_length
# ---------------------------------------------------------------------------


class TestMaxInputLength:
    def test_short_input_passes(self):
        guard = max_input_length(1000)
        result = guard.check("Short message")
        assert result.action == GuardAction.PASSED

    def test_long_input_blocked(self):
        guard = max_input_length(10)
        result = guard.check("This message is too long")
        assert result.action == GuardAction.BLOCKED
        assert "too long" in result.reason.lower()

    def test_input_only_phase(self):
        guard = max_input_length(100)
        assert guard.phase == GuardPhase.INPUT


# ---------------------------------------------------------------------------
# Built-in guards: keyword_block
# ---------------------------------------------------------------------------


class TestKeywordBlock:
    def test_clean_text_passes(self):
        guard = keyword_block(["password", "secret"])
        result = guard.check("Hello, how can I help?")
        assert result.action == GuardAction.PASSED

    def test_keyword_detected_blocks(self):
        guard = keyword_block(["password", "secret"])
        result = guard.check("My password is abc123")
        assert result.action == GuardAction.BLOCKED
        assert "password" in result.reason

    def test_case_insensitive_by_default(self):
        guard = keyword_block(["SECRET"])
        result = guard.check("this is a secret message")
        assert result.action == GuardAction.BLOCKED

    def test_case_sensitive(self):
        guard = keyword_block(["SECRET"], case_sensitive=True)
        result = guard.check("this is a secret message")
        assert result.action == GuardAction.PASSED
        result = guard.check("this is a SECRET message")
        assert result.action == GuardAction.BLOCKED

    def test_custom_phase(self):
        guard = keyword_block(["x"], phase=GuardPhase.INPUT)
        assert guard.phase == GuardPhase.INPUT


# ---------------------------------------------------------------------------
# ChatBot integration tests
# ---------------------------------------------------------------------------


class TestChatBotGuardrailIntegration:
    def test_chatbot_has_guardrail_method(self):
        from talk_box import ChatBot

        bot = ChatBot()
        assert hasattr(bot, "guardrail")

    def test_guardrail_chaining(self):
        from talk_box import ChatBot

        bot = ChatBot().guardrail(no_pii()).guardrail(max_response_length(500))
        assert len(bot._guard_pipeline.guards) == 2

    def test_guardrail_type_check(self):
        from talk_box import ChatBot

        bot = ChatBot()
        with pytest.raises(TypeError, match="Expected a Guard instance"):
            bot.guardrail("not a guard")  # type: ignore[arg-type]

    def test_input_guard_blocks_message(self):
        from talk_box import ChatBot

        bot = ChatBot().guardrail(max_input_length(5))
        convo = bot.chat("This message is way too long")
        last_msg = convo.get_last_message()
        assert "Blocked by guardrail" in last_msg.content

    def test_input_guard_rewrites_message(self):
        from talk_box import ChatBot

        bot = ChatBot().guardrail(no_pii())
        convo = bot.chat("Email me at user@example.com")
        # The user message in conversation should have PII stripped
        messages = convo.get_messages()
        user_msg = messages[0]
        assert "[EMAIL]" in user_msg.content
        assert "user@example.com" not in user_msg.content

    def test_output_guard_rewrites_response(self):
        from talk_box import ChatBot

        disclaimer = "This is not advice."
        bot = ChatBot().guardrail(disclaimer_required(disclaimer))
        convo = bot.chat("Hello")
        last_msg = convo.get_last_message()
        assert disclaimer in last_msg.content

    def test_output_guard_blocks_response(self):
        from talk_box import ChatBot

        # Use a guard that blocks any output not containing citations
        bot = ChatBot().guardrail(must_cite_sources())
        convo = bot.chat("Hello")
        last_msg = convo.get_last_message()
        assert "blocked by guardrail" in last_msg.content.lower()

    def test_guard_stats(self):
        from talk_box import ChatBot

        bot = ChatBot().guardrail(no_pii())
        bot.chat("Clean message")
        bot.chat("Email: a@b.com")
        stats = bot.guard_stats()
        assert "no_pii" in stats

    def test_multiple_guards_compose(self):
        from talk_box import ChatBot

        bot = (
            ChatBot()
            .guardrail(no_pii())
            .guardrail(keyword_block(["forbidden"], phase=GuardPhase.INPUT))
        )
        # PII guard rewrites, keyword guard blocks
        convo = bot.chat("forbidden content")
        last_msg = convo.get_last_message()
        assert "Blocked by guardrail" in last_msg.content


# ---------------------------------------------------------------------------
# mock_responses tests
# ---------------------------------------------------------------------------


class TestMockResponses:
    def test_mock_response_returned(self):
        from talk_box import ChatBot

        bot = ChatBot().mock_responses(["Hello from mock!"])
        convo = bot.chat("Hi")
        assert convo.get_last_message().content == "Hello from mock!"

    def test_mock_responses_consumed_in_order(self):
        from talk_box import ChatBot

        bot = ChatBot().mock_responses(["First", "Second", "Third"])
        c1 = bot.chat("a")
        c2 = bot.chat("b")
        c3 = bot.chat("c")
        assert c1.get_last_message().content == "First"
        assert c2.get_last_message().content == "Second"
        assert c3.get_last_message().content == "Third"

    def test_mock_exhausted_falls_back_to_echo(self):
        from talk_box import ChatBot

        bot = ChatBot().mock_responses(["Only one"])
        bot.chat("first")
        convo = bot.chat("second")
        # After mock is exhausted, should fall back to echo
        assert "Echo:" in convo.get_last_message().content

    def test_mock_with_output_guard(self):
        from talk_box import ChatBot

        bot = (
            ChatBot().guardrail(disclaimer_required("DISCLAIMER")).mock_responses(["Some advice."])
        )
        convo = bot.chat("Help me")
        content = convo.get_last_message().content
        # Guard should append disclaimer to the mock response
        assert "Some advice." in content
        assert "DISCLAIMER" in content

    def test_mock_with_input_guard(self):
        from talk_box import ChatBot

        bot = ChatBot().guardrail(no_pii()).mock_responses(["Got your message."])
        convo = bot.chat("Email me at user@test.com")
        # Input should be rewritten, mock response still used
        user_msg = convo.get_messages()[0]
        assert "[EMAIL]" in user_msg.content
        assert convo.get_last_message().content == "Got your message."

    def test_mock_responses_chaining(self):
        from talk_box import ChatBot

        bot = ChatBot().mock_responses(["resp"]).model("gpt-4")
        convo = bot.chat("test")
        assert convo.get_last_message().content == "resp"


# ---------------------------------------------------------------------------
# resolve_guards tests
# ---------------------------------------------------------------------------


class TestResolveGuards:
    def test_resolve_string_spec(self):
        guards = resolve_guards(["no_pii"])
        assert len(guards) == 1
        assert guards[0].name == "no_pii"

    def test_resolve_dict_spec_with_kwargs(self):
        guards = resolve_guards([{"disclaimer_required": {"disclaimer_text": "Not advice."}}])
        assert len(guards) == 1
        assert guards[0].name == "disclaimer_required"
        # Verify the guard actually works
        result = guards[0].check("Hello")
        assert result.action == GuardAction.REWRITTEN
        assert "Not advice." in result.message

    def test_resolve_multiple_specs(self):
        guards = resolve_guards(
            [
                "no_pii",
                {"disclaimer_required": {"disclaimer_text": "Disclaimer."}},
            ]
        )
        assert len(guards) == 2
        assert guards[0].name == "no_pii"
        assert guards[1].name == "disclaimer_required"

    def test_resolve_empty_list(self):
        guards = resolve_guards([])
        assert guards == []

    def test_resolve_unknown_guard_string(self):
        with pytest.raises(ValueError, match="Unknown guard 'nonexistent'"):
            resolve_guards(["nonexistent"])

    def test_resolve_unknown_guard_dict(self):
        with pytest.raises(ValueError, match="Unknown guard 'nonexistent'"):
            resolve_guards([{"nonexistent": {}}])

    def test_resolve_dict_with_multiple_keys(self):
        with pytest.raises(ValueError, match="exactly one key"):
            resolve_guards([{"no_pii": {}, "tone_check": {}}])

    def test_resolve_invalid_type(self):
        with pytest.raises(TypeError, match="str or dict"):
            resolve_guards([42])

    def test_resolve_all_builtin_guards(self):
        specs = [
            "no_pii",
            {"max_response_length": {"max_chars": 100}},
            {"tone_check": {"expected_tone": "professional"}},
            {"disclaimer_required": {"disclaimer_text": "Test."}},
            "must_cite_sources",
            {"max_input_length": {"max_chars": 500}},
            {"keyword_block": {"keywords": ["bad"]}},
        ]
        guards = resolve_guards(specs)
        assert len(guards) == 7


# ---------------------------------------------------------------------------
# Persona-aware guard defaults tests
# ---------------------------------------------------------------------------


class TestPersonaDefaultGuards:
    def test_financial_advisor_gets_guards(self):
        from talk_box import ChatBot

        bot = ChatBot().persona_pack("financial_advisor")
        stats = bot.guard_stats()
        assert "no_pii" in stats
        assert "disclaimer_required" in stats

    def test_financial_advisor_disclaimer_works(self):
        from talk_box import ChatBot

        bot = ChatBot().persona_pack("financial_advisor").mock_responses(["Save 20% of income."])
        convo = bot.chat("How do I save?")
        content = convo.get_last_message().content
        assert "not personalized financial advice" in content

    def test_legal_info_gets_guards(self):
        from talk_box import ChatBot

        bot = ChatBot().persona_pack("legal_info")
        stats = bot.guard_stats()
        assert "no_pii" in stats
        assert "disclaimer_required" in stats

    def test_legal_info_disclaimer_works(self):
        from talk_box import ChatBot

        bot = ChatBot().persona_pack("legal_info").mock_responses(["Fair use allows limited use."])
        convo = bot.chat("What is fair use?")
        content = convo.get_last_message().content
        assert "not legal advice" in content

    def test_hr_advisor_gets_guards(self):
        from talk_box import ChatBot

        bot = ChatBot().persona_pack("hr_advisor")
        stats = bot.guard_stats()
        assert "no_pii" in stats
        assert "disclaimer_required" in stats

    def test_customer_support_gets_pii_only(self):
        from talk_box import ChatBot

        bot = ChatBot().persona_pack("customer_support_tier1")
        stats = bot.guard_stats()
        assert "no_pii" in stats
        assert "disclaimer_required" not in stats

    def test_sales_assistant_gets_pii_only(self):
        from talk_box import ChatBot

        bot = ChatBot().persona_pack("sales_assistant")
        stats = bot.guard_stats()
        assert "no_pii" in stats
        assert "disclaimer_required" not in stats

    def test_code_reviewer_gets_no_guards(self):
        from talk_box import ChatBot

        bot = ChatBot().persona_pack("code_reviewer")
        stats = bot.guard_stats()
        assert stats == {}

    def test_default_guards_false_skips_guards(self):
        from talk_box import ChatBot

        bot = ChatBot().persona_pack("financial_advisor", default_guards=False)
        stats = bot.guard_stats()
        assert stats == {}

    def test_pii_guard_fires_on_persona_bot(self):
        from talk_box import ChatBot

        bot = ChatBot().persona_pack("customer_support_tier1").mock_responses(["I'll help you."])
        convo = bot.chat("My email is test@example.com")
        user_msg = convo.get_messages()[0]
        assert "[EMAIL]" in user_msg.content

    def test_manual_guard_added_after_persona_defaults(self):
        from talk_box import ChatBot

        bot = ChatBot().persona_pack("financial_advisor").guardrail(max_response_length(50))
        stats = bot.guard_stats()
        assert "no_pii" in stats
        assert "disclaimer_required" in stats
        assert "max_response_length" in stats


class TestStateScopedGuardrails:
    """Tests for state-scoped guardrails (guards that only run in specific pathway states)."""

    def _make_blocking_guard(self, name: str, **kwargs) -> Guard:
        def block_all(text: str) -> GuardResult:
            return GuardResult.blocked("Blocked by test guard")

        return Guard(name=name, func=block_all, **kwargs)

    def test_guard_no_states_runs_always(self):
        guard = self._make_blocking_guard("always_on")
        assert guard.states is None
        assert guard.applies_to_state(None) is True
        assert guard.applies_to_state("review") is True
        assert guard.applies_to_state("anything") is True

    def test_guard_with_states_matches(self):
        guard = self._make_blocking_guard("review_only", states=["review", "escalation"])
        assert guard.applies_to_state("review") is True
        assert guard.applies_to_state("escalation") is True
        assert guard.applies_to_state("greeting") is False
        assert guard.applies_to_state(None) is False

    def test_pipeline_skips_state_scoped_guard(self):
        pipeline = GuardPipeline()

        # A guard that only runs in "review" state
        review_guard = self._make_blocking_guard("review_blocker", states=["review"])
        pipeline.add(review_guard)

        # In "greeting" state, should pass
        result = pipeline.run("hello", GuardPhase.INPUT, current_state="greeting")
        assert not result.blocked
        assert result.text == "hello"

    def test_pipeline_runs_state_scoped_guard_when_matching(self):
        pipeline = GuardPipeline()

        review_guard = self._make_blocking_guard("review_blocker", states=["review"])
        pipeline.add(review_guard)

        # In "review" state, should block
        result = pipeline.run("hello", GuardPhase.INPUT, current_state="review")
        assert result.blocked

    def test_pipeline_state_scoped_with_no_state(self):
        """State-scoped guard is skipped when no state is active."""
        pipeline = GuardPipeline()

        review_guard = self._make_blocking_guard("review_only", states=["review"])
        pipeline.add(review_guard)

        result = pipeline.run("hello", GuardPhase.INPUT, current_state=None)
        assert not result.blocked

    def test_pipeline_mixes_scoped_and_global_guards(self):
        """Global guards run alongside state-scoped guards."""
        pipeline = GuardPipeline()

        # Global guard that rewrites
        def rewrite_fn(text: str) -> GuardResult:
            return GuardResult.rewrite(text.upper(), reason="uppercased")

        global_guard = Guard(name="uppercaser", func=rewrite_fn)
        pipeline.add(global_guard)

        # State-scoped guard that blocks
        scoped_guard = self._make_blocking_guard("scoped", states=["strict"])
        pipeline.add(scoped_guard)

        # In "normal" state: global runs (rewrite), scoped is skipped
        result = pipeline.run("hello", GuardPhase.INPUT, current_state="normal")
        assert not result.blocked
        assert result.text == "HELLO"

        # In "strict" state: global runs (rewrite), then scoped blocks
        result2 = pipeline.run("hello", GuardPhase.INPUT, current_state="strict")
        assert result2.blocked

    def test_guardrail_decorator_with_states(self):
        """The @guardrail decorator supports the states parameter."""
        from talk_box.guardrails import guardrail

        @guardrail(states=["review"])
        def review_guard(text: str) -> GuardResult:
            return GuardResult.blocked("Review mode only")

        assert review_guard.states == ["review"]
        assert review_guard.applies_to_state("review") is True
        assert review_guard.applies_to_state("chat") is False
