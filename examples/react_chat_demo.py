#!/usr/bin/env python3
"""
Example: React Chat Interface - Analogous to chat.show("browser")

This demonstrates the seamless React chat interface that works just like
the traditional browser interface but with modern React components.
"""

import talk_box as tb

# Import React chat support (adds "react" mode to show())
import talk_box.react_chat


def main():
    print("🚀 Talk Box React Chat Demo")
    print("=" * 40)

    # Create and configure a chatbot - EXACTLY like traditional approach
    bot = (
        tb.ChatBot()
        .model("gpt-4")
        .preset("technical_advisor")
        .temperature(0.3)
        .persona("You are a friendly Python programming expert")
    )

    print("Bot configured:")
    print(f"  Model: {bot._config.get('model')}")
    print(f"  Preset: {bot._config.get('preset')}")
    print(f"  Temperature: {bot._config.get('temperature')}")
    print()

    print("Traditional approach:")
    print("  bot.show('browser')    # Opens traditional interface")
    print()
    print("New React approach:")
    print("  bot.show('react')      # Opens React interface")
    print()

    choice = input("Launch React interface? (y/n): ").strip().lower()

    if choice in ["y", "yes"]:
        print("\n🚀 Launching React chat interface...")
        # This is the ONLY line needed - exactly analogous to bot.show("browser")
        bot.show("react")

        input("\nPress Enter when you're done chatting to exit...")
    else:
        print("Demo skipped. Run again and choose 'y' to see the React interface!")


if __name__ == "__main__":
    main()
