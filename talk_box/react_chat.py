"""
Automatic React Chat Integration for Talk Box

Import this module to automatically add React chat support to ChatBot.show()
"""

try:
    from .react_chat_integration import add_react_chat_support

    add_react_chat_support()
except ImportError:
    # Module not available, skip integration
    pass
