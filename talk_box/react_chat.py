def enable_react_support():
    """
    Enable React chat support by patching the ChatBot class.
    Call this function to add React support without auto-initialization.
    """
    try:
        from .react_chat_integration import add_react_chat_support

        return add_react_chat_support()
    except Exception as e:
        print(f"Warning: Could not enable React chat support: {e}")
        return False


# Automatically enable React support when this module is imported
# This is safe because it only patches methods, doesn't start servers
enable_react_support()
