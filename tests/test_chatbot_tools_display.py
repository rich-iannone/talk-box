import re
import talk_box as tb


def test_chatbot_tools_html_shows_selected_count():
    bot = tb.ChatBot().tools(["calculate", "validate_email", "current_time"]).model("gpt-4")
    html = bot._repr_html_()
    match = re.search(r"🛠️ Tools \((\d+) enabled\)", html)
    assert match, "Tools section header missing"
    assert match.group(1) == "3"

    # Ensure no unexpected built-in tools leaked into listing
    # (e.g., 'text_stats' should not appear since not requested)
    assert "text_stats" not in html


def test_chatbot_tools_html_all_case():
    bot = tb.ChatBot().tools("all")
    html = bot._repr_html_()
    match = re.search(r"🛠️ Tools \((\d+) enabled\)", html)
    assert match

    # For 'all', we expect at least the 14 built-ins; count should be >=14
    assert int(match.group(1)) >= 14
