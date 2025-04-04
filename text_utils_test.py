from utils import text_utils

def test_regex_replace1():
    text = "2022.30"
    pattern = "(\\d{4}).*"
    replacement = "\\1"
    expected_output = "2022"
    output = text_utils.regex_replace(text, pattern, replacement)
    assert expected_output==output

    replacement = r"\1"
    output = text_utils.regex_replace(text, pattern, replacement)
    assert expected_output == output