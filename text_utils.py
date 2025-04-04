import re

from fuzzywuzzy import fuzz
import random, string


def identify_similar_values(text, values_to_compare, threshold=70):
    """
    Arguments:
        text:str
            text to compare against
        values_to_compare:list of str
            list of string to which we have to compare the given text against
        threshold:float
            threshold (max=100) above which items should be identified as similar
    """
    return [v for v in values_to_compare if fuzz.ratio(text,v)>threshold]


def regex_replace(text:str, pattern, replacement, flags=re.I):
    """
    Given a text & a regex pattern replace it
    Arguments:
        text:str
            text on which replacement needs to be performed
        pattern:str or re.pattern
            regex pattern to search for
        replacement:str
            replacement string
    """
    return re.sub(pattern, replacement, text, flags=flags)

def generate_random_unique_id(length=5):
    """
    Returns a unique random string
    """
    return ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(length))