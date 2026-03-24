import re

def natural_sort(l):
    """
    Sorts a list of strings in 'natural' order (e.g., img_9 before img_10).
    """
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(l, key=alphanum_key)
