"""
char_
"""


def detect_lang_from_unicode():
    pass


def is_digit_char(uchar):
    return uchar in "0123456789"


def contains_digit(text):
    return any(is_digit_char(ch) for ch in text)


def get_digit_count(text):
    pass


def is_all_digit(text):
    return all(is_digit_char(char) for char in text)


def get_digit_count(text):
    digit_count = 0
    for char in text:
        if char in "0123456789":
            digit_count += 1
    return digit_count


def has_space(text):
    pass


def is_all_space(text):
    pass


def get_space_count(text):
    space_count = 0
    for char in text:
        if len(char.strip()) == 0:
            space_count += 1
    return space_count
