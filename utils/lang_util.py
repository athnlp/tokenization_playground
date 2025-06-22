"""
这个detect_language函数通过定义一系列语言字符的Unicode范围，然后使用regex包来检查输入字符串是否包含这些范围内的字符，
从而尝试确定字符串可能使用的语言。函数返回一个列表，包含所有匹配的语言名称；如果没有检测到已定义范围内的字符，则返回['Unknown']。

请注意，由于某些语言（如中文和日文）共享字符集的部分范围，这可能导致某些字符串被识别为多种语言。
此外，Latin范围非常广泛，几乎包括了所有西方语言的基本字母，因此可能需要更细致的逻辑来区分使用拉丁字母的具体语言。


通过检查特定的字母和重音符号来区分一些使用拉丁字母的语言。
然而，需要强调的是，这种方法的准确性受限于所选语言特征的全面性和独特性。
例如，English的检测范围仅限于基本的A-Z字母，这可能导致它与其他使用相同字母集的语言重叠。
此外，有些语言（如法语和西班牙语）在某些情况下可能共享特定的重音符号，这可能导致一个字符串被错误地识别为多种语言。

## common language
English | 简体中文 | 繁體中文 | 한국어 | Español | 日本語 | हिन्दी | Русский | Рortuguês | తెలుగు | Français | Deutsch | Tiếng Việt |
"""

import re
from typing import List

# 由于大部分是'latin'，所以就不统计了。
common_lang = ["Chinese", "Japanese-Kana", "Korean", "Arabic", "number"]

# Unicode range of different language
language_ranges = {
    (
        "Arabic",
        "ar",
    ): r"[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF\uFB50-\uFDFF\uFE70-\uFEFF]",
    # 'CJK'  https://en.wikipedia.org/wiki/CJK_Unified_Ideographs
    ("Chinese", "zh"): r"[\u4e00-\u9fff]",
    ("Japanese", "ja"): r"[\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FFF\u3400-\u4DBF]",
    # https://stackoverflow.com/questions/19899554/unicode-range-for-japanese
    # Kana type refers to Japanese hiragana and katakana characters that represent phonetic sounds in the Japanese language.
    (
        "Japanese-Kana",
        "ja-kana",
    ): r"[\u3040-\u309F\u30A0-\u30FF]",  # Hiragana  & Katakana
    ("Korean", "ko"): r"[\uac00-\ud7a3]",
    # 拉丁字母系列
    # ('Latin', 'la'): r'[\u0000-\u007F\u0080-\u00FF]',
    # ('English', 'en'): r'[A-Za-z]',  # 这可能会与其他使用基本拉丁字母的语言重叠
    # ('French', 'fr'): r'[\u00C0-\u00FF]',
    # ('German', 'de'): r'[\u00C4\u00D6\u00DC\u00E4\u00F6\u00FC\u00DF]',
    # ('Spanish-特有'): r'[\u00C1\u00E1\u00C9\u00E9\u00CD\u00ED\u00D3\u00F3\u00DA\u00FA\u00D1\u00F1\u00FC]',  # 西班牙语特有字符集合
    # 斯拉夫语系列
    # ('Cyrillic', ''): r'[\u0400-\u04FF\u0500-\u052F\u2DE0-\u2DFF\uA640-\uA69F]',
    #
    # 'Greek': r'[\u0370-\u03FF\u1F00-\u1FFF]',   # 希腊字母
    # 'Hebrew': r'[\u0590-\u05FF\uFB1D-\uFB4F]',  # 希伯来语
}


def detect_language_by_unicode(text: str) -> List:
    """
    :param text:
    :return:
    """
    detected_languages = []
    for language, pattern in language_ranges.items():
        if re.search(pattern, text):
            detected_languages.append(language)

    return detected_languages


if __name__ == "__main__":
    # 测试函数
    test_strings = {
        # 拉丁语系
        "Hello, world!": "English/Latin",
        "Hola": "Spanish",
        "Bonjour": "French",
        "Guten Tag": "German",
        "Empieza donde estás. ": "Spanish",
        # CJK
        "你好": "Chinese",
        "こんにちは": "Japanese",
        "안녕하세요": "Korean",
        # 其他
        "Привет": "Russian/Cyrillic",
        "مرحبا": "Arabic",
    }

    for s, expected in test_strings.items():
        # print(f"'{s}' === Detected lang: {detect_language(s)} === Expected: {expected}")
        print(
            f"'{s}'\nDetected lang: {detect_language_by_unicode(s)}\nExpected lang: {expected}"
        )
