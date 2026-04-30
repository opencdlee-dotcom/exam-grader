"""
Language detection for multilingual exam grading.
Uses character-set heuristics (no external API needed).
Priority: Spanish, Mandarin/Chinese, Korean, Arabic.
"""
import re


def detect_language(text: str) -> str:
    """
    Detect primary language of text. Returns ISO 639-1 code.
    Returns 'en' as default for ambiguous/English text.
    """
    if not text or len(text.strip()) < 10:
        return "en"

    # CJK (Chinese, Japanese, Korean) character ranges
    cjk_chars = len(re.findall(r'[\u4e00-\u9fff\u3400-\u4dbf]', text))  # Chinese
    hangul_chars = len(re.findall(r'[\uac00-\ud7af\u1100-\u11ff]', text))  # Korean
    arabic_chars = len(re.findall(r'[\u0600-\u06ff\u0750-\u077f]', text))  # Arabic

    total_chars = max(len(text.replace(' ', '')), 1)

    if cjk_chars / total_chars > 0.15:
        return "zh"
    if hangul_chars / total_chars > 0.15:
        return "ko"
    if arabic_chars / total_chars > 0.15:
        return "ar"

    # Spanish heuristics (common words + special chars)
    spanish_markers = ['el', 'la', 'los', 'las', 'es', 'en', 'de', 'que', 'por', 'con',
                       'un', 'una', 'del', 'al', 'se', 'no', 'más', 'también', 'pero']
    spanish_special = len(re.findall(r'[áéíóúüñ¿¡]', text.lower()))
    words = text.lower().split()
    spanish_word_hits = sum(1 for w in words if w in spanish_markers)

    if spanish_special >= 2 or (spanish_word_hits >= 3 and len(words) >= 10):
        return "es"

    return "en"


def is_non_english(text: str) -> bool:
    return detect_language(text) != "en"
