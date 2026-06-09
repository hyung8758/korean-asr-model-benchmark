import re


def normalize_text(text: str) -> str:
    text = text.strip()
    text = re.sub(r"\([^)]*\)|\[[^\]]*\]|<[^>]*>|\{[^}]*\}", " ", text)
    text = text.lower()
    text = re.sub(r"[^0-9a-z가-힣ㄱ-ㅎㅏ-ㅣ\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def char_tokens(text: str) -> list[str]:
    return list(normalize_text(text).replace(" ", ""))


def word_tokens(text: str) -> list[str]:
    normalized = normalize_text(text)
    if not normalized:
        return []
    return normalized.split()
