"""Deterministic anti-hallucination filter for Whisper ASR transcripts."""

from __future__ import annotations

import re

# Inline subtitle credit patterns often emitted by Whisper on silent/breath pauses
_INLINE_CREDIT_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(
        r"(?:Субтитры\s+(?:сделал|создавал|подготовил|добавил|подогнал|перев[её]л|написал)\s+"
        r"(?:Dima\s*Torzok|DimaTorzok|Дима\s*Т[оа]рз[оа]к|Дима\s*Тромозок|Amara\.org|[\w.-]{2,32})"
        r"|(?:Dima\s*Torzok|DimaTorzok)\s+сделал\s+субтитры)"
        r"[.!?…,:;\s]*",
        re.IGNORECASE | re.UNICODE,
    ),
    re.compile(
        r"(?:Редактор\s+субтитров|Корректор\s+субтитров)\s+[\w.-]+(?:\s+[\w.-]+)?[.!?…,:;\s]*",
        re.IGNORECASE | re.UNICODE,
    ),
    re.compile(
        r"(?:Subtitles?\s+(?:by|made\s+by|prepared\s+by)|Captioned\s+by)\s+[\w.-]+(?:\s+[\w.-]+)?[.!?…,:;\s]*",
        re.IGNORECASE,
    ),
    re.compile(r"\bAmara\.org\b[.!?…,:;\s]*", re.IGNORECASE),
)

_QUOTE_CONTEXT_RE = re.compile(
    r"(?:например|example|вроде|скажем|пишет|цитата|типа|e\.g\.)\s*[:,]?\s*[\"«']?\s*$",
    re.IGNORECASE | re.UNICODE,
)

# Standalone noise / tag tokens
_NOISE_TAG_RE = re.compile(
    r"(?:^|\s)(?:"
    r"\[(?:музыка|смех|аплодисменты|тишина|music|applause|laughter|silence|blank_audio)\]"
    r"|\((?:музыка|смех|кашель|вздох|music|cough|sigh|laughter|applause)\)"
    r"|(?:Cough|Sigh|Laughter|Applause|Music|Blank_audio)\."
    r")(?=\s|$|[,.!?])",
    re.IGNORECASE | re.UNICODE,
)

# Isolated interjection / boilerplate phrases (dropped only when they make up the entire segment/text)
_STANDALONE_FILLER_NOISE_RE = re.compile(
    r"^(?:hmm+|mm-hmm+|uh+|um+|cough|sigh|продолжение\s+следует|"
    r"спасибо\s+за\s+просмотр|подписывайтесь\s+на\s+канал|ставьте\s+лайки|"
    r"thanks?\s+for\s+watching|thank\s+you\s+for\s+watching|"
    r"please\s+subscribe|like\s+and\s+subscribe|to\s+be\s+continued)[.!?…\s]*$",
    re.IGNORECASE | re.UNICODE,
)


def clean_whisper_hallucinations(text: str) -> str:
    """Strip known Whisper subtitle hallucinations and standalone sound artifacts."""
    if not text or not text.strip():
        return ""

    out = text.strip()
    if _STANDALONE_FILLER_NOISE_RE.match(out):
        return ""

    for pat in _INLINE_CREDIT_PATTERNS:
        # Walk matches right-to-left so indices remain valid; preserve explicit user examples ("например, ...")
        matches = list(pat.finditer(out))
        for m in reversed(matches):
            prefix = out[: m.start()]
            if _QUOTE_CONTEXT_RE.search(prefix) or prefix.endswith(('"', "«", "'")):
                continue
            out = (out[: m.start()] + " " + out[m.end() :]).strip()

    out = _NOISE_TAG_RE.sub(" ", out)
    out = re.sub(r"\s+([,.!?;:])", r"\1", out)
    out = re.sub(r"\s{2,}", " ", out).strip()

    if not out or _STANDALONE_FILLER_NOISE_RE.match(out):
        return ""
    return out
