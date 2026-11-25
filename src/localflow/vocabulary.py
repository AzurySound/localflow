"""Personal vocabulary parsing, prompt biasing, and deterministic replacements."""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass

_ENTRY_SPLIT_RE = re.compile(r"[\n\r,;]+")
_ARROW_SPLIT_RE = re.compile(r"\s*(?:->|→)\s*")
_MAX_WHISPER_VOCAB_CHARS = 800


@dataclass(frozen=True, slots=True)
class ParsedVocabulary:
    """Structured personal vocabulary of preferred terms and src -> dst replacements."""

    terms: tuple[str, ...]
    replacements: tuple[tuple[str, str], ...]


def _normalize_spaces(text: str) -> str:
    return " ".join(text.split())


def parse_vocabulary(raw: str) -> ParsedVocabulary:
    """Parse comma/newline/semicolon-separated terms and `src -> dst` replacement rules."""
    if not raw or not raw.strip():
        return ParsedVocabulary(terms=(), replacements=())

    terms: list[str] = []
    seen_terms: set[str] = set()
    replacements: list[tuple[str, str]] = []
    seen_sources: set[str] = set()

    def _add_term(term: str) -> None:
        key = term.casefold()
        if key not in seen_terms:
            seen_terms.add(key)
            terms.append(term)

    for raw_entry in _ENTRY_SPLIT_RE.split(raw):
        entry = _normalize_spaces(raw_entry)
        if not entry:
            continue

        if "->" in entry or "→" in entry:
            parts = _ARROW_SPLIT_RE.split(entry, maxsplit=1)
            if len(parts) != 2:
                continue
            src = _normalize_spaces(parts[0])
            dst = _normalize_spaces(parts[1])
            if not src or not dst:
                continue
            if src == dst:
                _add_term(dst)
                continue
            src_key = src.casefold()
            if src_key not in seen_sources:
                seen_sources.add(src_key)
                replacements.append((src, dst))
        else:
            _add_term(entry)

    return ParsedVocabulary(terms=tuple(terms), replacements=tuple(replacements))


def build_whisper_initial_prompt(base_prompt: str, vocab: ParsedVocabulary) -> str:
    """Merge base Whisper initial_prompt with personal terms and replacement targets."""
    base = base_prompt.strip()
    seen: set[str] = set()
    candidates: list[str] = []

    for item in vocab.terms:
        norm = _normalize_spaces(item).rstrip("*")
        if not norm:
            continue
        key = norm.casefold()
        if key not in seen:
            seen.add(key)
            candidates.append(norm)

    for _src, dst in vocab.replacements:
        norm = _normalize_spaces(dst).rstrip("*")
        if not norm:
            continue
        key = norm.casefold()
        if key not in seen:
            seen.add(key)
            candidates.append(norm)

    if not candidates:
        return base

    kept: list[str] = []
    current_len = 0
    for item in candidates:
        added_len = len(item) if not kept else len(item) + 2  # ", "
        if current_len + added_len > _MAX_WHISPER_VOCAB_CHARS:
            break
        kept.append(item)
        current_len += added_len

    if not kept:
        return base

    terms_block = f"{', '.join(kept)}."
    if base:
        return f"{base} {terms_block}"
    return terms_block


def format_vocabulary_for_llm(
    vocab: ParsedVocabulary,
    *,
    lang: str | None = None,
) -> str:
    """Format personal terms and replacement rules concisely for LLM system prompt."""
    is_en = (lang or "").strip().lower().startswith("en")
    lines: list[str] = []
    if vocab.terms:
        prefix = "Preferred terms & spelling" if is_en else "Предпочитаемые термины и написание"
        lines.append(f"{prefix}: {', '.join(vocab.terms)}.")
    if vocab.replacements:
        rules = "; ".join(f'"{src}" -> "{dst}"' for src, dst in vocab.replacements)
        prefix = "Mandatory replacements" if is_en else "Обязательные автозамены"
        lines.append(f"{prefix}: {rules}.")
    return "\n".join(lines)


def _apply_case_rule(matched: str, dst: str) -> str:
    if any(ch.isupper() for ch in dst):
        return dst
    if len(matched) > 1 and matched.isupper() and any(ch.isalpha() for ch in matched):
        return dst.upper()
    if matched and matched[0].isupper():
        return dst[0].upper() + dst[1:]
    return dst


def apply_vocabulary_replacements_with_count(
    text: str,
    replacements: Sequence[tuple[str, str]],
) -> tuple[str, int]:
    """Apply whole-word (or prefix-wildcard `*`) case-insensitive replacements and return `(new_text, count)`."""
    if not text or not replacements:
        return (text, 0)

    deduped: list[tuple[str, str]] = []
    seen_sources: set[str] = set()
    for raw_src, raw_dst in replacements:
        src = _normalize_spaces(raw_src)
        dst = _normalize_spaces(raw_dst)
        if not src or not dst:
            continue
        key = src.casefold()
        if key in seen_sources:
            continue
        seen_sources.add(key)
        deduped.append((src, dst))

    if not deduped:
        return (text, 0)

    deduped.sort(key=lambda pair: len(pair[0]), reverse=True)
    exact_lookup: dict[str, str] = {}
    prefix_rules: list[tuple[str, str]] = []
    patterns: list[str] = []

    for src, dst in deduped:
        if src.endswith("*") and len(src) > 1:
            stem = src[:-1]
            stem_pat = r"\s+".join(re.escape(part) for part in stem.split()) + r"\w*"
            patterns.append(stem_pat)
            prefix_rules.append((stem.casefold(), dst))
        else:
            patterns.append(r"\s+".join(re.escape(part) for part in src.split()))
            exact_lookup[src.casefold()] = dst

    combined = re.compile(r"(?<!\w)(?:" + "|".join(patterns) + r")(?!\w)", re.IGNORECASE)
    fix_count = 0

    def _replace(match: re.Match[str]) -> str:
        nonlocal fix_count
        matched = match.group(0)
        norm = _normalize_spaces(matched)
        key = norm.casefold()

        dst = exact_lookup.get(key)
        if dst is not None:
            fix_count += 1
            return _apply_case_rule(matched, dst)

        for stem_key, raw_dst in prefix_rules:
            if key.startswith(stem_key):
                suffix = norm[len(stem_key) :]
                target = (raw_dst[:-1] + suffix) if raw_dst.endswith("*") else raw_dst
                fix_count += 1
                return _apply_case_rule(matched, target)

        return matched

    result = combined.sub(_replace, text)
    return (result, fix_count)


def apply_vocabulary_replacements(
    text: str,
    replacements: Sequence[tuple[str, str]],
) -> str:
    """Apply whole-word case-insensitive replacements in a single longest-match pass."""
    result, _count = apply_vocabulary_replacements_with_count(text, replacements)
    return result
