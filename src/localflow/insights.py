"""Pure local usage insights analytics computed from TranscriptionEntry history."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from difflib import SequenceMatcher
import re

from localflow.i18n import tr
from localflow.storage import TranscriptionEntry

_WORD_RE = re.compile(r"[^\W_]+", re.UNICODE)
_BASELINE_TYPING_WPM = 40.0


@dataclass(frozen=True, slots=True)
class DailyActivity:
    day: date
    dictations: int
    words: int


@dataclass(frozen=True, slots=True)
class CategoryStat:
    key: str
    label: str
    count: int
    words: int
    percent: int


@dataclass(frozen=True, slots=True)
class NamedCount:
    name: str
    count: int
    words: int
    percent: int


@dataclass(frozen=True, slots=True)
class UsageInsights:
    wpm: int
    top_percent: int
    total_fixes: int
    words_corrected: int
    dictionary_fixes: int
    total_words: int
    total_dictations: int
    total_duration_sec: float
    time_saved_min: float
    unique_apps_count: int
    categories: list[CategoryStat]
    current_streak_days: int
    longest_streak_days: int
    daily_activity: dict[date, DailyActivity]
    # Secondary ("Your voice") metrics
    avg_words_per_dictation: float
    avg_duration_sec: float
    longest_dictation_words: int
    llm_usage_pct: int
    top_apps: list[NamedCount]
    model_usage: list[NamedCount]


def tokenize_words(text: str) -> list[str]:
    return _WORD_RE.findall((text or "").lower())


def count_word_fixes(raw_text: str, final_text: str) -> int:
    """Count word-level insertions, deletions, and replacements between raw ASR and final text."""
    raw_words = tokenize_words(raw_text)
    final_words = tokenize_words(final_text)
    if not raw_words and not final_words:
        return 0
    if raw_words == final_words:
        return 0

    matcher = SequenceMatcher(None, raw_words, final_words, autojunk=False)
    fixes = 0
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            continue
        fixes += max(i2 - i1, j2 - j1)
    return fixes


def _estimate_top_percent(wpm: int) -> int:
    """Map dictation WPM onto a friendly Top-X% percentile gauge (matching Wispr Flow)."""
    if wpm <= 0:
        return 100
    if wpm >= 140:
        return 5
    if wpm >= 110:
        return 10
    if wpm >= 85:
        return 20
    if wpm >= 65:
        return 30
    if wpm >= 50:
        return 40
    if wpm >= 35:
        return 55
    return 75


def _parse_local_date(iso_ts: str) -> date | None:
    try:
        return datetime.fromisoformat(iso_ts).astimezone().date()
    except Exception:
        try:
            return date.fromisoformat(iso_ts[:10])
        except Exception:
            return None


def _compute_streaks(active_days: set[date], today: date) -> tuple[int, int]:
    if not active_days:
        return 0, 0

    sorted_days = sorted(active_days)
    longest = 1
    run = 1
    for i in range(1, len(sorted_days)):
        if (sorted_days[i] - sorted_days[i - 1]).days == 1:
            run += 1
            if run > longest:
                longest = run
        else:
            run = 1

    # Current streak counts if user dictated today or yesterday
    anchor = today if today in active_days else (today - timedelta(days=1) if (today - timedelta(days=1)) in active_days else None)
    if anchor is None:
        return 0, longest

    current = 0
    cursor = anchor
    while cursor in active_days:
        current += 1
        cursor -= timedelta(days=1)
    return current, longest


def compute_usage_insights(
    entries: list[TranscriptionEntry],
    *,
    today: date | None = None,
    lang: str | None = None,
) -> UsageInsights:
    """Compute all Usage Insights metrics from historical TranscriptionEntry items."""
    ref_today = today or datetime.now().astimezone().date()

    total_words = 0
    total_duration = 0.0
    total_llm_entries = 0
    words_corrected = 0
    dictionary_fixes = 0
    longest_words = 0

    daily_counts: dict[date, list[int]] = {}  # day -> [dictations, words]
    cat_counts: Counter[str] = Counter()
    cat_words: Counter[str] = Counter()
    app_counts: Counter[str] = Counter()
    app_words: Counter[str] = Counter()
    model_counts: Counter[str] = Counter()
    model_words: Counter[str] = Counter()

    for entry in entries:
        wc = len(tokenize_words(entry.final_text))
        dur = max(0.0, float(entry.duration_sec))
        total_words += wc
        total_duration += dur
        if wc > longest_words:
            longest_words = wc

        if entry.used_llm:
            total_llm_entries += 1

        if entry.is_command:
            bucket = "command"
            # In command mode, final_text is the transformed selection rather than a raw correction
            d_fix = max(0, int(entry.dict_fixes))
            dictionary_fixes += d_fix
        else:
            raw_cat = (entry.app_category or "general").strip().lower()
            cat_alias = {
                "ide_terminal": "ide",
                "ide": "ide",
                "ai_chat": "ide",
                "messenger": "messenger",
                "mail_docs": "docs",
                "docs": "docs",
                "general": "general",
            }
            bucket = cat_alias.get(raw_cat, "general")
            total_diff = count_word_fixes(entry.raw_text, entry.final_text)
            d_fix = min(max(0, int(entry.dict_fixes)), total_diff) if total_diff > 0 else max(0, int(entry.dict_fixes))
            dictionary_fixes += d_fix
            words_corrected += max(0, total_diff - d_fix)

        cat_counts[bucket] += 1
        cat_words[bucket] += wc

        app_name = (entry.app_name or "").strip()
        if app_name:
            app_counts[app_name] += 1
            app_words[app_name] += wc

        model_key = (entry.llm_model or "").strip() if entry.used_llm else "RAW ASR"
        if not model_key:
            model_key = "Ollama LLM"
        model_counts[model_key] += 1
        model_words[model_key] += wc

        day = _parse_local_date(entry.created_at)
        if day is not None:
            pair = daily_counts.setdefault(day, [0, 0])
            pair[0] += 1
            pair[1] += wc

    n = len(entries)
    wpm = int(round((total_words / total_duration) * 60.0)) if total_duration >= 1.0 and total_words > 0 else 0
    top_pct = _estimate_top_percent(wpm)

    typing_minutes = total_words / _BASELINE_TYPING_WPM if total_words > 0 else 0.0
    dictating_minutes = total_duration / 60.0
    time_saved_min = max(0.0, typing_minutes - dictating_minutes)

    # Category breakdown matching Wispr Flow Desktop Usage card
    cat_specs = [
        ("messenger", tr("insights.cat.messenger", lang)),
        ("ide", tr("insights.cat.ide", lang)),
        ("docs", tr("insights.cat.docs", lang)),
        ("command", tr("insights.cat.command", lang)),
        ("general", tr("insights.cat.general", lang)),
    ]
    denom_words = total_words if total_words > 0 else 0
    categories: list[CategoryStat] = []
    for key, label in cat_specs:
        c = cat_counts.get(key, 0)
        w = cat_words.get(key, 0)
        if denom_words > 0:
            pct = int(round((w / denom_words) * 100.0))
        elif n > 0:
            pct = int(round((c / n) * 100.0))
        else:
            pct = 0
        categories.append(CategoryStat(key=key, label=label, count=c, words=w, percent=pct))

    # Sort active categories to top while keeping zero ones in logical order
    categories.sort(key=lambda item: (-item.percent, -item.count))

    daily_activity = {
        d: DailyActivity(day=d, dictations=vals[0], words=vals[1])
        for d, vals in daily_counts.items()
    }
    current_streak, longest_streak = _compute_streaks(set(daily_activity.keys()), ref_today)

    top_apps = [
        NamedCount(
            name=name,
            count=cnt,
            words=app_words[name],
            percent=int(round((cnt / n) * 100.0)) if n > 0 else 0,
        )
        for name, cnt in app_counts.most_common(6)
    ]
    model_usage = [
        NamedCount(
            name=name,
            count=cnt,
            words=model_words[name],
            percent=int(round((cnt / n) * 100.0)) if n > 0 else 0,
        )
        for name, cnt in model_counts.most_common(5)
    ]

    return UsageInsights(
        wpm=wpm,
        top_percent=top_pct,
        total_fixes=words_corrected + dictionary_fixes,
        words_corrected=words_corrected,
        dictionary_fixes=dictionary_fixes,
        total_words=total_words,
        total_dictations=n,
        total_duration_sec=total_duration,
        time_saved_min=time_saved_min,
        unique_apps_count=len(app_counts) if app_counts else sum(1 for c in categories if c.count > 0),
        categories=categories,
        current_streak_days=current_streak,
        longest_streak_days=longest_streak,
        daily_activity=daily_activity,
        avg_words_per_dictation=(total_words / n) if n > 0 else 0.0,
        avg_duration_sec=(total_duration / n) if n > 0 else 0.0,
        longest_dictation_words=longest_words,
        llm_usage_pct=int(round((total_llm_entries / n) * 100.0)) if n > 0 else 0,
        top_apps=top_apps,
        model_usage=model_usage,
    )
