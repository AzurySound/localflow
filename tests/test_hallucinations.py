from __future__ import annotations

from dataclasses import dataclass

from localflow.asr.hallucinations import clean_whisper_hallucinations
from localflow.asr.whisper_engine import WhisperEngine


@dataclass
class _FakeSeg:
    text: str
    no_speech_prob: float = 0.0
    avg_logprob: float = -0.2
    start: float = 0.0
    end: float = 1.0


def test_clean_whisper_hallucinations_removes_subtitles_and_noise() -> None:
    # Inline Russian credit hallucination inside dictation (#424 case)
    raw_inline = "Продолжаем слушать дальше. Субтитры сделал DimaTorzok Если глава не готова, пауза."
    cleaned = clean_whisper_hallucinations(raw_inline)
    assert "DimaTorzok" not in cleaned
    assert "Субтитры" not in cleaned
    assert "Продолжаем слушать дальше." in cleaned
    assert "Если глава не готова, пауза." in cleaned

    # Standalone artifacts vanish completely
    assert clean_whisper_hallucinations("Субтитры создавал DimaTorzok") == ""
    assert clean_whisper_hallucinations("Субтитры подготовил Дима Торзок.") == ""
    assert clean_whisper_hallucinations("Продолжение следует...") == ""
    assert clean_whisper_hallucinations("Subtitles by Amara.org") == ""
    assert clean_whisper_hallucinations("Cough.") == ""
    assert clean_whisper_hallucinations("Hmm.") == ""
    assert clean_whisper_hallucinations("Thank you for watching!") == ""


def test_clean_whisper_hallucinations_preserves_explicit_user_examples() -> None:
    quoted = 'Например, "субтитры подготовил DimaTorzok" это артефакт Whisper'
    assert clean_whisper_hallucinations(quoted) == quoted


def test_whisper_engine_filter_segment_text_drops_low_conf_non_speech() -> None:
    noisy = _FakeSeg(text="Субтитры сделал DimaTorzok", no_speech_prob=0.82, avg_logprob=-1.2)
    assert WhisperEngine._filter_segment_text(noisy) == ""

    low_prob_cough = _FakeSeg(text="неразборчивый шум", no_speech_prob=0.75, avg_logprob=-1.1)
    assert WhisperEngine._filter_segment_text(low_prob_cough) == ""

    valid = _FakeSeg(text="Привет, давай проверим систему.", no_speech_prob=0.1, avg_logprob=-0.2)
    assert WhisperEngine._filter_segment_text(valid) == "Привет, давай проверим систему."
