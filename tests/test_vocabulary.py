import threading
import unittest
from types import SimpleNamespace

import numpy as np

from localflow.asr.whisper_engine import WhisperEngine
from localflow.vocabulary import (
    ParsedVocabulary,
    apply_vocabulary_replacements,
    build_whisper_initial_prompt,
    format_vocabulary_for_llm,
    parse_vocabulary,
)


class _FakeModel:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def transcribe(self, audio, **kwargs):
        self.calls.append({"audio_size": int(audio.size), **kwargs})
        prompt = kwargs.get("initial_prompt") or ""
        return ([SimpleNamespace(text=f"ok:{prompt}")], None)


class TestVocabulary(unittest.TestCase):
    def test_parse_empty_and_whitespace(self) -> None:
        self.assertEqual(parse_vocabulary(""), ParsedVocabulary(terms=(), replacements=()))
        self.assertEqual(parse_vocabulary("   \n\t,;  "), ParsedVocabulary(terms=(), replacements=()))

    def test_parse_terms_and_replacements_mixed_separators(self) -> None:
        raw = (
            "Antigravity, Wispr Flow; Ollama\r\n"
            "PySide6\n"
            "курс -> курсор, виспр → Wispr Flow;   \n"
            "antigravity, КУРС -> дубль\n"
            "-> missing_src\n"
            "missing_dst ->\n"
            "→\n"
            "FastAPI -> FastAPI\n"
        )
        parsed = parse_vocabulary(raw)
        self.assertEqual(
            parsed.terms,
            ("Antigravity", "Wispr Flow", "Ollama", "PySide6", "FastAPI"),
        )
        self.assertEqual(
            parsed.replacements,
            (("курс", "курсор"), ("виспр", "Wispr Flow")),
        )

    def test_build_whisper_initial_prompt_empty_and_populated(self) -> None:
        empty_vocab = ParsedVocabulary(terms=(), replacements=())
        self.assertEqual(
            build_whisper_initial_prompt("  Base prompt.  ", empty_vocab),
            "Base prompt.",
        )
        self.assertEqual(build_whisper_initial_prompt("   ", empty_vocab), "")

        vocab = parse_vocabulary("Antigravity, Wispr Flow, курс -> курсор, виспр → Wispr Flow")
        prompt = build_whisper_initial_prompt("Base prompt.", vocab)
        self.assertEqual(
            prompt,
            "Base prompt. Antigravity, Wispr Flow, курсор.",
        )
        self.assertNotIn("Термины", prompt)

        no_base = build_whisper_initial_prompt("", vocab)
        self.assertEqual(no_base, "Antigravity, Wispr Flow, курсор.")

    def test_build_whisper_initial_prompt_respects_soft_char_cap(self) -> None:
        many_terms = ", ".join(f"TermNumber{i:03d}" for i in range(200))
        vocab = parse_vocabulary(many_terms)
        prompt = build_whisper_initial_prompt("Base.", vocab)
        self.assertTrue(prompt.startswith("Base. TermNumber000"))
        self.assertLessEqual(len(prompt), 900)
        self.assertIn("TermNumber000", prompt)
        self.assertNotIn("TermNumber199", prompt)

    def test_format_vocabulary_for_llm(self) -> None:
        self.assertEqual(
            format_vocabulary_for_llm(ParsedVocabulary(terms=(), replacements=())),
            "",
        )

        terms_only = parse_vocabulary("Antigravity, LocalFlow")
        self.assertEqual(
            format_vocabulary_for_llm(terms_only),
            "Предпочитаемые термины и написание: Antigravity, LocalFlow.",
        )

        repl_only = parse_vocabulary("курс -> курсор")
        self.assertEqual(
            format_vocabulary_for_llm(repl_only),
            'Обязательные автозамены: "курс" -> "курсор".',
        )

        combined = parse_vocabulary("Antigravity, курс -> курсор")
        self.assertEqual(
            format_vocabulary_for_llm(combined),
            'Предпочитаемые термины и написание: Antigravity.\n'
            'Обязательные автозамены: "курс" -> "курсор".',
        )

    def test_apply_vocabulary_replacements_boundaries_case_and_single_pass(self) -> None:
        rules = [
            ("курс", "курсор"),
            ("бот", "Bot"),
            ("чат бот", "ChatGPT"),
            ("апи", "API"),
            ("C++", "C++20"),
            (".env", ".env.local"),
            ("а", "б"),
            ("б", "в"),
        ]

        # Whole-word Cyrillic boundary: "конкурс" and "курсор" must stay untouched
        text = "Курс, курс и конкурс, а также КУРС на курсор."
        self.assertEqual(
            apply_vocabulary_replacements(text, [("курс", "курсор")]),
            "Курсор, курсор и конкурс, а также КУРСОР на курсор.",
        )

        # Longest match wins ("чат бот" > "бот") and single-pass prevents "а"->"б"->"в" chain
        self.assertEqual(
            apply_vocabulary_replacements("чат   бот и бот: а, б", rules),
            "ChatGPT и Bot: б, в",
        )

        # Mixed case dst ("API") and punctuation terms ("C++", ".env")
        self.assertEqual(
            apply_vocabulary_replacements("Нужен апи, C++ и .env файл", rules),
            "Нужен API, C++20 и .env.local файл",
        )

    def test_apply_vocabulary_replacements_backslash_safe(self) -> None:
        result = apply_vocabulary_replacements(
            "путь к файлу",
            [("путь", r"C:\new\1\dir")],
        )
        self.assertEqual(result, r"C:\new\1\dir к файлу")


class TestWhisperEngineInitialPrompt(unittest.TestCase):
    def test_set_initial_prompt_and_per_call_override(self) -> None:
        engine = WhisperEngine(initial_prompt="initial-a")
        fake_model = _FakeModel()
        engine._model = fake_model
        audio = np.ones(1600, dtype=np.float32) * 0.2

        res1 = engine.transcribe(audio)
        self.assertEqual(res1, "ok:initial-a")
        self.assertEqual(fake_model.calls[-1]["initial_prompt"], "initial-a")

        engine.set_initial_prompt("updated-b")
        res2 = engine.transcribe(audio)
        self.assertEqual(res2, "ok:updated-b")
        self.assertEqual(fake_model.calls[-1]["initial_prompt"], "updated-b")

        # Per-call override without mutating engine.initial_prompt
        res3 = engine.transcribe(audio, initial_prompt="override-c")
        self.assertEqual(res3, "ok:override-c")
        self.assertEqual(engine.initial_prompt, "updated-b")

        # Empty string override clears initial_prompt (passed as None to faster-whisper)
        res4 = engine.transcribe(audio, initial_prompt="")
        self.assertEqual(res4, "ok:")
        self.assertIsNone(fake_model.calls[-1]["initial_prompt"])
        self.assertEqual(engine.initial_prompt, "updated-b")

    def test_concurrent_set_initial_prompt_and_transcribe_is_thread_safe(self) -> None:
        engine = WhisperEngine(initial_prompt="seed")
        engine._model = _FakeModel()
        audio = np.ones(1600, dtype=np.float32) * 0.2
        errors: list[Exception] = []

        def writer() -> None:
            try:
                for i in range(50):
                    engine.set_initial_prompt(f"prompt-{i}")
            except Exception as exc:  # pragma: no cover
                errors.append(exc)

        def reader() -> None:
            try:
                for _ in range(50):
                    out = engine.transcribe(audio)
                    self.assertTrue(out.startswith("ok:"))
            except Exception as exc:  # pragma: no cover
                errors.append(exc)

        threads = [threading.Thread(target=writer), threading.Thread(target=reader)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        self.assertEqual(errors, [])


if __name__ == "__main__":
    unittest.main()
