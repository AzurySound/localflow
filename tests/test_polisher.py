import json
import unittest

import httpx

from localflow.config import DEFAULT_LLM_SYSTEM_PROMPT
from localflow.injector.window_context import classify_window
from localflow.llm.ollama_polisher import OllamaPolisher, build_polish_system_prompt
from localflow.vocabulary import parse_vocabulary


class TestOllamaPolisher(unittest.TestCase):
    def test_polish_success_and_payload_format(self) -> None:
        captured_json: dict = {}

        def handler(request: httpx.Request) -> httpx.Response:
            nonlocal captured_json
            captured_json = json.loads(request.content.decode("utf-8"))
            return httpx.Response(
                200,
                json={"message": {"content": "Привет, давай сделаем рефакторинг API."}},
            )

        client = httpx.Client(transport=httpx.MockTransport(handler))
        polisher = OllamaPolisher(model="qwen2.5:7b-instruct", system_prompt="SYS", client=client)

        res = polisher.polish("ну типа привет давай сделаем рефакторинг API")
        self.assertTrue(res.used_llm)
        self.assertIsNone(res.error)
        self.assertEqual(res.text, "Привет, давай сделаем рефакторинг API.")
        self.assertFalse(captured_json["stream"])
        self.assertFalse(captured_json["think"])
        self.assertEqual(captured_json["model"], "qwen2.5:7b-instruct")
        self.assertIn("num_predict", captured_json["options"])
        self.assertGreaterEqual(captured_json["options"]["num_predict"], 128)
        self.assertEqual(captured_json["messages"][0]["content"], "SYS")

    def test_polish_falls_back_to_raw_on_http_error(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(503, text="Service Unavailable")

        client = httpx.Client(transport=httpx.MockTransport(handler))
        polisher = OllamaPolisher(client=client)

        res = polisher.polish("сырой текст диктовки")
        self.assertFalse(res.used_llm)
        self.assertIsNotNone(res.error)
        self.assertEqual(res.text, "сырой текст диктовки")

    def test_list_models(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={"models": [{"name": "qwen2.5:7b-instruct"}, {"name": "qwen3:8b"}]})

        client = httpx.Client(transport=httpx.MockTransport(handler))
        polisher = OllamaPolisher(client=client)
        self.assertEqual(polisher.list_models(), ["qwen2.5:7b-instruct", "qwen3:8b"])

    def test_build_polish_system_prompt_backward_compatible(self) -> None:
        self.assertEqual(
            build_polish_system_prompt(DEFAULT_LLM_SYSTEM_PROMPT),
            DEFAULT_LLM_SYSTEM_PROMPT,
        )
        self.assertEqual(
            build_polish_system_prompt("SYS", window_context="   \n ", vocabulary="  "),
            "SYS",
        )

    def test_build_polish_system_prompt_categories_and_str_context(self) -> None:
        ide_ctx = classify_window("Antigravity.exe", "app.py — Antigravity")
        ide_prompt = build_polish_system_prompt(DEFAULT_LLM_SYSTEM_PROMPT, window_context=ide_ctx)
        self.assertIn("Active window: Antigravity", ide_prompt)
        self.assertIn("camelCase", ide_prompt)
        self.assertIn("snake_case", ide_prompt)
        self.assertIn("filenames", ide_prompt)
        self.assertIn("CLI flags", ide_prompt)
        self.assertIn("commands", ide_prompt)
        self.assertIn("do not translate", ide_prompt)

        msg_ctx = classify_window("Telegram.exe", "Telegram — Команда")
        msg_prompt = build_polish_system_prompt(DEFAULT_LLM_SYSTEM_PROMPT, window_context=msg_ctx)
        self.assertIn("Active window: Telegram", msg_prompt)
        self.assertIn("conversational tone", msg_prompt)

        mail_ctx = classify_window("OUTLOOK.EXE", "Re: План")
        mail_prompt = build_polish_system_prompt(DEFAULT_LLM_SYSTEM_PROMPT, window_context=mail_ctx)
        self.assertIn("Active window: Outlook", mail_prompt)
        self.assertIn("paragraphs", mail_prompt)
        self.assertIn("bullet lists", mail_prompt)

        gen_prompt = build_polish_system_prompt("BASE", window_context="general")
        self.assertIn("neutral", gen_prompt)

        # String with multiline newlines gets sanitized to 1 line and classified
        str_prompt = build_polish_system_prompt("BASE", window_context="Telegram\nChat")
        self.assertIn("Telegram Chat", str_prompt)
        self.assertIn("conversational tone", str_prompt)

    def test_polish_with_window_context_and_vocabulary_replacements(self) -> None:
        captured_json: dict = {}

        def handler(request: httpx.Request) -> httpx.Response:
            nonlocal captured_json
            captured_json = json.loads(request.content.decode("utf-8"))
            return httpx.Response(
                200,
                json={"message": {"content": "Поставь курс в файл app.py."}},
            )

        client = httpx.Client(transport=httpx.MockTransport(handler))
        polisher = OllamaPolisher(
            system_prompt="SYS",
            client=client,
            vocabulary="Antigravity, LocalFlow\nкурс -> курсор",
        )
        ctx = classify_window("Code.exe", "app.py - Visual Studio Code")
        res = polisher.polish("поставь курс в файл app.py", window_context=ctx)

        self.assertTrue(res.used_llm)
        self.assertEqual(res.text, "Поставь курсор в файл app.py.")
        sys_msg = captured_json["messages"][0]["content"]
        self.assertIn("Active window: VS Code", sys_msg)
        self.assertIn("camelCase", sys_msg)
        self.assertIn("Antigravity, LocalFlow", sys_msg)
        self.assertIn('"курс" -> "курсор"', sys_msg)

    def test_polish_applies_vocabulary_replacements_on_http_error_fallback(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(503, text="down")

        client = httpx.Client(transport=httpx.MockTransport(handler))
        vocab = parse_vocabulary("курс -> курсор")
        polisher = OllamaPolisher(client=client, vocabulary=vocab)

        res = polisher.polish("сдвинь курс влево")
        self.assertFalse(res.used_llm)
        self.assertIsNotNone(res.error)
        self.assertEqual(res.text, "сдвинь курсор влево")

    def test_transform_selection_success_and_num_predict_scaling(self) -> None:
        captured_payloads: list[dict] = []

        def handler(request: httpx.Request) -> httpx.Response:
            captured_payloads.append(json.loads(request.content.decode("utf-8")))
            return httpx.Response(200, json={"message": {"content": "Translated text."}})

        client = httpx.Client(transport=httpx.MockTransport(handler))
        polisher = OllamaPolisher(
            system_prompt=DEFAULT_LLM_SYSTEM_PROMPT,
            client=client,
            vocabulary="LocalFlow",
        )
        ctx = classify_window("Antigravity.exe", "README.md - Antigravity")

        # Small selection (< 86 chars -> min floor 256)
        res_small = polisher.transform_selection(
            "  indented_line()",
            "переведи комментарий",
            window_context=ctx,
        )
        self.assertTrue(res_small.used_llm)
        self.assertEqual(res_small.text, "Translated text.")
        payload = captured_payloads[-1]
        self.assertEqual(payload["options"]["num_predict"], 256)
        sys_msg = payload["messages"][0]["content"]
        user_msg = payload["messages"][1]["content"]
        # Must NOT reuse ASR dictation prompt
        self.assertNotIn("сырая расшифровка речи", sys_msg)
        self.assertIn("Active window: Antigravity", sys_msg)
        self.assertIn("camelCase", sys_msg)
        self.assertIn("LocalFlow", sys_msg)
        self.assertIn("<selected_text>\n  indented_line()\n</selected_text>", user_msg)
        self.assertIn("<command>\nпереведи комментарий\n</command>", user_msg)

        # Medium selection (300 chars -> 900)
        polisher.transform_selection("a" * 300, "сделай короче")
        self.assertEqual(captured_payloads[-1]["options"]["num_predict"], 900)

        # Large selection (1000 chars -> capped at 2048)
        polisher.transform_selection("a" * 1000, "оформи списком")
        self.assertEqual(captured_payloads[-1]["options"]["num_predict"], 2048)

    def test_transform_selection_empty_inputs_skip_http(self) -> None:
        called = False

        def handler(request: httpx.Request) -> httpx.Response:
            nonlocal called
            called = True
            return httpx.Response(200, json={"message": {"content": "should not run"}})

        client = httpx.Client(transport=httpx.MockTransport(handler))
        polisher = OllamaPolisher(client=client)

        res_no_sel = polisher.transform_selection("   ", "переведи на английский")
        self.assertFalse(res_no_sel.used_llm)
        self.assertEqual(res_no_sel.text, "")
        self.assertIsNotNone(res_no_sel.error)

        res_no_cmd = polisher.transform_selection("hello world", "   ")
        self.assertFalse(res_no_cmd.used_llm)
        self.assertEqual(res_no_cmd.text, "")
        self.assertIsNotNone(res_no_cmd.error)
        self.assertFalse(called)

    def test_transform_selection_error_paths_never_return_command_or_selection(self) -> None:
        responses = [
            httpx.Response(503, text="Unavailable"),
            httpx.Response(200, text="not-json"),
            httpx.Response(200, json={"message": None}),
            httpx.Response(200, json={"message": {"content": None}}),
            httpx.Response(200, json={"message": {"content": 123}}),
            httpx.Response(200, json={"message": {"content": "   "}}),
        ]

        for idx, resp in enumerate(responses):
            def handler(request: httpx.Request, r=resp) -> httpx.Response:
                return r

            client = httpx.Client(transport=httpx.MockTransport(handler))
            polisher = OllamaPolisher(client=client)
            res = polisher.transform_selection("выделенный текст", "сделай короче")
            with self.subTest(case=idx):
                self.assertFalse(res.used_llm)
                self.assertEqual(res.text, "")
                self.assertTrue(res.error)

        # Timeout exception path
        def timeout_handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ReadTimeout("timed out")

        client = httpx.Client(transport=httpx.MockTransport(timeout_handler))
        polisher = OllamaPolisher(client=client)
        res_timeout = polisher.transform_selection("выделенный текст", "переведи")
        self.assertFalse(res_timeout.used_llm)
        self.assertEqual(res_timeout.text, "")
        self.assertIn("timed out", res_timeout.error or "")

    def test_polish_falls_back_on_null_or_non_string_content(self) -> None:
        for payload in (
            {"message": {"content": None}},
            {"message": {"content": 42}},
            {"message": {"content": "   "}},
        ):
            with self.subTest(payload=payload):
                def handler(request: httpx.Request, body=payload) -> httpx.Response:
                    return httpx.Response(200, json=body)

                client = httpx.Client(transport=httpx.MockTransport(handler))
                polisher = OllamaPolisher(client=client, vocabulary="курс -> курсор")
                res = polisher.polish("сдвинь курс")
                self.assertFalse(res.used_llm)
                self.assertEqual(res.text, "сдвинь курсор")
                self.assertEqual(res.error, "Empty LLM response")

    def test_build_polish_system_prompt_ignores_bare_category_words_in_titles(self) -> None:
        cases = [
            ("Antigravity — general.py", "camelCase"),
            ("Telegram — General Chat", "conversational tone"),
            ("Gmail — General Inbox", "paragraphs and bullet lists"),
            ("Notepad — messenger notes", "neutral, clean, natural style"),
            (
                "Active window: Chrome | title=general discussion | category=messenger",
                "conversational tone",
            ),
        ]
        for ctx_str, expected_fragment in cases:
            with self.subTest(ctx_str=ctx_str):
                prompt = build_polish_system_prompt("BASE", window_context=ctx_str)
                self.assertIn(expected_fragment, prompt)

    def test_transform_selection_preserves_line_one_indentation(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200,
                json={"message": {"content": "\n    def foo():\n        return 42\n"}},
            )

        client = httpx.Client(transport=httpx.MockTransport(handler))
        polisher = OllamaPolisher(client=client)
        res = polisher.transform_selection(
            "    def foo():\n        return 1",
            "верни 42",
        )
        self.assertTrue(res.used_llm)
        self.assertEqual(res.text, "    def foo():\n        return 42")


    def test_polish_rejects_system_prompt_leak_and_hallucinated_expansion(self) -> None:
        leaked = (
            "Active window: Antigravity | category=ide_terminal | process=Antigravity.exe\n\n"
            "Let's check the connection."
        )
        client = httpx.Client(
            transport=httpx.MockTransport(
                lambda req: httpx.Response(200, json={"message": {"content": leaked}})
            )
        )
        polisher = OllamaPolisher(client=client)
        res = polisher.polish("И ещё одна проверка связи.")
        self.assertFalse(res.used_llm)
        self.assertEqual(res.text, "И ещё одна проверка связи.")
        self.assertIn("leaked system prompt", res.error or "")

        # Hallucinated long essay from a 3-word dictation
        essay = "We need to create a new category. " * 12
        client2 = httpx.Client(
            transport=httpx.MockTransport(
                lambda req: httpx.Response(200, json={"message": {"content": essay}})
            )
        )
        polisher2 = OllamaPolisher(client=client2)
        res2 = polisher2.polish("Более сложная проверка связи.")
        self.assertFalse(res2.used_llm)
        self.assertEqual(res2.text, "Более сложная проверка связи.")
        self.assertIn("expanded", res2.error or "")

    def test_polish_wraps_transcript_strips_echoed_tags_and_uses_compact_window_ctx(self) -> None:
        from localflow.styles import STYLE_PROFILES

        captured: dict = {}

        def handler(request: httpx.Request) -> httpx.Response:
            nonlocal captured
            captured = json.loads(request.content.decode("utf-8"))
            return httpx.Response(
                200,
                json={"message": {"content": "<transcript>\nЛибо fix делается.\n</transcript>"}},
            )

        client = httpx.Client(transport=httpx.MockTransport(handler))
        polisher = OllamaPolisher(system_prompt="BASE", client=client)
        ctx = classify_window("Antigravity.exe", "chat — Antigravity")
        res = polisher.polish(
            "ну либо fix делается",
            window_context=ctx,
            style_profile=STYLE_PROFILES["ai_prompt"],
        )
        self.assertTrue(res.used_llm)
        self.assertEqual(res.text, "Либо fix делается.")
        user_msg = captured["messages"][1]["content"]
        self.assertEqual(user_msg, "<transcript>\nну либо fix делается\n</transcript>")
        sys_msg = captured["messages"][0]["content"]
        self.assertIn("Active window: Antigravity", sys_msg)
        self.assertNotIn("category=ide_terminal", sys_msg)
        self.assertNotIn("process=Antigravity.exe", sys_msg)


if __name__ == "__main__":
    unittest.main()
