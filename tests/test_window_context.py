import ctypes
import unittest

from localflow.injector.window_context import (
    ERROR_INSUFFICIENT_BUFFER,
    PROCESS_QUERY_LIMITED_INFORMATION,
    ActiveWindowContext,
    classify_window,
    format_window_context_line,
    get_active_window_context,
)


class _MockUser32:
    def __init__(self, hwnd: int | None, title: str, pid: int) -> None:
        self._hwnd = hwnd
        self._title = title
        self._pid = pid

    def GetForegroundWindow(self):
        return self._hwnd

    def GetWindowTextLengthW(self, hwnd):
        return len(self._title)

    def GetWindowTextW(self, hwnd, buf, max_count):
        text = self._title[: max(0, max_count - 1)]
        buf.value = text
        return len(text)

    def GetWindowThreadProcessId(self, hwnd, pid_ptr):
        pid_ptr[0] = self._pid
        return 1


class _MockKernel32:
    def __init__(
        self,
        image_path: str | None,
        *,
        handle: int | None = 101,
        insufficient_until_size: int = 0,
        raise_on_query: Exception | None = None,
    ) -> None:
        self._image_path = image_path
        self._handle = handle
        self._insufficient_until_size = insufficient_until_size
        self._raise_on_query = raise_on_query
        self.open_calls: list[tuple[int, bool, int]] = []
        self.closed_handles: list[int] = []
        self.query_buffer_sizes: list[int] = []

    def OpenProcess(self, access: int, inherit: bool, pid: int):
        self.open_calls.append((access, inherit, pid))
        return self._handle

    def QueryFullProcessImageNameW(self, handle, flags, buf, size_ptr):
        buf_size = int(size_ptr[0])
        self.query_buffer_sizes.append(buf_size)
        if self._raise_on_query is not None:
            raise self._raise_on_query
        if self._image_path is None:
            ctypes.set_last_error(5)
            return 0
        if self._insufficient_until_size and buf_size < self._insufficient_until_size:
            ctypes.set_last_error(ERROR_INSUFFICIENT_BUFFER)
            return 0
        buf.value = self._image_path
        size_ptr[0] = len(self._image_path)
        return 1

    def CloseHandle(self, handle):
        self.closed_handles.append(handle)
        return 1


class TestWindowContext(unittest.TestCase):
    def test_get_active_window_context_core_apps(self) -> None:
        cases = [
            (
                r"C:\Program Files\Antigravity\Antigravity.exe",
                "app.py - Antigravity",
                "Antigravity",
                "ide_terminal",
            ),
            (
                r"C:\Users\dev\AppData\Local\Programs\Microsoft VS Code\Code.exe",
                "ollama_polisher.py - chat - Visual Studio Code",
                "VS Code",
                "ide_terminal",
            ),
            (
                r"C:\Users\dev\AppData\Roaming\Telegram Desktop\Telegram.exe",
                "Telegram — Избранное",
                "Telegram",
                "messenger",
            ),
            (
                r"C:\Program Files\Google\Chrome\Application\chrome.exe",
                "Issue #42 - Google Chrome",
                "Chrome",
                "general",
            ),
            (
                r"C:\Program Files\WindowsApps\WindowsTerminal.exe",
                "PowerShell 7",
                "Terminal",
                "ide_terminal",
            ),
        ]

        for exe_path, title, expected_app, expected_category in cases:
            with self.subTest(expected_app=expected_app):
                u32 = _MockUser32(hwnd=42, title=title, pid=1234)
                k32 = _MockKernel32(image_path=exe_path)
                ctx = get_active_window_context(user32=u32, kernel32=k32)
                self.assertIsNotNone(ctx)
                assert ctx is not None
                self.assertEqual(ctx.app_name, expected_app)
                self.assertEqual(ctx.category, expected_category)
                self.assertEqual(k32.open_calls, [(PROCESS_QUERY_LIMITED_INFORMATION, False, 1234)])
                self.assertEqual(k32.closed_handles, [101])

                line = format_window_context_line(ctx)
                self.assertIn(expected_app, line)
                self.assertIn(expected_category, line)
                self.assertNotIn("\n", line)
                self.assertNotIn("\r", line)

    def test_chrome_tab_classification_variants(self) -> None:
        self.assertEqual(
            classify_window("chrome.exe", "Gmail - Входящие - Google Chrome").category,
            "mail_docs",
        )
        self.assertEqual(
            classify_window("chrome.exe", "Telegram Web - Google Chrome").category,
            "messenger",
        )
        self.assertEqual(
            classify_window("chrome.exe", "GitHub Codespaces - Google Chrome").category,
            "ide_terminal",
        )
        self.assertEqual(
            classify_window("chrome.exe", "Daily Mail - Google Chrome").category,
            "general",
        )

    def test_mail_docs_apps(self) -> None:
        self.assertEqual(classify_window("OUTLOOK.EXE", "Re: Release").category, "mail_docs")
        self.assertEqual(classify_window("Obsidian.exe", "Notes").category, "mail_docs")
        self.assertEqual(classify_window("WINWORD.EXE", "Spec.docx").category, "mail_docs")

    def test_foreground_none_or_zero_returns_none(self) -> None:
        for hwnd in (None, 0):
            u32 = _MockUser32(hwnd=hwnd, title="Ignored", pid=10)
            k32 = _MockKernel32(image_path=r"C:\app.exe")
            self.assertIsNone(get_active_window_context(user32=u32, kernel32=k32))
            self.assertEqual(k32.open_calls, [])

    def test_pid_zero_and_open_process_denied_fallback_to_title(self) -> None:
        u32_pid0 = _MockUser32(hwnd=1, title="Antigravity Workspace", pid=0)
        k32_pid0 = _MockKernel32(image_path=None)
        ctx = get_active_window_context(user32=u32_pid0, kernel32=k32_pid0)
        self.assertIsNotNone(ctx)
        assert ctx is not None
        self.assertEqual(ctx.app_name, "Antigravity")
        self.assertEqual(ctx.category, "ide_terminal")
        self.assertEqual(k32_pid0.open_calls, [])

        # Both process and title empty -> None
        u32_empty = _MockUser32(hwnd=1, title="   ", pid=0)
        self.assertIsNone(get_active_window_context(user32=u32_empty, kernel32=k32_pid0))

        # OpenProcess returns None -> falls back to title
        u32_denied = _MockUser32(hwnd=2, title="Telegram", pid=555)
        k32_denied = _MockKernel32(image_path=None, handle=None)
        ctx_denied = get_active_window_context(user32=u32_denied, kernel32=k32_denied)
        self.assertIsNotNone(ctx_denied)
        assert ctx_denied is not None
        self.assertEqual(ctx_denied.app_name, "Telegram")
        self.assertEqual(ctx_denied.category, "messenger")
        self.assertEqual(k32_denied.closed_handles, [])

    def test_query_insufficient_buffer_retries_and_closes_handle(self) -> None:
        u32 = _MockUser32(hwnd=5, title="Terminal", pid=77)
        k32 = _MockKernel32(
            image_path=r"C:\Program Files\WindowsTerminal.exe",
            handle=999,
            insufficient_until_size=4096,
        )
        ctx = get_active_window_context(user32=u32, kernel32=k32)
        self.assertIsNotNone(ctx)
        assert ctx is not None
        self.assertEqual(ctx.app_name, "Terminal")
        self.assertEqual(k32.query_buffer_sizes, [1024, 4096])
        self.assertEqual(k32.closed_handles, [999])

    def test_query_failure_and_exception_always_close_handle(self) -> None:
        u32 = _MockUser32(hwnd=5, title="Some Window", pid=77)
        k32_fail = _MockKernel32(image_path=None, handle=303)
        ctx = get_active_window_context(user32=u32, kernel32=k32_fail)
        self.assertIsNotNone(ctx)
        assert ctx is not None
        self.assertEqual(ctx.category, "general")
        self.assertEqual(k32_fail.closed_handles, [303])

        k32_exc = _MockKernel32(
            image_path=None,
            handle=404,
            raise_on_query=RuntimeError("mock failure"),
        )
        with self.assertRaises(RuntimeError):
            get_active_window_context(user32=u32, kernel32=k32_exc)
        self.assertEqual(k32_exc.closed_handles, [404])

    def test_empty_window_title_with_valid_process(self) -> None:
        u32 = _MockUser32(hwnd=9, title="", pid=88)
        k32 = _MockKernel32(image_path=r"C:\Tools\Code.exe")
        ctx = get_active_window_context(user32=u32, kernel32=k32)
        self.assertIsNotNone(ctx)
        assert ctx is not None
        self.assertEqual(ctx.app_name, "VS Code")
        self.assertEqual(ctx.category, "ide_terminal")
        line = format_window_context_line(ctx)
        self.assertNotIn("title=", line)

    def test_format_window_context_line_sanitizes_and_truncates(self) -> None:
        self.assertEqual(format_window_context_line(None), "")

        ctx = ActiveWindowContext(
            process_name="Code.exe\n",
            title="line1\r\nline2\tline3\u2028" + ("x" * 200),
            app_name="VS\nCode",
            category="ide_terminal",
        )
        line = format_window_context_line(ctx)
        self.assertNotIn("\n", line)
        self.assertNotIn("\r", line)
        self.assertNotIn("\u2028", line)
        self.assertIn("Active window: VS Code", line)
        self.assertTrue(line.endswith("..."))

    def test_title_only_fallback_prioritizes_browsers_over_tab_substrings(self) -> None:
        browser_cases = [
            ("Telegram Web - Google Chrome", "Chrome", "messenger"),
            ("Visual Studio Code Docs - Google Chrome", "Chrome", "general"),
            ("Installing Windows Terminal - Google Chrome", "Chrome", "general"),
            ("GitHub Codespaces - Microsoft Edge", "Edge", "ide_terminal"),
            ("Gmail - Mozilla Firefox", "Firefox", "mail_docs"),
            ("Telegram - Yandex Browser", "Yandex Browser", "messenger"),
        ]
        for title, expected_app, expected_cat in browser_cases:
            with self.subTest(title=title):
                ctx = classify_window("", title)
                self.assertEqual(ctx.app_name, expected_app)
                self.assertEqual(ctx.category, expected_cat)

        native_cases = [
            ("app.py - Antigravity", "Antigravity", "ide_terminal"),
            ("test.py - Visual Studio Code", "VS Code", "ide_terminal"),
            ("Windows Terminal", "Terminal", "ide_terminal"),
            ("Windows PowerShell", "Terminal", "ide_terminal"),
            ("Command Prompt", "Terminal", "ide_terminal"),
            ("Telegram", "Telegram", "messenger"),
        ]
        for title, expected_app, expected_cat in native_cases:
            with self.subTest(title=title):
                ctx = classify_window("", title)
                self.assertEqual(ctx.app_name, expected_app)
                self.assertEqual(ctx.category, expected_cat)


if __name__ == "__main__":
    unittest.main()
