"""Win32 active window metadata retrieval and formatting category classification."""

from __future__ import annotations

import ctypes
import ctypes.wintypes as wintypes
import ntpath
import re
from dataclasses import dataclass
from typing import Any, Literal

WindowCategory = Literal["ide_terminal", "messenger", "mail_docs", "ai_chat", "general"]

PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
ERROR_INSUFFICIENT_BUFFER = 122

_CONTROL_CHAR_RE = re.compile(r"[\x00-\x1f\x7f\u0085\u2028\u2029]+")
_MULTI_SPACE_RE = re.compile(r"\s+")

if hasattr(ctypes, "WinDLL"):
    user32: Any = ctypes.WinDLL("user32", use_last_error=True)
    kernel32: Any = ctypes.WinDLL("kernel32", use_last_error=True)

    user32.GetForegroundWindow.argtypes = []
    user32.GetForegroundWindow.restype = wintypes.HWND

    user32.GetWindowTextLengthW.argtypes = [wintypes.HWND]
    user32.GetWindowTextLengthW.restype = ctypes.c_int

    user32.GetWindowTextW.argtypes = [wintypes.HWND, wintypes.LPWSTR, ctypes.c_int]
    user32.GetWindowTextW.restype = ctypes.c_int

    user32.GetWindowThreadProcessId.argtypes = [
        wintypes.HWND,
        ctypes.POINTER(wintypes.DWORD),
    ]
    user32.GetWindowThreadProcessId.restype = wintypes.DWORD

    kernel32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    kernel32.OpenProcess.restype = wintypes.HANDLE

    kernel32.QueryFullProcessImageNameW.argtypes = [
        wintypes.HANDLE,
        wintypes.DWORD,
        wintypes.LPWSTR,
        ctypes.POINTER(wintypes.DWORD),
    ]
    kernel32.QueryFullProcessImageNameW.restype = wintypes.BOOL

    kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
    kernel32.CloseHandle.restype = wintypes.BOOL
else:
    user32 = None
    kernel32 = None


@dataclass(frozen=True, slots=True)
class ActiveWindowContext:
    """Normalized metadata for the active foreground window."""

    process_name: str
    title: str
    app_name: str
    category: WindowCategory
    is_browser: bool = False


_DIRECT_PROCESS_MAP: dict[str, tuple[str, WindowCategory]] = {
    # IDE / Editors / Terminals
    "antigravity.exe": ("Antigravity", "ide_terminal"),
    "code.exe": ("VS Code", "ide_terminal"),
    "code - insiders.exe": ("VS Code", "ide_terminal"),
    "vscodium.exe": ("VSCodium", "ide_terminal"),
    "codium.exe": ("VSCodium", "ide_terminal"),
    "cursor.exe": ("Cursor", "ide_terminal"),
    "windsurf.exe": ("Windsurf", "ide_terminal"),
    "zed.exe": ("Zed", "ide_terminal"),
    "devenv.exe": ("Visual Studio", "ide_terminal"),
    "pycharm64.exe": ("PyCharm", "ide_terminal"),
    "pycharm.exe": ("PyCharm", "ide_terminal"),
    "idea64.exe": ("IntelliJ IDEA", "ide_terminal"),
    "idea.exe": ("IntelliJ IDEA", "ide_terminal"),
    "webstorm64.exe": ("WebStorm", "ide_terminal"),
    "webstorm.exe": ("WebStorm", "ide_terminal"),
    "goland64.exe": ("GoLand", "ide_terminal"),
    "goland.exe": ("GoLand", "ide_terminal"),
    "clion64.exe": ("CLion", "ide_terminal"),
    "clion.exe": ("CLion", "ide_terminal"),
    "rider64.exe": ("Rider", "ide_terminal"),
    "rider.exe": ("Rider", "ide_terminal"),
    "rustrover64.exe": ("RustRover", "ide_terminal"),
    "rustrover.exe": ("RustRover", "ide_terminal"),
    "datagrip64.exe": ("DataGrip", "ide_terminal"),
    "datagrip.exe": ("DataGrip", "ide_terminal"),
    "sublime_text.exe": ("Sublime Text", "ide_terminal"),
    "notepad++.exe": ("Notepad++", "ide_terminal"),
    "windowsterminal.exe": ("Terminal", "ide_terminal"),
    "wt.exe": ("Terminal", "ide_terminal"),
    "openconsole.exe": ("Terminal", "ide_terminal"),
    "alacritty.exe": ("Terminal", "ide_terminal"),
    "wezterm-gui.exe": ("Terminal", "ide_terminal"),
    "wezterm.exe": ("Terminal", "ide_terminal"),
    "tabby.exe": ("Terminal", "ide_terminal"),
    "warp.exe": ("Terminal", "ide_terminal"),
    "hyper.exe": ("Terminal", "ide_terminal"),
    "mintty.exe": ("Terminal", "ide_terminal"),
    "conemu64.exe": ("Terminal", "ide_terminal"),
    "conemu.exe": ("Terminal", "ide_terminal"),
    "powershell.exe": ("PowerShell", "ide_terminal"),
    "pwsh.exe": ("PowerShell", "ide_terminal"),
    "cmd.exe": ("Command Prompt", "ide_terminal"),
    "wsl.exe": ("Terminal", "ide_terminal"),
    "bash.exe": ("Terminal", "ide_terminal"),
    "ubuntu.exe": ("Terminal", "ide_terminal"),
    # Messengers
    "telegram.exe": ("Telegram", "messenger"),
    "ayugram.exe": ("Telegram", "messenger"),
    "unigram.exe": ("Telegram", "messenger"),
    "discord.exe": ("Discord", "messenger"),
    "discordptb.exe": ("Discord", "messenger"),
    "discordcanary.exe": ("Discord", "messenger"),
    "slack.exe": ("Slack", "messenger"),
    "whatsapp.exe": ("WhatsApp", "messenger"),
    "whatsapp.root.exe": ("WhatsApp", "messenger"),
    "teams.exe": ("Microsoft Teams", "messenger"),
    "ms-teams.exe": ("Microsoft Teams", "messenger"),
    "signal.exe": ("Signal", "messenger"),
    "viber.exe": ("Viber", "messenger"),
    "skype.exe": ("Skype", "messenger"),
    "mattermost.exe": ("Mattermost", "messenger"),
    "element.exe": ("Element", "messenger"),
    # Mail / Docs
    "outlook.exe": ("Outlook", "mail_docs"),
    "olk.exe": ("Outlook", "mail_docs"),
    "thunderbird.exe": ("Thunderbird", "mail_docs"),
    "winword.exe": ("Microsoft Word", "mail_docs"),
    "onenote.exe": ("OneNote", "mail_docs"),
    "obsidian.exe": ("Obsidian", "mail_docs"),
    "notion.exe": ("Notion", "mail_docs"),
    "soffice.bin": ("LibreOffice Writer", "mail_docs"),
    "swriter.exe": ("LibreOffice Writer", "mail_docs"),
    # Standalone AI chat apps
    "claude.exe": ("Claude", "ai_chat"),
    "chatgpt.exe": ("ChatGPT", "ai_chat"),
    "perplexity.exe": ("Perplexity", "ai_chat"),
    "lm studio.exe": ("LM Studio", "ai_chat"),
    "jan.exe": ("Jan", "ai_chat"),
    "cherry studio.exe": ("Cherry Studio", "ai_chat"),
    # General built-ins
    "notepad.exe": ("Notepad", "general"),
    "explorer.exe": ("Explorer", "general"),
}

_BROWSER_PROCESS_MAP: dict[str, str] = {
    "chrome.exe": "Chrome",
    "msedge.exe": "Edge",
    "firefox.exe": "Firefox",
    "brave.exe": "Brave",
    "opera.exe": "Opera",
    "vivaldi.exe": "Vivaldi",
    "arc.exe": "Arc",
    "browser.exe": "Yandex Browser",
}

_BROWSER_TITLE_MARKERS: tuple[tuple[str, str], ...] = (
    ("google chrome", "Chrome"),
    ("microsoft edge", "Edge"),
    ("mozilla firefox", "Firefox"),
    ("yandex browser", "Yandex Browser"),
)

_BROWSER_AI_KEYWORDS = (
    "chatgpt",
    "claude",
    "gemini",
    "perplexity",
    "deepseek",
    "grok",
    "google ai studio",
    "ai studio",
    "copilot",
    "notebooklm",
    "qwen chat",
    "le chat",
    "poe.com",
    "mistral",
)

_BROWSER_IDE_KEYWORDS = (
    "github codespaces",
    "codespaces",
    "github.dev",
    "vscode.dev",
    "gitpod",
    "stackblitz",
    "codesandbox",
)

_BROWSER_MESSENGER_KEYWORDS = (
    "telegram",
    "slack",
    "discord",
    "whatsapp",
    "microsoft teams",
    "vk мессенджер",
    "вконтакте",
)

_BROWSER_MAIL_DOCS_KEYWORDS = (
    "gmail",
    "google docs",
    "google документы",
    "google sheets",
    "google таблицы",
    "excel",
    "outlook",
    "яндекс почта",
    "yandex mail",
    "mail.ru",
    "protonmail",
    "proton mail",
    "notion",
    "confluence",
    "google keep",
)


def _sanitize_single_line(text: str) -> str:
    if not text:
        return ""
    cleaned = _CONTROL_CHAR_RE.sub(" ", text)
    return _MULTI_SPACE_RE.sub(" ", cleaned).strip()


def _classify_browser_title(title_low: str) -> WindowCategory:
    if any(kw in title_low for kw in _BROWSER_AI_KEYWORDS):
        return "ai_chat"
    if any(kw in title_low for kw in _BROWSER_IDE_KEYWORDS):
        return "ide_terminal"
    if any(kw in title_low for kw in _BROWSER_MESSENGER_KEYWORDS):
        return "messenger"
    if any(kw in title_low for kw in _BROWSER_MAIL_DOCS_KEYWORDS):
        return "mail_docs"
    return "general"


def classify_window(process_name: str, title: str) -> ActiveWindowContext:
    """Classify an active window by executable name and window title."""
    proc = _sanitize_single_line(ntpath.basename((process_name or "").strip()))
    proc_low = proc.lower()
    clean_title = _sanitize_single_line(title)
    title_low = clean_title.lower()

    if proc_low in _DIRECT_PROCESS_MAP:
        app_name, category = _DIRECT_PROCESS_MAP[proc_low]
        return ActiveWindowContext(
            process_name=proc,
            title=clean_title,
            app_name=app_name,
            category=category,
        )

    if proc_low in _BROWSER_PROCESS_MAP:
        app_name = _BROWSER_PROCESS_MAP[proc_low]
        category = _classify_browser_title(title_low)
        return ActiveWindowContext(
            process_name=proc,
            title=clean_title,
            app_name=app_name,
            category=category,
            is_browser=True,
        )

    if not proc_low:
        for marker, browser_name in _BROWSER_TITLE_MARKERS:
            if marker in title_low:
                return ActiveWindowContext(
                    "",
                    clean_title,
                    browser_name,
                    _classify_browser_title(title_low),
                    is_browser=True,
                )
        if "antigravity" in title_low:
            return ActiveWindowContext("", clean_title, "Antigravity", "ide_terminal")
        if "visual studio code" in title_low or "vs code" in title_low:
            return ActiveWindowContext("", clean_title, "VS Code", "ide_terminal")
        if (
            "windows terminal" in title_low
            or "powershell" in title_low
            or "command prompt" in title_low
        ):
            return ActiveWindowContext("", clean_title, "Terminal", "ide_terminal")
        if "telegram" in title_low:
            return ActiveWindowContext("", clean_title, "Telegram", "messenger")
        cat = _classify_browser_title(title_low)
        return ActiveWindowContext("", clean_title, "Unknown", cat)

    stem = proc[:-4] if proc_low.endswith(".exe") else proc
    if not stem:
        app_name = "Unknown"
    elif stem.islower():
        app_name = stem.capitalize()
    else:
        app_name = stem

    return ActiveWindowContext(
        process_name=proc,
        title=clean_title,
        app_name=app_name,
        category="general",
    )


def format_window_context_line(
    ctx: ActiveWindowContext | None,
    *,
    max_title_length: int = 120,
) -> str:
    """Format active window context as a single sanitized line without CR/LF."""
    if ctx is None:
        return ""

    clean_app = _sanitize_single_line(ctx.app_name) or "Unknown"
    clean_cat = _sanitize_single_line(ctx.category) or "general"
    clean_proc = _sanitize_single_line(ctx.process_name)
    clean_title = _sanitize_single_line(ctx.title)

    limit = max(4, max_title_length)
    if len(clean_title) > limit:
        clean_title = clean_title[: limit - 3].rstrip() + "..."

    parts = [
        f"Active window: {clean_app}",
        f"category={clean_cat}",
    ]
    if clean_proc:
        parts.append(f"process={clean_proc}")
    if clean_title:
        parts.append(f"title={clean_title}")
    return " | ".join(parts)


def get_active_window_context(
    user32: Any = None,
    kernel32: Any = None,
) -> ActiveWindowContext | None:
    """Inspect the Win32 foreground window and return classified metadata, or None."""
    u32 = user32 if user32 is not None else globals().get("user32")
    k32 = kernel32 if kernel32 is not None else globals().get("kernel32")
    if u32 is None or k32 is None:
        return None

    hwnd = u32.GetForegroundWindow()
    if not hwnd:
        return None

    title = ""
    length = u32.GetWindowTextLengthW(hwnd)
    if length and length > 0:
        buf = ctypes.create_unicode_buffer(int(length) + 1)
        copied = u32.GetWindowTextW(hwnd, buf, int(length) + 1)
        if copied and copied > 0:
            title = buf.value[: int(copied)] if int(copied) <= int(length) else buf.value

    process_name = ""
    pid = wintypes.DWORD(0)
    u32.GetWindowThreadProcessId(hwnd, ctypes.pointer(pid))
    pid_val = int(pid.value)

    if pid_val > 0:
        handle = k32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, pid_val)
        if handle:
            try:
                for buf_chars in (1024, 4096, 32768):
                    ctypes.set_last_error(0)
                    pbuf = ctypes.create_unicode_buffer(buf_chars)
                    size = wintypes.DWORD(buf_chars)
                    ok = k32.QueryFullProcessImageNameW(
                        handle,
                        0,
                        pbuf,
                        ctypes.pointer(size),
                    )
                    if ok:
                        used = int(size.value)
                        raw_path = pbuf.value[:used] if 0 < used <= buf_chars else pbuf.value
                        process_name = ntpath.basename(raw_path)
                        break
                    if ctypes.get_last_error() != ERROR_INSUFFICIENT_BUFFER:
                        break
            finally:
                k32.CloseHandle(handle)

    if not process_name and not title.strip():
        return None

    return classify_window(process_name, title)
