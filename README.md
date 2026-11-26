# LocalFlow

<p align="center">
  <strong>Local, privacy-first, open-source Wispr Flow alternative for Windows.</strong><br>
  Offline voice dictation, smart LLM text cleanup &amp; local meeting Notetaker. 100% private, zero cloud dependencies.
</p>

<p align="center">
  <a href="https://github.com/AzurySound/localflow/releases/latest">
    <img src="https://img.shields.io/badge/Download-LocalFlow%20Setup%20(v0.3.0)-2563eb?style=for-the-badge&logo=windows&logoColor=white" alt="Download">
  </a>
</p>

<p align="center">
  <img src="docs/screenshots/flow_bar_recording.gif" alt="LocalFlow Flow Bar" width="260" style="border-radius: 10px;">
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Platform-Windows%2010%20%2F%2011-blue?logo=windows" alt="Platform">
  <img src="https://img.shields.io/badge/License-MIT-green" alt="MIT License">
  <img src="https://img.shields.io/badge/Engine-faster--whisper%20%2B%20llama.cpp-violet" alt="Engine">
  <img src="https://img.shields.io/badge/Tests-252%20passing-brightgreen" alt="Tests">
</p>

---

## How It Works

| Hotkey | Action |
|---|---|
| `2× CapsLock` | **Start dictating** anywhere in Windows. |
| `CapsLock` | **Finish & Paste**: transcribes, cleans up speech, and pastes into the active field. |
| `Ctrl + 2× CapsLock` | **Command on Selection**: highlight text and tell LocalFlow what to do (*"summarize"*, *"translate"*, *"make a list"*). |
| `Esc` | **Cancel** without pasting. |

<p align="center">
  <img src="docs/screenshots/flow_bar_showcase.png" alt="LocalFlow Flow Bar States" width="680">
</p>

---

## Key Features

- **100% Local & Private**: Audio and text never leave your computer. No accounts, API keys, or subscriptions.
- **Smart Speech Cleanup**: Removes filler words (*"um"*, *"uh"*, *"you know"*, *"ну то есть"*), fixes stutters, glues awkward pause breaks, and resolves mid-sentence corrections (*"let's add a button, actually let's finish for today"* &rarr; keeps only the final thought).
- **IT & Technical Terms**: Preserves English programming terms, CLI flags, camelCase, and snake_case in Latin script without weird transliteration.
- **App-Aware Styles**: Automatically formats text to match what you are doing:
  - 🤖 **AI Chats & IDEs** (Antigravity, Cursor, ChatGPT): crisp prompts condensed into clean plain-text lists without bold noise.
  - 💬 **Messengers** (Telegram, Slack, Discord): natural conversational tone.
  - 📄 **Docs & Email** (Word, Notion, Outlook): clean paragraphs and bullet points.
  - ⌨ **Raw Mode**: direct transcript without LLM edits.
- **Standalone Engine**: Includes an embedded `llama.cpp` runtime with Vulkan GPU acceleration & 1-click model downloads. External **Ollama** is also supported.
- **Bilingual & Themed**: English and Russian UI with Dark, Light, and System themes.

---

## Notetaker (Local Meeting Notes)

Record and summarize online or in-person calls locally — no bots joining your call, and no calendar connection required.

- **Dual-Stream Capture (`Mic + System Audio`)**: Records your microphone together with Windows system audio via WASAPI loopback, with echo-bleed suppression.
- **Speaker Diarization**: Distinguishes `You` from remote participants (`Speaker 1`, `Speaker 2`, …) using ONNX speaker embeddings, post-meeting cluster refinement, and click-to-rename chips.
- **Meeting Side Window**: Live streaming transcript bubbles, transcript search, manual *My thoughts* notes, **Pause / Resume**, and parallel voice dictation while a meeting is active.
- **AI Summary & Q&A**: Generates structured summaries (**TL;DR**, **Key Points**, **Action Items**) in your configured UI language, plus an interactive *Ask anything about this meeting* Q&A bar.

<p align="center">
  <img src="docs/screenshots/notetaker_transcript.png" alt="LocalFlow Notetaker Live Transcript" width="380">&nbsp;&nbsp;
  <img src="docs/screenshots/notetaker_summary.png" alt="LocalFlow Notetaker Summary & Action Items" width="380">
</p>

<p align="center">
  <img src="docs/screenshots/notetaker_hub.png" alt="LocalFlow Notetaker Hub" width="780">
</p>

---

## Screenshots

<p align="center">
  <img src="docs/screenshots/settings_window.png" alt="LocalFlow Settings" width="740">
</p>

<p align="center">
  <img src="docs/screenshots/insights_window.png" alt="Usage Analytics" width="740">
</p>

---

## Quick Install

1. Download **[LocalFlow-Setup-0.3.0.exe](https://github.com/AzurySound/localflow/releases/latest)** *(or the zero-install `LocalFlow-0.3.0-portable.zip`)*.
2. Run the installer (installs per-user, no admin rights required).
3. On first launch, download the recommended lightweight models in 1 click:
   - **Speech Recognition**: Whisper `small` (~460 MB) or `large-v3-turbo` (~1.6 GB).
   - **Smart Cleanup**: Embedded `Qwen 2.5 1.5B` (~1.0 GB) or `Gemma 3 4B` (~2.5 GB).

*(Optional: switch to Ollama in Settings if you already run models locally).*

---

## Architecture

<p align="center">
  <img src="docs/screenshots/architecture_diagram.png" alt="Architecture Diagram" width="800">
</p>

---

<details>
<summary><strong>For Developers: Building & Running from Source</strong></summary>

### Prerequisites
- Windows 10 / 11 (x64)
- Python 3.12
- [uv](https://github.com/astral-sh/uv) (recommended)

### Run Locally
```powershell
git clone https://github.com/AzurySound/localflow.git
cd localflow
uv sync --python 3.12
uv run localflow
```

### Run Tests
```powershell
uv run pytest -q
# 252 passed, 41 subtests passed
```

### Build Installer
```powershell
# Requires Inno Setup 6 (winget install JRSoftware.InnoSetup)
.\packaging\build_release.ps1
```
Output binaries will be placed in `release\`.

</details>

---

## License

MIT License — see [LICENSE](LICENSE) for details.
