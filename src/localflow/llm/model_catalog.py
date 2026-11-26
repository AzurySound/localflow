"""Catalog of embedded GGUF LLM models and Whisper presets for standalone LocalFlow."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Literal

from localflow.config import get_app_data_dir
from localflow.i18n import tr

ModelTier = Literal["default", "recommended", "max"]


@dataclass(frozen=True, slots=True)
class EmbeddedModelSpec:
    id: str
    name: str
    filename: str
    url: str
    size_mb: int
    tier: ModelTier
    badge: str
    description: str
    recommended_vram_gb: int


@dataclass(frozen=True, slots=True)
class WhisperPresetSpec:
    name: str
    label: str
    size_mb: int
    tier: ModelTier
    description: str


EMBEDDED_LLM_MODELS: tuple[EmbeddedModelSpec, ...] = (
    EmbeddedModelSpec(
        id="qwen25-1.5b",
        name="Qwen 2.5 1.5B (Q4_K_M)",
        filename="Qwen2.5-1.5B-Instruct-Q4_K_M.gguf",
        url=(
            "https://huggingface.co/bartowski/Qwen2.5-1.5B-Instruct-GGUF/"
            "resolve/main/Qwen2.5-1.5B-Instruct-Q4_K_M.gguf"
        ),
        size_mb=986,
        tier="default",
        badge="Default · ~1.0 GB",
        description="Lightest model for a quick start — runs even on laptops without a GPU.",
        recommended_vram_gb=2,
    ),
    EmbeddedModelSpec(
        id="gemma3-4b",
        name="Gemma 3 4B (Q4_K_M)",
        filename="google_gemma-3-4b-it-Q4_K_M.gguf",
        url=(
            "https://huggingface.co/bartowski/google_gemma-3-4b-it-GGUF/"
            "resolve/main/google_gemma-3-4b-it-Q4_K_M.gguf"
        ),
        size_mb=2490,
        tier="recommended",
        badge="Recommended · ~2.5 GB",
        description="Best balance of speed and multilingual accuracy (4–6 GB VRAM or CPU).",
        recommended_vram_gb=4,
    ),
    EmbeddedModelSpec(
        id="gemma4-e4b",
        name="Gemma 4 E4B (Q4_K_M)",
        filename="gemma-4-E4B-it-Q4_K_M.gguf",
        url=(
            "https://huggingface.co/lmstudio-community/gemma-4-E4B-it-GGUF/"
            "resolve/main/gemma-4-E4B-it-Q4_K_M.gguf"
        ),
        size_mb=5340,
        tier="max",
        badge="Max quality · ~5.3 GB",
        description="Highest quality cleanup, auto-styles, and complex prompts (8+ GB VRAM).",
        recommended_vram_gb=8,
    ),
)

DEFAULT_EMBEDDED_MODEL_ID = "qwen25-1.5b"
RECOMMENDED_EMBEDDED_MODEL_ID = "gemma3-4b"

DIARIZATION_MODEL_FILENAME = "wespeaker_en_voxceleb_resnet34_LM.onnx"
DIARIZATION_MODEL_URL = (
    "https://github.com/k2-fsa/sherpa-onnx/releases/download/"
    "speaker-recongition-models/wespeaker_en_voxceleb_resnet34_LM.onnx"
)
DIARIZATION_MODEL_SIZE_MB = 26


def get_diarization_models_dir(base_dir: Path | None = None) -> Path:
    root = base_dir if base_dir is not None else get_app_data_dir()
    return root / "models" / "diarization"


def find_diarization_model_path(base_dir: Path | None = None) -> Path | None:
    candidate = get_diarization_models_dir(base_dir) / DIARIZATION_MODEL_FILENAME
    if candidate.is_file() and candidate.stat().st_size > 1_000_000:
        return candidate
    return None


def get_model_badge(spec: EmbeddedModelSpec, lang: str | None = None) -> str:
    key = f"catalog.{spec.id}.badge"
    val = tr(key, lang)
    return spec.badge if val == key else val


def get_model_description(spec: EmbeddedModelSpec, lang: str | None = None) -> str:
    key = f"catalog.{spec.id}.desc"
    val = tr(key, lang)
    return spec.description if val == key else val


WHISPER_PRESETS: tuple[WhisperPresetSpec, ...] = (
    WhisperPresetSpec(
        name="small",
        label="small (~460 MB · Default)",
        size_mb=460,
        tier="default",
        description="Compact model for a quick start and CPU usage.",
    ),
    WhisperPresetSpec(
        name="medium",
        label="medium (~1.5 GB · Balanced)",
        size_mb=1500,
        tier="recommended",
        description="High multilingual accuracy.",
    ),
    WhisperPresetSpec(
        name="large-v3-turbo",
        label="large-v3-turbo (~1.6 GB · Recommended)",
        size_mb=1600,
        tier="max",
        description="Best accuracy and speed on GPU (recommended).",
    ),
)


def get_model_spec(model_id: str) -> EmbeddedModelSpec | None:
    for spec in EMBEDDED_LLM_MODELS:
        if spec.id == model_id:
            return spec
    return None


def get_llm_models_dir(base_dir: Path | None = None) -> Path:
    root = base_dir or get_app_data_dir()
    path = root / "models" / "llm"
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_llama_runtime_dir(base_dir: Path | None = None) -> Path:
    root = base_dir or get_app_data_dir()
    path = root / "runtime" / "llama"
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_model_path(
    model_id: str,
    *,
    custom_path: str = "",
    base_dir: Path | None = None,
) -> Path | None:
    if custom_path and custom_path.strip():
        return Path(custom_path.strip())
    spec = get_model_spec(model_id)
    if spec is None:
        return None
    return get_llm_models_dir(base_dir) / spec.filename


def is_model_downloaded(
    model_id: str,
    *,
    custom_path: str = "",
    base_dir: Path | None = None,
) -> bool:
    p = get_model_path(model_id, custom_path=custom_path, base_dir=base_dir)
    return p is not None and p.is_file() and p.stat().st_size > 1_000_000


def delete_model_file(model_id: str, *, base_dir: Path | None = None) -> bool:
    spec = get_model_spec(model_id)
    if spec is None:
        return False
    models_dir = get_llm_models_dir(base_dir)
    removed = False
    for target in (models_dir / spec.filename, models_dir / (spec.filename + ".part")):
        if target.is_file():
            try:
                target.unlink()
                removed = True
            except OSError:
                pass
    return removed


def find_llama_server_binary(base_dir: Path | None = None) -> Path | None:
    """Locate llama-server.exe across user runtime dir, PyInstaller bundle, or dev tree."""
    exe_name = "llama-server.exe" if sys.platform == "win32" else "llama-server"
    candidates: list[Path] = []

    # 1. Downloaded runtime inside %APPDATA%\LocalFlow\runtime\llama\
    candidates.append(get_llama_runtime_dir(base_dir) / exe_name)

    # 2. Bundled inside PyInstaller frozen package
    meipass = getattr(sys, "_MEIPASS", None)
    if meipass:
        candidates.append(Path(meipass) / "runtime" / "llama" / exe_name)
    if getattr(sys, "frozen", False):
        candidates.append(Path(sys.executable).resolve().parent / "runtime" / "llama" / exe_name)

    # 3. Dev workspace fallback
    repo_root = Path(__file__).resolve().parents[3]
    candidates.append(repo_root / "packaging" / "runtime" / "llama" / exe_name)

    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


def is_llama_runtime_installed(base_dir: Path | None = None) -> bool:
    return find_llama_server_binary(base_dir) is not None
