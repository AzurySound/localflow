# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec for thin standalone LocalFlow Windows bundle (downloads models & llama.cpp on demand)."""

from pathlib import Path
from PyInstaller.utils.hooks import collect_data_files, collect_dynamic_libs

PROJECT_ROOT = Path(SPECPATH).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
ICON_PATH = PROJECT_ROOT / "assets" / "localflow.ico"
VERSION_FILE = PROJECT_ROOT / "packaging" / "version_info.txt"

datas = []
binaries = []

# Required runtime assets for faster-whisper (silero_vad_v6.onnx) and sounddevice (portaudio DLL)
datas += collect_data_files("faster_whisper", includes=["assets/*"])
datas += collect_data_files("_sounddevice_data")
binaries += collect_dynamic_libs("ctranslate2")
binaries += collect_dynamic_libs("onnxruntime")
binaries += collect_dynamic_libs("av")
binaries += collect_dynamic_libs("numpy")

if ICON_PATH.is_file():
    datas.append((str(ICON_PATH), "assets"))

# Aggressively exclude ~600 MB of unused Qt modules and dev tools to keep installer compact (~115 MB)
excludes = [
    "PySide6.QtWebEngine",
    "PySide6.QtWebEngineCore",
    "PySide6.QtWebEngineWidgets",
    "PySide6.QtWebEngineQuick",
    "PySide6.QtWebChannel",
    "PySide6.QtWebSockets",
    "PySide6.QtPdf",
    "PySide6.QtPdfWidgets",
    "PySide6.QtQuick",
    "PySide6.QtQuickWidgets",
    "PySide6.QtQuickControls2",
    "PySide6.QtQuick3D",
    "PySide6.QtQml",
    "PySide6.Qt3DCore",
    "PySide6.Qt3DRender",
    "PySide6.Qt3DInput",
    "PySide6.Qt3DExtras",
    "PySide6.Qt3DAnimation",
    "PySide6.QtMultimedia",
    "PySide6.QtMultimediaWidgets",
    "PySide6.QtCharts",
    "PySide6.QtGraphs",
    "PySide6.QtGraphsWidgets",
    "PySide6.QtDataVisualization",
    "PySide6.QtDesigner",
    "PySide6.QtHelp",
    "PySide6.QtPositioning",
    "PySide6.QtLocation",
    "PySide6.QtBluetooth",
    "PySide6.QtNfc",
    "PySide6.QtSensors",
    "PySide6.QtSerialPort",
    "PySide6.QtSerialBus",
    "PySide6.QtRemoteObjects",
    "PySide6.QtScxml",
    "PySide6.QtStateMachine",
    "PySide6.QtSpatialAudio",
    "PySide6.QtTextToSpeech",
    "PySide6.QtSql",
    "PySide6.QtTest",
    "PySide6.QtPrintSupport",
    "PySide6.QtOpenGL",
    "PySide6.QtOpenGLWidgets",
    "PySide6.QtSvg",
    "PySide6.QtSvgWidgets",
    "PySide6.QtXml",
    "PySide6.QtConcurrent",
    "PySide6.QtDBus",
    "PySide6.QtHttpServer",
    "tkinter",
    "matplotlib",
    "pytest",
    "pytestqt",
    "unittest",
]

hiddenimports = [
    "localflow",
    "localflow.app",
    "localflow.autostart",
    "localflow.single_instance",
    "localflow.llm.embedded_polisher",
    "localflow.llm.llama_server",
    "localflow.llm.downloader",
    "localflow.llm.model_catalog",
    "faster_whisper",
    "ctranslate2",
    "onnxruntime",
    "sounddevice",
    "pyaudiowpatch",
    "tokenizers",
]

a = Analysis(
    [str(SRC_DIR / "localflow" / "__main__.py")],
    pathex=[str(SRC_DIR)],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=excludes,
    noarchive=False,
    optimize=1,
)

# Filter out any accidentally pulled Qt6WebEngine/Pdf/Qml/Quick/3D/Multimedia DLLs
_DROP_BIN_PREFIXES = (
    "qt6webengine",
    "qt6pdf",
    "qt6quick",
    "qt6qml",
    "qt63d",
    "qt6multimedia",
    "qt6charts",
    "qt6graphs",
    "qt6designer",
    "qt6sql",
    "qt6opengl",
    "qt6printsupport",
    "qt6test",
    "opengl32sw",
    "d3dcompiler",
)
a.binaries = [
    b for b in a.binaries if not Path(b[0]).name.lower().startswith(_DROP_BIN_PREFIXES)
]

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="LocalFlow",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    disable_windowed_traceback=False,
    icon=str(ICON_PATH) if ICON_PATH.is_file() else None,
    version=str(VERSION_FILE) if VERSION_FILE.is_file() else None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    name="LocalFlow",
)
