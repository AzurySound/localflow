import json
import tempfile
import unittest
from pathlib import Path

from pydantic import ValidationError

from localflow.config import AppConfig, apply_performance_profile, load_config, save_config


class TestConfig(unittest.TestCase):
    def test_save_and_load_roundtrip(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cfg_path = Path(tmp) / "config.json"
            cfg = AppConfig(double_tap_ms=280, ollama_model="qwen3:8b", llm_enabled=False)
            save_config(cfg, cfg_path)

            loaded = load_config(cfg_path)
            self.assertEqual(loaded.double_tap_ms, 280)
            self.assertEqual(loaded.ollama_model, "qwen3:8b")
            self.assertFalse(loaded.llm_enabled)

    def test_defaults_for_new_fields(self) -> None:
        cfg = AppConfig()
        self.assertEqual(cfg.custom_vocabulary, "")
        self.assertTrue(cfg.live_preview_enabled)
        self.assertFalse(cfg.silence_auto_stop_enabled)
        self.assertEqual(cfg.silence_auto_stop_sec, 2.5)

    def test_roundtrip_with_vocabulary_and_new_fields(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cfg_path = Path(tmp) / "config.json"
            vocab_raw = "Antigravity, Wispr Flow\nкурс -> курсор\nвиспр → Wispr Flow"
            cfg = AppConfig(
                custom_vocabulary=vocab_raw,
                live_preview_enabled=False,
                silence_auto_stop_enabled=True,
                silence_auto_stop_sec=4.25,
            )
            save_config(cfg, cfg_path)

            loaded = load_config(cfg_path)
            self.assertEqual(loaded.custom_vocabulary, vocab_raw)
            self.assertFalse(loaded.live_preview_enabled)
            self.assertTrue(loaded.silence_auto_stop_enabled)
            self.assertEqual(loaded.silence_auto_stop_sec, 4.25)

    def test_legacy_config_without_new_fields_loads_defaults(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cfg_path = Path(tmp) / "config.json"
            legacy_payload = {
                "performance_profile": "light_gpu",
                "double_tap_ms": 300,
                "ollama_model": "qwen3.5:2b",
            }
            cfg_path.write_text(json.dumps(legacy_payload), encoding="utf-8")

            loaded = load_config(cfg_path)
            self.assertEqual(loaded.double_tap_ms, 300)
            self.assertEqual(loaded.ollama_model, "qwen3.5:2b")
            self.assertEqual(loaded.custom_vocabulary, "")
            self.assertTrue(loaded.live_preview_enabled)
            self.assertFalse(loaded.silence_auto_stop_enabled)
            self.assertEqual(loaded.silence_auto_stop_sec, 2.5)

    def test_silence_auto_stop_sec_bounds(self) -> None:
        cfg_min = AppConfig(silence_auto_stop_sec=0.5)
        cfg_max = AppConfig(silence_auto_stop_sec=15.0)
        self.assertEqual(cfg_min.silence_auto_stop_sec, 0.5)
        self.assertEqual(cfg_max.silence_auto_stop_sec, 15.0)

        with self.assertRaises(ValidationError):
            AppConfig(silence_auto_stop_sec=0.49)
        with self.assertRaises(ValidationError):
            AppConfig(silence_auto_stop_sec=15.01)

    def test_profiles_preserve_new_config_fields(self) -> None:
        vocab_raw = "Antigravity, LocalFlow\nкурс -> курсор"
        base = AppConfig(
            custom_vocabulary=vocab_raw,
            live_preview_enabled=False,
            silence_auto_stop_enabled=True,
            silence_auto_stop_sec=5.5,
        )
        for profile in ("quality", "light_gpu", "zero_vram", "custom"):
            updated = apply_performance_profile(
                base,
                profile,
                available_ollama_models=["qwen2.5:14b", "qwen3.5:2b"],
            )
            self.assertEqual(updated.custom_vocabulary, vocab_raw)
            self.assertFalse(updated.live_preview_enabled)
            self.assertTrue(updated.silence_auto_stop_enabled)
            self.assertEqual(updated.silence_auto_stop_sec, 5.5)


if __name__ == "__main__":
    unittest.main()
