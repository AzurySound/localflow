import unittest

from localflow.config import AppConfig, apply_performance_profile


class TestPerformanceProfiles(unittest.TestCase):
    def test_light_gpu_picks_compact_model_when_available(self) -> None:
        cfg = AppConfig(ollama_model="qwen2.5:14b")
        updated = apply_performance_profile(cfg, "light_gpu", available_ollama_models=["qwen2.5:14b", "gemma4:e4b", "qwen3.5:2b"])
        self.assertEqual(updated.performance_profile, "light_gpu")
        self.assertEqual(updated.whisper_compute_type, "int8_float16")
        self.assertEqual(updated.whisper_beam_size, 2)
        self.assertEqual(updated.ollama_model, "qwen3.5:2b")
        self.assertEqual(updated.ollama_keep_alive, "2m")
        self.assertEqual(updated.ollama_num_ctx, 1024)
        self.assertIsNone(updated.ollama_num_gpu)

    def test_light_gpu_prefers_gemma4_e4b_over_legacy(self) -> None:
        cfg = AppConfig(ollama_model="qwen2.5:14b")
        updated = apply_performance_profile(cfg, "light_gpu", available_ollama_models=["qwen2.5:14b", "gemma4:e4b"])
        self.assertEqual(updated.ollama_model, "gemma4:e4b")

    def test_light_gpu_keeps_existing_model_if_no_compact_installed(self) -> None:
        cfg = AppConfig(ollama_model="qwen2.5:14b")
        updated = apply_performance_profile(cfg, "light_gpu", available_ollama_models=["qwen2.5:14b"])
        self.assertEqual(updated.ollama_model, "qwen2.5:14b")

    def test_zero_vram_forces_cpu_for_both_asr_and_ollama(self) -> None:
        cfg = AppConfig()
        updated = apply_performance_profile(cfg, "zero_vram", available_ollama_models=["qwen3:4b", "qwen2.5:14b"])
        self.assertEqual(updated.whisper_model, "small")
        self.assertEqual(updated.whisper_device, "cpu")
        self.assertEqual(updated.whisper_compute_type, "int8")
        self.assertEqual(updated.ollama_num_gpu, 0)
        self.assertEqual(updated.ollama_model, "qwen3:4b")

    def test_quality_uses_float16_and_warmup(self) -> None:
        cfg = AppConfig()
        updated = apply_performance_profile(cfg, "quality", available_ollama_models=["qwen2.5:3b", "qwen2.5:14b"])
        self.assertEqual(updated.whisper_compute_type, "float16")
        self.assertEqual(updated.whisper_beam_size, 5)
        self.assertEqual(updated.ollama_model, "qwen2.5:14b")
        self.assertTrue(updated.ollama_warmup_on_start)


if __name__ == "__main__":
    unittest.main()
