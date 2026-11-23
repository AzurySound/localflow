import unittest

import numpy as np

from localflow.audio.recorder import trim_edge_silence
from localflow.audio.sound_fx import get_sound_buffer


class TestAudioAndSoundFx(unittest.TestCase):
    def test_trim_edge_silence_all_silence_returns_empty(self) -> None:
        audio = np.zeros(16000, dtype=np.float32)
        trimmed = trim_edge_silence(audio)
        self.assertEqual(trimmed.size, 0)

    def test_trim_edge_silence_preserves_speech_with_padding(self) -> None:
        audio = np.zeros(16000, dtype=np.float32)
        audio[8000:8800] = 0.25  # 50 ms speech burst in the middle
        trimmed = trim_edge_silence(audio, sample_rate=16000, pad_ms=100)
        # 100ms before + 800 samples + 100ms after = 1600 + 800 + 1600 = 4000
        self.assertEqual(len(trimmed), 4000)
        self.assertGreater(float(np.max(np.abs(trimmed))), 0.2)

    def test_sound_cues_are_valid_wav_headers(self) -> None:
        for kind in ("start", "stop", "cancel"):
            buf = get_sound_buffer(kind)
            self.assertIsNotNone(buf)
            assert buf is not None
            self.assertTrue(buf.startswith(b"RIFF"))
            self.assertIn(b"WAVE", buf[:16])


if __name__ == "__main__":
    unittest.main()
