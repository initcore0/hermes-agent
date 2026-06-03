"""Tests for Silero TTS provider (local, PyTorch-based, multi-language).

Strategy: patch ``_import_silero`` and inject mock ``silero`` / ``torch`` into
sys.modules so the real packages are never touched.
"""

import json
import wave
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Helpers -- build a fully self-contained mock that never touches real packages
# ---------------------------------------------------------------------------

def _build_mocks():
    """Return (mock_silero_mod, mock_torch, mock_silero_tts_fn, mock_model).

    The mock silero module has a ``silero_tts`` callable that returns
    (mock_model, example_text).  The mock model's ``apply_tts`` returns a
    MagicMock that behaves like a torch tensor.
    """
    mock_model = MagicMock()
    mock_tensor = MagicMock()
    mock_tensor.cpu.return_value.numpy.return_value = np.zeros(
        4800, dtype=np.float32
    )
    mock_model.apply_tts.return_value = mock_tensor

    mock_silero_tts_fn = MagicMock(return_value=(mock_model, "Пример текста"))

    mock_silero_mod = MagicMock()
    mock_silero_mod.silero_tts = mock_silero_tts_fn

    mock_torch = MagicMock()
    mock_no_grad_ctx = MagicMock()
    mock_no_grad_ctx.__enter__ = MagicMock(return_value=None)
    mock_no_grad_ctx.__exit__ = MagicMock(return_value=False)
    mock_torch.no_grad = MagicMock(return_value=mock_no_grad_ctx)

    return mock_silero_mod, mock_torch, mock_silero_tts_fn, mock_model


# ---------------------------------------------------------------------------
# _check_silero_available
# ---------------------------------------------------------------------------


class TestCheckSileroAvailable:

    def test_returns_true_when_silero_installed(self):
        with patch("importlib.util.find_spec", return_value=MagicMock()):
            from tools.tts_tool import _check_silero_available
            assert _check_silero_available() is True

    def test_returns_false_when_silero_missing(self):
        with patch("importlib.util.find_spec", return_value=None):
            from tools.tts_tool import _check_silero_available
            assert _check_silero_available() is False

    def test_returns_false_on_exception(self):
        with patch("importlib.util.find_spec", side_effect=Exception("boom")):
            from tools.tts_tool import _check_silero_available
            assert _check_silero_available() is False


# ---------------------------------------------------------------------------
# _generate_silero_tts
# ---------------------------------------------------------------------------


class TestGenerateSileroTts:

    @pytest.fixture(autouse=True)
    def _clear_cache(self):
        """Clear the model cache before each test."""
        from tools import tts_tool
        tts_tool._silero_model_cache.clear()

    # -- individual tests --

    def test_default_config_loads_model(self, tmp_path):
        """Default config uses v5_ru language and kseniya voice."""
        mock_silero_mod, mock_torch, mock_silero_tts_fn, mock_model = _build_mocks()

        output = str(tmp_path / "out.mp3")
        with patch("tools.tts_tool._import_silero", return_value=mock_silero_mod), \
             patch.dict("sys.modules", {"silero": mock_silero_mod, "torch": mock_torch}), \
             patch("shutil.which", return_value=None):
            from tools.tts_tool import _generate_silero_tts
            _generate_silero_tts("Hello world", output, {})

        mock_silero_tts_fn.assert_called_once_with(
            language="ru", speaker="v5_ru"
        )
        mock_model.apply_tts.assert_called_once()
        call_kwargs = mock_model.apply_tts.call_args[1]
        assert call_kwargs["speaker"] == "kseniya"
        assert call_kwargs["sample_rate"] == 48000

    def test_custom_config_propagated(self, tmp_path):
        """User-provided silero config overrides defaults."""
        mock_silero_mod, mock_torch, mock_silero_tts_fn, mock_model = _build_mocks()
        cfg = {
            "silero": {
                "language": "en",
                "model": "v3_en",
                "voice": "en_0",
                "sample_rate": 24000,
            }
        }

        output = str(tmp_path / "out.mp3")
        with patch("tools.tts_tool._import_silero", return_value=mock_silero_mod), \
             patch.dict("sys.modules", {"silero": mock_silero_mod, "torch": mock_torch}), \
             patch("shutil.which", return_value=None):
            from tools.tts_tool import _generate_silero_tts
            _generate_silero_tts("Hello world", output, cfg)

        mock_silero_tts_fn.assert_called_once_with(
            language="en", speaker="v3_en"
        )
        call_kwargs = mock_model.apply_tts.call_args[1]
        assert call_kwargs["speaker"] == "en_0"
        assert call_kwargs["sample_rate"] == 24000

    def test_writes_wav_file(self, tmp_path):
        """Output WAV file is written with correct format."""
        mock_silero_mod, mock_torch, mock_silero_tts_fn, mock_model = _build_mocks()
        output = str(tmp_path / "out.wav")

        with patch("tools.tts_tool._import_silero", return_value=mock_silero_mod), \
             patch.dict("sys.modules", {"silero": mock_silero_mod, "torch": mock_torch}), \
             patch("shutil.which", return_value=None):
            from tools.tts_tool import _generate_silero_tts
            _generate_silero_tts("test", output, {})

        assert Path(output).exists()
        with wave.open(output, "rb") as wf:
            assert wf.getnchannels() == 1
            assert wf.getsampwidth() == 2  # 16-bit
            assert wf.getframerate() == 48000

    def test_float32_scaled_to_int16(self, tmp_path):
        """Audio tensor values in [-1,1] are correctly scaled to int16 range.

        This is the critical regression test: without multiplying by 32767,
        direct astype(np.int16) truncates fractional values to zero, producing
        silent audio.
        """
        mock_silero_mod, mock_torch, mock_silero_tts_fn, mock_model = _build_mocks()
        # tiny positive value -- would become 0 without proper scaling
        small_audio = np.full(4800, 0.01, dtype=np.float32)
        mock_model.apply_tts.return_value.cpu.return_value.numpy.return_value = small_audio

        output = str(tmp_path / "out.wav")
        with patch("tools.tts_tool._import_silero", return_value=mock_silero_mod), \
             patch.dict("sys.modules", {"silero": mock_silero_mod, "torch": mock_torch}), \
             patch("shutil.which", return_value=None):
            from tools.tts_tool import _generate_silero_tts
            _generate_silero_tts("test", output, {})

        with wave.open(output, "rb") as wf:
            frames = wf.readframes(wf.getnframes())
        audio_int16 = np.frombuffer(frames, dtype=np.int16)
        # 0.01 * 32767 = 327.67 -> clipped to 327
        assert audio_int16[0] == 327
        # Not all zeros -- the critical regression check
        assert audio_int16.max() > 0

    def test_model_caching_skips_reload(self, tmp_path):
        """Second call with same cache key does not reload the model."""
        mock_silero_mod, mock_torch, mock_silero_tts_fn, mock_model = _build_mocks()

        output = str(tmp_path / "out1.mp3")
        with patch("tools.tts_tool._import_silero", return_value=mock_silero_mod), \
             patch.dict("sys.modules", {"silero": mock_silero_mod, "torch": mock_torch}), \
             patch("shutil.which", return_value=None):
            from tools.tts_tool import _generate_silero_tts
            _generate_silero_tts("First", output, {})

        first_load_count = mock_silero_tts_fn.call_count

        # Second call with same config -- should hit cache
        output2 = str(tmp_path / "out2.mp3")
        with patch("tools.tts_tool._import_silero", return_value=mock_silero_mod), \
             patch.dict("sys.modules", {"silero": mock_silero_mod, "torch": mock_torch}), \
             patch("shutil.which", return_value=None):
            from tools.tts_tool import _generate_silero_tts
            _generate_silero_tts("Second", output2, {})

        # silero_tts should still only have been called once (cache hit)
        assert mock_silero_tts_fn.call_count == first_load_count

    def test_output_mp3_without_ffmpeg(self, tmp_path):
        """When output ends in .mp3 and ffmpeg is absent, WAV is renamed to .mp3."""
        mock_silero_mod, mock_torch, mock_silero_tts_fn, mock_model = _build_mocks()
        output = str(tmp_path / "out.mp3")

        with patch("tools.tts_tool._import_silero", return_value=mock_silero_mod), \
             patch.dict("sys.modules", {"silero": mock_silero_mod, "torch": mock_torch}), \
             patch("shutil.which", return_value=None):
            from tools.tts_tool import _generate_silero_tts
            result = _generate_silero_tts("Hello", output, {})

        # Without ffmpeg, WAV is renamed to the .mp3 path
        assert Path(result).exists()
        assert result == output

    def test_torch_no_grad_used(self, tmp_path):
        """Synthesis runs inside torch.no_grad() context."""
        mock_silero_mod, mock_torch, mock_silero_tts_fn, mock_model = _build_mocks()
        output = str(tmp_path / "out.wav")

        with patch("tools.tts_tool._import_silero", return_value=mock_silero_mod), \
             patch.dict("sys.modules", {"silero": mock_silero_mod, "torch": mock_torch}), \
             patch("shutil.which", return_value=None):
            from tools.tts_tool import _generate_silero_tts
            _generate_silero_tts("test", output, {})

        mock_torch.no_grad.assert_called_once()
        mock_torch.no_grad.return_value.__enter__.assert_called_once()
        mock_torch.no_grad.return_value.__exit__.assert_called_once()


# ---------------------------------------------------------------------------
# text_to_speech_tool dispatch
# ---------------------------------------------------------------------------


class TestSileroDispatch:

    def test_dispatch_calls_silero_when_available(self, tmp_path):
        """Provider='silero' dispatches to _generate_silero_tts."""
        mock_silero_mod, mock_torch, mock_silero_tts_fn, mock_model = _build_mocks()

        with patch("tools.tts_tool._import_silero", return_value=mock_silero_mod), \
             patch.dict("sys.modules", {"silero": mock_silero_mod, "torch": mock_torch}), \
             patch("shutil.which", return_value=None), \
             patch("tools.tts_tool._load_tts_config",
                   return_value={"provider": "silero"}):
            from tools.tts_tool import text_to_speech_tool
            result = text_to_speech_tool(
                text="Hello",
                output_path=str(tmp_path / "out.mp3"),
            )

        data = json.loads(result)
        assert data["success"] is True

    def test_dispatch_returns_error_when_import_fails(self, tmp_path):
        """Provider='silero' returns error JSON when package is missing."""
        with patch("tools.tts_tool._import_silero",
                   side_effect=ImportError("no module")), \
             patch("tools.tts_tool._load_tts_config",
                   return_value={"provider": "silero"}):
            from tools.tts_tool import text_to_speech_tool
            result = text_to_speech_tool(
                text="Hello",
                output_path=str(tmp_path / "out.mp3"),
            )

        data = json.loads(result)
        assert data["success"] is False
        assert "silero" in data["error"].lower()
        assert "not installed" in data["error"].lower()


# ---------------------------------------------------------------------------
# PROVIDER_MAX_TEXT_LENGTH & BUILTIN_TTS_PROVIDERS
# ---------------------------------------------------------------------------


class TestSileroConstants:

    def test_silero_in_builtin_providers(self):
        from tools.tts_tool import BUILTIN_TTS_PROVIDERS
        assert "silero" in BUILTIN_TTS_PROVIDERS

    def test_silero_max_text_length(self):
        from tools.tts_tool import PROVIDER_MAX_TEXT_LENGTH
        assert PROVIDER_MAX_TEXT_LENGTH.get("silero") == 5000

    def test_silero_defaults(self):
        from tools.tts_tool import (
            DEFAULT_SILERO_LANGUAGE,
            DEFAULT_SILERO_MODEL,
            DEFAULT_SILERO_VOICE,
            DEFAULT_SILERO_SAMPLE_RATE,
        )
        assert DEFAULT_SILERO_LANGUAGE == "ru"
        assert DEFAULT_SILERO_MODEL == "v5_ru"
        assert DEFAULT_SILERO_VOICE == "kseniya"
        assert DEFAULT_SILERO_SAMPLE_RATE == 48000


# ---------------------------------------------------------------------------
# check_tts_requirements
# ---------------------------------------------------------------------------


class TestCheckTtsRequirements:

    def test_returns_true_when_silero_available(self):
        with patch("tools.tts_tool._check_silero_available", return_value=True):
            from tools.tts_tool import check_tts_requirements
            assert check_tts_requirements() is True

    def test_returns_false_when_no_provider(self):
        with patch("tools.tts_tool._check_silero_available", return_value=False):
            from tools.tts_tool import check_tts_requirements
            with patch("tools.tts_tool._import_edge_tts",
                       side_effect=ImportError("no")), \
                 patch("tools.tts_tool._import_kittentts",
                       side_effect=ImportError("no")), \
                 patch("tools.tts_tool._import_piper",
                       side_effect=ImportError("no")):
                assert check_tts_requirements() is False


# ---------------------------------------------------------------------------
# lazy_deps registration
# ---------------------------------------------------------------------------


class TestSileroLazyDeps:

    def test_silero_in_lazy_deps(self):
        from tools.lazy_deps import LAZY_DEPS
        assert "tts.silero" in LAZY_DEPS
        deps = LAZY_DEPS["tts.silero"]
        assert "silero==0.5.5" in deps
        assert any("scipy" in d for d in deps)
