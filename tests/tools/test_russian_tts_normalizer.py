"""Tests for the Russian TTS normalizer (tools.russian_tts_normalizer)."""

import sys
import importlib
import os

# Ensure tools/ is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from tools.russian_tts_normalizer import normalize_for_russian_tts


# ---------------------------------------------------------------------------
# GPU model names
# ---------------------------------------------------------------------------

def test_gpu_rtx5090():
    out = normalize_for_russian_tts("Запусти RTX5090")
    assert "эр-ти-икс пятьдесят девяносто" in out


def test_gpu_rtx3090():
    out = normalize_for_russian_tts("RTX 3090 с 24GB")
    assert "эр-ти-икс тридцать девяносто" in out


def test_gpu_rtx4090():
    out = normalize_for_russian_tts("RTX4090")
    assert "эр-ти-икс сорок девяносто" in out


# ---------------------------------------------------------------------------
# Model names with version + size + quant
# ---------------------------------------------------------------------------

def test_model_qwen3_6_27b():
    out = normalize_for_russian_tts("Qwen3.6-27B")
    assert "кью-вен" in out.lower()
    assert "три точка шесть" in out.lower()
    assert "двадцать семь би" in out.lower()


def test_model_qwen_with_quant():
    out = normalize_for_russian_tts("Qwen3.6-27B-Q4_K_M")
    assert "кью-вен" in out.lower()
    assert "три точка шесть" in out.lower()
    assert "двадцать семь би" in out.lower()
    assert "кью четыре" in out.lower()


def test_standalone_quant():
    out = normalize_for_russian_tts("Модель в формате Q4_K_M")
    assert "кью четыре" in out.lower()


# ---------------------------------------------------------------------------
# Technical acronyms
# ---------------------------------------------------------------------------

def test_acronym_openai():
    out = normalize_for_russian_tts("OpenAI API вернул JSON через HTTPS.")
    assert "оупен-эй-ай" in out.lower()
    assert "эй-пи-ай" in out.lower()
    assert "джейсон" in out.lower()
    assert "эйч-ти-ти-пи-эс" in out.lower()


def test_acronym_gguf():
    out = normalize_for_russian_tts("Формат GGUF весит 15GB")
    assert "джи-джи-ю-эф" in out.lower()


# ---------------------------------------------------------------------------
# llama.cpp compound term
# ---------------------------------------------------------------------------

def test_llama_cpp():
    out = normalize_for_russian_tts("перезапусти llama.cpp server")
    assert "лама си-пи-пи" in out.lower()
    assert "сервер" in out.lower()


# ---------------------------------------------------------------------------
# Pure Russian text should stay intact
# ---------------------------------------------------------------------------

def test_pure_russian():
    out = normalize_for_russian_tts("Привет, как дела? Сегодня хорошая погода.")
    assert "Привет" in out
    assert "как дела" in out
    assert "погода" in out


# ---------------------------------------------------------------------------
# URLs should be preserved
# ---------------------------------------------------------------------------

def test_url_preserved():
    out = normalize_for_russian_tts("Скачай с https://github.com/user/repo и запусти.")
    assert "https://github.com/user/repo" in out


def test_localhost():
    out = normalize_for_russian_tts("Запусти на localhost.")
    assert "локалхост" in out.lower()


# ---------------------------------------------------------------------------
# File extensions
# ---------------------------------------------------------------------------

def test_file_extension_py():
    out = normalize_for_russian_tts("Файл config.py")
    assert "пайтон" in out.lower()


def test_file_extension_json():
    out = normalize_for_russian_tts("Файл data.json")
    assert "джейсон" in out.lower()


# ---------------------------------------------------------------------------
# Numbers + units
# ---------------------------------------------------------------------------

def test_number_gb():
    out = normalize_for_russian_tts("24GB VRAM")
    assert "двадцать четыре гигабайт" in out.lower()
    assert "ви-рэм" in out.lower()


def test_number_k():
    out = normalize_for_russian_tts("128k параметров")
    assert "сто двадцать восемь тысяч" in out.lower()


def test_number_fps():
    out = normalize_for_russian_tts("60fps")
    assert "шестьдесят эф-пи-эс" in out.lower()


def test_number_w():
    out = normalize_for_russian_tts("350W")
    assert "триста пятьдесят ватт" in out.lower()


# ---------------------------------------------------------------------------
# Markdown stripping
# ---------------------------------------------------------------------------

def test_markdown_stripped():
    out = normalize_for_russian_tts("**жирный текст** и *курсив*")
    assert "**" not in out
    assert "жирный" in out
    assert "курсив" in out


# ---------------------------------------------------------------------------
# Configurable options
# ---------------------------------------------------------------------------

def test_disable_markdown_stripping():
    out = normalize_for_russian_tts("**жирный**", strip_markdown=False)
    assert "**" in out


def test_disable_number_conversion():
    out = normalize_for_russian_tts("24GB VRAM", convert_numbers=False)
    assert "24" in out


def test_disable_transliteration():
    # "server" is in TECH_REPLACEMENTS so it still becomes "сервер"
    # Test with a word NOT in the dict
    out = normalize_for_russian_tts("hello world", transliterate_latin=False)
    assert "hello" in out.lower() or "world" in out.lower()


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

def test_empty_input():
    assert normalize_for_russian_tts("") == ""
    assert normalize_for_russian_tts(None) is None
    assert normalize_for_russian_tts("   ") == "   "
