#!/usr/bin/env python3
"""
Russian TTS normalizer — preprocessing layer for Silero Russian voice output.
"""

import logging
import re

logger = logging.getLogger(__name__)

_num2words = None
_translit_fn = None


def _get_num2words():
    global _num2words
    if _num2words is None:
        try:
            from num2words import num2words as _n2w
            _num2words = _n2w
        except ImportError:
            logger.debug("num2words not installed — numbers won't be converted to words")
            _num2words = None
    return _num2words


def _get_translit():
    global _translit_fn
    if _translit_fn is None:
        try:
            from transliterate import translit as _tl
            _translit_fn = _tl
        except ImportError:
            logger.debug("transliterate not installed — English words in Russian text won't be transliterated")
    return _translit_fn


TECH_REPLACEMENTS = {
    "llama.cpp": "лама си-пи-пи",
    "RTX": "эр-ти-икс", "GTX": "джи-ти-икс",
    "GPU": "джи-пи-ю", "CPU": "си-пи-ю",
    "RAM": "рэм", "VRAM": "ви-рэм",
    "LLM": "эл-эл-эм", "AI": "эй-ай", "API": "эй-пи-ай",
    "TTS": "ти-ти-эс", "STT": "эс-ти-ти", "VAD": "ви-эй-ди",
    "MCP": "эм-си-пи",
    "HTTP": "эйч-ти-ти-пи", "HTTPS": "эйч-ти-ти-пи-эс",
    "SSH": "эс-эс-эйч", "JSON": "джейсон",
    "GGUF": "джи-джи-ю-эф", "GGML": "джи-джи-эм-эл",
    "CUDA": "куда", "MPS": "эм-пи-эс",
    "VLLM": "ви-эл-эл-эм", "FFMPEG": "эфф-мепег",
    "FPS": "эф-пи-эс", "VR": "ви-эр", "ML": "эм-эл",
    "NLP": "эн-эл-пи", "RAG": "рэг", "TDP": "ти-ди-пи",
    "Docker": "докер", "docker": "докер",
    "GitHub": "гитхаб", "github": "гитхаб",
    "OpenAI": "оупен-эй-ай", "openai": "оупен-эй-ай",
    "Whisper": "виспер", "whisper": "виспер",
    "Silero": "силеро", "silero": "силеро",
    "Piper": "пайпер", "piper": "пайпер",
    "Discord": "дискорд", "discord": "дискорд",
    "Telegram": "телеграм", "telegram": "телеграм",
    "Hermes": "гермес", "hermes": "гермес",
    "Qwen": "кью-вен", "qwen": "кью-вен",
    "Gemma": "гемма", "gemma": "гемма",
    "Llama": "лама", "llama": "лама",
    "DeepSeek": "дип-сик", "deepseek": "дип-сик",
    "Mistral": "мистраль", "mistral": "мистраль",
    "Ollama": "олама", "ollama": "олама",
    "Linux": "линукс", "linux": "линукс",
    "Ubuntu": "убунту", "ubuntu": "убунту",
    "Python": "пайтон", "python": "пайтон",
    "PyTorch": "пай-торч", "pytorch": "пай-торч",
    "NumPy": "нум-пай", "numpy": "нум-пай",
    "Coolify": "кул-ифай", "coolify": "кул-ифай",
    "localhost": "локалхост", "repo": "репозиторий",
    "server": "сервер", "branch": "бранч",
    "commit": "коммит", "push": "пуш", "pull": "пул",
    "merge": "мердж", "clone": "клон", "build": "билд",
    "deploy": "деплой", "config": "конфиг", "debug": "дебаг",
    "error": "эррор", "warning": "воунинг",
    "log": "лог", "logs": "логи",
    "path": "пэт", "file": "файл", "folder": "фолдер",
    "process": "процесс", "memory": "мемори", "disk": "диск",
    "port": "порт", "host": "хост", "token": "токен",
    "performance": "перформанс", "inference": "инференс",
    "training": "тренировка", "backup": "бэкап",
    "release": "релиз", "bug": "баг", "mode": "моде",
}

_TECH_REPLACEMENTS_SORTED = sorted(TECH_REPLACEMENTS.items(), key=lambda x: -len(x[0]))
_TECH_MAP = {k.lower(): v for k, v in TECH_REPLACEMENTS.items()}

_tech_words_only = [k for k, _ in _TECH_REPLACEMENTS_SORTED if '.' not in k]
if _tech_words_only:
    _TECH_WORD_PATTERN = re.compile(
        r'\b(' + '|'.join(re.escape(k) for k in _tech_words_only) + r')\b',
        re.IGNORECASE,
    )
else:
    _TECH_WORD_PATTERN = None

_COMPOUND_ITEMS = [
    (re.escape(k), v) for k, v in TECH_REPLACEMENTS.items() if '.' in k
]

GPU_MODELS = {
    "rtx5090": "эр-ти-икс пятьдесят девяносто",
    "rtx5080": "эр-ти-икс пятьдесят восемьдесят",
    "rtx4090": "эр-ти-икс сорок девяносто",
    "rtx4080": "эр-ти-икс сорок восемьдесят",
    "rtx4070": "эр-ти-икс сорок семьдесят",
    "rtx3090": "эр-ти-икс тридцать девяносто",
    "rtx3080": "эр-ти-икс тридцать восемьдесят",
    "rtx3070": "эр-ти-икс тридцать семьдесят",
    "gtx1660": "джи-ти-икс шестнадцать шестьдесят",
    "gtx1080": "джи-ти-икс десять восемьдесят",
    "gtx1070": "джи-ти-икс десять семьдесят",
    "gtx1060": "джи-ти-икс десять шестьдесят",
}

_GPU_PATTERN = re.compile(r'\b((?:RTX|GTX)\s?\d{4})\b', re.IGNORECASE)


def _gpu_model_replacer(match):
    raw = match.group(1)
    key = raw.lower().replace(' ', '')
    return GPU_MODELS.get(key) or raw

_MODEL_PATTERN = re.compile(
    r'\b((?:Qwen|Gemma|Llama|Phi|Mistral|DeepSeek|Yi|Mixtral|Falcon|GPT|Claude)'
    r'(?:[\d.]+(?:-[.\d]+)*)?'
    r'(?:[-_]\d+[Bb])?'
    r'(?:[-_]Q\d+_\w*)?)\b',
    re.IGNORECASE,
)

def _pronounce_letters(text):
    letter_map = {
        'A': 'эй', 'B': 'би', 'C': 'си', 'D': 'ди', 'E': 'и',
        'F': 'эф', 'G': 'джи', 'H': 'эйч', 'I': 'ай', 'J': 'джей',
        'K': 'кэй', 'L': 'эл', 'M': 'эм', 'N': 'эн', 'O': 'оу',
        'P': 'пи', 'Q': 'кью', 'R': 'эр', 'S': 'эс', 'T': 'ти',
        'U': 'ю', 'V': 'ви', 'W': 'дэбл-ю', 'X': 'икс', 'Y': 'уай',
        'Z': 'зэт',
    }
    parts = []
    for part in text.split('_'):
        if part.isalpha():
            parts.append("-".join(letter_map.get(c, c) for c in part.upper()))
        else:
            parts.append(part)
    return "-".join(parts)


def _model_name_replacer(match):
    raw = match.group(1)
    base_match = re.match(r'^([A-Za-z]+)', raw)
    if not base_match:
        return raw
    base = base_match.group(1)
    remainder = raw[len(base):]
    base_lower = base.lower()
    base_ru = _TECH_MAP.get(base_lower) or _transliterate_word(base)
    remainder_clean = remainder.strip()
    quant_placeholders = {}
    temp_text = remainder_clean
    for i, qm in enumerate(re.finditer(r'Q(\d+)_(\w+)', remainder_clean, re.IGNORECASE)):
        ph = f"\x01Q{i}\x01"
        quant_placeholders[ph] = qm.group(0)
        temp_text = temp_text.replace(qm.group(0), ph, 1)
    tokens = re.split(r'-', temp_text)
    parts = []
    for token in tokens:
        if not token:
            continue
        for ph, orig in quant_placeholders.items():
            if ph in token:
                token = token.replace(ph, orig)
        token = token.strip()
        quant_match = re.match(r'^Q(\d+)_(\w+)$', token, re.IGNORECASE)
        if quant_match:
            q_num_ru = _num_to_words(quant_match.group(1))
            suffix_ru = _pronounce_letters(quant_match.group(2))
            parts.append(f"кью {q_num_ru} {suffix_ru}")
            continue
        size_match = re.match(r'^(\d+)[Bb]$', token)
        if size_match:
            num_ru = _num_to_words(size_match.group(1))
            parts.append(f"{num_ru} би")
            continue
        if re.match(r'^[\d.]+$', token):
            version_parts = token.split('.')
            version_ru = " точка ".join(
                (_num_to_words(vp) if vp else '') for vp in version_parts
            )
            parts.append(version_ru)
            continue
        parts.append(_transliterate_word(token))
    result = base_ru
    if parts:
        result += " " + ", ".join(parts)
    return result

_QUANT_PATTERN = re.compile(r'\b(Q\d+_\w+)\b')


def _quant_replacer(match):
    token = match.group(1)
    quant_match = re.match(r'^Q(\d+)_(\w+)$', token, re.IGNORECASE)
    if not quant_match:
        return token
    q_num_ru = _num_to_words(quant_match.group(1))
    suffix_ru = _pronounce_letters(quant_match.group(2))
    return f"кью {q_num_ru} {suffix_ru}"

_UNIT_MAP = {
    'GB': 'гигабайт', 'gb': 'гигабайт',
    'MB': 'мегабайт', 'mb': 'мегабайт',
    'KB': 'килобайт', 'kb': 'килобайт',
    'TB': 'терабайт', 'tb': 'терабайт',
    'K': 'тысяч', 'k': 'тысяч',
    'W': 'ватт', 'w': 'ватт',
    'FPS': 'эф-пи-эс', 'fps': 'эф-пи-эс',
    'MHz': 'мегагерц', 'mhz': 'мегагерц',
    'GHz': 'гигагерц', 'ghz': 'гигагерц',
    '%': 'процентов', 'Hz': 'герц', 'hz': 'герц',
}

_NUMBER_UNIT_PATTERN = re.compile(
    r'\b(\d+(?:\.\d+)?)\s*('
    + '|'.join(re.escape(u) for u in sorted(_UNIT_MAP.keys(), key=lambda x: -len(x)))
    + r')\b'
)


def _number_unit_replacer(match):
    num_str = match.group(1)
    unit = match.group(2)
    unit_ru = _UNIT_MAP.get(unit) or unit
    if '.' in num_str:
        parts = num_str.split('.')
        num_ru = f"{_num_to_words(parts[0])} точка {' '.join(_num_to_words(d) for d in parts[1])}"
    else:
        num_ru = _num_to_words(num_str)
    unit_ru = _decline_unit(unit_ru, num_str)
    return f"{num_ru} {unit_ru}"


def _decline_unit(unit_ru, num_str):
    if '-' in unit_ru or unit_ru in ('тысяч', 'процентов', 'мегагерц', 'гигагерц', 'герц'):
        return unit_ru
    try:
        val = int(num_str.split('.')[0])
    except (ValueError, IndexError):
        return unit_ru
    last_digit = val % 10
    last_two = val % 100
    if 11 <= last_two <= 19:
        return unit_ru + ('ов' if unit_ru.endswith('т') else 'ов')
    if last_digit == 1:
        return unit_ru
    if 2 <= last_digit <= 4:
        return unit_ru + ('а' if unit_ru.endswith('т') else 'а')
    return unit_ru + ('ов' if unit_ru.endswith('т') else 'ов')

_EXT_MAP = {
    '.cpp': 'си-пи-пи', '.js': 'джаваскрипт', '.ts': 'тайпскрипт',
    '.py': 'пайтон', '.json': 'джейсон', '.yaml': 'ямл', '.yml': 'ямл',
    '.toml': 'томл', '.xml': 'экс-эм-эл', '.html': 'эч-ти-мэл',
    '.css': 'си-эс-эс', '.md': 'эм-ди', '.txt': 'тэкст',
    '.wav': 'вэйв', '.mp3': 'эм-пи-три', '.ogg': 'огг',
    '.flac': 'флейк', '.onnx': 'онн-кс', '.gguf': 'джи-джи-ю-эф',
    '.pth': 'пай-торч', '.sh': 'шей', '.bash': 'бэш',
    '.env': 'эн-ви', '.git': 'гит',
}

_EXT_PATTERN = re.compile(
    r'(' + '|'.join(re.escape(e) for e in _EXT_MAP.keys()) + r')(?=\s|$|[^a-zA-Z0-9.])'
)


def _ext_replacer(match):
    ext = match.group(1)
    return f" {_EXT_MAP.get(ext) or ext}"

_BARE_NUMBER_PATTERN = re.compile(r'\b(\d{1,6})\b')


def _bare_number_replacer(match):
    num_str = match.group(1)
    try:
        val = int(num_str)
        if val > 999999 or (1900 <= val <= 2100):
            return num_str
        return _num_to_words(num_str)
    except (ValueError, Exception):
        return num_str

_URL_PATTERN = re.compile(
    r'(?:https?://|ftp://|ssh://|git@)[^\s<>"\')\]]+',
    re.IGNORECASE,
)

_MD_CODEBLOCK = re.compile(r'```[\s\S]*?```')
_MD_INLINE_CODE = re.compile(r'`([^`]+)`')
_MD_BOLD = re.compile(r'\*\*(.+?)\*\*')
_MD_ITALIC = re.compile(r'\*(.+?)\*')
_MD_UNDERLINE = re.compile(r'__(.+?)__')
_MD_LINK = re.compile(r'\[([^\]]+)\]\(([^)]+)\)')
_MD_HEADING = re.compile(r'^#{1,6}\s+', re.MULTILINE)
_MD_IMAGE = re.compile(r'!\[([^\]]*)\]\([^)]+\)')
_MD_STRIKE = re.compile(r'~~(.+?)~~')


def _strip_markdown(text):
    text = _MD_IMAGE.sub(r'\1', text)
    text = _MD_LINK.sub(r'\1', text)
    text = _MD_CODEBLOCK.sub(lambda m: m.group(0).replace('```', '').strip(), text)
    text = _MD_INLINE_CODE.sub(r'\1', text)
    text = _MD_BOLD.sub(r'\1', text)
    text = _MD_ITALIC.sub(r'\1', text)
    text = _MD_UNDERLINE.sub(r'\1', text)
    text = _MD_STRIKE.sub(r'\1', text)
    text = _MD_HEADING.sub('', text)
    return text


def _num_to_words(num_str):
    try:
        n2w = _get_num2words()
        if n2w:
            return n2w(int(num_str), lang='ru')
        return num_str
    except (ValueError, Exception):
        return num_str


def _transliterate_word(word):
    if not word:
        return word
    trans_fn = _get_translit()
    if trans_fn:
        try:
            return trans_fn(word, language_code='ru')
        except Exception:
            return word
    return word


def _transliterate_latin_words(text):
    words = text.split(' ')
    result_words = []
    for word in words:
        if not word:
            result_words.append(word)
            continue
        if any(ord(c) < 32 for c in word):
            result_words.append(word)
            continue
        has_cyrillic = any('\u0400' <= c <= '\u04FF' for c in word)
        has_latin = any(c.isascii() and c.isalpha() for c in word)
        if has_cyrillic:
            result_words.append(word)
        elif has_latin:
            result_words.append(_transliterate_word(word))
        else:
            result_words.append(word)
    return ' '.join(result_words)


def _normalize_whitespace(text):
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\s+([.,;:!?)\]»])', r'\1', text)
    text = re.sub(r'([.,;:!?\[\(«])(\S)', r'\1 \2', text)
    text = re.sub(r'(\.)([.,;])', r'\1', text)
    text = re.sub(r'\s+-\s+', ' - ', text)
    return text.strip()


def normalize_for_russian_tts(
    text,
    *,
    strip_markdown=True,
    convert_numbers=True,
    transliterate_latin=True,
):
    if not text or not text.strip():
        return text

    result = text

    # Step 0: Protect URLs (BEFORE Markdown stripping)
    url_placeholder_map = {}
    placeholder_idx = [0]

    def _url_protector(match):
        url = match.group(0)
        placeholder = f"\x01URL{placeholder_idx[0]}\x01"
        url_placeholder_map[placeholder] = url
        placeholder_idx[0] += 1
        return placeholder

    result = _URL_PATTERN.sub(_url_protector, result)

    # Step 1: Strip Markdown
    if strip_markdown:
        result = _strip_markdown(result)

    # Step 2: Compound terms
    for escaped_key, replacement in _COMPOUND_ITEMS:
        result = re.sub(escaped_key, replacement, result, flags=re.IGNORECASE)

    # Step 3: GPU model names
    result = _GPU_PATTERN.sub(_gpu_model_replacer, result)

    # Step 4: Model name patterns
    result = _MODEL_PATTERN.sub(_model_name_replacer, result)

    # Step 5: Standalone quantization tags
    result = _QUANT_PATTERN.sub(_quant_replacer, result)

    # Step 6: File extensions
    result = _EXT_PATTERN.sub(_ext_replacer, result)

    # Step 7: Dictionary lookup
    if _TECH_WORD_PATTERN:
        def _tech_word_replacer(match):
            word = match.group(0)
            replacement = _TECH_MAP.get(word.lower())
            return replacement if replacement is not None else word
        result = _TECH_WORD_PATTERN.sub(_tech_word_replacer, result)

    # Step 8: Number + unit
    if convert_numbers:
        result = _NUMBER_UNIT_PATTERN.sub(_number_unit_replacer, result)

    # Step 9: Bare numbers
    if convert_numbers:
        result = _BARE_NUMBER_PATTERN.sub(_bare_number_replacer, result)

    # Step 10: Transliterate
    if transliterate_latin:
        result = _transliterate_latin_words(result)

    # Step 11: Normalize whitespace
    result = _normalize_whitespace(result)

    # Step 12: Restore URLs (AFTER whitespace normalization)
    for placeholder, url in url_placeholder_map.items():
        result = result.replace(placeholder, url)

    return result
