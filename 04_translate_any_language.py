#!/usr/bin/env python3
"""
Phase 4: Translate Any Language

This script is designed to be called directly or by:

08_run_full_dubbing_pipeline.py

Supported engines:

1. indictrans2
   Best for Indian language translation:
   - English → Hindi/Kannada/Tamil/Telugu/etc.
   - Indian language → English
   - Indian language → Indian language

2. nllb
   Best for international language translation:
   - English → Spanish/French/German/Chinese/etc.
   - Spanish/French/etc. → English
   - International language → international language

Input:
- JSON transcript segments from Phase 3

Output:
- translated JSON segments
- translated SRT
- translated TXT

Expected command style:

python 04_translate_any_language.py \
  --input-json transcript.json \
  --source-language english \
  --target-language hindi \
  --engine indictrans2 \
  --model ai4bharat/indictrans2-en-indic-1B \
  --output-json hindi_segments.json \
  --output-srt hindi.srt \
  --output-txt hindi.txt \
  --device cuda \
  --batch-size 8
"""

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple


# ============================================================
# LANGUAGE NORMALIZATION
# ============================================================

LANGUAGE_ALIASES = {
    # English
    "english": "english",
    "en": "english",

    # Indian languages
    "hindi": "hindi",
    "hi": "hindi",

    "kannada": "kannada",
    "kn": "kannada",

    "tamil": "tamil",
    "ta": "tamil",

    "telugu": "telugu",
    "te": "telugu",

    "malayalam": "malayalam",
    "ml": "malayalam",

    "marathi": "marathi",
    "mr": "marathi",

    "bengali": "bengali",
    "bangla": "bengali",
    "bn": "bengali",

    "gujarati": "gujarati",
    "gu": "gujarati",

    "punjabi": "punjabi",
    "pa": "punjabi",

    "odia": "odia",
    "oriya": "odia",
    "or": "odia",

    "assamese": "assamese",
    "as": "assamese",

    # International languages
    "spanish": "spanish",
    "es": "spanish",

    "french": "french",
    "fr": "french",

    "german": "german",
    "de": "german",

    "italian": "italian",
    "it": "italian",

    "portuguese": "portuguese",
    "pt": "portuguese",

    "chinese": "chinese",
    "mandarin": "chinese",
    "zh": "chinese",
    "zh-cn": "chinese",

    "japanese": "japanese",
    "ja": "japanese",

    "korean": "korean",
    "ko": "korean",

    "arabic": "arabic",
    "ar": "arabic",

    "russian": "russian",
    "ru": "russian",

    "dutch": "dutch",
    "nl": "dutch",

    "polish": "polish",
    "pl": "polish",

    "turkish": "turkish",
    "tr": "turkish",
}


INDIC_LANGUAGES = {
    "hindi",
    "kannada",
    "tamil",
    "telugu",
    "malayalam",
    "marathi",
    "bengali",
    "gujarati",
    "punjabi",
    "odia",
    "assamese",
}


INDIC_CODES = {
    "english": "eng_Latn",

    "hindi": "hin_Deva",
    "kannada": "kan_Knda",
    "tamil": "tam_Taml",
    "telugu": "tel_Telu",
    "malayalam": "mal_Mlym",
    "marathi": "mar_Deva",
    "bengali": "ben_Beng",
    "gujarati": "guj_Gujr",
    "punjabi": "pan_Guru",
    "odia": "ory_Orya",
    "assamese": "asm_Beng",
}


NLLB_CODES = {
    "english": "eng_Latn",
    "spanish": "spa_Latn",
    "french": "fra_Latn",
    "german": "deu_Latn",
    "italian": "ita_Latn",
    "portuguese": "por_Latn",
    "chinese": "zho_Hans",
    "japanese": "jpn_Jpan",
    "korean": "kor_Hang",
    "arabic": "arb_Arab",
    "russian": "rus_Cyrl",
    "dutch": "nld_Latn",
    "polish": "pol_Latn",
    "turkish": "tur_Latn",

    # Some Indian languages are also available in NLLB.
    # These are fallback codes only. Prefer IndicTrans2 for Indian languages.
    "hindi": "hin_Deva",
    "bengali": "ben_Beng",
    "gujarati": "guj_Gujr",
    "marathi": "mar_Deva",
    "tamil": "tam_Taml",
    "telugu": "tel_Telu",
    "malayalam": "mal_Mlym",
    "kannada": "kan_Knda",
    "punjabi": "pan_Guru",
    "odia": "ory_Orya",
    "assamese": "asm_Beng",
}


def normalize_language(language: str) -> str:
    key = language.strip().lower()
    return LANGUAGE_ALIASES.get(key, key)


def is_indic_language(language: str) -> bool:
    return normalize_language(language) in INDIC_LANGUAGES


# ============================================================
# FILE HELPERS
# ============================================================

def seconds_to_srt_time(seconds: float) -> str:
    seconds = max(0.0, float(seconds))

    milliseconds = int(round((seconds - int(seconds)) * 1000))
    total_seconds = int(seconds)

    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    secs = total_seconds % 60

    if milliseconds >= 1000:
        milliseconds -= 1000
        secs += 1

    return f"{hours:02d}:{minutes:02d}:{secs:02d},{milliseconds:03d}"


def load_segments(input_json: Path) -> List[Dict]:
    if not input_json.exists():
        raise FileNotFoundError(f"Input JSON not found: {input_json}")

    with input_json.open("r", encoding="utf-8") as file:
        data = json.load(file)

    if not isinstance(data, list):
        raise ValueError("Input JSON must be a list of segment objects.")

    return data


def write_json(segments: List[Dict], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(segments, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def write_srt(segments: List[Dict], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    lines = []

    for index, segment in enumerate(segments, start=1):
        text = get_translated_text(segment).strip()

        if not text:
            continue

        start = seconds_to_srt_time(float(segment["start"]))
        end = seconds_to_srt_time(float(segment["end"]))

        lines.append(str(index))
        lines.append(f"{start} --> {end}")
        lines.append(text)
        lines.append("")

    output_path.write_text("\n".join(lines), encoding="utf-8")


def write_txt(segments: List[Dict], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    lines = []

    for segment in segments:
        text = get_translated_text(segment).strip()

        if text:
            lines.append(text)

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def get_source_text(segment: Dict) -> str:
    for key in ["text", "source_text", "transcript", "original_text"]:
        value = segment.get(key)

        if isinstance(value, str) and value.strip():
            return value.strip()

    return ""


def get_translated_text(segment: Dict) -> str:
    for key in ["translated_text", "translation", "target_text", "text"]:
        value = segment.get(key)

        if isinstance(value, str) and value.strip():
            return value.strip()

    return ""


def batch_items(items: List[str], batch_size: int) -> List[List[str]]:
    return [
        items[index:index + batch_size]
        for index in range(0, len(items), batch_size)
    ]


# ============================================================
# TRANSLATION VALIDATION + CLEANUP
# ============================================================

REPEATED_PUNCTUATION_PATTERN = re.compile(r"([\-–—_\.])(?:\s*\1){5,}")
REPEATED_SINGLE_TOKEN_PATTERN = re.compile(r"\b([A-Za-zÀ-ÿ])\b(?:\s+\1\b){8,}", re.IGNORECASE)
REPEATED_WORD_PATTERN = re.compile(r"\b([\wÀ-ÿ]+)\b(?:\s+\1\b){6,}", re.IGNORECASE)
MULTISPACE_PATTERN = re.compile(r"\s+")


COMMON_SHORT_TRANSLATION_FALLBACKS = {
    "spanish": {
        "external": "externo",
        "internal": "interno",
        "yes": "sí",
        "no": "no",
        "and": "y",
        "or": "o",
        "brain": "cerebro",
        "mind": "mente",
        "body": "cuerpo",
        "cell": "célula",
        "cells": "células",
        "energy": "energía",
        "memory": "memoria",
        "consciousness": "conciencia",
        "genome": "genoma",
        "protein": "proteína",
        "proteins": "proteínas",
        "dna": "ADN",
        "rna": "ARN",
        "atma": "Atma",
        "ruh": "Ruh",
        "ru": "Ruh",
    },
    "french": {
        "external": "externe",
        "internal": "interne",
        "yes": "oui",
        "no": "non",
    },
}


def clean_translation_text(text: str) -> str:
    """
    Removes obvious decoder garbage without changing normal translation text.
    """

    if not isinstance(text, str):
        return ""

    cleaned = text.strip()
    cleaned = MULTISPACE_PATTERN.sub(" ", cleaned)

    # Remove hallucinated repeated punctuation tails such as "- - - - -".
    cleaned = REPEATED_PUNCTUATION_PATTERN.sub("", cleaned).strip()

    # Remove long repeated single-letter/word tails.
    cleaned = REPEATED_SINGLE_TOKEN_PATTERN.sub("", cleaned).strip()
    cleaned = REPEATED_WORD_PATTERN.sub(r"\1", cleaned).strip()

    # Tidy spaces before punctuation.
    cleaned = re.sub(r"\s+([,.;:!?])", r"\1", cleaned)
    cleaned = re.sub(r"([¿¡])\s+", r"\1", cleaned)

    return cleaned.strip()


def is_repetition_garbage(text: str) -> bool:
    if not text or not text.strip():
        return True

    stripped = text.strip()

    if REPEATED_PUNCTUATION_PATTERN.search(stripped):
        return True

    if REPEATED_SINGLE_TOKEN_PATTERN.search(stripped):
        return True

    if REPEATED_WORD_PATTERN.search(stripped):
        return True

    tokens = stripped.split()
    if len(tokens) >= 10:
        unique_ratio = len(set(token.lower() for token in tokens)) / max(1, len(tokens))
        if unique_ratio < 0.18:
            return True

    return False


def looks_like_unfinished_translation(text: str) -> bool:
    stripped = text.strip()

    if not stripped:
        return True

    unfinished_endings = (
        " y la",
        " y el",
        " de la",
        " de el",
        " del",
        " que",
        " and the",
        " of the",
        " and",
        " or",
    )

    lowered = stripped.lower()

    if lowered.endswith(unfinished_endings):
        return True

    if stripped.endswith(("...", "…")):
        return True

    return False


def translation_length_ratio(source_text: str, translated_text: str) -> float:
    source_len = max(1, len(source_text.strip()))
    return len(translated_text.strip()) / source_len


def validate_translation_pair(
    source_text: str,
    translated_text: str,
    target_language: str,
    max_length_ratio: float,
    min_length_ratio: float,
) -> Tuple[bool, List[str]]:
    """
    Returns (is_bad, reasons).
    This catches common decoder failures before TTS sees the text.
    """

    reasons = []
    source_text = source_text.strip()
    translated_text = translated_text.strip()

    if not translated_text:
        reasons.append("empty_translation")

    if is_repetition_garbage(translated_text):
        reasons.append("repetition_garbage")

    ratio = translation_length_ratio(source_text, translated_text)

    if source_text and ratio > max_length_ratio and len(translated_text) > 120:
        reasons.append(f"too_long_ratio_{ratio:.2f}")

    if source_text and ratio < min_length_ratio and len(source_text) > 20:
        reasons.append(f"too_short_ratio_{ratio:.2f}")

    if looks_like_unfinished_translation(translated_text):
        reasons.append("possibly_unfinished")

    # Very short sources are easily mistranslated into garbage.
    if len(source_text.split()) <= 2 and is_repetition_garbage(translated_text):
        reasons.append("short_source_decoder_failure")

    return bool(reasons), reasons


def fallback_short_translation(source_text: str, target_language: str) -> Optional[str]:
    mapping = COMMON_SHORT_TRANSLATION_FALLBACKS.get(normalize_language(target_language), {})
    key = source_text.strip().lower().strip(".,:;!?()[]{}\"'")
    return mapping.get(key)


def translate_texts_by_engine(
    texts: List[str],
    args,
    source_language: str,
    target_language: str,
) -> List[str]:
    if args.engine == "indictrans2":
        return translate_with_indictrans2(
            texts=texts,
            source_language=source_language,
            target_language=target_language,
            model_name=args.model,
            source_indic_code=args.source_indic_code,
            target_indic_code=args.target_indic_code,
            device=args.device,
            batch_size=args.batch_size,
            max_length=args.max_length,
        )

    if args.engine == "nllb":
        return translate_with_nllb(
            texts=texts,
            source_language=source_language,
            target_language=target_language,
            model_name=args.model,
            source_nllb_code=args.source_nllb_code,
            target_nllb_code=args.target_nllb_code,
            device=args.device,
            batch_size=args.batch_size,
            max_length=args.max_length,
        )

    if args.engine == "seamlessm4t":
        raise NotImplementedError(
            "SeamlessM4T text translation is reserved for a later upgrade. "
            "Use --engine nllb for international text translation now."
        )

    raise ValueError(f"Unsupported translation engine: {args.engine}")


def validate_clean_and_retry_translations(
    source_texts: List[str],
    translations: List[str],
    args,
    source_language: str,
    target_language: str,
) -> Tuple[List[str], List[Dict]]:
    """
    Cleans obvious garbage, retries bad translations, and returns a validation report.
    """

    if len(source_texts) != len(translations):
        raise RuntimeError(
            f"Translation count mismatch. Input texts: {len(source_texts)}, "
            f"translations: {len(translations)}"
        )

    cleaned_translations = [clean_translation_text(item) for item in translations]
    validation_report = []

    if args.disable_validation:
        return cleaned_translations, validation_report

    for retry_round in range(args.retry_bad_translations + 1):
        bad_indices = []

        for index, (source_text, translated_text) in enumerate(zip(source_texts, cleaned_translations)):
            is_bad, reasons = validate_translation_pair(
                source_text=source_text,
                translated_text=translated_text,
                target_language=target_language,
                max_length_ratio=args.max_length_ratio,
                min_length_ratio=args.min_length_ratio,
            )

            if is_bad:
                bad_indices.append(index)

                validation_report.append(
                    {
                        "index": index,
                        "retry_round": retry_round,
                        "source_text": source_text,
                        "translation_before": translated_text,
                        "reasons": reasons,
                    }
                )

        if not bad_indices:
            break

        if retry_round >= args.retry_bad_translations:
            break

        print("\n============================================================")
        print(f"Translation validator found {len(bad_indices)} suspicious segments.")
        print(f"Retry round {retry_round + 1}/{args.retry_bad_translations}")
        print("============================================================")

        retry_texts = [source_texts[index] for index in bad_indices]
        retry_outputs = translate_texts_by_engine(
            texts=retry_texts,
            args=args,
            source_language=source_language,
            target_language=target_language,
        )

        for bad_index, retry_output in zip(bad_indices, retry_outputs):
            cleaned_translations[bad_index] = clean_translation_text(retry_output)

    # Final pass after all retries.
    final_bad = []
    for index, (source_text, translated_text) in enumerate(zip(source_texts, cleaned_translations)):
        is_bad, reasons = validate_translation_pair(
            source_text=source_text,
            translated_text=translated_text,
            target_language=target_language,
            max_length_ratio=args.max_length_ratio,
            min_length_ratio=args.min_length_ratio,
        )

        if not is_bad:
            continue

        fallback = fallback_short_translation(source_text, target_language)

        if fallback:
            print(f"Using short fallback for segment {index}: {source_text!r} -> {fallback!r}")
            cleaned_translations[index] = fallback
            validation_report.append(
                {
                    "index": index,
                    "action": "short_fallback",
                    "source_text": source_text,
                    "translation_before": translated_text,
                    "translation_after": fallback,
                    "reasons": reasons,
                }
            )
            continue

        final_bad.append((index, reasons))
        validation_report.append(
            {
                "index": index,
                "action": args.bad_translation_action,
                "source_text": source_text,
                "translation_before": translated_text,
                "reasons": reasons,
            }
        )

        if args.bad_translation_action == "fail":
            raise RuntimeError(
                f"Bad translation remains after retry at text index {index}: {reasons}\n"
                f"Source: {source_text}\nTranslation: {translated_text}"
            )

        if args.bad_translation_action == "fallback_source":
            cleaned_translations[index] = source_text

        elif args.bad_translation_action == "mark":
            cleaned_translations[index] = f"[REVIEW_TRANSLATION] {translated_text or source_text}"

        elif args.bad_translation_action == "keep":
            pass

        else:
            raise ValueError(f"Unsupported bad translation action: {args.bad_translation_action}")

    if final_bad:
        print("\nWARNING: Some suspicious translations remain after retries.")
        for index, reasons in final_bad[:20]:
            print(f"  Text index {index}: {', '.join(reasons)}")

    return cleaned_translations, validation_report


# ============================================================
# NLLB TRANSLATION
# ============================================================

def translate_with_nllb(
    texts: List[str],
    source_language: str,
    target_language: str,
    model_name: str,
    source_nllb_code: Optional[str],
    target_nllb_code: Optional[str],
    device: str,
    batch_size: int,
    max_length: int,
) -> List[str]:
    try:
        import torch
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    except ImportError as error:
        raise ImportError(
            "torch/transformers is not installed. "
            "Use the multi-translate-tts environment for NLLB translation."
        ) from error

    source_language = normalize_language(source_language)
    target_language = normalize_language(target_language)

    src_code = source_nllb_code or NLLB_CODES.get(source_language)
    tgt_code = target_nllb_code or NLLB_CODES.get(target_language)

    if not src_code:
        raise ValueError(f"No NLLB source code found for language: {source_language}")

    if not tgt_code:
        raise ValueError(f"No NLLB target code found for language: {target_language}")

    print("============================================================")
    print("NLLB Translation")
    print("============================================================")
    print(f"Model           : {model_name}")
    print(f"Source language : {source_language}")
    print(f"Target language : {target_language}")
    print(f"Source NLLB     : {src_code}")
    print(f"Target NLLB     : {tgt_code}")
    print(f"Device          : {device}")
    print(f"Batch size      : {batch_size}")
    print("============================================================")

    tokenizer = AutoTokenizer.from_pretrained(model_name, src_lang=src_code)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    if device == "cuda" and torch.cuda.is_available():
        model = model.to("cuda")
        torch_device = "cuda"
    else:
        torch_device = "cpu"

    forced_bos_token_id = tokenizer.convert_tokens_to_ids(tgt_code)

    translated_texts = []

    for batch in batch_items(texts, batch_size):
        inputs = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        )

        inputs = {
            key: value.to(torch_device)
            for key, value in inputs.items()
        }

        generated_tokens = model.generate(
            **inputs,
            forced_bos_token_id=forced_bos_token_id,
            max_length=max_length,
            num_beams=5,
        )

        outputs = tokenizer.batch_decode(
            generated_tokens,
            skip_special_tokens=True,
        )

        translated_texts.extend([output.strip() for output in outputs])

    return translated_texts


# ============================================================
# INDIC TRANS2 TRANSLATION
# ============================================================

def translate_with_indictrans2(
    texts: List[str],
    source_language: str,
    target_language: str,
    model_name: str,
    source_indic_code: Optional[str],
    target_indic_code: Optional[str],
    device: str,
    batch_size: int,
    max_length: int,
) -> List[str]:
    """
    IndicTrans2 translation.

    This function supports two paths:

    Path A:
    If the IndicTransToolkit package is installed and exposes the processor,
    it will use the official-style preprocessing/postprocessing.

    Path B:
    If toolkit imports fail, it falls back to Hugging Face AutoTokenizer +
    AutoModelForSeq2SeqLM.

    This keeps the script usable across slightly different IndicTrans2 setups.
    """

    source_language = normalize_language(source_language)
    target_language = normalize_language(target_language)

    src_code = source_indic_code or INDIC_CODES.get(source_language)
    tgt_code = target_indic_code or INDIC_CODES.get(target_language)

    if not src_code:
        raise ValueError(f"No Indic source code found for language: {source_language}")

    if not tgt_code:
        raise ValueError(f"No Indic target code found for language: {target_language}")

    print("============================================================")
    print("IndicTrans2 Translation")
    print("============================================================")
    print(f"Model           : {model_name}")
    print(f"Source language : {source_language}")
    print(f"Target language : {target_language}")
    print(f"Source Indic    : {src_code}")
    print(f"Target Indic    : {tgt_code}")
    print(f"Device          : {device}")
    print(f"Batch size      : {batch_size}")
    print("============================================================")

    try:
        return translate_with_indictrans2_toolkit(
            texts=texts,
            source_code=src_code,
            target_code=tgt_code,
            model_name=model_name,
            device=device,
            batch_size=batch_size,
            max_length=max_length,
        )
    except Exception as toolkit_error:
        print("\nToolkit path failed or unavailable.")
        print(f"Toolkit error: {toolkit_error}")
        print("Falling back to Hugging Face Seq2Seq path.\n")

        return translate_with_indictrans2_hf_fallback(
            texts=texts,
            source_code=src_code,
            target_code=tgt_code,
            model_name=model_name,
            device=device,
            batch_size=batch_size,
            max_length=max_length,
        )


def translate_with_indictrans2_toolkit(
    texts: List[str],
    source_code: str,
    target_code: str,
    model_name: str,
    device: str,
    batch_size: int,
    max_length: int,
) -> List[str]:
    try:
        import torch
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    except ImportError as error:
        raise ImportError(
            "torch/transformers is not installed in this environment."
        ) from error

    # IndicTransToolkit has had slightly different import paths in examples/installations.
    # Try common possibilities.
    IndicProcessor = None

    import_errors = []

    try:
        from IndicTransToolkit.processor import IndicProcessor as ToolkitIndicProcessor
        IndicProcessor = ToolkitIndicProcessor
    except Exception as error:
        import_errors.append(error)

    if IndicProcessor is None:
        try:
            from IndicTransToolkit import IndicProcessor as ToolkitIndicProcessor
            IndicProcessor = ToolkitIndicProcessor
        except Exception as error:
            import_errors.append(error)

    if IndicProcessor is None:
        raise ImportError(f"Could not import IndicProcessor. Errors: {import_errors}")

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_name,
        trust_remote_code=True,
    )

    if device == "cuda" and torch.cuda.is_available():
        model = model.to("cuda")
        torch_device = "cuda"
    else:
        torch_device = "cpu"

    ip = IndicProcessor(inference=True)

    translations = []

    for batch in batch_items(texts, batch_size):
        batch = [text.strip() for text in batch]

        preprocessed_batch = ip.preprocess_batch(
            batch,
            src_lang=source_code,
            tgt_lang=target_code,
        )

        inputs = tokenizer(
            preprocessed_batch,
            truncation=True,
            padding=True,
            return_tensors="pt",
            max_length=max_length,
        )

        inputs = {
            key: value.to(torch_device)
            for key, value in inputs.items()
        }

        with torch.no_grad():
            generated_tokens = model.generate(
                **inputs,
                max_length=max_length,
                num_beams=5,
                num_return_sequences=1,
            )

        decoded = tokenizer.batch_decode(
            generated_tokens,
            skip_special_tokens=True,
        )

        postprocessed = ip.postprocess_batch(
            decoded,
            lang=target_code,
        )

        translations.extend([item.strip() for item in postprocessed])

    return translations


def translate_with_indictrans2_hf_fallback(
    texts: List[str],
    source_code: str,
    target_code: str,
    model_name: str,
    device: str,
    batch_size: int,
    max_length: int,
) -> List[str]:
    try:
        import torch
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    except ImportError as error:
        raise ImportError(
            "torch/transformers is not installed in this environment."
        ) from error

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
    )

    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_name,
        trust_remote_code=True,
    )

    if device == "cuda" and torch.cuda.is_available():
        model = model.to("cuda")
        torch_device = "cuda"
    else:
        torch_device = "cpu"

    translations = []

    for batch in batch_items(texts, batch_size):
        # Many IndicTrans2 HF examples expect the language tags in the text
        # if preprocessing is not available.
        tagged_batch = [
            f"{source_code} {target_code} {text.strip()}"
            for text in batch
        ]

        inputs = tokenizer(
            tagged_batch,
            truncation=True,
            padding=True,
            return_tensors="pt",
            max_length=max_length,
        )

        inputs = {
            key: value.to(torch_device)
            for key, value in inputs.items()
        }

        with torch.no_grad():
            generated_tokens = model.generate(
                **inputs,
                max_length=max_length,
                num_beams=5,
                num_return_sequences=1,
            )

        decoded = tokenizer.batch_decode(
            generated_tokens,
            skip_special_tokens=True,
        )

        translations.extend([item.strip() for item in decoded])

    return translations


# ============================================================
# MAIN TRANSLATION LOGIC
# ============================================================

def apply_translations_to_segments(
    segments: List[Dict],
    translations: List[str],
    source_language: str,
    target_language: str,
    engine: str,
    model_name: str,
) -> List[Dict]:
    translated_segments = []

    translation_index = 0

    for segment in segments:
        source_text = get_source_text(segment)

        new_segment = dict(segment)

        new_segment["source_text"] = source_text
        new_segment["source_language"] = source_language
        new_segment["target_language"] = target_language
        new_segment["translation_engine"] = engine
        new_segment["translation_model"] = model_name

        if source_text.strip():
            translated = translations[translation_index]
            translation_index += 1
        else:
            translated = ""

        new_segment["translated_text"] = translated
        new_segment["translation"] = translated
        new_segment["target_text"] = translated

        translated_segments.append(new_segment)

    return translated_segments


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Phase 4 translation script for IndicTrans2 or NLLB."
    )

    parser.add_argument(
        "--input-json",
        required=True,
        help="Input transcript JSON from Phase 3.",
    )

    parser.add_argument(
        "--source-language",
        required=True,
        help="Source language name/code.",
    )

    parser.add_argument(
        "--target-language",
        required=True,
        help="Target language name/code.",
    )

    parser.add_argument(
        "--engine",
        default="indictrans2",
        choices=["indictrans2", "nllb", "seamlessm4t"],
        help="Translation engine.",
    )

    parser.add_argument(
        "--model",
        required=True,
        help="Translation model name.",
    )

    parser.add_argument(
        "--output-json",
        required=True,
        help="Output translated segments JSON.",
    )

    parser.add_argument(
        "--output-srt",
        required=True,
        help="Output translated SRT.",
    )

    parser.add_argument(
        "--output-txt",
        required=True,
        help="Output translated TXT transcript.",
    )

    parser.add_argument(
        "--device",
        default="cuda",
        help="Device: cuda or cpu.",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Translation batch size.",
    )

    parser.add_argument(
        "--max-length",
        type=int,
        default=256,
        help="Maximum sequence length for translation.",
    )

    parser.add_argument(
        "--source-indic-code",
        default=None,
        help="Optional source IndicTrans2 language code.",
    )

    parser.add_argument(
        "--target-indic-code",
        default=None,
        help="Optional target IndicTrans2 language code.",
    )

    parser.add_argument(
        "--source-nllb-code",
        default=None,
        help="Optional source NLLB language code.",
    )

    parser.add_argument(
        "--target-nllb-code",
        default=None,
        help="Optional target NLLB language code.",
    )

    parser.add_argument(
        "--disable-validation",
        action="store_true",
        help="Disable translation validation/cleanup. Not recommended for production.",
    )

    parser.add_argument(
        "--retry-bad-translations",
        type=int,
        default=2,
        help="Retry suspicious translations N times before applying bad-translation action.",
    )

    parser.add_argument(
        "--bad-translation-action",
        default="mark",
        choices=["mark", "keep", "fallback_source", "fail"],
        help="What to do if a translation remains suspicious after retries.",
    )

    parser.add_argument(
        "--max-length-ratio",
        type=float,
        default=4.0,
        help="Flag translations longer than this source/target character ratio.",
    )

    parser.add_argument(
        "--min-length-ratio",
        type=float,
        default=0.12,
        help="Flag translations shorter than this source/target character ratio for longer source text.",
    )

    parser.add_argument(
        "--validation-report-json",
        default=None,
        help="Optional path for validation report JSON. Default: output_json stem + _validation_report.json.",
    )

    parser.add_argument(
        "--always-write-validation-report",
        action="store_true",
        help="Write an empty validation report even when no suspicious segments are found.",
    )

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    input_json = Path(args.input_json)
    output_json = Path(args.output_json)
    output_srt = Path(args.output_srt)
    output_txt = Path(args.output_txt)

    source_language = normalize_language(args.source_language)
    target_language = normalize_language(args.target_language)

    print("\n============================================================")
    print("PHASE 4 TRANSLATION")
    print("============================================================")
    print(f"Input JSON       : {input_json}")
    print(f"Source language  : {source_language}")
    print(f"Target language  : {target_language}")
    print(f"Engine           : {args.engine}")
    print(f"Model            : {args.model}")
    print(f"Device           : {args.device}")
    print(f"Batch size       : {args.batch_size}")
    print(f"Validation       : {'disabled' if args.disable_validation else 'enabled'}")
    print(f"Bad action       : {args.bad_translation_action}")
    print(f"Retry bad        : {args.retry_bad_translations}")
    print(f"Output JSON      : {output_json}")
    print(f"Output SRT       : {output_srt}")
    print(f"Output TXT       : {output_txt}")
    print("============================================================\n")

    segments = load_segments(input_json)

    source_texts = [
        get_source_text(segment)
        for segment in segments
        if get_source_text(segment).strip()
    ]

    if not source_texts:
        raise RuntimeError("No source text found in input JSON.")

    translations = translate_texts_by_engine(
        texts=source_texts,
        args=args,
        source_language=source_language,
        target_language=target_language,
    )

    translations, validation_report = validate_clean_and_retry_translations(
        source_texts=source_texts,
        translations=translations,
        args=args,
        source_language=source_language,
        target_language=target_language,
    )

    if args.validation_report_json:
        report_path = Path(args.validation_report_json)
    else:
        report_path = output_json.with_name(output_json.stem + "_validation_report.json")

    if validation_report:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(
            json.dumps(validation_report, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"Translation validation report written: {report_path}")
    elif args.always_write_validation_report:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text("[]\n", encoding="utf-8")
        print(f"Empty translation validation report written: {report_path}")

    translated_segments = apply_translations_to_segments(
        segments=segments,
        translations=translations,
        source_language=source_language,
        target_language=target_language,
        engine=args.engine,
        model_name=args.model,
    )

    write_json(translated_segments, output_json)
    write_srt(translated_segments, output_srt)
    write_txt(translated_segments, output_txt)

    print("\n============================================================")
    print("TRANSLATION COMPLETE")
    print("============================================================")
    print(f"Segments written : {len(translated_segments)}")
    print(f"JSON             : {output_json}")
    print(f"SRT              : {output_srt}")
    print(f"TXT              : {output_txt}")
    print("============================================================\n")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print("\nERROR:")
        print(error)
        sys.exit(1)