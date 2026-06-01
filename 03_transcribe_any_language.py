#!/usr/bin/env python3
"""
Phase 3: Transcribe Any Language

This script is designed to be called directly or by:

08_run_full_dubbing_pipeline.py

Supported engines:

1. faster-whisper
   Best for:
   - English
   - Spanish
   - French
   - German
   - Arabic
   - Chinese
   - Japanese
   - many international languages

2. indic-conformer
   Best for:
   - Hindi
   - Kannada
   - Tamil
   - Telugu
   - Malayalam
   - Marathi
   - Bengali
   - Gujarati
   - Punjabi
   - Odia
   - Assamese

Output files:

- JSON segments
- SRT subtitles
- TXT transcript

Expected command style:

python 03_transcribe_any_language.py \
  --input-audio input.wav \
  --source-language english \
  --engine faster-whisper \
  --model large-v3 \
  --output-json transcript.json \
  --output-srt transcript.srt \
  --output-txt transcript.txt \
  --device cuda \
  --compute-type float16
"""

import argparse
import json
import math
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple


LANGUAGE_ALIASES = {
    "english": "english",
    "en": "english",

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
}


WHISPER_LANGUAGE_CODES = {
    "english": "en",
    "hindi": "hi",
    "kannada": "kn",
    "tamil": "ta",
    "telugu": "te",
    "malayalam": "ml",
    "marathi": "mr",
    "bengali": "bn",
    "gujarati": "gu",
    "punjabi": "pa",
    "odia": "or",
    "assamese": "as",
    "spanish": "es",
    "french": "fr",
    "german": "de",
    "italian": "it",
    "portuguese": "pt",
    "chinese": "zh",
    "japanese": "ja",
    "korean": "ko",
    "arabic": "ar",
    "russian": "ru",
}


INDIC_LANGUAGE_CODES = {
    "hindi": "hi",
    "kannada": "kn",
    "tamil": "ta",
    "telugu": "te",
    "malayalam": "ml",
    "marathi": "mr",
    "bengali": "bn",
    "gujarati": "gu",
    "punjabi": "pa",
    "odia": "or",
    "assamese": "as",
}


def normalize_language(language: str) -> str:
    key = language.strip().lower()
    return LANGUAGE_ALIASES.get(key, key)


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


def write_srt(segments: List[Dict], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    lines = []

    for index, segment in enumerate(segments, start=1):
        start = seconds_to_srt_time(segment["start"])
        end = seconds_to_srt_time(segment["end"])
        text = segment.get("text", "").strip()

        if not text:
            continue

        lines.append(str(index))
        lines.append(f"{start} --> {end}")
        lines.append(text)
        lines.append("")

    output_path.write_text("\n".join(lines), encoding="utf-8")


def write_txt(segments: List[Dict], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    text = "\n".join(
        segment.get("text", "").strip()
        for segment in segments
        if segment.get("text", "").strip()
    )

    output_path.write_text(text + "\n", encoding="utf-8")


def write_json(segments: List[Dict], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(segments, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def ffprobe_duration(audio_path: Path) -> float:
    command = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(audio_path),
    ]

    result = subprocess.run(command, capture_output=True, text=True, check=True)
    return float(result.stdout.strip())


def extract_audio_chunk(
    input_audio: Path,
    output_audio: Path,
    start: float,
    duration: float,
    sample_rate: int = 16000,
) -> None:
    command = [
        "ffmpeg",
        "-y",
        "-ss",
        str(start),
        "-t",
        str(duration),
        "-i",
        str(input_audio),
        "-ar",
        str(sample_rate),
        "-ac",
        "1",
        "-sample_fmt",
        "s16",
        str(output_audio),
    ]

    subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)


def transcribe_with_faster_whisper(
    input_audio: Path,
    source_language: str,
    model_name: str,
    device: str,
    compute_type: str,
    beam_size: int,
    vad_filter: bool,
) -> List[Dict]:
    try:
        from faster_whisper import WhisperModel
    except ImportError as error:
        raise ImportError(
            "faster-whisper is not installed in this environment. "
            "Activate/use audio-asr environment and install faster-whisper."
        ) from error

    language = normalize_language(source_language)
    whisper_language = WHISPER_LANGUAGE_CODES.get(language)

    print("============================================================")
    print("Faster-Whisper Transcription")
    print("============================================================")
    print(f"Input audio     : {input_audio}")
    print(f"Model           : {model_name}")
    print(f"Language        : {language}")
    print(f"Whisper code    : {whisper_language}")
    print(f"Device          : {device}")
    print(f"Compute type    : {compute_type}")
    print(f"Beam size       : {beam_size}")
    print(f"VAD filter      : {vad_filter}")
    print("============================================================")

    model = WhisperModel(
        model_name,
        device=device,
        compute_type=compute_type,
    )

    segments_generator, info = model.transcribe(
        str(input_audio),
        language=whisper_language,
        beam_size=beam_size,
        vad_filter=vad_filter,
        word_timestamps=True,
    )

    print(f"Detected language: {info.language}")
    print(f"Language probability: {info.language_probability}")

    output_segments = []

    for idx, segment in enumerate(segments_generator):
        text = segment.text.strip()

        if not text:
            continue

        words = []

        if getattr(segment, "words", None):
            for word in segment.words:
                words.append(
                    {
                        "word": word.word,
                        "start": float(word.start) if word.start is not None else None,
                        "end": float(word.end) if word.end is not None else None,
                        "probability": float(word.probability) if word.probability is not None else None,
                    }
                )

        output_segments.append(
            {
                "id": idx,
                "start": float(segment.start),
                "end": float(segment.end),
                "text": text,
                "source_language": language,
                "engine": "faster-whisper",
                "model": model_name,
                "words": words,
            }
        )

        print(f"[{segment.start:.2f} → {segment.end:.2f}] {text}")

    return output_segments


def load_transformers_asr_pipeline(
    model_name: str,
    device: str,
):
    try:
        import torch
        from transformers import pipeline
    except ImportError as error:
        raise ImportError(
            "transformers/torch is not installed in this environment. "
            "Activate/use audio-asr environment and install required packages."
        ) from error

    if device == "cuda" and torch.cuda.is_available():
        pipeline_device = 0
    else:
        pipeline_device = -1

    return pipeline(
        task="automatic-speech-recognition",
        model=model_name,
        trust_remote_code=True,
        device=pipeline_device,
    )


def transcribe_with_indic_conformer(
    input_audio: Path,
    source_language: str,
    model_name: str,
    device: str,
    chunk_seconds: float,
) -> List[Dict]:
    """
    Indic Conformer fallback implementation.

    This uses a Transformers ASR pipeline and chunks the audio.
    Some Indic ASR models may have their own preferred inference method.
    If your installed ai4bharat model needs a custom call, keep this CLI interface
    but replace only the internals of this function.
    """

    language = normalize_language(source_language)
    indic_code = INDIC_LANGUAGE_CODES.get(language, language)

    if chunk_seconds <= 0:
        chunk_seconds = 20.0

    print("============================================================")
    print("Indic Conformer Transcription")
    print("============================================================")
    print(f"Input audio     : {input_audio}")
    print(f"Model           : {model_name}")
    print(f"Language        : {language}")
    print(f"Indic code      : {indic_code}")
    print(f"Device          : {device}")
    print(f"Chunk seconds   : {chunk_seconds}")
    print("============================================================")

    asr = load_transformers_asr_pipeline(
        model_name=model_name,
        device=device,
    )

    total_duration = ffprobe_duration(input_audio)
    total_chunks = math.ceil(total_duration / chunk_seconds)

    output_segments = []

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_dir_path = Path(temp_dir)

        for chunk_index in range(total_chunks):
            start = chunk_index * chunk_seconds
            duration = min(chunk_seconds, total_duration - start)

            if duration <= 0:
                continue

            chunk_path = temp_dir_path / f"chunk_{chunk_index:04d}.wav"

            extract_audio_chunk(
                input_audio=input_audio,
                output_audio=chunk_path,
                start=start,
                duration=duration,
                sample_rate=16000,
            )

            try:
                result = asr(str(chunk_path))
            except TypeError:
                result = asr(str(chunk_path), generate_kwargs={"language": indic_code})

            if isinstance(result, dict):
                text = result.get("text", "").strip()
            else:
                text = str(result).strip()

            if not text:
                continue

            segment = {
                "id": len(output_segments),
                "start": float(start),
                "end": float(start + duration),
                "text": text,
                "source_language": language,
                "engine": "indic-conformer",
                "model": model_name,
            }

            output_segments.append(segment)

            print(f"[{segment['start']:.2f} → {segment['end']:.2f}] {text}")

    return output_segments


def transcribe_with_openai_whisper(
    input_audio: Path,
    source_language: str,
    model_name: str,
) -> List[Dict]:
    """
    Optional fallback for the original whisper package.

    Use only if you installed openai-whisper.
    faster-whisper is recommended for production.
    """

    try:
        import whisper
    except ImportError as error:
        raise ImportError(
            "openai-whisper is not installed. "
            "Use --engine faster-whisper or install openai-whisper."
        ) from error

    language = normalize_language(source_language)
    whisper_language = WHISPER_LANGUAGE_CODES.get(language)

    print("============================================================")
    print("OpenAI Whisper Transcription")
    print("============================================================")
    print(f"Input audio     : {input_audio}")
    print(f"Model           : {model_name}")
    print(f"Language        : {language}")
    print(f"Whisper code    : {whisper_language}")
    print("============================================================")

    model = whisper.load_model(model_name)
    result = model.transcribe(
        str(input_audio),
        language=whisper_language,
        word_timestamps=True,
    )

    output_segments = []

    for idx, segment in enumerate(result.get("segments", [])):
        text = segment.get("text", "").strip()

        if not text:
            continue

        output_segments.append(
            {
                "id": idx,
                "start": float(segment.get("start", 0.0)),
                "end": float(segment.get("end", 0.0)),
                "text": text,
                "source_language": language,
                "engine": "whisper",
                "model": model_name,
            }
        )

        print(f"[{segment.get('start', 0.0):.2f} → {segment.get('end', 0.0):.2f}] {text}")

    return output_segments


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Phase 3 transcription script for faster-whisper, Indic Conformer, or Whisper."
    )

    parser.add_argument(
        "--input-audio",
        required=True,
        help="Input audio file path.",
    )

    parser.add_argument(
        "--source-language",
        required=True,
        help="Source language name/code. Example: english, hindi, kannada, es, fr.",
    )

    parser.add_argument(
        "--engine",
        default="faster-whisper",
        choices=["faster-whisper", "indic-conformer", "whisper"],
        help="Transcription engine.",
    )

    parser.add_argument(
        "--model",
        default="large-v3",
        help="Model name. Example: large-v3, medium, ai4bharat/indicconformer.",
    )

    parser.add_argument(
        "--output-json",
        required=True,
        help="Output transcript JSON path.",
    )

    parser.add_argument(
        "--output-srt",
        required=True,
        help="Output SRT subtitle path.",
    )

    parser.add_argument(
        "--output-txt",
        required=True,
        help="Output TXT transcript path.",
    )

    parser.add_argument(
        "--device",
        default="cuda",
        help="Device: cuda or cpu.",
    )

    parser.add_argument(
        "--compute-type",
        default="float16",
        help="faster-whisper compute type: float16, int8_float16, int8, float32.",
    )

    parser.add_argument(
        "--beam-size",
        type=int,
        default=5,
        help="Beam size for faster-whisper.",
    )

    parser.add_argument(
        "--no-vad-filter",
        action="store_true",
        help="Disable VAD filter for faster-whisper.",
    )

    parser.add_argument(
        "--chunk-seconds",
        type=float,
        default=20.0,
        help="Chunk seconds for Indic Conformer fallback mode.",
    )

    parser.add_argument(
        "--source-indic-code",
        default=None,
        help="Optional IndicTrans/Indic language code passed from Phase 8.",
    )

    parser.add_argument(
        "--source-nllb-code",
        default=None,
        help="Optional NLLB language code passed from Phase 8.",
    )

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    input_audio = Path(args.input_audio)
    output_json = Path(args.output_json)
    output_srt = Path(args.output_srt)
    output_txt = Path(args.output_txt)

    if not input_audio.exists():
        raise FileNotFoundError(f"Input audio not found: {input_audio}")

    source_language = normalize_language(args.source_language)

    print("\n============================================================")
    print("PHASE 3 TRANSCRIPTION")
    print("============================================================")
    print(f"Input audio       : {input_audio}")
    print(f"Source language   : {source_language}")
    print(f"Engine            : {args.engine}")
    print(f"Model             : {args.model}")
    print(f"Device            : {args.device}")
    print(f"Output JSON       : {output_json}")
    print(f"Output SRT        : {output_srt}")
    print(f"Output TXT        : {output_txt}")
    print("============================================================\n")

    if args.engine == "faster-whisper":
        segments = transcribe_with_faster_whisper(
            input_audio=input_audio,
            source_language=source_language,
            model_name=args.model,
            device=args.device,
            compute_type=args.compute_type,
            beam_size=args.beam_size,
            vad_filter=not args.no_vad_filter,
        )

    elif args.engine == "indic-conformer":
        segments = transcribe_with_indic_conformer(
            input_audio=input_audio,
            source_language=source_language,
            model_name=args.model,
            device=args.device,
            chunk_seconds=args.chunk_seconds,
        )

    elif args.engine == "whisper":
        segments = transcribe_with_openai_whisper(
            input_audio=input_audio,
            source_language=source_language,
            model_name=args.model,
        )

    else:
        raise ValueError(f"Unsupported engine: {args.engine}")

    if not segments:
        raise RuntimeError("No transcription segments were produced.")

    write_json(segments, output_json)
    write_srt(segments, output_srt)
    write_txt(segments, output_txt)

    print("\n============================================================")
    print("TRANSCRIPTION COMPLETE")
    print("============================================================")
    print(f"Segments written : {len(segments)}")
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