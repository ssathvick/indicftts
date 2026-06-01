#!/usr/bin/env python3
"""
Phase 8: Full AI Dubbing Production Runner

This script is the main production controller for the local AI audio/video dubbing pipeline.

It supports:

1. Indian languages
   - ASR: faster-whisper or Indic Conformer
   - Translation: IndicTrans2
   - TTS: IndicF5

2. International languages
   - ASR: faster-whisper
   - Translation: NLLB / SeamlessM4T
   - TTS: XTTS-v2 / other compatible international TTS script

The script does not merge all Python dependencies into one environment.
That would break because ASR, IndicTrans2, IndicF5, and XTTS use different package versions.

Instead, this script orchestrates separate Conda environments:

audio-asr
indic-translate
indic-tts
multi-translate-tts

Expected lower-level scripts:

03_transcribe_any_language.py
04_translate_any_language.py
05_tts_any_indian_language.py
05_tts_international_xtts.py

Main workflow:

Input video
→ Extract audio
→ Transcribe
→ Translate
→ Generate TTS
→ Polish dubbed audio
→ Optionally mix with original audio
→ Mux with video
→ Save final MP4

Example:

python /mnt/d/aistudio/audio_pipeline/scripts/08_run_full_dubbing_pipeline.py \
  --input-video /mnt/d/aistudio/audio_pipeline/input/videos/demo.mp4 \
  --source-language english \
  --target-language hindi \
  --ref-audio /mnt/d/aistudio/audio_pipeline/input/prompts/male_hindi_ref_24k_mono.wav \
  --ref-text-file /mnt/d/aistudio/audio_pipeline/input/prompts/male_hindi_ref.txt \
  --output-name demo_hindi \
  --audio-denoise \
  --mix-original-audio \
  --soft-subtitles
"""

import argparse
import json
import shlex
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional


# ============================================================
# GLOBAL PATHS
# ============================================================

PROJECT_ROOT = Path("/mnt/d/aistudio/audio_pipeline")

SCRIPTS_DIR = PROJECT_ROOT / "scripts/final"

TRANSCRIBE_SCRIPT = SCRIPTS_DIR / "03_transcribe_any_language.py"
TRANSLATE_SCRIPT = SCRIPTS_DIR / "04_translate_any_language.py"
INDIC_TTS_SCRIPT = SCRIPTS_DIR / "05_tts_any_indian_language_no_scaling.py"
INTERNATIONAL_TTS_SCRIPT = SCRIPTS_DIR / "05_tts_international_xtts.py"

INPUT_AUDIO_DIR = PROJECT_ROOT / "output" / "audio"
TRANSCRIPT_DIR = PROJECT_ROOT / "output" / "transcripts"
TRANSLATION_DIR = PROJECT_ROOT / "output" / "translations"
TTS_DIR = PROJECT_ROOT / "output" / "tts"
FINAL_VIDEO_DIR = PROJECT_ROOT / "output" / "final_video"
JOBS_DIR = PROJECT_ROOT / "output" / "jobs"
LOG_DIR = PROJECT_ROOT / "logs"
CONFIG_DIR = PROJECT_ROOT / "config"


# ============================================================
# LANGUAGE ROUTER
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


INTERNATIONAL_LANGUAGES = {
    "english",
    "spanish",
    "french",
    "german",
    "italian",
    "portuguese",
    "chinese",
    "japanese",
    "korean",
    "arabic",
    "russian",
    "dutch",
    "polish",
    "turkish",
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
}


XTTS_CODES = {
    "english": "en",
    "spanish": "es",
    "french": "fr",
    "german": "de",
    "italian": "it",
    "portuguese": "pt",
    "chinese": "zh-cn",
    "japanese": "ja",
    "korean": "ko",
    "arabic": "ar",
    "russian": "ru",
    "dutch": "nl",
    "polish": "pl",
    "turkish": "tr",
}


def normalize_language(language: str) -> str:
    key = language.strip().lower()
    return LANGUAGE_ALIASES.get(key, key)


def is_indic_language(language: str) -> bool:
    return normalize_language(language) in INDIC_LANGUAGES


def is_international_language(language: str) -> bool:
    return normalize_language(language) in INTERNATIONAL_LANGUAGES


# ============================================================
# GENERAL UTILITIES
# ============================================================

def now_id() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def ensure_directories() -> None:
    for directory in [
        INPUT_AUDIO_DIR,
        TRANSCRIPT_DIR,
        TRANSLATION_DIR,
        TTS_DIR,
        FINAL_VIDEO_DIR,
        JOBS_DIR,
        LOG_DIR,
        CONFIG_DIR,
    ]:
        directory.mkdir(parents=True, exist_ok=True)


def shell_quote(command: List[str]) -> str:
    return " ".join(shlex.quote(str(part)) for part in command)


def run_command(
    command: List[str],
    log_file: Optional[Path] = None,
    allow_failure: bool = False,
) -> subprocess.CompletedProcess:
    print("\n------------------------------------------------------------")
    print("Running command:")
    print(shell_quote(command))
    print("------------------------------------------------------------")

    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)

        with log_file.open("a", encoding="utf-8") as log:
            log.write("\n\n------------------------------------------------------------\n")
            log.write(f"COMMAND @ {datetime.now().isoformat()}\n")
            log.write(shell_quote(command) + "\n")
            log.write("------------------------------------------------------------\n")

            result = subprocess.run(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )

            print(result.stdout)
            log.write(result.stdout)

    else:
        result = subprocess.run(command)

    if result.returncode != 0 and not allow_failure:
        raise RuntimeError(
            f"Command failed with exit code {result.returncode}:\n"
            f"{shell_quote(command)}"
        )

    return result


def conda_run(env_name: str, command: List[str], log_file: Optional[Path] = None) -> None:
    run_command(["conda", "run", "-n", env_name] + command, log_file=log_file)


def append_extra_args(command: List[str], extra_args: Optional[str]) -> List[str]:
    if extra_args:
        command.extend(shlex.split(extra_args))
    return command


def ffprobe_duration(path: Path) -> Optional[float]:
    try:
        result = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                str(path),
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        return float(result.stdout.strip())
    except Exception:
        return None


def write_json(path: Path, data: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def validate_file_exists(path: Path, label: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{label} not found: {path}")


# ============================================================
# MODEL ROUTER
# ============================================================

def choose_asr_defaults(args) -> None:
    source = normalize_language(args.source_language)

    if args.asr_engine == "auto":
        if is_indic_language(source):
            args.asr_engine = "indic-conformer"
        else:
            args.asr_engine = "faster-whisper"

    if args.asr_model == "auto":
        if args.asr_engine == "indic-conformer":
            args.asr_model = "ai4bharat/indicconformer"
        elif args.asr_engine == "faster-whisper":
            args.asr_model = "large-v3"
        else:
            args.asr_model = "large-v3"


def choose_translation_defaults(args) -> None:
    source = normalize_language(args.source_language)
    target = normalize_language(args.target_language)

    if args.translation_engine == "auto":
        if is_indic_language(source) or is_indic_language(target):
            args.translation_engine = "indictrans2"
        else:
            args.translation_engine = "nllb"

    if args.translation_model == "auto":
        if args.translation_engine == "indictrans2":
            if source == "english" and is_indic_language(target):
                args.translation_model = "ai4bharat/indictrans2-en-indic-1B"
            elif is_indic_language(source) and target == "english":
                args.translation_model = "ai4bharat/indictrans2-indic-en-1B"
            elif is_indic_language(source) and is_indic_language(target):
                args.translation_model = "ai4bharat/indictrans2-indic-indic-1B"
            else:
                args.translation_engine = "nllb"
                args.translation_model = "facebook/nllb-200-distilled-600M"

        elif args.translation_engine == "nllb":
            args.translation_model = "facebook/nllb-200-distilled-600M"

        elif args.translation_engine == "seamlessm4t":
            args.translation_model = "facebook/seamless-m4t-v2-large"


def choose_tts_defaults(args) -> None:
    target = normalize_language(args.target_language)

    if args.tts_engine == "auto":
        if is_indic_language(target):
            args.tts_engine = "indicf5"
        else:
            args.tts_engine = "xtts_v2"

    if args.tts_model == "auto":
        if args.tts_engine == "indicf5":
            args.tts_model = "ai4bharat/IndicF5"
        elif args.tts_engine in {"xtts", "xtts_v2"}:
            args.tts_model = "tts_models/multilingual/multi-dataset/xtts_v2"
        elif args.tts_engine == "seamlessm4t":
            args.tts_model = "facebook/seamless-m4t-v2-large"


def choose_translation_env(args) -> str:
    if args.translation_engine == "indictrans2":
        return "indic-translate"

    if args.translation_engine in {"nllb", "seamlessm4t"}:
        return "multi-translate-tts"

    if is_indic_language(args.target_language):
        return "indic-translate"

    return "multi-translate-tts"


def choose_tts_env(args) -> str:
    if args.tts_engine == "indicf5":
        return "indic-tts"

    if args.tts_engine in {"xtts", "xtts_v2", "parler", "seamlessm4t"}:
        return "multi-translate-tts"

    if is_indic_language(args.target_language):
        return "indic-tts"

    return "multi-translate-tts"


def choose_tts_script(args) -> Path:
    if args.tts_script:
        return Path(args.tts_script)

    if args.tts_engine == "indicf5" or is_indic_language(args.target_language):
        return INDIC_TTS_SCRIPT

    return INTERNATIONAL_TTS_SCRIPT


def resolve_auto_options(args) -> None:
    args.source_language = normalize_language(args.source_language)
    args.target_language = normalize_language(args.target_language)

    choose_asr_defaults(args)
    choose_translation_defaults(args)
    choose_tts_defaults(args)

    args.source_indic_code = INDIC_CODES.get(args.source_language)
    args.target_indic_code = INDIC_CODES.get(args.target_language)

    args.source_nllb_code = NLLB_CODES.get(args.source_language)
    args.target_nllb_code = NLLB_CODES.get(args.target_language)

    args.target_xtts_code = XTTS_CODES.get(args.target_language)

    args.translation_env = choose_translation_env(args)
    args.tts_env = choose_tts_env(args)
    args.resolved_tts_script = choose_tts_script(args)


# ============================================================
# PHASE FUNCTIONS
# ============================================================

def extract_audio(input_video: Path, output_audio: Path, args, log_file: Path) -> None:
    command = [
        "ffmpeg",
        "-y",
        "-i",
        str(input_video),
        "-vn",
        "-ar",
        str(args.extract_sample_rate),
        "-ac",
        str(args.extract_channels),
        "-sample_fmt",
        "s16",
        str(output_audio),
    ]
    run_command(command, log_file=log_file)


def run_transcription_stage(
    audio_file: Path,
    transcript_json: Path,
    transcript_srt: Path,
    transcript_txt: Path,
    args,
    log_file: Path,
) -> None:
    script = Path(args.transcribe_script)
    validate_file_exists(script, "Transcription script")

    command = [
        "python",
        str(script),
        "--input-audio",
        str(audio_file),
        "--source-language",
        args.source_language,
        "--engine",
        args.asr_engine,
        "--model",
        args.asr_model,
        "--output-json",
        str(transcript_json),
        "--output-srt",
        str(transcript_srt),
        "--output-txt",
        str(transcript_txt),
    ]

    if args.source_indic_code:
        command.extend(["--source-indic-code", args.source_indic_code])

    if args.source_nllb_code:
        command.extend(["--source-nllb-code", args.source_nllb_code])

    if args.asr_device:
        command.extend(["--device", args.asr_device])

    if args.asr_compute_type:
        command.extend(["--compute-type", args.asr_compute_type])

    if args.asr_chunk_seconds > 0:
        command.extend(["--chunk-seconds", str(args.asr_chunk_seconds)])

    append_extra_args(command, args.asr_extra_args)

    conda_run("audio-asr", command, log_file=log_file)


def run_translation_stage(
    transcript_json: Path,
    translated_json: Path,
    translated_srt: Path,
    translated_txt: Path,
    args,
    log_file: Path,
) -> None:
    script = Path(args.translate_script)
    validate_file_exists(script, "Translation script")

    command = [
        "python",
        str(script),
        "--input-json",
        str(transcript_json),
        "--source-language",
        args.source_language,
        "--target-language",
        args.target_language,
        "--engine",
        args.translation_engine,
        "--model",
        args.translation_model,
        "--output-json",
        str(translated_json),
        "--output-srt",
        str(translated_srt),
        "--output-txt",
        str(translated_txt),
    ]

    if args.source_indic_code:
        command.extend(["--source-indic-code", args.source_indic_code])

    if args.target_indic_code:
        command.extend(["--target-indic-code", args.target_indic_code])

    if args.source_nllb_code:
        command.extend(["--source-nllb-code", args.source_nllb_code])

    if args.target_nllb_code:
        command.extend(["--target-nllb-code", args.target_nllb_code])

    if args.translation_device:
        command.extend(["--device", args.translation_device])

    if args.translation_batch_size > 0:
        command.extend(["--batch-size", str(args.translation_batch_size)])

    append_extra_args(command, args.translation_extra_args)

    conda_run(args.translation_env, command, log_file=log_file)


def run_tts_stage(
    translated_json: Path,
    output_name: str,
    args,
    log_file: Path,
) -> Path:
    script = Path(args.resolved_tts_script)
    validate_file_exists(script, "TTS script")

    command = [
        "python",
        str(script),
        "--language",
        args.target_language,
        "--input-json",
        str(translated_json),
        "--output-dir",
        str(TTS_DIR),
        "--model",
        args.tts_model,
        "--output-name",
        output_name,
        "--export-sample-rate",
        str(args.tts_export_sample_rate),
        "--audio-channels",
        str(args.tts_audio_channels),
    ]

    if args.target_xtts_code:
        command.extend(["--xtts-language-code", args.target_xtts_code])

    if args.ref_audio:
        command.extend(["--ref-audio", str(Path(args.ref_audio))])

    if args.ref_text:
        command.extend(["--ref-text", args.ref_text])

    if args.ref_text_file:
        command.extend(["--ref-text-file", str(Path(args.ref_text_file))])

    if args.tts_limit > 0:
        command.extend(["--limit", str(args.tts_limit)])

    if args.tts_export_mp3:
        command.append("--export-mp3")

    if args.keep_model_rate_wav:
        command.append("--keep-model-rate-wav")

    append_extra_args(command, args.tts_extra_args)

    conda_run(args.tts_env, command, log_file=log_file)

    expected_wav = TTS_DIR / f"{output_name}_{args.tts_export_sample_rate // 1000}k.wav"

    if not expected_wav.exists():
        raise FileNotFoundError(
            f"TTS completed, but expected output was not found:\n{expected_wav}\n"
            "Check the naming convention in your TTS script."
        )

    return expected_wav


def build_audio_filter(args) -> str:
    filters = []

    if args.audio_highpass > 0:
        filters.append(f"highpass=f={args.audio_highpass}")

    if args.audio_lowpass > 0:
        filters.append(f"lowpass=f={args.audio_lowpass}")

    if args.audio_denoise:
        filters.append(f"afftdn=nf={args.audio_denoise_strength}")

    if args.audio_deesser:
        filters.append("deesser")

    if args.audio_normalize == "dynaudnorm":
        filters.append("dynaudnorm")

    elif args.audio_normalize == "loudnorm":
        filters.append(
            f"loudnorm=I={args.loudnorm_i}:TP={args.loudnorm_tp}:LRA={args.loudnorm_lra}"
        )

    if args.audio_volume != 1.0:
        filters.append(f"volume={args.audio_volume}")

    return ",".join(filters)


def polish_audio(input_audio: Path, output_audio: Path, args, log_file: Path) -> None:
    audio_filter = build_audio_filter(args)

    command = [
        "ffmpeg",
        "-y",
        "-i",
        str(input_audio),
    ]

    if audio_filter:
        command.extend(["-af", audio_filter])

    command.extend(
        [
            "-ar",
            str(args.final_audio_sample_rate),
            "-ac",
            str(args.final_audio_channels),
            "-sample_fmt",
            "s16",
            str(output_audio),
        ]
    )

    run_command(command, log_file=log_file)


def extract_original_audio_low(input_video: Path, output_audio: Path, args, log_file: Path) -> None:
    command = [
        "ffmpeg",
        "-y",
        "-i",
        str(input_video),
        "-vn",
        "-af",
        f"volume={args.original_audio_volume}",
        "-ar",
        str(args.final_audio_sample_rate),
        "-ac",
        str(args.final_audio_channels),
        str(output_audio),
    ]
    run_command(command, log_file=log_file)


def mix_audio(original_low_audio: Path, dubbed_audio: Path, mixed_audio: Path, args, log_file: Path) -> None:
    filter_complex = (
        f"[0:a]volume={args.original_audio_volume}[bg];"
        f"[1:a]volume={args.dubbed_audio_volume}[voice];"
        "[bg][voice]amix=inputs=2:duration=longest:dropout_transition=0,"
        f"loudnorm=I={args.loudnorm_i}:TP={args.loudnorm_tp}:LRA={args.loudnorm_lra}"
    )

    command = [
        "ffmpeg",
        "-y",
        "-i",
        str(original_low_audio),
        "-i",
        str(dubbed_audio),
        "-filter_complex",
        filter_complex,
        "-ar",
        str(args.final_audio_sample_rate),
        "-ac",
        str(args.final_audio_channels),
        str(mixed_audio),
    ]

    run_command(command, log_file=log_file)


def mux_final_video(
    input_video: Path,
    final_audio: Path,
    output_video: Path,
    subtitle_file: Optional[Path],
    args,
    log_file: Path,
) -> None:
    if args.burn_subtitles and subtitle_file and subtitle_file.exists():
        subtitle_path = str(subtitle_file).replace(":", "\\:")

        command = [
            "ffmpeg",
            "-y",
            "-i",
            str(input_video),
            "-i",
            str(final_audio),
            "-vf",
            f"subtitles='{subtitle_path}'",
            "-map",
            "0:v:0",
            "-map",
            "1:a:0",
            "-c:v",
            args.video_codec,
            "-preset",
            args.video_preset,
            "-crf",
            str(args.video_crf),
            "-c:a",
            "aac",
            "-b:a",
            args.final_audio_bitrate,
            "-shortest",
            str(output_video),
        ]
    else:
        command = [
            "ffmpeg",
            "-y",
            "-i",
            str(input_video),
            "-i",
            str(final_audio),
            "-map",
            "0:v:0",
            "-map",
            "1:a:0",
            "-c:v",
            "copy",
            "-c:a",
            "aac",
            "-b:a",
            args.final_audio_bitrate,
            "-shortest",
            str(output_video),
        ]

    run_command(command, log_file=log_file)


def create_soft_subtitle_video(
    input_video: Path,
    final_audio: Path,
    subtitle_file: Path,
    output_video: Path,
    args,
    log_file: Path,
) -> None:
    command = [
        "ffmpeg",
        "-y",
        "-i",
        str(input_video),
        "-i",
        str(final_audio),
        "-i",
        str(subtitle_file),
        "-map",
        "0:v:0",
        "-map",
        "1:a:0",
        "-map",
        "2:0",
        "-c:v",
        "copy",
        "-c:a",
        "aac",
        "-b:a",
        args.final_audio_bitrate,
        "-c:s",
        "mov_text",
        "-metadata:s:s:0",
        f"language={args.target_language}",
        "-shortest",
        str(output_video),
    ]

    run_command(command, log_file=log_file)


# ============================================================
# ARGUMENT PARSER
# ============================================================

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Full AI dubbing production runner for Indian and international languages."
    )

    # Core
    parser.add_argument("--input-video", required=True)
    parser.add_argument("--source-language", required=True)
    parser.add_argument("--target-language", required=True)
    parser.add_argument("--output-name", default=None)
    parser.add_argument("--work-id", default=None)

    # Script paths
    parser.add_argument("--transcribe-script", default=str(TRANSCRIBE_SCRIPT))
    parser.add_argument("--translate-script", default=str(TRANSLATE_SCRIPT))
    parser.add_argument("--tts-script", default=None)

    # ASR
    parser.add_argument(
        "--asr-engine",
        default="auto",
        choices=["auto", "faster-whisper", "indic-conformer", "whisper"],
    )
    parser.add_argument("--asr-model", default="auto")
    parser.add_argument("--asr-device", default="cuda")
    parser.add_argument("--asr-compute-type", default="float16")
    parser.add_argument("--asr-chunk-seconds", type=float, default=0)
    parser.add_argument("--asr-extra-args", default=None)

    # Translation
    parser.add_argument(
        "--translation-engine",
        default="auto",
        choices=["auto", "indictrans2", "nllb", "seamlessm4t"],
    )
    parser.add_argument("--translation-model", default="auto")
    parser.add_argument("--translation-device", default="cuda")
    parser.add_argument("--translation-batch-size", type=int, default=8)
    parser.add_argument("--translation-extra-args", default=None)

    # TTS
    parser.add_argument(
        "--tts-engine",
        default="auto",
        choices=["auto", "indicf5", "xtts", "xtts_v2", "parler", "seamlessm4t"],
    )
    parser.add_argument("--tts-model", default="auto")
    parser.add_argument("--ref-audio", default=None)
    parser.add_argument("--ref-text", default=None)
    parser.add_argument("--ref-text-file", default=None)
    parser.add_argument("--tts-export-sample-rate", type=int, default=48000)
    parser.add_argument("--tts-audio-channels", type=int, default=1)
    parser.add_argument("--tts-export-mp3", action="store_true")
    parser.add_argument("--tts-limit", type=int, default=0)
    parser.add_argument("--keep-model-rate-wav", action="store_true")
    parser.add_argument("--tts-extra-args", default=None)

    # Audio extraction
    parser.add_argument("--extract-sample-rate", type=int, default=16000)
    parser.add_argument("--extract-channels", type=int, default=1)

    # Audio polishing
    parser.add_argument("--final-audio-sample-rate", type=int, default=48000)
    parser.add_argument("--final-audio-channels", type=int, default=2)
    parser.add_argument("--audio-highpass", type=int, default=80)
    parser.add_argument("--audio-lowpass", type=int, default=11000)
    parser.add_argument("--audio-denoise", action="store_true")
    parser.add_argument("--audio-denoise-strength", default="-25")
    parser.add_argument("--audio-deesser", action="store_true")
    parser.add_argument(
        "--audio-normalize",
        default="loudnorm",
        choices=["none", "dynaudnorm", "loudnorm"],
    )
    parser.add_argument("--audio-volume", type=float, default=1.0)
    parser.add_argument("--loudnorm-i", default="-16")
    parser.add_argument("--loudnorm-tp", default="-1.5")
    parser.add_argument("--loudnorm-lra", default="11")

    # Mixing
    parser.add_argument("--mix-original-audio", action="store_true")
    parser.add_argument("--original-audio-volume", type=float, default=0.10)
    parser.add_argument("--dubbed-audio-volume", type=float, default=1.0)

    # Video export
    parser.add_argument("--final-audio-bitrate", default="192k")
    parser.add_argument("--burn-subtitles", action="store_true")
    parser.add_argument("--soft-subtitles", action="store_true")
    parser.add_argument("--video-codec", default="libx264")
    parser.add_argument("--video-preset", default="medium")
    parser.add_argument("--video-crf", type=int, default=18)

    # Resume/skip
    parser.add_argument("--skip-transcription", action="store_true")
    parser.add_argument("--skip-translation", action="store_true")
    parser.add_argument("--skip-tts", action="store_true")
    parser.add_argument("--existing-transcript-json", default=None)
    parser.add_argument("--existing-translated-json", default=None)
    parser.add_argument("--existing-tts-audio", default=None)

    # Control
    parser.add_argument("--dry-run", action="store_true")

    return parser


# ============================================================
# MAIN
# ============================================================

def print_plan(args, paths: Dict[str, Path]) -> None:
    print("\n============================================================")
    print("PHASE 8 FULL DUBBING PRODUCTION PLAN")
    print("============================================================")
    print(f"Input video             : {paths['input_video']}")
    print(f"Source language         : {args.source_language}")
    print(f"Target language         : {args.target_language}")
    print(f"Output name             : {paths['base_name']}")
    print(f"Work ID                 : {paths['work_id']}")
    print()
    print(f"ASR engine/model        : {args.asr_engine} / {args.asr_model}")
    print(f"ASR device/compute      : {args.asr_device} / {args.asr_compute_type}")
    print()
    print(f"Translation env         : {args.translation_env}")
    print(f"Translation engine      : {args.translation_engine}")
    print(f"Translation model       : {args.translation_model}")
    print(f"Source Indic code       : {args.source_indic_code}")
    print(f"Target Indic code       : {args.target_indic_code}")
    print(f"Source NLLB code        : {args.source_nllb_code}")
    print(f"Target NLLB code        : {args.target_nllb_code}")
    print()
    print(f"TTS env                 : {args.tts_env}")
    print(f"TTS engine              : {args.tts_engine}")
    print(f"TTS model               : {args.tts_model}")
    print(f"TTS script              : {args.resolved_tts_script}")
    print(f"XTTS target code        : {args.target_xtts_code}")
    print(f"Reference audio         : {args.ref_audio}")
    print(f"Reference text file     : {args.ref_text_file}")
    print()
    print(f"Mix original audio      : {args.mix_original_audio}")
    print(f"Burn subtitles          : {args.burn_subtitles}")
    print(f"Soft subtitles          : {args.soft_subtitles}")
    print()
    print(f"Final video             : {paths['final_video']}")
    print(f"Log file                : {paths['log_file']}")
    print("============================================================\n")


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    ensure_directories()
    resolve_auto_options(args)

    input_video = Path(args.input_video)
    validate_file_exists(input_video, "Input video")

    base_name = args.output_name or f"{input_video.stem}_{args.target_language}_{now_id()}"
    work_id = args.work_id or base_name

    job_dir = JOBS_DIR / work_id
    job_dir.mkdir(parents=True, exist_ok=True)

    log_file = LOG_DIR / f"{work_id}.log"
    manifest_path = job_dir / "job_manifest.json"

    extracted_audio = INPUT_AUDIO_DIR / f"{base_name}_extracted_{args.extract_sample_rate}hz.wav"

    transcript_json = TRANSCRIPT_DIR / f"{base_name}_transcript.json"
    transcript_srt = TRANSCRIPT_DIR / f"{base_name}_transcript.srt"
    transcript_txt = TRANSCRIPT_DIR / f"{base_name}_transcript.txt"

    translated_json = TRANSLATION_DIR / f"{base_name}_{args.target_language}_segments.json"
    translated_srt = TRANSLATION_DIR / f"{base_name}_{args.target_language}.srt"
    translated_txt = TRANSLATION_DIR / f"{base_name}_{args.target_language}.txt"

    tts_output_name = f"{base_name}_{args.target_language}_dubbed_track"

    polished_audio = TTS_DIR / f"{base_name}_{args.target_language}_dubbed_polished.wav"
    original_low_audio = TTS_DIR / f"{base_name}_original_low.wav"
    mixed_audio = TTS_DIR / f"{base_name}_{args.target_language}_mixed_final.wav"

    final_video = FINAL_VIDEO_DIR / f"{base_name}_{args.target_language}_dubbed.mp4"
    final_video_softsubs = FINAL_VIDEO_DIR / f"{base_name}_{args.target_language}_dubbed_softsubs.mp4"

    paths = {
        "input_video": input_video,
        "base_name": Path(base_name),
        "work_id": Path(work_id),
        "job_dir": job_dir,
        "log_file": log_file,
        "manifest_path": manifest_path,
        "extracted_audio": extracted_audio,
        "transcript_json": transcript_json,
        "transcript_srt": transcript_srt,
        "transcript_txt": transcript_txt,
        "translated_json": translated_json,
        "translated_srt": translated_srt,
        "translated_txt": translated_txt,
        "polished_audio": polished_audio,
        "original_low_audio": original_low_audio,
        "mixed_audio": mixed_audio,
        "final_video": final_video,
        "final_video_softsubs": final_video_softsubs,
    }

    print_plan(args, paths)

    manifest = {
        "work_id": work_id,
        "base_name": base_name,
        "created_at": datetime.now().isoformat(),
        "input_video": str(input_video),
        "source_language": args.source_language,
        "target_language": args.target_language,
        "router": {
            "asr_engine": args.asr_engine,
            "asr_model": args.asr_model,
            "translation_env": args.translation_env,
            "translation_engine": args.translation_engine,
            "translation_model": args.translation_model,
            "tts_env": args.tts_env,
            "tts_engine": args.tts_engine,
            "tts_model": args.tts_model,
            "tts_script": str(args.resolved_tts_script),
            "source_indic_code": args.source_indic_code,
            "target_indic_code": args.target_indic_code,
            "source_nllb_code": args.source_nllb_code,
            "target_nllb_code": args.target_nllb_code,
            "target_xtts_code": args.target_xtts_code,
        },
        "paths": {key: str(value) for key, value in paths.items()},
        "settings": {
            "audio_denoise": args.audio_denoise,
            "audio_normalize": args.audio_normalize,
            "mix_original_audio": args.mix_original_audio,
            "burn_subtitles": args.burn_subtitles,
            "soft_subtitles": args.soft_subtitles,
        },
    }

    write_json(manifest_path, manifest)

    if args.dry_run:
        print("Dry run complete. No processing was executed.")
        print(f"Manifest saved: {manifest_path}")
        return

    # --------------------------------------------------------
    # Stage 1: Extract audio + transcribe
    # --------------------------------------------------------

    if args.skip_transcription:
        if not args.existing_transcript_json:
            raise ValueError("--skip-transcription requires --existing-transcript-json")

        transcript_json = Path(args.existing_transcript_json)
        validate_file_exists(transcript_json, "Existing transcript JSON")

    else:
        extract_audio(input_video, extracted_audio, args, log_file)

        run_transcription_stage(
            audio_file=extracted_audio,
            transcript_json=transcript_json,
            transcript_srt=transcript_srt,
            transcript_txt=transcript_txt,
            args=args,
            log_file=log_file,
        )

    # --------------------------------------------------------
    # Stage 2: Translate
    # --------------------------------------------------------

    if args.skip_translation:
        if not args.existing_translated_json:
            raise ValueError("--skip-translation requires --existing-translated-json")

        translated_json = Path(args.existing_translated_json)
        validate_file_exists(translated_json, "Existing translated JSON")

    else:
        run_translation_stage(
            transcript_json=transcript_json,
            translated_json=translated_json,
            translated_srt=translated_srt,
            translated_txt=translated_txt,
            args=args,
            log_file=log_file,
        )

    # --------------------------------------------------------
    # Stage 3: TTS
    # --------------------------------------------------------

    if args.skip_tts:
        if not args.existing_tts_audio:
            raise ValueError("--skip-tts requires --existing-tts-audio")

        tts_audio = Path(args.existing_tts_audio)
        validate_file_exists(tts_audio, "Existing TTS audio")

    else:
        tts_audio = run_tts_stage(
            translated_json=translated_json,
            output_name=tts_output_name,
            args=args,
            log_file=log_file,
        )

    # --------------------------------------------------------
    # Stage 4: Audio polish
    # --------------------------------------------------------

    polish_audio(
        input_audio=tts_audio,
        output_audio=polished_audio,
        args=args,
        log_file=log_file,
    )

    final_audio = polished_audio

    # --------------------------------------------------------
    # Stage 5: Optional mix with original audio
    # --------------------------------------------------------

    if args.mix_original_audio:
        extract_original_audio_low(
            input_video=input_video,
            output_audio=original_low_audio,
            args=args,
            log_file=log_file,
        )

        mix_audio(
            original_low_audio=original_low_audio,
            dubbed_audio=polished_audio,
            mixed_audio=mixed_audio,
            args=args,
            log_file=log_file,
        )

        final_audio = mixed_audio

    # --------------------------------------------------------
    # Stage 6: Mux final video
    # --------------------------------------------------------

    mux_final_video(
        input_video=input_video,
        final_audio=final_audio,
        output_video=final_video,
        subtitle_file=translated_srt,
        args=args,
        log_file=log_file,
    )

    if args.soft_subtitles and translated_srt.exists():
        create_soft_subtitle_video(
            input_video=input_video,
            final_audio=final_audio,
            subtitle_file=translated_srt,
            output_video=final_video_softsubs,
            args=args,
            log_file=log_file,
        )

    # --------------------------------------------------------
    # Final manifest update
    # --------------------------------------------------------

    manifest["completed_at"] = datetime.now().isoformat()
    manifest["durations"] = {
        "input_video_seconds": ffprobe_duration(input_video),
        "tts_audio_seconds": ffprobe_duration(tts_audio),
        "final_audio_seconds": ffprobe_duration(final_audio),
        "final_video_seconds": ffprobe_duration(final_video),
    }
    manifest["final_outputs"] = {
        "transcript_json": str(transcript_json),
        "translated_json": str(translated_json),
        "translated_srt": str(translated_srt),
        "tts_audio": str(tts_audio),
        "polished_audio": str(polished_audio),
        "final_audio": str(final_audio),
        "final_video": str(final_video),
        "soft_subtitle_video": str(final_video_softsubs) if args.soft_subtitles else None,
        "manifest": str(manifest_path),
        "log_file": str(log_file),
    }

    write_json(manifest_path, manifest)

    print("\n============================================================")
    print("PHASE 8 PRODUCTION RUN COMPLETE")
    print("============================================================")
    print(f"Final video             : {final_video}")
    print(f"Final audio             : {final_audio}")
    print(f"Translated subtitles    : {translated_srt}")
    print(f"Manifest                : {manifest_path}")
    print(f"Log file                : {log_file}")

    if args.soft_subtitles:
        print(f"Soft subtitle video     : {final_video_softsubs}")

    print("============================================================\n")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print("\nERROR:")
        print(exc)
        sys.exit(1)