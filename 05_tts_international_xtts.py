#!/usr/bin/env python3
"""
Phase 5 International TTS using XTTS-v2

This script is designed to be called directly or by:

08_run_full_dubbing_pipeline.py

Purpose:

Translated international-language segment JSON
→ XTTS-v2 TTS
→ aligned segment WAV files
→ full dubbed WAV track
→ optional MP3 export
→ automatically splits long text chunks to avoid XTTS 400-token limit

Supported target languages depend on XTTS-v2, commonly including:

English, Spanish, French, German, Italian, Portuguese, Polish,
Turkish, Russian, Dutch, Czech, Arabic, Chinese, Japanese, Korean,
Hungarian, Hindi

Expected command style:

python 05_tts_international_xtts.py \
  --language spanish \
  --xtts-language-code es \
  --input-json /mnt/d/aistudio/audio_pipeline/output/translations/demo_spanish_segments.json \
  --output-dir /mnt/d/aistudio/audio_pipeline/output/tts \
  --model tts_models/multilingual/multi-dataset/xtts_v2 \
  --ref-audio /mnt/d/aistudio/audio_pipeline/input/prompts/spanish_ref_24k_mono.wav \
  --ref-text-file /mnt/d/aistudio/audio_pipeline/input/prompts/spanish_ref.txt \
  --output-name demo_spanish_dubbed_track \
  --export-sample-rate 48000 \
  --audio-channels 1 \
  --export-mp3

Note:
XTTS-v2 mainly needs reference audio for voice cloning.
The --ref-text / --ref-text-file arguments are accepted for interface compatibility
with IndicF5 and Phase 8, but XTTS-v2 may not use the text directly.
"""

import argparse
import json
import subprocess
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import librosa
import numpy as np
import soundfile as sf
from tqdm import tqdm


MODEL_SAMPLE_RATE = 24000


LANGUAGE_ALIASES = {
    "english": "english",
    "en": "english",

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

    "czech": "czech",
    "cs": "czech",

    "hungarian": "hungarian",
    "hu": "hungarian",

    "hindi": "hindi",
    "hi": "hindi",
}


XTTS_LANGUAGE_CODES = {
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
    "czech": "cs",
    "hungarian": "hu",
    "hindi": "hi",
}


# ============================================================
# BASIC HELPERS
# ============================================================

def normalize_language(language: str) -> str:
    key = language.strip().lower()
    return LANGUAGE_ALIASES.get(key, key)


def resolve_xtts_language_code(language: str, xtts_language_code: Optional[str]) -> str:
    if xtts_language_code and xtts_language_code.strip():
        return xtts_language_code.strip().lower()

    normalized = normalize_language(language)
    code = XTTS_LANGUAGE_CODES.get(normalized)

    if not code:
        raise ValueError(
            f"No XTTS language code found for language: {language}. "
            "Please pass --xtts-language-code manually."
        )

    return code


def load_reference_text(ref_text: Optional[str], ref_text_file: Optional[str]) -> str:
    """
    This is kept mainly for Phase 8 interface compatibility.

    XTTS-v2 usually uses reference audio, not reference text, but keeping this
    allows common command style across Indian and international TTS scripts.
    """

    if ref_text_file:
        path = Path(ref_text_file)

        if not path.exists():
            raise FileNotFoundError(f"Reference text file not found: {path}")

        text = path.read_text(encoding="utf-8").strip()

        if not text:
            raise ValueError(f"Reference text file is empty: {path}")

        return text

    if ref_text and ref_text.strip():
        return ref_text.strip()

    return ""


def load_segments(input_json: Path) -> List[Dict]:
    if not input_json.exists():
        raise FileNotFoundError(f"Input JSON not found: {input_json}")

    with input_json.open("r", encoding="utf-8") as file:
        data = json.load(file)

    if not isinstance(data, list):
        raise ValueError("Input JSON must be a list of segment objects.")

    return data


def get_segment_text(segment: Dict) -> str:
    possible_keys = [
        "translated_text",
        "translation",
        "target_text",
        "text",
    ]

    for key in possible_keys:
        value = segment.get(key)

        if isinstance(value, str) and value.strip():
            return value.strip()

    return ""


def get_segment_start_end(segment: Dict, index: int) -> Tuple[float, float]:
    if "start" not in segment or "end" not in segment:
        raise ValueError(
            f"Segment {index} is missing 'start' or 'end'. "
            "The translated JSON must preserve timing fields from transcription."
        )

    start = float(segment["start"])
    end = float(segment["end"])

    if end <= start:
        end = start + 0.5

    return start, end


def normalize_audio(audio: np.ndarray, peak_target: float = 0.90) -> np.ndarray:
    audio = np.asarray(audio, dtype=np.float32)

    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)

    if audio.size == 0:
        return audio.astype(np.float32)

    peak = float(np.max(np.abs(audio)))

    if peak > 0:
        audio = audio / peak * peak_target

    return audio.astype(np.float32)


def safe_time_stretch(
    audio: np.ndarray,
    target_samples: int,
    max_stretch_rate: float,
) -> np.ndarray:
    if len(audio) <= target_samples:
        return audio

    stretch_rate = len(audio) / target_samples
    safe_rate = min(stretch_rate, max_stretch_rate)

    try:
        stretched = librosa.effects.time_stretch(audio, rate=safe_rate)
        return stretched.astype(np.float32)
    except Exception as error:
        print(f"Warning: time-stretch failed. Will trim instead. Error: {error}")
        return audio


def fit_to_duration(
    audio: np.ndarray,
    target_duration: float,
    sample_rate: int,
    max_stretch_rate: float,
    allow_time_stretch: bool,
) -> np.ndarray:
    target_samples = max(1, int(target_duration * sample_rate))

    if len(audio) == 0:
        return np.zeros(target_samples, dtype=np.float32)

    if allow_time_stretch and len(audio) > target_samples:
        audio = safe_time_stretch(
            audio=audio,
            target_samples=target_samples,
            max_stretch_rate=max_stretch_rate,
        )

    if len(audio) > target_samples:
        audio = audio[:target_samples]

    elif len(audio) < target_samples:
        pad = target_samples - len(audio)
        audio = np.pad(audio, (0, pad), mode="constant")

    return audio.astype(np.float32)


def apply_fade(audio: np.ndarray, sample_rate: int, fade_ms: int) -> np.ndarray:
    if fade_ms <= 0 or len(audio) == 0:
        return audio

    fade_samples = int(sample_rate * fade_ms / 1000)

    if fade_samples <= 0:
        return audio

    fade_samples = min(fade_samples, len(audio) // 2)

    if fade_samples <= 0:
        return audio

    fade_in = np.linspace(0.0, 1.0, fade_samples, dtype=np.float32)
    fade_out = np.linspace(1.0, 0.0, fade_samples, dtype=np.float32)

    audio[:fade_samples] *= fade_in
    audio[-fade_samples:] *= fade_out

    return audio.astype(np.float32)


def run_ffmpeg_export(
    input_wav: Path,
    output_audio: Path,
    export_sample_rate: int,
    audio_channels: int,
    bitrate: Optional[str] = None,
) -> None:
    output_audio.parent.mkdir(parents=True, exist_ok=True)

    command = [
        "ffmpeg",
        "-y",
        "-i",
        str(input_wav),
        "-ar",
        str(export_sample_rate),
        "-ac",
        str(audio_channels),
    ]

    if bitrate and output_audio.suffix.lower() == ".mp3":
        command.extend(["-b:a", bitrate])

    command.append(str(output_audio))

    print("\nRunning FFmpeg export:")
    print(" ".join(command))

    subprocess.run(command, check=True)


def write_manifest(manifest: List[Dict], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(manifest, file, ensure_ascii=False, indent=2)


def split_text_for_xtts(text: str, max_chars: int = 300) -> List[str]:
    """
    Split long text into smaller chunks before sending it to XTTS-v2.

    XTTS-v2 has a maximum text/token limit. We use a conservative
    character-based splitter because token count varies by language and tokenizer.

    The original segment timing is preserved later by concatenating the chunk
    audio and fitting the combined audio back to the original segment duration.
    """

    text = " ".join(text.strip().split())

    if not text:
        return []

    if max_chars <= 0 or len(text) <= max_chars:
        return [text]

    sentence_breaks = {".", "?", "!", ";", ":", ",", "।", "，", "。", "！", "？"}

    rough_chunks = []
    current = ""

    for word in text.split():
        candidate = f"{current} {word}".strip()

        if len(candidate) <= max_chars:
            current = candidate
            continue

        if current:
            rough_chunks.append(current.strip())

        current = word

    if current:
        rough_chunks.append(current.strip())

    final_chunks = []

    for chunk in rough_chunks:
        if len(chunk) <= max_chars:
            final_chunks.append(chunk)
            continue

        temp = ""
        for char in chunk:
            temp += char

            if char in sentence_breaks and len(temp) >= max_chars * 0.5:
                final_chunks.append(temp.strip())
                temp = ""
            elif len(temp) >= max_chars:
                final_chunks.append(temp.strip())
                temp = ""

        if temp.strip():
            final_chunks.append(temp.strip())

    return [chunk for chunk in final_chunks if chunk.strip()]


# ============================================================
# XTTS MODEL HELPERS
# ============================================================

def load_xtts_model(model_name: str, device: str):
    import os

    os.environ["COQUI_TOS_AGREED"] = "1"

    try:
        import torch
        from TTS.api import TTS

        # PyTorch 2.6+ safe loading fix for XTTS-v2.
        # Only do this for trusted model sources such as the official Coqui XTTS-v2 model.
        try:
            from torch.serialization import add_safe_globals
            from TTS.tts.configs.xtts_config import XttsConfig
            from TTS.tts.models.xtts import XttsAudioConfig, XttsArgs
            from TTS.config.shared_configs import BaseDatasetConfig

            add_safe_globals([
                XttsConfig,
                XttsAudioConfig,
                XttsArgs,
                BaseDatasetConfig,
            ])

            print("PyTorch safe globals added for XTTS-v2.")
        except Exception as safe_error:
            print(f"Warning: could not add all XTTS safe globals: {safe_error}")

    except ImportError as error:
        raise ImportError(
            "Coqui TTS is not installed. "
            "Use the multi-translate-tts environment and install TTS==0.22.0."
        ) from error

    print("Loading XTTS model...")
    print(f"Model: {model_name}")

    tts = TTS(model_name)

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    if device == "cuda" and torch.cuda.is_available():
        try:
            tts = tts.to("cuda")
            print("XTTS model moved to CUDA.")
        except Exception as error:
            print(f"Warning: could not move XTTS model to CUDA: {error}")
            print("Continuing on model default device.")
    else:
        print("Using CPU or model default device.")

    return tts


def generate_xtts_segment_audio(
    tts,
    text: str,
    speaker_wav: Path,
    language_code: str,
) -> np.ndarray:
    """
    Generate audio using XTTS-v2.

    Coqui TTS returns a waveform list/array when using tts.tts().
    """

    audio = tts.tts(
        text=text,
        speaker_wav=str(speaker_wav),
        language=language_code,
    )

    audio = np.asarray(audio, dtype=np.float32)

    if audio.ndim > 1:
        audio = np.squeeze(audio)

    audio = normalize_audio(audio)

    return audio.astype(np.float32)


# ============================================================
# MAIN TTS TRACK GENERATION
# ============================================================

def generate_tts_track(args) -> None:
    language = normalize_language(args.language)
    xtts_language_code = resolve_xtts_language_code(language, args.xtts_language_code)

    input_json = Path(args.input_json)
    output_dir = Path(args.output_dir)
    ref_audio_path = Path(args.ref_audio)
    reference_text = load_reference_text(args.ref_text, args.ref_text_file)

    if not ref_audio_path.exists():
        raise FileNotFoundError(f"Reference audio not found: {ref_audio_path}")

    segments = load_segments(input_json)

    if args.limit > 0:
        segments = segments[:args.limit]

    if not segments:
        raise ValueError("No segments found in input JSON.")

    output_name = args.output_name or f"{language}_dubbed_track"

    raw_wav_path = output_dir / f"{output_name}_24k.wav"
    final_wav_path = output_dir / f"{output_name}_{args.export_sample_rate // 1000}k.wav"
    final_mp3_path = output_dir / f"{output_name}_{args.export_sample_rate // 1000}k.mp3"
    manifest_path = output_dir / f"{output_name}_manifest.json"
    segment_dir = output_dir / f"{output_name}_segments"

    output_dir.mkdir(parents=True, exist_ok=True)
    segment_dir.mkdir(parents=True, exist_ok=True)

    print("\n============================================================")
    print("PHASE 5: INTERNATIONAL LANGUAGE TTS USING XTTS-V2")
    print("============================================================")
    print(f"Language              : {language}")
    print(f"XTTS language code    : {xtts_language_code}")
    print(f"Input JSON            : {input_json}")
    print(f"Output directory      : {output_dir}")
    print(f"Output name           : {output_name}")
    print(f"Model                 : {args.model}")
    print(f"Device                : {args.device}")
    print(f"Reference audio       : {ref_audio_path}")
    print(f"Reference text file   : {args.ref_text_file if args.ref_text_file else 'Not used'}")
    print(f"Reference text chars  : {len(reference_text)}")
    print(f"Model sample rate     : {MODEL_SAMPLE_RATE}")
    print(f"Export sample rate    : {args.export_sample_rate}")
    print(f"Audio channels        : {args.audio_channels}")
    print(f"Max stretch rate      : {args.max_stretch_rate}")
    print(f"Time stretch enabled  : {not args.disable_time_stretch}")
    print(f"Max XTTS chunk chars : {args.max_chars_per_xtts_chunk}")
    print(f"Chunk pause seconds  : {args.chunk_pause_seconds}")
    print("============================================================\n")

    valid_end_times = [
        float(segment["end"])
        for segment in segments
        if "end" in segment
    ]

    if not valid_end_times:
        raise ValueError("No valid segment end times found in input JSON.")

    max_end = max(valid_end_times)
    full_track_samples = int((max_end + args.tail_padding_seconds) * MODEL_SAMPLE_RATE)
    full_track = np.zeros(full_track_samples, dtype=np.float32)

    tts = load_xtts_model(
        model_name=args.model,
        device=args.device,
    )

    manifest = []

    for index, segment in enumerate(tqdm(segments, desc=f"Generating {language} XTTS")):
        start, end = get_segment_start_end(segment, index)
        duration = max(0.1, end - start)
        text = get_segment_text(segment)

        if not text:
            print(f"Skipping empty segment {index}")
            continue

        if args.max_chars_per_segment > 0 and len(text) > args.max_chars_per_segment:
            print(
                f"Warning: segment {index} text is long "
                f"({len(text)} chars). XTTS may be slow or unstable."
            )

        print("\n------------------------------------------------------------")
        print(f"Segment {index}")
        print(f"Time     : {start:.2f}s → {end:.2f}s")
        print(f"Duration : {duration:.2f}s")
        print(f"Text     : {text[:200]}")
        print("------------------------------------------------------------")

        try:
            text_chunks = split_text_for_xtts(
                text=text,
                max_chars=args.max_chars_per_xtts_chunk,
            )

            print(f"XTTS chunks: {len(text_chunks)}")

            generated_parts = []

            for chunk_index, text_chunk in enumerate(text_chunks):
                print(
                    f"  Chunk {chunk_index + 1}/{len(text_chunks)} "
                    f"({len(text_chunk)} chars): {text_chunk[:120]}"
                )

                chunk_audio = generate_xtts_segment_audio(
                    tts=tts,
                    text=text_chunk,
                    speaker_wav=ref_audio_path,
                    language_code=xtts_language_code,
                )

                generated_parts.append(chunk_audio)

                if chunk_index < len(text_chunks) - 1:
                    pause = np.zeros(int(args.chunk_pause_seconds * MODEL_SAMPLE_RATE), dtype=np.float32)
                    generated_parts.append(pause)

            if generated_parts:
                generated_audio = np.concatenate(generated_parts).astype(np.float32)
            else:
                generated_audio = np.zeros(int(duration * MODEL_SAMPLE_RATE), dtype=np.float32)

        except Exception as error:
            if args.continue_on_error:
                print(f"ERROR generating segment {index}. Inserting silence. Error: {error}")
                generated_audio = np.zeros(int(duration * MODEL_SAMPLE_RATE), dtype=np.float32)
            else:
                raise

        generated_duration = len(generated_audio) / MODEL_SAMPLE_RATE

        fitted_audio = fit_to_duration(
            audio=generated_audio,
            target_duration=duration,
            sample_rate=MODEL_SAMPLE_RATE,
            max_stretch_rate=args.max_stretch_rate,
            allow_time_stretch=not args.disable_time_stretch,
        )

        fitted_audio = apply_fade(
            audio=fitted_audio,
            sample_rate=MODEL_SAMPLE_RATE,
            fade_ms=args.fade_ms,
        )

        segment_audio_path = segment_dir / f"{language}_segment_{index:04d}.wav"
        sf.write(str(segment_audio_path), fitted_audio, MODEL_SAMPLE_RATE)

        start_sample = int(start * MODEL_SAMPLE_RATE)
        end_sample = start_sample + len(fitted_audio)

        if end_sample > len(full_track):
            full_track = np.pad(
                full_track,
                (0, end_sample - len(full_track)),
                mode="constant",
            )

        full_track[start_sample:end_sample] += fitted_audio

        manifest.append(
            {
                "index": index,
                "language": language,
                "xtts_language_code": xtts_language_code,
                "start": start,
                "end": end,
                "target_duration_seconds": duration,
                "generated_duration_seconds": generated_duration,
                "final_duration_seconds": len(fitted_audio) / MODEL_SAMPLE_RATE,
                "text": text,
                "segment_audio": str(segment_audio_path),
                "source_segment": segment,
            }
        )

    full_track = normalize_audio(full_track, peak_target=args.peak_target)

    sf.write(str(raw_wav_path), full_track, MODEL_SAMPLE_RATE)
    write_manifest(manifest, manifest_path)

    print("\nRaw model-rate WAV saved:")
    print(raw_wav_path)

    print("\nManifest saved:")
    print(manifest_path)

    run_ffmpeg_export(
        input_wav=raw_wav_path,
        output_audio=final_wav_path,
        export_sample_rate=args.export_sample_rate,
        audio_channels=args.audio_channels,
    )

    if args.export_mp3:
        run_ffmpeg_export(
            input_wav=raw_wav_path,
            output_audio=final_mp3_path,
            export_sample_rate=args.export_sample_rate,
            audio_channels=args.audio_channels,
            bitrate=args.mp3_bitrate,
        )

    if not args.keep_model_rate_wav:
        try:
            raw_wav_path.unlink()
            print(f"\nRemoved temporary raw model-rate WAV: {raw_wav_path}")
        except Exception as error:
            print(f"\nWarning: could not remove raw model-rate WAV: {error}")

    print("\n============================================================")
    print("PHASE 5 INTERNATIONAL TTS COMPLETE")
    print("============================================================")
    print(f"Final WAV : {final_wav_path}")

    if args.export_mp3:
        print(f"Final MP3 : {final_mp3_path}")

    print(f"Segments  : {segment_dir}")
    print(f"Manifest  : {manifest_path}")
    print("============================================================\n")


# ============================================================
# ARGUMENT PARSER
# ============================================================

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Phase 5: Convert translated international-language text segments into TTS audio using XTTS-v2."
    )

    parser.add_argument(
        "--language",
        required=True,
        help="Target language name/code: spanish, french, chinese, es, fr, zh, etc.",
    )

    parser.add_argument(
        "--xtts-language-code",
        default=None,
        help="XTTS language code. Example: es, fr, de, zh-cn, ja, ko. If omitted, resolved from --language.",
    )

    parser.add_argument(
        "--input-json",
        required=True,
        help="Translated segments JSON path from Phase 4.",
    )

    parser.add_argument(
        "--output-dir",
        default="/mnt/d/aistudio/audio_pipeline/output/tts",
        help="Output directory for generated TTS audio.",
    )

    parser.add_argument(
        "--output-name",
        default=None,
        help="Base output name. If omitted, language_dubbed_track will be used.",
    )

    parser.add_argument(
        "--model",
        default="tts_models/multilingual/multi-dataset/xtts_v2",
        help="Coqui TTS model name. Default: tts_models/multilingual/multi-dataset/xtts_v2.",
    )

    parser.add_argument(
        "--device",
        default="cuda",
        choices=["cuda", "cpu", "auto"],
        help="Device preference. Default: cuda.",
    )

    parser.add_argument(
        "--ref-audio",
        required=True,
        help="Reference speaker audio path. Recommended: clean 24 kHz mono WAV.",
    )

    parser.add_argument(
        "--ref-text",
        default=None,
        help="Accepted for interface compatibility. XTTS-v2 may not use this directly.",
    )

    parser.add_argument(
        "--ref-text-file",
        default=None,
        help="Accepted for interface compatibility. XTTS-v2 may not use this directly.",
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Process only first N segments for testing. Use 0 for full file.",
    )

    parser.add_argument(
        "--export-sample-rate",
        type=int,
        default=48000,
        help="Final export sample rate. Recommended: 48000 for video.",
    )

    parser.add_argument(
        "--audio-channels",
        type=int,
        default=1,
        help="Final export channels. Recommended: 1 for voice, 2 for stereo.",
    )

    parser.add_argument(
        "--export-mp3",
        action="store_true",
        help="Also export MP3.",
    )

    parser.add_argument(
        "--mp3-bitrate",
        default="192k",
        help="MP3 bitrate when --export-mp3 is enabled.",
    )

    parser.add_argument(
        "--keep-model-rate-wav",
        action="store_true",
        help="Keep the raw 24 kHz model-rate WAV.",
    )

    parser.add_argument(
        "--max-stretch-rate",
        type=float,
        default=1.25,
        help="Maximum time-stretch compression rate. Recommended: 1.15 to 1.25.",
    )

    parser.add_argument(
        "--disable-time-stretch",
        action="store_true",
        help="Disable time-stretching and only trim/pad generated speech.",
    )

    parser.add_argument(
        "--fade-ms",
        type=int,
        default=8,
        help="Small fade-in/out in milliseconds for each segment.",
    )

    parser.add_argument(
        "--tail-padding-seconds",
        type=float,
        default=1.0,
        help="Extra silence after final segment.",
    )

    parser.add_argument(
        "--peak-target",
        type=float,
        default=0.90,
        help="Peak normalization target for generated full track.",
    )

    parser.add_argument(
        "--max-chars-per-segment",
        type=int,
        default=350,
        help="Warning threshold for very long text segments. Use 0 to disable warning.",
    )

    parser.add_argument(
        "--max-chars-per-xtts-chunk",
        type=int,
        default=250,
        help="Maximum characters sent to XTTS at once. Helps avoid XTTS 400-token limit.",
    )

    parser.add_argument(
        "--chunk-pause-seconds",
        type=float,
        default=0.12,
        help="Small silence inserted between XTTS chunks from the same original segment.",
    )

    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="If a segment fails, insert silence instead of stopping.",
    )

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    generate_tts_track(args)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print("\nERROR:")
        print(error)
        sys.exit(1)