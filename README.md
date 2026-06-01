# indicftts
# Local AI Audio Translation and Dubbing Pipeline

This repository contains automation scripts for setting up and running a local AI-based audio/video translation and dubbing pipeline inside **WSL2 Ubuntu**.

The goal of this project is to take an input video, transcribe the speech, translate it into a target language, generate translated speech, improve the audio output, and create a final dubbed video.

The pipeline supports both:

* Indian language workflows using Indic models
* International language workflows using multilingual translation and TTS models

---

## What This Repository Contains

This repository contains only the required setup and production scripts.

Recommended script structure:

```text
scripts/
├── 00_wsl2_translate_setup.sh
├── 03_transcribe_any_language.py
├── 04_translate_any_language.py
├── 05_tts_any_indian_language.py
├── 05_tts_international_xtts.py
└── 08_run_full_dubbing_pipeline.py
```

---

## Pipeline Overview

The complete workflow is:

```text
Input Video
→ Audio Extraction
→ Transcription
→ Translation
→ Text-to-Speech
→ Audio Cleanup / Normalization
→ Video Mixing / Export
→ Final Dubbed Video
```

Earlier this workflow required multiple manual phases:

```text
Phase 01: WSL2 and base environment setup
Phase 02: PyTorch and GPU setup
Phase 03: Audio separation and transcription
Phase 04: Translation
Phase 05: Text-to-speech generation
Phase 06: Audio/video mixing
Phase 08: Full production automation
```

This repository automates those steps using two main files:

```text
00_wsl2_translate_setup.sh
08_run_full_dubbing_pipeline.py
```

---

## Main Scripts

### 1. `00_wsl2_translate_setup.sh`

This is the **one-time machine setup script**.

Run this once on a fresh WSL2 Ubuntu machine.

It installs and configures:

* Ubuntu system dependencies
* Miniforge / Conda
* Mamba
* Conda environment paths
* Hugging Face cache paths
* FFmpeg
* PyTorch
* Faster-Whisper dependencies
* IndicTrans2 dependencies
* IndicF5 dependencies
* XTTS / international TTS dependencies
* TorchCodec
* Required folder structure

It creates separate Conda environments for each major part of the pipeline:

```text
audio-asr
indic-translate
indic-tts
multi-translate-tts
```

---

### 2. `03_transcribe_any_language.py`

This script performs speech-to-text transcription.

Supported engines include:

```text
faster-whisper
indic-conformer
whisper
```

Typical usage is through the production runner, but it can also be run directly.

Example:

```bash
conda run -n audio-asr python scripts/03_transcribe_any_language.py \
  --input-audio /mnt/d/aistudio/audio_pipeline/output/audio/demo.wav \
  --source-language english \
  --engine faster-whisper \
  --model large-v3 \
  --output-json /mnt/d/aistudio/audio_pipeline/output/transcripts/demo_transcript.json \
  --output-srt /mnt/d/aistudio/audio_pipeline/output/transcripts/demo_transcript.srt \
  --output-txt /mnt/d/aistudio/audio_pipeline/output/transcripts/demo_transcript.txt \
  --device cuda \
  --compute-type float16
```

---

### 3. `04_translate_any_language.py`

This script translates transcript segments.

Supported translation routes:

```text
Indian languages       → IndicTrans2
International languages → NLLB
```

Example Indian language translation:

```bash
conda run -n indic-translate python scripts/04_translate_any_language.py \
  --input-json /mnt/d/aistudio/audio_pipeline/output/transcripts/demo_transcript.json \
  --source-language english \
  --target-language hindi \
  --engine indictrans2 \
  --model ai4bharat/indictrans2-en-indic-1B \
  --output-json /mnt/d/aistudio/audio_pipeline/output/translations/demo_hindi_segments.json \
  --output-srt /mnt/d/aistudio/audio_pipeline/output/translations/demo_hindi.srt \
  --output-txt /mnt/d/aistudio/audio_pipeline/output/translations/demo_hindi.txt
```

Example international translation:

```bash
conda run -n multi-translate-tts python scripts/04_translate_any_language.py \
  --input-json /mnt/d/aistudio/audio_pipeline/output/transcripts/demo_transcript.json \
  --source-language english \
  --target-language spanish \
  --engine nllb \
  --model facebook/nllb-200-distilled-600M \
  --output-json /mnt/d/aistudio/audio_pipeline/output/translations/demo_spanish_segments.json \
  --output-srt /mnt/d/aistudio/audio_pipeline/output/translations/demo_spanish.srt \
  --output-txt /mnt/d/aistudio/audio_pipeline/output/translations/demo_spanish.txt
```

---

### 4. `05_tts_any_indian_language.py`

This script generates Indian-language speech using Indic TTS models such as IndicF5.

Supported Indian languages include:

```text
Hindi
Kannada
Tamil
Telugu
Malayalam
Marathi
Bengali
Gujarati
Punjabi
Odia
Assamese
```

Example:

```bash
conda run -n indic-tts python scripts/05_tts_any_indian_language.py \
  --language hindi \
  --input-json /mnt/d/aistudio/audio_pipeline/output/translations/demo_hindi_segments.json \
  --output-dir /mnt/d/aistudio/audio_pipeline/output/tts \
  --model ai4bharat/IndicF5 \
  --ref-audio /mnt/d/aistudio/audio_pipeline/input/prompts/male_hindi_ref_24k_mono.wav \
  --ref-text-file /mnt/d/aistudio/audio_pipeline/input/prompts/male_hindi_ref.txt \
  --output-name demo_hindi_dubbed_track \
  --export-sample-rate 48000 \
  --audio-channels 1 \
  --export-mp3
```

Recommended reference audio format:

```text
WAV
24 kHz
Mono
10 to 15 seconds
Clean dry voice
No music
No room echo
No background noise
```

---

### 5. `05_tts_international_xtts.py`

This script generates international-language speech using XTTS-v2.

Supported languages commonly include:

```text
English
Spanish
French
German
Italian
Portuguese
Chinese
Japanese
Korean
Arabic
Russian
Dutch
Polish
Turkish
```

Example:

```bash
conda run -n multi-translate-tts python scripts/05_tts_international_xtts.py \
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
```

Note: XTTS-v2 may require license confirmation. The setup script may configure:

```bash
COQUI_TOS_AGREED=1
```

Use XTTS-v2 only in accordance with its license terms.

---

### 6. `08_run_full_dubbing_pipeline.py`

This is the **main production runner**.

It automatically controls:

* Audio extraction
* Transcription
* Translation
* TTS generation
* Audio cleanup
* Original audio mixing
* Subtitle handling
* Final video export

It automatically selects the correct environment and model route based on the source and target language.

Indian language route:

```text
Transcription: Faster-Whisper / Indic Conformer
Translation : IndicTrans2
TTS         : IndicF5
```

International language route:

```text
Transcription: Faster-Whisper
Translation : NLLB
TTS         : XTTS-v2
```

---

## Installation on a Fresh WSL2 Machine

### Step 1: Clone the repository

Inside WSL2 Ubuntu:

```bash
cd ~
git clone <your-repository-url>
cd <your-repository-folder>
```

If your scripts are inside a `scripts` folder, move or copy them to the project location:

```bash
mkdir -p /mnt/d/aistudio/audio_pipeline/scripts
cp scripts/* /mnt/d/aistudio/audio_pipeline/scripts/
```

Or if the repository itself is the scripts folder:

```bash
mkdir -p /mnt/d/aistudio/audio_pipeline/scripts
cp * /mnt/d/aistudio/audio_pipeline/scripts/
```

---

### Step 2: Run the one-time setup script

Run without `sudo`:

```bash
chmod +x /mnt/d/aistudio/audio_pipeline/scripts/00_wsl2_translate_setup.sh
bash /mnt/d/aistudio/audio_pipeline/scripts/00_wsl2_translate_setup.sh
```

After setup finishes:

```bash
source ~/.bashrc
```

If Conda is still not visible in the same terminal:

```bash
exec bash -l
```

Check:

```bash
conda --version
mamba --version
conda env list
```

Expected Conda environments:

```text
audio-asr
indic-translate
indic-tts
multi-translate-tts
```

---

## Recommended Folder Structure

The setup script creates this working structure:

```text
/mnt/d/aistudio/audio_pipeline/
├── input/
│   ├── videos/
│   ├── audio/
│   └── prompts/
├── output/
│   ├── audio/
│   ├── transcripts/
│   ├── translations/
│   ├── tts/
│   ├── final_video/
│   └── jobs/
├── scripts/
├── config/
├── models/
└── logs/
```

On Windows, this appears as:

```text
D:\aistudio\audio_pipeline
```

---

## Hugging Face Authentication

Some models may require Hugging Face authentication.

Login once inside WSL2:

```bash
source ~/.bashrc
conda activate indic-tts
huggingface-cli login
```

Or, if your installed Hugging Face Hub version supports the newer CLI:

```bash
hf auth login
```

Then test:

```bash
huggingface-cli whoami
```

The token is shared across environments through:

```bash
HF_HOME=~/aistudio/models/huggingface
HF_HUB_CACHE=~/aistudio/models/huggingface/hub
```

Do not hardcode your Hugging Face token inside scripts.

---

## Preparing Reference Voice Audio

Reference voice files should be placed in:

```text
/mnt/d/aistudio/audio_pipeline/input/prompts/
```

Recommended format:

```text
24 kHz
Mono
WAV
Clean voice
No background music
No echo
No room noise
```

Convert any reference audio to the recommended format:

```bash
ffmpeg -y \
  -i /mnt/d/aistudio/audio_pipeline/input/prompts/original_ref.wav \
  -ar 24000 \
  -ac 1 \
  -sample_fmt s16 \
  /mnt/d/aistudio/audio_pipeline/input/prompts/clean_ref_24k_mono.wav
```

Create a matching text file with the exact words spoken in the reference audio:

```text
/mnt/d/aistudio/audio_pipeline/input/prompts/clean_ref.txt
```

Example:

```text
Hello, this is a clean reference voice sample for speech generation.
```

For IndicF5, the reference text should match the reference audio closely.

---

## Running the Full Production Pipeline

### Example 1: English to Hindi

```bash
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
```

---

### Example 2: English to Kannada

```bash
python /mnt/d/aistudio/audio_pipeline/scripts/08_run_full_dubbing_pipeline.py \
  --input-video /mnt/d/aistudio/audio_pipeline/input/videos/demo.mp4 \
  --source-language english \
  --target-language kannada \
  --ref-audio /mnt/d/aistudio/audio_pipeline/input/prompts/male_kannada_ref_24k_mono.wav \
  --ref-text-file /mnt/d/aistudio/audio_pipeline/input/prompts/male_kannada_ref.txt \
  --output-name demo_kannada \
  --audio-denoise \
  --mix-original-audio \
  --soft-subtitles
```

---

### Example 3: English to Spanish

```bash
python /mnt/d/aistudio/audio_pipeline/scripts/08_run_full_dubbing_pipeline.py \
  --input-video /mnt/d/aistudio/audio_pipeline/input/videos/demo.mp4 \
  --source-language english \
  --target-language spanish \
  --ref-audio /mnt/d/aistudio/audio_pipeline/input/prompts/spanish_ref_24k_mono.wav \
  --ref-text-file /mnt/d/aistudio/audio_pipeline/input/prompts/spanish_ref.txt \
  --output-name demo_spanish \
  --audio-denoise \
  --mix-original-audio \
  --soft-subtitles
```

---

### Example 4: English to French

```bash
python /mnt/d/aistudio/audio_pipeline/scripts/08_run_full_dubbing_pipeline.py \
  --input-video /mnt/d/aistudio/audio_pipeline/input/videos/demo.mp4 \
  --source-language english \
  --target-language french \
  --ref-audio /mnt/d/aistudio/audio_pipeline/input/prompts/french_ref_24k_mono.wav \
  --ref-text-file /mnt/d/aistudio/audio_pipeline/input/prompts/french_ref.txt \
  --output-name demo_french \
  --audio-denoise \
  --mix-original-audio \
  --soft-subtitles
```

---

## Natural Audio Mode vs Source-Timed Dubbing

The TTS scripts can be configured to generate more natural audio without aggressive time scaling.

For cleaner narration-style translated audio, avoid forced source timing.

Recommended:

```bash
--tts-extra-args "--natural-pause-seconds 0.25"
```

If source timing sync is required, use the timing sync option only when necessary:

```bash
--tts-extra-args "--sync-to-source-timing"
```

Forced time scaling can sometimes make generated speech sound hollow, shallow, or inconsistent. Natural concatenation usually sounds better for educational narration and documentary-style videos.

---

## Output Files

Final outputs are saved in:

```text
/mnt/d/aistudio/audio_pipeline/output/final_video/
```

Other generated assets are saved in:

```text
/mnt/d/aistudio/audio_pipeline/output/audio/
/mnt/d/aistudio/audio_pipeline/output/transcripts/
/mnt/d/aistudio/audio_pipeline/output/translations/
/mnt/d/aistudio/audio_pipeline/output/tts/
```

Logs are saved in:

```text
/mnt/d/aistudio/audio_pipeline/logs/
```

Job manifests are saved in:

```text
/mnt/d/aistudio/audio_pipeline/output/jobs/
```

---

## Troubleshooting

### Conda command not found

Run:

```bash
source ~/.bashrc
```

If still not found:

```bash
exec bash -l
```

Then check:

```bash
conda --version
```

---

### Hugging Face authentication required

Run:

```bash
conda activate indic-tts
huggingface-cli login
```

or:

```bash
hf auth login
```

---

### TorchCodec error

If you see:

```text
TorchCodec is required for load_with_torchcodec
```

Run:

```bash
conda activate multi-translate-tts
conda install -c conda-forge "ffmpeg<8" -y
pip install torchcodec
```

For Indic TTS:

```bash
conda activate indic-tts
pip install torchcodec
```

---

### XTTS license prompt

XTTS-v2 may ask for license confirmation.

The setup script may add:

```bash
COQUI_TOS_AGREED=1
```

Use XTTS-v2 only according to its license terms.

---

### Audio sounds hollow or like it is coming from a well

Try:

```text
1. Use a clean dry reference audio.
2. Avoid original audio mixing at first.
3. Use mono voice output.
4. Avoid aggressive time scaling.
5. Reduce or disable heavy denoise.
6. Use natural audio mode instead of source-timed sync.
```

Recommended test:

```bash
--tts-audio-channels 1 \
--final-audio-channels 1 \
--tts-extra-args "--natural-pause-seconds 0.25"
```

---

### Translation hallucination or repeated symbols

If translation output contains repeated symbols or repeated words, use the validation-enabled Phase 04 script and review the generated validation report.

For production output, manually review the translated `.srt` or `.txt` before final TTS.

---

## Recommended Workflow

For a fresh machine:

```bash
git clone <your-repository-url>
mkdir -p /mnt/d/aistudio/audio_pipeline/scripts
cp <your-repository-folder>/* /mnt/d/aistudio/audio_pipeline/scripts/

chmod +x /mnt/d/aistudio/audio_pipeline/scripts/00_wsl2_translate_setup.sh
bash /mnt/d/aistudio/audio_pipeline/scripts/00_wsl2_translate_setup.sh

source ~/.bashrc
conda env list
```

Then place:

```text
Input video      → /mnt/d/aistudio/audio_pipeline/input/videos/
Reference audio  → /mnt/d/aistudio/audio_pipeline/input/prompts/
Reference text   → /mnt/d/aistudio/audio_pipeline/input/prompts/
```

Run:

```bash
python /mnt/d/aistudio/audio_pipeline/scripts/08_run_full_dubbing_pipeline.py \
  --input-video /mnt/d/aistudio/audio_pipeline/input/videos/demo.mp4 \
  --source-language english \
  --target-language hindi \
  --ref-audio /mnt/d/aistudio/audio_pipeline/input/prompts/male_hindi_ref_24k_mono.wav \
  --ref-text-file /mnt/d/aistudio/audio_pipeline/input/prompts/male_hindi_ref.txt \
  --output-name demo_hindi \
  --soft-subtitles
```

---

## Notes

This project is intended for local AI experimentation, research, learning, and production pipeline development.

Please check the licenses of all models used in the pipeline before using them for commercial work.

Model downloads may be large and will happen during first use. Subsequent runs will reuse the cached model files.

---

## Summary

This repository turns a multi-phase AI translation and dubbing process into a practical automated workflow.

With one setup script and one production runner, a WSL2 machine can be prepared for local audio/video translation, Indian-language dubbing, international-language dubbing, and final video export.

```text
Clone repository
→ Run setup script once
→ Place video and reference voice
→ Run production command
→ Get final translated video
```

