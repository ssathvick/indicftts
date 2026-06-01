#!/usr/bin/env bash
set -e

# Do not run this script with sudo.
# If you run with sudo, $HOME becomes /root and the aistudio folder will be created under /root,
# not under your normal Linux user home directory.
if [ "$EUID" -eq 0 ]; then
  echo "ERROR: Do not run this script with sudo or as root."
  echo "Run it as your normal WSL user:"
  echo "  bash /mnt/d/aistudio/audio_pipeline/scripts/00_wsl2_translate_setup.sh"
  exit 1
fi


# ============================================================
# FULL AI AUDIO PIPELINE BOOTSTRAP FOR WSL2 UBUNTU
# ============================================================
#
# Assumption:
#   - Windows base WSL2 Ubuntu is already installed.
#   - Nothing else is installed.
#
# This script installs and configures:
#   - Ubuntu dependencies
#   - Miniforge / Conda / Mamba
#   - Conda env path
#   - Conda package cache path
#   - Hugging Face cache path
#   - Project folder structure on D: drive
#   - Indian language ASR / Translation / TTS environments
#   - International language Translation / TTS environment
#
# Environments created:
#   1. audio-asr
#      For Phase 3 transcription:
#      faster-whisper, audio tools, transformers base
#
#   2. indic-translate
#      For Phase 4 Indian language translation:
#      IndicTrans2-compatible environment
#
#   3. indic-tts
#      For Phase 5 Indian language TTS:
#      IndicF5-compatible environment
#
#   4. multi-translate-tts
#      For international languages:
#      NLLB translation + XTTS-v2 multilingual TTS
#
# Recommended run:
#   bash ~/setup_audio_pipeline_wsl2.sh
#
# ============================================================


# ============================================================
# USER CONFIGURABLE PATHS
# ============================================================

# Always install under the current normal WSL user's home directory.
# Example: /home/sathvick/aistudio
AISTUDIO_HOME="$HOME/aistudio"

echo "Using Linux user home:"
echo "$HOME"

echo "AI Studio home will be:"
echo "$AISTUDIO_HOME"


MINIFORGE_DIR="$AISTUDIO_HOME/miniforge3"
CONDA_ENVS_DIR="$AISTUDIO_HOME/conda_envs"
CONDA_PKGS_DIR="$AISTUDIO_HOME/conda_pkgs"

HF_HOME_DIR="$AISTUDIO_HOME/models/huggingface"
HF_HUB_CACHE_DIR="$HF_HOME_DIR/hub"

PROJECT_ROOT="/mnt/d/aistudio/audio_pipeline"

PYTORCH_INDEX_URL="https://download.pytorch.org/whl/cu128"


# ============================================================
# LOGGING HELPERS
# ============================================================

section() {
  echo
  echo "============================================================"
  echo "$1"
  echo "============================================================"
}

run() {
  echo
  echo "------------------------------------------------------------"
  echo "Running: $*"
  echo "------------------------------------------------------------"
  "$@"
}


# ============================================================
# SAFETY CHECKS
# ============================================================

section "Checking WSL2 environment"

if grep -qi microsoft /proc/version; then
  echo "WSL environment detected."
else
  echo "WARNING: This does not look like WSL."
  echo "Continue only if you intentionally want this on Linux."
fi

if [ ! -d "/mnt/d" ]; then
  echo "ERROR: /mnt/d was not found."
  echo "Your Windows D: drive is not mounted inside WSL2."
  echo "Please confirm that D: drive exists in Windows."
  exit 1
fi


# ============================================================
# CREATE BASE FOLDERS
# ============================================================

section "Creating AI Studio folders"

# Local Linux home-side folders.
# These hold Miniforge, Conda environments, Conda package cache, and Hugging Face cache.
mkdir -p "$AISTUDIO_HOME"
mkdir -p "$MINIFORGE_DIR"
mkdir -p "$CONDA_ENVS_DIR"
mkdir -p "$CONDA_PKGS_DIR"
mkdir -p "$AISTUDIO_HOME/models"
mkdir -p "$HF_HOME_DIR"
mkdir -p "$HF_HUB_CACHE_DIR"

# Shared Windows D: project folders.
# These are visible from Windows as D:\aistudio\audio_pipeline.
mkdir -p "$PROJECT_ROOT"
mkdir -p "$PROJECT_ROOT/input/videos"
mkdir -p "$PROJECT_ROOT/input/audio"
mkdir -p "$PROJECT_ROOT/input/prompts"
mkdir -p "$PROJECT_ROOT/output/audio"
mkdir -p "$PROJECT_ROOT/output/transcripts"
mkdir -p "$PROJECT_ROOT/output/translations"
mkdir -p "$PROJECT_ROOT/output/tts"
mkdir -p "$PROJECT_ROOT/output/final_video"
mkdir -p "$PROJECT_ROOT/models"
mkdir -p "$PROJECT_ROOT/scripts"
mkdir -p "$PROJECT_ROOT/config"
mkdir -p "$PROJECT_ROOT/logs"

echo "Linux AI Studio folder created:"
ls -ld "$AISTUDIO_HOME"

echo "Miniforge target folder:"
echo "$MINIFORGE_DIR"

echo "Conda environments folder:"
echo "$CONDA_ENVS_DIR"

echo "Conda packages folder:"
echo "$CONDA_PKGS_DIR"

echo "Hugging Face cache folder:"
echo "$HF_HOME_DIR"

echo "Windows-shared project folder created:"
echo "$PROJECT_ROOT"


# ============================================================
# INSTALL UBUNTU SYSTEM PACKAGES
# ============================================================

section "Installing Ubuntu system packages"

run sudo apt update

run sudo apt install -y \
  git \
  wget \
  curl \
  build-essential \
  ffmpeg \
  libsndfile1 \
  ca-certificates \
  nano \
  unzip \
  bzip2 \
  pkg-config \
  cmake \
  python3 \
  python3-pip \
  python3-venv


# ============================================================
# INSTALL MINIFORGE
# ============================================================

section "Installing or updating Miniforge / Conda / Mamba"

cd "$AISTUDIO_HOME"

INSTALLER="$AISTUDIO_HOME/Miniforge3-Linux-x86_64.sh"

if [ -x "$MINIFORGE_DIR/bin/conda" ]; then
  echo "Miniforge/Conda already installed:"
  echo "$MINIFORGE_DIR"
  echo "Updating existing Miniforge installation with installer -u option..."
else
  echo "Miniforge/Conda executable not found at:"
  echo "$MINIFORGE_DIR/bin/conda"

  if [ -d "$MINIFORGE_DIR" ]; then
    echo "Miniforge target directory already exists but conda is missing:"
    echo "$MINIFORGE_DIR"
    echo "Installer will use -u to update/repair the existing directory."
  else
    echo "Miniforge target directory does not exist yet. Creating clean install."
  fi
fi

echo "Downloading Miniforge installer..."
curl -L -o "$INSTALLER" \
  "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh"

echo "Installing/updating Miniforge at:"
echo "$MINIFORGE_DIR"

if [ -d "$MINIFORGE_DIR" ]; then
  # -u updates/repairs an existing prefix instead of failing when the directory already exists.
  bash "$INSTALLER" -b -u -p "$MINIFORGE_DIR"
else
  bash "$INSTALLER" -b -p "$MINIFORGE_DIR"
fi

rm -f "$INSTALLER"

# Make Conda available immediately in the current shell session.
export PATH="$MINIFORGE_DIR/bin:$MINIFORGE_DIR/condabin:$PATH"
hash -r || true

# Verify Miniforge/Conda installation immediately.
if [ ! -x "$MINIFORGE_DIR/bin/conda" ]; then
  echo "ERROR: Conda was not found after Miniforge installation/update."
  echo "Expected path:"
  echo "$MINIFORGE_DIR/bin/conda"
  echo
  echo "Check these folders:"
  echo "  ls -la $AISTUDIO_HOME"
  echo "  ls -la $MINIFORGE_DIR"
  exit 1
fi

echo "Conda installed at:"
echo "$MINIFORGE_DIR/bin/conda"
"$MINIFORGE_DIR/bin/conda" --version


# ============================================================
# CONFIGURE BASHRC PATHS
# ============================================================

section "Configuring shell paths"

BASHRC="$HOME/.bashrc"

touch "$BASHRC"

add_line_if_missing() {
  local line="$1"
  grep -qxF "$line" "$BASHRC" || echo "$line" >> "$BASHRC"
}

add_line_if_missing ""
add_line_if_missing "# ============================================================"
add_line_if_missing "# AI Audio Pipeline paths"
add_line_if_missing "# ============================================================"
add_line_if_missing "export AISTUDIO_HOME=\"$AISTUDIO_HOME\""
add_line_if_missing "export PATH=\"$MINIFORGE_DIR/bin:$MINIFORGE_DIR/condabin:\$PATH\""
add_line_if_missing "export CONDA_ENVS_PATH=\"$CONDA_ENVS_DIR\""
add_line_if_missing "export CONDA_PKGS_DIRS=\"$CONDA_PKGS_DIR\""
add_line_if_missing "export HF_HOME=\"$HF_HOME_DIR\""
add_line_if_missing "export HF_HUB_CACHE=\"$HF_HUB_CACHE_DIR\""
add_line_if_missing "export COQUI_TOS_AGREED=1"
add_line_if_missing "source \"$MINIFORGE_DIR/etc/profile.d/conda.sh\""

export PATH="$MINIFORGE_DIR/bin:$MINIFORGE_DIR/condabin:$PATH"
export CONDA_ENVS_PATH="$CONDA_ENVS_DIR"
export CONDA_PKGS_DIRS="$CONDA_PKGS_DIR"
export HF_HOME="$HF_HOME_DIR"
export HF_HUB_CACHE="$HF_HUB_CACHE_DIR"
export COQUI_TOS_AGREED="1"

source "$MINIFORGE_DIR/etc/profile.d/conda.sh"

echo "Shell paths configured."


# ============================================================
# CONFIGURE CONDA
# ============================================================

section "Configuring Conda"

run conda config --set auto_activate_base false
run conda config --add envs_dirs "$CONDA_ENVS_DIR" || true
run conda config --add pkgs_dirs "$CONDA_PKGS_DIR" || true
run conda config --set channel_priority strict

# Initialize Conda for future interactive WSL terminals.
# This ensures "conda activate" works after opening a new terminal.
run conda init bash || true

if command -v mamba >/dev/null 2>&1; then
  echo "Mamba is already available."
else
  echo "Installing Mamba into base environment..."
  run conda install -n base -c conda-forge mamba -y
fi

echo
echo "Conda version:"
conda --version

echo
echo "Mamba version:"
mamba --version


# ============================================================
# HELPER FUNCTIONS
# ============================================================

env_exists() {
  conda env list | awk '{print $1}' | grep -qx "$1"
}

create_env_if_missing() {
  local env_name="$1"
  local python_version="$2"

  if env_exists "$env_name"; then
    echo "Environment already exists: $env_name"
  else
    run mamba create -n "$env_name" "python=$python_version" -y
  fi
}

pip_upgrade_base() {
  local env_name="$1"

  run conda run -n "$env_name" python -m pip install --upgrade pip setuptools wheel
}

pip_install() {
  local env_name="$1"
  shift

  run conda run -n "$env_name" python -m pip install "$@"
}

conda_install_env() {
  local env_name="$1"
  shift

  run mamba install -n "$env_name" -c conda-forge "$@" -y
}

install_torch_cuda() {
  local env_name="$1"

  run conda run -n "$env_name" python -m pip install \
    torch torchvision torchaudio \
    --index-url "$PYTORCH_INDEX_URL"
}

verify_torch() {
  local env_name="$1"

  run conda run -n "$env_name" python -c "import torch; print('Torch:', torch.__version__); print('CUDA version:', torch.version.cuda); print('CUDA available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
}

verify_transformers() {
  local env_name="$1"

  run conda run -n "$env_name" python -c "import transformers; print('Transformers:', transformers.__version__); from transformers import AutoModel; print('AutoModel import OK')"
}

run_pip_check() {
  local env_name="$1"

  echo
  echo "Running pip check for $env_name..."
  conda run -n "$env_name" python -m pip check || true
}


# ============================================================
# PHASE 3 ENVIRONMENT: AUDIO ASR
# ============================================================

section "Creating Phase 3 environment: audio-asr"

create_env_if_missing "audio-asr" "3.10"
pip_upgrade_base "audio-asr"
install_torch_cuda "audio-asr"

pip_install "audio-asr" \
  "numpy==1.26.4" \
  "scipy" \
  "soundfile" \
  "librosa" \
  "pydub" \
  "tqdm" \
  "pandas" \
  "ffmpeg-python" \
  "faster-whisper" \
  "ctranslate2" \
  "transformers==4.40.2" \
  "accelerate" \
  "sentencepiece" \
  "protobuf==4.25.8" \
  "huggingface_hub"

verify_torch "audio-asr"
verify_transformers "audio-asr"

run conda run -n "audio-asr" python -c "import faster_whisper; print('faster-whisper OK')"

run_pip_check "audio-asr"


# ============================================================
# PHASE 4 ENVIRONMENT: INDIC TRANSLATION
# ============================================================

section "Creating Phase 4 environment: indic-translate"

create_env_if_missing "indic-translate" "3.10"
pip_upgrade_base "indic-translate"
install_torch_cuda "indic-translate"

pip_install "indic-translate" \
  "numpy==1.26.4" \
  "scipy" \
  "pandas" \
  "tqdm" \
  "sentencepiece" \
  "sacremoses" \
  "indic-nlp-library" \
  "protobuf==4.25.8" \
  "huggingface_hub==0.25.2" \
  "transformers==4.40.2" \
  "accelerate" \
  "safetensors"

verify_torch "indic-translate"
verify_transformers "indic-translate"

run_pip_check "indic-translate"


# ============================================================
# PHASE 5 ENVIRONMENT: INDIC TTS
# ============================================================

section "Creating Phase 5 environment: indic-tts"

create_env_if_missing "indic-tts" "3.10"
pip_upgrade_base "indic-tts"
install_torch_cuda "indic-tts"

pip_install "indic-tts" \
  "transformers==4.49.0" \
  "accelerate==0.33.0" \
  "huggingface_hub==0.29.0" \
  "safetensors==0.4.3" \
  "soundfile==0.12.1" \
  "scipy==1.13.0" \
  "numpy==1.26.4" \
  "sentencepiece==0.2.0" \
  "protobuf==4.25.8" \
  "pydub==0.25.1" \
  "librosa" \
  "tqdm" \
  "pandas" \
  "torchcodec"

pip_install "indic-tts" \
  "git+https://github.com/AI4Bharat/IndicF5.git"

verify_torch "indic-tts"
verify_transformers "indic-tts"

run conda run -n "indic-tts" python -c "import torchcodec; print('TorchCodec OK')"

run_pip_check "indic-tts"


# ============================================================
# INTERNATIONAL LANGUAGE ENVIRONMENT:
# NLLB TRANSLATION + XTTS-V2 TTS
# ============================================================

section "Creating international environment: multi-translate-tts"

create_env_if_missing "multi-translate-tts" "3.10"
pip_upgrade_base "multi-translate-tts"
install_torch_cuda "multi-translate-tts"

pip_install "multi-translate-tts" \
  "numpy==1.26.4" \
  "scipy" \
  "pandas" \
  "tqdm" \
  "soundfile" \
  "librosa" \
  "pydub" \
  "ffmpeg-python" \
  "sentencepiece" \
  "sacremoses" \
  "protobuf==4.25.8" \
  "huggingface_hub" \
  "transformers==4.49.0" \
  "accelerate==0.33.0" \
  "safetensors==0.4.3" \
  "torchcodec"

# Keep FFmpeg below 8 inside the Conda environment for TorchCodec / TorchAudio compatibility.
conda_install_env "multi-translate-tts" "ffmpeg<8"

# Coqui TTS package for XTTS-v2.
# This provides:
#   tts_models/multilingual/multi-dataset/xtts_v2
pip_install "multi-translate-tts" \
  "TTS==0.22.0"

# Reinstall TorchCodec after Coqui TTS, in case TTS dependency resolution changed Torch/TorchAudio components.
pip_install "multi-translate-tts" \
  "torchcodec"

verify_torch "multi-translate-tts"
verify_transformers "multi-translate-tts"

run conda run -n "multi-translate-tts" python -c "import torchcodec; print('TorchCodec OK')"
run conda run -n "multi-translate-tts" python -c "from TTS.api import TTS; print('Coqui TTS / XTTS environment OK')"

run_pip_check "multi-translate-tts"


# ============================================================
# CREATE DEFAULT CONFIG FILES
# ============================================================

section "Creating default config files"

LANG_CONFIG="$PROJECT_ROOT/config/languages.json"

cat > "$LANG_CONFIG" << 'EOF'
{
  "indic_languages": {
    "hindi": {
      "indic_code": "hin_Deva",
      "short_code": "hi",
      "translation_engine": "indictrans2",
      "tts_engine": "indicf5",
      "environment": "indic-tts"
    },
    "kannada": {
      "indic_code": "kan_Knda",
      "short_code": "kn",
      "translation_engine": "indictrans2",
      "tts_engine": "indicf5",
      "environment": "indic-tts"
    },
    "tamil": {
      "indic_code": "tam_Taml",
      "short_code": "ta",
      "translation_engine": "indictrans2",
      "tts_engine": "indicf5",
      "environment": "indic-tts"
    },
    "telugu": {
      "indic_code": "tel_Telu",
      "short_code": "te",
      "translation_engine": "indictrans2",
      "tts_engine": "indicf5",
      "environment": "indic-tts"
    },
    "malayalam": {
      "indic_code": "mal_Mlym",
      "short_code": "ml",
      "translation_engine": "indictrans2",
      "tts_engine": "indicf5",
      "environment": "indic-tts"
    },
    "marathi": {
      "indic_code": "mar_Deva",
      "short_code": "mr",
      "translation_engine": "indictrans2",
      "tts_engine": "indicf5",
      "environment": "indic-tts"
    },
    "bengali": {
      "indic_code": "ben_Beng",
      "short_code": "bn",
      "translation_engine": "indictrans2",
      "tts_engine": "indicf5",
      "environment": "indic-tts"
    },
    "gujarati": {
      "indic_code": "guj_Gujr",
      "short_code": "gu",
      "translation_engine": "indictrans2",
      "tts_engine": "indicf5",
      "environment": "indic-tts"
    },
    "punjabi": {
      "indic_code": "pan_Guru",
      "short_code": "pa",
      "translation_engine": "indictrans2",
      "tts_engine": "indicf5",
      "environment": "indic-tts"
    },
    "odia": {
      "indic_code": "ory_Orya",
      "short_code": "or",
      "translation_engine": "indictrans2",
      "tts_engine": "indicf5",
      "environment": "indic-tts"
    },
    "assamese": {
      "indic_code": "asm_Beng",
      "short_code": "as",
      "translation_engine": "indictrans2",
      "tts_engine": "indicf5",
      "environment": "indic-tts"
    }
  },
  "international_languages": {
    "spanish": {
      "nllb_code": "spa_Latn",
      "xtts_code": "es",
      "translation_engine": "nllb",
      "translation_model": "facebook/nllb-200-distilled-600M",
      "tts_engine": "xtts_v2",
      "tts_model": "tts_models/multilingual/multi-dataset/xtts_v2",
      "environment": "multi-translate-tts"
    },
    "french": {
      "nllb_code": "fra_Latn",
      "xtts_code": "fr",
      "translation_engine": "nllb",
      "translation_model": "facebook/nllb-200-distilled-600M",
      "tts_engine": "xtts_v2",
      "tts_model": "tts_models/multilingual/multi-dataset/xtts_v2",
      "environment": "multi-translate-tts"
    },
    "german": {
      "nllb_code": "deu_Latn",
      "xtts_code": "de",
      "translation_engine": "nllb",
      "translation_model": "facebook/nllb-200-distilled-600M",
      "tts_engine": "xtts_v2",
      "tts_model": "tts_models/multilingual/multi-dataset/xtts_v2",
      "environment": "multi-translate-tts"
    },
    "italian": {
      "nllb_code": "ita_Latn",
      "xtts_code": "it",
      "translation_engine": "nllb",
      "translation_model": "facebook/nllb-200-distilled-600M",
      "tts_engine": "xtts_v2",
      "tts_model": "tts_models/multilingual/multi-dataset/xtts_v2",
      "environment": "multi-translate-tts"
    },
    "portuguese": {
      "nllb_code": "por_Latn",
      "xtts_code": "pt",
      "translation_engine": "nllb",
      "translation_model": "facebook/nllb-200-distilled-600M",
      "tts_engine": "xtts_v2",
      "tts_model": "tts_models/multilingual/multi-dataset/xtts_v2",
      "environment": "multi-translate-tts"
    },
    "chinese": {
      "nllb_code": "zho_Hans",
      "xtts_code": "zh-cn",
      "translation_engine": "nllb",
      "translation_model": "facebook/nllb-200-distilled-600M",
      "tts_engine": "xtts_v2",
      "tts_model": "tts_models/multilingual/multi-dataset/xtts_v2",
      "environment": "multi-translate-tts"
    },
    "japanese": {
      "nllb_code": "jpn_Jpan",
      "xtts_code": "ja",
      "translation_engine": "nllb",
      "translation_model": "facebook/nllb-200-distilled-600M",
      "tts_engine": "xtts_v2",
      "tts_model": "tts_models/multilingual/multi-dataset/xtts_v2",
      "environment": "multi-translate-tts"
    },
    "korean": {
      "nllb_code": "kor_Hang",
      "xtts_code": "ko",
      "translation_engine": "nllb",
      "translation_model": "facebook/nllb-200-distilled-600M",
      "tts_engine": "xtts_v2",
      "tts_model": "tts_models/multilingual/multi-dataset/xtts_v2",
      "environment": "multi-translate-tts"
    },
    "arabic": {
      "nllb_code": "arb_Arab",
      "xtts_code": "ar",
      "translation_engine": "nllb",
      "translation_model": "facebook/nllb-200-distilled-600M",
      "tts_engine": "xtts_v2",
      "tts_model": "tts_models/multilingual/multi-dataset/xtts_v2",
      "environment": "multi-translate-tts"
    },
    "russian": {
      "nllb_code": "rus_Cyrl",
      "xtts_code": "ru",
      "translation_engine": "nllb",
      "translation_model": "facebook/nllb-200-distilled-600M",
      "tts_engine": "xtts_v2",
      "tts_model": "tts_models/multilingual/multi-dataset/xtts_v2",
      "environment": "multi-translate-tts"
    }
  }
}
EOF

echo "Created:"
echo "$LANG_CONFIG"


VOICE_CONFIG="$PROJECT_ROOT/config/voices.json"

cat > "$VOICE_CONFIG" << 'EOF'
{
  "default": {
    "narrator": {
      "description": "Default narrator voice profile. Replace paths with your own reference audio and reference text.",
      "ref_audio": "/mnt/d/aistudio/audio_pipeline/input/prompts/narrator_ref_24k_mono.wav",
      "ref_text_file": "/mnt/d/aistudio/audio_pipeline/input/prompts/narrator_ref.txt"
    }
  },
  "hindi": {
    "narrator": {
      "ref_audio": "/mnt/d/aistudio/audio_pipeline/input/prompts/male_hindi_ref_24k_mono.wav",
      "ref_text_file": "/mnt/d/aistudio/audio_pipeline/input/prompts/male_hindi_ref.txt"
    }
  },
  "kannada": {
    "narrator": {
      "ref_audio": "/mnt/d/aistudio/audio_pipeline/input/prompts/male_kannada_ref_24k_mono.wav",
      "ref_text_file": "/mnt/d/aistudio/audio_pipeline/input/prompts/male_kannada_ref.txt"
    }
  },
  "spanish": {
    "narrator": {
      "ref_audio": "/mnt/d/aistudio/audio_pipeline/input/prompts/spanish_ref_24k_mono.wav",
      "ref_text_file": "/mnt/d/aistudio/audio_pipeline/input/prompts/spanish_ref.txt"
    }
  },
  "french": {
    "narrator": {
      "ref_audio": "/mnt/d/aistudio/audio_pipeline/input/prompts/french_ref_24k_mono.wav",
      "ref_text_file": "/mnt/d/aistudio/audio_pipeline/input/prompts/french_ref.txt"
    }
  }
}
EOF

echo "Created:"
echo "$VOICE_CONFIG"


# ============================================================
# FINAL VERIFICATION SUMMARY
# ============================================================

section "Final verification"

echo
echo "Conda environments:"
conda env list

echo
echo "Project root:"
echo "$PROJECT_ROOT"

echo
echo "Miniforge:"
echo "$MINIFORGE_DIR"

echo
echo "Conda envs path:"
echo "$CONDA_ENVS_DIR"

echo
echo "Conda packages path:"
echo "$CONDA_PKGS_DIR"

echo
echo "Hugging Face cache:"
echo "$HF_HOME_DIR"
echo
echo "Coqui XTTS license prompt bypass variable for accepted non-commercial/commercial-license use:"
echo "COQUI_TOS_AGREED=$COQUI_TOS_AGREED"

echo
echo "Language config:"
echo "$LANG_CONFIG"

echo
echo "Voice config:"
echo "$VOICE_CONFIG"


# ============================================================
# FINAL INSTRUCTIONS
# ============================================================

section "BOOTSTRAP COMPLETE"

echo "Run these commands now:"
echo
echo "source ~/.bashrc"
echo "conda --version"
echo "mamba --version"
echo

echo "If conda is still not found in the same terminal, run:"
echo
echo "exec bash -l"
echo

echo "Then check environments:"
echo
echo "conda env list"
echo

echo "Activate Phase 3 ASR:"
echo
echo "conda activate audio-asr"
echo

echo "Activate Phase 4 Indian translation:"
echo
echo "conda activate indic-translate"
echo

echo "Activate Phase 5 Indian TTS:"
echo
echo "conda activate indic-tts"
echo

echo "Activate international translation/TTS:"
echo
echo "conda activate multi-translate-tts"
echo

echo "Your fresh WSL2 machine is now ready for:"
echo
echo "1. Indian language transcription, translation, and TTS"
echo "2. International language translation and TTS"
echo "3. Future Phase 8 full pipeline orchestration"
echo

echo "Done."