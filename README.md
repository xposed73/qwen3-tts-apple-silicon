# Qwen3-TTS (Apple Silicon)

Run Alibaba's **Qwen3-TTS** text-to-speech AI locally and offline on macOS with Apple Silicon (M1, M2, M3, M4) using Apple MLX.

Features a minimalist **Gradio Web UI** and is managed exclusively with **[`uv`](https://docs.astral.sh/uv/)**.

---

## Features

- **100% Local & Offline** - Runs natively on Apple Silicon GPU and Unified Memory via MLX. No cloud, no API keys.
- **Minimalist Gradio Web UI** - Clean, distraction-free interface with waveform player and instant playback.
- **Studio Voices with Emotion Control** - 9 built-in Qwen3 voices with natural language tone and emotion steering.
- **Voice Cloning** - Clone voices from short reference audio clips.
- **Voice Design** - Create custom voices from descriptive text prompts.
- **Managed with `uv`** - Fast, reproducible dependency management without `pip` or virtualenv headaches.

---

## Quick Start

### 1. Prerequisites

```bash
brew install ffmpeg uv
```

### 2. Clone & Setup

```bash
git clone https://github.com/xposed73/qwen3-tts-apple-silicon.git
cd qwen3-tts-apple-silicon

# Install all dependencies with uv:
uv sync
```

### 3. Download Models

Place downloaded model folders in the `models/` directory:

| Model | Description | Download |
|---|---|---|
| **CustomVoice (1.7B)** | 9 preset voices + emotion control *(Recommended)* | [Download (8-bit)](https://huggingface.co/mlx-community/Qwen3-TTS-12Hz-1.7B-CustomVoice-8bit) |
| **Base (1.7B)** | Zero-shot voice cloning from audio | [Download (8-bit)](https://huggingface.co/mlx-community/Qwen3-TTS-12Hz-1.7B-Base-8bit) |
| **VoiceDesign (1.7B)**| Generate voices from text description | [Download (8-bit)](https://huggingface.co/mlx-community/Qwen3-TTS-12Hz-1.7B-VoiceDesign-8bit) |

*(0.6B Lite models are also supported: [CustomVoice 0.6B](https://huggingface.co/mlx-community/Qwen3-TTS-12Hz-0.6B-CustomVoice-8bit) · [Base 0.6B](https://huggingface.co/mlx-community/Qwen3-TTS-12Hz-0.6B-Base-8bit) · [VoiceDesign 0.6B](https://huggingface.co/mlx-community/Qwen3-TTS-12Hz-0.6B-VoiceDesign-8bit))*

Directory structure:
```
models/
├── Qwen3-TTS-12Hz-1.7B-CustomVoice-8bit/
├── Qwen3-TTS-12Hz-1.7B-Base-8bit/
└── Qwen3-TTS-12Hz-1.7B-VoiceDesign-8bit/
```

*(If a model folder is not downloaded locally, MLX can download it automatically from Hugging Face on first use).*

---

## Launch Web UI

Start the Gradio interface:

```bash
uv run python app.py
```

Open your browser to: **`http://127.0.0.1:7860`**

---

## Built-in Voices

| Speaker | Description |
|---|---|
| **Vivian** | Female · Confident, natural, bilingual (EN / ZH) |
| **Serena** | Female · Calm, elegant, narrator (EN / ZH) |
| **Ryan** | Male · Conversational, dynamic (EN) |
| **Aiden** | Male · Clear, youthful American (EN) |
| **Uncle_Fu** | Male · Mature, warm storyteller (ZH) |
| **Dylan** | Male · Energetic, modern (ZH) |
| **Eric** | Male · Standard, corporate business (ZH) |
| **Ono_Anna** | Female · Playful, youthful (JA) |
| **Sohee** | Female · Soft, gentle (KO) |

---

## Parameters

- **Voice**: Choose from the 9 studio voices.
- **Speed**: Adjust playback speed multiplier (`0.5x` – `2.0x`, default `1.0x`).
- **Emotion / Tone**: Steer emotion and style (e.g., *"Whispering quietly"*, *"Excited and cheerful"*, *"Calm bedtime story"*).
- **Advanced Options**: Temperature, Top-P, Top-K, Repetition Penalty, Max Tokens, and Language code.

---

## Project Structure

```
qwen3-tts-apple-silicon/
├── pyproject.toml       # uv configuration & dependencies
├── uv.lock              # Reproducible lockfile
├── app.py               # Minimal Gradio Web UI
├── core.py              # Shared TTS engine & speaker resolution
├── main.py              # Terminal CLI
├── models/              # Local MLX model weights
└── outputs/             # Generated audio files
```
