# Qwen3-TTS for Mac - Run AI Text-to-Speech Locally on Apple Silicon

Run **Qwen3-TTS** text-to-speech AI locally on your MacBook or Mac desktop with Apple Silicon (M1, M2, M3, M4, Pro/Max/Ultra). No cloud, no API keys, completely offline.

Now powered by **[`uv`](https://docs.astral.sh/uv/)** for fast, reliable package management, and featuring a rich **Gradio Web UI** with full parameter controls!

**Keywords:** Qwen TTS Mac, Qwen3 TTS Apple Silicon, MLX text to speech, local TTS Mac, voice cloning Mac, AI voice generator MacBook, Gradio TTS, uv

---

## Features

- **🌐 Interactive Gradio Web UI** - Simple, intuitive web interface with live audio player, waveform visualizer, and download.
- **🎭 Custom Voices with Emotion Control** - 11 built-in speakers (English, Chinese, Japanese, Korean) with custom emotion prompting (e.g. whispering, excited, sad, serious, bedtime story).
- **🎨 Voice Design** - Create entirely new voices simply by describing them ("deep British narrator", "energetic anime character", "gritty detective").
- **🧬 Zero-Shot Voice Cloning** - Clone any voice from a 5–10 second audio file or direct microphone recording.
- **🎛️ Comprehensive Speech Parameters** - Speed multiplier (0.5x to 2.0x), Temperature, Top-P, Top-K, Repetition Penalty, Max Tokens, and Language selection.
- **⚡ 100% Local & Optimized with MLX** - Runs natively on Apple Silicon GPU and Unified Memory.
- **🚀 Managed with `uv`** - Lightning-fast installation and lockfile reproducibility.

---

## Why MLX Models?

MLX models are specifically optimized for Apple Silicon unified memory architecture:

| Metric | Standard PyTorch Model | MLX Model |
|--------|------------------------|-----------|
| **RAM Usage** | 10+ GB | 2–3 GB |
| **CPU Temperature** | 80–90°C | 40–50°C |
| **Inference Time** | 15–30s | 2–4s |

*Tested on M4 MacBook Air (fanless) with 1.7B models*

---

## Quick Start with `uv`

### 1. Prerequisites

Make sure you have `ffmpeg` and `uv` installed:

```bash
# Install ffmpeg (if not already installed)
brew install ffmpeg

# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh
# or: brew install uv
```

### 2. Clone and Setup Environment

Clone the repository and sync dependencies with `uv`:

```bash
git clone https://github.com/kapi2800/qwen3-tts-apple-silicon.git
cd qwen3-tts-apple-silicon

# Automatically creates virtualenv and installs all dependencies in seconds:
uv sync
```

### 3. Download Models

Download the model weights you want and place them in the `models/` directory:

**Pro Models (1.7B) - Best Quality**

| Model | Use Case | Download |
|-------|----------|----------|
| CustomVoice | Preset voices + emotion control | [Download](https://huggingface.co/mlx-community/Qwen3-TTS-12Hz-1.7B-CustomVoice-8bit) |
| VoiceDesign | Create voices from text description | [Download](https://huggingface.co/mlx-community/Qwen3-TTS-12Hz-1.7B-VoiceDesign-8bit) |
| Base | Voice cloning from audio clips | [Download](https://huggingface.co/mlx-community/Qwen3-TTS-12Hz-1.7B-Base-8bit) |

**Lite Models (0.6B) - Faster, Less RAM**

| Model | Use Case | Download |
|-------|----------|----------|
| CustomVoice | Preset voices + emotion control | [Download](https://huggingface.co/mlx-community/Qwen3-TTS-12Hz-0.6B-CustomVoice-8bit) |
| VoiceDesign | Create voices from text description | [Download](https://huggingface.co/mlx-community/Qwen3-TTS-12Hz-0.6B-VoiceDesign-8bit) |
| Base | Voice cloning from audio clips | [Download](https://huggingface.co/mlx-community/Qwen3-TTS-12Hz-0.6B-Base-8bit) |

Put downloaded folders in `models/`:
```
models/
├── Qwen3-TTS-12Hz-1.7B-CustomVoice-8bit/
├── Qwen3-TTS-12Hz-1.7B-VoiceDesign-8bit/
└── Qwen3-TTS-12Hz-1.7B-Base-8bit/
```

*(Note: If a model folder is not downloaded locally, `mlx_audio` can also stream/download directly from HuggingFace on first use.)*

---

## Running the Application

### Option A: Launch the Gradio Web UI (Recommended)

Start the interactive Web UI:

```bash
uv run python app.py
```

Then open your browser to **`http://127.0.0.1:7860`**.

#### Web UI Features:
1. **🎭 Custom Voice Tab**:
   - Choose from 11 preset speakers.
   - Set custom Emotion instructions or click one-touch emotion chips (Natural, Happy & Excited, Sad & Soft, Angry, Whispering, Bedtime Story, News Anchor, High Tempo).
   - Speed slider (0.5x to 2.0x) with quick buttons for 0.8x Slow, 1.0x Normal, 1.3x Fast.
   - Temperature, Top-P, Top-K, Repetition Penalty, Max Tokens sliders.
   - One-click sample texts and `.txt` file drag-and-drop.
2. **🎨 Voice Design Tab**:
   - Describe any character voice in natural language.
   - Explore preset chips (Elderly Storyteller, British Narrator, Noir Detective, Anime Heroine, Chill Podcast Host).
3. **🧬 Voice Cloning Tab**:
   - Upload any `.wav` / `.mp3` / `.m4a` file or record directly with your microphone.
   - Reuse voices saved in your local voice library.
   - Enter optional transcript for enhanced pronunciation accuracy.
   - Save new voices to library with one click.
4. **📁 Output History Tab**:
   - Browse, inspect, and listen to recent generations stored in `outputs/`.
5. **⚙️ Models & System Tab**:
   - View local download status for all 6 models.
   - One-click memory cleaner to unload model cache from Apple Silicon unified memory.

---

### Option B: Terminal CLI Mode

Run the terminal-based interactive CLI:

```bash
uv run python main.py
```

```
========================================
 Qwen3-TTS Manager
========================================

  Pro Models (1.7B - Best Quality)
  ---------------------------------
  1. Custom Voice
  2. Voice Design
  3. Voice Cloning

  Lite Models (0.6B - Faster)
  ---------------------------
  4. Custom Voice
  5. Voice Design
  6. Voice Cloning

  w. Launch Gradio Web UI
  q. Exit

Select: 
```

---

## Parameter Reference Guide

| Parameter | Recommended Range | Description |
|-----------|-------------------|-------------|
| **Emotion / Instruct** | Text prompt | Controls speaker tone, emotion, and cadence (e.g. *"Excited and happy, speaking very fast"*, *"Whispering quietly"*). |
| **Speed** | `0.5x` – `2.0x` (default `1.0x`) | Adjusts the speaking speed multiplier. |
| **Temperature** | `0.1` – `1.5` (default `0.9`) | Higher values increase expressiveness and vocal variation; lower values produce more stable speech. |
| **Top-P** | `0.1` – `1.0` (default `1.0`) | Nucleus sampling probability cutoff. |
| **Top-K** | `1` – `100` (default `50`) | Limits token selection to top K candidates. |
| **Repetition Penalty**| `1.0` – `1.5` (default `1.05`)| Prevents stuttering or repetitive speech patterns. |
| **Max Tokens** | `256` – `4096` (default `2048`) | Maximum acoustic tokens generated. |
| **Language** | `auto`, `en`, `zh`, `ja`, `ko` | Language context for pronunciation. |

---

## Speakers Catalog

- **Vivian** (Bilingual English / Chinese - Warm, natural female)
- **Serena** (Bilingual English / Chinese - Clear, cheerful female)
- **Ryan** (English - Friendly modern male)
- **Aiden** (English - Dynamic young male)
- **Uncle_Fu** (Chinese - Mature, authoritative male)
- **Dylan** (Chinese - Youthful casual male)
- **Eric** (Chinese - Energetic resonant male)
- **Ono_Anna** (Japanese - Natural polite female)
- **Sohee** (Korean - Smooth gentle female)

---

## Project Structure

```
qwen3-tts-apple-silicon/
├── pyproject.toml       # uv project dependencies & configuration
├── uv.lock              # Reproducible dependency lockfile
├── app.py               # Gradio Web UI application
├── core.py              # Shared TTS engine and audio management
├── main.py              # Interactive Terminal CLI
├── models/              # Local MLX model weights
├── outputs/             # Generated audio files (CustomVoice, VoiceDesign, Clones)
└── voices/              # Saved reference voices for cloning
```

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| `uv not found` | Run `curl -LsSf https://astral.sh/uv/install.sh \| sh` or `brew install uv` |
| `ffmpeg not found` | Run `brew install ffmpeg` |
| Port 7860 already in use | Set custom port: `PORT=7861 uv run python app.py` |
| Memory usage high after many runs | Click **"Free Unified Memory"** in the **Models & System** tab or call `core.unload_model()` |

---

## Related Projects

- [Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS) - Alibaba's original Qwen3-TTS
- [MLX Audio](https://github.com/Blaizzy/mlx-audio) - MLX framework for audio models
- [MLX Community](https://huggingface.co/mlx-community) - Pre-converted MLX models
- [uv](https://github.com/astral-sh/uv) - Fast Python package installer and resolver
