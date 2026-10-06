"""
Core TTS engine and utility module for Qwen3-TTS on Apple Silicon using MLX.
Shared between the Gradio Web UI (app.py) and the CLI (main.py).
"""

import os
import sys
import gc
import re
import time
import shutil
import warnings
from datetime import datetime
from typing import Optional, Dict, List, Tuple, Any

import numpy as np
import soundfile as sf

# Suppress noisy library warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

try:
    import mlx.core as mx
    from mlx_audio.tts.utils import load_model
except ImportError as e:
    raise ImportError(
        f"Required MLX libraries not found: {e}\n"
        "Please run 'uv sync' to install all required dependencies."
    )

# Base Paths
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
OUTPUTS_DIR = os.path.join(PROJECT_ROOT, "outputs")
VOICES_DIR = os.path.join(PROJECT_ROOT, "voices")

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(OUTPUTS_DIR, exist_ok=True)
os.makedirs(VOICES_DIR, exist_ok=True)

# Model Registry
MODEL_REGISTRY: Dict[str, Dict[str, Any]] = {
    # 1.7B Pro Models (Best Quality)
    "qwen3-1.7b-custom": {
        "id": "qwen3-1.7b-custom",
        "display_name": "1.7B Custom Voice (Preset Speakers + Emotion)",
        "folder": "Qwen3-TTS-12Hz-1.7B-CustomVoice-8bit",
        "hf_repo": "mlx-community/Qwen3-TTS-12Hz-1.7B-CustomVoice-8bit",
        "type": "custom",
        "size": "1.7B",
        "subfolder": "CustomVoice",
        "recommended_for": "Preset voices, emotion control, speed adjustment"
    },
    "qwen3-1.7b-design": {
        "id": "qwen3-1.7b-design",
        "display_name": "1.7B Voice Design (Describe Any Voice)",
        "folder": "Qwen3-TTS-12Hz-1.7B-VoiceDesign-8bit",
        "hf_repo": "mlx-community/Qwen3-TTS-12Hz-1.7B-VoiceDesign-8bit",
        "type": "design",
        "size": "1.7B",
        "subfolder": "VoiceDesign",
        "recommended_for": "Creating novel voices via descriptive text prompts"
    },
    "qwen3-1.7b-base": {
        "id": "qwen3-1.7b-base",
        "display_name": "1.7B Voice Cloning (Clone Reference Audio)",
        "folder": "Qwen3-TTS-12Hz-1.7B-Base-8bit",
        "hf_repo": "mlx-community/Qwen3-TTS-12Hz-1.7B-Base-8bit",
        "type": "clone",
        "size": "1.7B",
        "subfolder": "Clones",
        "recommended_for": "Zero-shot voice cloning from 5-10s audio clips"
    },
    # 0.6B Lite Models (Faster, Lower Memory)
    "qwen3-0.6b-custom": {
        "id": "qwen3-0.6b-custom",
        "display_name": "0.6B Lite Custom Voice (Fast)",
        "folder": "Qwen3-TTS-12Hz-0.6B-CustomVoice-8bit",
        "hf_repo": "mlx-community/Qwen3-TTS-12Hz-0.6B-CustomVoice-8bit",
        "type": "custom",
        "size": "0.6B",
        "subfolder": "CustomVoice",
        "recommended_for": "Fast generation with preset speakers & emotion"
    },
    "qwen3-0.6b-design": {
        "id": "qwen3-0.6b-design",
        "display_name": "0.6B Lite Voice Design (Fast)",
        "folder": "Qwen3-TTS-12Hz-0.6B-VoiceDesign-8bit",
        "hf_repo": "mlx-community/Qwen3-TTS-12Hz-0.6B-VoiceDesign-8bit",
        "type": "design",
        "size": "0.6B",
        "subfolder": "VoiceDesign",
        "recommended_for": "Fast voice creation from descriptions"
    },
    "qwen3-0.6b-base": {
        "id": "qwen3-0.6b-base",
        "display_name": "0.6B Lite Voice Cloning (Fast)",
        "folder": "Qwen3-TTS-12Hz-0.6B-Base-8bit",
        "hf_repo": "mlx-community/Qwen3-TTS-12Hz-0.6B-Base-8bit",
        "type": "clone",
        "size": "0.6B",
        "subfolder": "Clones",
        "recommended_for": "Fast voice cloning with low RAM footprint"
    },
}

# Speakers catalog (Exact 9 supported speakers in Qwen3-TTS CustomVoice)
SPEAKER_CATALOG = {
    "English": [
        {"name": "Vivian", "desc": "Warm, natural female speaker (Bilingual EN/ZH)"},
        {"name": "Serena", "desc": "Clear, bright female speaker (Bilingual EN/ZH)"},
        {"name": "Ryan", "desc": "Friendly, modern male speaker"},
        {"name": "Aiden", "desc": "Youthful, energetic male speaker"},
    ],
    "Chinese": [
        {"name": "Vivian", "desc": "Warm, gentle female speaker"},
        {"name": "Serena", "desc": "Clear, friendly female speaker"},
        {"name": "Uncle_Fu", "desc": "Mature, authoritative male speaker"},
        {"name": "Dylan", "desc": "Casual, natural young male speaker"},
        {"name": "Eric", "desc": "Dynamic, resonant male speaker"},
    ],
    "Japanese": [
        {"name": "Ono_Anna", "desc": "Natural, polite Japanese female speaker"},
    ],
    "Korean": [
        {"name": "Sohee", "desc": "Smooth, gentle Korean female speaker"},
    ],
}

ALL_SPEAKERS = ["Vivian", "Serena", "Ryan", "Aiden", "Uncle_Fu", "Dylan", "Eric", "Ono_Anna", "Sohee"]

# Global cache for loaded model
_CURRENT_MODEL_KEY: Optional[str] = None
_LOADED_MODEL: Any = None


# Canonical speakers in Qwen3-TTS CustomVoice
QWEN3_CANONICAL_SPEAKERS = {
    "vivian": "Vivian",
    "serena": "Serena",
    "ryan": "Ryan",
    "aiden": "Aiden",
    "uncle_fu": "Uncle_Fu",
    "dylan": "Dylan",
    "eric": "Eric",
    "ono_anna": "Ono_Anna",
    "sohee": "Sohee",
}

# Known aliases and alternative names mapped gracefully
SPEAKER_ALIASES = {
    "ethan": "Ryan",       # Deep conversational male -> Ryan
    "chelsie": "Serena",   # Expressive female -> Serena
    "uncle fu": "Uncle_Fu",
    "unclefu": "Uncle_Fu",
    "ono anna": "Ono_Anna",
    "onoanna": "Ono_Anna",
    "anna": "Ono_Anna",
}


def resolve_speaker(voice: Optional[str], model: Any = None) -> Tuple[str, Optional[str]]:
    """
    Safely resolves any voice input to a valid speaker supported by Qwen3.
    Returns (resolved_speaker_name, note_message_or_None).
    Never throws ValueError, ensuring all inputs are handled gracefully.
    """
    if not voice or not str(voice).strip():
        return "Vivian", None

    v_clean = str(voice).strip()
    v_lower = v_clean.lower().replace("-", "_").replace(" ", "_")

    # 1. Get supported speakers from loaded model if available
    supported = []
    if model and hasattr(model, "supported_speakers") and model.supported_speakers:
        supported = [s.lower() for s in model.supported_speakers]
    elif model and hasattr(model, "config") and hasattr(model.config, "talker_config"):
        spk_id = getattr(model.config.talker_config, "spk_id", None) or {}
        supported = [s.lower() for s in spk_id.keys()]
    if not supported:
        supported = list(QWEN3_CANONICAL_SPEAKERS.keys())

    # 2. Direct case-insensitive match
    for s in supported:
        if v_lower == s or v_lower.replace("_", "") == s.replace("_", ""):
            canonical = QWEN3_CANONICAL_SPEAKERS.get(s, s.title())
            return canonical, None

    # 3. Check known aliases (e.g. ethan, chelsie, anna)
    if v_lower in SPEAKER_ALIASES:
        target = SPEAKER_ALIASES[v_lower]
        note = f"Voice '{v_clean}' mapped to Qwen3 studio voice '{target}'"
        print(f"[MLX TTS] {note}")
        return target, note

    # Check substring match
    for s in supported:
        if s in v_lower or v_lower in s:
            target = QWEN3_CANONICAL_SPEAKERS.get(s, s.title())
            note = f"Voice '{v_clean}' matched to '{target}'"
            print(f"[MLX TTS] {note}")
            return target, note

    # 4. Fallback: default to Vivian or Ryan
    fallback = "Vivian" if any(w in v_lower for w in ["female", "woman", "girl", "she", "her"]) else "Ryan"
    note = f"Voice '{v_clean}' not in Qwen3's 9 built-in studio voices. Using '{fallback}'."
    print(f"[MLX TTS] {note}")
    return fallback, note


def get_model_path_info(folder_or_name: str, hf_repo: Optional[str] = None) -> Tuple[Optional[str], bool]:
    """
    Check if a model exists locally in models/ directory or Hugging Face Hub cache.
    Returns (resolved_path, is_local_bool).
    """
    full_path = os.path.join(MODELS_DIR, folder_or_name)
    if os.path.exists(full_path):
        snapshots_dir = os.path.join(full_path, "snapshots")
        if os.path.exists(snapshots_dir):
            subfolders = [f for f in os.listdir(snapshots_dir) if not f.startswith('.')]
            if subfolders:
                return os.path.join(snapshots_dir, subfolders[0]), True
        return full_path, True

    # Check Hugging Face hub cache
    repo = hf_repo or folder_or_name
    hf_folder_name = "models--" + repo.replace("/", "--")
    hf_cache_dir = os.path.expanduser(f"~/.cache/huggingface/hub/{hf_folder_name}/snapshots")
    if os.path.exists(hf_cache_dir):
        snaps = [f for f in os.listdir(hf_cache_dir) if not f.startswith('.')]
        if snaps:
            return os.path.join(hf_cache_dir, snaps[0]), True

    return None, False


def get_available_models_info() -> List[Dict[str, Any]]:
    """Returns a list of all model definitions with download status."""
    info_list = []
    for key, spec in MODEL_REGISTRY.items():
        local_path, is_local = get_model_path_info(spec["folder"], spec.get("hf_repo"))
        info_list.append({
            "key": key,
            "display_name": spec["display_name"],
            "type": spec["type"],
            "size": spec["size"],
            "is_downloaded": is_local,
            "path": local_path or spec["hf_repo"],
            "folder": spec["folder"],
            "hf_repo": spec["hf_repo"],
            "subfolder": spec["subfolder"]
        })
    return info_list


def clean_memory():
    """Trigger Python garbage collection and clear MLX unified memory cache."""
    gc.collect()
    try:
        mx.clear_cache()
    except Exception:
        pass


def unload_model():
    """Unload the currently cached model from memory."""
    global _CURRENT_MODEL_KEY, _LOADED_MODEL
    _LOADED_MODEL = None
    _CURRENT_MODEL_KEY = None
    clean_memory()


def get_or_load_model(model_key: str, progress_callback=None) -> Any:
    """
    Load model into memory or return cached model instance.
    `model_key` can be a registry key (e.g., 'qwen3-1.7b-custom') or a folder/HF repo.
    """
    global _CURRENT_MODEL_KEY, _LOADED_MODEL

    if _CURRENT_MODEL_KEY == model_key and _LOADED_MODEL is not None:
        return _LOADED_MODEL

    # Find model spec
    target_path = None
    if model_key in MODEL_REGISTRY:
        spec = MODEL_REGISTRY[model_key]
        local_path, is_local = get_model_path_info(spec["folder"])
        target_path = local_path if is_local else spec["hf_repo"]
    else:
        # Direct path or repo
        local_path, is_local = get_model_path_info(model_key)
        target_path = local_path if is_local else model_key

    if not is_local:
        msg = f"Downloading model from Hugging Face ({target_path}). First-time download (~2.4 GB) may take a few minutes..."
        print(f"[MLX TTS] {msg}")
        if progress_callback:
            progress_callback(0.05, msg)
    elif progress_callback:
        progress_callback(0.1, f"Loading local weights from {os.path.basename(target_path)}...")

    # Free previous model before allocating new one
    unload_model()

    print(f"[MLX TTS] Loading model from: {target_path}")
    t0 = time.time()
    model = load_model(target_path)
    t1 = time.time()
    print(f"[MLX TTS] Model loaded in {t1 - t0:.2f}s")

    _LOADED_MODEL = model
    _CURRENT_MODEL_KEY = model_key
    return _LOADED_MODEL


def sanitize_filename(text: str, max_len: int = 24) -> str:
    """Generate safe filename snippet from text."""
    clean = re.sub(r'[^\w\s-]', '', text).strip().replace(' ', '_')
    return clean[:max_len] if clean else "audio"


def generate_tts(
    model_key: str,
    text: str,
    voice: Optional[str] = "Vivian",
    instruct: Optional[str] = None,
    speed: float = 1.0,
    temperature: float = 0.9,
    top_p: float = 1.0,
    top_k: int = 50,
    repetition_penalty: float = 1.05,
    max_tokens: int = 2048,
    lang_code: str = "auto",
    ref_audio: Optional[str] = None,
    ref_text: Optional[str] = None,
    output_subfolder: Optional[str] = None,
    progress_callback=None,
) -> Dict[str, Any]:
    """
    Synthesize text to speech using MLX and save output WAV.
    Returns dictionary with output file path and performance statistics.
    """
    text = (text or "").strip()
    if not text:
        raise ValueError("Input text cannot be empty.")

    if progress_callback:
        progress_callback(0.05, "Preparing model...")

    model = get_or_load_model(model_key, progress_callback=progress_callback)

    if progress_callback:
        progress_callback(0.3, "Synthesizing speech on Apple Silicon...")

    # Determine subfolder for saving output
    if not output_subfolder:
        spec = MODEL_REGISTRY.get(model_key, {})
        output_subfolder = spec.get("subfolder", "Generations")

    target_dir = os.path.join(OUTPUTS_DIR, output_subfolder)
    os.makedirs(target_dir, exist_ok=True)

    # Clean instruct: empty or blank becomes None
    inst = instruct.strip() if (instruct and instruct.strip()) else None

    # Reference audio check
    ref_audio_path = None
    if ref_audio and os.path.exists(ref_audio):
        ref_audio_path = ref_audio

    ref_t = ref_text.strip() if (ref_text and ref_text.strip()) else None

    t0 = time.time()
    gen_kwargs = {
        "text": text,
        "temperature": float(temperature),
        "speed": float(speed),
        "lang_code": str(lang_code),
        "top_p": float(top_p),
        "top_k": int(top_k),
        "repetition_penalty": float(repetition_penalty),
        "max_tokens": int(max_tokens),
    }

    speaker_note = None
    if voice:
        resolved_voice, speaker_note = resolve_speaker(voice, model)
        gen_kwargs["voice"] = resolved_voice

    if inst:
        gen_kwargs["instruct"] = inst
    if ref_audio_path:
        gen_kwargs["ref_audio"] = ref_audio_path
        if ref_t:
            gen_kwargs["ref_text"] = ref_t

    # Run generation generator
    results = list(model.generate(**gen_kwargs))
    t1 = time.time()
    proc_time = t1 - t0

    if not results:
        raise RuntimeError("Model generated no audio chunks.")

    sample_rate = getattr(results[0], "sample_rate", 24000)
    audio_arrays = [np.array(r.audio) for r in results]
    full_audio = np.concatenate(audio_arrays) if len(audio_arrays) > 1 else audio_arrays[0]

    audio_duration = len(full_audio) / float(sample_rate)
    rtf = proc_time / audio_duration if audio_duration > 0 else 0.0

    # Save to outputs
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    clean_snippet = sanitize_filename(text)
    out_filename = f"{timestamp}_{clean_snippet}.wav"
    out_path = os.path.join(target_dir, out_filename)

    sf.write(out_path, full_audio, sample_rate)

    if progress_callback:
        progress_callback(1.0, f"Done! {audio_duration:.2f}s audio generated.")

    return {
        "file_path": out_path,
        "audio_duration": audio_duration,
        "processing_time": proc_time,
        "real_time_factor": rtf,
        "sample_rate": sample_rate,
        "filename": out_filename,
        "subfolder": output_subfolder,
        "audio_array": full_audio,
        "speaker_used": gen_kwargs.get("voice"),
        "speaker_note": speaker_note,
    }


def get_saved_voices() -> List[Dict[str, str]]:
    """List all saved reference voices in the voices/ directory."""
    if not os.path.exists(VOICES_DIR):
        return []
    voices = []
    for f in sorted(os.listdir(VOICES_DIR)):
        if f.endswith(".wav"):
            name = f[:-4]
            wav_path = os.path.join(VOICES_DIR, f)
            txt_path = os.path.join(VOICES_DIR, f"{name}.txt")
            transcript = ""
            if os.path.exists(txt_path):
                try:
                    with open(txt_path, "r", encoding="utf-8") as tf:
                        transcript = tf.read().strip()
                except Exception:
                    pass
            voices.append({
                "name": name,
                "wav_path": wav_path,
                "transcript": transcript
            })
    return voices


def save_voice_to_library(name: str, audio_file_path: str, transcript: str = "") -> str:
    """Save an uploaded audio clip into the voices/ directory for reuse."""
    if not name or not name.strip():
        raise ValueError("Please provide a name for the voice.")
    if not audio_file_path or not os.path.exists(audio_file_path):
        raise ValueError("Reference audio file does not exist.")

    safe_name = sanitize_filename(name, max_len=30)
    dest_wav = os.path.join(VOICES_DIR, f"{safe_name}.wav")
    dest_txt = os.path.join(VOICES_DIR, f"{safe_name}.txt")

    # Read and convert to standard 24kHz mono wav using soundfile
    data, sr = sf.read(audio_file_path)
    if len(data.shape) > 1:
        data = data.mean(axis=1)  # Mono
    sf.write(dest_wav, data, sr)

    if transcript:
        with open(dest_txt, "w", encoding="utf-8") as f:
            f.write(transcript.strip())

    return dest_wav


def get_output_history(limit: int = 20) -> List[Dict[str, Any]]:
    """Retrieve recent generation files from outputs directory."""
    records = []
    if not os.path.exists(OUTPUTS_DIR):
        return records

    for root, _, files in os.walk(OUTPUTS_DIR):
        for f in files:
            if f.endswith(".wav"):
                full_path = os.path.join(root, f)
                try:
                    stat = os.stat(full_path)
                    subfolder = os.path.relpath(root, OUTPUTS_DIR)
                    records.append({
                        "filename": f,
                        "path": full_path,
                        "subfolder": subfolder,
                        "size_mb": round(stat.st_size / (1024 * 1024), 2),
                        "mtime": stat.st_mtime,
                        "timestamp": datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M:%S")
                    })
                except Exception:
                    pass

    records.sort(key=lambda x: x["mtime"], reverse=True)
    return records[:limit]
