"""
Qwen3-TTS Web UI for Apple Silicon (MLX)
Minimal, distraction-free interface powered by Gradio and uv.
"""

import os
from typing import Optional

import gradio as gr
import core

# Minimal CSS with clean monochrome styling
CUSTOM_CSS = """
.gradio-container {
    max-width: 920px !important;
    margin: 0 auto !important;
    padding-top: 28px !important;
    font-family: -apple-system, BlinkMacSystemFont, "SF Pro Text", "Segoe UI", Roboto, sans-serif !important;
}

.header-wrap {
    margin-bottom: 20px;
    padding-bottom: 14px;
    border-bottom: 1px solid var(--border-color-primary, #e4e4e7);
}

.header-title {
    font-size: 22px;
    font-weight: 600;
    letter-spacing: -0.02em;
    color: var(--body-text-color, #18181b);
    margin: 0 0 4px 0;
}

.header-sub {
    font-size: 13px;
    color: var(--body-text-color-subdued, #71717a);
    margin: 0;
}

.meta-line {
    font-size: 12.5px;
    color: var(--body-text-color-subdued, #71717a);
    margin-top: 8px;
    display: flex;
    gap: 8px;
    align-items: center;
    flex-wrap: wrap;
}

.meta-pill {
    background: var(--neutral-100, #f4f4f5);
    padding: 2px 8px;
    border-radius: 4px;
    font-size: 12px;
    font-weight: 500;
}
"""

SPEAKERS = [
    ("Vivian (Female · Natural)", "Vivian"),
    ("Serena (Female · Calm)", "Serena"),
    ("Ryan (Male · Conversational)", "Ryan"),
    ("Aiden (Male · Clear)", "Aiden"),
    ("Uncle_Fu (Male · Mature)", "Uncle_Fu"),
    ("Dylan (Male · Energetic)", "Dylan"),
    ("Eric (Male · Corporate)", "Eric"),
    ("Ono_Anna (Female · Japanese)", "Ono_Anna"),
    ("Sohee (Female · Korean)", "Sohee"),
]


def format_meta(result: dict) -> str:
    note = f"<span style='color: #a16207;'>· {result['speaker_note']}</span>" if result.get("speaker_note") else ""
    return f"""
    <div class="meta-line">
        <span class="meta-pill">{result['audio_duration']:.2f}s audio</span>
        <span>·</span>
        <span>{result['processing_time']:.2f}s latency ({result['real_time_factor']:.2f}x RTF)</span>
        <span>·</span>
        <span>Voice: <strong>{result.get('speaker_used', 'Default')}</strong></span>
        {note}
    </div>
    """


def handle_custom_voice_tts(
    text: str,
    voice: str,
    instruct: str,
    speed: float,
    temperature: float,
    top_p: float,
    top_k: int,
    repetition_penalty: float,
    max_tokens: int,
    lang_code: str,
    model_choice: str,
    progress=gr.Progress(track_tqdm=True),
):
    if not text or not text.strip():
        raise gr.Error("Please enter text to synthesize.")

    try:
        def update_progress(ratio, desc):
            progress(ratio, desc=desc)

        result = core.generate_tts(
            model_key=model_choice,
            text=text,
            voice=voice,
            instruct=instruct,
            speed=speed,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            max_tokens=max_tokens,
            lang_code=lang_code,
            output_subfolder="CustomVoice",
            progress_callback=update_progress,
        )
        return result["file_path"], format_meta(result)
    except Exception as e:
        raise gr.Error(f"Generation failed: {e}")


def handle_voice_design_tts(
    text: str,
    instruct: str,
    speed: float,
    temperature: float,
    top_p: float,
    top_k: int,
    repetition_penalty: float,
    max_tokens: int,
    lang_code: str,
    model_choice: str,
    progress=gr.Progress(track_tqdm=True),
):
    if not text or not text.strip():
        raise gr.Error("Please enter text to synthesize.")
    if not instruct or not instruct.strip():
        raise gr.Error("Please describe the voice you want to create.")

    try:
        def update_progress(ratio, desc):
            progress(ratio, desc=desc)

        result = core.generate_tts(
            model_key=model_choice,
            text=text,
            voice=None,
            instruct=instruct,
            speed=speed,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            max_tokens=max_tokens,
            lang_code=lang_code,
            output_subfolder="VoiceDesign",
            progress_callback=update_progress,
        )
        return result["file_path"], format_meta(result)
    except Exception as e:
        raise gr.Error(f"Voice Design failed: {e}")


def handle_voice_cloning_tts(
    text: str,
    ref_audio_file,
    saved_voice_choice: str,
    ref_transcript: str,
    speed: float,
    temperature: float,
    top_p: float,
    top_k: int,
    repetition_penalty: float,
    max_tokens: int,
    lang_code: str,
    model_choice: str,
    progress=gr.Progress(track_tqdm=True),
):
    if not text or not text.strip():
        raise gr.Error("Please enter text to synthesize.")

    target_audio = None
    target_transcript = ref_transcript

    if saved_voice_choice and saved_voice_choice != "None":
        for v in core.get_saved_voices():
            if v["name"] == saved_voice_choice:
                target_audio = v["wav_path"]
                if not target_transcript and v["transcript"]:
                    target_transcript = v["transcript"]
                break
    elif ref_audio_file:
        target_audio = ref_audio_file if isinstance(ref_audio_file, str) else getattr(ref_audio_file, "name", None)

    if not target_audio or not os.path.exists(target_audio):
        raise gr.Error("Please provide reference audio (upload or record).")

    try:
        def update_progress(ratio, desc):
            progress(ratio, desc=desc)

        result = core.generate_tts(
            model_key=model_choice,
            text=text,
            voice=None,
            instruct=None,
            speed=speed,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            max_tokens=max_tokens,
            lang_code=lang_code,
            ref_audio=target_audio,
            ref_text=target_transcript,
            output_subfolder="Clones",
            progress_callback=update_progress,
        )
        return result["file_path"], format_meta(result)
    except Exception as e:
        raise gr.Error(f"Voice Cloning failed: {e}")


def refresh_history():
    records = core.get_output_history(limit=15)
    if not records:
        return "<p style='color: #71717a; font-size: 13px; margin: 8px 0;'>No audio files generated yet.</p>", None

    html = "<div style='display: flex; flex-direction: column; gap: 6px; max-height: 380px; overflow-y: auto;'>"
    for r in records:
        html += f"""
        <div style='display: flex; justify-content: space-between; align-items: center; padding: 8px 12px; background: var(--neutral-50, #fafafa); border: 1px solid var(--border-color-primary, #e4e4e7); border-radius: 6px;'>
            <div>
                <span style='font-size: 13px; font-weight: 500;'>{r['filename']}</span>
                <span style='font-size: 11.5px; color: #71717a; margin-left: 8px;'>{r['subfolder']} · {r['size_mb']} MB</span>
            </div>
            <span style='font-size: 11px; color: #a1a1aa;'>{r['timestamp']}</span>
        </div>
        """
    html += "</div>"
    latest = records[0]["path"] if records else None
    return html, latest


def build_app():
    models_info = core.get_available_models_info()
    custom_models = []
    design_models = []
    clone_models = []

    for m in models_info:
        label = f"{m['display_name']} ({'Ready' if m['is_downloaded'] else 'Download required'})"
        if m["type"] == "custom":
            custom_models.append((label, m["key"]))
        elif m["type"] == "design":
            design_models.append((label, m["key"]))
        elif m["type"] == "clone":
            clone_models.append((label, m["key"]))

    saved_voices = ["None"] + [v["name"] for v in core.get_saved_voices()]

    with gr.Blocks(title="Qwen3-TTS", css=CUSTOM_CSS, theme=gr.themes.Monochrome()) as demo:
        # Minimal Header
        gr.HTML("""
        <div class="header-wrap">
            <div class="header-title">Qwen3-TTS</div>
            <div class="header-sub">Local AI speech synthesis for Apple Silicon</div>
        </div>
        """)

        with gr.Tabs():
            # TAB 1: Custom Voice (Primary)
            with gr.TabItem("Custom Voice"):
                custom_text = gr.Textbox(
                    label="Text",
                    placeholder="Enter text to synthesize...",
                    lines=3,
                    value="Hello! This is Qwen3-TTS running natively on Apple Silicon with MLX.",
                )

                with gr.Row():
                    custom_voice = gr.Dropdown(
                        label="Voice",
                        choices=SPEAKERS,
                        value="Vivian",
                        allow_custom_value=True,
                        scale=1,
                    )
                    custom_speed = gr.Slider(
                        label="Speed",
                        minimum=0.5,
                        maximum=2.0,
                        value=1.0,
                        step=0.05,
                        scale=1,
                    )
                    custom_instruct = gr.Textbox(
                        label="Emotion / Tone (Optional)",
                        placeholder="e.g. Whispering, Excited, Calm narration",
                        value="",
                        scale=1,
                    )

                # Collapsed Advanced Parameters
                with gr.Accordion("Advanced Options", open=False):
                    with gr.Row():
                        custom_temp = gr.Slider(label="Temperature", minimum=0.1, maximum=1.5, value=0.9, step=0.05)
                        custom_top_p = gr.Slider(label="Top-P", minimum=0.1, maximum=1.0, value=1.0, step=0.05)
                        custom_lang = gr.Dropdown(label="Language", choices=["auto", "en", "zh", "ja", "ko"], value="auto")
                    with gr.Row():
                        custom_top_k = gr.Slider(label="Top-K", minimum=1, maximum=100, value=50, step=1)
                        custom_rep_pen = gr.Slider(label="Repetition Penalty", minimum=1.0, maximum=1.5, value=1.05, step=0.01)
                        custom_max_tokens = gr.Slider(label="Max Tokens", minimum=256, maximum=4096, value=2048, step=64)
                    custom_model = gr.Dropdown(
                        label="Model",
                        choices=custom_models,
                        value="qwen3-1.7b-custom",
                    )

                custom_btn = gr.Button("Generate Audio", variant="primary", size="lg")

                custom_audio = gr.Audio(label="Output Audio", type="filepath", interactive=False)
                custom_meta = gr.HTML()

                custom_btn.click(
                    fn=handle_custom_voice_tts,
                    inputs=[
                        custom_text,
                        custom_voice,
                        custom_instruct,
                        custom_speed,
                        custom_temp,
                        custom_top_p,
                        custom_top_k,
                        custom_rep_pen,
                        custom_max_tokens,
                        custom_lang,
                        custom_model,
                    ],
                    outputs=[custom_audio, custom_meta],
                )

            # TAB 2: Voice Cloning
            with gr.TabItem("Voice Cloning"):
                with gr.Row():
                    with gr.Column(scale=1):
                        clone_audio_in = gr.Audio(
                            label="Reference Audio Sample (5–10s)",
                            sources=["upload", "microphone"],
                            type="filepath",
                        )
                        clone_saved_voice = gr.Dropdown(
                            label="Or Saved Voice",
                            choices=saved_voices,
                            value=saved_voices[0],
                        )
                        clone_ref_text = gr.Textbox(
                            label="Reference Transcript (Optional)",
                            placeholder="Words spoken in reference clip (improves accuracy)...",
                            lines=2,
                        )
                    with gr.Column(scale=1):
                        clone_text = gr.Textbox(
                            label="Text to Speak in Cloned Voice",
                            placeholder="Enter text...",
                            lines=4,
                            value="This audio was generated using zero-shot voice cloning on Apple Silicon.",
                        )
                        clone_speed = gr.Slider(label="Speed", minimum=0.5, maximum=2.0, value=1.0, step=0.05)

                with gr.Accordion("Advanced Options", open=False):
                    with gr.Row():
                        clone_temp = gr.Slider(label="Temperature", minimum=0.1, maximum=1.5, value=0.9, step=0.05)
                        clone_top_p = gr.Slider(label="Top-P", minimum=0.1, maximum=1.0, value=1.0, step=0.05)
                        clone_top_k = gr.Slider(label="Top-K", minimum=1, maximum=100, value=50, step=1)
                    with gr.Row():
                        clone_rep_pen = gr.Slider(label="Repetition Penalty", minimum=1.0, maximum=1.5, value=1.05, step=0.01)
                        clone_max_tokens = gr.Slider(label="Max Tokens", minimum=256, maximum=4096, value=2048, step=64)
                        clone_lang = gr.Dropdown(label="Language", choices=["auto", "en", "zh", "ja", "ko"], value="auto")
                    clone_model = gr.Dropdown(label="Model", choices=clone_models, value="qwen3-1.7b-base")

                clone_btn = gr.Button("Clone & Generate Audio", variant="primary", size="lg")
                clone_audio_out = gr.Audio(label="Output Audio", type="filepath", interactive=False)
                clone_meta = gr.HTML()

                clone_btn.click(
                    fn=handle_voice_cloning_tts,
                    inputs=[
                        clone_text,
                        clone_audio_in,
                        clone_saved_voice,
                        clone_ref_text,
                        clone_speed,
                        clone_temp,
                        clone_top_p,
                        clone_top_k,
                        clone_rep_pen,
                        clone_max_tokens,
                        clone_lang,
                        clone_model,
                    ],
                    outputs=[clone_audio_out, clone_meta],
                )

            # TAB 3: Voice Design
            with gr.TabItem("Voice Design"):
                design_instruct = gr.Textbox(
                    label="Voice Description",
                    placeholder="e.g. A calm, deep British documentary narrator with resonant cadence",
                    lines=2,
                    value="A calm British male documentary narrator with a deep resonant voice",
                )
                design_text = gr.Textbox(
                    label="Text",
                    placeholder="Enter text to speak...",
                    lines=3,
                    value="Knowledge is of no value unless you put it into practice.",
                )
                design_speed = gr.Slider(label="Speed", minimum=0.5, maximum=2.0, value=1.0, step=0.05)

                with gr.Accordion("Advanced Options", open=False):
                    with gr.Row():
                        design_temp = gr.Slider(label="Temperature", minimum=0.1, maximum=1.5, value=0.9, step=0.05)
                        design_top_p = gr.Slider(label="Top-P", minimum=0.1, maximum=1.0, value=1.0, step=0.05)
                        design_lang = gr.Dropdown(label="Language", choices=["auto", "en", "zh", "ja", "ko"], value="auto")
                    with gr.Row():
                        design_top_k = gr.Slider(label="Top-K", minimum=1, maximum=100, value=50, step=1)
                        design_rep_pen = gr.Slider(label="Repetition Penalty", minimum=1.0, maximum=1.5, value=1.05, step=0.01)
                        design_max_tokens = gr.Slider(label="Max Tokens", minimum=256, maximum=4096, value=2048, step=64)
                    design_model = gr.Dropdown(label="Model", choices=design_models, value="qwen3-1.7b-design")

                design_btn = gr.Button("Generate Designed Voice", variant="primary", size="lg")
                design_audio_out = gr.Audio(label="Output Audio", type="filepath", interactive=False)
                design_meta = gr.HTML()

                design_btn.click(
                    fn=handle_voice_design_tts,
                    inputs=[
                        design_text,
                        design_instruct,
                        design_speed,
                        design_temp,
                        design_top_p,
                        design_top_k,
                        design_rep_pen,
                        design_max_tokens,
                        design_lang,
                        design_model,
                    ],
                    outputs=[design_audio_out, design_meta],
                )

            # TAB 4: History
            with gr.TabItem("History"):
                with gr.Row():
                    hist_refresh = gr.Button("Refresh", size="sm")
                    hist_free_ram = gr.Button("Free Memory (Unload Model)", size="sm")

                hist_status = gr.Markdown("")
                hist_free_ram.click(
                    fn=lambda: (core.unload_model(), "Memory cache cleared."),
                    inputs=[],
                    outputs=[hist_status],
                )

                hist_list = gr.HTML()
                hist_player = gr.Audio(label="Play Selected", type="filepath")

                hist_refresh.click(fn=refresh_history, inputs=[], outputs=[hist_list, hist_player])
                demo.load(fn=refresh_history, inputs=[], outputs=[hist_list, hist_player])

    return demo


def main():
    demo = build_app()
    port = int(os.environ.get("PORT", 7860))
    host = os.environ.get("HOST", "127.0.0.1")
    print(f"\n* Qwen3-TTS running at http://{host}:{port}\n")
    demo.launch(server_name=host, server_port=port, share=False)


if __name__ == "__main__":
    main()
