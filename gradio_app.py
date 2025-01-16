import os
import json
import gradio as gr
import torch
import whisper
import numpy as np
import pyperclip
from datetime import datetime

class WhisperTranscriber:
    def __init__(self):
        self.force_cpu = self.load_saved_cpu_choice()
        self.device = "cpu" if self.force_cpu else ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        if self.device == "cuda":
            print(f"GPU: {torch.cuda.get_device_name(0)}")
        self.model = None
        self.current_model_name = self.load_saved_model_choice()
        
        self.available_models = {
            "tiny": {
                "name": "Tiny",
                "description": "Tiny model - Fastest, lowest accuracy (about 1GB)",
                "size": "~1GB"
            },
            "medium": {
                "name": "Medium",
                "description": "Medium model - Good balance of speed/accuracy (about 5GB)",
                "size": "~5GB"
            }
        }

    def load_saved_model_choice(self):
        try:
            if os.path.exists('model_config.json'):
                with open('model_config.json', 'r') as f:
                    config = json.load(f)
                    return config.get('model', 'tiny')
        except Exception as e:
            print(f"Error loading model config: {e}")
        return 'tiny'

    def load_saved_cpu_choice(self):
        try:
            if os.path.exists('model_config.json'):
                with open('model_config.json', 'r') as f:
                    config = json.load(f)
                    return config.get('force_cpu', False)
        except Exception as e:
            print(f"Error loading CPU config: {e}")
        return False

    def save_config(self):
        try:
            config = {
                'model': self.current_model_name,
                'force_cpu': self.force_cpu
            }
            with open('model_config.json', 'w') as f:
                json.dump(config, f)
        except Exception as e:
            print(f"Error saving config: {e}")

    def set_device_mode(self, force_cpu):
        if self.force_cpu != force_cpu:
            self.force_cpu = force_cpu
            self.device = "cpu" if force_cpu else ("cuda" if torch.cuda.is_available() else "cpu")
            print(f"Switching to device: {self.device}")
            
            if self.model is not None:
                self.model = None
                torch.cuda.empty_cache()
            
            self.save_config()
            return f"Switched to {self.device.upper()} mode"
        return f"Already in {self.device.upper()} mode"

    def load_model(self, model_name=None):
        if model_name is None:
            model_name = self.current_model_name
        
        try:
            if self.model is None or model_name != self.current_model_name:
                print(f"Loading Whisper {model_name} model...")
                self.model = whisper.load_model(model_name)
                
                if self.device == "cuda":
                    self.model = self.model.cuda()
                    torch.cuda.empty_cache()
                    print("Model loaded on GPU")
                else:
                    self.model = self.model.cpu()
                    print("Model loaded on CPU")
                
                self.current_model_name = model_name
                self.save_config()
            return self.model
        except Exception as e:
            print(f"Error loading model: {e}")
            return None

    def get_current_model_info(self):
        return {
            "name": self.available_models[self.current_model_name]["name"],
            "description": self.available_models[self.current_model_name]["description"],
            "status": "Loaded" if self.model is not None else "Not Loaded"
        }

    def change_model(self, model_name):
        if model_name not in self.available_models:
            return f"Error: Invalid model selection '{model_name}'"
        
        try:
            self.load_model(model_name)
            return f"Successfully switched to {self.available_models[model_name]['name']} model"
        except Exception as e:
            return f"Error loading model: {str(e)}"

    def preprocess_audio(self, audio):
        try:
            if audio is None:
                raise ValueError("No audio data received")
            
            if isinstance(audio, tuple):
                sample_rate, audio_data = audio
                
                if audio_data is None or len(audio_data) == 0:
                    raise ValueError("Empty audio data received")
                
                if isinstance(audio_data, np.ndarray):
                    audio_data = audio_data.astype(np.float32)
                    max_val = np.abs(audio_data).max()
                    if max_val > 1.0:
                        audio_data = audio_data / max_val
                    return audio_data
            
            if isinstance(audio, str):
                if not os.path.exists(audio):
                    raise ValueError(f"Audio file not found: {audio}")
                return audio
            
            raise ValueError(f"Unsupported audio format: {type(audio)}")
            
        except Exception as e:
            print(f"Error preprocessing audio: {e}")
            return None

    def transcribe_audio(self, audio):
        try:
            model = self.load_model()
            if model is None:
                return "Error: Model failed to load"

            processed_audio = self.preprocess_audio(audio)
            if processed_audio is None:
                return "Error: Failed to process audio input"
            
            transcribe_options = {
                "fp16": torch.cuda.is_available() and not self.force_cpu,
                "language": "en",
                "task": "transcribe"
            }
            
            print("Starting transcription...")
            result = model.transcribe(processed_audio, **transcribe_options)
            
            if isinstance(audio, str):
                base_name = os.path.splitext(os.path.basename(audio))[0]
            else:
                base_name = "recording"
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = f"transcription_{base_name}_{timestamp}.txt"
            
            with open(output_file, "w", encoding="utf-8") as f:
                f.write(result["text"])
            
            # Copy result to clipboard automatically
            transcription_text = result["text"]
            pyperclip.copy(transcription_text)
            print(f"Transcription saved to: {output_file}")
            print("Transcription copied to clipboard automatically")
            return transcription_text
            
        except Exception as e:
            error_msg = f"Error during transcription: {str(e)}"
            print(error_msg)
            return error_msg

def create_interface():
    transcriber = WhisperTranscriber()
    
    with gr.Blocks(title="Whisper Speech-to-Text") as interface:
        with gr.Tabs():
            with gr.Tab("Transcription"):
                gr.Markdown("# 🎙️ Whisper Speech-to-Text Converter")
                
                with gr.Row():
                    current_model = gr.Markdown(
                        f"Current Model: **{transcriber.get_current_model_info()['name']}** - "
                        f"{transcriber.get_current_model_info()['description']} "
                        f"(Running on {transcriber.device.upper()})"
                    )
                
                with gr.Row():
                    with gr.Column():
                        audio_input = gr.Audio(
                            label="Audio Input",
                            sources=["microphone", "upload"],
                            type="filepath",
                            streaming=False,
                            min_length=1,
                            max_length=300,
                            autoplay=False,
                            show_label=True,
                            elem_id="audio-input",
                            format="wav",  # Explicitly set format
                            waveform_options={"sample_rate": 16000},  # Set consistent sample rate
                            interactive=True
                        )
                        
                        with gr.Row():
                            transcribe_btn = gr.Button(
                                "🎯 Transcribe", 
                                variant="primary",
                                scale=1,
                                min_width=100
                            )
                    
                    with gr.Column():
                        text_output = gr.TextArea(
                            label="Transcription Result",
                            placeholder="Transcription will appear here...",
                            lines=10,
                            interactive=False,
                            show_copy_button=True
                        )
                        
                        # Add status message for copy operation
                        copy_status = gr.Textbox(
                            label="Status",
                            placeholder="",
                            interactive=False,
                            visible=False
                        )
                        
                        with gr.Row():
                            copy_btn = gr.Button(
                                "📋 Copy to Clipboard",
                                variant="secondary",
                                scale=1,
                                min_width=100
                            )
            
            with gr.Tab("Models"):
                gr.Markdown("## 🤖 Model Selection")
                
                with gr.Row():
                    with gr.Column():
                        model_dropdown = gr.Dropdown(
                            choices=list(transcriber.available_models.keys()),
                            value=transcriber.current_model_name,
                            label="Select Model",
                            info="Choose a model - larger models are more accurate but slower"
                        )
                        
                        cpu_toggle = gr.Checkbox(
                            value=transcriber.force_cpu,
                            label="Force CPU Mode",
                            info="Enable to force CPU usage even if CUDA is available"
                        )
                    
                    with gr.Column():
                        model_info = gr.Markdown(
                            f"""
                            ### Current Model Information
                            - **Name:** {transcriber.get_current_model_info()['name']}
                            - **Description:** {transcriber.get_current_model_info()['description']}
                            - **Status:** {transcriber.get_current_model_info()['status']}
                            - **Device:** {transcriber.device.upper()}
                            """
                        )
                
                change_model_btn = gr.Button("Apply Changes", variant="primary")
        
        with gr.Accordion("ℹ️ Information", open=False):
            gr.Markdown("""
            - This is a local-only interface (not accessible from the internet)
            - You can either record audio directly or upload an audio file
            - Supported formats: MP3 and WAV
            - Transcriptions are automatically saved to text files
            - Models will be downloaded automatically when first selected
            - CPU Mode can be forced in the Models tab if needed
            """)
        
        def safe_transcribe(audio):
            try:
                if audio is None:
                    return "No audio recorded. Please try again."
                return transcriber.transcribe_audio(audio)
            except Exception as e:
                print(f"Error in transcription: {str(e)}")
                return "Error processing audio. Please try recording again."

        def copy_to_clipboard(text):
            if text and isinstance(text, str) and text.strip():
                pyperclip.copy(text)
                return gr.update(value="✓ Text copied!", visible=True)
            return gr.update(value="⚠️ No text available to copy", visible=True)

        def apply_changes(model_name, force_cpu):
            device_message = transcriber.set_device_mode(force_cpu)
            result = transcriber.change_model(model_name)
            
            info = transcriber.get_current_model_info()
            model_info_text = f"""
            ### Current Model Information
            - **Name:** {info['name']}
            - **Description:** {info['description']}
            - **Status:** {info['status']}
            - **Device:** {transcriber.device.upper()}
            """
            
            current_model_text = (
                f"Current Model: **{info['name']}** - "
                f"{info['description']} "
                f"(Running on {transcriber.device.upper()})"
            )
            
            return f"{result}\n{device_message}", model_info_text, current_model_text
        
        # Set up event handlers
        transcribe_btn.click(
            fn=safe_transcribe,
            inputs=audio_input,
            outputs=text_output,
            api_name=False
        )
        
        copy_btn.click(
            fn=copy_to_clipboard,
            inputs=[text_output],
            outputs=copy_status
        )
        
        change_model_btn.click(
            fn=apply_changes,
            inputs=[model_dropdown, cpu_toggle],
            outputs=[
                gr.Textbox(visible=False),
                model_info,
                current_model
            ]
        )
    
    return interface

if __name__ == "__main__":
    demo = create_interface()
    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,
        inbrowser=True,
        show_api=False,
        auth=None,
        favicon_path=None
    )
