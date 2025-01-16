# 🎙️ Careless Whisper (Speech-to-Paste)

A user-friendly local application for converting speech to text using OpenAI's Whisper model. This tool provides a web interface for transcribing audio from either microphone recordings or uploaded audio files.

## 🚀 Features

- 🎤 Support for both microphone recording and audio file uploads
- 💾 Automatic saving of transcriptions to text files
- 📋 Automatic clipboard copying of transcription results
- 🔄 Multiple Whisper model options:
  - Tiny (1GB) - Fastest, suitable for basic transcription
  - Medium (5GB) - Better accuracy, requires more resources
- 🖥️ Flexible processing options:
  - Automatic GPU detection and usage
  - Optional CPU-only mode
- 🌐 Local-only interface for privacy and security

## 📋 Requirements

- Python 3.8 or higher
- CUDA-compatible GPU (optional, for faster processing)
- Windows/Linux/MacOS

## 🛠️ Installation

1. Clone or download this repository to your local machine
2. Run the installation script:
   - Windows: Double-click `install.bat`
   - Linux/MacOS: Run `./install.sh`

This will install all required dependencies including:
- PyTorch with CUDA support (if available)
- OpenAI Whisper
- Gradio interface
- Other required Python packages

## 🎯 Usage

1. Start the application:
   - Windows: Double-click `run-gradio.bat`
   - Linux/MacOS: Run `./run-gradio.sh`

2. The application will open in your default web browser at `http://127.0.0.1:7860`

3. Choose your transcription method:
   - Click the microphone icon to record audio directly
   - Click the upload button to process an existing audio file (MP3 or WAV)

4. Click "🎯 Transcribe" to start the conversion process

5. The transcription will:
   - Appear in the text box
   - Be automatically copied to your clipboard
   - Be saved as a text file in the application directory

## ⚙️ Configuration

### Model Selection
- Navigate to the "Models" tab to choose between:
  - Tiny model (~1GB) - Fastest, lower accuracy
  - Medium model (~5GB) - Better accuracy, slower processing

### Processing Mode
- In the "Models" tab, you can:
  - Enable/disable CPU-only mode
  - Force CPU usage even when GPU is available

## 📝 Output Files

Transcriptions are automatically saved with the following naming convention:
- For uploaded files: `transcription_[original-filename]_[timestamp].txt`
- For recordings: `transcription_recording_[timestamp].txt`

## ⚠️ Important Notes

- First-time model downloads occur automatically when selecting a new model
- GPU mode requires a CUDA-compatible graphics card
- Maximum audio length is limited to 300 seconds (5 minutes)
- The interface is accessible only locally for security
- Supported audio formats: MP3 and WAV
- Audio is processed at 16kHz sample rate for optimal results

## 🔧 Troubleshooting

1. If the application fails to start:
   - Ensure all dependencies are installed correctly
   - Check if Python is in your system PATH
   - Verify CUDA installation if using GPU mode

2. If transcription is slow:
   - Consider switching to GPU mode if available
   - Try the Tiny model for faster processing
   - Ensure no other resource-intensive applications are running

3. For GPU-related issues:
   - Update your graphics drivers
   - Verify CUDA installation
   - Try CPU mode as a fallback
