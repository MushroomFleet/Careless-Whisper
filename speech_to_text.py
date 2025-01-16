import os
import torch
import whisper
import soundfile as sf
from datetime import datetime

def check_cuda():
    """Check if CUDA is available and print device information."""
    if torch.cuda.is_available():
        device = torch.cuda.get_device_name(0)
        print(f"CUDA is available. Using GPU: {device}")
        return True
    else:
        print("WARNING: CUDA is not available. Using CPU instead.")
        print("For faster processing, please install CUDA and compatible GPU drivers.")
        return False

def load_model():
    """Load the Whisper Tiny model with CUDA if available."""
    try:
        print("Loading Whisper Tiny model...")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = whisper.load_model("tiny").to(device)
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

def transcribe_audio(model, audio_path):
    """Transcribe audio file to text using GPU acceleration when available."""
    try:
        # Load and process the audio file
        print(f"Processing audio file: {audio_path}")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Transcribe with CUDA acceleration
        result = model.transcribe(audio_path, fp16=torch.cuda.is_available())
        
        # Create output filename based on input filename
        base_name = os.path.splitext(os.path.basename(audio_path))[0]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"transcription_{base_name}_{timestamp}.txt"
        
        # Save transcription to file
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(result["text"])
        
        print(f"\nTranscription saved to: {output_file}")
        return True
    
    except Exception as e:
        print(f"Error during transcription: {e}")
        return False

def validate_audio_path(path):
    """Validate the audio file path and format."""
    if not os.path.exists(path):
        print("Error: File does not exist.")
        return False
    
    if not path.lower().endswith(('.mp3', '.wav')):
        print("Error: File must be in MP3 or WAV format.")
        return False
    
    return True

def main():
    # Print welcome message and check CUDA availability
    print("=" * 50)
    print("Whisper Tiny Speech-to-Text Converter")
    print("=" * 50)
    print("\nChecking CUDA availability...")
    check_cuda()
    
    # Load the model
    model = load_model()
    if model is None:
        print("Failed to load model. Exiting...")
        return
    
    while True:
        # Get audio file path from user
        print("\nEnter the path to your audio file (.mp3 or .wav)")
        print("Or type 'exit' to quit")
        audio_path = input("Path: ").strip('"')  # Remove quotes if present
        
        if audio_path.lower() == 'exit':
            break
        
        # Validate the audio path
        if not validate_audio_path(audio_path):
            continue
        
        # Process the audio file
        success = transcribe_audio(model, audio_path)
        
        if success:
            # Ask if user wants to process another file
            again = input("\nProcess another file? (y/n): ").lower()
            if again != 'y':
                break
        else:
            print("Failed to process audio file.")

    print("\nThank you for using Whisper Tiny Speech-to-Text Converter!")

if __name__ == "__main__":
    main()