from flask import Flask, render_template, request, send_file, jsonify, Response
import cv2
import pyaudio
import wave
import threading
import os
import whisper  
import warnings
from transformers import pipeline

app = Flask(__name__)

# Ensure static directory exists
os.makedirs("static", exist_ok=True)

# Constants
AUDIO_FILENAME = "static/recorded_audio.wav"
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
FPS = 20
AUDIO_RATE = 44100
AUDIO_CHANNELS = 1
AUDIO_FORMAT = pyaudio.paInt16
CHUNK = 2048

recording = False
frames = []
stream = None
audio = None

# Open webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ Error: Could not access webcam. Make sure it's not being used by another app.")
    exit(1)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
cap.set(cv2.CAP_PROP_FPS, FPS)


def record_audio():
    """Records audio in a separate thread."""
    global recording, stream, frames
    while recording:
        try:
            data = stream.read(CHUNK, exception_on_overflow=False)
            frames.append(data)
        except Exception as e:
            print(f"Audio recording error: {e}")


def generate_frames():
    cap = cv2.VideoCapture(0)  # Open the webcam
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    while True:
        success, frame = cap.read()
        if not success:
            break
        else:
            _, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()  # Ensure the webcam is released properly


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')



@app.route('/start', methods=['POST'])
def start_recording():
    """Starts audio recording"""
    global recording, stream, audio, frames
    if recording:
        return jsonify({"message": "Recording already in progress."})

    recording = True
    frames = []

    audio = pyaudio.PyAudio()
    stream = audio.open(format=AUDIO_FORMAT, channels=AUDIO_CHANNELS,
                        rate=AUDIO_RATE, input=True,
                        frames_per_buffer=CHUNK)

    audio_thread = threading.Thread(target=record_audio)
    audio_thread.start()

    return jsonify({"message": "Recording started."})


@app.route('/stop', methods=['POST'])
def stop_recording():
    """Stops recording and saves only the audio."""
    global recording, stream, audio
    if not recording:
        return jsonify({"message": "No active recording."})

    recording = False

    if stream:
        stream.stop_stream()
        stream.close()
    if audio:
        audio.terminate()

    # Save recorded audio
    try:
        with wave.open(AUDIO_FILENAME, 'wb') as wf:
            wf.setnchannels(AUDIO_CHANNELS)
            wf.setsampwidth(audio.get_sample_size(AUDIO_FORMAT))
            wf.setframerate(AUDIO_RATE)
            wf.writeframes(b''.join(frames))
    except Exception as e:
        return jsonify({"error": f"Error saving audio: {e}"})

    return jsonify({"message": "Recording stopped and audio saved."})


@app.route('/download_audio')
def download_audio():
    """Allows users to download the recorded audio file."""
    if not os.path.exists(AUDIO_FILENAME):
        return jsonify({"error": "No recorded audio found. Record first."})
    return send_file(AUDIO_FILENAME, as_attachment=True)


@app.route('/transcribe', methods=['POST'])
def transcribe_audio():
    """Transcribes the recorded audio using OpenAI Whisper."""
    if not os.path.exists(AUDIO_FILENAME):
        return jsonify({"error": "Audio file not found. Please record first."})

    try:
        warnings.filterwarnings("ignore", category=UserWarning)  # Suppress warnings
        model = whisper.load_model("base") 
        result = model.transcribe(AUDIO_FILENAME)
        return jsonify({"transcription": result["text"]})
    except Exception as e:
        return jsonify({"error": f"Transcription failed: {e}"})


# Initialize summarization model
summarizer = pipeline("summarization")

@app.route('/summarize', methods=['POST'])
def summarize_text():
    """Summarizes transcribed text."""
    data = request.get_json()
    text = data.get("text", "")

    if not text:
        return jsonify({"error": "No transcription found. Please transcribe first."})

    try:
        summary = summarizer(text, max_length=100, min_length=30, do_sample=False)[0]['summary_text']
        return jsonify({"summary": summary})
    except Exception as e:
        return jsonify({"error": f"Summarization failed: {e}"})


if __name__ == '__main__':
    app.run(debug=True)
