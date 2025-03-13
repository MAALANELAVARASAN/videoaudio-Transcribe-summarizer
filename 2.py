from flask import Flask, render_template, request, send_file, jsonify
import cv2
import pyaudio
import wave
import threading
import os
import subprocess

app = Flask(__name__)

# Constants
VIDEO_FILENAME = "static/recorded_video.avi"
AUDIO_FILENAME = "static/recorded_audio.wav"
EXTRACTED_AUDIO = "static/extracted_from_video.wav"
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
FPS = 20
AUDIO_RATE = 44100
AUDIO_CHANNELS = 1
AUDIO_FORMAT = pyaudio.paInt16
CHUNK = 2048

recording = False
cap = None
out = None
frames = []
stream = None
audio = None

def record_audio():
    global recording, stream, frames
    """Records audio in a separate thread."""
    while recording:
        try:
            data = stream.read(CHUNK, exception_on_overflow=False)
            frames.append(data)
        except Exception as e:
            print(f"Audio recording error: {e}")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/start', methods=['POST'])
def start_recording():
    global recording, cap, out, stream, audio, frames
    if recording:
        return jsonify({"message": "Recording already in progress."})

    recording = True
    frames = []

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(VIDEO_FILENAME, fourcc, FPS, (FRAME_WIDTH, FRAME_HEIGHT))

    audio = pyaudio.PyAudio()
    stream = audio.open(format=AUDIO_FORMAT, channels=AUDIO_CHANNELS,
                        rate=AUDIO_RATE, input=True,
                        frames_per_buffer=CHUNK)

    audio_thread = threading.Thread(target=record_audio)
    audio_thread.start()

    while recording:
        ret, frame = cap.read()
        if not ret:
            break
        out.write(frame)
        cv2.imshow('Recording', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    return jsonify({"message": "Recording started."})

@app.route('/stop', methods=['POST'])
def stop_recording():
    global recording, cap, out, stream, audio
    if not recording:
        return jsonify({"message": "No active recording."})

    recording = False
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    
    stream.stop_stream()
    stream.close()
    audio.terminate()

    with wave.open(AUDIO_FILENAME, 'wb') as wf:
        wf.setnchannels(AUDIO_CHANNELS)
        wf.setsampwidth(audio.get_sample_size(AUDIO_FORMAT))
        wf.setframerate(AUDIO_RATE)
        wf.writeframes(b''.join(frames))

    # Extract audio from video using FFmpeg
    ffmpeg_cmd = ["ffmpeg", "-i", VIDEO_FILENAME, "-q:a", "0", "-map", "a", EXTRACTED_AUDIO, "-y"]
    try:
        subprocess.run(ffmpeg_cmd, check=True)
        os.remove(VIDEO_FILENAME)
    except Exception as e:
        print(f"Error extracting audio: {e}")

    return jsonify({"message": "Recording stopped and audio extracted."})

@app.route('/download_audio')
def download_audio():
    return send_file(EXTRACTED_AUDIO, as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)
