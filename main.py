
import threading
from flask import Flask, Response, request, jsonify
import cv2
import numpy as np
import tensorflow as tf
from flask_cors import CORS
import time
from gaze_tracking import GazeTracking
from speechbrain.inference.interfaces import foreign_class
import requests
import os
import subprocess
import wave
import pyaudio
import firebase_admin
from firebase_admin import credentials, storage
from transformers import BertForSequenceClassification, BertTokenizer
import torch
import speech_recognition as sr

app = Flask(__name__)
CORS(app)

# Load the saved model
model = tf.keras.models.load_model('model/model.h5')

# Initialize Firebase Admin SDK
cred = credentials.Certificate("key.json")
firebase_admin.initialize_app(cred, {
    'storageBucket': 'interviewsimulator-252d6.appspot.com'
})

# Define the face cascade and emotions
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
emotions = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

cap = None  # Variable to hold the camera capture object
emotion_times = {emotion: 0 for emotion in emotions}

# Gaze tracking initialization
gaze = GazeTracking()

# Variables to track gaze and blink information
og_time = time.time()

# Variables to track time spent in each state
start_time = time.time()
time_left = 0
time_right = 0
time_center = 0
blink_count = 0
last_time = time.time()
result = None

# Audio settings
audio_format = pyaudio.paInt16
channels = 1  # Set the number of audio channels
audio_rate = 44100
chunk = 1024

# File paths
video_file = 'output.avi'
audio_file = 'output.wav'
output_file = 'output_with_audio.mp4'  # Output in AVI format


# Global variables for synchronization
frame_times = []
audio_times = []
stream = None
stop_flag = False
video_url = None

# Initialize PyAudio
audio_input = pyaudio.PyAudio()


def upload_to_firebase(file_path):
    global video_url
    bucket = storage.bucket()
    name = str(time.time()) + ".mp4"
    blob = bucket.blob(name)
    blob.upload_from_filename(file_path)
    # video_url = blob.public_url
    video_url = f"https://firebasestorage.googleapis.com/v0/b/{bucket.name}/o/{name}?alt=media"
    print(f"File {file_path} uploaded to Firebase Storage.")


# upload_to_firebase(file_path=output_file)


# Load the pre-trained BERT model and tokenizer
model_path = "Personality_detection_Classification_Save/"
bmodel = BertForSequenceClassification.from_pretrained(
    model_path, num_labels=5)
tokenizer = BertTokenizer.from_pretrained(model_path, do_lower_case=True)

# Define the mapping of labels to IDs and vice versa
bmodel.config.label2id = {
    "Extroversion": 0,
    "Neuroticism": 1,
    "Agreeableness": 2,
    "Conscientiousness": 3,
    "Openness": 4,
}

bmodel.config.id2label = {
    "0": "Extroversion",
    "1": "Neuroticism",
    "2": "Agreeableness",
    "3": "Conscientiousness",
    "4": "Openness",
}

# Define the personality detection function


def personality_detection(text):
    if len(text) < 20:
        return {
            "Extroversion": 0.0,
            "Neuroticism": 0.0,
            "Agreeableness": 0.0,
            "Conscientiousness": 0.0,
            "Openness": 0.0,
            "text": text
        }
    else:

        # Encoding input data
        dict_custom = {}
        Preprocess_part1 = text[:len(text)]
        Preprocess_part2 = text[len(text):]
        dict1 = tokenizer.encode_plus(
            Preprocess_part1, max_length=1024, padding=True, truncation=True)
        dict_custom['input_ids'] = [dict1['input_ids'], dict1['input_ids']]
        dict_custom['token_type_ids'] = [
            dict1['token_type_ids'], dict1['token_type_ids']]
        dict_custom['attention_mask'] = [
            dict1['attention_mask'], dict1['attention_mask']]
        outs = bmodel(torch.tensor(dict_custom['input_ids']), token_type_ids=None, attention_mask=torch.tensor(
            dict_custom['attention_mask']))
        b_logit_pred = outs[0]
        pred_label = torch.sigmoid(b_logit_pred)
        ret = {
            "Extroversion": float(pred_label[0][0]),
            "Neuroticism": float(pred_label[0][1]),
            "Agreeableness": float(pred_label[0][2]),
            "Conscientiousness": float(pred_label[0][3]),
            "Openness": float(pred_label[0][4]),  "text": text}
        return ret


def record_audio():
    global cap, start_time, og_time, result, audio_format, channels, audio_rate, device_index, chunk, audio_out, audio_times, stream, audio_input

    audio_out = wave.open(audio_file, 'wb')
    audio_out.setnchannels(channels)
    audio_out.setsampwidth(pyaudio.PyAudio().get_sample_size(audio_format))
    audio_out.setframerate(audio_rate)

    # Initialize PyAudio
    audio_input = pyaudio.PyAudio()

    # Find and print available audio devices
    print("Available audio devices:")
    for i in range(audio_input.get_device_count()):
        device_info = audio_input.get_device_info_by_index(i)
        print(f"{i}: {device_info['name']}")

    # Select the default audio input device
    device_index = None  # Specify the device index here if needed
    for i in range(audio_input.get_device_count()):
        if "Microphone" in audio_input.get_device_info_by_index(i)["name"]:
            device_index = i
            break

    # Check if a suitable device was found
    if device_index is None:
        print("Error: No suitable audio input device found.")
        exit()

    print(
        f"Using audio device: {audio_input.get_device_info_by_index(device_index)['name']}")

    stream = audio_input.open(format=audio_format,
                              channels=channels,
                              rate=audio_rate,
                              input=True,
                              input_device_index=device_index,
                              frames_per_buffer=chunk)

    while True:
        if stop_flag:
            break
        data = stream.read(chunk)
        audio_out.writeframes(data)
        audio_times.append(time.time())


def record_video():
    global cap, start_time, og_time, result, audio_format, channels, audio_rate, device_index, chunk, audio_out, audio_times, video_file

    cap = cv2.VideoCapture(0)
    start_time = time.time()
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    size = (frame_width, frame_height)
    result = cv2.VideoWriter(video_file,
                             cv2.VideoWriter_fourcc(*'XVID'),
                             3, size)


def predict_emotion(face):
    face = cv2.resize(face, (48, 48), interpolation=cv2.INTER_AREA)
    face = face.astype('float')/255.0
    face = np.expand_dims(face, axis=0)
    prediction = model.predict(face)
    emotion = emotions[np.argmax(prediction)]
    return emotion


def detect_emotion_and_gaze(frame):
    global start_time, emotion_times, blink_count, time_left, time_right, time_center, last_time

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    current_time = time.time()
    time_delta = current_time - last_time
    last_time = current_time

    for (x, y, w, h) in faces:
        face = gray[y:y+h, x:x+w]
        emotion = predict_emotion(face)
        emotion_times[emotion] += time_delta

        gaze.refresh(frame)

        text = ""
        if gaze.is_blinking():
            text = "Blinking"
            blink_count += time_delta

        elif gaze.is_right():
            text = "Looking right"
            time_right += time_delta

        elif gaze.is_left():
            text = "Looking left"
            time_left += time_delta

        elif gaze.is_center():
            text = "Looking center"
            time_center += time_delta

        cv2.putText(frame, text, (x, y-20),
                    cv2.FONT_HERSHEY_DUPLEX, 0.9, (147, 58, 31), 2)

        left_pupil = gaze.pupil_left_coords()
        right_pupil = gaze.pupil_right_coords()
        cv2.putText(frame, "Left pupil:  " + str(left_pupil), (x, y+h+30),
                    cv2.FONT_HERSHEY_DUPLEX, 0.9, (147, 58, 31), 1)
        cv2.putText(frame, "Right pupil: " + str(right_pupil), (x, y+h+60),
                    cv2.FONT_HERSHEY_DUPLEX, 0.9, (147, 58, 31), 1)

        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

        cv2.putText(frame, emotion, (x, y-50),
                    cv2.FONT_HERSHEY_DUPLEX, 1.6, (36, 255, 12), 2)

        start_time = time.time()

    return frame


def detect_emotion_and_gaze_real(frame):
    global start_time, emotion_times, blink_count, time_left, time_right, time_center, last_time

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    current_time = time.time()
    time_delta = current_time - last_time
    last_time = current_time

    for (x, y, w, h) in faces:
        face = gray[y:y+h, x:x+w]
        emotion = predict_emotion(face)
        emotion_times[emotion] += time_delta

        gaze.refresh(frame)

        text = ""
        if gaze.is_blinking():
            text = "Blinking"
            blink_count += time_delta

        elif gaze.is_right():
            text = "Looking right"
            time_right += time_delta

        elif gaze.is_left():
            text = "Looking left"
            time_left += time_delta

        elif gaze.is_center():
            text = "Looking center"
            time_center += time_delta

        start_time = time.time()

    return frame


def generate():
    global cap, start_time, og_time, result, audio_format, channels, audio_rate, device_index, chunk, audio_out, audio_times
    og_time = time.time()
    # Start recording threads
    audio_thread = threading.Thread(target=record_audio)
    video_thread = threading.Thread(target=record_video)

    audio_thread.start()
    video_thread.start()

    # Wait for threads to finish
    # audio_thread.join()
    video_thread.join()

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = detect_emotion_and_gaze(frame)
        result.write(frame)
        _, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        frame_times.append(time.time())
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')


def generate_real():
    global cap, start_time, og_time, result, audio_format, channels, audio_rate, device_index, chunk, audio_out, audio_times
    og_time = time.time()
    # Start recording threads
    audio_thread = threading.Thread(target=record_audio)
    video_thread = threading.Thread(target=record_video)

    audio_thread.start()
    video_thread.start()

    # Wait for threads to finish
    # audio_thread.join()
    video_thread.join()

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = detect_emotion_and_gaze_real(frame)
        result.write(frame)
        _, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        frame_times.append(time.time())
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')


@app.route('/video_feed')
def video_feed():
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame', headers={"Access-Control-Allow-Origin": "*"})


@app.route('/video_feed_real')
def video_feed_real():
    return Response(generate_real(), mimetype='multipart/x-mixed-replace; boundary=frame', headers={"Access-Control-Allow-Origin": "*"})


@app.route('/close_camera')
def close_camera():
    global cap, emotion_times, blink_count, last_time, time_left, time_right, time_center, og_time, start_time, result, frame_times, audio_times, video_file, audio_file, audio_out, stream, audio_input, stop_flag, video_url
    if cap is not None:
        # Release resources
        stop_flag = True
        cap.release()

        # Synchronize audio and video
        start_time = max(frame_times[0], audio_times[0])
        frame_times = [t - start_time for t in frame_times]
        audio_times = [t - start_time for t in audio_times]

        # Release resources
        audio_out.close()
        stream.stop_stream()
        stream.close()
        audio_input.terminate()
        result.release()

        # Merge video and audio using ffmpeg
        command = f"ffmpeg -y -i {video_file} -i {audio_file} -c:v libx264 -preset veryfast -crf 22 -c:a aac -strict experimental {output_file}"
        subprocess.call(command, shell=True)

        # Check if output file exists and has non-zero size
        if os.path.exists(output_file) and os.path.getsize(output_file) > 0:
            print("Recording stopped. Output file with audio and video:", output_file)
        else:
            print("Error: Output file not created or has zero size.")

        total_emotion_time = sum(emotion_times.values())
        if total_emotion_time == 0:
            # Handle the case when total_emotion_time is zero
            emotion_percentages = {emotion: 0 for emotion in emotion_times}
        else:
            # Calculate emotion percentages normally
            emotion_percentages = {emotion: (
                time_spent / total_emotion_time) * 100 for emotion, time_spent in emotion_times.items()}

        # Reset variables
        start_time = None
        emotion_times = {emotion: 0 for emotion in emotions}
        upload_to_firebase(file_path="output_with_audio.mp4")
        total_gaze_time = time_center + time_left + time_right
        if total_gaze_time == 0:
            time_left = 0
            time_right = 0
            time_center = 0
        else:
            time_left = time_left/total_gaze_time
            time_right = time_right/total_gaze_time
            time_center = time_center/total_gaze_time
        # Return additional information
        return jsonify({
            "emotion_percentages": emotion_percentages,
            "total_blinks": blink_count,
            "total_time_looking_left": (time_left) * 100,
            "total_time_looking_right": (time_right) * 100,
            "total_time_looking_center": (time_center) * 100,
            "total_time_seconds": round(time.time() - og_time, 2),
            "video_link": video_url
        })
    return 'Camera Closed'


@app.route('/audio_analysis', methods=['POST'])
def audio_analysis():
    # Check if the request contains a link
    if 'link' not in request.json:
        return jsonify({"error": "No link provided in the request"}), 400

    audio_link = request.json['link']

    # Download the audio file from Firebase
    try:
        response = requests.get(audio_link)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        return jsonify({"error": f"Failed to download the audio file: {str(e)}"}), 400

    # Save the audio file temporarily
    temp_webm_path = "temp.webm"
    temp_wav_path = "temp.wav"
    with open(temp_webm_path, 'wb') as f:
        f.write(response.content)

    # Convert the WebM file to WAV using ffmpeg
    try:
        subprocess.run(["ffmpeg", "-i", temp_webm_path, temp_wav_path])
    except Exception as e:
        return jsonify({"error": f"Failed to convert audio file to WAV: {str(e)}"}), 500
    # Initialize the recognizer
    recognizer = sr.Recognizer()

    # Load the audio file
    with sr.AudioFile(temp_wav_path) as source:
        audio_data = recognizer.record(source)  # Read the entire audio file

    try:
        # Perform speech recognition
        text = recognizer.recognize_google(audio_data)
    except sr.UnknownValueError:
        # Speech recognition could not understand the audio
        text = "Speech recognition could not understand the audio"
    except sr.RequestError as e:
        # Could not request results from Google Speech Recognition service
        text = f"Could not request results from Google Speech Recognition service: {str(e)}"

    # Load the converted WAV file and perform analysis
    classifier = foreign_class(source="speechbrain/emotion-recognition-wav2vec2-IEMOCAP",
                               pymodule_file="custom_interface.py", classname="CustomEncoderWav2vec2Classifier")
    out_prob, score, index, text_lab = classifier.classify_file(temp_wav_path)

# Make a POST request to personality_detection endpoint
    personality_detection_url = "http://127.0.0.1:8080/personality_detection"
    data = {"text": text}
    try:
        personality_response = requests.post(
            personality_detection_url, json=data)
        personality_response.raise_for_status()
        personality_traits = personality_response.json()
    except requests.exceptions.RequestException as e:
        print(e)
    # Remove the temporary files
    os.remove(temp_webm_path)
    os.remove(temp_wav_path)

    # Return the analysis result
    return jsonify({"result": text_lab, "personality_traits": personality_traits, "transcript": text}), 200


@app.route('/personality_detection', methods=['POST'])
def detect_personality():
    data = request.get_json()
    text = data.get('text', '')
    if text:
        personality_traits = personality_detection(text)
        return jsonify(personality_traits)
    else:
        return jsonify({"error": "Text not provided"}), 400


if __name__ == '__main__':
    app.run(port=8080, debug=True)
