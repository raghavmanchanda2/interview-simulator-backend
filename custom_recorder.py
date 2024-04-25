import cv2
import pyaudio
import numpy as np
import threading
import wave
import subprocess
import time
import os

# Video settings
frame_width = 640
frame_height = 480
fps = 30

# Audio settings
audio_format = pyaudio.paInt16
channels = 1  # Set the number of audio channels
audio_rate = 44100
chunk = 1024

# Duration of recording in seconds
record_duration = 10

# File paths
video_file = 'output.avi'
audio_file = 'output.wav'
output_file = 'output_with_audio.avi'  # Output in AVI format

# Video writer
fourcc = cv2.VideoWriter_fourcc(*'XVID')
video_out = cv2.VideoWriter(video_file, fourcc, fps,
                            (frame_width, frame_height))

# Audio recorder
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

# Flag to stop recording
stop_flag = False

# Global variables for synchronization
frame_times = []
audio_times = []


def record_audio():
    start_time = time.time()
    while time.time() - start_time < record_duration:
        if stop_flag:
            break
        data = stream.read(chunk)
        audio_out.writeframes(data)
        audio_times.append(time.time())


def record_video():
    start_time = time.time()
    cap = cv2.VideoCapture(0)
    while cap.isOpened() and time.time() - start_time < record_duration:
        if stop_flag:
            break
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, (frame_width, frame_height))
        video_out.write(frame)  # Write the frame
        frame_times.append(time.time())

    cap.release()


# Start recording threads
audio_thread = threading.Thread(target=record_audio)
video_thread = threading.Thread(target=record_video)

audio_thread.start()
video_thread.start()

# Wait for threads to finish
audio_thread.join()
video_thread.join()

# Synchronize audio and video
start_time = max(frame_times[0], audio_times[0])
frame_times = [t - start_time for t in frame_times]
audio_times = [t - start_time for t in audio_times]

# Release resources
video_out.release()
audio_out.close()
stream.stop_stream()
stream.close()
audio_input.terminate()

# Merge video and audio using ffmpeg
command = f"ffmpeg -y -i {video_file} -i {audio_file} -c:v copy -c:a aac -strict experimental {output_file}"
subprocess.call(command, shell=True)

# Check if output file exists and has non-zero size
if os.path.exists(output_file) and os.path.getsize(output_file) > 0:
    print("Recording stopped. Output file with audio and video:", output_file)
else:
    print("Error: Output file not created or has zero size.")
