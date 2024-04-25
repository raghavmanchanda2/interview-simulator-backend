import tensorflow as tf
import cv2
import numpy as np
import matplotlib.pyplot as plt
import time


# Load the saved model
model = tf.keras.models.load_model('model/model.h5')

# Define the face cascade and emotions
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
emotions = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

no_face_detection_alert = "Cannot Detect Face"
low_confidence_alert = "Cannot Detect Emotion"


def predict_emotion(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    frame_height, frame_width = frame.shape[:2]

    text_size = cv2.getTextSize(
        low_confidence_alert, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
    text_x = frame_width - text_size[0] - 10
    text_y = frame_height - 10

    prediction = None
    message = None

    for (x, y, w, h) in faces:
        face = gray[y:y+h, x:x+w]
        face = cv2.resize(face, (48, 48), interpolation=cv2.INTER_AREA)
        if np.sum([face]) != 0:
            face = face.astype('float')/255.0
            face = tf.keras.utils.img_to_array(face)
            face = np.expand_dims(face, axis=0)
            prediction = model.predict(face)
            if any(prob > .5 for prob in prediction[0]):
                emotion = emotions[np.argmax(prediction)]
                text_size = cv2.getTextSize(
                    emotion, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
                text_x = frame_width - text_size[0] - 10
                text_y = frame_height - 10
                cv2.putText(frame, emotion, (text_x, text_y),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

                # Display message for fear, anger, or sadness
                if emotion in ['Fear', 'Angry']:
                    message = "Please calm down and take a breather."
                elif emotion == 'Sad':
                    message = "Cheer up!!!"

                if message:
                    text_size = cv2.getTextSize(
                        message, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
                    message_x = int((frame_width - text_size[0]) / 2)
                    message_y = int((frame_height + text_size[1]) / 2)
                    cv2.putText(frame, message, (message_x, message_y),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            else:
                cv2.putText(frame, low_confidence_alert, (text_x, text_y),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        else:
            cv2.putText(frame, no_face_detection_alert, (text_x, text_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    return frame, prediction


emotion_durations = {emotion: 0 for emotion in emotions}
current_emotion = None
start_time = None

# Start the video capture and emotion detection
cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    if ret:
        frame, prediction = predict_emotion(frame)
        cv2.imshow('Live Facial Emotion Detection', frame)

        if prediction is not None:
            if current_emotion is not None:
                elapsed_time = time.time() - start_time
                emotion_durations[current_emotion] += elapsed_time

            current_emotion = emotions[np.argmax(prediction[0])]
            start_time = time.time()

    if cv2.waitKey(1) == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()

plt.figure(figsize=(10, 5))
plt.bar(emotions, emotion_durations.values(), color='blue')
plt.xlabel('Emotions')
plt.ylabel('Duration (s)')
plt.title('Emotion Duration')
plt.xticks(rotation=45)
plt.show()

# import tensorflow as tf
# import cv2
# import numpy as np
# import matplotlib.pyplot as plt
# from collections import defaultdict
# import time

# # Load the saved model
# model = tf.keras.models.load_model('model/model.h5')

# # Define the face cascade and emotions
# face_cascade = cv2.CascadeClassifier(
#     cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
# emotions = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# no_face_detection_alert = "Cannot Detect Face"
# low_confidence_alert = "Cannot Detect Emotion"


# def predict_emotion(frame, emotion_start_times, emotion_end_times):
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     faces = face_cascade.detectMultiScale(gray, 1.3, 5)
#     for (x, y, w, h) in faces:
#         face = gray[y:y+h, x:x+w]
#         face = cv2.resize(face, (48, 48), interpolation=cv2.INTER_AREA)
#         if np.sum([face]) != 0:
#             face = face.astype('float')/255.0
#             face = tf.keras.utils.img_to_array(face)
#             face = np.expand_dims(face, axis=0)
#             prediction = model.predict(face)
#             if any(prob > .5 for prob in prediction[0]):
#                 emotion = emotions[np.argmax(prediction)]
#                 if emotion not in emotion_start_times:
#                     emotion_start_times[emotion] = time.time()
#                 cv2.putText(frame, emotion, (x, y - 10),
#                             cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
#                 cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 0), 2)
#             else:
#                 cv2.putText(frame, low_confidence_alert, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0),
#                             2)
#         else:
#             cv2.putText(frame, no_face_detection_alert, (x, y - 10),
#                         cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
#             continue  # Skip processing if no face is detected

#         # Check for emotions ending
#         for emotion in list(emotion_start_times.keys()):
#             if not any(emotion == emotions[np.argmax(prediction)] for prediction in model.predict(face)):
#                 if emotion not in emotion_end_times:
#                     emotion_end_times[emotion] = time.time()

#     return frame


# def plot_emotion_times(emotion_start_times, emotion_end_times):
#     emotion_durations = defaultdict(float)
#     for emotion, start_time in emotion_start_times.items():
#         if emotion in emotion_end_times:
#             duration = emotion_end_times[emotion] - start_time
#             emotion_durations[emotion] += duration

#     plt.bar(emotion_durations.keys(), emotion_durations.values())
#     plt.xlabel('Emotion')
#     plt.ylabel('Time Spent (seconds)')
#     plt.title('Emotion Detection Over Time')
#     plt.show()


# # Start the video capture and emotion detection
# cap = cv2.VideoCapture(0)
# emotion_start_times = {}
# emotion_end_times = {}
# while True:
#     ret, frame = cap.read()
#     if ret:
#         frame = predict_emotion(frame, emotion_start_times, emotion_end_times)
#         cv2.imshow('Live Facial Emotion Detection', frame)
#     if cv2.waitKey(1) == ord('q'):
#         break

# # Plot the emotion detection over time
# plot_emotion_times(emotion_start_times, emotion_end_times)

# cap.release()
# cv2.destroyAllWindows()
