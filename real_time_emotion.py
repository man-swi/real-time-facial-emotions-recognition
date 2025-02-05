import cv2
import numpy as np
import tensorflow as tf

model = tf.keras.models.load_model("emotion_model.h5")

emotion_labels = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Opens webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Converting frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detecting faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:
        roi_gray = gray[y:y + h, x:x + w]  # Extracting face region
        roi_gray = cv2.resize(roi_gray, (48, 48))  # Resizing to match model input
        roi_gray = np.expand_dims(roi_gray, axis=0).reshape(-1, 48, 48, 1) / 255.0  # Normalizing

        # Predicting emotion
        predictions = model.predict(roi_gray)
        emotion = emotion_labels[np.argmax(predictions)]

        
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # output
    cv2.imshow("Emotion Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):  #'q' to exit
        break

cap.release()
cv2.destroyAllWindows()
