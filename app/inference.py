import mediapipe as mp
import numpy as np
from tensorflow import keras
from keras import layers
from modelutils import classifier_model
import cv2
import matplotlib.pyplot as plt
import os
import gdown


mp_face_detection = mp.solutions.face_detection
rescale = layers.Rescaling(scale=1./127.5, offset=-1)
classes = ['suprise', 'fear', 'disgust', 'happy','sad', 'angry', 'neutral']

def load_model():
    url = "https://drive.google.com/file/d/1UABX1ZD9XiJTk2a0CGx6SFsgjWpZyM8F/view?usp=sharing" 
    model_path = "facemodel.keras"

    if not os.path.exists(model_path):
        gdown.download(url, model_path, quiet=False)

    return keras.models.load_model(model_path)


classifier = load_model()
frame_results = []


# For webcam input:
def run_classifier_on_video():
    cap = cv2.VideoCapture(0)
    count = 0
    with mp_face_detection.FaceDetection(
        model_selection=1, min_detection_confidence=1) as face_detection:
        while cap.isOpened() and count <= 100:
            success, frame = cap.read()
            if not success:
                print("Ignoring empty camera frame.")
                    # If loading a video, use 'break' instead of 'continue'.
                continue

            # To improve performance, optionally mark the image as not writeable to
            # pass by reference.
            image = frame.copy()
            height, width, channel = image.shape

            image.flags.writeable = False
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = face_detection.process(image)

            # Draw the face detection annotations on the image.
            image.flags.writeable = True
            if results.detections:
                for detection in results.detections:

                    bb = detection.location_data.relative_bounding_box

                    x, y, w, h = int(bb.xmin*width), int(bb.ymin*height), int(bb.width*width), int(bb.height*height)
                    face = image[y:y + h, x:x + w]
                    face = rescale(cv2.resize(face, (224, 224)))
                    prediction = classifier.predict(np.expand_dims(face, axis=0))
                    prediction = np.argmax(prediction, axis=1)

                    frame_results.append(classes[prediction[0]])
                    cv2.putText(frame, classes[prediction[0]], (x, y), cv2.FONT_HERSHEY_COMPLEX, .75, (255, 100, 150), thickness=2)
                    cv2.rectangle(frame, pt1=(x, y), pt2=(x + w, y + h), color=(225, 255, 255), thickness=2)


            cv2.imshow('MediaPipe Face Detection', frame)
            if cv2.waitKey(5) & 0xFF == ord('q'):
                break
            count += 1

    cap.release()
    cv2.destroyAllWindows()
    val, index, counts = np.unique(frame_results, return_counts=True, return_inverse=True)
    max_index = list(counts).index(max(counts))
    return val[max_index]
