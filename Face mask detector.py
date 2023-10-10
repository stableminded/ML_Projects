import cv2
import os
import tensorflow as tf
import numpy as np


from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
face_classifier=cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
model = load_model('face_mask_detector.h5')
text_Mask = "Mask On"
text_No_Mask="Mask Off"
font=cv2.FONT_HERSHEY_SIMPLEX
scale=0.8
def predict(image):
    face_frame=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    face_frame=cv2.resize(face_frame,(224,224))
    face_frame=img_to_array(face_frame)
    face_frame=np.expand_dims(face_frame,axis=0)
    face_frame=preprocess_input(face_frame)
    prediction=model.predict(face_frame)
    
    return prediction[0][0]
 
def detector(gray_image, frame):
    faces = face_classifier.detectMultiScale(gray_image, 1.1, 5)

    for (x, y, w, h) in faces:
        roi_color = frame[y:y + h, x:x + w]
        mask = predict(roi_color)

        if mask > 0.5:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, text=text_Mask, org=(x + 50, y - 10), fontFace=font, fontScale=scale, color=(0, 255, 0), thickness=2)
        elif mask <= 0.5:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255  ), 2)
            cv2.putText(frame, text=text_No_Mask, org=(x + 50, y - 100), fontFace=font, fontScale=scale, color=(0, 0, 255), thickness=2)

    return frame

video_cap=cv2.VideoCapture(0)
while(True):
    
    ret,frame=video_cap.read()
    gray_frame=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
  
    if ret and frame.shape[0] > 0 and frame.shape[1] > 0:
     # Process and display the frame
        detect = detector(gray_frame, frame)
        cv2.imshow("Video", detect)
    else:
        print("Error: Frame capture failed or frame dimensions are invalid.")

    if cv2.waitKey(1)& 0xFF==ord("q"):
        break
video_cap.release()
cv2.destroyAllWindows()
    #Now lets go to the real time application