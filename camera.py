import cv2
from detection import AccidentDetectionModel
import numpy as np
import os

model = AccidentDetectionModel("model.json", 'model_weights.h5')
font = cv2.FONT_HERSHEY_SIMPLEX

def startapplication():
    video = cv2.VideoCapture('cars.mp4') # for camera use video = cv2.VideoCapture(0)
    while True:
        ret, frame = video.read()
        
        # Convert frame to grayscale if needed (remove if model expects BGR)
        # gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Resize frame to match model input size
        roi = cv2.resize(frame, (250, 250))
        
        # Predict accident
        pred, prob = model.predict_accident(roi[np.newaxis, :, :])
        
        if pred == "Accident":
            prob = round(prob[0][0] * 100, 2)
            
            # Uncomment to activate alert mechanism based on probability threshold
            # if prob > 90:
            #     os.system("say beep")
            
            # Display prediction and probability on the frame
            cv2.rectangle(frame, (0, 0), (280, 40), (0, 0, 0), -1)
            cv2.putText(frame, f"{pred} {prob}%", (20, 30), font, 1, (255, 255, 0), 2)

        # Display the video frame
        cv2.imshow('Video', frame)  

        # Check for user input to quit
        if cv2.waitKey(33) & 0xFF == ord('q'):
            break

    # Release video object and close all windows
    video.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    startapplication()
