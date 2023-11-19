# human-recognition
#recognition of humans from camera or by using video



import cv2 as cv
import numy as np
import matplotlib.pyplot as plt
#importing libraries

#loading cascade file for full_body
human_cascade=cv.CascadeClassifier(cv.data.haarcascades + 'haarcascade_fullbody.xml')

#getting video from default camera
cap=cv.VideoCapture(0)

#loading frames from video one by one
while cap.isOpened():
    ret,frame=cap.read()
    gray=cv.cvtColor(frame,cv.COLOR_BGR2GRAY)
    #changing i=orginal image to gray image

    #detecting humans coordinates in the each frame by cascade classifier
    humans=human_cascade.detectMultiScale(gray,2.6,0)

    #drawing a rectange around the facesin the each frame
    for (x, y, w, h) in humans:
        cv.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 3)


    #showing the image on the screan
    cv.imshow('frame',frame)

    #break the code in b/w the video
    if cv.waitKey(2) & 0xFF == ord('p'):
            break


cap.release()
cv.destroyAllWindows()
#used to clear space used by cam and release the camera
