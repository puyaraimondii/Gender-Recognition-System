import cv2
import sys
import datetime as dt
from time import sleep

cascPath = "haarcascade_frontalface_lib.xml"
faceCascade = cv2.CascadeClassifier(cascPath)
video_capture = cv2.VideoCapture(0)
detectface= 0
save_time= 0

while True:
    # Capture frame-by-frame
    ret, frame = video_capture.read()
    #transfer it into gray-scale map
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )
    # Draw a rectangle around the faces
    for (x, y, w, h) in faces:
        # to increase the accuracy of finding a real face
        if w+h>200:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
    # Display the resulting frame
    cv2.imshow('Video', frame)
    resized_image = cv2.resize(frame, (320, 200))
    #test if face is recognized
    if detectface != len(faces):
        detectface = len(faces)
        save_time = save_time + 1
        print (save_time)
        cv2.imwrite("test1.png", resized_image, [int(cv2.IMWRITE_PNG_COMPRESSION), 9])
    # refresh the screen
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    #or use following
    '''
    keycode = cv2.waitKey(1) 
    if keycode != -1: 
        keycode &= 0xFF

    '''
    
    #On some systems, waitKey() may return a value that encodes more than just the ASCII keycode. (A bug is known to occur on Linux when OpenCV uses GTK as its backend GUI 
    #library.) On all systems, we can ensure that we extract just the SCII keycode by reading the last byte from the return value like this: 
    #keycode = cv2.waitKey(1) 
    #if keycode != -1: 
    #keycode &= 0xFF



