import cv2
from random import randrange as r

trainedData=cv2.CascadeClassifier('haarcascade.xml')

cam=cv2.VideoCapture("video.mp4")

while True:
 success,frame=cam.read()

 grayimg=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

 faceCoordinates=trainedData.detectMultiScale(grayimg)

 for x,y,w,h in faceCoordinates:
    cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),2)

 cv2.imshow('window',frame)
 key=cv2.waitKey(1)
 if(key==81 or key==113):
    break

cam.release()

img=cv2.imread('a.jpg')

grayimg=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

faceCoordinates=trainedData.detectMultiScale(grayimg)

for x,y,w,h in faceCoordinates:
    cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),2)

cv2.imshow('window',img)
cv2.waitKey()

print('End of program')