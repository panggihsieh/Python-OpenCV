# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import cv2
import numpy as np
face=cv2.CascadeClassifier(r'E:\Python Data\data\haarcascades\haarcascade_frontalface_default.xml')
img=cv2.imread(r'E:\Python Data\data\image\who.jpg')


#gray=cv2.cvtColor(img,cv2.COLOR_BGR2BGRA)
faces=face.detectMultiScale(img)
#faces=face.detectMultiScale(
#        gray,
#        scaleFactor=1.1,
#        minNeighbors=5,
#        minSize=(30,30),
#        )
for (x,y,h,w) in faces:
    cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),1)
    text='{},{}'.format(x+w,y+h)
    cv2.putText(img,str(x)+','+str(y),(x+w,y+w-7), 1, 1, (0,0,255),1)
    #cv2.putText(img, text, (10, 40), cv2.FONT_HERSHEY_SIMPLEX,1, (0, 0, 255))
cv2.imshow('Face found',img)
cv2.waitKey(0)
cv2.destroyAllWindows()



