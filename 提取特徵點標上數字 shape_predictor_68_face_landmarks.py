# -*- coding: utf-8 -*-
"""
Created on Mon May 20 21:38:07 2019

@author: hsieh
"""
import cv2
import os
import numpy as np
from imutils import face_utils
import dlib
img=cv2.imread(r'E:\python data\data\image\men.jpg')
img2= cv2.resize(img, (0, 0), fx=2, fy=2, interpolation=cv2.INTER_NEAREST)
bw=cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)
predict=dlib.shape_predictor(r'E:\python data\data\dlib\landmarks.dat')
detector=dlib.get_frontal_face_detector()
faces=detector(img2)
n=0
for face in faces:
    x=face.left()
    y=face.top()
    w=face.width()
    h=face.height()
    shape=predict(bw,face)
    pt=face_utils.shape_to_np(shape)
    print(type(pt))
    cv2.rectangle(img2,(x,y),(x+w,y+h),(0,255,0))
    for x,y in pt:
        n +=1
        #cv2.circle(img,(x,y),1,(0,0,255))
        cv2.putText(img2,str(n),(x,y),1,1,(0,0,255))
cv2.imshow('logo',img2)
cv2.waitKey(0)
cv2.destroyAllWindows()




