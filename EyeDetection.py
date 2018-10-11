# -*- coding: utf-8 -*-
"""
Created on Thu Oct 11 23:09:51 2018

@author: Vaidik
"""

eye_cascade_path = 'F:/Python_Projects/Face_Recognition/haarcascade_eye.xml'
iris_cadcade_path = "E:/Internship/dasar_haartrain/haarcascade_iris (2).xml"

import cv2
import matplotlib.pyplot as plt

eye_cascade = cv2.CascadeClassifier(eye_cascade_path)
iris_cascade = cv2.CascadeClassifier(iris_cadcade_path)
imgPath = "E:\Internship\img.jpg"


img = cv2.imread(imgPath)
image = img.copy()
output = img.copy()
grayImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
eyes = eye_cascade.detectMultiScale(grayImg, 1.1, 5)

for (x, y, w, h) in eyes:
    print("X: ",x," Y: ",y ," W: ", w," H: ", h)
    nx = x-80
    ny = y*2.1
    if(nx<0):
        nx = x+87
        ny = y/2.1
    xc = int(((2*nx)+w)/2)
    yc = int(((2*ny)+h)/2)
    r = 8
    print("Xc: ",xc, " Yc: ", yc, " R: ", r)
    cv2.circle(image, (xc, yc), r, (0, 155, 0), -1)
    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 1)
    cv2.circle(output, (xc, yc), r, (0, 155, 0), -1)
    cv2.rectangle(output, (x, y), (x+w, y+h), (0, 255, 0), 1)
    

plt.imshow(img)
plt.imshow(image)
plt.imshow(output)
cv2.imshow("Detecting Eyes", img)
cv2.imshow("Changing Color", image)
cv2.imshow("Final Output", output)
'''
for (x, y, r) in eyes:
    cv2.circle(image, (x, y), r, (255, 0 , 0), -1)
    cv2.imshow("Image", image)
'''
cv2.waitKey(0)
cv2.destroyAllWindows()