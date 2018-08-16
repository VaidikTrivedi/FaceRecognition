# -*- coding: utf-8 -*-
"""
Created on Wed Jul 25 09:59:16 2018

@author: Vaidik
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Jul  4 19:12:19 2018

@author: Vaidik
"""

import cv2
import numpy as np
import os
import pyttsx3
from PIL import Image

recognizer = cv2.face.LBPHFaceRecognizer_create()
face_cascade_path = 'F:/Python_Projects/Face_Recognition/haarcascade_frontalface_default.xml'
eye_cascade_path = 'F:/Python_Projects/Face_Recognition/haarcascade_eye.xml'

face_cascade = cv2.CascadeClassifier(face_cascade_path)
eye_cascade = cv2.CascadeClassifier(eye_cascade_path)

names = ['', 'Vaidik', 'Dharik', 'Kafil', 'Sunil', 'Parth', 'Drumil','','','','Negative Image']

engine = pyttsx3.init();

def faceDetection():
    cap = cv2.VideoCapture(0)
    while (cap.isOpened()):
        rect, img = cap.read()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    #    print(len(faces))
        for(x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
            cv2.putText(img, "Face: " + str(len(faces)), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,255,0), 3)
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = img[y:y+h, x:x+w]
            eyes = eye_cascade.detectMultiScale(roi_gray)
            for(ex, ey, ew, eh) in eyes:
                cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)
                
        cv2.imshow("Face(s) Found!", img)
        k = cv2.waitKey(10)
        if(k==27):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    return

def faceCapture():
    #name = input("Enter Name: ")
    path = 'F:/Python_Projects/Face_Recognition/Face-Recognition-Train-YML-Python-master/dataSet/'
    #os.mkdir(path)
    count = 0;
    flag = 0;
    cap = cv2.VideoCapture(0)
    while (cap.isOpened()):
        rect, img = cap.read()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    #    print(len(faces))
        for(x, y, w, h) in faces:
            image = img.copy()
            image1 = img.copy()
            cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
            cv2.putText(image1, str(count), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,255,0), 3)
            #roi_gray = gray[y:y+h, x:x+w]
            crop_img = image[y:y+h, x:x+w]
            crop_img = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
            cv2.imshow("croped img", crop_img)
            cv2.imwrite(os.path.join(path, 'user.10.'+ str(count)+'.jpg'), crop_img)
            count = count + 1
            if(count>=50):
                flag = 1
                break
                
        cv2.imshow("Face(s) Found!", img)
        k = cv2.waitKey(10)
        if(k==27 or flag==1):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    return

def faceTrainer(path):
    imagePath = [os.path.join(path,f) for f in os.listdir(path)]
    faces = []
    Ids = []
    for image in imagePath:
        faceImg = Image.open(image).convert('L')
        faceNp = np.array(faceImg, 'uint8')
        Id = int(os.path.split(image)[-1].split('.')[1])
        faces.append(faceNp)
        print("Id: ", Id, "Person: ", names[Id])
        Ids.append(Id)
        cv2.imshow("Training...", faceNp)
        cv2.waitKey(10)
    return Ids, faces
    
def faceRecogition(path):
    global names
    count = 1
    img = cv2.imread(path)
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray_img, 1.5, 5)
    for(x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 255, 255), 5)
        Id, conf = recognizer.predict(gray_img[y:y+h, x:x+w])
        print("Id: ", Id)
        if(Id<7):
            name = names[Id]
        else:
            name = "Unknown"
            Id="Unknown"
        print("Person: ", name, "   ||  Confidence: ", 100 - conf)
        cv2.putText(img, str(count)+". "+str(name), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 4, (255, 255, 0), 7)
        count+=1
        tempimg = cv2.resize(img, (400, 400))
        cv2.imshow(str(name), tempimg)
        engine.say(str(name))
        engine.runAndWait()
    show_img = cv2.resize(img, (900,900))
    cv2.imshow("Image", show_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
        
    return


def retrainMode():
    path = 'F:/Python_Projects/Face_Recognition/Face-Recognition-Train-YML-Python-master/dataSet/'
    count = 100;
    flag = 0;
    cap = cv2.VideoCapture(0)
    while (cap.isOpened()):
        rect, img = cap.read()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.5, 5)
    #    print(len(faces))
        for(x, y, w, h) in faces:
            image = img.copy()
            image1 = img.copy()
            cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
            cv2.putText(image1, str(count), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,255,0), 3)
            #roi_gray = gray[y:y+h, x:x+w]
            crop_img = image[y:y+h, x:x+w]
            crop_img = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
            cv2.imshow("croped img", crop_img)
            cv2.imwrite(os.path.join(path, 'user.1.'+ str(count)+'.jpg'), crop_img)
            count = count + 1
            if(count>=219):
                flag = 1
                break
                
        cv2.imshow("Face(s) Found!", img)
        k = cv2.waitKey(10)
        if(k==27 or flag==1):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    return

def trainAgain(path):
    data_path = 'F:/Python_Projects/Face_Recognition/Face-Recognition-Train-YML-Python-master/dataSet/'
    imgcount = [os.path.join(data_path,f) for f in os.listdir(data_path)]
    img = cv2.imread(path)
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray_img, 1.5, 5)
    count = 0
    print("Person | ID")
    for i in names:
        print( i, count)
        count = count + 1
    for (x, y, w, h) in faces:
        count = 0
        crop_img = img[y:y+w, x:x+w]
        cv2.imshow("Id", crop_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        if input("This was Correct? [y/n] ") == 'y':
            continue
        Id = input("Enter Id: ")
        for lastCount in imgcount:
            if Id == os.path.split(lastCount)[-1].split('.')[1]:
                count+=1
        count+=1
        print("Photo Saved by this count: ",count)
        cv2.imwrite(os.path.join(data_path, 'user.'+Id+'.'+ str(count)+'.jpg'), crop_img)
    return

action = 0
while(action!=6):
    print("\n\n\n1. Face Detection \n2. Face Capturing \n3. Face Training\n4. Face Recognition\n5. Retrain Model\n6. Exit");
    action = int(input("Enter Your Choice: "))
    if(action==1):
        faceDetection();
    elif(action==2):
        faceCapture();
    elif(action==3):
        path = 'F:/Python_Projects/Face_Recognition/Face-Recognition-Train-YML-Python-master/dataSet/'
        Ids,faces=faceTrainer(path)
        recognizer.train(faces,np.array(Ids))
        recognizer.write('F:/Python_Projects/Face_Recognition/Face-Recognition-Train-YML-Python-master/trainningData.yml')
        cv2.destroyAllWindows()
    elif(action==4):
        recognizer.read('F:/Python_Projects/Face_Recognition/Face-Recognition-Train-YML-Python-master/trainningData.yml')
        path = input("Enter path: ")
        faceRecogition(path)
        while(True):
            check = input("Predictions are Right or Wrong [r/w]: ")
            if(check=='w' or check=='W'):
                print("I'm Sorry!\nPlease Train me by this Ids\n\n")
                trainAgain(path)
                break
            elif(check=='r' or check=='R'):
                print("I Know I'm smarter, because I developed by toppers of IT.")
                break
            else:
                print("!!  Wrong input  !!\nEnter Againg...")
    elif(action==5):
        retrainMode();
    elif(action==7):
        path = input("Enter path: ")
        trainAgain(path)
    elif(action==6):
        break
    else:
        print('\n!!Wrong Choice!!\n\n')    