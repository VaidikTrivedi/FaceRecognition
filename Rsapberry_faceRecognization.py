# -*- coding: utf-8 -*-
"""
Created on Mon Mar 18 10:12:50 2019

@author: Vaidik
"""

import cv2
import os
import numpy as np
from PIL import Image
import time
import math
choice = 0

faceDetect = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("trainningData.yml")
names = ['','', '', '', '', '', '', '', '', '', 'Unkown']
#print("All Names: ", names[0], names[1], names[10])
id=0
font=cv2.FONT_HERSHEY_COMPLEX_SMALL

path = 'F:/Python_Projects/Face_Recognition/Id_Names/'
namePaths = [os.path.join(path,f) for f in os.listdir(path)]
for name in namePaths:
        #print("name: ", name)
        Id = int(os.path.split(name)[-1].split('.')[1])
        print("ID: ", Id)
        nm = str(os.path.split(name)[1].split('user.'+str(Id)+'.')[1])
        print("nm: ", nm)
        names.insert(Id, str(nm))

print("Names: ", names)

def faceRecognizor(img):
    #cam = cv2.VideoCapture(0);
    #ret,img=cam.read()
    #print("Image: ", img)
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces=faceDetect.detectMultiScale(gray,1.3,5);
    for(x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),2)
        id,conf=recognizer.predict(gray[y:y+h,x:x+w])
        cv2.putText(img, str(names[id])+": "+str(int(conf)), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,255,0), 3)
        cv2.imshow("Face",img)
    cv2.waitKey(10)
    cv2.destroyAllWindows()
    
def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        pass
 
    try:
        import unicodedata
        unicodedata.numeric(s)
        return True
    except (TypeError, ValueError):
        pass
 
    return False
 

def addFace(): 
    flag=False 
    IdUse = False     
    Id=input("Enter ID: ")
    name=input("Enter Name: ")
    if(is_number(Id)):
        flag=True
        folderName = 'user.'+str(Id)+'.'+str(name)
    if(os.path.exists(path+folderName)):
        print("\n !! Id is already in used !!\n")
        IdUse = True
    else:
        print("Making folder by name: ", folderName)
        os.mkdir(path+folderName)
        print("Taking Images...")
    if(is_number(Id) and name.isalpha() and IdUse==False):
        cam = cv2.VideoCapture(0)
        harcascadePath = "haarcascade_frontalface_default.xml"
        detector=cv2.CascadeClassifier(harcascadePath)
        sampleNum=0
        while(True):
            ret, img = cam.read()
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = detector.detectMultiScale(gray, 1.3, 5)
            for (x,y,w,h) in faces:
                cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)        
                #incrementing sample number 
                sampleNum=sampleNum+1
                #saving the captured face in the dataset folder TrainingImage
                cv2.imwrite("TrainingImage\ "+name +"."+Id +'.'+ str(sampleNum) + ".jpg", gray[y:y+h,x:x+w])
                #display the frame
                cv2.putText(img,str(sampleNum),(x,y+h), font, 1,(255,255,255),2)
                cv2.imshow('frame',img)
            #wait for 100 miliseconds 
            if cv2.waitKey(100) & 0xFF == ord('q'):
                break
            # break if the sample number is morethan 100
            elif sampleNum>60:
                break
        cam.release()
        cv2.destroyAllWindows() 

    else:
        if(is_number(Id)):
            print("Enter Numeric Id")
        if(name.isalpha()):
            print("Enter Alphabetical Name")
    if(flag==True):
        print("Image Taking Complete.")    
    else:
        print("Enter right input")

def TrainImages():
    print("Trainning Images...")
    recognizer = cv2.face_LBPHFaceRecognizer.create()#recognizer = cv2.face.LBPHFaceRecognizer_create()#$cv2.createLBPHFaceRecognizer()
    #harcascadePath = "haarcascade_frontalface_default.xml"
    #detector =cv2.CascadeClassifier(harcascadePath)
    faces,Id = getImagesAndLabels("TrainingImage")
    recognizer.train(faces, np.array(Id))
    recognizer.write("TrainingImageLabel\Trainner.yml")
    print("Image Trainning Complete.")

def getImagesAndLabels(path):
    print("Getting Images And Labels...")
    #get the path of all the files in the folder
    imagePaths=[os.path.join(path,f) for f in os.listdir(path)] 
    #print(imagePaths)
    
    #create empth face list
    faces=[]
    #create empty ID list
    Ids=[]
    #now looping through all the image paths and loading the Ids and the images
    for imagePath in imagePaths:
        #loading the image and converting it to gray scale
        pilImage=Image.open(imagePath).convert('L')
        #Now we are converting the PIL image into numpy array
        imageNp=np.array(pilImage,'uint8')
        #getting the Id from the image
        Id=int(os.path.split(imagePath)[-1].split(".")[1])
        # extract the face from the training image sample
        faces.append(imageNp)
        Ids.append(Id)  
    print("All Labels and Images Collected.")
    return faces,Ids   
  

def RecognizeImages():
    print("Names: ", names)
    print("Tracking Images...")
    recognizer = cv2.face.LBPHFaceRecognizer_create()#cv2.createLBPHFaceRecognizer()
    #recognizer.read("F:/Python_Projects/Face_Recognition/Face-Recognition-Train-YML-Python-master/trainningData.yml")
    recognizer.read("TrainingImageLabel\Trainner.yml")
    harcascadePath = "haarcascade_frontalface_default.xml"
    faceCascade = cv2.CascadeClassifier(harcascadePath);    
    cam = cv2.VideoCapture(0)
    font = cv2.FONT_HERSHEY_SIMPLEX    
    tt = "" 
    while True:
        ret, im =cam.read()
        gray=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
        faces=faceCascade.detectMultiScale(gray, 1.2,5)    
        for(x,y,w,h) in faces:
            cv2.rectangle(im,(x,y),(x+w,y+h),(225,0,0),2)
            Id, conf = recognizer.predict(gray[y:y+h,x:x+w]) 
            conf = int(100-conf)
            print("Id: ",Id, " || Name: ", names[Id] ," || Confidence: ", conf, "%")                                  
            if(conf >= 60):
                #aa=df.loc[df['Id'] == Id]['Name'].values
                #print("In If")
                tt=str(names[Id]) 
            elif(conf < 60 and conf > 30):
                tt = names[Id]
                print("Low Confidence with Id: ", Id)
                #noOfFile=len(os.listdir("ImagesUnknown"))+1
                #cv2.imwrite("ImagesUnknown\Image"+str(noOfFile) + ".jpg", im[y:y+h,x:x+w])            
            else:
                Id='Unknown'                
                tt=str(Id) 
            cv2.putText(im,tt+"-"+str(conf)+"%",(x,y+h), font, 1,(255,255,255),2)           
        cv2.imshow('Faces',im) 
        if (cv2.waitKey(1)==ord('q')):
            break
    cam.release()
    cv2.destroyAllWindows()
    print("Tracking Complete.\n\n")

'''   
while(choice!=4):
    
    print("\n1. Add New Face\n2. Train Images\n3. Recognize Face\n4. End")
    choice = int(input("Enter Choice: "))
    if(choice==1):
        addFace()
    elif(choice==2):
        TrainImages()
    elif(choice==3):
        RecognizeImages()
    elif(choice==4):
        break
    else:
        print("Wrong Choice")
'''

cap = cv2.VideoCapture(0)
while(1):
        
    try:  #an error comes if it does not find anything in window as it cannot find contour of max area
          #therefore this try error statement
          
        ret, frame = cap.read()
        frame=cv2.flip(frame,1)
        image = frame
        kernel = np.ones((3,3),np.uint8)
        
        #define region of interest
        roi=frame[100:300, 100:300]
        
        
        cv2.rectangle(frame,(100,100),(300,300),(0,255,0),0)    
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        
        
         
    # define range of skin color in HSV
        lower_skin = np.array([0,20,70], dtype=np.uint8)
        upper_skin = np.array([20,255,255], dtype=np.uint8)
        
     #extract skin colur imagw  
        mask = cv2.inRange(hsv, lower_skin, upper_skin)
        
   
        
    #extrapolate the hand to fill dark spots within
        mask = cv2.dilate(mask,kernel,iterations = 4)
        
    #blur the image
        mask = cv2.GaussianBlur(mask,(5,5),100) 
        
        
        
    #find contours
        _,contours,hierarchy= cv2.findContours(mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    
   #find contour of max area(hand)
        cnt = max(contours, key = lambda x: cv2.contourArea(x))
        
    #approx the contour a little
        epsilon = 0.0005*cv2.arcLength(cnt,True)
        approx= cv2.approxPolyDP(cnt,epsilon,True)
       
        
    #make convex hull around hand
        hull = cv2.convexHull(cnt)
        
     #define area of hull and area of hand
        areahull = cv2.contourArea(hull)
        areacnt = cv2.contourArea(cnt)
      
    #find the percentage of area not covered by hand in convex hull
        arearatio=((areahull-areacnt)/areacnt)*100
    
     #find the defects in convex hull with respect to hand
        hull = cv2.convexHull(approx, returnPoints=False)
        defects = cv2.convexityDefects(approx, hull)
        
    # l = no. of defects
        l=0
        
    #code for finding no. of defects due to fingers
        for i in range(defects.shape[0]):
            s,e,f,d = defects[i,0]
            start = tuple(approx[s][0])
            end = tuple(approx[e][0])
            far = tuple(approx[f][0])
            #pt= (100,180)
            #print("PT: ",pt)
            
            # find length of all sides of triangle
            a = math.sqrt((end[0] - start[0])**2 + (end[1] - start[1])**2)
            b = math.sqrt((far[0] - start[0])**2 + (far[1] - start[1])**2)
            c = math.sqrt((end[0] - far[0])**2 + (end[1] - far[1])**2)
            s = (a+b+c)/2
            ar = math.sqrt(s*(s-a)*(s-b)*(s-c))
            
            #distance between point and convex hull
            d=(2*ar)/a
            
            # apply cosine rule here
            angle = math.acos((b**2 + c**2 - a**2)/(2*b*c)) * 57
            
        
            # ignore angles > 90 and ignore points very close to convex hull(they generally come due to noise)
            if angle <= 90 and d>30:
                l += 1
                cv2.circle(roi, far, 3, [255,0,0], -1)
            
            #draw lines around hand
            cv2.line(roi,start, end, [0,255,0], 2)
            
            
        l+=1
        
        #print corresponding gestures which are in their ranges
        font = cv2.FONT_HERSHEY_SIMPLEX
        if l==1:
            if areacnt<2000:
                cv2.putText(frame,'Put hand in the box',(0,50), font, 2, (0,0,255), 3, cv2.LINE_AA)
            else:
                if arearatio<12:
                    cv2.putText(frame,'0',(0,50), font, 2, (0,0,255), 3, cv2.LINE_AA)
                elif arearatio<17.5:
                    cv2.putText(frame,'Best of luck',(0,50), font, 2, (0,0,255), 3, cv2.LINE_AA)
                   
                else:
                    cv2.putText(frame,'1',(0,50), font, 2, (0,0,255), 3, cv2.LINE_AA)
                    print("Capturing Image after 5 seconds...")
                    time.sleep(5)
                    cv2.imshow("Captured...", image)
                    faceRecognizor(image)
                    
        elif l==2:
            cv2.putText(frame,'2',(0,50), font, 2, (0,0,255), 3, cv2.LINE_AA)
            
        elif l==3:
         
              if arearatio<27:
                    cv2.putText(frame,'3',(0,50), font, 2, (0,0,255), 3, cv2.LINE_AA)
              else:
                    cv2.putText(frame,'ok',(0,50), font, 2, (0,0,255), 3, cv2.LINE_AA)
                    
        elif l==4:
            cv2.putText(frame,'4',(0,50), font, 2, (0,0,255), 3, cv2.LINE_AA)
            
        elif l==5:
            cv2.putText(frame,'5',(0,50), font, 2, (0,0,255), 3, cv2.LINE_AA)
            
        elif l==6:
            cv2.putText(frame,'reposition',(0,50), font, 2, (0,0,255), 3, cv2.LINE_AA)
            
        else :
            cv2.putText(frame,'reposition',(10,50), font, 2, (0,0,255), 3, cv2.LINE_AA)
            
        #show the windows
        cv2.imshow('mask',mask)
        cv2.imshow('frame',frame)
    except:
        pass
        
    
    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break
    
cv2.destroyAllWindows()
cap.release()
 