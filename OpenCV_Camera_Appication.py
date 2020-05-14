# -*- coding: utf-8 -*-
"""
Created on Wed May  6 23:26:03 2020
@author: Swapnil Sangal
"""

import cv2
import tensorflow as tf
import numpy as np
from datetime import datetime

#Dictionary for Class Lables to the corresponding output layer for the CNN M1_D3
class_label_lookup={0:'Zero',1:'One',2:'Two',3:'Three',4:'Four',5:'Five',6:'A',7:'I',8:'Undefined'}

#Load the pretrained tensorflow Classification CNN model
loaded_model = tf.keras.models.load_model('Models/M1_D3')

#Initialising Input dimension on which the CNN was trained 
#Converts image to specified dimension before Classification
input_height = 64
input_width = 64

cap = cv2.VideoCapture(0)
width  = int(cap.get(3))  # float
height = int(cap.get(4)) # float

rect_pt1 = (int(width*0.70),int(height*0.20))
rect_pt2 = (int(width),int(height*0.80))

count=0

fourcc = cv2.VideoWriter_fourcc(*'XVID')

window_no = 1

last_n = 5
last_n_detections = [np.NaN]*last_n

while(True):

    ret, frame = cap.read()

    frame = cv2.flip(frame,1)
    
    # Running the Classifier at every n'th' Frame
    if count%10==0:
        count=0
        
        #Cropping the Region of Interest to feed to the CNN
        cropped = frame[rect_pt1[1]:rect_pt2[1] , rect_pt1[0]:rect_pt2[0]]
        cropped_gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
        '''
        #Implementing Image Enhancement Techniques here
        kernel = np.array([[0,-1,0],[-1,5,-1],[0,-1,0]])       
        cropped_gray_sharp = cv2.filter2D(cropped_gray, -1, kernel)    
        cropped_gray = cropped_gray_sharp
        '''  
        cv2.rectangle(frame, rect_pt1, rect_pt2, color=(0,255,0))
        cropped_gray_inpt = cv2.resize(cropped_gray,(input_height,input_width))
        cropped_gray_inpt = cropped_gray_inpt/255
        cropped_gray_inpt = cropped_gray_inpt.reshape(-1,input_width,input_height,1)

        predicted_class = loaded_model.predict_classes(cropped_gray_inpt)
        class_name = class_label_lookup[predicted_class[0]]
        last_n_detections = last_n_detections+[class_name]
        last_n_detections = last_n_detections[-last_n:]
        #print(last_n_detections)

    else:
        cv2.rectangle(frame, rect_pt1, rect_pt2, color=(255,0,0))
    # Frame Count Incrementer
    count = count+1

    cv2.namedWindow('Hand Gesture Project', cv2.WINDOW_NORMAL)
    
        
    k = cv2.waitKey(1)
    if k == ord('q'):
        break
    elif k==ord('h'):
        window_no = 2
    elif k==ord('n'):
        window_no = 1
    
    #Show the whole frame
    # Blurring the rest of the window
    blur_region1 = frame[0:height, 0:int(width*0.70)]
    blur_region2 = frame[0:int(height*0.20), int(width*0.70):width]
    blur_region3 = frame[int(height*0.80)+1:height, int(width*0.70):width]
    
    blur_frame = cv2.GaussianBlur(blur_region1, (51,51), -5)
    frame[0:height, 0:int(width*0.70)] = blur_frame
    
    blur_frame = cv2.GaussianBlur(blur_region2, (51,51), -5)
    frame[0:int(height*0.20), int(width*0.70):width] = blur_frame
    
    blur_frame = cv2.GaussianBlur(blur_region3, (51,51), -5)
    frame[int(height*0.80)+1:height, int(width*0.70):width] = blur_frame
    
    #Adding the text for detection
    frame = cv2.putText(frame, 'Class:'+str(class_name), org=(rect_pt1[0],rect_pt1[1]+25),fontFace=cv2.FONT_HERSHEY_SIMPLEX,fontScale=0.3,color=(255,0,0))
    
    #Adding Datetime
    now = datetime.now()
    now_date_text = now.strftime("%d %B,%Y")
    now_time_text = now.strftime("%H : %M : %S")
    welcome_text = 'Welcome'
    date_textsize = cv2.getTextSize(now_date_text, cv2.FONT_HERSHEY_SIMPLEX, 0.3, 2)[0]
    time_textsize = cv2.getTextSize(now_time_text, cv2.FONT_HERSHEY_SIMPLEX, 0.3, 2)[0]
    welcome_textsize = cv2.getTextSize(welcome_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
    
    frame = cv2.putText(frame, now_date_text, org=(10,10+int(date_textsize[1]/2)),fontFace=cv2.FONT_HERSHEY_SIMPLEX,fontScale=0.3,color=(255,0,0))
    frame = cv2.putText(frame, now_time_text, org=(width-int(time_textsize[0])-10,10+int(time_textsize[1]/2)),fontFace=cv2.FONT_HERSHEY_SIMPLEX,fontScale=0.3,color=(255,0,0))
    frame = cv2.putText(frame, welcome_text, org=(int((width/2)-welcome_textsize[0]/2),10+int(welcome_textsize[1]/2)),fontFace=cv2.FONT_HERSHEY_SIMPLEX,fontScale=0.3,color=(255,0,0))
    
    #Show the Frame
    cv2.imshow('Hand Gesture Project',frame)
    
    '''
    #Show the cropped and processed frame only
    cropped_gray = cv2.putText(cropped_gray, text, org=(10,20),fontFace=cv2.FONT_HERSHEY_SIMPLEX,fontScale=0.5,color=(255,0,0))
    cv2.imshow('Hand Gesture Project',cropped_gray)
    '''    
    
    

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
