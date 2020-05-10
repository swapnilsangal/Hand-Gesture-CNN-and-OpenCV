# -*- coding: utf-8 -*-
"""
Created on Wed May  6 23:26:03 2020

@author: Swapnil Sangal
"""

import cv2
import tensorflow as tf

#Dictionary for Class Lables to the corresponding output layer for the CNN
class_label_lookup={0:'Zero',1:'One',2:'Two',3:'Five',4:'I',5:'L',6:'U',7:'Unknown'}

#Load the pretrained tensorflow Classification CNN model
loaded_model = tf.keras.models.load_model('Models/M1_D1')

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


while(True):

    ret, frame = cap.read()

    frame = cv2.flip(frame,1)

    if count%50==0:
        count=0
        cv2.rectangle(frame, rect_pt1, rect_pt2, color=(0,255,0))

        cropped = frame[rect_pt1[1]:rect_pt2[1] , rect_pt1[0]:rect_pt2[0]]

        cropped_gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)

        cropped_gray_inpt = cv2.resize(cropped_gray,(input_height,input_width))
        cropped_gray_inpt = cropped_gray_inpt/255
        cropped_gray_inpt = cropped_gray_inpt.reshape(-1,input_width,input_height,1)

        predicted_class = loaded_model.predict_classes(cropped_gray_inpt)
        class_name = class_label_lookup[predicted_class[0]]


    else:
        cv2.rectangle(frame, rect_pt1, rect_pt2, color=(255,0,0))
    count = count+1

    frame = cv2.putText(frame, 'Class:'+str(class_name), org=(rect_pt1[0],rect_pt1[1]+25),fontFace=cv2.FONT_HERSHEY_SIMPLEX,fontScale=0.3,color=(255,0,0))

    cv2.namedWindow('frame', cv2.WINDOW_NORMAL)
    
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
    
    
    cv2.imshow('frame',frame)


    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
