import sys

import numpy as np
import cv2

im_gray = cv2.imread('st1.png', cv2.CV_LOAD_IMAGE_GRAYSCALE)
(thr, im_bw) = cv2.threshold(im_gray, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
cv2.imwrite('bw_image.png', im_bw)

im = cv2.imread('bw_image.png')
height, width, depth = im.shape
im = cv2.resize(im,(width*3, height*3))
im3 = im.copy()

gray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)

blur = cv2.GaussianBlur(gray,(5,5),0)
thresh = cv2.adaptiveThreshold(blur,255,1,1,11,2)
print thresh
#################      Now finding Contours         ###################

contours,hierarchy = cv2.findContours(thresh,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
print '-------------', len(contours)

samples =  np.empty((0,100))
responses = []
keys = [i for i in range(48,58) + range(97,123)] #ascii code 0-9, a=z

for c in contours:
    [x,y,w,h] = cv2.boundingRect(c)
    #if  h>20 and w >10 and w < 40:
    cv2.rectangle(im,(x,y),(x+w,y+h),(152,251,152),2)


for cnt in contours:
    if cv2.contourArea(cnt)>10:
        [x,y,w,h] = cv2.boundingRect(cnt)

        if  h>25 and w >10 and w < 35:
            cv2.rectangle(im,(x,y),(x+w,y+h),(0,0,255),2)
            roi = thresh[y:y+h,x:x+w]
            roismall = cv2.resize(roi,(10,10))
            cv2.imshow('norm',im)
            key = cv2.waitKey(0)

            if key == 27:  # (escape to quit)
                sys.exit()
            elif key in keys:
                cv2.rectangle(im,(x,y),(x+w,y+h),(0,255,255),2)
                print '-------->[ascii:%s] [key:%s]' % (key, chr(key))
                responses.append(int(key))
                sample = roismall.reshape((1,100))
                samples = np.append(samples,sample,0)

responses = np.array(responses,np.float32)
responses = responses.reshape((responses.size,1))
print "training complete"
print samples
print responses
np.savetxt('generalsamples.data',samples)
np.savetxt('generalresponses.data',responses)
