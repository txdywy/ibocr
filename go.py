import cv2
import numpy as np
from pprint import pprint
import sys

#######   training part    ############### 
samples = np.loadtxt('ib_samples.data',np.float32)
responses = np.loadtxt('ib_responses.data',np.float32)
responses = responses.reshape((responses.size,1))

model = cv2.KNearest()
model.train(samples,responses)

############################# testing part  #########################

#im = cv2.imread('st1.png')
im_gray = cv2.imread('st1.png', cv2.CV_LOAD_IMAGE_GRAYSCALE)
#(thr, im_bw) = cv2.threshold(im_gray, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
thr = 180 # to make font with proper with for counter finding
im_bw = cv2.threshold(im_gray, thr, 255, cv2.THRESH_BINARY)[1]
cv2.imwrite('bw_image.png', im_bw)

im = cv2.imread('bw_image.png')
height, width, depth = im.shape
height, width = height * 3, width * 3
print '=======', height, width 
im = cv2.resize(im,(width, height))

white = (255,255,255)
def dl(x, l=0, h=930):
    cv2.line(im, (x, l), (x, h), white)
    cv2.line(im, (x+1, l), (x+1, h), white)

# add white split up lines between different close columns
c = 7
x = 89
x0 = 10
while c > 0:
    if c == 2:
        x -= 2
    if c == 1:
        x += 8
    dl(x)
    dl(x+23)
    half = 1000
    off = 10
    if c == 1:
        off = 1
    dl(x+off, half, height)
    dl(x+23+off, half, height)
    x += 172
    c -= 1
    ww = 52
    if c == 0:
        ww = 55
    for i in range(x0, x0+ww):
        dl(i)
    for i in range(x0, x0+59):
        dl(i, half, height)
    x0 +=172



out = np.zeros(im.shape,np.uint8)
gray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
thresh = cv2.adaptiveThreshold(gray,255,1,1,11,2)

contours,hierarchy = cv2.findContours(thresh,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)

result = []
for cnt in contours:
    if cv2.contourArea(cnt)>10:
        [x,y,w,h] = cv2.boundingRect(cnt)
        if  h>28 and w >15 and w < 35:
            cv2.rectangle(im,(x,y),(x+w,y+h),(0,255,0),2)
            roi = thresh[y:y+h,x:x+w]
            roismall = cv2.resize(roi,(10,10))
            roismall = roismall.reshape((1,100))
            roismall = np.float32(roismall)
            retval, results, neigh_resp, dists = model.find_nearest(roismall, k = 1)
            #string = str(int((results[0][0])))
            string = chr(int(results[0][0]))
            #cv2.putText(out,string,(x,y+h),0,1,(0,255,0))
            cv2.putText(im,string,(x,y+h),0,1,(0,0,255))
            result.append([ord(string), x, y])

result = np.array(result)
print result
ind = np.lexsort((result[:,1],result[:,2]))  
print result[ind]
result = result[ind]
result = result.tolist()
result = [(chr(x[0]), x[1],x[2]) for x in result]

pprint(result)
size = len(result)
if size != 224 * 3:
    print '=========================error with size %s/672===================' % size
    sys.exit(0)
print size/3.0
c = 0
keys = []
while c < 224 * 3:
    s = []
    s.append(result[c])
    s.append(result[c+1])
    s.append(result[c+2])
    s = sorted(s, key=lambda x: x[1])
    s = ''.join([i[0] for i in s])
    keys.append(s)
    c += 3
print keys


cv2.imshow('im',im)
#cv2.imshow('out',out)
cv2.waitKey(0)
