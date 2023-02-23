import numpy as np

import cv2

a = np.zeros((240,320,1))
print(a.shape)

b = np.ones((120,240,1))
print(b.shape)
scale = 320/240

image = cv2.resize(b, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
print(image.shape)

w = image.shape[1]
h = image.shape[0]
image = np.reshape(image,(h,w,1))

left = np.random.uniform(320 - w)
top = np.random.uniform(240 - h)
                    
# convert to integer rect x1,y1,x2,y2
rect = np.array([int(left), int(top), int(left + w), int(top + h)])
a[rect[1]:rect[3], rect[0]:rect[2],:]=image


print("aaa",a.shape)

cv2.imshow("a",a)
cv2.waitKey(0)


b = np.ones((480,720,1))
print(b.shape)

h_scale = 480 / 240
w_scale = 720 / 320
scale = max(h_scale,w_scale)


a = np.zeros((int(240*scale),int(320*scale),1))
print("aaaa",a.shape)
a_w = a.shape[1]
a_h = a.shape[0]
left = np.random.uniform(a_w - 720)
top = np.random.uniform(a_h - 480)
                    
# convert to integer rect x1,y1,x2,y2
rect = np.array([int(left), int(top), int(left + 720), int(top + 480)])
a[rect[1]:rect[3], rect[0]:rect[2],:]=b

cv2.imshow('aa',a)
cv2.waitKey(0)



