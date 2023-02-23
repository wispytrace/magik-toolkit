import cv2
import numpy as np

def save_jzdl_imginput(save_path, input):
    f2 = open(save_path,'w')
    img1 = np.reshape(input,[-1])
    f2.write('unsigned char image[')
    f2.write(str(len(img1)))
    f2.write('] = {')       
    for index in range(len(img1)) :
        f2.write(str(img1[index]))
        f2.write(' ,')
    f2.write('};')
    f2.close()     

img = cv2.imread("./facePerson_416x416.jpg", 1)
h,w,c = img.shape
bak = np.zeros((h,w,4), np.uint8)
bak[:,:,:3] = img
img = bak
save_jzdl_imginput("facePerson_416x416.h", img)

