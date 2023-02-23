import cv2
import numpy as np 
import sys

argvs = (sys.argv)
if len(sys.argv) != 6:
    print ("img_path w h c flag")
    exit(0)
img_path = argvs[1]
w = int(argvs[2])
h = int(argvs[3])
c = int(argvs[4])
flag = argvs[5]

def save_imginput(save_path, input, is_show=0):
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

if flag == "img":
    print ("img generate img_input.h")
    img = cv2.imread(img_path, 1)
#    img = np.reshape(img, (1080,810,3))
    h,w,c = img.shape
    print (h,w,c)
elif flag == "bin":
    print ("bin generate img_input.h")
    img = np.fromfile(img_path, np.uint8)
    img = np.reshape(img, (h, w, c))

img_res = np.zeros((h,w,4), np.uint8)
img_res[:,:,0:3] = img
save_imginput("img_input.h", img_res)
cv2.imshow("img_res", img_res)
cv2.waitKey(0)


