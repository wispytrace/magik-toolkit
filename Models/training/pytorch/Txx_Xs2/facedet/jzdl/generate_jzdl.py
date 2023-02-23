import cv2
import numpy as np

def save_jzdl_imginput(save_path, input, is_show=0):
    f2 = open(save_path,'w')
    img1 = np.reshape(input,[-1])
    f2.write('unsigned char image[')
    f2.write(str(len(img1)))
    f2.write('] = {')
                
    for index in range(len(img1)) :
        f2.write(str(img1[index]))
        f2.write(' ,')
            #    print(img1[index])
    f2.write('};')
    f2.close()     
    if is_show:
        cv2.imshow("img", input)
        cv2.waitKey(0)   

img_202 = cv2.imread("5.jpg", 1)
img_202 = cv2.cvtColor(img_202, cv2.COLOR_BGR2RGB)
print (img_202.shape)
save_jzdl_imginput("img_input.h", img_202, 0)

