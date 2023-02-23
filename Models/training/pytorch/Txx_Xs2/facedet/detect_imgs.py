"""
This code is used to batch detect images in a folder.
"""
import argparse
import os
import sys

import cv2
import numpy as np

from vision_quantize.ssd.config.fd_config import define_img_size  #quantize
from vision_quantize.ssd.mb_tiny_fd import create_mb_tiny_fd, create_mb_tiny_fd_predictor

class_dict = {0:"BackGround",1:"person",2:"face",3:"1111",4:"head"}
parser = argparse.ArgumentParser(
    description='detect_imgs')

parser.add_argument('--net_type', default="slim", type=str,
                    help='The network architecture ,optional: RFB (higher precision) or slim (faster)')
parser.add_argument('--input_size', default=320, type=int,
                    help='define network input size,default optional value 128/160/320/480/640/1280')
parser.add_argument('--threshold', default=0.4, type=float,
                    help='score threshold')
parser.add_argument('--candidate_size', default=1500, type=int,
                    help='nms candidate size')
parser.add_argument('--path', default="./imgs", type=str,
                    help='imgs dir')
parser.add_argument('--test_device', default="cuda:0", type=str,
                    help='cuda:0 or cpu')  #cuda:0
args = parser.parse_args()
define_img_size(args.input_size)  # must put define_img_size() before 'import create_mb_tiny_fd, create_mb_tiny_fd_predictor'
def yuv2bgr(filename, height, width, startfrm):
    """
    :param filename: 待处理 YUV 视频的名字
    :param height: YUV 视频中图像的高
    :param width: YUV 视频中图像的宽
    :param startfrm: 起始帧
    :return: None
    """
    fp = open(filename, 'rb')

    framesize = height * width * 3 // 2 # 一帧图像所含的像素个数
    h_h = height // 2
    h_w = width // 2

    fp.seek(0, 2) # 设置文件指针到文件流的尾部
    ps = fp.tell() # 当前文件指针位置
    numfrm = ps // framesize # 计算输出帧数
    fp.seek(framesize * startfrm, 0)

    for i in range(numfrm - startfrm):
        Yt = np.zeros(shape=(height, width), dtype='uint8', order='C')
        Ut = np.zeros(shape=(h_h, h_w), dtype='uint8', order='C')
        Vt = np.zeros(shape=(h_h, h_w), dtype='uint8', order='C')

    for m in range(height):
        for n in range(width):
            Yt[m, n] = ord(fp.read(1))
    for m in range(h_h):
        for n in range(h_w):
            Ut[m, n] = ord(fp.read(1))
    for m in range(h_h):
        for n in range(h_w):
            Vt[m, n] = ord(fp.read(1))

    img = np.concatenate((Yt.reshape(-1), Ut.reshape(-1), Vt.reshape(-1)))
    img = img.reshape((height * 3 // 2, width)).astype('uint8') # YUV 的存储格式为：NV12（YYYY UV）

    # 由于 opencv 不能直接读取 YUV 格式的文件, 所以要转换一下格式
    bgr_img = cv2.cvtColor(img, cv2.COLOR_YUV2BGR_NV12) # 注意 YUV 的存储格式
    scale = 320/width
    bgr_img = cv2.resize(bgr_img, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC) 

    cv2.imshow('orig_image', bgr_img)
    cv2.waitKey(0)
    ##cv2.imwrite('yuv2bgr/%d.jpg' % (i + 1), bgr_img)
    print("Extract frame %d " % (i + 1))

    fp.close()
    print("job done!")
    return bgr_img
  #quantize
#from vision.ssd.mb_tiny_RFB_fd import create_Mb_Tiny_RFB_fd, create_Mb_Tiny_RFB_fd_predictor

result_path = "./detect_imgs_results"
label_path = "./models/voc-model-labels.txt"
test_device = args.test_device



def read_img(path, w,h):
    with open(path,'r') as f:
        data = f.readline()
        data = data.strip(' ').split(',')[:]
        print(len(data))
        data = list(map(np.uint8,data))
        ##print(data)
        ##data_p = np.reshape(np.array(data),(1,17,17,1))
        data_p = np.reshape(np.array(data),(w,h,3))
        ##data_p.astype(np.uint8)
        print('data_p shape :{}'.format(data_p.shape))
        
    ##cv2.imshow('data_p', data_p)
    ##cv2.waitKey(0)
    return data_p


class_names = [name.strip() for name in open(label_path).readlines()]
if args.net_type == 'slim':

    model_path = "./models/4bit_clip2_model/slim-Epoch-208-Loss-3.0186328684842145.pth"
    #model_path = "./models/640_face_4bit_new/slim-Epoch-33-Loss-2.6307342493975603.pth"
    net = create_mb_tiny_fd(len(class_names), is_test=True, device=test_device)
    predictor = create_mb_tiny_fd_predictor(net, candidate_size=args.candidate_size, device=test_device)
else:
    print("The net type is wrong!")
    sys.exit(1)
net.load(model_path)

#output_name_and_params(net)
#print("weight::",net['base_net.0.Conv2d.weight'])
listdir = os.listdir(args.path)
#listdir.sort(key=lambda s:int(s.split("_")[-1].split(".")[0]))
test_Path = args.path +"/facebox.txt"
sum = 0

#exit()
# cv2.namedWindow('image')
# cv2.setMouseCallback('image',OnMouseAction)
#f = open("error.txt",'w')

for file_path in listdir:
    img_path = os.path.join(args.path, file_path)
   
    #orig_image = yuv2bgr(img_path,1080,1920,0)
    print(file_path)
    image = cv2.imread(img_path)
    orig_image = image
    #image2 = orig_image.copy()
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    #image = read_img("1920x1080.data",1080,1920)
    #orig_image = image
    boxes, labels, probs = predictor.predict(image, args.candidate_size / 2, args.threshold)
    sum += boxes.size(0)
    if 1:
        for i in range(boxes.size(0)):
            box = boxes[i, :]
            print("box:[%d]:[%.2f,%.2f,%.2f,%.2f] score:%.2f"%(i,box[0], box[1], box[2], box[3],probs[i]))
            cv2.rectangle(orig_image, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 0, 255), 2)
            # label = f"""{voc_dataset.class_names[labels[i]]}: {probs[i]:.2f}"""
            probss = f"{probs[i]:.2f}"
            label = class_dict[int(labels[i])]
            cv2.putText(orig_image, probss, (int(box[0]), int(box[1] - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            txtr = file_path + " "+" ".join(str(int(i)) for i in [box[0], box[1], box[2], box[3]]) +"\n"
            if float(probs[i]) > 0.8:
                txt_w = file_path + " "+" ".join(str(int(i)) for i in [box[0], box[1], box[2], box[3]]) +"\n"
        
            else:
                continue
        
        #f.write(txt_w)
        #cv2.putText(image, str(boxes.size(0)), (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
        cv2.imshow('old orig_image', orig_image)
        cv2.waitKey(0)
    #print(f"Found {len(probs)} faces. The output image is {result_path}")
print(sum)
