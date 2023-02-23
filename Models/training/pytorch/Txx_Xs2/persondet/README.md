1.版本要求
同君正技术人员确认环境

2.数据的准备 
以COCO_2017为例，COCO2017数据集下载好并解压，annotation 文件的格式为 .json 格式
下载地址:
  图片：
    http://images.cocodataset.org/zips/train2017.zip
    http://images.cocodataset.org/zips/test2017.zip
    http://images.cocodataset.org/zips/val2017.zip
  标注:
    http://images.cocodataset.org/annotations/stuff_annotations_trainval2017.zip
    http://images.cocodataset.org/annotations/image_info_test2017.zip
    http://images.cocodataset.org/annotations/annotations_trainval2017.zip

生成数据的脚本在COCO_forYOLO文件夹下。
  (1)运行 python batch_split_annotation_foryolo.py (注意修改程序中的coco路径'coco_data_dir=')；
  (2)运行完会在coco路径下生成person/data/images，person/data/ImageSets，person/data/labels三个文件夹；将ImageSets下的train2017.txt,val2017.txt,test2017.txt放入persondet/data/coco文件夹下,txt文件里存储图片的绝对路径
	
3.训练
 (1)可通过kmeans_anchor.py得到对应自己数据的anchor并修改cfg/persondet.cfg内的anchor参数，也可用原始anchor参数;
 (2)训练配置参数可通过python3 train.py -h查看选取所需参数，示例如下，若需要其他操作可选择相应的参数;
    python train.py --batch-size 16 --accumulate 4 --cfg cfg/tmp.cfg --data data/coco.data --img-size 416 --weights '' --device 0
 (3)train.py里wdir = 'weights/32bit' + os.sep  # weights dir设置保存模型路径
    模型mAP>0.4保留，也可自己设置 train.py: if epoch > 0 and fi > 0.4:  #mAP>0.4
    				  	     torch.save (chkpt,wdir + '/backup%g.pt' % epoch) 

4.测试模型与检测图片
通过python3 detect.py -h查看选取所需参数,通过设置source可检测图片和视频,检测结果默认保存在output文件下
- Image:  `--source file.jpg`
- Video:  `--source file.mp4`
- Directory:  `--source dir/`
示例如下，若需要其他操作可选择相应的参数;
python3 detect.py --cfg cfg/tmp.cfg --name data/coco.names --weights weights/best.pt --source tmp --img-size 416
注：检测的模型配置一定要和训练时候的配置一致

5.网络量化需改cfg/*.cfg文件
float:
  ##0 or 1
  is_quantize = 0
  ## 32/8/4	    
  bita = 32
  bitw = 32
8bit:
  is_quantize = 1	    
  bita = 8
  bitw = 8
4bit:
  is_quantize = 1	    
  bita = 4
  bitw = 4

6.模型转换.onnx
 生成.onnx文件：python3 convert_onnx.py
    input：指定.cfg和.pt路径
    output：.onnx,位置和.pt在同一目录

7.测试精度结果
  Class	   Images   Targets	    P	      R	  mAP@0.5	 F1
    all     5e+03   1.1e+04     0.513     0.504     0.483     0.508