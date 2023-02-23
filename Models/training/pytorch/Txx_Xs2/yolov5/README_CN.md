1.版本要求
python
pytorch
cuda
cudnn
gcc
以上各版本注意与发布插件时提供的的保持一致，如有更改，请及时告知君正技术人员以同步更新插件
其余依赖库可参考requirements.txt或实际运行提示按需安装

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
 (1)sh multi_gpu_train.sh/sh singlel_gpu_train.sh
 (2)可通过kmeans_anchor.py得到对应自己数据的anchor并修改cfg/persondet.cfg内的anchor参数，也可用原始anchor参数;
 (3)训练配置参数可参考models/commom.py中is_quantize,bitw,bita等参数:
    32bit, 设置:is_quantize = 0, bita = 32, 其余见具体代码
    8bit, 设置:is_quantize = 1, bita = 8, 其余见具体代码
    4bit, 设置:is_quantize = 1, bita = 4, 其余见具体代码
    另，浮点训练时没有预训练模型，--weight为‘’, 8bit训练时加载已得到的精度足够的浮点模型, 同时适当降低学习率；以此类推, 4bit加载8bit,如果效果达到还可以基于此尝试w2a4, 这样一步步推进, 效果更加.
 (4)train.py里--project可设置保存模型路径, 在runs/train/project下, 保存的有两个模型，best.pt本次训练到目前最好的,last.pt本次训练到目前最新的, 测试的结果保存在result.txt中, 可随时查看.


4.测试模型与检测图片
(测试时务必注意common中的bit设置与测试模型的对应,否则结果会有误差)
通过python3 detect.py -h查看选取所需参数,通过设置source可检测图片和视频,检测结果可设置显示(--view-img)或保存(--save-img)
- Image:  `--source file.jpg`
- Video:  `--source file.mp4`
- Directory:  `--source dir/`
示例如下，若需要其他操作可选择相应的参数;
sh detect.sh
注：检测的模型配置一定要和训练时候的配置(bitwidth)一致
测试模型精度:
sh test.sh
待测试的模型通过---weights指定, 验证集通过data/coco-person.yaml中的val2017.txt确定,其余参数根据实际需要给定


5.模型转换.onnx
 生成.onnx文件：sh convert_onnx.sh
    input：指定待转模型的路径
    output：.onnx,位置和.pt在同一目录