
版本要求:
 python : 3.7 
 torch: >= 1.6  建议 1.8

NEW！！2020.12.03 相关修改内容
	1.修改数据训练数据增强部分。文件transforms.py
	  * 添加Padding（） 等比缩放策略 多余部分填充灰色边框。（主要修改）
	  * RandomSampleCrop 修改了对 随机w,h判断 
	    原始     if h / w != 1:  
            修改为   h / w < 0.75 or h / w > 1.3: 
	  * 对应修改预测部分 缩放与填充的转换处理。

 
1. 数据的准备 
提供训练数据集的地址：
The clean widerface data pack after filtering out the 10px*10px small face:
https://pan.baidu.com/share/init?surl=MR0ZOKHUP_ArILjbAn03sw   提取代码： cbiu
验证数据集：点击：http://mmlab.ie.cuhk.edu.hk/projects/WIDERFace/，进入网页，
下载WIDER Face Validation Images 
下载完成后。直接解压到/data/目录下即可
data/
    wider_face_add_lm_10_10/
      	Annotations/
      	ImageSets/
	    Main/
		  test.txt  
		  trainval.txt			
      	JPEGImages/
    WIDER_val/
       images/


2.训练
训练配置参数主要在train-version-slim.sh 脚本中。
run sh train-version-slim.sh



3.网络量化与参数配置
（2个文件与参数需要手动配置）
文件1： vision_quantize/nn/mb_tiny.py
IS_QUANTIZE = True     #如果非量化。请改为False。量化为True
BITA = 4               #如果8bit量化时  BITA 改为8即可  
BITW = 4	      	    #与BITA值一致即可
SPECIAL_BIT = 4        #4bit 量化为4  8bit 为8
CLIP_MAX_VALUE = 3.0   #CLIP值8bit量化时建议为3.0  4bit 3.0与2.0都可以。可能依据训练数据而变。

文件2： vision_quantize/ssd/mb_tiny_fd.py
参数同上，俩个文件量化参数保持一致。


4.测试模型与检测图片
训练好模型后。如果对模型测试。
./widerface_evaluate 文件下按里面README文件步骤走就可以。测试数据原本wider数据val即可


图片检测detect_imgs.py 里边将模型路径替换成训练好的模型即可。

5.模型转换onnx。
python convert_to_onnx.py 
转换好的模型默认会存放在 ./model/onnx/ 目录下

6.WIDER_val 测试集量化流程测试结果
	#result
	float
	Easy:   0.7799
	Medium: 0.6878
	Hard:   0.3955

	#8bit clip3.0 batch 24 lr 1e-2
	Easy:   0.7784
	Medium: 0.6839
	Hard:   0.3799

	#4bit  5 4        
	    clip3.0						clip2.0  
	Easy:   0.7741               Easy:   0.7738 
	Medium: 0.6773	          	 Medium: 0.6801
	Hard:   0.3808           	 Hard:   0.3882
