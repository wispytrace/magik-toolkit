#########################################################################
#  File Name: convert_jzdl.sh
#  Author: Magik Development Team
#  Created Time: Wed 19 Feb 2020 10:22:05 AM CST
#  Description:
#########################################################################
../../../../../TransformKit/magik-transform-tools \
	--framework onnx \
	--target_device Txx \
	--outputpath jzdl_cpp_inference/yolov5.mk.h \
	--inputpath ./runs/train/yolov5-4bit/best.onnx \
	--mean 0 \
	--var 255 \
	--img_width 320 \
	--img_height 416 \
	--img_channel 3 \
	--input_prepad true
