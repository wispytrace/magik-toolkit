#########################################################################
#  File Name: convert_jzdl.sh
#  Author: Magik Development Team
#  Created Time: Wed 19 Feb 2020 10:22:05 AM CST
#  Description:
#########################################################################

onnx_model=./models/onnx/slim-Epoch-208-Loss-3.onnx
../../../../../TransformKit/magik-transform-tools \
	--framework onnx \
	--target_device Txx \
	--outputpath ./jzdl/facedet.mk.h \
	--inputpath $onnx_model \
	--mean 128 \
	--var 128 \
	--img_width 320 \
	--img_height 240 \
	--img_channel 3 \
	--input_prepad false
