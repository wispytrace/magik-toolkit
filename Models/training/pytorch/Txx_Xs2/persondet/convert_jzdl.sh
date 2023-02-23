#########################################################################
#  File Name: convert_jzdl.sh
#  Author: Magik Development Team
#  Created Time: Wed 19 Feb 2020 10:22:05 AM CST
#  Description:
#########################################################################

onnx_model=./weights/4-5bit/backup3.onnx
../../../../../TransformKit/magik-transform-tools \
--framework onnx \
--target_device Txx \
--outputpath ./jzdl/persondet.mk.h \
--inputpath $onnx_model \
--mean 0 \
--var 255 \
--img_width 416 \
--img_height 320 \
--img_channel 3 \
--input_prepad true 
