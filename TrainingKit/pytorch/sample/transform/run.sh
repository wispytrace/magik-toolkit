path=../../../../TransformKit

$path/magik-transform-tools \
--framework onnx \
--target_device T40 \
--outputpath t40_graph_mnist.mk.h \
--inputpath ../ckpt/net-w8a8/checkpoint.onnx \
--mean 0,0,0 \
--var 255,255,255 \
--img_width 32 \
--img_height 32 \
--img_channel 3 \

cp t40_graph_mnist.bin ../venus_sample_mnist//
