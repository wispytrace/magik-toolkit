../../../TransformKit/magik-transform-tools \
--inputpath  yolov5s.onnx \
--outputpath ./venus_sample_yolov5s/yolov5s_a1_magik.mk.h \
--config cfg/magik_a1.cfg \
--save_quantize_model true
cp venus_sample_yolov5s/makefile_files/Makefile_a1 venus_sample_yolov5s/Makefile
