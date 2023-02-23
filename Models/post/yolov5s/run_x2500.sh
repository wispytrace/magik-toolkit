../../../TransformKit/magik-transform-tools \
--inputpath  yolov5s.onnx \
--outputpath ./venus_sample_yolov5s/yolov5s_x2500_magik.mk.h \
--config cfg/magik_x2500.cfg \
--save_quantize_model true
cp venus_sample_yolov5s/makefile_files/Makefile_x2500 venus_sample_yolov5s/Makefile
