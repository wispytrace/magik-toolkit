../../../TransformKit/magik-transform-tools \
--inputpath  yolov5s.onnx \
--outputpath ./venus_sample_yolov5s/yolov5s_t40_magik.mk.h \
--config cfg/magik_t40.cfg \
--save_quantize_model true
cp venus_sample_yolov5s/makefile_files/Makefile_t40 venus_sample_yolov5s/Makefile
