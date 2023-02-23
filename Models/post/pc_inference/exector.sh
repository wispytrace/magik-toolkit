cd ../yolov5s/
sh run_t41.sh
cd -
make clean
make -j12
MAGIK_CPP_DUMPDATA=true  ./pc_inference_bin  ../yolov5s/venus_sample_yolov5s/save-magik/model_quant.mgk fall_1054_sys.jpg ../yolov5s/venus_sample_yolov5s/ 640 384 3
