python train.py
python convert_onnx.py
cd transform
sh run.sh 
cd ../venus_sample_mnist 
make clean
make
