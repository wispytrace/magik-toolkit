***convert model from caffe to onnx***
===========================================================
```bash
conda create -n test python=3.6
conda activate test
pip install -r requirement.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
python caffe2onnx/caffe2onnx.py model/resnet18.prototxt model/resnet18.caffemodel -o resnet18.onnx

```
