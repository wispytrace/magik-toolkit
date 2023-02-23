1. 模型的训练
   示例代码默认训练cifar100的数据集, 运行代码时会自动下载到data下,运行命令参见train.sh:
   python train.py -net mobilenetv2 -prefix 32bit -b 128 -lr 0.1 -num_classes 100
   -net --- 要训练的网络
   -prefix --- 模型存储的目录,(这里是在./checkpoint/mobilenetv2/32bit/, 对应的log在runs/mobilenetv2/32bit下)
   -numpy_classes --- 训练类别数,不用cifar100时注意对应修改
   其余参数的设置见conf/global_setting.py, 显卡的设置在train.py中的os.environ["CUDA_VISIBLE_DEVICES"] = '0,1', 各项按需修改

2. 模型的修改
   cifar100的输入尺寸为32x32,偏小,实际用mobilenet训练时对第一层做了修改,参见models/mobilenet.py, 具体如下:
   ##for inmagenet(224x224)
   # self.conv1 = qConv(3, self.inplanes, kernel_size=7, stride=2, pad=3, bias=False, bn=True, act=True, first=True)
   ## for cifar100(32x32)
   self.conv1 = qConv(3, self.inplanes, kernel_size=3, stride=1, pad=1, bias=False, bn=True, act=True, first=True)
   如果使用imagenet等224x224的输入,或想要保持原始mobilenetv2的结构,改为上面一种定义方式即可.

3. bit位的设置
   训练配置参数在model/basic.py中, 其中,
   32bit: IS_QUANTIZE = 0 BITW = 32, 其余见具体代码
   8bit: IS_QUANTIZE = 1 BITW = 8, 其余见具体代码
   4bit: IS_QUANTIZE = 1 BITW = 4, 其余见具体代码
   另，浮点训练时没有预训练模型，--weight为‘’，8bit训练时加载已得到的精度足够的32bit模型，同时适当降低学习率；以此类推，4bit加载8bit，这样一步步推进，效果更加.

4. 模型的测试
   测试模型精度:
   python test.py -net mobilenetv2 -weights ./checkpoint/mobilenetv2/4bit/*.pth -b 128
   测试单张图片:
   python test_image.py -net mobilenetv2 -weights checkpoint/mobilenetv2/4bit/*.pth 
   注：检测的模型配置一定要和训练时候的配置(BITW等)一致

5. 模型的onnx转换
   python convert_onnx.py -net mobilenetv2 -weights ./checkpoint/mobilenetv2/4bit/*.pth
