# magik-toolkit

magik 开发平台.

1、Inferencekit 文件夹是板端推理库，文档参考路径Docs/ch/Ingenic Neural Network Inference Framework Venus Programming Manual.pdf.

2、Modlels文件夹下包含训练及转换代码，训练量化使用文档Docs/ch/Magik Training Quantization Used Guide.pdf；后量化使用文档Docs/ch/Magik Post-Training-Quantization User Guide.pdf；

3、TrainingKit 包括pytorch的训练whl包及对应sample，文档参考路径Docs/ch/Detailed Explanation of PyTorch Training Quantization Operators.pdf.

4、TransformKit 转换工具.

5、ThirdParty 使用的第三方推理库(opencv lib).



Note:
T40切换到T41流程：
训练量化：
    1、基于ops构建的训练量化模型，从git库TrainingKit下获取最新的安装whl包并安装（安装之前需要卸载以前的whl包pip uninstall magik-trainingkit*，卸载之前最好备份当前训练环境）；
    2、修改网络代码中的target_device为T41重新生成onnx模型（生成onnx时需要以网络结构定义的形式进行加载pt参数，这样T41参数才会生效，生成onnx之后可通过netron进行可视化，进一步确认节点属性target_device是否更改为T41）；
    3、从TransformKit下面获取最新的转换工具，可直接使用以前的转换脚本，只需要把shell脚本中对应的target_device修改为T41；
    4、T41对应的上板推理库需要替换成InferenceKit/nna2/T41相关的库及头文件，以前写的推理代码及接口可直接使用，不需更改；
后量化：
    1、从TransformKit下面获取最新的转换工具，可直接使用以前的转换脚本及代码，只需要把shell脚本中的target_device修改为T41；
    2、T41对应的上板推理库需要替换成InferenceKit/nna2/T41相关的库及头文件，以前写的推理代码及接口可直接使用，不需更改；
注：需要根据上述git库拉取最新的训练量化whl包、转换工具及对应的venus库；转换及上板过程中有任何问题，请及时反馈；
