venus lib sample used ways:
1、release(发布库)
   编译: make build_type=release
   在当前文件夹下生成venus_mnist_bin_uclibc_release可执行文件，拷贝venus库(libvenus.so)、opencv库(libopencv_core.so.3.3,libopencv_highgui.so.3.3,libopencv_imgcodecs.so.3.3,libopencv_imgproc.so.3.3,libopencv_videoio.so.3.3)、可执行文件(venus_mnist_bin_uclibc_release)、模型文件(t40_graph_mnist.bin)、测试图片(test1.bmp)至开发板运行即可(./venus_mnist_bin_uclibc_release t40_graph_mnist.bin test1.bmp)
   (注：运行前添加库路径至LD_LIBRARY_PATH: export LD_LIBRARY_PATH=$lib_path:$LD_LIBRARY_PATH).
   (清除make build_type=release clean)

2、debug(用于核对数据的库，调用该库会dump网络中间层数据,方便进行数据核对)
   首先使用generate_img_input.py生成对应的输入头文件(可以读取图片或者二进制文件,二进制文件默认是uint8数据格式).
   [python generate_img_input.py img_path w h c img]  or  [python generate_img_input.py bin_path w h c bin]
   编译 make build_type=debug 
   在当前文件夹下生成venus_mnist_bin_uclibc_debug可执行文件，拷贝venus库(libvenus.d.so)、可执行文件(venus_mnist_bin_uclibc_debug)、模型文件(t40_graph_mnist.bin)至开发板运行即可(./venus_mnist_bin_uclibc_debug t40_graph_mnist.bin)
   (注：运行前添加库路径至LD_LIBRARY_PATH: export LD_LIBRARY_PATH=$lib_path:$LD_LIBRARY_PATH).
   清除 make build_type=debug clean

3、profile(网络结构可视化及网络每层运行时间统计)
   编译: make build_type=profile
   在当前文件夹下生成venus_mnist_bin_uclibc_profile可执行文件，拷贝venus库(libvenus.p.so)、opencv库(libopencv_core.so.3.3,libopencv_highgui.so.3.3,libopencv_imgcodecs.so.3.3,libopencv_imgproc.so.3.3,libopencv_videoio.so.3.3)、可执行文件(venus_mnist_bin_uclibc_profile)、模型文件(t40_graph_mnist.bin)、测试图片(test1.bmp)至开发板运行即可(./venus_mnist_bin_uclibc_profile t40_graph_mnist.bin test1.bmp)
   (注：运行前添加库路径至LD_LIBRARY_PATH: export LD_LIBRARY_PATH=$lib_path:$LD_LIBRARY_PATH).
   (清除make build_type=profile clean)

4、nmem(用于统计模型运行时nmem内存占用情况，运行程序，内存使用情况保存在/tmp/nmem_memory.txt)
   编译: make build_type=nmem
   在当前文件夹下生成venus_mnist_bin_uclibc_nmem可执行文件，拷贝venus库(libvenus.m.so)、opencv库(libopencv_core.so.3.3,libopencv_highgui.so.3.3,libopencv_imgcodecs.so.3.3,libopencv_imgproc.so.3.3,libopencv_videoio.so.3.3)、可执行文件(venus_mnist_bin_uclibc_nmem)、模型文件(t40_graph_mnist.bin)、测试图片(test1.bmp)至开发板运行即可(./venus_mnist_bin_uclibc_nmem t40_graph_mnist.bin test1.bmp)
   (注：运行前添加库路径至LD_LIBRARY_PATH: export LD_LIBRARY_PATH=$lib_path:$LD_LIBRARY_PATH).
   (清除make build_type=nmem clean)
