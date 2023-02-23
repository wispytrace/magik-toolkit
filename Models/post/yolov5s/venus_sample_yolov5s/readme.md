venus lib sample used ways:
1、release(发布库)
   编译: make build_type=release
   在当前文件夹下生成venus_yolov5s_bin_uclibc_release可执行文件，拷贝venus库(libvenus.so)、可执行文件(venus_yolov5s_bin_uclibc_release)、模型文件(yolov5s_*_magik.bin)、测试图片(bus.jpg)至开发板运行即可(./venus_yolov5s_bin_uclibc_release yolov5s_*_magik.bin fall_1054_sys.jpg)
   (注：运行前添加库路径至LD_LIBRARY_PATH: export LD_LIBRARY_PATH=$lib_path:$LD_LIBRARY_PATH).
   (清除make build_type=release clean)

2、debug(用于核对数据的库，调用该库会dump网络中间层数据,方便进行数据核对)
   编译 make build_type=debug 
   在当前文件夹下生成venus_yolov5s_bin_uclibc_debug可执行文件，拷贝venus库(libvenus.d.so)、可执行文件(venus_yolov5s_bin_uclibc_debug)、模型文件(yolov5s_*_magik.bin)及输入文件（magik_input_nhwc_1_640_480_3.bin）至开发板运行即可(./venus_yolov5s_bin_uclibc_debug yolov5s_*_magik.bin magik_input_nhwc_1_640_480_3.bin)
   (注：运行前添加库路径至LD_LIBRARY_PATH: export LD_LIBRARY_PATH=$lib_path:$LD_LIBRARY_PATH).
   (清除 make build_type=debug clean)

3、profile(网络结构可视化及网络每层运行时间统计)
   编译: make build_type=profile
   在当前文件夹下生成venus_yolov5s_bin_uclibc_profile可执行文件，拷贝venus库(libvenus.p.so)、可执行文件(venus_yolov5s_bin_uclibc_profile)、模型文件(yolov5s_*_magik.bin)、测试图片(bus.jpg)至开发板运行即可(./venus_yolov5s_bin_uclibc_profile yolov5s_*_magik.bin fall_1054_sys.jpg)
   (注：运行前添加库路径至LD_LIBRARY_PATH: export LD_LIBRARY_PATH=$lib_path:$LD_LIBRARY_PATH).
   (清除make build_type=profile clean)

4、nmem(用于统计模型运行时nmem内存占用情况，运行程序，内存使用情况保存在/tmp/nmem_memory.txt)
   编译: make build_type=nmem
   在当前文件夹下生成venus_yolov5s_bin_uclibc_nmem可执行文件，拷贝venus库(libvenus.m.so)、可执行文件(venus_yolov5s_bin_uclibc_nmem)、模型文件(yolov5s_*_magik.bin)、测试图片(bus.jpg)至开发板运行即可(./venus_yolov5s_bin_uclibc_nmem yolov5s_*_magik.bin fall_1054_sys.jpg)
   (注：运行前添加库路径至LD_LIBRARY_PATH: export LD_LIBRARY_PATH=$lib_path:$LD_LIBRARY_PATH).
   (清除make build_type=nmem clean)
