#include <stdio.h>
#include <algorithm>
#include <vector>
#include <iostream>
#include <string>
#include <stdint.h>
#include "net.h"
#include "img_input.h"
#include "fstream"
#include <iomanip>
#include "utils.h"

//#define MEM_MODEL
using namespace std;
#ifdef MEM_MODEL
#include "magik_model_mnist.h"
#endif

int main(int argc, char* argv[]) {

    printf("imagedata size:%d\n", sizeof(image));

    jzdl::BaseNet *mnist = jzdl::net_create();
#ifdef MEM_MODEL
    mnist->load_model((const char*)magik_model_mnist, true);
#else
    std::string model_file_path = "./magik_model_mnist.bin";
    mnist->load_model(model_file_path.c_str());
#endif

    std::vector<uint32_t> input_shape = mnist->get_input_shape();
    int input_index = mnist->get_model_input_index();
    int output_index = mnist->get_model_output_index();
    printf("input_shape.h=%d, input_shape.w=%d, input_shape.c=%d\n", input_shape[0], input_shape[1], input_shape[2]);
    printf("input_index=%d, output_index=%d\n", input_index, output_index);
    jzdl::Mat<uint8_t> src(28, 28, 1, (uint8_t*)image); // w,h,c
    jzdl::Mat<uint8_t> dst(input_shape[0], input_shape[1], input_shape[2]);
    jzdl::resize(src, dst);
    jzdl::image_sub(dst, 128);


    jzdl::Mat<int8_t> img(input_shape[0], input_shape[1], input_shape[2], (int8_t*)dst.data);
    jzdl::Mat<float> out;

    printf("########Input Done#######\n");

    mnist->input(img);
    mnist->run(out);

    printf("%d,%d,%d\n",out.h,out.w,out.c);
    ofstream InputFile("img_feature_jzdl.h");
    for (int i = 0; i < out.h * out.w * out.c; i++)
    {
        float temp = static_cast<float>(out.data[i]);

        InputFile  << temp;
        if ((i+1) % 1 == 0)
        {
            InputFile << endl;
        }
    }
    jzdl::net_destory(mnist);

    return 0;
}
