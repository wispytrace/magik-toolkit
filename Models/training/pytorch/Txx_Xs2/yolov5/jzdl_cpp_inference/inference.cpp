#include <stdio.h>
#include <algorithm>
#include <vector>
#include <iostream>
#include <string>
#include <stdint.h>
#include "net.h"
#include "fstream"
#include <iomanip>
#include "utils.h"
#include "img_input.h"
#include "post_process.hpp"
#include "drawing.hpp"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

//#define MEM_MODEL
using namespace std;
using namespace jzdl;
#ifdef MEM_MODEL
#include "magik_model_t31.mk.h"
#endif

static const uint8_t color[3] = {0xff, 0, 0};

int main(int argc, char* argv[]) {
    jzdl::BaseNet *mnist = jzdl::net_create();
#ifdef MEM_MODEL
    mnist->load_model((const char*)magik_model_t31_mk, true);
#else
    std::string model_file_path = "./magik_model_yolov5.bin";
    mnist->load_model(model_file_path.c_str());
#endif

    std::vector<uint32_t> input_shape = mnist->get_input_shape();
    int input_index = mnist->get_model_input_index();
    int output_index = mnist->get_model_output_index();
    printf("input_shape.w=%d, input_shape.h=%d, input_shape.c=%d\n", input_shape[0], input_shape[1], input_shape[2]);
    printf("input_index=%d, output_index=%d\n", input_index, output_index);
    //jzdl::Mat<uint8_t> src(320, 416, 3); // w,h,c
    jzdl::Mat<uint8_t> src(320, 416, 3, (uint8_t*)image);//img size whc
    jzdl::Mat<uint8_t> dst(input_shape[0], input_shape[1], input_shape[2]);
    jzdl::resize(src, dst);
    jzdl::image_sub(dst, 128);

    jzdl::Mat<int8_t> img_in(input_shape[0], input_shape[1], input_shape[2], (int8_t*)dst.data);
    jzdl::Mat<float> out;
    
    printf("########Input Done#######\n");

    mnist->input(img_in);
    mnist->run(out);

    printf("%d,%d,%d\n",out.h,out.w,out.c);
    ofstream InputFile("img_feature_jzdl.h");
    for (int i = 0; i < out.h * out.w * out.c; i++) {
        float temp = static_cast<float>(out.data[i]);

        InputFile  << temp;
        if ((i+1) % 1 == 0) {
            InputFile << endl;
        }
    }

    Img img = {
        .w = 320,
        .h = 416,
        .c = 3,
        .w_stride = 320*3,
        .data = image
    };

    std::vector<ObjectBox> face_list;
    printf("##########Output Done###########\n");
    /*post process*/
    float *p;
    p=out;
   
    postprocess(p, face_list, input_shape[0], input_shape[1]);// w h

    for (int i = 0; i < face_list.size(); i++) {
        Point pt1 = {
            .x = (int)face_list[i].x1,
            .y = (int)face_list[i].y1
        };
        Point pt2 = {
            .x = (int)face_list[i].x2,
            .y = (int)face_list[i].y2
        };
        sample_draw_box_for_image(&img, pt1, pt2, color, 2);
        printf("index:%d \n",i);
        printf("minx:%f \n", face_list[i].x1);
        printf("miny:%f \n", face_list[i].y1);
        printf("maxx:%f \n", face_list[i].x2);
        printf("maxy:%f \n", face_list[i].y2);
        printf("score:%f \n", face_list[i].score);
        printf("classID:%d \n", face_list[i].classid);
    }
    stbi_write_bmp("result.bmp", 320, 416, 3, img.data);// w h
    jzdl::net_destory(mnist);
    return 0;
}
