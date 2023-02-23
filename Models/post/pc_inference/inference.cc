/*
 *        (C) COPYRIGHT Ingenic Limited.
 *             ALL RIGHTS RESERVED
 *
 * File       : magikExecutor.cc
 * Authors    : lqwang
 * Create Time: 2021-04-08:09:16:52
 * Description:
 *
 */

#include <fstream>
#include <iostream>
#include "args_parser.h"
#include "graph_executor.h"
#include "common.h"
#include "tensor.h"
#include <ctime>
#include <post_process.hpp>
#define STB_IMAGE_IMPLEMENTATION
#include "./stb/stb_image.h"
#include "./stb/drawing.hpp"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "./stb/stb_image_write.h"
#define STB_IMAGE_RESIZE_IMPLEMENTATION
#include "./stb/stb_image_resize.h"
static const uint8_t color[3] = {0xff, 0, 0};

struct object_rect {
    int x;
    int y;
    int width;
    int height;
};

void write_output_bin(const uint8_t* in_ptr, int h, int w, int c, std::string save_path)
{
	int size = h*w*c;
    std::ofstream owput;
    owput.open(save_path, std::ios::binary);
    if (!owput || !owput.is_open() || !owput.good()) {
        owput.close();
        return ;
    }
    owput.write((char *)in_ptr, size * sizeof(uint8_t));
    owput.close();
    return ;
}

int resize_uniform(uint8_t* src_data, int ori_w, int ori_h, uint8_t* dst_data, int dst_w, int dst_h, int c, object_rect &effect_area)
{
    int w = ori_w;
    int h = ori_h;
    std::cout << "src: (" << h << ", " << w << ")" << std::endl;

    float ratio_src = w*1.0 / h;
    float ratio_dst = dst_w*1.0 / dst_h;

    int tmp_w=0;
    int tmp_h=0;
    if (ratio_src > ratio_dst) {
        tmp_w = dst_w;
        tmp_h = floor((dst_w*1.0 / w) * h);
    } else if (ratio_src < ratio_dst){
        tmp_h = dst_h;
        tmp_w = floor((dst_h*1.0 / h) * w);
    } else {
        stbir_resize_uint8(src_data, ori_w, ori_h, 0, dst_data, dst_w, dst_h, 0, c);
        effect_area.x = 0;
        effect_area.y = 0;
        effect_area.width = dst_w;
        effect_area.height = dst_h;
        return 0;
    }

    std::cout << "tmp: (" << tmp_h << ", " << tmp_w << ")" << std::endl;

    uint8_t* tmp_data = (uint8_t*)malloc(tmp_w*tmp_h*c);
    stbir_resize_uint8(src_data, ori_w, ori_h, 0, tmp_data, tmp_w, tmp_h, 0, c);

    if (tmp_w != dst_w) { //高对齐，宽没对齐
        int index_w = floor((dst_w - tmp_w) / 2.0);
        std::cout << "index_w: " << index_w << std::endl;
        for (int i=0; i<dst_h; i++) {
            memcpy(dst_data+i*dst_w*3 + index_w*3, tmp_data+i*tmp_w*3, tmp_w*3);
        }
        effect_area.x = index_w;
        effect_area.y = 0;
        effect_area.width = tmp_w;
        effect_area.height = tmp_h;
    } else if (tmp_h != dst_h) { //宽对齐， 高没有对齐
        int index_h = floor((dst_h - tmp_h) / 2.0);
        std::cout << "index_h: " << index_h << std::endl;
        memcpy(dst_data+index_h*dst_w*3, tmp_data, tmp_w*tmp_h*3);
        effect_area.x = 0;
        effect_area.y = index_h;
        effect_area.width = tmp_w;
        effect_area.height = tmp_h;
    } else {
        printf("error\n");
    }
    free(tmp_data);
    return 0;
}

using namespace magik::transformkit::magikexecutor;
int main(int argc, char** argv) {

    if (argc != 7){
        printf("%s model_path img_path save_input_path in_w in_h in_c\n", argv[0]);
        exit(0);
    }
    std::string model_path = argv[1];
    std::string img_path = argv[2];
    std::string save_input_path = argv[3];
    int in_w = atoi(argv[4]);
    int in_h = atoi(argv[5]);
    int c = atoi(argv[6]);

    int comp = 0;
    int ori_img_w = 0;
    int ori_img_h = 0;
    unsigned char *imagedata = stbi_load(img_path.c_str(), &ori_img_w, &ori_img_h, &comp, c); // image format is bgr

    object_rect res_area;
    uint8_t* dst_data = (uint8_t*)malloc(in_w*in_h*c);
    resize_uniform(imagedata, ori_img_w, ori_img_h, dst_data, in_w, in_h, c, res_area);
//    stbi_write_bmp("result.bmp", in_w, in_h, c, dst_data);// w h
    std::string save_path = save_input_path + "/magik_input_nhwc_1_" + std::to_string(in_h) + "_" + std::to_string(in_w) + "_" + std::to_string(c) + ".bin";
    printf("%s\n", save_path.c_str());
    write_output_bin(dst_data, in_w, in_h, c, save_path);
    printf("resize_w:%d resize_h:%d\n", in_w, in_h);

    magik::transformkit::magikexecutor::GraphExecutor graphExecutor(model_path);
    graphExecutor.set_inplace(false);
    
    std::vector<std::string> input_names = graphExecutor.get_input_names();
    std::vector<std::string> output_names = graphExecutor.get_output_names();
    Tensor* tensor = new Tensor({1, in_h, in_w, c}, Tensor::DataType::DT_FLOAT);
    tensor->set_name(input_names[0]);

    for (int i = 0; i < tensor->total(); ++i) {
        tensor->mutable_data<float>(i) = (float)dst_data[i];
    }

    graphExecutor.set_input(tensor);

    std::cout<<"start inference ....................................................."<<std::endl;
    graphExecutor.work();

    const Tensor* tensor_res2 = graphExecutor.get_node_tensor(output_names[2]);
    const Tensor* tensor_res0 = graphExecutor.get_node_tensor(output_names[0]);
    const Tensor* tensor_res1 = graphExecutor.get_node_tensor(output_names[1]);

    printf("%d %d %d %d\n", tensor_res2->shape()[0], tensor_res2->shape()[1], tensor_res2->shape()[2], tensor_res2->shape()[3]);
    printf("%d %d %d %d\n", tensor_res0->shape()[0], tensor_res0->shape()[1], tensor_res0->shape()[2], tensor_res0->shape()[3]);
    printf("%d %d %d %d\n", tensor_res1->shape()[0], tensor_res1->shape()[1], tensor_res1->shape()[2], tensor_res1->shape()[3]);

	std::vector<float> out;
    for(int i = 0; i < tensor_res2->total(); i ++)
		out.push_back(tensor_res2->data<float>(i));
    for(int i = 0; i < tensor_res0->total(); i ++)
		out.push_back(tensor_res0->data<float>(i));
    for(int i = 0; i < tensor_res1->total(); i ++)
		out.push_back(tensor_res1->data<float>(i));

    std::vector<ObjectBox> face_list;
    printf("##########Output Done###########\n");
    /*post process*/
    float *p;
    p=out.data();

    Img img = {
        .w = in_w,
        .h = in_h,
        .c = 3,
        .w_stride = in_w*3,
        .data = dst_data
    };

    postprocess(p, face_list, in_w, in_h);// w h
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
    stbi_write_bmp("result.bmp", in_w, in_h, 3, img.data);// w h
    free(dst_data);

    delete tensor;
    return 0;
}
