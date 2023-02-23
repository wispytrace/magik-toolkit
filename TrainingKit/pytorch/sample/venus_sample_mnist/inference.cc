/*
 *         (C) COPYRIGHT Ingenic Limited
 *              ALL RIGHT RESERVED
 *
 * File        : model_run.cc
 * Authors     : klyu
 * Create Time : 2020-10-28 12:22:44 (CST)
 * Description :
 *
 */
#include "venus.h"
#include <fstream>
#include <iostream>
#include <memory.h>
#include <memory>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <string>
#include <vector>

#define IS_ALIGN_64(x) (((size_t)x) & 0x3F)

#ifdef VENUS_PROFILE
#define RUN_CNT 10
#else
#define RUN_CNT 1
#endif


#ifdef VENUS_DEBUG
#include "img_input.h"
#else
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
#endif


int main(int argc, char **argv) {
    /*set*/
    cpu_set_t mask;
    CPU_ZERO(&mask);
    CPU_SET(0, &mask);
    if (sched_setaffinity(0, sizeof(mask), &mask) == -1) {
            printf("warning: could not set CPU affinity, continuing...\n");
    }

#ifdef VENUS_DEBUG
    int ret = 0;
    if (argc != 2)
    {
        printf("%s model_path\n", argv[0]);
        exit(0);
    }

    int in_w = 32, in_h = 32;
    std::unique_ptr<venus::Tensor> input;
    ret = venus::venus_init();
    if (0 != ret) {
        fprintf(stderr, "venus init failed.\n");
        exit(0);
    }

    std::unique_ptr<venus::BaseNet> test_net;
    test_net = venus::net_create(venus::TensorFormat::NHWC);
    std::string model_path = argv[1];
    ret = test_net->load_model(model_path.c_str());

    input = test_net->get_input(0);
    magik::venus::shape_t rgba_input_shape = input->shape();
    printf("model-->%d ,%d %d \n",rgba_input_shape[1], rgba_input_shape[2], rgba_input_shape[3]);
    input->reshape({1, in_h, in_w , 4});
    uint8_t *indata = input->mudata<uint8_t>();
    std::cout << "input shape:" << std::endl;
    printf("-->%d %d \n",in_h, in_w);
    int data_cnt = 1;
    for (auto i : input->shape()) 
    {
        std::cout << i << ",";
        data_cnt *= i;
    }
    std::cout << std::endl;

    for (int j = 0; j < data_cnt; j++) 
    {
        indata[j] = image[j];
    }
    test_net->run();
#else
    int ret = 0;
    if (argc != 3)
    {
        printf("%s model_path img_path\n", argv[0]);
        exit(0);
    }

    std::string model_path = argv[1];
	std::string img_path = argv[2];
    cv::Mat image;
    image = cv::imread(argv[2]);
    int ori_img_w = image.cols;
    int ori_img_h = image.rows;
    printf("w:%d h:%d\n", ori_img_w, ori_img_h);

    int in_w = 32, in_h = 32;
    ret = venus::venus_init();
    if (0 != ret) 
	{
        fprintf(stderr, "venus init failed.\n");
        return ret;
    }

    std::unique_ptr<venus::BaseNet> test_net = venus::net_create();
    if (!test_net) 
	{
        fprintf(stderr, "create network handle falied.\n");
        return -1;
    }

    ret = test_net->load_model(model_path.c_str());

    if (0 != ret) {
        fprintf(stderr, "Load model failed.\n");
        return ret;
    }
    printf("Load model over.\n");
    size_t mem_size;
    ret = test_net->get_forward_memory_size(mem_size);
    std::cout << "Forward memory size: " << mem_size << std::endl;
    std::unique_ptr<venus::Tensor> input = test_net->get_input(0);
    magik::venus::shape_t rgba_input_shape = input->shape();
    printf("model-->%d ,%d %d \n",rgba_input_shape[1], rgba_input_shape[2], rgba_input_shape[3]);
    input->reshape({1, in_h, in_w , 4});
    uint8_t *indata = input->mudata<uint8_t>();
    std::cout << "input shape:" << std::endl;
    printf("-->%d %d \n",in_h, in_w);

    if (IS_ALIGN_64(indata) != 0) {
        fprintf(stderr, "input addr not align to 64 bytes.\n");
        return -1;
    }
    int data_cnt = 1;
    for (auto i : input->shape()) {
        std::cout << i << ",";
        data_cnt *= i;
    }
    std::cout << std::endl;

    for(int i = 0; i < ori_img_w * ori_img_h; i++)
    {
        for (int j = 0 ; j < 4; j ++)
        {
            indata[i*4 + 0] = image.data[i*3 + 0];
            indata[i*4 + 1] = image.data[i*3 + 1];
            indata[i*4 + 2] = image.data[i*3 + 2];
            indata[i*4 + 3] = 0;
        }
    }

    for (int i = 0; i < RUN_CNT; i++) 
	{
        test_net->run();
    }

    auto out0 = test_net->get_output(0);
    const float *out_ptr = out0->data<float>();

    std::cout << "output0 Shape: " << std::endl;
    int out_size = 1;
    for (auto i : out0->shape()) {
        std::cout << i << ",";
        out_size *= i;
    }
	for(int i = 0 ; i < out_size; i ++)
		printf("%f ", out_ptr[i]);
    printf("\n");

    ret = venus::venus_deinit();
    if (0 != ret) {
        fprintf(stderr, "venus deinit failed.\n");
        return ret;
    }
#endif
    return 0;
}
