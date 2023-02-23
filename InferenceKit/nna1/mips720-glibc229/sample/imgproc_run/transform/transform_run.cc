/*
 *         (C) COPYRIGHT Ingenic Limited
 *              ALL RIGHT RESERVED
 *
 * File        : transform_run.cc
 * Authors     : zfni
 * Create Time : 2022-02-15 09:24:21 (CST)
 * Description :
 *
 */

#include "venus.h"
#include <errno.h>
#include <fcntl.h>
#include <fstream>
#include <iostream>
#include <memory.h>
#include <memory>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <string>
#include <unistd.h>
#include <vector>
int write_bin_to_file(const char *file_path, char *buf, int size_buf) {
    FILE *fid = fopen(file_path, "wb");
    for (int i = 0; i < size_buf; i++) {
        fwrite(&buf[i], sizeof(char), 1, fid);
    }
    fclose(fid);
    return 0;
}

int similar_transform_run(int argc, char **argv) {
    /*set*/
    cpu_set_t mask;
    CPU_ZERO(&mask);
    CPU_SET(0, &mask);
    if (sched_setaffinity(0, sizeof(mask), &mask) == -1) {
        printf("warning: could not set CPU affinity, continuing...\n");
    }

    int ret = 0;

    ret = venus::venus_init();
    if (0 != ret) {
        fprintf(stderr, "venus init failed.\n");
        exit(0);
    }

    magik::venus::Tensor input_src_tensor({1, 3, 2, 1}, magik::venus::TensorFormat::NHWC);
    magik::venus::Tensor input_dst_tensor({1, 3, 2, 1}, magik::venus::TensorFormat::NHWC);

    float *in_src_data = input_src_tensor.mudata<float>();
    float *in_dst_data = input_dst_tensor.mudata<float>();

    int size = 2 * 3;

    float src_addr[size] = {174.403, 180.562, 333.289, 180.54, 258.049, 359.579};
    float dst_addr[size] = {76.5892, 103.3926, 147.0636, 103.0028, 112.2792, 184.5696};
    for (int i = 0; i < size; i++) {
        in_src_data[i] = src_addr[i];
        in_dst_data[i] = dst_addr[i];
    }

    magik::venus::Tensor output_tensor({1, 3, 3, 1}, magik::venus::TensorFormat::NHWC);
    float *p = output_tensor.mudata<float>();
    magik::venus::similar_transform(input_dst_tensor, input_src_tensor, output_tensor);

    auto shape = output_tensor.shape();
    int out_size = 1;
    printf("shape\n");
    for (int i = 0; i < 4; i++) {
        out_size *= shape[i];
        printf("%d ", shape[i]);
    }
    printf("\n");

    float *out_data = output_tensor.mudata<float>();
    write_bin_to_file("similar_transform.bin", (char *)out_data, out_size * sizeof(float));

    ret = venus::venus_deinit();
    if (0 != ret) {
        fprintf(stderr, "venus deinit failed.\n");
        return ret;
    }
    return 0;
}
int get_affine_transform_run(int argc, char **argv) {
    /*set*/
    cpu_set_t mask;
    CPU_ZERO(&mask);
    CPU_SET(0, &mask);
    if (sched_setaffinity(0, sizeof(mask), &mask) == -1) {
        printf("warning: could not set CPU affinity, continuing...\n");
    }

    int ret = 0;
    if (argc < 3) {
        printf("%s iw ih\n", argv[0]);
        exit(0);
    }

    ret = venus::venus_init();
    if (0 != ret) {
        fprintf(stderr, "venus init failed.\n");
        exit(0);
    }

    magik::venus::Tensor input_src_tensor({1, 3, 2, 1}, magik::venus::TensorFormat::NHWC);
    magik::venus::Tensor input_dst_tensor({1, 3, 2, 1}, magik::venus::TensorFormat::NHWC);

    float *in_src_data = input_src_tensor.mudata<float>();
    float *in_dst_data = input_dst_tensor.mudata<float>();

    int size = 2 * 3;
    int iw = atoi(argv[1]);
    int ih = atoi(argv[2]);
    float src_addr[size] = {iw - 1, 0, iw - 1, ih - 1, 0, ih - 1};
    float dst_addr[size] = {ih, iw - 1, 0, iw - 1, 0, 0};
    for (int i = 0; i < size; i++) {
        in_src_data[i] = src_addr[i];
        in_dst_data[i] = dst_addr[i];
    }

    magik::venus::Tensor output_tensor({1, 3, 2, 1}, magik::venus::TensorFormat::NHWC);
    float *p = output_tensor.mudata<float>();
    magik::venus::similar_transform(input_dst_tensor, input_src_tensor, output_tensor);

    auto shape = output_tensor.shape();
    int out_size = 1;
    printf("shape\n");
    for (int i = 0; i < 4; i++) {
        out_size *= shape[i];
        printf("%d ", shape[i]);
    }
    printf("\n");
    for (int i = 0; i < 6; i++) {
        printf("%f\n", p[i]);
    }

    float *out_data = output_tensor.mudata<float>();
    write_bin_to_file("get_affine_transform.bin", (char *)out_data, out_size * sizeof(float));

    ret = venus::venus_deinit();
    if (0 != ret) {
        fprintf(stderr, "venus deinit failed.\n");
        return ret;
    }
    return 0;
}

int main(int argc, char **argv) {
    printf("input number 0(similar_transform_run)  1(get_affine_transform_run):\n");
    int i = 0;
    scanf("%d", &i);
    if (i == 0)
        similar_transform_run(argc, argv);
    else if (i == 1)
        get_affine_transform_run(argc, argv);
    return 0;
}
