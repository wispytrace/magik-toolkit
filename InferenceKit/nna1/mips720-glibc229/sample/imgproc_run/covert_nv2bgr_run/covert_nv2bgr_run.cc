/*
 *         (C) COPYRIGHT Ingenic Limited
 *              ALL RIGHT RESERVED
 *
 * File        : covert_nv2bgr_run.cc
 * Authors     : zfni
 * Create Time : 2022-02-15 09:24:21 (CST)
 * Description :
 *
 */

#include "venus.h"
#include <errno.h>
#include <fstream>
#include <iostream>
#include <memory.h>
#include <memory>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <string>
#include <vector>
int write_bin_to_file(const char *file_path, char *buf, int size_buf) {
    FILE *fid = fopen(file_path, "wb");
    for (int i = 0; i < size_buf; i++) {
        fwrite(&buf[i], sizeof(char), 1, fid);
    }
    fclose(fid);
    return 0;
}

int main(int argc, char **argv) {
    /*set*/
    cpu_set_t mask;
    CPU_ZERO(&mask);
    CPU_SET(0, &mask);
    if (sched_setaffinity(0, sizeof(mask), &mask) == -1) {
        printf("warning: could not set CPU affinity, continuing...\n");
    }

    int ret = 0;
    if (argc != 4) {
        printf("%s nv12_path in_w in_h\n", argv[0]);
        exit(0);
    }
    ret = venus::venus_init();
    if (0 != ret) {
        fprintf(stderr, "venus init failed.\n");
        exit(0);
    }

    FILE *fp = NULL;
    int input_w = atoi(argv[2]);
    int input_h = atoi(argv[3]);
    int nv12_size = input_w * input_h * 1.5;
    unsigned char *nv12_data = (unsigned char *)malloc(nv12_size);
    fp = fopen(argv[1], "rb");
    if (fp) {
        int ret = fread(nv12_data, 1, nv12_size, fp);
        printf("\033[32mret %d\033[0m\n", ret);
        fclose(fp);
    } else {
        printf("open nv12 file failed\n");
        return -1;
    }

    magik::venus::Tensor output_tensor({1, input_h, input_w, 4});

    magik::venus::Tensor temp_ori_input({1, input_h, input_w, 1}, magik::venus::TensorFormat::NV12);
    uint8_t *tensor_data = temp_ori_input.mudata<uint8_t>();
    int src_size = int(input_h * input_w * 1.5);
    magik::venus::memcopy((void *)tensor_data, (void *)nv12_data, src_size * sizeof(uint8_t));

    magik::venus::warp_covert_nv2bgr(temp_ori_input, output_tensor);

    int size = input_w * input_h * 4;
    uint8_t *outdata = output_tensor.mudata<uint8_t>();
    write_bin_to_file("warp_covert_nv2bgr.bin", (char *)outdata, size * sizeof(uint8_t));

    ret = venus::venus_deinit();
    if (0 != ret) {
        fprintf(stderr, "venus deinit failed.\n");
        return ret;
    }

    return 0;
}
