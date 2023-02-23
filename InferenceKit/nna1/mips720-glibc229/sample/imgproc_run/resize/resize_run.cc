/*
 *         (C) COPYRIGHT Ingenic Limited
 *              ALL RIGHT RESERVED
 *
 * File        : resize_run.cc
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

int warp_resize_run(int argc, char **argv) {
    /*set*/
    cpu_set_t mask;
    CPU_ZERO(&mask);
    CPU_SET(0, &mask);
    if (sched_setaffinity(0, sizeof(mask), &mask) == -1) {
        printf("warning: could not set CPU affinity, continuing...\n");
    }

    int ret = 0;
    if (argc < 8) {
        printf("%s file_path iw ih ic ow oh oc\n", argv[0]);
        exit(0);
    }

    ret = venus::venus_init();
    if (0 != ret) {
        fprintf(stderr, "venus init failed.\n");
        exit(0);
    }
    /*image infos*/
    std::string filepath = argv[1];
    int iw = atoi(argv[2]);
    int ih = atoi(argv[3]);
    int ic = atoi(argv[4]);

    int ow = atoi(argv[5]);
    int oh = atoi(argv[6]);
    int oc = atoi(argv[7]);

    /*Bs Param*/
    magik::venus::BsExtendParam param;
    param.pad_val = 0;
    param.coef_off_enable = false;
    param.pad_type = magik::venus::BsPadType::NONE;
    param.in_layout = magik::venus::ChannelLayout::BGRA;
    param.out_layout = magik::venus::ChannelLayout::BGRA;
    magik::venus::TensorFormat src_format = magik::venus::TensorFormat::NHWC;
    magik::venus::TensorFormat dst_format = magik::venus::TensorFormat::NHWC;
    int src_size = ih * iw * ic;
    int dst_size = oh * ow * oc;
    if (ic == 3) {
        param.in_layout = magik::venus::ChannelLayout::RGB;
    } else if (ic == 4) {
        param.in_layout = magik::venus::ChannelLayout::BGRA;
    } else if (ic == 1) {
        param.in_layout = magik::venus::ChannelLayout::GRAY;
    } else {
        src_size = ih * iw * 3 / 2;
        param.in_layout = magik::venus::ChannelLayout::NV12;
        src_format = magik::venus::TensorFormat::NV12;
    }
    if (oc == 3) {
        param.out_layout = magik::venus::ChannelLayout::RGB;
    } else if (oc == 4) {
        param.out_layout = magik::venus::ChannelLayout::BGRA;
    } else if (oc == 1) {
        param.out_layout = magik::venus::ChannelLayout::GRAY;
    } else {
        dst_size = oh * ow * 3 / 2;
        param.out_layout = magik::venus::ChannelLayout::NV12;
        dst_format = magik::venus::TensorFormat::NV12;
    }
    magik::venus::Tensor src_img({1, ih, iw, ic}, src_format);
    magik::venus::Tensor dst_img({1, oh, ow, oc}, dst_format);
    uint8_t *in_addr = src_img.mudata<uint8_t>();
    uint8_t *dst_addr = dst_img.mudata<uint8_t>();
    int handle = open(filepath.c_str(), O_RDONLY);
    if (handle == -1) {
        printf("Error: %s:%d open failed\n", __func__, __LINE__);
        return -1;
    }
    if (src_size != read(handle, in_addr, src_size)) {
        printf("Error %s:%d read failed(src_size:%d)\n", __func__, __LINE__, src_size);
        return -1;
    }
    close(handle);
    /*resize api*/
    ret = warp_resize(src_img, dst_img, &param);

    write_bin_to_file("warp_resize.bin", (char *)dst_addr, dst_size * sizeof(uint8_t));
    ret = venus::venus_deinit();
    if (0 != ret) {
        fprintf(stderr, "venus deinit failed.\n");
        return ret;
    }
    return 0;
}

int crop_resize_run(int argc, char **argv) {
    /*set*/
    cpu_set_t mask;
    CPU_ZERO(&mask);
    CPU_SET(0, &mask);
    if (sched_setaffinity(0, sizeof(mask), &mask) == -1) {
        printf("warning: could not set CPU affinity, continuing...\n");
    }

    int ret = 0;
    if (argc < 9) {
        printf("%s file_path iw ih ic ow oh oc box_num\n", argv[0]);
        exit(0);
    }

    ret = venus::venus_init();
    if (0 != ret) {
        fprintf(stderr, "venus init failed.\n");
        exit(0);
    }
    /*image infos*/
    std::string filepath = argv[1];
    int iw = atoi(argv[2]);
    int ih = atoi(argv[3]);
    int ic = atoi(argv[4]);

    int ow = atoi(argv[5]);
    int oh = atoi(argv[6]);
    int oc = atoi(argv[7]);

    int box_num = atoi(argv[8]);

    /*Bs Param*/
    magik::venus::BsExtendParam param;
    param.pad_val = 0;
    param.coef_off_enable = false;
    param.pad_type = magik::venus::BsPadType::NONE;
    param.in_layout = magik::venus::ChannelLayout::BGRA;
    param.out_layout = magik::venus::ChannelLayout::BGRA;
    magik::venus::TensorFormat src_format = magik::venus::TensorFormat::NHWC;
    magik::venus::TensorFormat dst_format = magik::venus::TensorFormat::NHWC;
    int src_size = ih * iw * ic;
    int dst_size = oh * ow * oc;
    if (ic != 4) {
        src_size = ih * iw * 3 / 2;
        param.in_layout = magik::venus::ChannelLayout::NV12;
        src_format = magik::venus::TensorFormat::NV12;
    }
    if (oc != 4) {
        dst_size = oh * ow * 3 / 2;
        param.out_layout = magik::venus::ChannelLayout::NV12;
        dst_format = magik::venus::TensorFormat::NV12;
    }

    std::vector<magik::venus::Tensor> dst_imgs;
    std::vector<magik::venus::Bbox_t> infos;
    for (int i = 0; i < box_num; i++) {
        float x0 = i * 64;
        float y0 = i * 64;
        float x1 = x0 + 256;
        float y1 = y0 + 128;
        magik::venus::Bbox_t info = {x0, y0, x1, y1};
        infos.push_back(info);
        /*creat Tensor*/
        magik::venus::Tensor dst_img({1, oh, ow, oc}, dst_format);
        uint8_t *dst_addr = dst_img.mudata<uint8_t>();
        dst_imgs.push_back(dst_img);
    }

    magik::venus::Tensor src_img({1, ih, iw, ic}, src_format);
    uint8_t *in_addr = src_img.mudata<uint8_t>();
    int handle = open(filepath.c_str(), O_RDONLY);
    if (handle == -1) {
        printf("Error: %s:%d open failed\n", __func__, __LINE__);
        return -1;
    }
    if (src_size != read(handle, in_addr, src_size)) {
        printf("Error %s:%d read failed(src_size:%d)\n", __func__, __LINE__, src_size);
        return -1;
    }
    close(handle);
    /*resize api*/
    ret = crop_resize(src_img, dst_imgs, infos, &param);
    for (int i = 0; i < box_num; i++) {
        char output_name[2048];
        sprintf(output_name, "./crop_resize_%d.bin", i);
        uint8_t *out_addr = dst_imgs[i].mudata<uint8_t>();
        write_bin_to_file(output_name, (char *)out_addr, dst_size * sizeof(uint8_t));
    }

    ret = venus::venus_deinit();
    if (0 != ret) {
        fprintf(stderr, "venus deinit failed.\n");
        return ret;
    }
    return 0;
}

int crop_common_resize_run(int argc, char **argv) {
    /*set*/
    cpu_set_t mask;
    CPU_ZERO(&mask);
    CPU_SET(0, &mask);
    if (sched_setaffinity(0, sizeof(mask), &mask) == -1) {
        printf("warning: could not set CPU affinity, continuing...\n");
    }

    int ret = 0;
    if (argc < 9) {
        printf("%s file_path iw ih ic ow oh oc box_num\n", argv[0]);
        exit(0);
    }

    ret = venus::venus_init();
    if (0 != ret) {
        fprintf(stderr, "venus init failed.\n");
        exit(0);
    }
    /*image infos*/
    std::string filepath = argv[1];
    int iw = atoi(argv[2]);
    int ih = atoi(argv[3]);
    int ic = atoi(argv[4]);

    int ow = atoi(argv[5]);
    int oh = atoi(argv[6]);
    int oc = atoi(argv[7]);

    int box_num = atoi(argv[8]);

    /*Bs Param*/
    magik::venus::BsCommonParam param;
    param.pad_val = 0;
    param.coef_off_enable = false;
    param.pad_type = magik::venus::BsPadType::NONE;
    param.input_height = ih;
    param.input_width = iw;
    param.input_line_stride = iw;
    param.in_layout = magik::venus::ChannelLayout::NV12;
    param.out_layout = magik::venus::ChannelLayout::BGRA;
    magik::venus::TensorFormat src_format = magik::venus::TensorFormat::NV12;
    magik::venus::TensorFormat dst_format = magik::venus::TensorFormat::NHWC;
    int src_size = ih * iw * ic;
    int dst_size = oh * ow * oc;
    if (ic != 4) {
        src_size = ih * iw * 3 / 2;
        param.in_layout = magik::venus::ChannelLayout::NV12;
        src_format = magik::venus::TensorFormat::NV12;
    }
    if (oc != 4) {
        dst_size = oh * ow * 3 / 2;
        param.out_layout = magik::venus::ChannelLayout::NV12;
        dst_format = magik::venus::TensorFormat::NV12;
    }

    std::vector<magik::venus::Tensor> dst_imgs;
    std::vector<magik::venus::Bbox_t> infos;
    for (int i = 0; i < box_num; i++) {
        float x0 = i * 64;
        float y0 = i * 64;
        float x1 = x0 + 256;
        float y1 = y0 + 128;
        magik::venus::Bbox_t info = {x0, y0, x1, y1};
        infos.push_back(info);
        /*creat Tensor*/
        magik::venus::Tensor dst_img({1, oh, ow, oc}, dst_format);
        uint8_t *dst_addr = dst_img.mudata<uint8_t>();
        dst_imgs.push_back(dst_img);
    }

    magik::venus::Tensor src_img({1, ih, iw, ic}, src_format);
    uint8_t *in_addr = src_img.mudata<uint8_t>();
    int handle = open(filepath.c_str(), O_RDONLY);
    if (handle == -1) {
        printf("Error: %s:%d open failed\n", __func__, __LINE__);
        return -1;
    }
    if (src_size != read(handle, in_addr, src_size)) {
        printf("Error %s:%d read failed(src_size:%d)\n", __func__, __LINE__, src_size);
        return -1;
    }
    close(handle);
    /*resize api*/
    ret = crop_common_resize(in_addr, dst_imgs, infos, magik::venus::AddressLocate::NMEM_VIRTUAL,
                             &param);
    for (int i = 0; i < box_num; i++) {
        char output_name[2048];
        sprintf(output_name, "./crop_common_resize_%d.bin", i);
        uint8_t *out_addr = dst_imgs[i].mudata<uint8_t>();
        write_bin_to_file(output_name, (char *)out_addr, dst_size * sizeof(uint8_t));
    }

    ret = venus::venus_deinit();
    if (0 != ret) {
        fprintf(stderr, "venus deinit failed.\n");
        return ret;
    }
    return 0;
}

int common_resize_run(int argc, char **argv) {
    /*set*/
    cpu_set_t mask;
    CPU_ZERO(&mask);
    CPU_SET(0, &mask);
    if (sched_setaffinity(0, sizeof(mask), &mask) == -1) {
        printf("warning: could not set CPU affinity, continuing...\n");
    }

    int ret = 0;
    if (argc < 8) {
        printf("%s file_path iw ih ic ow oh oc\n", argv[0]);
        exit(0);
    }

    ret = venus::venus_init();
    if (0 != ret) {
        fprintf(stderr, "venus init failed.\n");
        exit(0);
    }
    /*image infos*/
    std::string filepath = argv[1];
    int iw = atoi(argv[2]);
    int ih = atoi(argv[3]);
    int ic = atoi(argv[4]);

    int ow = atoi(argv[5]);
    int oh = atoi(argv[6]);
    int oc = atoi(argv[7]);

    /*Bs Param*/
    magik::venus::BsCommonParam param;
    param.pad_val = 0;
    param.coef_off_enable = false;
    param.pad_type = magik::venus::BsPadType::NONE;
    param.input_height = ih;
    param.input_width = iw;
    param.input_line_stride = iw;
    param.in_layout = magik::venus::ChannelLayout::NV12;
    param.out_layout = magik::venus::ChannelLayout::BGRA;
    magik::venus::TensorFormat src_format = magik::venus::TensorFormat::NV12;
    magik::venus::TensorFormat dst_format = magik::venus::TensorFormat::NHWC;
    int src_size = ih * iw * 3 / 2;
    int dst_size = oh * ow * oc;
    /*input: nv12 or gray*/
    if (ic == 1) {
        param.in_layout = magik::venus::ChannelLayout::GRAY;
        src_size = ih * iw;
        src_format = magik::venus::TensorFormat::NHWC;
    }
    /*output: 1:gray 2:nv12 3:grb 4:bgra*/
    if (oc == 3) {
        param.out_layout = magik::venus::ChannelLayout::RGB;
    } else if (oc == 4) {
        param.out_layout = magik::venus::ChannelLayout::BGRA;
    } else if (oc == 1) {
        param.out_layout = magik::venus::ChannelLayout::GRAY;
    } else {
        dst_size = oh * ow * 3 / 2;
        param.out_layout = magik::venus::ChannelLayout::NV12;
        dst_format = magik::venus::TensorFormat::NV12;
    }
    magik::venus::Tensor src_img({1, ih, iw, ic}, src_format);
    magik::venus::Tensor dst_img({1, oh, ow, oc}, dst_format);
    uint8_t *in_addr = src_img.mudata<uint8_t>();
    uint8_t *dst_addr = dst_img.mudata<uint8_t>();
    int handle = open(filepath.c_str(), O_RDONLY);
    if (handle == -1) {
        printf("Error: %s:%d open failed\n", __func__, __LINE__);
        return -1;
    }
    if (src_size != read(handle, in_addr, src_size)) {
        printf("Error %s:%d read failed(src_size:%d)\n", __func__, __LINE__, src_size);
        return -1;
    }
    close(handle);
    /*resize api*/
    ret = warp_resize(src_img, dst_img, &param);

    write_bin_to_file("common_resize.bin", (char *)dst_addr, dst_size * sizeof(uint8_t));
    ret = venus::venus_deinit();
    if (0 != ret) {
        fprintf(stderr, "venus deinit failed.\n");
        return ret;
    }
    return 0;
}

int main(int argc, char **argv) {
    printf("input number 0(warp_resize_bgra)  1(crop_resize_run)  2(crop_common_resize_run) "
           "3(common_resize_run):\n");
    int i = 0;
    scanf("%d", &i);
    if (i == 0)
        warp_resize_run(argc, argv);
    else if (i == 1)
        crop_resize_run(argc, argv);
    else if (i == 2)
        crop_common_resize_run(argc, argv);
    else if (i == 3)
        common_resize_run(argc, argv);
    return 0;
}
