/*
 *        (C) COPYRIGHT Ingenic Limited.
 *             ALL RIGHTS RESERVED
 *
 * File       : common_data_struct.h
 * Authors    : lqwang
 * Create Time: 2022-03-02:11:18:25
 * Description:
 *
 */

#include <cstdint>

#ifndef __COMMON_DATA_STRUCT_H__
#define __COMMON_DATA_STRUCT_H__

struct Img {
    int w;
    int h;
    int c;
    int w_stride;
    uint8_t* data;
};

struct Point {
    int x;
    int y;
};
    
#endif /* __COMMON_DATA_STRUCT_H__ */

