/*
 *         (C) COPYRIGHT Ingenic Limited
 *              ALL RIGHT RESERVED
 *
 * File        : net.h
 * Authors     : klyu(kanglong.yu@ingenic.com)
 * Create Time : 2019-12-26:10:00:00
 * Description :
 *
 */

#ifndef JZDL_NET_H
#define JZDL_NET_H
#include "mat.h"
#include <stdio.h>
#if !BUILD_RTOS
#include <string>
#include <vector>
using namespace std;
#else
#include "String.h"
#include "Vector.h"
using namespace jzdl::rtos;
#endif
#include <stdint.h>
namespace jzdl {

class BaseNet {
  public:
    BaseNet();
    virtual ~BaseNet() = 0;
    virtual int load_model(const char *model_file, bool memory_model = false);
    virtual vector<uint32_t> get_input_shape(void) const; /*return input shape: w, h, c*/
    virtual int get_model_input_index(void) const;   /*just for model debug*/
    virtual int get_model_output_index(void) const;  /*just for model debug*/

    virtual int input(const Mat<float> &in, int blob_index = -999);
    virtual int input(const Mat<int8_t> &in, int blob_index = -999);
    virtual int input(const Mat<int32_t> &in, int blob_index = -999);

    virtual int run(Mat<float> &feat, int blob_index = -999);
};

BaseNet *net_create();
void net_destory(BaseNet *net);

} // namespace jzdl
#endif
